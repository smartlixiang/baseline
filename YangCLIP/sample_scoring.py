import argparse
import random
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader_with_index import DATASET_CHOICES, build_train_dataset


SEED_CHOICES = [22, 42, 96]
DATASET_DEFAULT_BATCH = {
    "cifar10": 256,
    "cifar100": 256,
    "tiny-imagenet": 64,
}
KNN_K = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_user_path(path_arg: str, script_dir: Path) -> Path:
    user_path = Path(path_arg)
    if user_path.is_absolute():
        return user_path.resolve()

    candidates = [
        (script_dir.parent / user_path).resolve(),
        (script_dir / user_path).resolve(),
        (Path.cwd() / user_path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return (script_dir / user_path).resolve()


def build_model(device: torch.device, clip_path: str):
    clip_model_path = Path(clip_path)
    if not clip_model_path.exists():
        raise FileNotFoundError(
            f"CLIP model file not found: {clip_model_path}. "
            "Please prepare local file `clip_model/ViT-B-32.pt`."
        )
    model, preprocess = clip.load(str(clip_model_path), device=device)
    model = model.float()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, preprocess


def compute_density_scores(
    features: np.ndarray,
    labels: np.ndarray,
    k: int = KNN_K,
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "features and labels must have same number of samples, "
            f"got {features.shape[0]} and {labels.shape[0]}"
        )

    num_samples = features.shape[0]
    density = np.zeros(num_samples, dtype=np.float32)

    for cls in tqdm(np.unique(labels), desc="Density (class-wise KNN)"):
        cls_idx = np.where(labels == cls)[0]
        cls_feats = features[cls_idx]
        cls_size = cls_feats.shape[0]
        if cls_size <= 1:
            density[cls_idx] = 0.0
            continue

        k_eff = min(k, cls_size - 1)
        sim = cls_feats @ cls_feats.T
        np.fill_diagonal(sim, -np.inf)

        topk = np.partition(sim, kth=cls_size - k_eff, axis=1)[:, -k_eff:]
        density[cls_idx] = topk.mean(axis=1).astype(np.float32)

    return density


def main():
    parser = argparse.ArgumentParser(description="Sample scoring for YangCLIP")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser.add_argument("--seed", type=int, required=True, choices=SEED_CHOICES)
    parser.add_argument("--clip_path", type=str, default="clip_model/ViT-B-32.pt")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = DATASET_DEFAULT_BATCH[args.dataset]

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = Path(__file__).resolve().parent
    clip_path = resolve_user_path(args.clip_path, script_dir)
    data_root = resolve_user_path(args.data_root, script_dir)

    model, preprocess = build_model(device, str(clip_path))

    ckpt_path = Path("checkpoints") / args.dataset / f"seed_{args.seed}" / "adapter.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    image_adapter = nn.Linear(512, 512).to(device)
    text_adapter = nn.Linear(512, 512).to(device)
    image_adapter.load_state_dict(state["image_adapter"])
    text_adapter.load_state_dict(state["text_adapter"])
    image_adapter.eval()
    text_adapter.eval()

    train_dataset, class_names = build_train_dataset(args.dataset, preprocess, data_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    prompts = [f"A photo of a {name}" for name in class_names]
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        adapted_text_features = F.normalize(text_adapter(text_features), dim=-1)

    num_samples = len(train_dataset)
    scores = np.zeros(num_samples, dtype=np.float32)
    features = np.zeros((num_samples, 512), dtype=np.float32)
    labels_all = np.zeros(num_samples, dtype=np.int64)

    with torch.no_grad():
        for images, labels, indices in tqdm(train_loader, desc="Scoring"):
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(images).float()
            adapted_image_features = F.normalize(image_adapter(image_features), dim=-1)
            batch_text_features = adapted_text_features[labels]
            matchness = F.cosine_similarity(adapted_image_features, batch_text_features, dim=-1)

            np_indices = indices.numpy()
            scores[np_indices] = matchness.detach().cpu().numpy().astype(np.float32)
            # SDS uses CLIP image feature neighborhood density.
            clip_image_features = F.normalize(image_features, dim=-1)
            features[np_indices] = clip_image_features.detach().cpu().numpy().astype(np.float32)
            labels_all[np_indices] = labels.detach().cpu().numpy().astype(np.int64)

    density = compute_density_scores(features=features, labels=labels_all, k=KNN_K)

    save_dir = Path("Pruning_Scores") / args.dataset / str(args.seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    score_path = save_dir / "scores.npy"
    density_path = save_dir / "density.npy"
    np.save(score_path, scores)
    np.save(density_path, density)
    print(f"Saved scores to {score_path}")
    print(f"Saved density to {density_path}")


if __name__ == "__main__":
    main()
