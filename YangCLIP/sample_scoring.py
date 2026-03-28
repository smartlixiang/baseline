import argparse
import os
import random

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset as dataset_lib
from utils import normalize_dataset_name, obtain_classnames

DEFAULT_CLIP_MODEL_PATH = "clip_model/ViT-B-32.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_clip_local(path: str, device: torch.device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CLIP checkpoint not found at '{path}'. Please place local weights there.")
    model, preprocess = clip.load(path, device=device)
    return model.float(), preprocess


def main():
    parser = argparse.ArgumentParser(description="Compute YangCLIP sample scores.")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "tiny-imagenet"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--clip_model_path", type=str, default=DEFAULT_CLIP_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--score_dir", type=str, default="scores")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_name = normalize_dataset_name(args.dataset)

    adapter_path = args.adapter_path or os.path.join("adapter_ckpt", dataset_name, f"adapter_seed_{args.seed}.pt")
    if not os.path.isfile(adapter_path):
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = load_clip_local(args.clip_model_path, device)
    model.eval()

    input_dim = model.text_projection.shape[1]
    adapter_img = nn.Linear(input_dim, input_dim).to(device)
    adapter_text = nn.Linear(input_dim, input_dim).to(device)

    ckpt = torch.load(adapter_path, map_location=device)
    adapter_img.load_state_dict(ckpt["adapter_img"])
    adapter_text.load_state_dict(ckpt["adapter_text"])
    adapter_img.eval()
    adapter_text.eval()

    transform = transforms.Compose([preprocess])

    # 关键修复：
    # 不再把 data_root 拼成具体子目录，直接传 data 给 build_dataset。
    train_dataset = dataset_lib.build_dataset(
        dataset_name,
        args.data_root,
        train=True,
        transform=transform,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    n = len(train_dataset)
    targets = np.array(train_dataset.targets)

    class_names = train_dataset.classes if dataset_name == "tiny-imagenet" else obtain_classnames(dataset_name)
    text_inputs = torch.cat(
        [clip.tokenize(f"A photo of a {c}.") for c in tqdm(class_names, desc="Tokenizing class prompts")]
    ).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs).float()
        ft_text_features = F.normalize(adapter_text(text_features), dim=-1)

    image_features = torch.zeros((n, input_dim), dtype=torch.float32)
    sa_scores = torch.full((n,), -1.0, dtype=torch.float32)

    for index, images, target in tqdm(loader, desc="Extracting train features"):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            batch_image_features = model.encode_image(images).float()
            img_out = F.normalize(adapter_img(batch_image_features), dim=-1)
            txt_out = ft_text_features[target]
            matchness = F.cosine_similarity(img_out, txt_out, dim=-1)

        image_features[index] = img_out.detach().cpu()
        sa_scores[index] = matchness.detach().cpu()

    sd_scores = torch.zeros(n, dtype=torch.float32)
    unique_classes = np.unique(targets)
    k = 50 if dataset_name in ("cifar100", "tiny-imagenet") else 100

    for cls in tqdm(unique_classes, desc="Computing SD / KNN"):
        cls_idx = np.where(targets == cls)[0]
        cls_feats = image_features[cls_idx].numpy()
        n_neighbors = min(k, len(cls_idx))
        if n_neighbors <= 1:
            continue

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(cls_feats)
        distances, _ = nbrs.kneighbors(cls_feats)
        nearest_neighbor_distances = distances[:, 1:] if distances.shape[1] > 1 else distances
        sd_scores[cls_idx] = torch.tensor(nearest_neighbor_distances.mean(axis=1), dtype=torch.float32)

    def _norm(v: torch.Tensor) -> torch.Tensor:
        denom = (v.max() - v.min()).clamp(min=1e-12)
        return (v - v.min()) / denom

    sa_norm = _norm(sa_scores)
    sd_norm = _norm(sd_scores)
    final_score = sa_norm + sd_norm

    out_dir = os.path.join(args.score_dir, dataset_name, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "sa_scores.npy"), sa_scores.numpy())
    np.save(os.path.join(out_dir, "sd_scores.npy"), sd_scores.numpy())
    np.save(os.path.join(out_dir, "sa_norm.npy"), sa_norm.numpy())
    np.save(os.path.join(out_dir, "sd_norm.npy"), sd_norm.numpy())
    np.savez(os.path.join(out_dir, "scores.npz"), score=final_score.numpy())
    np.save(os.path.join(out_dir, "targets.npy"), targets)
    torch.save(image_features, os.path.join(out_dir, "image_features.pt"))

    print(f"[sample_scoring] saved: {out_dir}")


if __name__ == "__main__":
    main()
