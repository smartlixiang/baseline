import argparse
import os
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


def load_model_and_adapter(dataset, seed, clip_path, device):
    clip_model_path = Path(clip_path)
    if not clip_model_path.exists():
        raise FileNotFoundError(
            f"CLIP model file not found: {clip_model_path}. "
            "Please prepare local file `clip_model/ViT-B-32.pt`."
        )
    model, preprocess = clip.load(str(clip_model_path), device=device)
    model = model.float()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    image_adapter = nn.Linear(512, 512).to(device)
    text_adapter = nn.Linear(512, 512).to(device)

    ckpt_path = Path("checkpoints") / dataset / f"seed_{seed}" / "adapter.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    image_adapter.load_state_dict(state["image_adapter"])
    text_adapter.load_state_dict(state["text_adapter"])
    image_adapter.eval()
    text_adapter.eval()

    return model, preprocess, image_adapter, text_adapter


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


def main():
    parser = argparse.ArgumentParser(description="Generate sample mask with trained adapter")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser.add_argument("--seed", type=int, required=True, choices=SEED_CHOICES)
    parser.add_argument("--keep_ratio", type=float, required=True)
    parser.add_argument("--clip_path", type=str, default="clip_model/ViT-B-32.pt")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if not (0 < args.keep_ratio <= 1):
        raise ValueError("keep_ratio must be in (0, 1].")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = Path(__file__).resolve().parent
    clip_path = resolve_user_path(args.clip_path, script_dir)
    data_root = resolve_user_path(args.data_root, script_dir)
    model, preprocess, image_adapter, text_adapter = load_model_and_adapter(
        args.dataset, args.seed, str(clip_path), device
    )
    train_dataset, class_names = build_train_dataset(args.dataset, preprocess, data_root)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    prompts = [f"A photo of a {name}" for name in class_names]
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        class_text_features = model.encode_text(text_tokens).float()
        adapted_text = F.normalize(text_adapter(class_text_features), dim=-1)

    num_samples = len(train_dataset)
    scores = torch.zeros(num_samples, dtype=torch.float32)

    with torch.no_grad():
        for images, _, indices in tqdm(loader):
            images = images.to(device)
            image_features = model.encode_image(images).float()
            adapted_image = F.normalize(image_adapter(image_features), dim=-1)

            logits = adapted_image @ adapted_text.t()
            conf = torch.softmax(logits, dim=1).max(dim=1).values
            scores[indices] = conf.cpu()

    keep_num = int(num_samples * args.keep_ratio)
    topk = torch.topk(scores, k=keep_num, largest=True).indices

    mask = np.zeros(num_samples, dtype=np.uint8)
    mask[topk.numpy()] = 1

    dataset = args.dataset
    seed = args.seed
    keep_ratio = args.keep_ratio
    save_path = f"mask/{dataset}/{seed}/mask_{int(keep_ratio * 100)}.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez(save_path, mask=mask)
    print(f"Saved mask to {save_path}")


if __name__ == "__main__":
    main()
