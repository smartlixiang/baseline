import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

SEEDS = [22, 42, 96]
DATASET_CONFIG: Dict[str, Dict[str, int]] = {
    "CIFAR10": {"batch_size": 256, "epochs": 30},
    "CIFAR100": {"batch_size": 256, "epochs": 30},
    "Tiny-ImageNet": {"batch_size": 64, "epochs": 30},
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def _read_tiny_imagenet_class_names(tiny_root: Path, class_ids: List[str]) -> List[str]:
    wnids_path = tiny_root / "wnids.txt"
    words_path = tiny_root / "words.txt"

    # 默认使用类目录名（wnid）
    class_names = list(class_ids)

    # 如果存在 words.txt，优先用可读类别名
    if words_path.exists():
        wnid_to_words: Dict[str, str] = {}
        with words_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    wnid, names = parts[0], parts[1]
                    wnid_to_words[wnid] = names.split(",")[0].strip()
        class_names = [wnid_to_words.get(cid, cid) for cid in class_ids]
    elif wnids_path.exists():
        # 仅验证 wnids.txt 可用，若无 words 则继续使用 wnid
        with wnids_path.open("r", encoding="utf-8") as f:
            valid_ids = {line.strip() for line in f if line.strip()}
        class_names = [cid if cid in valid_ids else cid for cid in class_ids]

    return class_names



def build_dataloader(
    dataset_name: str,
    preprocess,
    data_root: Path,
    seed: int,
    num_workers: int,
) -> Tuple[DataLoader, List[str]]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    if dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root=str(data_root),
            train=True,
            transform=preprocess,
            download=True,
        )
        class_names = train_dataset.classes
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(
            root=str(data_root),
            train=True,
            transform=preprocess,
            download=True,
        )
        class_names = train_dataset.classes
    elif dataset_name == "Tiny-ImageNet":
        tiny_root = data_root / "tiny-imagenet-200"
        train_dir = tiny_root / "train"
        if not train_dir.exists():
            raise FileNotFoundError(
                f"Tiny-ImageNet train dir not found: {train_dir}. "
                "Expected structure: tiny-imagenet-200/train and tiny-imagenet-200/val"
            )
        train_dataset = datasets.ImageFolder(root=str(train_dir), transform=preprocess)
        class_ids = train_dataset.classes
        class_names = _read_tiny_imagenet_class_names(tiny_root, class_ids)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=DATASET_CONFIG[dataset_name]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=False,
    )
    return train_loader, class_names



def build_model(device: torch.device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model = model.float()

    # 冻结 CLIP backbone
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    image_adapter = nn.Linear(512, 512).to(device)
    text_adapter = nn.Linear(512, 512).to(device)

    return model, preprocess, image_adapter, text_adapter



def train_one_epoch(
    model,
    image_adapter: nn.Module,
    text_adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    class_text_features: torch.Tensor,
    device: torch.device,
    epoch: int,
    temperature: float = 0.07,
) -> float:
    image_adapter.train()
    text_adapter.train()

    total_loss = 0.0
    total_samples = 0

    batch_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for images, labels in batch_bar:
        images = images.to(device)
        labels = labels.to(device)
        bsz = images.size(0)

        with torch.no_grad():
            image_features = model.encode_image(images).float()
            text_features = class_text_features[labels].float()

        image_features = image_adapter(image_features)
        text_features = text_adapter(text_features)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits = image_features @ text_features.T / temperature
        targets = torch.arange(bsz, device=device)

        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.T, targets)
        loss = (loss_i + loss_t) / 2.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * bsz
        total_samples += bsz
        batch_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_samples, 1)



def main() -> None:
    parser = argparse.ArgumentParser(description="Train YangCLIP adapters with CLIP InfoNCE objective")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["CIFAR10", "CIFAR100", "Tiny-ImageNet", "all"],
        help="Dataset to train on. Use 'all' for all supported datasets.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Data root path relative to YangCLIP directory.",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = Path(__file__).resolve().parent
    data_root = (base_dir / args.data_root).resolve()
    ckpt_root = base_dir / "adapter_ckpt"

    datasets_to_run = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]

    for dataset_name in datasets_to_run:
        cfg = DATASET_CONFIG[dataset_name]
        epochs = cfg["epochs"]

        for seed in SEEDS:
            print(f"\n===== Dataset: {dataset_name} | Seed: {seed} =====")
            set_seed(seed)

            model, preprocess, image_adapter, text_adapter = build_model(device)
            train_loader, class_names = build_dataloader(
                dataset_name=dataset_name,
                preprocess=preprocess,
                data_root=data_root,
                seed=seed,
                num_workers=args.num_workers,
            )

            prompts = [f"A photo of a {name}" for name in class_names]
            text_tokens = clip.tokenize(prompts).to(device)
            with torch.no_grad():
                class_text_features = model.encode_text(text_tokens).float()

            optimizer = Adam(
                list(image_adapter.parameters()) + list(text_adapter.parameters()),
                lr=1e-4,
                weight_decay=0.0,
            )
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

            epoch_bar = tqdm(range(1, epochs + 1), desc=f"{dataset_name}-seed{seed}")
            for epoch in epoch_bar:
                avg_loss = train_one_epoch(
                    model=model,
                    image_adapter=image_adapter,
                    text_adapter=text_adapter,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    class_text_features=class_text_features,
                    device=device,
                    epoch=epoch,
                    temperature=args.temperature,
                )
                scheduler.step()
                epoch_bar.set_postfix(epoch=epoch, loss=f"{avg_loss:.4f}")

            save_dir = ckpt_root / dataset_name / str(seed)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(image_adapter.state_dict(), save_dir / "adapater_img.pth")
            torch.save(text_adapter.state_dict(), save_dir / "adapater_text.pth")
            print(f"Saved adapters to: {save_dir}")


if __name__ == "__main__":
    main()
