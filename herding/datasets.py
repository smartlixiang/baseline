from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}

IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class IndexedDataset(Dataset):
    """Wrap a dataset to additionally return sample indices."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label, index


def build_train_dataset(dataset_name: str, data_root: str):
    """Build train set with deterministic preprocessing (no augmentation)."""
    if dataset_name not in ("cifar10", "cifar100", "tiny-imagenet", "tiny-imagenet-200"):
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if dataset_name in CIFAR_STATS:
        mean, std = CIFAR_STATS[dataset_name]
    else:
        mean, std = IMAGENET_STATS

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root=data_root, train=True, transform=transform, download=True)
    elif dataset_name == "cifar100":
        dataset = datasets.CIFAR100(root=data_root, train=True, transform=transform, download=True)
    else:
        train_dir = Path(data_root) / "tiny-imagenet-200" / "train"
        if not train_dir.is_dir():
            raise FileNotFoundError(
                "Tiny-ImageNet train directory not found. "
                f"Expected: {train_dir}"
            )
        dataset = datasets.ImageFolder(root=str(train_dir), transform=transform)

    return dataset


def build_train_loader(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, Dataset]:
    dataset = build_train_dataset(dataset_name=dataset_name, data_root=data_root)
    indexed_dataset = IndexedDataset(dataset)
    loader = DataLoader(
        indexed_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset
