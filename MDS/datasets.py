from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor

DATASETS = ("cifar10", "cifar100", "tiny-imagenet")
NUM_CLASSES: Dict[str, int] = {
    "cifar10": 10,
    "cifar100": 100,
    "tiny-imagenet": 200,
}


def get_eval_transform(dataset: str) -> Compose:
    if dataset in ("cifar10", "cifar100"):
        return Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    if dataset == "tiny-imagenet":
        return Compose([
            ToTensor(),
            Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
        ])
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_train_transform(dataset: str) -> Compose:
    # Keep transform simple and stable for reproducibility.
    return get_eval_transform(dataset)


class IndexedDataset(Dataset):
    """Wrap a dataset so each sample returns (index, image, target)."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        return index, image, target


def build_train_dataset(dataset: str, data_root: str | Path, for_feature: bool = False) -> Dataset:
    data_root = Path(data_root)
    transform = get_eval_transform(dataset) if for_feature else get_train_transform(dataset)

    if dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform)
    elif dataset == "cifar100":
        ds = torchvision.datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=transform)
    elif dataset == "tiny-imagenet":
        tiny_train = data_root / "tiny-imagenet-200" / "train"
        if not tiny_train.exists():
            raise FileNotFoundError(f"Tiny-ImageNet train directory not found: {tiny_train}")
        ds = ImageFolder(root=str(tiny_train), transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return IndexedDataset(ds)


def build_test_dataset(dataset: str, data_root: str | Path) -> Dataset:
    data_root = Path(data_root)
    transform = get_eval_transform(dataset)

    if dataset == "cifar10":
        return torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=transform)
    if dataset == "cifar100":
        return torchvision.datasets.CIFAR100(root=str(data_root), train=False, download=True, transform=transform)
    if dataset == "tiny-imagenet":
        tiny_val = data_root / "tiny-imagenet-200" / "val"
        if not tiny_val.exists():
            raise FileNotFoundError(f"Tiny-ImageNet val directory not found: {tiny_val}")
        return ImageFolder(root=str(tiny_val), transform=transform)

    raise ValueError(f"Unsupported dataset: {dataset}")


def get_dataset_size_and_classes(dataset: str) -> Tuple[int, int]:
    if dataset == "cifar10":
        return 50_000, 10
    if dataset == "cifar100":
        return 50_000, 100
    if dataset == "tiny-imagenet":
        return 100_000, 200
    raise ValueError(f"Unsupported dataset: {dataset}")
