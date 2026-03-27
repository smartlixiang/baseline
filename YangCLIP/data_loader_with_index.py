from pathlib import Path
from typing import List

from torch.utils.data import Dataset
from torchvision import datasets


DATASET_CHOICES = ["cifar10", "cifar100", "tiny-imagenet"]


class IndexedDataset(Dataset):
    """Wrap a dataset to return (image, label, index)."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label, index


def _read_tiny_imagenet_class_names(tiny_root: Path, class_ids: List[str]) -> List[str]:
    words_path = tiny_root / "words.txt"
    if not words_path.exists():
        return class_ids

    wnid_to_words = {}
    with words_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                wnid, names = parts[0], parts[1]
                wnid_to_words[wnid] = names.split(",")[0].strip()

    return [wnid_to_words.get(class_id, class_id) for class_id in class_ids]


def build_train_dataset(dataset_name: str, preprocess, data_root: Path):
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        base_dataset = datasets.CIFAR10(
            root=str(data_root),
            train=True,
            transform=preprocess,
            download=True,
        )
        class_names = base_dataset.classes
    elif dataset_name == "cifar100":
        base_dataset = datasets.CIFAR100(
            root=str(data_root),
            train=True,
            transform=preprocess,
            download=True,
        )
        class_names = base_dataset.classes
    elif dataset_name == "tiny-imagenet":
        tiny_root = data_root / "tiny-imagenet-200"
        train_dir = tiny_root / "train"
        if not train_dir.exists():
            raise FileNotFoundError(
                f"Tiny-ImageNet train dir not found: {train_dir}. "
                "Expected structure: tiny-imagenet-200/train"
            )

        base_dataset = datasets.ImageFolder(root=str(train_dir), transform=preprocess)
        class_names = _read_tiny_imagenet_class_names(tiny_root, base_dataset.classes)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return IndexedDataset(base_dataset), class_names
