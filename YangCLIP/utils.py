"""Utility constants and helpers for YangCLIP scripts.

Notes:
- Added dataset-name/path normalization to support unified external names
  (cifar10/cifar100/tiny-imagenet) while keeping backward compatibility.
"""
import os
from typing import Dict, List

cifar10_classes = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

cifar100_classes = [
    'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 'palm tree', 'pear', 'pickup truck', 'pine tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow tree', 'wolf', 'woman', 'worm',
]


def normalize_dataset_name(dataset: str) -> str:
    """Normalize dataset aliases to unified lowercase names for new scripts."""
    key = dataset.strip().lower().replace('_', '-').replace(' ', '')
    mapping = {
        'cifar10': 'cifar10',
        'cifar-10': 'cifar10',
        'cifar100': 'cifar100',
        'cifar-100': 'cifar100',
        'tinyimagenet': 'tiny-imagenet',
        'tiny-imagenet': 'tiny-imagenet',
        # keep old defaults compatible
        'cifar10legacy': 'cifar10',
    }
    if dataset in ('CIFAR10', 'CIFAR100'):
        return dataset.lower()
    if key in mapping:
        return mapping[key]
    raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of: cifar10, cifar100, tiny-imagenet")


def get_dataset_subdir(dataset: str) -> str:
    """Dataset subdir under data_root; fixed by requirement."""
    d = normalize_dataset_name(dataset)
    if d == 'cifar10':
        return 'cifar-10-batches-py'
    if d == 'cifar100':
        return 'cifar-100-python'
    if d == 'tiny-imagenet':
        return 'tiny-imagenet-200'
    raise ValueError(d)


def obtain_classnames(dataset: str):
    """Return class names for CIFAR datasets (Tiny-ImageNet uses ImageFolder classes)."""
    d = normalize_dataset_name(dataset)
    if d == 'cifar10':
        return cifar10_classes
    if d == 'cifar100':
        return cifar100_classes
    raise ValueError("tiny-imagenet class names should come from ImageFolder.classes")


def load_tiny_imagenet_wnid_to_name(data_root: str) -> Dict[str, str]:
    """Load Tiny-ImageNet WNID -> natural-language class name mapping.

    Mapping source:
    - wnids.txt: authoritative class-id list used by the dataset.
    - words.txt: WordNet id to comma-separated English synonyms.
    """
    tiny_root = os.path.join(data_root, "tiny-imagenet-200")
    wnids_path = os.path.join(tiny_root, "wnids.txt")
    words_path = os.path.join(tiny_root, "words.txt")

    if not os.path.isfile(wnids_path):
        raise FileNotFoundError(
            f"Tiny-ImageNet metadata missing: '{wnids_path}'. "
            "Cannot build natural-language prompts from WNIDs."
        )
    if not os.path.isfile(words_path):
        raise FileNotFoundError(
            f"Tiny-ImageNet metadata missing: '{words_path}'. "
            "Cannot build natural-language prompts from WNIDs."
        )

    with open(wnids_path, "r", encoding="utf-8") as f:
        wnids = [line.strip() for line in f if line.strip()]

    words_map: Dict[str, str] = {}
    with open(words_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            wnid, names = parts
            first_name = names.split(",")[0].strip()
            if first_name:
                words_map[wnid.strip()] = first_name

    missing = [wnid for wnid in wnids if wnid not in words_map]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            "Tiny-ImageNet words mapping is incomplete. "
            f"{len(missing)} WNID(s) from wnids.txt missing in words.txt. "
            f"Examples: {preview}"
        )

    return {wnid: words_map[wnid] for wnid in wnids}


def resolve_class_names(dataset_name: str, data_root: str, class_names: List[str]) -> List[str]:
    """Resolve dataset class identifiers into prompt-ready English labels.

    For Tiny-ImageNet, `class_names` typically come from ImageFolder.classes,
    i.e. WNID directory names (e.g. n01443537), which are not suitable as CLIP
    text prompts. We map each WNID to an English class phrase via words.txt.
    """
    d = normalize_dataset_name(dataset_name)
    if d in ("cifar10", "cifar100"):
        return list(class_names)

    if d != "tiny-imagenet":
        raise ValueError(f"Unsupported dataset for class-name resolving: {dataset_name}")

    wnid_to_name = load_tiny_imagenet_wnid_to_name(data_root)
    resolved: List[str] = []
    missing: List[str] = []
    for wnid in class_names:
        name = wnid_to_name.get(wnid)
        if name is None:
            missing.append(wnid)
        else:
            resolved.append(name)

    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            "Found Tiny-ImageNet class id(s) that cannot be mapped to English names: "
            f"{preview}. Check class_names/ImageFolder directory names and metadata files."
        )
    return resolved
