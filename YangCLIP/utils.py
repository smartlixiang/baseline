"""Utility constants and helpers for YangCLIP scripts.

Notes:
- Added dataset-name/path normalization to support unified external names
  (cifar10/cifar100/tiny-imagenet) while keeping backward compatibility.
"""

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
