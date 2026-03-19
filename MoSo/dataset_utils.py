from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path('./data')
DATASET_SUBDIRS = {
    'cifar10': 'cifar10',
    'cifar100': 'cifar100',
    'tiny': 'tiny-imagenet-200',
}
DATASET_NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny': 200,
}


def resolve_data_root(data_root: str | Path) -> Path:
    return Path(data_root).expanduser().resolve()


def resolve_dataset_path(dataset: str, data_root: str | Path) -> Path:
    if dataset not in DATASET_SUBDIRS:
        raise ValueError(f'Unsupported dataset: {dataset}')
    return resolve_data_root(data_root) / DATASET_SUBDIRS[dataset]


def _assert_tiny_val_ready(tiny_root: Path) -> None:
    val_root = tiny_root / 'val'
    if not val_root.exists():
        raise FileNotFoundError(f'Tiny-ImageNet val directory not found: {val_root}')

    if (val_root / 'images').exists():
        raise RuntimeError(
            'Tiny-ImageNet val directory is not arranged as an ImageFolder dataset yet. '
            'Please run `python MoSo/prepare_tiny_imagenet.py --data_root ./data` first.'
        )

    class_dirs = [p for p in val_root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(
            f'No class folders found under {val_root}. '
            'Please prepare Tiny-ImageNet val first with MoSo/prepare_tiny_imagenet.py.'
        )


def build_transforms(dataset: str, trainaug: int = 0) -> Dict[str, transforms.Compose]:
    if dataset == 'tiny':
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2302, 0.2265, 0.2262]
        default_train = transforms.Compose([
            transforms.RandomResizedCrop(55),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        randaug_train = transforms.Compose([
            transforms.RandomResizedCrop(55),
            transforms.Resize(64),
            transforms.RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        augmix_train = transforms.Compose([
            transforms.RandomResizedCrop(55),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test = transforms.Compose([
            transforms.Resize(int(64 / 0.875)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        if trainaug == 0:
            train = default_train
        elif trainaug == 3:
            train = augmix_train
        else:
            train = randaug_train
        return {'train': train, 'test': test}

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    default_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    autoaug_train = transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    randaug_train = transforms.Compose([
        transforms.RandAugment(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    augmix_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    if trainaug == 0:
        train = default_train
    elif trainaug == 1:
        train = autoaug_train
    elif trainaug == 2:
        train = randaug_train
    elif trainaug == 3:
        train = augmix_train
    else:
        raise ValueError(f'Unsupported trainaug value: {trainaug}')
    test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return {'train': train, 'test': test}


def build_train_dataset(dataset: str, data_root: str | Path, transform=None, trainaug: int = 0):
    transforms_map = build_transforms(dataset, trainaug=trainaug)
    transform = transform or transforms_map['train']
    dataset_root = resolve_dataset_path(dataset, data_root)
    if dataset == 'cifar10':
        return torchvision.datasets.CIFAR10(root=str(dataset_root), train=True, download=True, transform=transform)
    if dataset == 'cifar100':
        return torchvision.datasets.CIFAR100(root=str(dataset_root), train=True, download=True, transform=transform)
    if dataset == 'tiny':
        _assert_tiny_val_ready(dataset_root)
        return ImageFolder(root=str(dataset_root / 'train'), transform=transform)
    raise ValueError(f'Unsupported dataset: {dataset}')


def build_test_dataset(dataset: str, data_root: str | Path, transform=None, trainaug: int = 0):
    transforms_map = build_transforms(dataset, trainaug=trainaug)
    transform = transform or transforms_map['test']
    dataset_root = resolve_dataset_path(dataset, data_root)
    if dataset == 'cifar10':
        return torchvision.datasets.CIFAR10(root=str(dataset_root), train=False, download=True, transform=transform)
    if dataset == 'cifar100':
        return torchvision.datasets.CIFAR100(root=str(dataset_root), train=False, download=True, transform=transform)
    if dataset == 'tiny':
        _assert_tiny_val_ready(dataset_root)
        return ImageFolder(root=str(dataset_root / 'val'), transform=transform)
    raise ValueError(f'Unsupported dataset: {dataset}')


def build_eval_train_dataset(dataset: str, data_root: str | Path, trainaug: int = 0):
    return build_train_dataset(dataset, data_root, transform=build_transforms(dataset, trainaug=trainaug)['test'], trainaug=trainaug)


def get_dataset_targets(dataset_obj):
    if hasattr(dataset_obj, 'targets'):
        return list(dataset_obj.targets)
    raise AttributeError('Dataset does not expose targets in a supported way.')
