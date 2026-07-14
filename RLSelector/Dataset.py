import os
from collections import defaultdict

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100

from Augmentation import *

DATASET_NCLASSES = {
    'CIFAR10': 10,
    'CIFAR100': 100,
    'Tiny-Imagenet': 200,
}
DATASET_SIZES = {
    'CIFAR10': (32, 32),
    'CIFAR100': (32, 32),
    'Tiny-Imagenet': (64, 64),
}
DATASET_NORMALIZATION = {
    'CIFAR10': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'CIFAR100': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'Tiny-Imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
DATASET_NUM = {
    'CIFAR10': 50000,
    'CIFAR100': 50000,
    'Tiny-Imagenet': 100000,
}

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.ppm')


def normalize_dataset_name(dataset_name):
    """Normalize command-line dataset aliases to the names used internally."""
    name = dataset_name.strip()
    key = name.lower().replace('_', '').replace('-', '')
    if key == 'cifar10':
        return 'CIFAR10'
    if key == 'cifar100':
        return 'CIFAR100'
    if key in {'tinyimagenet', 'tinyimagenet200'}:
        return 'Tiny-Imagenet'
    if name in DATASET_NCLASSES:
        return name
    raise ValueError('Dataset name: "{}" does not exist.'.format(dataset_name))


def dataset_output_name(dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == 'CIFAR10':
        return 'cifar10'
    if dataset_name == 'CIFAR100':
        return 'cifar100'
    if dataset_name == 'Tiny-Imagenet':
        return 'tinyimagenet'
    raise ValueError(dataset_name)


def _resolve_tiny_imagenet_root(root):
    """Return the directory that contains train/, val/, wnids.txt for Tiny-ImageNet."""
    candidates = [
        root,
        os.path.join(root, 'tiny-imagenet-200'),
        os.path.join(root, 'Tiny-Imagenet'),
        os.path.join(root, 'Tiny-ImageNet'),
        os.path.join(root, 'tinyimagenet'),
        os.path.join(root, 'tiny_imagenet'),
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, 'train')) and os.path.isdir(os.path.join(path, 'val')):
            return path
    raise FileNotFoundError(
        'Cannot find Tiny-ImageNet under root="{}". Expected a directory such as '
        'data/tiny-imagenet-200 containing train/ and val/.'.format(root)
    )


class TinyImageNet(Dataset):
    """Minimal Tiny-ImageNet-200 loader.

    The expected official structure is:
        tiny-imagenet-200/
          train/<wnid>/images/*.JPEG
          val/images/*.JPEG
          val/val_annotations.txt
          wnids.txt

    The order of training samples is deterministic, so the generated mask can be
    reused as long as the same loader is used during later selected-set training.
    """

    def __init__(self, root, train=True, transform=None):
        self.root = _resolve_tiny_imagenet_root(root)
        self.train = train
        self.transform = transform
        self.samples = []
        self.targets = []

        train_root = os.path.join(self.root, 'train')
        wnids_file = os.path.join(self.root, 'wnids.txt')
        if os.path.isfile(wnids_file):
            with open(wnids_file, 'r') as f:
                classes = [line.strip() for line in f if line.strip()]
            classes = [c for c in classes if os.path.isdir(os.path.join(train_root, c))]
        else:
            classes = [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]
            classes = sorted(classes)

        if len(classes) != 200:
            raise RuntimeError('Tiny-ImageNet should contain 200 classes, but found {}.'.format(len(classes)))

        self.classes = classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        if train:
            self._load_train_split()
        else:
            self._load_val_split()

    def _load_train_split(self):
        train_root = os.path.join(self.root, 'train')
        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            cls_dir = os.path.join(train_root, cls_name)
            image_paths = []
            for dirpath, _, filenames in os.walk(cls_dir):
                for filename in filenames:
                    if filename.lower().endswith(IMG_EXTENSIONS):
                        image_paths.append(os.path.join(dirpath, filename))
            for image_path in sorted(image_paths):
                self.samples.append((image_path, cls_idx))
                self.targets.append(cls_idx)

        if len(self.samples) != DATASET_NUM['Tiny-Imagenet']:
            raise RuntimeError(
                'Tiny-ImageNet train split should contain {} images, but found {}.'.format(
                    DATASET_NUM['Tiny-Imagenet'], len(self.samples)
                )
            )

    def _load_val_split(self):
        """Load Tiny-ImageNet validation split.

        Supported layouts:
        1. Official layout:
        val/images/*.JPEG
        val/val_annotations.txt

        2. ImageFolder-like layout:
        val/<wnid>/images/*.JPEG
        or
        val/<wnid>/*.JPEG
        """
        val_root = os.path.join(self.root, 'val')
        image_root = os.path.join(val_root, 'images')
        ann_path = os.path.join(val_root, 'val_annotations.txt')

        # Official Tiny-ImageNet validation layout.
        if os.path.isfile(ann_path) and os.path.isdir(image_root):
            annotations = {}
            with open(ann_path, 'r') as f:
                for line in f:
                    # Use split() instead of split('\t') to tolerate tabs/spaces.
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        annotations[parts[0]] = parts[1]

            for filename in sorted(annotations.keys()):
                cls_name = annotations[filename]
                if cls_name not in self.class_to_idx:
                    continue

                image_path = os.path.join(image_root, filename)
                if not os.path.isfile(image_path):
                    continue

                cls_idx = self.class_to_idx[cls_name]
                self.samples.append((image_path, cls_idx))
                self.targets.append(cls_idx)

            if len(self.samples) > 0:
                return

        # Fallback: ImageFolder-like validation layout.
        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            cls_dir = os.path.join(val_root, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            image_paths = []
            for dirpath, _, filenames in os.walk(cls_dir):
                for filename in filenames:
                    if filename.lower().endswith(IMG_EXTENSIONS):
                        image_paths.append(os.path.join(dirpath, filename))

            for image_path in sorted(image_paths):
                self.samples.append((image_path, cls_idx))
                self.targets.append(cls_idx)

        if len(self.samples) == 0:
            raise RuntimeError(
                'Loaded zero Tiny-ImageNet validation images. Please check the dataset layout under "{}". '
                'Expected either val/images + val_annotations.txt, or val/<class>/images/*.JPEG.'.format(
                    val_root
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        img = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def init_dataset(root, dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    image_size = DATASET_SIZES[dataset_name][0]
    padding = 4 if image_size == 32 else 8

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=padding),
        transforms.ToTensor(),
        transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]),
        Cutout(n_holes=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]),
    ])

    if dataset_name == 'CIFAR10':
        trainset = IndexMapDataset(CIFAR10(root=root, train=True, transform=transform_train))
        testset = CIFAR10(root=root, train=False, transform=transform_test)
    elif dataset_name == 'CIFAR100':
        trainset = IndexMapDataset(CIFAR100(root=root, train=True, transform=transform_train))
        testset = CIFAR100(root=root, train=False, transform=transform_test)
    elif dataset_name == 'Tiny-Imagenet':
        trainset = IndexMapDataset(TinyImageNet(root=root, train=True, transform=transform_train))
        testset = TinyImageNet(root=root, train=False, transform=transform_test)
    else:
        assert 0, 'Dataset name: "{}" does not exist.'.format(dataset_name)

    return trainset, testset


class IndexMapDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.class2index = defaultdict(list)
        for idx, t in enumerate(self.dataset.targets):
            self.class2index[t].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        return index, img, target

    @property
    def index_map(self):
        return self.class2index
