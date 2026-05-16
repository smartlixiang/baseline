import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from Augmentation import *
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100

DATASET_NCLASSES = {
    'CIFAR10': 10,
    'CIFAR100': 100,
    'Tiny-Imagenet': 200,
}
DATASET_SIZES = {
    'CIFAR10': (32, 32),
    'CIFAR100': (32, 32),
    'Tiny-Imagenet': (64, 64)
}
DATASET_NORMALIZATION = {
    'CIFAR10': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'CIFAR100': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'Tiny-Imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}
DATASET_NUM = {
    'CIFAR10': 50000,
    'CIFAR100': 50000,
    'Tiny-Imagenet': 100000,
}

def init_dataset(root, dataset_name):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]),
        Cutout(n_holes=1, length=16)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*DATASET_NORMALIZATION[dataset_name])
    ])

    if dataset_name == 'CIFAR10':
        trainset = IndexMapDataset(CIFAR10(root=root, train=True, transform=transform_train))
        testset = CIFAR10(root=root, train=False, transform=transform_test)
    elif dataset_name == 'CIFAR100':
        trainset = IndexMapDataset(CIFAR100(root=root, train=True, transform=transform_train))
        testset = CIFAR100(root=root, train=False, transform=transform_test)
    elif dataset_name == 'Tiny-Imagenet':
        pass
    else:
        assert 0, 'Dataset name: \"{}\" does not exist.'.format(dataset_name)

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