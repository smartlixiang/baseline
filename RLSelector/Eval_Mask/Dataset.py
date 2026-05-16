import sys
sys.path.append('..')
import torchvision.transforms as transforms

from Augmentation import *
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

DATASET_NCLASSES = {
    'CIFAR10': 10,
    'CIFAR100': 100,
    'Tiny-Imagenet': 200
}

DATASET_NORMALIZATION = {
    'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    'CIFAR100': ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    'Tiny-Imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}

def init_dataset(root, dataset_name, batch_size, mask):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]),
        Cutout(1, 16)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*DATASET_NORMALIZATION[dataset_name])
    ])

    if dataset_name == 'CIFAR10':
        trainset = CIFAR10(root=root, train=True, transform=transform_train)
        trainset = MaskedDataset(trainset, mask)
        testset = CIFAR10(root=root, train=False, transform=transform_test)
    elif dataset_name == 'CIFAR100':
        trainset = CIFAR100(root=root, train=True, transform=transform_train)
        trainset = MaskedDataset(trainset, mask)
        testset = CIFAR100(root=root, train=False, transform=transform_test)
    elif dataset_name == 'Tiny-Imagenet':
        pass
    else:
        assert 0, 'Dataset name: \"{}\" does not exist.'.format(dataset_name)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader


class MaskedDataset(Dataset):
    def __init__(self, dataset, mask=None):
        self.dataset = dataset
        self.mask = mask

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, index):
        index = self.mask[index]
        img, target = self.dataset[index]
        return img, target