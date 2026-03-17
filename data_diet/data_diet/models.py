from functools import partial
from typing import Sequence

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_channels: Sequence[int], num_classes: int):
        super().__init__()
        layers = []
        in_ch = 3
        for nc in num_channels:
            layers.extend([
                nn.Conv2d(in_ch, nc, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True),
            ])
            in_ch = nc
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x, train=False):
        del train
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        return self.head(x)


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.9, eps=1e-5)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.9, eps=1e-5)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, momentum=0.9, eps=1e-5),
            )

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = torch.relu(out + identity)
        return out


class BottleneckResNetBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        outplanes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.9, eps=1e-5)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.9, eps=1e-5)
        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes, momentum=0.9, eps=1e-5)
        nn.init.zeros_(self.bn3.weight)
        self.downsample = None
        if stride != 1 or inplanes != outplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes, momentum=0.9, eps=1e-5),
            )

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = torch.relu(out + identity)
        return out


class ResNet(nn.Module):
    def __init__(self, stage_sizes, block_cls, num_classes, num_filters=64, lowres=True):
        super().__init__()
        self.lowres = lowres
        self.inplanes = num_filters
        self.conv_init = nn.Conv2d(3, num_filters, kernel_size=3 if lowres else 7,
                                   stride=1 if lowres else 2, padding=1 if lowres else 3, bias=False)
        self.bn_init = nn.BatchNorm2d(num_filters, momentum=0.9, eps=1e-5)

        layers = []
        for i, block_count in enumerate(stage_sizes):
            planes = num_filters * (2 ** i)
            stride = 1 if i == 0 else 2
            layers.append(self._make_stage(block_cls, planes, block_count, stride))
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(num_filters * (2 ** (len(stage_sizes) - 1)) * block_cls.expansion, num_classes)

    def _make_stage(self, block_cls, planes, blocks, stride):
        blocks_list = [block_cls(self.inplanes, planes, stride=stride)]
        self.inplanes = planes * block_cls.expansion
        for _ in range(1, blocks):
            blocks_list.append(block_cls(self.inplanes, planes, stride=1))
        return nn.Sequential(*blocks_list)

    def forward(self, x, train=True):
        del train
        x = torch.relu(self.bn_init(self.conv_init(x)))
        if not self.lowres:
            x = torch.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layers(x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)


def get_model(args):
    if args.model == 'resnet18_lowres':
        return ResNet18(num_classes=args.num_classes, lowres=True)
    if args.model == 'resnet50_lowres':
        return ResNet50(num_classes=args.num_classes, lowres=True)
    if args.model == 'simple_cnn_0':
        return SimpleCNN(num_channels=[32, 64, 128], num_classes=args.num_classes)
    raise NotImplementedError(f'Unknown model: {args.model}')


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())
