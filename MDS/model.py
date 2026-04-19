from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50


def build_resnet50(num_classes: int) -> nn.Module:
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = nn.Sequential(*list(classifier.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return torch.flatten(x, 1)
