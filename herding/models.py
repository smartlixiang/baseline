from __future__ import annotations

import warnings

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18FeatureExtractor(nn.Module):
    """ResNet18 backbone returning global pooled embeddings before final FC layer."""

    def __init__(self, prefer_pretrained: bool = True):
        super().__init__()

        weights = None
        if prefer_pretrained:
            try:
                weights = ResNet18_Weights.IMAGENET1K_V1
            except Exception as exc:
                warnings.warn(
                    "Failed to resolve pretrained ResNet18 weights; falling back to random init. "
                    f"Reason: {exc}"
                )

        try:
            backbone = resnet18(weights=weights)
        except Exception as exc:
            if weights is not None:
                warnings.warn(
                    "Failed to load pretrained ResNet18 weights (download or cache issue). "
                    "Falling back to randomly initialized ResNet18. "
                    f"Reason: {exc}"
                )
                backbone = resnet18(weights=None)
            else:
                raise RuntimeError(f"Unable to build ResNet18 model: {exc}") from exc

        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_dim = 512

    def forward(self, x):
        return self.backbone(x)
