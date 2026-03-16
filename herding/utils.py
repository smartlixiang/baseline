import os
import random
from pathlib import Path

import numpy as np
import torch


DEFAULT_KEEP_RATIOS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
DEFAULT_SEEDS = [22, 42, 96]
DEFAULT_DATASETS = ["cifar10", "cifar100"]


def set_seed(seed: int) -> None:
    """Set Python / NumPy / PyTorch random seeds with deterministic CuDNN config."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Use single GPU when available, else fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def keep_ratio_to_cut_ratio(keep_ratio: float) -> int:
    """Convert keep ratio to integer tag for mask naming (cut_ratio=keep percentage)."""
    return int(round(keep_ratio * 100))


def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)
