from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch


def _ensure_score_tensor(scores: Sequence[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(scores, torch.Tensor):
        tensor = scores.detach().cpu().clone().float()
    else:
        tensor = torch.tensor(scores, dtype=torch.float32)
    return tensor.squeeze()


def nopt2_selection(scores: Sequence[float] | torch.Tensor, keep_fraction: float, targets: Iterable[int]) -> List[int]:
    tar = torch.tensor(list(targets))
    pool = _ensure_score_tensor(scores)
    if len(pool) != len(tar):
        raise AssertionError(f'len(score)={len(pool)} does not match len(targets)={len(tar)}')
    size = int(len(pool) * keep_fraction)
    if size <= 0:
        return []
    num_classes = int(tar.max().item()) + 1
    index_all: List[int] = []
    for class_idx in range(num_classes):
        temp_pool = pool * (tar == class_idx).float()
        _, index = temp_pool.topk(int(size / num_classes))
        index_all.extend(index.tolist())
    return index_all


def random_selection(scores: Sequence[float] | torch.Tensor, keep_fraction: float) -> List[int]:
    pool = _ensure_score_tensor(scores)
    size = int(len(pool) * keep_fraction)
    if size <= 0:
        return []
    random_scores = torch.rand(pool.shape)
    _, index = random_scores.topk(size)
    return index.tolist()


def select_indices_from_scores(
    scores: Sequence[float] | torch.Tensor,
    keep_fraction: float,
    targets: Iterable[int],
    random_mode: bool = False,
) -> np.ndarray:
    if not 0 <= keep_fraction <= 1:
        raise ValueError(f'keep_fraction must be within [0, 1], got {keep_fraction}')
    if random_mode:
        selected = random_selection(scores, keep_fraction)
    else:
        selected = nopt2_selection(scores, keep_fraction, targets)
    return np.asarray(selected, dtype=np.int64)


def make_binary_mask(total_size: int, selected_index: Sequence[int]) -> np.ndarray:
    mask = np.zeros(total_size, dtype=np.uint8)
    if len(selected_index) > 0:
        mask[np.asarray(selected_index, dtype=np.int64)] = 1
    return mask


def load_score_file(score_path: str | Path) -> torch.Tensor:
    score = torch.load(Path(score_path), map_location='cpu')
    return _ensure_score_tensor(score)
