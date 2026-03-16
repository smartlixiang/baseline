from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils import ensure_dir, keep_ratio_to_cut_ratio


@torch.no_grad()
def extract_features(model, dataloader, device: torch.device, normalize_features: bool = True):
    """Extract sample-wise embeddings and labels from a train dataloader."""
    model.eval()
    model.to(device)

    dataset_size = len(dataloader.dataset)
    feature_dim = model.feature_dim

    all_features = torch.zeros(dataset_size, feature_dim, dtype=torch.float32)
    all_labels = torch.zeros(dataset_size, dtype=torch.long)

    progress = tqdm(dataloader, desc="Feature extraction", leave=False)
    for images, labels, indices in progress:
        images = images.to(device, non_blocking=True)
        features = model(images)
        if normalize_features:
            features = F.normalize(features, p=2, dim=1)

        all_features[indices] = features.detach().cpu()
        all_labels[indices] = labels

    return all_features, all_labels


def herding_select_classwise(
    features: torch.Tensor,
    labels: torch.Tensor,
    keep_ratio: float,
    num_classes: int,
) -> np.ndarray:
    """Run standard class-wise feature herding and return 0/1 global mask."""
    num_samples = features.shape[0]
    selected_global = torch.zeros(num_samples, dtype=torch.bool)

    class_progress = tqdm(range(num_classes), desc=f"Herding keep={keep_ratio:.1f}", leave=False)
    for class_id in class_progress:
        class_indices = torch.where(labels == class_id)[0]
        class_features = features[class_indices]
        class_count = class_features.shape[0]

        if class_count == 0:
            continue

        target_count = int(round(class_count * keep_ratio))
        target_count = max(1, min(class_count, target_count))

        class_mean = class_features.mean(dim=0)

        selected_local = torch.zeros(class_count, dtype=torch.bool)
        running_sum = torch.zeros_like(class_mean)

        for k in range(1, target_count + 1):
            available = torch.where(~selected_local)[0]
            candidates = class_features[available]
            candidate_means = (running_sum.unsqueeze(0) + candidates) / k
            distances = ((candidate_means - class_mean.unsqueeze(0)) ** 2).sum(dim=1)

            best_pos = available[torch.argmin(distances)]
            selected_local[best_pos] = True
            running_sum = running_sum + class_features[best_pos]

        selected_global[class_indices[selected_local]] = True

    return selected_global.to(torch.uint8).cpu().numpy()


def generate_masks_for_keep_ratios(
    features: torch.Tensor,
    labels: torch.Tensor,
    keep_ratios: List[float],
    num_classes: int,
    dataset_name: str,
    seed: int,
    output_root: str,
) -> Dict[int, Path]:
    """Generate and save masks for every keep ratio under required directory format."""
    output_paths: Dict[int, Path] = {}
    for keep_ratio in keep_ratios:
        cut_ratio = keep_ratio_to_cut_ratio(keep_ratio)
        mask = herding_select_classwise(
            features=features,
            labels=labels,
            keep_ratio=keep_ratio,
            num_classes=num_classes,
        )

        save_dir = Path(output_root) / dataset_name / str(seed)
        ensure_dir(save_dir)
        save_path = save_dir / f"mask_{cut_ratio}.npz"
        np.savez(save_path, mask=mask)
        output_paths[cut_ratio] = save_path

    return output_paths
