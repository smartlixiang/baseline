from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import NUM_CLASSES, build_train_dataset
from .model import ResNet50FeatureExtractor, build_resnet50
from .utils import ensure_dir


def _get_median_prototypes(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    prototypes = np.zeros((num_classes, features.shape[1]), dtype=features.dtype)
    for class_id in range(num_classes):
        class_features = features[(labels == class_id).nonzero()[0], :]
        prototypes[class_id] = np.median(class_features, axis=0)
    return prototypes


def _get_distances(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    prototypes = _get_median_prototypes(features, labels, num_classes)
    prototype_per_sample = prototypes[labels]
    return np.linalg.norm(features - prototype_per_sample, axis=1)


def select_middle_band_indices(distances: np.ndarray, keep_ratio: int) -> np.ndarray:
    rate = keep_ratio / 100.0
    low = 0.5 - rate / 2
    high = 0.5 + rate / 2
    sorted_idx = distances.argsort()
    n = distances.shape[0]
    low_idx = round(n * low)
    high_idx = round(n * high)
    return sorted_idx[low_idx:high_idx]


def build_moderate_mask(num_samples: int, selected_indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(num_samples, dtype=np.int64)
    mask[selected_indices] = 1
    return mask


def validate_mask(mask: np.ndarray, expected_ones: int, n_samples: int) -> None:
    assert mask.ndim == 1, f"mask must be 1D, got shape={mask.shape}"
    assert mask.shape[0] == n_samples, f"mask length {mask.shape[0]} != {n_samples}"
    unique_values = np.unique(mask)
    assert set(unique_values.tolist()).issubset({0, 1}), f"mask has non-binary values: {unique_values}"
    num_selected = int(mask.sum())
    assert num_selected == expected_ones, f"mask selected {num_selected}, expected {expected_ones}"


def extract_train_features(
    dataset: str,
    data_root: str,
    seed: int,
    ckpt_path: str | Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Path:
    dataset_obj = build_train_dataset(dataset, data_root, for_feature=True)
    loader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    classifier = build_resnet50(NUM_CLASSES[dataset])
    state = torch.load(ckpt_path, map_location="cpu")
    classifier.load_state_dict(state)
    model = ResNet50FeatureExtractor(classifier).to(device)
    model.eval()

    n_samples = len(dataset_obj)
    feature_dim = 2048
    features = np.zeros((n_samples, feature_dim), dtype=np.float32)
    labels = np.zeros((n_samples,), dtype=np.int64)

    with torch.no_grad():
        for indices, images, target in tqdm(loader, desc=f"Extract {dataset} seed={seed}"):
            images = images.to(device, non_blocking=True)
            batch_features = model(images).cpu().numpy()
            features[indices.numpy()] = batch_features
            labels[indices.numpy()] = target.numpy()

    cache_dir = ensure_dir(Path("MDS") / "cache" / dataset / str(seed))
    cache_path = cache_dir / "train_features.npz"
    np.savez(cache_path, features=features, labels=labels)
    return cache_path


def load_or_extract_features(
    dataset: str,
    data_root: str,
    seed: int,
    ckpt_path: str | Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    force_recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = Path("MDS") / "cache" / dataset / str(seed) / "train_features.npz"
    if cache_path.exists() and not force_recompute:
        arr = np.load(cache_path)
        return arr["features"], arr["labels"]

    cache_path = extract_train_features(
        dataset=dataset,
        data_root=data_root,
        seed=seed,
        ckpt_path=ckpt_path,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    arr = np.load(cache_path)
    return arr["features"], arr["labels"]


def generate_masks_for_ratios(
    dataset: str,
    seed: int,
    keep_ratios: Iterable[int],
    features: np.ndarray,
    labels: np.ndarray,
    mask_root: str | Path = "mask/MDS",
) -> None:
    distances = _get_distances(features, labels, NUM_CLASSES[dataset])
    out_dir = ensure_dir(Path(mask_root) / dataset / str(seed))
    n = features.shape[0]

    for keep_ratio in tqdm(list(keep_ratios), desc=f"Select {dataset} seed={seed}"):
        selected = select_middle_band_indices(distances, keep_ratio)
        mask = build_moderate_mask(n, selected)
        validate_mask(mask, expected_ones=len(selected), n_samples=n)

        save_path = out_dir / f"mask_{keep_ratio}.npz"
        np.savez(save_path, mask=mask)
        print(
            f"[mask] dataset={dataset} seed={seed} keep_ratio={keep_ratio} "
            f"num_samples={n} selected={int(mask.sum())} selected_ratio={mask.mean():.6f}"
        )
