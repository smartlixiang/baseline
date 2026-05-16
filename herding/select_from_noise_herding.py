#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click label-noise experiment script for the Herding baseline.

Run from the baseline repository root, e.g.:
    CUDA_VISIBLE_DEVICES=0 python select_from_noise_herding.py --dataset cifar10 --seed 22

Expected repository layout:
    baseline/
    ├── data/
    ├── noise/{dataset}/noise_list_{seed}.txt
    ├── herding/
    └── select_from_noise_herding.py

This script loads the fixed noise list, replaces training labels, extracts ResNet18
features, runs class-wise feature herding under noisy labels, saves masks for
keep ratios 30/50/70, and records how many selected samples are injected-noise
samples.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

DATASETS = ("cifar10", "cifar100")
SEEDS = (22, 42, 96)
KEEP_RATIOS = (30, 50, 70)
NOISE_RATE = 0.20

CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


@dataclass
class Config:
    dataset: str
    seed: int
    data_root: str
    noise_root: str
    cache_root: str
    mask_root: str
    batch_size: int = 128
    num_workers: int = 4
    use_pretrained: bool = True
    force: bool = False

    @property
    def noise_list_path(self) -> Path:
        return Path(self.noise_root) / self.dataset / f"noise_list_{self.seed}.txt"

    @property
    def cache_path(self) -> Path:
        tag = "pretrained" if self.use_pretrained else "random"
        return Path(self.cache_root) / self.dataset / f"seed_{self.seed}" / f"features_resnet18_{tag}.pt"

    @property
    def run_summary_path(self) -> Path:
        return Path(self.mask_root) / self.dataset / str(self.seed) / "summary.csv"

    @property
    def global_summary_path(self) -> Path:
        return Path(self.mask_root) / "summary.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_noise_list(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Noise list not found: {path}")
    arr = np.loadtxt(path, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Noise list must have two columns, got shape={arr.shape}: {path}")
    return arr


def apply_noise(clean_targets: np.ndarray, noise_list: np.ndarray, num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_targets = np.asarray(clean_targets, dtype=np.int64)
    noisy_targets = clean_targets.copy()
    sample_ids = noise_list[:, 0].astype(np.int64)
    new_labels = noise_list[:, 1].astype(np.int64)

    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample ids in noise list.")
    if np.any(sample_ids < 0) or np.any(sample_ids >= len(clean_targets)):
        raise ValueError("Noise sample id out of range.")
    if np.any(new_labels < 0) or np.any(new_labels >= num_classes):
        raise ValueError("Noisy label out of range.")
    if np.any(new_labels == clean_targets[sample_ids]):
        bad = int(np.sum(new_labels == clean_targets[sample_ids]))
        raise ValueError(f"{bad} noisy labels are identical to clean labels.")

    noisy_targets[sample_ids] = new_labels
    is_noisy = np.zeros(len(clean_targets), dtype=bool)
    is_noisy[sample_ids] = True
    actual = float(is_noisy.mean())
    if abs(actual - NOISE_RATE) > 1e-6:
        raise ValueError(f"Noise rate mismatch: expected {NOISE_RATE}, got {actual}")
    return noisy_targets, sample_ids, is_noisy


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        return image, target, index


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, prefer_pretrained: bool = True):
        super().__init__()
        weights = None
        if prefer_pretrained:
            try:
                weights = ResNet18_Weights.IMAGENET1K_V1
            except Exception as exc:
                warnings.warn(f"Failed to resolve ImageNet weights; using random init. Reason: {exc}")
        try:
            backbone = resnet18(weights=weights)
        except Exception as exc:
            warnings.warn(
                "Failed to load/download ImageNet pretrained ResNet18 weights; "
                f"falling back to random init. Reason: {exc}"
            )
            backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_noisy_train_dataset(cfg: Config):
    mean, std = CIFAR_STATS[cfg.dataset]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if cfg.dataset == "cifar10":
        ds = datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=transform)
    elif cfg.dataset == "cifar100":
        ds = datasets.CIFAR100(root=cfg.data_root, train=True, download=True, transform=transform)
    else:
        raise ValueError(cfg.dataset)
    clean_targets = np.asarray(ds.targets, dtype=np.int64)
    noise_list = read_noise_list(cfg.noise_list_path)
    noisy_targets, noisy_sample_ids, is_noisy = apply_noise(clean_targets, noise_list, NUM_CLASSES[cfg.dataset])
    ds.targets = noisy_targets.tolist()
    return ds, clean_targets, noisy_targets, noisy_sample_ids, is_noisy


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.to(device).eval()
    n = len(loader.dataset)
    features = torch.zeros(n, 512, dtype=torch.float32)
    labels = torch.zeros(n, dtype=torch.long)
    for images, targets, indices in tqdm(loader, desc="Extract ResNet18 features", dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        feats = model(images)
        feats = F.normalize(feats, p=2, dim=1)
        features[indices] = feats.detach().cpu()
        labels[indices] = targets.long().cpu()
    return features, labels


def load_or_extract_features(cfg: Config, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds, clean_targets, noisy_targets, noisy_sample_ids, is_noisy = build_noisy_train_dataset(cfg)
    if cfg.cache_path.exists() and not cfg.force:
        print(f"[cache] load {cfg.cache_path}")
        payload = torch.load(cfg.cache_path, map_location="cpu")
        return payload["features"], payload["labels"], clean_targets, noisy_targets, noisy_sample_ids, is_noisy

    loader = DataLoader(
        IndexedDataset(ds), batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
    )
    model = ResNet18FeatureExtractor(prefer_pretrained=cfg.use_pretrained)
    features, labels = extract_features(model, loader, device)
    cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "labels": labels, "clean_targets": clean_targets, "noisy_targets": noisy_targets}, cfg.cache_path)
    print(f"[cache] saved {cfg.cache_path}")
    return features, labels, clean_targets, noisy_targets, noisy_sample_ids, is_noisy


def exact_class_quotas(labels: torch.Tensor, keep_ratio: int, num_classes: int) -> Dict[int, int]:
    n = int(labels.numel())
    total_k = int(round(n * keep_ratio / 100.0))
    counts = np.asarray([(labels == c).sum().item() for c in range(num_classes)], dtype=np.int64)
    raw = counts * (keep_ratio / 100.0)
    quotas = np.floor(raw).astype(np.int64)
    remainder = total_k - int(quotas.sum())
    if remainder > 0:
        frac_order = np.argsort(-(raw - quotas))
        for c in frac_order:
            if remainder <= 0:
                break
            if quotas[c] < counts[c]:
                quotas[c] += 1
                remainder -= 1
    elif remainder < 0:
        frac_order = np.argsort(raw - quotas)
        for c in frac_order:
            if remainder >= 0:
                break
            if quotas[c] > 0:
                quotas[c] -= 1
                remainder += 1
    assert int(quotas.sum()) == total_k
    return {c: int(quotas[c]) for c in range(num_classes)}


def herding_classwise(features: torch.Tensor, labels: torch.Tensor, keep_ratio: int, num_classes: int) -> np.ndarray:
    n = features.shape[0]
    selected_global = torch.zeros(n, dtype=torch.bool)
    quotas = exact_class_quotas(labels, keep_ratio, num_classes)
    for class_id in tqdm(range(num_classes), desc=f"Herding keep={keep_ratio}%", dynamic_ncols=True):
        q = quotas[class_id]
        if q <= 0:
            continue
        class_indices = torch.where(labels == class_id)[0]
        class_features = features[class_indices]
        if class_features.numel() == 0:
            continue
        q = min(q, class_features.shape[0])
        class_mean = class_features.mean(dim=0)
        selected_local = torch.zeros(class_features.shape[0], dtype=torch.bool)
        running_sum = torch.zeros_like(class_mean)
        for k in range(1, q + 1):
            available = torch.where(~selected_local)[0]
            candidates = class_features[available]
            candidate_means = (running_sum.unsqueeze(0) + candidates) / k
            distances = ((candidate_means - class_mean.unsqueeze(0)) ** 2).sum(dim=1)
            best_pos = available[torch.argmin(distances)]
            selected_local[best_pos] = True
            running_sum += class_features[best_pos]
        selected_global[class_indices[selected_local]] = True
    mask = selected_global.to(torch.uint8).numpy()
    expected = int(round(n * keep_ratio / 100.0))
    assert int(mask.sum()) == expected, f"selected {mask.sum()}, expected {expected}"
    return mask


def save_mask(cfg: Config, keep_ratio: int, mask: np.ndarray, clean_targets, noisy_targets, noisy_sample_ids, is_noisy) -> dict:
    selected = np.where(mask == 1)[0].astype(np.int64)
    num_noisy_selected = int(is_noisy[selected].sum())
    ratio = float(num_noisy_selected / max(1, len(selected)))
    out_dir = Path(cfg.mask_root) / cfg.dataset / str(cfg.seed) / "herding"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mask_{keep_ratio}.npz"
    np.savez(
        out_path,
        mask=mask.astype(np.uint8), selected_indices=selected,
        clean_targets=clean_targets.astype(np.int64), noisy_targets=noisy_targets.astype(np.int64),
        noisy_sample_ids=noisy_sample_ids.astype(np.int64), is_noisy=is_noisy.astype(np.uint8),
        dataset=np.array(cfg.dataset), seed=np.array(cfg.seed), method=np.array("herding"),
        keep_ratio=np.array(keep_ratio), num_selected=np.array(len(selected)),
        num_noisy_selected=np.array(num_noisy_selected), noise_ratio_in_mask=np.array(ratio),
    )
    print(f"[mask] kr={keep_ratio} selected={len(selected)} noisy_selected={num_noisy_selected} ratio={ratio:.4f} -> {out_path}")
    return {
        "dataset": cfg.dataset, "seed": cfg.seed, "method": "herding", "keep_ratio": keep_ratio,
        "num_selected": len(selected), "num_noisy_selected": num_noisy_selected,
        "noise_ratio_in_mask": ratio, "mask_path": str(out_path),
    }


def write_summary(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "seed", "method", "keep_ratio", "num_selected", "num_noisy_selected", "noise_ratio_in_mask", "mask_path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"[summary] {path}")


def append_global_summary(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = []
    fields = ["dataset", "seed", "method", "keep_ratio", "num_selected", "num_noisy_selected", "noise_ratio_in_mask", "mask_path"]
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
    new_keys = {(str(r["dataset"]), str(r["seed"]), str(r["method"]), str(r["keep_ratio"])) for r in rows}
    old = [r for r in old if (str(r["dataset"]), str(r["seed"]), str(r["method"]), str(r["keep_ratio"])) not in new_keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(old + rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Herding selection under fixed 20% label noise.")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)
    p.add_argument("--force", action="store_true")
    p.add_argument("--disable-pretrained", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    cfg = Config(
        dataset=args.dataset, seed=args.seed,
        data_root=str(root / "data"), noise_root=str(root / "noise"),
        cache_root=str(root / "herding" / "noise_cache"),
        mask_root=str(root / "herding" / "noise_masks"),
        use_pretrained=not args.disable_pretrained, force=args.force,
    )
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, labels, clean_targets, noisy_targets, noisy_sample_ids, is_noisy = load_or_extract_features(cfg, device)
    rows = []
    for kr in KEEP_RATIOS:
        mask = herding_classwise(features, labels, kr, NUM_CLASSES[cfg.dataset])
        rows.append(save_mask(cfg, kr, mask, clean_targets, noisy_targets, noisy_sample_ids, is_noisy))
    write_summary(cfg.run_summary_path, rows)
    append_global_summary(cfg.global_summary_path, rows)
    print("[done]")


if __name__ == "__main__":
    main()
