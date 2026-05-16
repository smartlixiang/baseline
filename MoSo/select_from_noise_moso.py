#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click label-noise experiment script for MoSo.

Run from the baseline repository root, e.g.:
    CUDA_VISIBLE_DEVICES=0 python select_from_noise_moso.py --dataset cifar10 --seed 22

This script trains a MoSo surrogate model on noisy labels, computes the MoSo
sample-removal-effect score using the same core formula as MoSo/scoring.py,
exports masks for keep ratios 30/50/70, and records how many selected samples
are injected-noise samples.

Note: exact MoSo scoring computes per-sample gradients and is expensive.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet50 as tv_resnet50
from tqdm import tqdm

DATASETS = ("cifar10", "cifar100")
SEEDS = (22, 42, 96)
KEEP_RATIOS = (30, 50, 70)
NOISE_RATE = 0.20
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def train_transform():
    # MoSo build_transforms(trainaug=0) for CIFAR.
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


@dataclass
class Config:
    dataset: str
    seed: int
    data_root: str
    noise_root: str
    exp_root: str
    mask_root: str
    model: str = "resnet50"
    epochs: int = 50
    batch_size: int = 256
    score_batch_size: int = 1
    num_workers: int = 4
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    force: bool = False

    @property
    def noise_list_path(self) -> Path:
        return Path(self.noise_root) / self.dataset / f"noise_list_{self.seed}.txt"

    @property
    def run_dir(self) -> Path:
        return Path(self.exp_root) / self.dataset / f"seed_{self.seed}"

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "ckpt"

    @property
    def best_ckpt_path(self) -> Path:
        return self.ckpt_dir / "best.pth"

    @property
    def last_ckpt_path(self) -> Path:
        return self.ckpt_dir / "last.pth"

    @property
    def score_path(self) -> Path:
        return self.run_dir / "score" / "moso_score.npy"

    @property
    def run_summary_path(self) -> Path:
        return Path(self.mask_root) / self.dataset / str(self.seed) / "summary.csv"

    @property
    def global_summary_path(self) -> Path:
        return Path(self.mask_root) / "summary.csv"


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset, return_index: bool = False):
        self.base = base
        self.return_index = return_index

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        if self.return_index:
            return index, image, target
        return image, target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    ids = noise_list[:, 0].astype(np.int64)
    labels = noise_list[:, 1].astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate sample ids in noise list.")
    if np.any(ids < 0) or np.any(ids >= len(clean_targets)):
        raise ValueError("Noise sample id out of range.")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("Noisy label out of range.")
    if np.any(labels == clean_targets[ids]):
        raise ValueError("Noise list contains unchanged labels.")
    noisy_targets[ids] = labels
    is_noisy = np.zeros(len(clean_targets), dtype=bool)
    is_noisy[ids] = True
    if abs(float(is_noisy.mean()) - NOISE_RATE) > 1e-6:
        raise ValueError(f"Noise rate mismatch: {is_noisy.mean()} vs expected {NOISE_RATE}")
    return noisy_targets, ids, is_noisy


def build_cifar(cfg: Config, train: bool, noisy: bool, transform):
    if cfg.dataset == "cifar10":
        ds = datasets.CIFAR10(root=cfg.data_root, train=train, download=True, transform=transform)
    elif cfg.dataset == "cifar100":
        ds = datasets.CIFAR100(root=cfg.data_root, train=train, download=True, transform=transform)
    else:
        raise ValueError(cfg.dataset)
    clean_targets = np.asarray(ds.targets, dtype=np.int64)
    if train and noisy:
        noisy_targets, ids, is_noisy = apply_noise(clean_targets, read_noise_list(cfg.noise_list_path), NUM_CLASSES[cfg.dataset])
        ds.targets = noisy_targets.tolist()
        return ds, clean_targets, noisy_targets, ids, is_noisy
    return ds


def build_moso_model(cfg: Config) -> nn.Module:
    num_classes = NUM_CLASSES[cfg.dataset]
    moso_dir = Path.cwd() / "MoSo"
    if str(moso_dir) not in sys.path:
        sys.path.insert(0, str(moso_dir))
    try:
        from models import ResNet18, ResNet50  # type: ignore
        if cfg.model == "resnet18":
            return ResNet18(-10, num_classes)
        if cfg.model == "resnet50":
            return ResNet50(-10, num_classes)
    except Exception as exc:
        warnings.warn(f"Failed to import MoSo.models, fallback to torchvision resnet50. Reason: {exc}")
    model = tv_resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += int((preds == targets).sum().item())
        total += int(targets.numel())
    return correct / max(1, total)


def train_surrogate(cfg: Config, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_ds, clean_targets, noisy_targets, noisy_ids, is_noisy = build_cifar(cfg, True, True, train_transform())
    if cfg.best_ckpt_path.exists() and cfg.last_ckpt_path.exists() and not cfg.force:
        print(f"[train] found checkpoints, skip training: {cfg.best_ckpt_path}")
        return clean_targets, noisy_targets, noisy_ids, is_noisy
    test_ds = build_cifar(cfg, False, False, eval_transform())
    train_loader = DataLoader(IndexedDataset(train_ds, return_index=False), batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=100, shuffle=False, num_workers=cfg.num_workers,
                             pin_memory=torch.cuda.is_available())
    model = build_moso_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[25, 38], gamma=0.1)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.run_dir / "train_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "test_acc", "best_acc"])
        writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            total_loss = total = 0
            pbar = tqdm(train_loader, desc=f"MoSo train {cfg.dataset} seed={cfg.seed} {epoch}/{cfg.epochs}", dynamic_ncols=True)
            for images, targets in pbar:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * targets.size(0)
                total += targets.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            scheduler.step()
            acc = evaluate(model, test_loader, device)
            if acc > best_acc:
                best_acc = acc
                torch.save({"net": model.state_dict(), "acc": acc, "epoch": epoch}, cfg.best_ckpt_path)
            torch.save({"net": model.state_dict(), "acc": acc, "epoch": epoch}, cfg.last_ckpt_path)
            writer.writerow({"epoch": epoch, "train_loss": total_loss / max(1, total), "test_acc": acc, "best_acc": best_acc})
            f.flush()
            print(f"[train] epoch={epoch} test_acc={acc:.4f} best={best_acc:.4f}")
    return clean_targets, noisy_targets, noisy_ids, is_noisy


def load_best_model(cfg: Config, device: torch.device) -> nn.Module:
    model = build_moso_model(cfg).to(device)
    ckpt = torch.load(cfg.best_ckpt_path, map_location=device)
    state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    model.load_state_dict(state)
    return model


def grad_vector_for_sample(model: nn.Module, image: torch.Tensor, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    logits = model(image)
    loss = criterion(logits, target)
    params = [p for p in model.parameters() if p.requires_grad]
    g = grad(loss, params, create_graph=False, retain_graph=False)
    return torch.nn.utils.parameters_to_vector([x.detach() for x in g])


def compute_moso_scores(cfg: Config, device: torch.device, clean_targets, noisy_targets, noisy_ids, is_noisy) -> np.ndarray:
    if cfg.score_path.exists() and not cfg.force:
        print(f"[score] load {cfg.score_path}")
        return np.load(cfg.score_path)
    eval_ds, *_ = build_cifar(cfg, True, True, eval_transform())
    loader = DataLoader(IndexedDataset(eval_ds, return_index=True), batch_size=1, shuffle=False,
                        num_workers=0, pin_memory=False)
    model = load_best_model(cfg, device).eval()
    criterion = nn.CrossEntropyLoss()
    n = len(eval_ds)
    print("[score] first pass: compute average gradient")
    overall_grad = None
    count = 0
    for _, images, targets in tqdm(loader, desc="MoSo overall grad", dynamic_ncols=True):
        images = images.to(device)
        targets = targets.to(device)
        g = grad_vector_for_sample(model, images, targets, criterion)
        count += 1
        if overall_grad is None:
            overall_grad = g.clone()
        else:
            overall_grad.mul_((count - 1) / count).add_(g, alpha=1.0 / count)
        del g
    if overall_grad is None:
        raise RuntimeError("No samples for MoSo scoring.")
    print("[score] second pass: compute MoSo score for each sample")
    scores = np.zeros(n, dtype=np.float32)
    for indices, images, targets in tqdm(loader, desc="MoSo sample scores", dynamic_ncols=True):
        images = images.to(device)
        targets = targets.to(device)
        g = grad_vector_for_sample(model, images, targets, criterion)
        score = ((overall_grad - g / n) * g).sum().detach().cpu().item()
        scores[int(indices.item())] = float(score)
        del g
    cfg.score_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cfg.score_path, scores)
    torch.save(torch.tensor(scores), cfg.run_dir / "score" / "moso_score.pth")
    print(f"[score] saved {cfg.score_path}")
    return scores


def exact_class_top_mask(scores: np.ndarray, labels: np.ndarray, keep_ratio: int, num_classes: int) -> np.ndarray:
    n = len(scores)
    total_k = int(round(n * keep_ratio / 100.0))
    counts = np.asarray([(labels == c).sum() for c in range(num_classes)], dtype=np.int64)
    raw = counts * (keep_ratio / 100.0)
    quotas = np.floor(raw).astype(np.int64)
    rem = total_k - int(quotas.sum())
    if rem > 0:
        for c in np.argsort(-(raw - quotas)):
            if rem <= 0:
                break
            if quotas[c] < counts[c]:
                quotas[c] += 1; rem -= 1
    elif rem < 0:
        for c in np.argsort(raw - quotas):
            if rem >= 0:
                break
            if quotas[c] > 0:
                quotas[c] -= 1; rem += 1
    selected = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        q = int(quotas[c])
        if q <= 0 or len(idx) == 0:
            continue
        order = idx[np.argsort(-scores[idx], kind="mergesort")[:q]]
        selected.extend(order.tolist())
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) != total_k:
        raise RuntimeError(f"Selected {len(selected)}, expected {total_k}")
    mask = np.zeros(n, dtype=np.uint8)
    mask[selected] = 1
    return mask


def save_mask(cfg: Config, keep_ratio: int, mask: np.ndarray, scores: np.ndarray, clean_targets, noisy_targets, noisy_ids, is_noisy) -> dict:
    selected = np.where(mask == 1)[0].astype(np.int64)
    num_noisy_selected = int(is_noisy[selected].sum())
    ratio = float(num_noisy_selected / max(1, len(selected)))
    out_dir = Path(cfg.mask_root) / cfg.dataset / str(cfg.seed) / "MoSo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mask_{keep_ratio}.npz"
    np.savez(out_path, mask=mask.astype(np.uint8), selected_indices=selected, scores=scores.astype(np.float32),
             clean_targets=clean_targets.astype(np.int64), noisy_targets=noisy_targets.astype(np.int64),
             noisy_sample_ids=noisy_ids.astype(np.int64), is_noisy=is_noisy.astype(np.uint8),
             dataset=np.array(cfg.dataset), seed=np.array(cfg.seed), method=np.array("MoSo"), keep_ratio=np.array(keep_ratio),
             num_selected=np.array(len(selected)), num_noisy_selected=np.array(num_noisy_selected),
             noise_ratio_in_mask=np.array(ratio))
    print(f"[mask] kr={keep_ratio} selected={len(selected)} noisy_selected={num_noisy_selected} ratio={ratio:.4f} -> {out_path}")
    return {"dataset": cfg.dataset, "seed": cfg.seed, "method": "MoSo", "keep_ratio": keep_ratio,
            "num_selected": len(selected), "num_noisy_selected": num_noisy_selected,
            "noise_ratio_in_mask": ratio, "mask_path": str(out_path)}


def write_summary(path: Path, rows: List[dict]) -> None:
    fields = ["dataset", "seed", "method", "keep_ratio", "num_selected", "num_noisy_selected", "noise_ratio_in_mask", "mask_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"[summary] {path}")


def append_global_summary(path: Path, rows: List[dict]) -> None:
    fields = ["dataset", "seed", "method", "keep_ratio", "num_selected", "num_noisy_selected", "noise_ratio_in_mask", "mask_path"]
    old = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
    keys = {(str(r["dataset"]), str(r["seed"]), str(r["method"]), str(r["keep_ratio"])) for r in rows}
    old = [r for r in old if (str(r["dataset"]), str(r["seed"]), str(r["method"]), str(r["keep_ratio"])) not in keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(old + rows)


def parse_args():
    p = argparse.ArgumentParser(description="MoSo selection under fixed 20% label noise.")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path.cwd()
    cfg = Config(dataset=args.dataset, seed=args.seed, data_root=str(root / "data"), noise_root=str(root / "noise"),
                 exp_root=str(root / "MoSo" / "noise_exps"), mask_root=str(root / "MoSo" / "noise_masks"), force=args.force)
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_targets, noisy_targets, noisy_ids, is_noisy = train_surrogate(cfg, device)
    scores = compute_moso_scores(cfg, device, clean_targets, noisy_targets, noisy_ids, is_noisy)
    rows = []
    for kr in KEEP_RATIOS:
        mask = exact_class_top_mask(scores, noisy_targets, kr, NUM_CLASSES[cfg.dataset])
        rows.append(save_mask(cfg, kr, mask, scores, clean_targets, noisy_targets, noisy_ids, is_noisy))
    write_summary(cfg.run_summary_path, rows)
    append_global_summary(cfg.global_summary_path, rows)
    print("[done]")


if __name__ == "__main__":
    main()
