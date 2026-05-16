#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noise-label data selection for data_diet baselines.

Place this file at:
    baseline/data_diet/select_from_noise.py

Run from baseline root:
    CUDA_VISIBLE_DEVICES=0 python data_diet/select_from_noise.py --dataset cifar10 --seed 22
    CUDA_VISIBLE_DEVICES=1 python data_diet/select_from_noise.py --dataset cifar100 --seed 42

Directory assumptions:
    baseline/
    ├── data/
    ├── noise/
    │   ├── cifar10/noise_list_22.txt
    │   ├── cifar10/noise_list_42.txt
    │   ├── cifar10/noise_list_96.txt
    │   ├── cifar100/noise_list_22.txt
    │   ├── cifar100/noise_list_42.txt
    │   └── cifar100/noise_list_96.txt
    └── data_diet/
        ├── data_diet/
        │   └── models.py
        └── select_from_noise.py

Main function:
    1. Load CIFAR-10 / CIFAR-100 from baseline/data.
    2. Load fixed noise list from baseline/noise/{dataset}/noise_list_{seed}.txt.
    3. Replace clean training labels with noisy labels.
    4. Sort training data by noisy labels, matching data_diet's class-sorted convention.
    5. Train one proxy model with the same seed for all randomness.
    6. Save checkpoint at score_epoch=20 and at final epoch=200.
    7. Compute E2LN and GraNd at epoch 20.
    8. Compute Forgetting from full training trajectory.
    9. Map all scores back to torchvision original training order.
    10. Generate mask_30.npz, mask_50.npz, mask_70.npz.
    11. Count selected injected-noise samples for each mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parent

# Make sure `from data_diet.models import get_model` works when running from baseline.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_diet.models import get_model  # noqa: E402


# ---------------------------------------------------------------------------
# Fixed experiment defaults
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = {"cifar10", "cifar100"}
VALID_SEEDS = {22, 42, 96}

DEFAULT_KEEP_RATIOS = [30, 50, 70]
DEFAULT_NOISE_RATE = 0.20

DEFAULT_MODEL = "resnet18_lowres"
DEFAULT_EPOCHS = 200
DEFAULT_SCORE_EPOCH = 20

DEFAULT_TRAIN_BATCH_SIZE = 128
DEFAULT_TEST_BATCH_SIZE = 1024

DEFAULT_LR = 0.1
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_NESTEROV = True
DEFAULT_DECAY_FACTOR = 0.2
DEFAULT_DECAY_EPOCHS = [60, 120, 160]

DEFAULT_EL2N_BATCH_SIZE = 1024
DEFAULT_GRAND_BATCH_SIZE = 32

DEFAULT_EVAL_EVERY_EPOCHS = 10


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    dataset: str
    seed: int

    data_root: str
    noise_root: str

    output_root: str
    mask_root: str

    model: str = DEFAULT_MODEL
    epochs: int = DEFAULT_EPOCHS
    score_epoch: int = DEFAULT_SCORE_EPOCH
    noise_rate: float = DEFAULT_NOISE_RATE

    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE
    test_batch_size: int = DEFAULT_TEST_BATCH_SIZE

    lr: float = DEFAULT_LR
    momentum: float = DEFAULT_MOMENTUM
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    nesterov: bool = DEFAULT_NESTEROV
    decay_factor: float = DEFAULT_DECAY_FACTOR
    decay_epochs: Tuple[int, ...] = tuple(DEFAULT_DECAY_EPOCHS)

    keep_ratios: Tuple[int, ...] = tuple(DEFAULT_KEEP_RATIOS)

    el2n_batch_size: int = DEFAULT_EL2N_BATCH_SIZE
    grand_batch_size: int = DEFAULT_GRAND_BATCH_SIZE

    eval_every_epochs: int = DEFAULT_EVAL_EVERY_EPOCHS
    force: bool = False

    @property
    def exp_name(self) -> str:
        return f"{self.dataset}_noise20_seed{self.seed}"

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.dataset / f"seed_{self.seed}"

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "ckpts"

    @property
    def score_dir(self) -> Path:
        return self.run_dir / "scores"

    @property
    def meta_dir(self) -> Path:
        return self.run_dir / "meta"

    @property
    def run_summary_path(self) -> Path:
        return self.run_dir / "summary.csv"

    @property
    def global_summary_path(self) -> Path:
        return Path(self.mask_root) / "summary.csv"

    @property
    def noise_list_path(self) -> Path:
        return Path(self.noise_root) / self.dataset / f"noise_list_{self.seed}.txt"

    @property
    def score_step(self) -> int:
        # It will be filled accurately after loading data because steps_per_epoch
        # depends on the dataset size and train batch size.
        raise RuntimeError("Use score_step_from_steps_per_epoch instead.")

    def score_step_from_steps_per_epoch(self, steps_per_epoch: int) -> int:
        return self.score_epoch * steps_per_epoch

    def final_step_from_steps_per_epoch(self, steps_per_epoch: int) -> int:
        return self.epochs * steps_per_epoch


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_clean_dir(path: Path, force: bool) -> None:
    if path.exists() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_hot(labels: np.ndarray, num_classes: int, dtype=np.float32) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    return (labels[:, None] == np.arange(num_classes)).astype(dtype)


def normalize_cifar_images(x: np.ndarray) -> np.ndarray:
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 1, 1, 3) * 255.0
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 1, 1, 3) * 255.0
    return (x.astype(np.float32) - mean) / std


def augment_cifar_batch(x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Match data_diet's CIFAR augmentation style:
      - reflect padding by 4
      - random 32x32 crop
      - random horizontal flip

    The input is already normalized, following the original data_diet convention.
    """
    xp = np.pad(x, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
    out = np.empty_like(x)

    for i in range(x.shape[0]):
        top = rng.randint(0, 9)
        left = rng.randint(0, 9)
        out[i] = xp[i, top:top + x.shape[1], left:left + x.shape[2], :]

    flip = rng.rand(x.shape[0]) < 0.5
    out[flip] = out[flip, :, ::-1, :]
    return out


def cross_entropy_from_onehot(logits: torch.Tensor, labels_onehot: torch.Tensor) -> torch.Tensor:
    targets = labels_onehot.argmax(dim=-1)
    return F.cross_entropy(logits, targets)


def correct_from_onehot(logits: torch.Tensor, labels_onehot: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1) == labels_onehot.argmax(dim=-1)


def accuracy_from_onehot(logits: torch.Tensor, labels_onehot: torch.Tensor) -> torch.Tensor:
    return correct_from_onehot(logits, labels_onehot).float().mean()


def get_lr(step: int, cfg: ExperimentConfig, steps_per_epoch: int) -> float:
    lr = cfg.lr
    for i, epoch in enumerate(cfg.decay_epochs):
        if step >= epoch * steps_per_epoch:
            lr = cfg.lr * (cfg.decay_factor ** (i + 1))
    return lr


# ---------------------------------------------------------------------------
# Data loading with fixed label noise
# ---------------------------------------------------------------------------

def load_raw_cifar(dataset_name: str, data_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        train_set = datasets.CIFAR10(root=str(data_root), train=True, download=True)
        test_set = datasets.CIFAR10(root=str(data_root), train=False, download=True)
        num_classes = 10
    elif dataset_name == "cifar100":
        train_set = datasets.CIFAR100(root=str(data_root), train=True, download=True)
        test_set = datasets.CIFAR100(root=str(data_root), train=False, download=True)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    x_train = np.asarray(train_set.data)
    y_train = np.asarray(train_set.targets, dtype=np.int64)
    x_test = np.asarray(test_set.data)
    y_test = np.asarray(test_set.targets, dtype=np.int64)

    return x_train, y_train, x_test, y_test, num_classes


def read_noise_list(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Noise list not found: {path}")

    arr = np.loadtxt(path, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Noise list must be a two-column txt file, got shape={arr.shape}: {path}")

    return arr


def apply_noise_and_validate(
    clean_targets: np.ndarray,
    noise_list: np.ndarray,
    num_classes: int,
    expected_noise_rate: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        noisy_targets: original-order noisy labels
        noisy_sample_ids: original-order ids whose labels are modified
        is_noisy: bool array in original training order
    """
    n = clean_targets.shape[0]
    noisy_targets = clean_targets.copy()

    noisy_sample_ids = noise_list[:, 0].astype(np.int64)
    noisy_labels = noise_list[:, 1].astype(np.int64)

    if len(np.unique(noisy_sample_ids)) != len(noisy_sample_ids):
        raise ValueError("Duplicate sample ids found in noise list.")

    if np.any(noisy_sample_ids < 0) or np.any(noisy_sample_ids >= n):
        raise ValueError("Some sample ids in noise list are out of range.")

    if np.any(noisy_labels < 0) or np.any(noisy_labels >= num_classes):
        raise ValueError("Some noisy labels are out of valid class range.")

    old_labels = clean_targets[noisy_sample_ids]
    if np.any(noisy_labels == old_labels):
        bad = int(np.sum(noisy_labels == old_labels))
        raise ValueError(f"{bad} noisy labels are identical to clean labels, which is not allowed.")

    actual_rate = len(noisy_sample_ids) / n
    if abs(actual_rate - expected_noise_rate) > 1e-6:
        raise ValueError(
            f"Noise rate mismatch: expected={expected_noise_rate}, actual={actual_rate}, "
            f"num_noisy={len(noisy_sample_ids)}, num_train={n}"
        )

    noisy_targets[noisy_sample_ids] = noisy_labels

    is_noisy = np.zeros(n, dtype=bool)
    is_noisy[noisy_sample_ids] = True

    return noisy_targets, noisy_sample_ids, is_noisy


@dataclass
class LoadedData:
    x_train_sorted: np.ndarray
    y_train_sorted: np.ndarray
    x_test_sorted: np.ndarray
    y_test_sorted: np.ndarray

    clean_targets_orig: np.ndarray
    noisy_targets_orig: np.ndarray
    noisy_sample_ids_orig: np.ndarray
    is_noisy_orig: np.ndarray

    orig_ids_sorted: np.ndarray
    train_sort_order: np.ndarray

    num_classes: int
    num_train: int
    num_test: int
    steps_per_epoch: int


def load_noisy_data(cfg: ExperimentConfig) -> LoadedData:
    data_root = Path(cfg.data_root)
    noise_list_path = cfg.noise_list_path

    print(f"[DATA] loading dataset={cfg.dataset} from {data_root}")
    x_train_raw, clean_targets, x_test_raw, y_test_clean, num_classes = load_raw_cifar(cfg.dataset, data_root)

    print(f"[NOISE] loading {noise_list_path}")
    noise_list = read_noise_list(noise_list_path)
    noisy_targets, noisy_sample_ids, is_noisy = apply_noise_and_validate(
        clean_targets=clean_targets,
        noise_list=noise_list,
        num_classes=num_classes,
        expected_noise_rate=cfg.noise_rate,
    )

    print(
        f"[NOISE] num_train={len(clean_targets)} | "
        f"num_noisy={len(noisy_sample_ids)} | "
        f"actual_rate={len(noisy_sample_ids) / len(clean_targets):.4f}"
    )

    # Normalize first, matching original data_diet implementation.
    x_train = normalize_cifar_images(x_train_raw)
    x_test = normalize_cifar_images(x_test_raw)

    # Original data_diet sorts training samples by class label.
    # Here class label must be the noisy label, not the clean label.
    train_sort_order = np.argsort(noisy_targets, kind="mergesort")
    orig_ids_sorted = np.arange(len(noisy_targets), dtype=np.int64)[train_sort_order]

    x_train_sorted = x_train[train_sort_order]
    y_train_sorted = one_hot(noisy_targets[train_sort_order], num_classes)

    # Test set stays clean. Sorting test set has no effect on training or final scores,
    # but keeps the convention consistent with data_diet.
    test_sort_order = np.argsort(y_test_clean, kind="mergesort")
    x_test_sorted = x_test[test_sort_order]
    y_test_sorted = one_hot(y_test_clean[test_sort_order], num_classes)

    steps_per_epoch = max(1, int(len(noisy_targets) // cfg.train_batch_size))

    return LoadedData(
        x_train_sorted=x_train_sorted,
        y_train_sorted=y_train_sorted,
        x_test_sorted=x_test_sorted,
        y_test_sorted=y_test_sorted,
        clean_targets_orig=clean_targets,
        noisy_targets_orig=noisy_targets,
        noisy_sample_ids_orig=noisy_sample_ids,
        is_noisy_orig=is_noisy,
        orig_ids_sorted=orig_ids_sorted,
        train_sort_order=train_sort_order,
        num_classes=num_classes,
        num_train=len(noisy_targets),
        num_test=len(y_test_clean),
        steps_per_epoch=steps_per_epoch,
    )


def save_data_meta(cfg: ExperimentConfig, data: LoadedData) -> None:
    cfg.meta_dir.mkdir(parents=True, exist_ok=True)

    np.save(cfg.meta_dir / "clean_targets_orig.npy", data.clean_targets_orig)
    np.save(cfg.meta_dir / "noisy_targets_orig.npy", data.noisy_targets_orig)
    np.save(cfg.meta_dir / "noisy_sample_ids_orig.npy", data.noisy_sample_ids_orig)
    np.save(cfg.meta_dir / "is_noisy_orig.npy", data.is_noisy_orig.astype(np.uint8))
    np.save(cfg.meta_dir / "train_sort_order.npy", data.train_sort_order)
    np.save(cfg.meta_dir / "orig_ids_sorted.npy", data.orig_ids_sorted)

    save_json(
        cfg.meta_dir / "noise_info.json",
        {
            "dataset": cfg.dataset,
            "seed": cfg.seed,
            "noise_rate": cfg.noise_rate,
            "num_train": data.num_train,
            "num_noisy": int(data.noisy_sample_ids_orig.shape[0]),
            "actual_noise_rate": float(data.noisy_sample_ids_orig.shape[0] / data.num_train),
            "noise_list_path": str(cfg.noise_list_path),
        },
    )


# ---------------------------------------------------------------------------
# Model / checkpoint
# ---------------------------------------------------------------------------

def make_model(cfg: ExperimentConfig, num_classes: int) -> torch.nn.Module:
    class Args:
        pass

    args = Args()
    args.model = cfg.model
    args.num_classes = num_classes
    return get_model(args)


def make_optimizer(cfg: ExperimentConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov,
    )


def save_checkpoint(path: Path, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    print(f"[CKPT] saved {path}")


def load_model_from_checkpoint(
    cfg: ExperimentConfig,
    num_classes: int,
    ckpt_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = make_model(cfg, num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    n = x.shape[0]
    total_loss = 0.0
    total_acc = 0.0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        xb = torch.from_numpy(x[start:end]).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
        yb = torch.from_numpy(y[start:end]).to(device=device, dtype=torch.float32)

        logits = model(xb)
        loss = cross_entropy_from_onehot(logits, yb)
        acc = accuracy_from_onehot(logits, yb)

        bs = end - start
        total_loss += float(loss.item()) * bs
        total_acc += float(acc.item()) * bs

    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Forgetting statistics
# ---------------------------------------------------------------------------

@dataclass
class ForgetStats:
    prev_correct: np.ndarray
    num_forgets: np.ndarray
    never_correct_mask: np.ndarray


def init_forget_stats(num_train: int) -> ForgetStats:
    return ForgetStats(
        prev_correct=np.zeros(num_train, dtype=np.int32),
        num_forgets=np.zeros(num_train, dtype=np.float32),
        never_correct_mask=np.ones(num_train, dtype=bool),
    )


def update_forget_stats(stats: ForgetStats, idxs: np.ndarray, correct_now: np.ndarray) -> ForgetStats:
    correct_now = correct_now.astype(np.int32)

    forgotten = stats.prev_correct[idxs] > correct_now
    stats.num_forgets[idxs[forgotten]] += 1.0

    stats.prev_correct[idxs] = correct_now
    stats.never_correct_mask[idxs[correct_now.astype(bool)]] = False

    return stats


def final_forget_scores(stats: ForgetStats) -> np.ndarray:
    scores = stats.num_forgets.copy()
    scores[stats.never_correct_mask] = np.inf
    return scores


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def iter_train_batches(
    x: np.ndarray,
    y: np.ndarray,
    cfg: ExperimentConfig,
    total_steps: int,
) -> Iterable[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Produce batches in the same spirit as data_diet.data.train_batches:
      - random permutation controlled by train seed
      - drop incomplete tail
      - reshuffle when current epoch is exhausted
      - random crop/flip augmentation for CIFAR
    """
    rng = np.random.RandomState(cfg.seed)
    num_examples = x.shape[0]

    order = rng.permutation(num_examples).astype(np.int64)
    start_idx = 0
    curr_step = 1

    while curr_step <= total_steps:
        end_idx = start_idx + cfg.train_batch_size

        if end_idx > num_examples:
            order = rng.permutation(num_examples).astype(np.int64)
            start_idx = 0
            continue

        idxs = order[start_idx:end_idx]
        xb = x[idxs].copy()
        yb = y[idxs]

        xb = augment_cifar_batch(xb, rng)

        yield curr_step, idxs, xb, yb

        curr_step += 1
        start_idx = end_idx


def train_proxy_model(cfg: ExperimentConfig, data: LoadedData, device: torch.device) -> None:
    score_step = cfg.score_step_from_steps_per_epoch(data.steps_per_epoch)
    final_step = cfg.final_step_from_steps_per_epoch(data.steps_per_epoch)

    score_ckpt_path = cfg.ckpt_dir / f"checkpoint_{score_step}.pt"
    final_ckpt_path = cfg.ckpt_dir / f"checkpoint_{final_step}.pt"
    forget_path = cfg.score_dir / "forgetting_sorted.npy"

    if (
        score_ckpt_path.exists()
        and final_ckpt_path.exists()
        and forget_path.exists()
        and not cfg.force
    ):
        print("[TRAIN] existing checkpoints and forgetting scores found; skip training.")
        print(f"        score checkpoint: {score_ckpt_path}")
        print(f"        final checkpoint: {final_ckpt_path}")
        print(f"        forgetting scores: {forget_path}")
        return

    print("[TRAIN] start proxy model training")
    print(f"        model={cfg.model}")
    print(f"        epochs={cfg.epochs}, steps_per_epoch={data.steps_per_epoch}, total_steps={final_step}")
    print(f"        score_epoch={cfg.score_epoch}, score_step={score_step}")

    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.score_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(cfg.seed)
    model = make_model(cfg, data.num_classes).to(device)
    optimizer = make_optimizer(cfg, model)

    stats = init_forget_stats(data.num_train)

    metrics_path = cfg.run_dir / "train_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "test_loss",
                "test_acc",
                "elapsed_sec",
            ],
        )
        writer.writeheader()

        t0 = time.time()
        pbar = tqdm(
            iter_train_batches(data.x_train_sorted, data.y_train_sorted, cfg, final_step),
            total=final_step,
            desc=f"train:{cfg.dataset}:seed{cfg.seed}",
            dynamic_ncols=True,
        )

        for step, idxs, xb_np, yb_np in pbar:
            model.train()

            xb = torch.from_numpy(xb_np).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(yb_np).to(device=device, dtype=torch.float32)

            lr = get_lr(step, cfg, data.steps_per_epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = cross_entropy_from_onehot(logits, yb)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_acc = accuracy_from_onehot(logits, yb)
                correct_now = correct_from_onehot(logits, yb).int().detach().cpu().numpy()

            stats = update_forget_stats(stats, idxs, correct_now)

            epoch_float = step / data.steps_per_epoch
            pbar.set_postfix(
                step=step,
                epoch=f"{epoch_float:.1f}",
                loss=f"{loss.item():.4f}",
                acc=f"{train_acc.item():.3f}",
                lr=f"{lr:.5f}",
            )

            should_eval = (
                step == 1
                or step == score_step
                or step == final_step
                or step % (cfg.eval_every_epochs * data.steps_per_epoch) == 0
            )

            test_loss = math.nan
            test_acc = math.nan

            if should_eval:
                test_loss, test_acc = evaluate(
                    model=model,
                    x=data.x_test_sorted,
                    y=data.y_test_sorted,
                    batch_size=cfg.test_batch_size,
                    device=device,
                )

                writer.writerow(
                    {
                        "step": step,
                        "epoch": step / data.steps_per_epoch,
                        "lr": lr,
                        "train_loss": float(loss.item()),
                        "train_acc": float(train_acc.item()),
                        "test_loss": float(test_loss),
                        "test_acc": float(test_acc),
                        "elapsed_sec": time.time() - t0,
                    }
                )
                f.flush()

                print(
                    f"[EVAL] step={step:6d} epoch={step / data.steps_per_epoch:7.2f} "
                    f"train_acc={float(train_acc.item()):.4f} "
                    f"test_acc={test_acc:.4f} lr={lr:.5f}"
                )

            if step == score_step:
                save_checkpoint(score_ckpt_path, step, model, optimizer)

            if step == final_step:
                save_checkpoint(final_ckpt_path, step, model, optimizer)

        pbar.close()

    forget_scores_sorted = final_forget_scores(stats)
    np.save(forget_path, forget_scores_sorted)

    # Save a copy using the original data_diet naming style as well.
    np.save(cfg.score_dir / f"forget_scores_ckpt_{final_step}.npy", forget_scores_sorted)

    print(f"[FORGET] saved sorted-order forgetting scores: {forget_path}")


# ---------------------------------------------------------------------------
# E2LN / GraNd scoring
# ---------------------------------------------------------------------------

def to_torch_batch(x: np.ndarray, y: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xb = torch.from_numpy(x).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    yb = torch.from_numpy(y).to(device=device, dtype=torch.float32)
    return xb, yb


@torch.no_grad()
def el2n_scores_for_batch(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    xb, yb = to_torch_batch(x, y, device)
    probs = F.softmax(model(xb), dim=-1)
    scores = torch.linalg.norm(probs - yb, dim=-1)
    return scores.detach().cpu().numpy().astype(np.float32)


def grand_scores_for_batch(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    xb, yb = to_torch_batch(x, y, device)
    targets = yb.argmax(dim=-1)

    # Try to match the original vectorized torch.func implementation.
    try:
        from torch.func import functional_call, grad, vmap

        params = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}

        def loss_single(params_, buffers_, x_single, y_single):
            logits = functional_call(model, (params_, buffers_), (x_single.unsqueeze(0),))
            return F.cross_entropy(logits, y_single.unsqueeze(0))

        per_grad = vmap(
            grad(loss_single),
            in_dims=(None, None, 0, 0),
        )(params, buffers, xb, targets)

        sq = torch.zeros(xb.shape[0], device=device)
        for g in per_grad.values():
            sq += g.reshape(g.shape[0], -1).pow(2).sum(dim=1)

        return torch.sqrt(sq).detach().cpu().numpy().astype(np.float32)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("[WARN] torch.func GraNd OOM; falling back to per-sample gradients.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"[WARN] torch.func GraNd failed: {repr(e)}")
            print("[WARN] falling back to per-sample gradients.")

    except Exception as e:
        print(f"[WARN] torch.func GraNd failed: {repr(e)}")
        print("[WARN] falling back to per-sample gradients.")

    # Robust fallback: compute one sample at a time.
    scores: List[float] = []
    model.zero_grad(set_to_none=True)

    for i in range(xb.shape[0]):
        logits = model(xb[i:i + 1])
        loss = F.cross_entropy(logits, targets[i:i + 1])
        grads = torch.autograd.grad(
            loss,
            model.parameters(),
            retain_graph=False,
            create_graph=False,
        )
        gnorm = torch.sqrt(sum(g.pow(2).sum() for g in grads)).item()
        scores.append(float(gnorm))

    return np.asarray(scores, dtype=np.float32)


def compute_scores_sorted(
    score_type: str,
    model: torch.nn.Module,
    data: LoadedData,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    n = data.num_train
    outputs: List[np.ndarray] = []

    pbar = tqdm(
        range(0, n, batch_size),
        desc=f"score:{score_type}",
        dynamic_ncols=True,
    )

    for start in pbar:
        end = min(start + batch_size, n)
        x = data.x_train_sorted[start:end]
        y = data.y_train_sorted[start:end]

        if score_type == "el2n":
            outputs.append(el2n_scores_for_batch(model, x, y, device))
        elif score_type == "grand":
            outputs.append(grand_scores_for_batch(model, x, y, device))
        else:
            raise ValueError(f"Unknown score_type: {score_type}")

    scores = np.concatenate(outputs, axis=0)
    if scores.shape[0] != n:
        raise RuntimeError(f"Score length mismatch: got {scores.shape[0]}, expected {n}")

    return scores.astype(np.float32)


def sorted_scores_to_original(scores_sorted: np.ndarray, orig_ids_sorted: np.ndarray) -> np.ndarray:
    scores_orig = np.empty_like(scores_sorted)
    scores_orig[orig_ids_sorted] = scores_sorted
    return scores_orig


def compute_all_scores(cfg: ExperimentConfig, data: LoadedData, device: torch.device) -> Dict[str, np.ndarray]:
    score_step = cfg.score_step_from_steps_per_epoch(data.steps_per_epoch)
    final_step = cfg.final_step_from_steps_per_epoch(data.steps_per_epoch)

    score_ckpt_path = cfg.ckpt_dir / f"checkpoint_{score_step}.pt"
    if not score_ckpt_path.exists():
        raise FileNotFoundError(f"Score checkpoint not found: {score_ckpt_path}")

    score_paths_orig = {
        "E2LN": cfg.score_dir / "E2LN_scores_orig.npy",
        "GraNd": cfg.score_dir / "GraNd_scores_orig.npy",
        "Forgetting": cfg.score_dir / "Forgetting_scores_orig.npy",
    }

    if all(p.exists() for p in score_paths_orig.values()) and not cfg.force:
        print("[SCORE] existing original-order scores found; load them.")
        return {method: np.load(path) for method, path in score_paths_orig.items()}

    model = load_model_from_checkpoint(
        cfg=cfg,
        num_classes=data.num_classes,
        ckpt_path=score_ckpt_path,
        device=device,
    )

    # E2LN
    e2ln_sorted_path = cfg.score_dir / f"E2LN_scores_sorted_ckpt_{score_step}.npy"
    e2ln_orig_path = score_paths_orig["E2LN"]

    if e2ln_orig_path.exists() and not cfg.force:
        e2ln_orig = np.load(e2ln_orig_path)
    else:
        print(f"[SCORE] computing E2LN at step={score_step}")
        e2ln_sorted = compute_scores_sorted(
            score_type="el2n",
            model=model,
            data=data,
            batch_size=cfg.el2n_batch_size,
            device=device,
        )
        e2ln_orig = sorted_scores_to_original(e2ln_sorted, data.orig_ids_sorted)
        np.save(e2ln_sorted_path, e2ln_sorted)
        np.save(e2ln_orig_path, e2ln_orig)

    # GraNd
    grand_sorted_path = cfg.score_dir / f"GraNd_scores_sorted_ckpt_{score_step}.npy"
    grand_orig_path = score_paths_orig["GraNd"]

    if grand_orig_path.exists() and not cfg.force:
        grand_orig = np.load(grand_orig_path)
    else:
        print(f"[SCORE] computing GraNd at step={score_step}")
        grand_sorted = compute_scores_sorted(
            score_type="grand",
            model=model,
            data=data,
            batch_size=cfg.grand_batch_size,
            device=device,
        )
        grand_orig = sorted_scores_to_original(grand_sorted, data.orig_ids_sorted)
        np.save(grand_sorted_path, grand_sorted)
        np.save(grand_orig_path, grand_orig)

    # Forgetting
    forget_sorted_path = cfg.score_dir / "forgetting_sorted.npy"
    forget_orig_path = score_paths_orig["Forgetting"]

    if not forget_sorted_path.exists():
        raise FileNotFoundError(f"Sorted forgetting score not found: {forget_sorted_path}")

    if forget_orig_path.exists() and not cfg.force:
        forget_orig = np.load(forget_orig_path)
    else:
        print(f"[SCORE] converting Forgetting scores from sorted order to original order, final_step={final_step}")
        forget_sorted = np.load(forget_sorted_path).astype(np.float32)
        forget_orig = sorted_scores_to_original(forget_sorted, data.orig_ids_sorted)
        np.save(forget_orig_path, forget_orig)

    return {
        "E2LN": e2ln_orig.astype(np.float32),
        "GraNd": grand_orig.astype(np.float32),
        "Forgetting": forget_orig.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Mask generation and summaries
# ---------------------------------------------------------------------------

def make_keep_high_mask(scores_orig: np.ndarray, keep_ratio_percent: int) -> Tuple[np.ndarray, np.ndarray]:
    n = scores_orig.shape[0]
    k = int(round(n * keep_ratio_percent / 100.0))

    if k <= 0 or k > n:
        raise ValueError(f"Invalid keep ratio: {keep_ratio_percent}, selected k={k}, n={n}")

    # Higher score is kept, matching original formal mask script's keep_high behavior.
    order_desc = np.argsort(-scores_orig, kind="mergesort")
    selected_indices = np.sort(order_desc[:k]).astype(np.int64)

    mask = np.zeros(n, dtype=np.uint8)
    mask[selected_indices] = 1

    return mask, selected_indices


def save_mask_and_get_row(
    cfg: ExperimentConfig,
    data: LoadedData,
    method: str,
    scores_orig: np.ndarray,
    keep_ratio: int,
) -> dict:
    mask, selected_indices = make_keep_high_mask(scores_orig, keep_ratio)

    selected_noisy = data.is_noisy_orig[selected_indices]
    num_selected = int(mask.sum())
    num_noisy_selected = int(selected_noisy.sum())
    noise_ratio_in_mask = float(num_noisy_selected / max(1, num_selected))

    out_dir = Path(cfg.mask_root) / cfg.dataset / str(cfg.seed) / method
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"mask_{keep_ratio}.npz"

    np.savez(
        out_path,
        mask=mask.astype(np.uint8),
        selected_indices=selected_indices.astype(np.int64),
        scores=scores_orig.astype(np.float32),
        noisy_sample_ids=data.noisy_sample_ids_orig.astype(np.int64),
        is_noisy=data.is_noisy_orig.astype(np.uint8),
        clean_targets=data.clean_targets_orig.astype(np.int64),
        noisy_targets=data.noisy_targets_orig.astype(np.int64),
        dataset=np.array(cfg.dataset),
        method=np.array(method),
        seed=np.array(cfg.seed),
        keep_ratio=np.array(keep_ratio),
        num_selected=np.array(num_selected),
        num_noisy_total=np.array(len(data.noisy_sample_ids_orig)),
        num_noisy_selected=np.array(num_noisy_selected),
        noise_ratio_total=np.array(len(data.noisy_sample_ids_orig) / data.num_train),
        noise_ratio_in_mask=np.array(noise_ratio_in_mask),
    )

    print(
        f"[MASK] method={method:10s} kr={keep_ratio:2d} "
        f"selected={num_selected:5d} "
        f"noisy_selected={num_noisy_selected:5d} "
        f"noise_ratio_in_mask={noise_ratio_in_mask:.4f} "
        f"path={out_path}"
    )

    return {
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "method": method,
        "keep_ratio": keep_ratio,
        "num_train": data.num_train,
        "num_selected": num_selected,
        "num_noisy_total": int(len(data.noisy_sample_ids_orig)),
        "noise_ratio_total": float(len(data.noisy_sample_ids_orig) / data.num_train),
        "num_noisy_selected": num_noisy_selected,
        "noise_ratio_in_mask": noise_ratio_in_mask,
        "mask_path": str(out_path),
    }


def write_summary_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "seed",
        "method",
        "keep_ratio",
        "num_train",
        "num_selected",
        "num_noisy_total",
        "noise_ratio_total",
        "num_noisy_selected",
        "noise_ratio_in_mask",
        "mask_path",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[SUMMARY] saved {path}")


def update_global_summary(path: Path, new_rows: List[dict]) -> None:
    """
    Update global summary by removing rows with the same
    dataset/seed/method/keep_ratio and appending the new rows.
    """
    if not new_rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    key_fields = ["dataset", "seed", "method", "keep_ratio"]
    all_rows: List[dict] = []

    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)

    new_keys = {
        tuple(str(row[k]) for k in key_fields)
        for row in new_rows
    }

    kept_rows = []
    for row in all_rows:
        key = tuple(str(row[k]) for k in key_fields)
        if key not in new_keys:
            kept_rows.append(row)

    merged = kept_rows + new_rows

    fieldnames = [
        "dataset",
        "seed",
        "method",
        "keep_ratio",
        "num_train",
        "num_selected",
        "num_noisy_total",
        "noise_ratio_total",
        "num_noisy_selected",
        "noise_ratio_in_mask",
        "mask_path",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged:
            writer.writerow(row)

    print(f"[SUMMARY] updated global summary: {path}")


def generate_masks(cfg: ExperimentConfig, data: LoadedData, scores: Dict[str, np.ndarray]) -> None:
    rows: List[dict] = []

    for method in ["E2LN", "GraNd", "Forgetting"]:
        method_scores = scores[method]

        if method_scores.shape[0] != data.num_train:
            raise RuntimeError(
                f"{method} score length mismatch: got {method_scores.shape[0]}, expected={data.num_train}"
            )

        for kr in cfg.keep_ratios:
            row = save_mask_and_get_row(
                cfg=cfg,
                data=data,
                method=method,
                scores_orig=method_scores,
                keep_ratio=int(kr),
            )
            rows.append(row)

    write_summary_csv(cfg.run_summary_path, rows)
    update_global_summary(cfg.global_summary_path, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E2LN/GraNd/Forgetting selection on fixed label-noise CIFAR datasets."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(SUPPORTED_DATASETS),
        help="Dataset name: cifar10 or cifar100.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=sorted(VALID_SEEDS),
        help="Experiment seed. Must be one of 22, 42, 96.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun and overwrite existing checkpoints/scores under this experiment directory.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    dataset = args.dataset.lower()
    seed = int(args.seed)

    data_root = BASELINE_ROOT / "data"
    noise_root = BASELINE_ROOT / "noise"

    output_root = SCRIPT_DIR / "noise_exps"
    mask_root = SCRIPT_DIR / "noise_masks"

    return ExperimentConfig(
        dataset=dataset,
        seed=seed,
        data_root=str(data_root),
        noise_root=str(noise_root),
        output_root=str(output_root),
        mask_root=str(mask_root),
        force=bool(args.force),
    )


def validate_runtime_environment(cfg: ExperimentConfig) -> None:
    baseline_cwd = Path.cwd().resolve()
    expected = BASELINE_ROOT.resolve()

    if baseline_cwd != expected:
        print(f"[WARN] current working directory is {baseline_cwd}")
        print(f"[WARN] expected baseline root is      {expected}")
        print("[WARN] script will still use paths relative to its own location.")

    if not Path(cfg.data_root).exists():
        raise FileNotFoundError(f"Data root not found: {cfg.data_root}")

    if not Path(cfg.noise_root).exists():
        raise FileNotFoundError(f"Noise root not found: {cfg.noise_root}")

    if not cfg.noise_list_path.exists():
        raise FileNotFoundError(f"Noise list not found: {cfg.noise_list_path}")


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    validate_runtime_environment(cfg)

    print("[CONFIG]")
    print(json.dumps(asdict(cfg), indent=4, ensure_ascii=False))
    print(f"[DEVICE] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(f"[DEVICE] torch.cuda.is_available={torch.cuda.is_available()}")

    ensure_clean_dir(cfg.run_dir, force=cfg.force)
    ensure_clean_dir(cfg.ckpt_dir, force=False)
    ensure_clean_dir(cfg.score_dir, force=False)
    ensure_clean_dir(cfg.meta_dir, force=False)
    Path(cfg.mask_root).mkdir(parents=True, exist_ok=True)

    save_json(cfg.run_dir / "config.json", asdict(cfg))

    set_all_seeds(cfg.seed)
    device = get_device()
    print(f"[DEVICE] using {device}")

    data = load_noisy_data(cfg)
    save_data_meta(cfg, data)

    print("[DATA SUMMARY]")
    print(f"  dataset          = {cfg.dataset}")
    print(f"  num_classes      = {data.num_classes}")
    print(f"  num_train        = {data.num_train}")
    print(f"  num_test         = {data.num_test}")
    print(f"  steps_per_epoch  = {data.steps_per_epoch}")
    print(f"  score_step       = {cfg.score_step_from_steps_per_epoch(data.steps_per_epoch)}")
    print(f"  final_step       = {cfg.final_step_from_steps_per_epoch(data.steps_per_epoch)}")
    print(f"  noisy_samples    = {len(data.noisy_sample_ids_orig)}")

    train_proxy_model(cfg, data, device)

    scores = compute_all_scores(cfg, data, device)

    generate_masks(cfg, data, scores)

    print("[DONE]")
    print(f"  run_dir  = {cfg.run_dir}")
    print(f"  mask_dir = {Path(cfg.mask_root) / cfg.dataset / str(cfg.seed)}")
    print(f"  summary  = {cfg.run_summary_path}")


if __name__ == "__main__":
    main()