#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate CIFAR-100 data-selection runtime for smartlixiang/baseline.

Run this file from the repository root, for example:

    python estimate_cifar100_runtime.py --device cuda:0

The estimator benchmarks short, method-specific computation kernels on the
current machine and extrapolates them to the normal CIFAR-100 scripts in this
repository. Dataset download, dataset construction, checkpoint loading, and
other one-off data-loading work are deliberately excluded.

Default wall-clock assumptions:
  * target keep ratios: 60%, 70%, 80%, 90%; ratios run sequentially;
  * four independent RTX 3090 machines are available;
  * only repetitions/trials inside one ratio may run in parallel;
  * 3-4 repeated runs therefore occupy one wave; MoSo's 8 trials occupy two;
  * MoSo is conservatively rerun for every target keep ratio, as requested.

The output contains per-stage estimates, method totals, JSON, and CSV files.
It does not modify method checkpoints, masks, caches, or experiment outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gc
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required to run this estimator.") from exc


N_TRAIN = 50_000
N_TEST = 10_000
N_CLASSES = 100
CIFAR_SHAPE = (3, 32, 32)
DEFAULT_RATIOS = (60, 70, 80, 90)
ALL_METHODS = (
    "Random",
    "Herding",
    "EL2N",
    "GraNd",
    "Forgetting",
    "MDS",
    "MoSo",
    "YangCLIP",
    "RLSelector",
)


@dataclass
class StageEstimate:
    method: str
    stage: str
    ratio: Optional[int]
    measured_unit_seconds: float
    multiplier: float
    estimated_seconds: float
    parallel_waves: int
    confidence: str
    note: str


@dataclass
class MethodEstimate:
    method: str
    estimated_seconds: float
    stages: list[StageEstimate]
    status: str = "ok"
    warning: str = ""


class RuntimeEstimatorError(RuntimeError):
    pass


def load_module_from_path(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def temporary_sys_path(path: Path):
    value = str(path.resolve())
    sys.path.insert(0, value)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(value)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_torch(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def set_cudnn_mode(deterministic: bool) -> None:
    if not hasattr(torch.backends, "cudnn"):
        return
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def timed_repeated(
    fn: Callable[[], Any],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    for _ in range(max(0, warmup)):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / repeats


def cpu_timed(fn: Callable[[], Any], repeats: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(max(1, repeats)):
        fn()
    return (time.perf_counter() - start) / max(1, repeats)


def waves(repetitions: int, machines: int) -> int:
    return max(1, math.ceil(repetitions / max(1, machines)))


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "n/a"
    seconds = max(0.0, seconds)
    days, rem = divmod(seconds, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    if days >= 1:
        return f"{int(days)}d {int(hours):02d}h {int(minutes):02d}m"
    if hours >= 1:
        return f"{int(hours)}h {int(minutes):02d}m {secs:04.1f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {secs:04.1f}s"
    return f"{secs:.3f}s"


def make_cpu_batch(
    batch_size: int,
    image_hw: int = 32,
    num_classes: int = N_CLASSES,
    pin_memory: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, 3, image_hw, image_hw, dtype=torch.float32)
    y = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
    if pin_memory:
        x = x.pin_memory()
        y = y.pin_memory()
    return x, y


def model_logits(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def benchmark_train_batch(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    warmup: int,
    repeats: int,
    forgetting_tracking: bool = False,
    output_logits_index: int = 0,
) -> float:
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    x_cpu, y_cpu = make_cpu_batch(batch_size, pin_memory=device.type == "cuda")
    tracked = np.zeros(batch_size, dtype=np.int64)

    def step() -> None:
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        if isinstance(out, tuple):
            logits = out[output_logits_index]
        else:
            logits = out
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        if forgetting_tracking:
            correct = logits.detach().argmax(dim=1).eq(y).int().cpu().numpy()
            np.copyto(tracked, correct)

    return timed_repeated(step, device, warmup, repeats)


def benchmark_eval_batch(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    warmup: int,
    repeats: int,
    output_logits_index: int = 0,
) -> float:
    model = model.to(device)
    model.eval()
    x_cpu, _ = make_cpu_batch(batch_size, pin_memory=device.type == "cuda")

    def step() -> None:
        x = x_cpu.to(device, non_blocking=True)
        with torch.no_grad():
            out = model(x)
            logits = out[output_logits_index] if isinstance(out, tuple) else out
            _ = logits.detach().cpu()

    return timed_repeated(step, device, warmup, repeats)


def benchmark_feature_batch(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    image_hw: int,
    warmup: int,
    repeats: int,
    normalize: bool = False,
) -> float:
    model = model.to(device)
    model.eval()
    x_cpu, _ = make_cpu_batch(batch_size, image_hw=image_hw, pin_memory=device.type == "cuda")

    def step() -> None:
        x = x_cpu.to(device, non_blocking=True)
        with torch.no_grad():
            feat = model(x)
            if isinstance(feat, tuple):
                feat = feat[0]
            if normalize:
                feat = F.normalize(feat, dim=1)
            _ = feat.detach().cpu()

    return timed_repeated(step, device, warmup, repeats)


def benchmark_sort_masks(ratios: Sequence[int], keep_high: bool = True) -> float:
    rng = np.random.default_rng(2026)
    scores = rng.standard_normal(N_TRAIN, dtype=np.float32)

    def run() -> None:
        for ratio in ratios:
            keep_n = int(round(N_TRAIN * ratio / 100.0))
            order = np.argsort(scores)
            keep_idx = order[-keep_n:] if keep_high else order[:keep_n]
            keep_idx = np.sort(keep_idx)
            mask = np.zeros(N_TRAIN, dtype=np.bool_)
            mask[keep_idx] = True
            _ = int(mask.sum())

    return cpu_timed(run, repeats=3)


def benchmark_mean_and_masks(runs: int, ratios: Sequence[int]) -> float:
    rng = np.random.default_rng(2026)
    arrays = rng.standard_normal((runs, N_TRAIN), dtype=np.float32)

    def run() -> None:
        scores = arrays.mean(axis=0)
        for ratio in ratios:
            keep_n = int(round(N_TRAIN * ratio / 100.0))
            order = np.argsort(scores)
            idx = order[-keep_n:]
            mask = np.zeros(N_TRAIN, dtype=np.uint8)
            mask[idx] = 1
            _ = int(mask.sum())

    return cpu_timed(run, repeats=3)


def check_repo(repo_root: Path) -> dict[str, bool]:
    required = {
        "data_diet": repo_root / "data_diet" / "data_diet" / "models.py",
        "herding": repo_root / "herding" / "herding_select.py",
        "MDS": repo_root / "MDS" / "run_mds.py",
        "MoSo": repo_root / "MoSo" / "surrogate_training.py",
        "YangCLIP": repo_root / "YangCLIP" / "run_all.sh",
        "RLSelector": repo_root / "RLSelector" / "train.py",
    }
    return {name: path.is_file() for name, path in required.items()}


def estimate_random(args, device: torch.device) -> MethodEstimate:
    rng = np.random.default_rng(args.seed)

    def generate() -> None:
        for ratio in args.ratios:
            k = int(round(N_TRAIN * ratio / 100.0))
            selected = rng.choice(N_TRAIN, size=k, replace=False)
            mask = np.zeros(N_TRAIN, dtype=np.uint8)
            mask[selected] = 1
            _ = int(mask.sum())

    elapsed = cpu_timed(generate, repeats=5)
    seed_waves = waves(args.seed_runs, args.machines)
    total = elapsed * seed_waves
    stage = StageEstimate(
        method="Random",
        stage="sample_and_build_four_masks",
        ratio=None,
        measured_unit_seconds=elapsed,
        multiplier=seed_waves,
        estimated_seconds=total,
        parallel_waves=seed_waves,
        confidence="high",
        note=f"{args.seed_runs} seeds occupy {seed_waves} parallel wave(s).",
    )
    return MethodEstimate("Random", total, [stage])


def estimate_herding(args, device: torch.device) -> MethodEstimate:
    model_mod = load_module_from_path(
        "runtime_herding_models", args.repo_root / "herding" / "models.py"
    )
    set_cudnn_mode(deterministic=True)
    model = model_mod.ResNet18FeatureExtractor(prefer_pretrained=False)
    batch_sec = benchmark_feature_batch(
        model,
        device,
        batch_size=128,
        image_hw=32,
        warmup=args.warmup,
        repeats=args.benchmark_batches,
        normalize=True,
    )
    seed_waves = waves(args.seed_runs, args.machines)
    extraction = batch_sec * math.ceil(N_TRAIN / 128) * seed_waves
    stages = [
        StageEstimate(
            "Herding",
            "resnet18_feature_extraction",
            None,
            batch_sec,
            math.ceil(N_TRAIN / 128) * seed_waves,
            extraction,
            seed_waves,
            "medium-high",
            f"ImageNet weight loading is excluded; {args.seed_runs} seeds occupy {seed_waves} wave(s).",
        )
    ]

    d = 512
    probe_classes = args.herding_probe_classes
    probe_n = args.herding_probe_class_size
    rng = np.random.default_rng(args.seed)

    def one_ratio_probe(ratio: int) -> float:
        features = torch.from_numpy(
            rng.standard_normal((probe_classes * probe_n, d), dtype=np.float32)
        )
        labels = torch.arange(probe_classes).repeat_interleave(probe_n)
        keep = ratio / 100.0

        def run() -> None:
            selected_global = torch.zeros(features.shape[0], dtype=torch.bool)
            for class_id in range(probe_classes):
                class_indices = torch.where(labels == class_id)[0]
                class_features = features[class_indices]
                class_count = class_features.shape[0]
                target_count = max(1, min(class_count, int(round(class_count * keep))))
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
            _ = int(selected_global.sum())

        measured = cpu_timed(run, repeats=1)
        full_n = N_TRAIN // N_CLASSES
        probe_m = max(1, int(round(probe_n * keep)))
        full_m = max(1, int(round(full_n * keep)))

        def work(n: int, m: int) -> float:
            return m * (2 * n - m + 1) / 2.0

        scale = (N_CLASSES / probe_classes) * (work(full_n, full_m) / work(probe_n, probe_m))
        scale *= seed_waves
        estimated = measured * scale
        stages.append(
            StageEstimate(
                "Herding",
                "classwise_greedy_selection",
                ratio,
                measured,
                scale,
                estimated,
                seed_waves,
                "medium",
                f"Scaled by the exact candidate-comparison count and {seed_waves} seed wave(s), from {probe_classes}x{probe_n} probe classes to 100x500.",
            )
        )
        return estimated

    selection_total = sum(one_ratio_probe(r) for r in args.ratios)
    total = extraction + selection_total
    clear_torch(device)
    return MethodEstimate("Herding", total, stages)


def load_data_diet_modules(repo_root: Path):
    models = load_module_from_path(
        "runtime_data_diet_models", repo_root / "data_diet" / "data_diet" / "models.py"
    )
    scores = load_module_from_path(
        "runtime_data_diet_scores", repo_root / "data_diet" / "data_diet" / "scores.py"
    )
    return models, scores


def make_data_diet_model(models_mod) -> nn.Module:
    return models_mod.ResNet18(num_classes=N_CLASSES, lowres=True)


def benchmark_data_diet_train(
    models_mod,
    device: torch.device,
    args,
    forgetting: bool,
) -> tuple[float, float]:
    model = make_data_diet_model(models_mod)
    train_batch = benchmark_train_batch(
        model,
        device,
        batch_size=32,
        warmup=args.warmup,
        repeats=args.benchmark_batches,
        forgetting_tracking=forgetting,
    )
    eval_batch = benchmark_eval_batch(
        model,
        device,
        batch_size=128,
        warmup=args.warmup,
        repeats=max(2, args.benchmark_batches // 2),
    )
    return train_batch, eval_batch


def estimate_el2n(args, device: torch.device) -> MethodEstimate:
    models_mod, scores_mod = load_data_diet_modules(args.repo_root)
    set_cudnn_mode(deterministic=True)
    train_batch, eval_batch = benchmark_data_diet_train(models_mod, device, args, False)
    train_epoch = train_batch * math.ceil(N_TRAIN / 32)
    eval_epoch = eval_batch * math.ceil(N_TEST / 128)
    run_waves = waves(args.data_diet_runs, args.machines)
    train_epochs = args.el2n_score_epoch
    training = train_epoch * train_epochs * run_waves
    validation = eval_epoch * (train_epochs + 1) * run_waves if args.include_proxy_validation else 0.0

    model = make_data_diet_model(models_mod).to(device).eval()
    score_batch_size = 100
    x = np.random.default_rng(args.seed).standard_normal(
        (score_batch_size, 32, 32, 3), dtype=np.float32
    )
    targets = np.random.default_rng(args.seed + 1).integers(0, N_CLASSES, score_batch_size)
    y = np.eye(N_CLASSES, dtype=np.float32)[targets]

    def score() -> None:
        _ = scores_mod._el2n_scores(model, x, y, device)

    score_batch = timed_repeated(score, device, args.warmup, max(2, args.benchmark_batches // 2))
    scoring = score_batch * math.ceil(N_TRAIN / score_batch_size) * run_waves
    aggregate = benchmark_mean_and_masks(args.data_diet_runs, args.ratios)

    stages = [
        StageEstimate("EL2N", "proxy_training_to_score_epoch", None, train_batch, math.ceil(N_TRAIN / 32) * train_epochs * run_waves, train_epoch * train_epochs * run_waves, run_waves, "medium-high", f"Score checkpoint is epoch {train_epochs}; {args.data_diet_runs} runs use {run_waves} parallel wave(s)."),
        *([StageEstimate("EL2N", "proxy_validation", None, eval_batch, math.ceil(N_TEST / 128) * (train_epochs + 1) * run_waves, validation, run_waves, "medium", "Optional literal-script overhead: initial validation plus one validation per epoch.")] if args.include_proxy_validation else []),
        StageEstimate("EL2N", "el2n_full_dataset_scoring", None, score_batch, math.ceil(N_TRAIN / score_batch_size) * run_waves, scoring, run_waves, "medium-high", "Per-run scoring is parallel across the available machines."),
        StageEstimate("EL2N", "mean_aggregation_and_four_masks", None, aggregate, 1.0, aggregate, 1, "high", "The repository sorts independently for every keep ratio."),
    ]
    total = training + validation + scoring + aggregate
    clear_torch(device)
    return MethodEstimate("EL2N", total, stages)


def estimate_grand(args, device: torch.device) -> MethodEstimate:
    models_mod, scores_mod = load_data_diet_modules(args.repo_root)
    set_cudnn_mode(deterministic=True)
    train_batch, eval_batch = benchmark_data_diet_train(models_mod, device, args, False)
    train_epoch = train_batch * math.ceil(N_TRAIN / 32)
    eval_epoch = eval_batch * math.ceil(N_TEST / 128)
    run_waves = waves(args.data_diet_runs, args.machines)
    train_epochs = args.grand_score_epoch
    training = train_epoch * train_epochs * run_waves
    validation = eval_epoch * (train_epochs + 1) * run_waves if args.include_proxy_validation else 0.0

    model = make_data_diet_model(models_mod).to(device).eval()
    score_batch_size = 5
    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((score_batch_size, 32, 32, 3), dtype=np.float32)
    labels = rng.integers(0, N_CLASSES, score_batch_size)
    y = np.eye(N_CLASSES, dtype=np.float32)[labels]

    def score() -> None:
        _ = scores_mod._grand_scores(model, x, y, device)

    score_batch = timed_repeated(score, device, warmup=1, repeats=max(1, args.grand_probe_batches))
    scoring = score_batch * math.ceil(N_TRAIN / score_batch_size) * run_waves
    aggregate = benchmark_mean_and_masks(args.data_diet_runs, args.ratios)

    stages = [
        StageEstimate("GraNd", "proxy_training_to_score_epoch", None, train_batch, math.ceil(N_TRAIN / 32) * train_epochs * run_waves, train_epoch * train_epochs * run_waves, run_waves, "medium-high", f"Score checkpoint is epoch {train_epochs}; repeated runs fit in {run_waves} wave(s)."),
        *([StageEstimate("GraNd", "proxy_validation", None, eval_batch, math.ceil(N_TEST / 128) * (train_epochs + 1) * run_waves, validation, run_waves, "medium", "Optional literal-script validation overhead.")] if args.include_proxy_validation else []),
        StageEstimate("GraNd", "per_sample_gradient_norm_scoring", None, score_batch, math.ceil(N_TRAIN / score_batch_size) * run_waves, scoring, run_waves, "medium-low", "Uses the repository torch.func/vmap path when available; this is the most hardware-sensitive GraNd stage."),
        StageEstimate("GraNd", "mean_aggregation_and_four_masks", None, aggregate, 1.0, aggregate, 1, "high", "Four ratios share the same averaged score but are sorted separately."),
    ]
    total = training + validation + scoring + aggregate
    clear_torch(device)
    return MethodEstimate("GraNd", total, stages)


def estimate_forgetting(args, device: torch.device) -> MethodEstimate:
    models_mod, _ = load_data_diet_modules(args.repo_root)
    set_cudnn_mode(deterministic=True)
    train_batch, eval_batch = benchmark_data_diet_train(models_mod, device, args, True)
    train_epoch = train_batch * math.ceil(N_TRAIN / 32)
    eval_epoch = eval_batch * math.ceil(N_TEST / 128)
    run_waves = waves(args.data_diet_runs, args.machines)
    epochs = args.forgetting_epochs
    training = train_epoch * epochs * run_waves
    validation = eval_epoch * (epochs + 1) * run_waves if args.include_proxy_validation else 0.0
    aggregate = benchmark_mean_and_masks(args.data_diet_runs, args.ratios)
    stages = [
        StageEstimate("Forgetting", "proxy_training_with_forgetting_tracking", None, train_batch, math.ceil(N_TRAIN / 32) * epochs * run_waves, train_epoch * epochs * run_waves, run_waves, "medium-high", f"One measured batch is extrapolated to {epochs} epochs; forgetting-event updates are inside the measured step."),
        *([StageEstimate("Forgetting", "proxy_validation", None, eval_batch, math.ceil(N_TEST / 128) * (epochs + 1) * run_waves, validation, run_waves, "medium", "Optional literal-script validation overhead.")] if args.include_proxy_validation else []),
        StageEstimate("Forgetting", "mean_aggregation_and_four_masks", None, aggregate, 1.0, aggregate, 1, "high", "Final forgetting scores are averaged and then sorted for all four masks."),
    ]
    total = training + validation + aggregate
    clear_torch(device)
    return MethodEstimate("Forgetting", total, stages)


def estimate_mds(args, device: torch.device) -> MethodEstimate:
    model_mod = load_module_from_path("runtime_mds_model", args.repo_root / "MDS" / "model.py")
    set_cudnn_mode(deterministic=True)
    classifier = model_mod.build_resnet50(N_CLASSES)
    train_batch = benchmark_train_batch(
        classifier,
        device,
        batch_size=128,
        warmup=args.warmup,
        repeats=args.benchmark_batches,
    )
    eval_batch = benchmark_eval_batch(
        classifier,
        device,
        batch_size=128,
        warmup=args.warmup,
        repeats=max(2, args.benchmark_batches // 2),
    )
    train_epoch = train_batch * math.ceil(N_TRAIN / 128)
    eval_epoch = eval_batch * math.ceil(N_TEST / 128)
    seed_waves = waves(args.seed_runs, args.machines)
    training = train_epoch * args.mds_epochs * seed_waves
    validation = eval_epoch * args.mds_epochs * seed_waves if args.include_proxy_validation else 0.0

    feature_model = model_mod.ResNet50FeatureExtractor(classifier)
    feature_batch = benchmark_feature_batch(
        feature_model,
        device,
        batch_size=128,
        image_hw=32,
        warmup=args.warmup,
        repeats=max(2, args.benchmark_batches // 2),
    )
    feature_total = feature_batch * math.ceil(N_TRAIN / 128) * seed_waves

    rng = np.random.default_rng(args.seed)
    probe_classes = args.mds_probe_classes
    probe_n_class = args.mds_probe_class_size
    probe_dim = args.mds_probe_dim
    probe_n = probe_classes * probe_n_class
    features = rng.standard_normal((probe_n, probe_dim), dtype=np.float32)
    labels = np.arange(probe_classes, dtype=np.int64).repeat(probe_n_class)

    def distance_probe() -> None:
        prototypes = np.zeros((probe_classes, probe_dim), dtype=np.float32)
        for c in range(probe_classes):
            prototypes[c] = np.median(features[labels == c], axis=0)
        _ = np.linalg.norm(features - prototypes[labels], axis=1)

    distance_measured = cpu_timed(distance_probe, repeats=2)
    full_n_class = N_TRAIN // N_CLASSES
    log_scale = math.log(max(2, full_n_class)) / math.log(max(2, probe_n_class))
    distance_scale = (
        (N_CLASSES / probe_classes)
        * (full_n_class / probe_n_class)
        * (2048 / probe_dim)
        * log_scale
        * seed_waves
    )
    distance_total = distance_measured * distance_scale
    sort_total = benchmark_sort_masks(args.ratios, keep_high=True) * seed_waves

    stages = [
        StageEstimate("MDS", "resnet50_proxy_training", None, train_batch, math.ceil(N_TRAIN / 128) * args.mds_epochs * seed_waves, train_epoch * args.mds_epochs * seed_waves, seed_waves, "medium-high", f"Three seeds fit in {seed_waves} wave(s)."),
        *([StageEstimate("MDS", "validation_for_best_checkpoint", None, eval_batch, math.ceil(N_TEST / 128) * args.mds_epochs * seed_waves, validation, seed_waves, "medium", "Optional literal-script overhead: MDS/train_base.py evaluates every epoch to choose best.pth.")] if args.include_proxy_validation else []),
        StageEstimate("MDS", "resnet50_feature_extraction", None, feature_batch, math.ceil(N_TRAIN / 128) * seed_waves, feature_total, seed_waves, "medium-high", "One feature pass per seed; checkpoint loading is excluded."),
        StageEstimate("MDS", "class_medians_and_distances", None, distance_measured, distance_scale, distance_total, seed_waves, "medium-low", "Scaled from a balanced synthetic probe; median cost includes an n log n correction."),
        StageEstimate("MDS", "four_middle_band_sorts_and_masks", None, sort_total / seed_waves, seed_waves, sort_total, seed_waves, "high", "The current implementation calls argsort independently for each ratio."),
    ]
    total = training + validation + feature_total + distance_total + sort_total
    clear_torch(device)
    return MethodEstimate("MDS", total, stages)


def estimate_moso(args, device: torch.device) -> MethodEstimate:
    resnet_mod = load_module_from_path(
        "runtime_moso_resnet", args.repo_root / "MoSo" / "models" / "resnet.py"
    )
    set_cudnn_mode(deterministic=False)
    model = resnet_mod.ResNet50(-10, N_CLASSES)
    train_batch = benchmark_train_batch(
        model,
        device,
        batch_size=256,
        warmup=args.warmup,
        repeats=max(2, args.benchmark_batches // 2),
    )
    eval_batch = benchmark_eval_batch(
        model,
        device,
        batch_size=100,
        warmup=args.warmup,
        repeats=max(2, args.benchmark_batches // 2),
    )
    support_n = math.ceil(N_TRAIN / args.moso_trials)
    train_epoch_one_trial = train_batch * math.ceil(support_n / 256)
    eval_epoch = eval_batch * math.ceil(N_TEST / 100)
    one_trial_training = (train_epoch_one_trial + eval_epoch) * args.moso_epochs

    model = model.to(device).eval()
    x_cpu, y_cpu = make_cpu_batch(1, pin_memory=device.type == "cuda")
    params = [p for p in model.parameters() if p.requires_grad]

    def one_gradient() -> None:
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        vector = torch.nn.utils.parameters_to_vector(grads)
        _ = (vector.square().sum() + vector.sum() * 0.0).detach().cpu()

    grad_sec = timed_repeated(
        one_gradient,
        device,
        warmup=1,
        repeats=max(1, args.moso_gradient_probes),
    )

    cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def copy_and_load() -> None:
        cloned = copy.deepcopy(model)
        cloned.load_state_dict(cpu_state)
        del cloned

    copy_sec = timed_repeated(copy_and_load, device, warmup=0, repeats=2)
    one_checkpoint_score = 2.0 * support_n * grad_sec
    one_trial_scoring = args.moso_sampled_checkpoints * (one_checkpoint_score + copy_sec)
    trial_waves = waves(args.moso_trials, args.machines)

    export_one = benchmark_sort_masks((args.ratios[0],), keep_high=True)
    stages: list[StageEstimate] = []
    total = 0.0
    for ratio in args.ratios:
        score_ratio = one_trial_scoring * trial_waves
        train_only_ratio = train_epoch_one_trial * args.moso_epochs * trial_waves
        eval_only_ratio = eval_epoch * args.moso_epochs * trial_waves
        validation_ratio = eval_only_ratio if args.include_proxy_validation else 0.0
        ratio_total = train_only_ratio + validation_ratio + score_ratio + export_one
        total += ratio_total
        stages.extend([
            StageEstimate("MoSo", "eight_trial_surrogate_training", ratio, train_batch, math.ceil(support_n / 256) * args.moso_epochs * trial_waves, train_only_ratio, trial_waves, "medium", f"Eight trials are scheduled as {trial_waves} waves on {args.machines} machines; each trial trains on about N/8 samples for {args.moso_epochs} epochs."),
            *([StageEstimate("MoSo", "trial_validation_and_checkpoint_selection", ratio, eval_batch, math.ceil(N_TEST / 100) * args.moso_epochs * trial_waves, validation_ratio, trial_waves, "medium", "Optional literal-script overhead: surrogate_training.py evaluates every trial at every epoch because ckptfreq=1.")] if args.include_proxy_validation else []),
            StageEstimate("MoSo", "ten_checkpoint_exact_gradient_scoring", ratio, grad_sec, 2 * support_n * args.moso_sampled_checkpoints * trial_waves, score_ratio, trial_waves, "low-medium", "Each sampled checkpoint makes two per-sample gradient passes over one trial support set. This stage is highly sensitive to PyTorch/CUDA versions."),
            StageEstimate("MoSo", "mask_export", ratio, export_one, 1.0, export_one, 1, "high", "Sorting and mask construction only; disk writing is excluded."),
        ])

    clear_torch(device)
    return MethodEstimate(
        "MoSo",
        total,
        stages,
        warning="Conservative convention: the full 8-trial training/scoring pipeline is repeated independently for every target keep ratio.",
    )


def build_yangclip_image_encoder(args, device: torch.device):
    yang_dir = args.repo_root / "YangCLIP"
    ckpt = yang_dir / "clip_model" / "ViT-B-32.pt"
    if ckpt.is_file():
        try:
            with temporary_sys_path(yang_dir):
                import clip  # type: ignore
                model, _ = clip.load(str(ckpt), device=device)
            model = model.float().eval()
            dim = int(model.text_projection.shape[1])

            class Encoder(nn.Module):
                def __init__(self, clip_model):
                    super().__init__()
                    self.clip_model = clip_model

                def forward(self, x):
                    return self.clip_model.encode_image(x).float()

            return Encoder(model), dim, "actual local CLIP ViT-B/32"
        except Exception as exc:
            fallback_reason = f"CLIP load failed: {exc}"
    else:
        fallback_reason = f"missing {ckpt}"

    from torchvision.models import vit_b_32

    vit = vit_b_32(weights=None)
    dim = vit.heads.head.in_features
    vit.heads = nn.Identity()
    return vit.to(device), dim, f"torchvision ViT-B/32 compute proxy ({fallback_reason})"


def benchmark_yangclip_adapter_batch(
    encoder: nn.Module,
    dim: int,
    device: torch.device,
    args,
) -> tuple[float, int]:
    target_batch = 256
    batch = target_batch
    last_error = ""
    while batch >= 8:
        try:
            encoder = encoder.to(device).eval()
            for p in encoder.parameters():
                p.requires_grad = False
            adapter_img = nn.Linear(dim, dim).to(device)
            adapter_txt = nn.Linear(dim, dim).to(device)
            optimizer = torch.optim.Adam(
                list(adapter_img.parameters()) + list(adapter_txt.parameters()), lr=1e-4
            )
            x_cpu, y_cpu = make_cpu_batch(batch, image_hw=224, pin_memory=device.type == "cuda")
            text_features = torch.randn(N_CLASSES, dim, device=device)

            def step() -> None:
                x = x_cpu.to(device, non_blocking=True)
                y = y_cpu.to(device, non_blocking=True)
                with torch.no_grad():
                    image_feat = encoder(x).float()
                img_out = F.normalize(adapter_img(image_feat), dim=-1)
                txt_out = F.normalize(adapter_txt(text_features[y]), dim=-1)
                logits = img_out @ txt_out.t()
                labels = torch.arange(batch, device=device)
                loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            sec = timed_repeated(step, device, warmup=1, repeats=max(2, args.benchmark_batches // 2))
            return sec, batch
        except torch.cuda.OutOfMemoryError as exc:
            last_error = str(exc)
            clear_torch(device)
            batch //= 2
    raise RuntimeEstimatorError(f"YangCLIP adapter benchmark OOM even at batch 8: {last_error}")


def benchmark_yangclip_score_batch(
    encoder: nn.Module,
    dim: int,
    device: torch.device,
    batch: int,
    args,
) -> float:
    encoder = encoder.to(device).eval()
    adapter_img = nn.Linear(dim, dim).to(device).eval()
    text_features = F.normalize(torch.randn(N_CLASSES, dim, device=device), dim=-1)
    x_cpu, y_cpu = make_cpu_batch(batch, image_hw=224, pin_memory=device.type == "cuda")

    def step() -> None:
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        with torch.no_grad():
            image_feat = encoder(x).float()
            img_out = F.normalize(adapter_img(image_feat), dim=-1)
            match = F.cosine_similarity(img_out, text_features[y], dim=-1)
            _ = torch.stack((img_out.mean(), match.mean())).detach().cpu()

    return timed_repeated(step, device, warmup=1, repeats=max(2, args.benchmark_batches // 2))


def benchmark_knn_scaled(args) -> tuple[float, float]:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeEstimatorError("scikit-learn is required for YangCLIP KNN timing") from exc

    c_probe = args.yangclip_knn_probe_classes
    n_probe = args.yangclip_knn_probe_class_size
    d = 512
    rng = np.random.default_rng(args.seed)
    feature_sets = [rng.standard_normal((n_probe, d), dtype=np.float32) for _ in range(c_probe)]

    def run() -> None:
        for feats in feature_sets:
            neighbors = min(50, len(feats))
            nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm="auto").fit(feats)
            distances, _ = nbrs.kneighbors(feats)
            _ = distances[:, 1:].mean(axis=1)

    measured = cpu_timed(run, repeats=1)
    full_n = N_TRAIN // N_CLASSES
    scale = (N_CLASSES / c_probe) * (full_n / n_probe) ** 2
    return measured, scale


def benchmark_yangclip_optimization(
    ratio: int,
    device: torch.device,
    args,
) -> tuple[float, int, bool]:
    torch.manual_seed(args.seed + ratio)
    similarity = torch.rand(N_TRAIN, device=device)
    diversity = torch.rand(N_TRAIN, device=device)
    similarity = similarity / similarity.mean().clamp(min=1e-12)
    diversity = diversity / diversity.mean().clamp(min=1e-12)
    w = nn.Parameter(0.01 * torch.ones(N_TRAIN, device=device))
    optimizer = torch.optim.SGD([w], lr=1e-3, momentum=0.9)
    k = int(round(N_TRAIN * ratio / 100.0))
    max_probe = min(args.yangclip_opt_probe_steps, args.yangclip_opt_max_epochs)
    converged = False

    synchronize(device)
    start = time.perf_counter()
    steps = 0
    for epoch in range(max_probe):
        x = torch.sigmoid(100.0 * w)
        loss1 = -torch.mean(x * similarity)
        loss2 = -torch.mean(x * diversity) * 0.1
        hard_x = (x > 0.5).float()
        st_x = hard_x - x.detach() + x
        loss3 = torch.sqrt((((st_x.sum() - k) / N_TRAIN) ** 2)) * 2.0
        loss = loss1 + loss2 + loss3
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps = epoch + 1
        if float(loss3.detach().item()) < 1e-3:
            converged = True
            break
    synchronize(device)
    elapsed = time.perf_counter() - start
    if converged:
        return elapsed, steps, True
    per_step = elapsed / max(1, steps)
    return per_step * args.yangclip_opt_max_epochs, steps, False


def estimate_yangclip(args, device: torch.device) -> MethodEstimate:
    set_cudnn_mode(deterministic=True)
    encoder, dim, encoder_note = build_yangclip_image_encoder(args, device)
    adapter_batch_sec, measured_batch = benchmark_yangclip_adapter_batch(encoder, dim, device, args)
    # Scale by sample throughput if the target batch did not fit.
    per_sample = adapter_batch_sec / measured_batch
    adapter_epoch = per_sample * N_TRAIN
    seed_waves = waves(args.seed_runs, args.machines)
    adapter_total = adapter_epoch * args.yangclip_adapter_epochs * seed_waves

    score_batch_sec = benchmark_yangclip_score_batch(encoder, dim, device, measured_batch, args)
    scoring_total = (score_batch_sec / measured_batch) * N_TRAIN * seed_waves
    knn_measured, knn_scale = benchmark_knn_scaled(args)
    knn_total = knn_measured * knn_scale * seed_waves

    stages = [
        StageEstimate("YangCLIP", "clip_adapter_training", None, adapter_batch_sec, (N_TRAIN / measured_batch) * args.yangclip_adapter_epochs * seed_waves, adapter_total, seed_waves, "medium", f"{encoder_note}; measured batch={measured_batch}, target script batch=256. Model loading and image preprocessing are excluded."),
        StageEstimate("YangCLIP", "clip_feature_and_semantic_scoring", None, score_batch_sec, (N_TRAIN / measured_batch) * seed_waves, scoring_total, seed_waves, "medium", "One score pass per seed; three seeds fit in one four-machine wave."),
        StageEstimate("YangCLIP", "classwise_knn_diversity", None, knn_measured, knn_scale * seed_waves, knn_total, seed_waves, "low-medium", "Scaled quadratically in the 500 samples per CIFAR-100 class."),
    ]
    optimization_total = 0.0
    for ratio in args.ratios:
        opt_sec, probe_steps, converged = benchmark_yangclip_optimization(ratio, device, args)
        optimization_total += opt_sec * seed_waves
        note = (
            f"Converged in {probe_steps} steps on synthetic 50k scores."
            if converged
            else f"Did not converge within {probe_steps} probe steps; projected to the script maximum of {args.yangclip_opt_max_epochs} steps."
        )
        stages.append(
            StageEstimate("YangCLIP", "optimization_based_mask", ratio, opt_sec, seed_waves, opt_sec * seed_waves, seed_waves, "medium" if converged else "low-medium", note)
        )

    total = adapter_total + scoring_total + knn_total + optimization_total
    clear_torch(device)
    return MethodEstimate("YangCLIP", total, stages)


def estimate_rlselector(args, device: torch.device) -> MethodEstimate:
    if device.type != "cuda":
        raise RuntimeEstimatorError("RLSelector's A2C implementation hardcodes .cuda(); use a CUDA device.")
    resnet_mod = load_module_from_path(
        "runtime_rl_resnet", args.repo_root / "RLSelector" / "Network" / "ResNet.py"
    )
    a2c_mod = load_module_from_path(
        "runtime_rl_a2c", args.repo_root / "RLSelector" / "A2C.py"
    )
    set_cudnn_mode(deterministic=True)
    torch.cuda.set_device(device)
    seed_waves = waves(args.seed_runs, args.machines)
    stages: list[StageEstimate] = []
    total = 0.0

    # Distance matrix is ratio-independent but recomputed every epoch in each ratio run.
    probe_classes = args.rl_distance_probe_classes
    class_size = N_TRAIN // N_CLASSES
    probe_feature_map = torch.randn(probe_classes * class_size, 512, device=device)
    class_indices = [torch.arange(c * class_size, (c + 1) * class_size, device=device) for c in range(probe_classes)]

    def distance_probe() -> None:
        outputs = []
        for idx in class_indices:
            feature = probe_feature_map[idx]
            feature_sq = torch.sum(feature ** 2, dim=1, keepdim=True)
            dist_sq = feature_sq + feature_sq.t() - 2 * feature @ feature.t()
            dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))
            outputs.append(dist.mean(dim=0).cpu())
        _ = sum(float(x.mean()) for x in outputs)

    distance_measured = timed_repeated(distance_probe, device, warmup=1, repeats=1)
    distance_one_epoch = distance_measured * (N_CLASSES / probe_classes)

    for ratio in args.ratios:
        model = resnet_mod.ResNet18(num_classes=N_CLASSES).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        agent_args = SimpleNamespace(state_dim=512, action_dim=1, epoches=args.rl_epochs)
        agent = a2c_mod.A2C(agent_args).to(device)
        x_cpu, y_cpu = make_cpu_batch(256, pin_memory=True)
        feature_map = torch.zeros(N_TRAIN, 512, device=device)
        mask_pruner = torch.ones(N_TRAIN)
        dis_loss = torch.rand(N_TRAIN)
        batch_indices = torch.arange(256)

        def pretrain_step() -> None:
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            logits, feature = model(x)
            feature_map[batch_indices] = feature.detach()
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        pretrain_batch = timed_repeated(pretrain_step, device, args.warmup, max(2, args.benchmark_batches // 2))

        def rl_step() -> None:
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            logits, feature = model(x)
            state = feature.detach().squeeze()
            action = agent.action(state)
            mask = action.detach().squeeze()
            mask_pruner[batch_indices] = mask.cpu()
            feature_map[batch_indices] = state
            per_sample = F.cross_entropy(logits, y, reduction="none")
            selected_loss = (per_sample * mask).mean()
            optimizer.zero_grad(set_to_none=True)
            selected_loss.backward()
            optimizer.step()
            current_cr = torch.where(mask_pruner > 0.5)[0].shape[0] / mask_pruner.shape[0]
            gap = current_cr - ratio / 100.0
            max_gamma = 1 - ratio / 100.0 if gap >= 0 else ratio / 100.0
            gamma = abs(gap) / max(max_gamma, 1e-12)
            reward_1 = 5.0 * (1 - gamma)
            reward_2 = (dis_loss * mask_pruner).mean().item()
            agent.update(state, action, reward_1 + reward_2)

        rl_batch = timed_repeated(rl_step, device, warmup=1, repeats=max(2, args.benchmark_batches // 2))
        eval_batch = benchmark_eval_batch(
            model,
            device,
            batch_size=256,
            warmup=1,
            repeats=max(2, args.benchmark_batches // 2),
            output_logits_index=0,
        )
        pretrain_epoch = pretrain_batch * math.ceil(N_TRAIN / 256)
        rl_train_epoch = rl_batch * math.ceil(N_TRAIN / 256)
        eval_epoch = eval_batch * math.ceil(N_TEST / 256)
        mask_time = cpu_timed(lambda: torch.topk(mask_pruner, k=int(round(N_TRAIN * ratio / 100.0))), repeats=3)
        ratio_total = (
            pretrain_epoch
            + args.rl_epochs * (distance_one_epoch + rl_train_epoch + eval_epoch)
            + mask_time
        ) * seed_waves
        total += ratio_total
        stages.extend([
            StageEstimate("RLSelector", "one_epoch_feature_warmup", ratio, pretrain_batch, math.ceil(N_TRAIN / 256) * seed_waves, pretrain_epoch * seed_waves, seed_waves, "medium-high", "The script performs this additional full pass only in epoch 0 of each ratio run."),
            StageEstimate("RLSelector", "classwise_distance_matrix_each_epoch", ratio, distance_measured, (N_CLASSES / probe_classes) * args.rl_epochs * seed_waves, distance_one_epoch * args.rl_epochs * seed_waves, seed_waves, "medium", f"Measured on {probe_classes} full 500-sample classes and scaled to 100 classes."),
            StageEstimate("RLSelector", "rl_guided_training", ratio, rl_batch, math.ceil(N_TRAIN / 256) * args.rl_epochs * seed_waves, rl_train_epoch * args.rl_epochs * seed_waves, seed_waves, "medium-high", "Includes ResNet18 backward, mask reward, actor update, and critic update."),
            StageEstimate("RLSelector", "validation_each_epoch", ratio, eval_batch, math.ceil(N_TEST / 256) * args.rl_epochs * seed_waves, eval_epoch * args.rl_epochs * seed_waves, seed_waves, "medium", "Matches RLSelector/train.py."),
            StageEstimate("RLSelector", "topk_mask_export", ratio, mask_time, seed_waves, mask_time * seed_waves, seed_waves, "high", "One target ratio per independent run."),
        ])
        del model, optimizer, agent, feature_map, mask_pruner, dis_loss
        clear_torch(device)

    return MethodEstimate("RLSelector", total, stages)


ESTIMATORS: dict[str, Callable[[Any, torch.device], MethodEstimate]] = {
    "Random": estimate_random,
    "Herding": estimate_herding,
    "EL2N": estimate_el2n,
    "GraNd": estimate_grand,
    "Forgetting": estimate_forgetting,
    "MDS": estimate_mds,
    "MoSo": estimate_moso,
    "YangCLIP": estimate_yangclip,
    "RLSelector": estimate_rlselector,
}


def print_assumptions(args, device: torch.device, repo_status: dict[str, bool]) -> None:
    print("=" * 88)
    print("CIFAR-100 data-selection runtime estimator")
    print("=" * 88)
    print(f"Repository root : {args.repo_root}")
    print(f"Device          : {device}")
    if device.type == "cuda":
        print(f"GPU             : {torch.cuda.get_device_name(device)}")
    print(f"Machines        : {args.machines} (one GPU per machine)")
    print(f"Keep ratios     : {', '.join(map(str, args.ratios))}% — processed sequentially")
    print(f"Methods         : {', '.join(args.methods)}")
    validation_mode = "included" if args.include_proxy_validation else "excluded except RLSelector"
    print("Excluded        : dataset download/construction, initial file loading, checkpoint I/O, final disk writes")
    print(f"Proxy validation: {validation_mode}")
    print("Parallel rule   : only repetitions/trials inside one keep-ratio job are parallelized")
    print("Repository files:")
    for name, ok in repo_status.items():
        print(f"  {name:<12} {'OK' if ok else 'MISSING'}")
    print()


def print_results(results: Sequence[MethodEstimate]) -> None:
    print("\n" + "=" * 88)
    print("Estimated wall-clock totals")
    print("=" * 88)
    print(f"{'Method':<14} {'Status':<10} {'Estimated time':>20}  Notes")
    print("-" * 88)
    for result in results:
        note = result.warning or ""
        print(f"{result.method:<14} {result.status:<10} {format_duration(result.estimated_seconds):>20}  {note}")

    print("\nPer-stage detail")
    print("-" * 110)
    for result in results:
        print(f"\n[{result.method}] total={format_duration(result.estimated_seconds)} status={result.status}")
        if result.warning:
            print(f"  warning: {result.warning}")
        for stage in result.stages:
            ratio = "shared" if stage.ratio is None else f"keep={stage.ratio}%"
            print(
                f"  {ratio:<10} {stage.stage:<42} {format_duration(stage.estimated_seconds):>14} "
                f"confidence={stage.confidence}"
            )
            print(f"             {stage.note}")


def write_outputs(args, device: torch.device, results: Sequence[MethodEstimate]) -> None:
    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "repo_root": str(args.repo_root),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "machines": args.machines,
        "ratios": list(args.ratios),
        "methods": list(args.methods),
        "exclusions": [
            "dataset download and construction",
            "initial checkpoint/model file loading",
            "checkpoint serialization",
            "final mask disk writes",
            *( [] if args.include_proxy_validation else ["proxy validation unrelated to score computation (except RLSelector)"] ),
        ],
    }
    payload = {
        "metadata": metadata,
        "results": [
            {
                "method": r.method,
                "estimated_seconds": r.estimated_seconds,
                "estimated_human": format_duration(r.estimated_seconds),
                "status": r.status,
                "warning": r.warning,
                "stages": [asdict(s) | {"estimated_human": format_duration(s.estimated_seconds)} for s in r.stages],
            }
            for r in results
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "stage",
                "ratio",
                "measured_unit_seconds",
                "multiplier",
                "parallel_waves",
                "estimated_seconds",
                "estimated_human",
                "confidence",
                "note",
                "method_status",
                "method_warning",
            ],
        )
        writer.writeheader()
        for result in results:
            if not result.stages:
                writer.writerow({
                    "method": result.method,
                    "stage": "",
                    "ratio": "",
                    "estimated_seconds": result.estimated_seconds,
                    "estimated_human": format_duration(result.estimated_seconds),
                    "method_status": result.status,
                    "method_warning": result.warning,
                })
                continue
            for stage in result.stages:
                writer.writerow({
                    "method": result.method,
                    "stage": stage.stage,
                    "ratio": "" if stage.ratio is None else stage.ratio,
                    "measured_unit_seconds": f"{stage.measured_unit_seconds:.9f}",
                    "multiplier": f"{stage.multiplier:.6f}",
                    "parallel_waves": stage.parallel_waves,
                    "estimated_seconds": f"{stage.estimated_seconds:.6f}",
                    "estimated_human": format_duration(stage.estimated_seconds),
                    "confidence": stage.confidence,
                    "note": stage.note,
                    "method_status": result.status,
                    "method_warning": result.warning,
                })

    print(f"\nJSON written to: {args.output_json}")
    print(f"CSV written to : {args.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Benchmark short kernels and estimate CIFAR-100 selection runtime.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--machines", type=int, default=4)
    parser.add_argument("--ratios", nargs="+", type=int, default=list(DEFAULT_RATIOS))
    parser.add_argument("--methods", nargs="+", choices=list(ALL_METHODS) + ["all"], default=["all"])
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--seed-runs", type=int, default=3, help="Normal seed count for non-MoSo methods.")
    parser.add_argument("--data-diet-runs", type=int, default=3, help="Repeated runs for EL2N/GraNd/Forgetting.")
    parser.add_argument("--benchmark-batches", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--strict", action="store_true", help="Stop immediately when one method cannot be benchmarked.")
    parser.add_argument("--dry-run", action="store_true", help="Validate assumptions and files without running benchmarks.")
    parser.add_argument(
        "--include-proxy-validation",
        action="store_true",
        help="Include validation passes that exist in literal run scripts but are not needed to compute the selection scores/masks. RLSelector validation remains included because it selects saved masks.",
    )

    # Data Diet.
    parser.add_argument("--el2n-score-epoch", type=int, default=20)
    parser.add_argument("--grand-score-epoch", type=int, default=20)
    parser.add_argument("--forgetting-epochs", type=int, default=200)
    parser.add_argument("--grand-probe-batches", type=int, default=1)

    # Herding / MDS CPU probes.
    parser.add_argument("--herding-probe-classes", type=int, default=8)
    parser.add_argument("--herding-probe-class-size", type=int, default=100)
    parser.add_argument("--mds-epochs", type=int, default=200)
    parser.add_argument("--mds-probe-classes", type=int, default=10)
    parser.add_argument("--mds-probe-class-size", type=int, default=100)
    parser.add_argument("--mds-probe-dim", type=int, default=256)

    # MoSo.
    parser.add_argument("--moso-trials", type=int, default=8)
    parser.add_argument("--moso-epochs", type=int, default=50)
    parser.add_argument("--moso-sampled-checkpoints", type=int, default=10)
    parser.add_argument("--moso-gradient-probes", type=int, default=1)

    # YangCLIP.
    parser.add_argument("--yangclip-adapter-epochs", type=int, default=30)
    parser.add_argument("--yangclip-knn-probe-classes", type=int, default=8)
    parser.add_argument("--yangclip-knn-probe-class-size", type=int, default=100)
    parser.add_argument("--yangclip-opt-probe-steps", type=int, default=5_000)
    parser.add_argument("--yangclip-opt-max-epochs", type=int, default=100_000)

    # RLSelector.
    parser.add_argument("--rl-epochs", type=int, default=200)
    parser.add_argument("--rl-distance-probe-classes", type=int, default=5)

    parser.add_argument("--output-json", type=Path, default=Path("runtime_estimate_cifar100.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("runtime_estimate_cifar100.csv"))
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    if "all" in args.methods:
        args.methods = list(ALL_METHODS)
    else:
        # Preserve user order while removing duplicates.
        args.methods = list(dict.fromkeys(args.methods))
    args.ratios = tuple(dict.fromkeys(args.ratios))
    if any(r <= 0 or r > 100 for r in args.ratios):
        parser.error("All keep ratios must be in (0, 100].")
    if args.machines < 1:
        parser.error("--machines must be >= 1")
    if args.benchmark_batches < 1:
        parser.error("--benchmark-batches must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise SystemExit(f"Unsupported device: {device}")

    repo_status = check_repo(args.repo_root)
    print_assumptions(args, device, repo_status)
    if args.dry_run:
        print("Dry run complete. No benchmark was executed.")
        return 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results: list[MethodEstimate] = []

    for method in args.methods:
        print("\n" + "=" * 88)
        print(f"Benchmarking {method}")
        print("=" * 88)
        estimator = ESTIMATORS[method]
        try:
            result = estimator(args, device)
            results.append(result)
            print(f"{method}: {format_duration(result.estimated_seconds)}")
        except Exception as exc:
            clear_torch(device)
            message = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] {method}: {message}", file=sys.stderr)
            if args.strict:
                traceback.print_exc()
                return 1
            results.append(MethodEstimate(method, float("nan"), [], status="failed", warning=message))

    print_results(results)
    write_outputs(args, device, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())