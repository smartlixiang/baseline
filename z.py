#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unify mask paths for label-noise experiments.

Run from baseline root:

    python z.py

Purpose:
    Copy existing label-noise experiment masks from each baseline method's
    local noise_masks directory to the unified layout consumed by noise_train.py:

        [mode]/[dataset]/[seed]/mask_[kr].npz

    All converted mode names are prefixed with "noise_", for example:

        noise_GraNd/cifar10/22/mask_30.npz
        noise_MDS/cifar100/96/mask_70.npz

This script only handles local label-noise experiment results under baseline/.
It does not copy summary.csv and does not delete original results by default.
"""

from __future__ import annotations

import argparse
import filecmp
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


MASK_RE = re.compile(r"^mask_(\d+)\.npz$")


@dataclass(frozen=True)
class SourceSpec:
    method_dir: str
    method_name_map: Dict[str, str]


SOURCE_SPECS: List[SourceSpec] = [
    SourceSpec(
        method_dir="data_diet",
        method_name_map={
            "E2LN": "noise_E2LN",
            "EL2N": "noise_E2LN",
            "GraNd": "noise_GraNd",
            "grand": "noise_GraNd",
            "Forgetting": "noise_Forgetting",
            "forgetting": "noise_Forgetting",
        },
    ),
    SourceSpec(
        method_dir="herding",
        method_name_map={
            "herding": "noise_herding",
            "Herding": "noise_herding",
        },
    ),
    SourceSpec(
        method_dir="MDS",
        method_name_map={
            "MDS": "noise_MDS",
            "mds": "noise_MDS",
            "ModerateDS": "noise_MDS",
            "Moderate-DS": "noise_MDS",
            "Moderate_DS": "noise_MDS",
        },
    ),
    SourceSpec(
        method_dir="MoSo",
        method_name_map={
            "MoSo": "noise_MoSo",
            "moso": "noise_MoSo",
        },
    ),
    SourceSpec(
        method_dir="YangCLIP",
        method_name_map={
            "YangCLIP": "noise_YangCLIP",
            "yangclip": "noise_YangCLIP",
            "Yangclip": "noise_YangCLIP",
        },
    ),
]


@dataclass
class CopyTask:
    source: Path
    target: Path
    method_dir: str
    source_method: str
    target_mode: str
    dataset: str
    seed: str
    keep_ratio: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy label-noise experiment masks to unified paths: "
            "[noise_mode]/[dataset]/[seed]/mask_[kr].npz"
        )
    )
    parser.add_argument(
        "--baseline-root",
        type=str,
        default=".",
        help="Path to baseline root. Default: current directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target mask files if they already exist.",
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help=(
            "Delete original mask files after successful copy. "
            "Default is false, so original results are kept."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned operations without copying files.",
    )
    return parser.parse_args()


def is_dataset_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name.startswith("."):
        return False
    if path.name.lower() in {"summary", "summaries", "__pycache__"}:
        return False
    return True


def discover_tasks(baseline_root: Path) -> List[CopyTask]:
    tasks: List[CopyTask] = []

    for spec in SOURCE_SPECS:
        source_root = baseline_root / spec.method_dir / "noise_masks"
        if not source_root.exists():
            print(f"[MISS] {source_root} does not exist, skip {spec.method_dir}.")
            continue

        for dataset_dir in sorted(source_root.iterdir()):
            if not is_dataset_dir(dataset_dir):
                continue
            dataset = dataset_dir.name

            for seed_dir in sorted(dataset_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                seed = seed_dir.name

                for method_dir in sorted(seed_dir.iterdir()):
                    if not method_dir.is_dir():
                        continue

                    source_method = method_dir.name
                    target_mode = spec.method_name_map.get(source_method)
                    if target_mode is None:
                        print(
                            f"[SKIP] Unknown method folder under label-noise masks: "
                            f"{method_dir}"
                        )
                        continue

                    for mask_path in sorted(method_dir.glob("mask_*.npz")):
                        m = MASK_RE.match(mask_path.name)
                        if not m:
                            print(f"[SKIP] Non-standard mask name: {mask_path}")
                            continue

                        keep_ratio = int(m.group(1))
                        target_path = (
                            baseline_root
                            / target_mode
                            / dataset
                            / seed
                            / f"mask_{keep_ratio}.npz"
                        )

                        tasks.append(
                            CopyTask(
                                source=mask_path,
                                target=target_path,
                                method_dir=spec.method_dir,
                                source_method=source_method,
                                target_mode=target_mode,
                                dataset=dataset,
                                seed=seed,
                                keep_ratio=keep_ratio,
                            )
                        )

    return tasks


def copy_task(task: CopyTask, overwrite: bool, delete_original: bool, dry_run: bool) -> str:
    if not task.source.exists():
        return f"[MISS] source not found: {task.source}"

    if task.target.exists():
        if filecmp.cmp(task.source, task.target, shallow=False):
            if delete_original and not dry_run:
                task.source.unlink()
                return f"[SAME+DEL] {task.target}"
            return f"[SAME] {task.target}"

        if not overwrite:
            return (
                f"[EXIST] target exists and differs, skip without --overwrite: "
                f"{task.target}"
            )

    if dry_run:
        return f"[DRY] {task.source} -> {task.target}"

    task.target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(task.source, task.target)

    if delete_original:
        if filecmp.cmp(task.source, task.target, shallow=False):
            task.source.unlink()
            return f"[COPY+DEL] {task.source} -> {task.target}"
        return f"[COPY] {task.source} -> {task.target} [WARN: not deleted; copy check failed]"

    return f"[COPY] {task.source} -> {task.target}"


def print_plan_summary(tasks: List[CopyTask]) -> None:
    if not tasks:
        print("[INFO] No mask files discovered.")
        return

    counts: Dict[str, int] = {}
    for task in tasks:
        key = f"{task.target_mode}/{task.dataset}/{task.seed}"
        counts[key] = counts.get(key, 0) + 1

    print("[PLAN] discovered mask files:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]} file(s)")


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline_root).resolve()

    if not baseline_root.exists():
        raise FileNotFoundError(f"baseline root not found: {baseline_root}")

    print(f"[ROOT] {baseline_root}")
    print(f"[MODE] overwrite={args.overwrite}, delete_original={args.delete_original}, dry_run={args.dry_run}")
    print()

    tasks = discover_tasks(baseline_root)
    print_plan_summary(tasks)
    print()

    copied = 0
    skipped_or_same = 0

    for task in tasks:
        msg = copy_task(
            task=task,
            overwrite=args.overwrite,
            delete_original=args.delete_original,
            dry_run=args.dry_run,
        )
        print(msg)

        if msg.startswith("[COPY") or msg.startswith("[DRY"):
            copied += 1
        else:
            skipped_or_same += 1

    print()
    print("[DONE]")
    print(f"  discovered        : {len(tasks)}")
    print(f"  copied/planned    : {copied}")
    print(f"  skipped/same/miss : {skipped_or_same}")
    print()
    print("Unified target examples:")
    print("  noise_GraNd/cifar10/22/mask_30.npz")
    print("  noise_MDS/cifar100/96/mask_70.npz")
    print("  noise_YangCLIP/cifar10/42/mask_50.npz")


if __name__ == "__main__":
    main()
