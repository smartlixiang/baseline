#!/usr/bin/env python3
"""
Convert existing data_diet selection results to train_after_selection.py-compatible .npz masks.

This script performs format conversion only. It does NOT change sample-set semantics:
- it reuses original `*.mask.npy` files whenever possible,
- falls back to `*.idx.npy` only when mask files are missing,
- preserves global training-sample indices exactly as produced on the source machine.

Goal: ensure the same selected samples on the data_diet server can be read as identical
selections on other machines via `np.load(path)["mask"]`.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DATASET_TRAIN_SIZE = {
    "cifar10": 50000,
    "cifar100": 50000,
}

# Input method aliases -> canonical method key.
METHOD_ALIASES = {
    "el2n": "el2n",
    "grand": "grand",
    "forgetting": "forgetting",
}

# Canonical method key -> required output dir name.
METHOD_OUTPUT_DIR = {
    "el2n": "E2LN",  # Intentionally E2LN per requirement.
    "grand": "GraNd",
    "forgetting": "Forgetting",
}

KEEP_RE = re.compile(r"keep(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class SourceChoice:
    dataset: str
    seed: str
    method_key: str
    keep_ratio: int
    source_file: Path
    source_kind: str  # "mask" or "idx"


def normalize_method(user_method: str) -> Optional[str]:
    key = user_method.strip().lower()
    return METHOD_ALIASES.get(key)


def extract_keep_ratio(path: Path) -> Optional[int]:
    m = KEEP_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1))


def load_mask_from_mask_file(path: Path) -> np.ndarray:
    arr = np.load(path)
    return np.asarray(arr).reshape(-1)


def load_mask_from_idx_file(path: Path, dataset: str) -> np.ndarray:
    n = DATASET_TRAIN_SIZE[dataset]
    idx = np.asarray(np.load(path)).reshape(-1)
    if idx.size == 0:
        raise ValueError(f"idx file is empty: {path}")
    if not np.issubdtype(idx.dtype, np.integer):
        if np.all(np.equal(idx, idx.astype(np.int64))):
            idx = idx.astype(np.int64)
        else:
            raise ValueError(f"idx file contains non-integer values: {path}")

    idx = idx.astype(np.int64, copy=False)
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError(f"idx out of range [0, {n - 1}] in {path}")

    mask = np.zeros(n, dtype=np.uint8)
    mask[idx] = 1
    return mask


def validate_mask(mask: np.ndarray, dataset: str, src: Path) -> np.ndarray:
    expected = DATASET_TRAIN_SIZE[dataset]
    mask = np.asarray(mask).reshape(-1)
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1D, got shape={mask.shape}, src={src}")
    if mask.shape[0] != expected:
        raise ValueError(
            f"mask length mismatch for {dataset}: expected {expected}, got {mask.shape[0]}, src={src}"
        )

    # Normalize to {0,1} and validate semantics.
    if mask.dtype == np.bool_:
        out = mask.astype(np.uint8)
    else:
        uniq = np.unique(mask)
        if not set(int(x) for x in uniq.tolist()).issubset({0, 1}):
            raise ValueError(f"mask has non-binary values {uniq[:10]} in {src}")
        out = mask.astype(np.uint8)

    selected = int(out.sum())
    if selected <= 0:
        raise ValueError(f"mask has no selected samples in {src}")
    return out


def iter_dataset_seed_dirs(mask_root: Path, datasets: Sequence[str], seeds: Optional[Sequence[str]]) -> Iterable[Tuple[str, str, Path]]:
    wanted_seeds = set(seeds) if seeds else None
    for dataset in datasets:
        dataset_dir = mask_root / dataset
        if not dataset_dir.exists():
            continue
        for seed_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            seed = seed_dir.name
            if wanted_seeds is not None and seed not in wanted_seeds:
                continue
            yield dataset, seed, seed_dir


def method_from_dirname(name: str) -> Optional[str]:
    low = name.lower()
    if low == "el2n":
        return "el2n"
    if low == "grand":
        return "grand"
    if low == "forgetting":
        return "forgetting"
    return None


def collect_sources(mask_root: Path, datasets: Sequence[str], methods: Sequence[str], seeds: Optional[Sequence[str]]) -> List[SourceChoice]:
    wanted_methods = set(methods)
    grouped: Dict[Tuple[str, str, str, int], Dict[str, List[Path]]] = {}

    for dataset, seed, seed_dir in iter_dataset_seed_dirs(mask_root, datasets, seeds):
        for method_dir in [p for p in seed_dir.iterdir() if p.is_dir()]:
            method_key = method_from_dirname(method_dir.name)
            if method_key is None or method_key not in wanted_methods:
                continue

            for npy_path in method_dir.rglob("*.npy"):
                suffix = npy_path.name.lower()
                kind = None
                if suffix.endswith(".mask.npy"):
                    kind = "mask"
                elif suffix.endswith(".idx.npy"):
                    kind = "idx"
                if kind is None:
                    continue

                keep = extract_keep_ratio(npy_path)
                if keep is None:
                    continue

                key = (dataset, seed, method_key, keep)
                grouped.setdefault(key, {"mask": [], "idx": []})[kind].append(npy_path)

    choices: List[SourceChoice] = []
    for (dataset, seed, method_key, keep), files in sorted(grouped.items()):
        kind = "mask" if files["mask"] else "idx"
        candidates = sorted(files[kind], key=lambda p: str(p))
        source = candidates[0]
        choices.append(
            SourceChoice(
                dataset=dataset,
                seed=seed,
                method_key=method_key,
                keep_ratio=keep,
                source_file=source,
                source_kind=kind,
            )
        )

    return choices


def convert_one(choice: SourceChoice, out_root: Path, dry_run: bool, overwrite: bool, verbose: bool) -> Tuple[Path, int]:
    out_method = METHOD_OUTPUT_DIR[choice.method_key]
    target = out_root / out_method / choice.dataset / "resnet50" / choice.seed / f"mask_{choice.keep_ratio}.npz"

    if target.exists() and not overwrite:
        if verbose:
            print(f"[SKIP] exists (use --overwrite): {target}")
        # Return current mask count if possible for uniform logging.
        with np.load(target) as data:
            existing_mask = np.asarray(data["mask"]).reshape(-1)
        return target, int(existing_mask.astype(np.uint8).sum())

    if choice.source_kind == "mask":
        mask = load_mask_from_mask_file(choice.source_file)
    else:
        mask = load_mask_from_idx_file(choice.source_file, choice.dataset)

    mask = validate_mask(mask, choice.dataset, choice.source_file)
    selected_count = int(mask.sum())

    print(
        f"[CONVERT] source={choice.source_file} | target={target} | "
        f"selected_count={selected_count} | keep_ratio={choice.keep_ratio}"
    )

    if dry_run:
        return target, selected_count

    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, mask=mask.astype(np.uint8))
    return target, selected_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert data_diet masks/idx into .npz masks for train_after_selection.py")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10", "cifar100"],
        help="Datasets to process (default: cifar10 cifar100)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["EL2N", "GraNd", "forgetting"],
        help="Methods to process (default: EL2N GraNd forgetting)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="Optional seed list. If omitted, auto-scan existing seeds.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print conversion actions without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target .npz files.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    datasets = [d.lower() for d in args.datasets]
    unknown_dataset = [d for d in datasets if d not in DATASET_TRAIN_SIZE]
    if unknown_dataset:
        raise ValueError(f"Unsupported datasets: {unknown_dataset}. Supported: {sorted(DATASET_TRAIN_SIZE)}")

    method_keys: List[str] = []
    for m in args.methods:
        norm = normalize_method(m)
        if norm is None:
            raise ValueError(
                f"Unsupported method: {m}. Supported aliases: EL2N, GraNd, forgetting (case-insensitive)."
            )
        method_keys.append(norm)

    mask_root = Path(__file__).resolve().parent / "mask"
    out_root = Path(__file__).resolve().parent

    if args.verbose:
        print(f"[INFO] mask root: {mask_root}")
        print(f"[INFO] output root: {out_root}")
        print(f"[INFO] datasets={datasets}, methods={method_keys}, seeds={args.seeds}")

    choices = collect_sources(mask_root=mask_root, datasets=datasets, methods=method_keys, seeds=args.seeds)

    if not choices:
        print("[INFO] No matching source files found. Nothing to convert.")
        return

    converted = 0
    for choice in choices:
        convert_one(choice, out_root=out_root, dry_run=args.dry_run, overwrite=args.overwrite, verbose=args.verbose)
        converted += 1

    print(f"[DONE] Processed {converted} conversion task(s). dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
