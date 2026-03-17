from __future__ import annotations
from herding.utils import (
    DEFAULT_DATASETS,
    DEFAULT_KEEP_RATIOS,
    DEFAULT_SEEDS,
    ensure_dir,
    get_device,
    set_seed,
)
from herding.models import ResNet18FeatureExtractor
from herding.herding_select import extract_features, generate_masks_for_keep_ratios
from herding.datasets import build_train_loader

import argparse
from pathlib import Path
import sys

import torch
from tqdm import tqdm

# Make sure `import herding.*` works when running as:
# - from repo root:   python herding/run_all_herding.py
# - from herding dir: python run_all_herding.py
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Run class-wise feature herding for CIFAR / Tiny-ImageNet.")

    default_base_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(default_base_dir / "data"),
        help="Root directory for datasets (supports CIFAR and Tiny-ImageNet).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(default_base_dir / "mask"),
        help="Root directory for saved masks.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(default_base_dir / "cache"),
        help="Feature cache directory (to avoid repeated extraction).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--keep-ratios", nargs="+", type=float, default=DEFAULT_KEEP_RATIOS)
    parser.add_argument("--disable-cache", action="store_true", help="Always re-extract features.")
    parser.add_argument(
        "--disable-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained initialization for ResNet18.",
    )
    return parser.parse_args()


def validate_args(args):
    for dataset in args.datasets:
        if dataset not in ("cifar10", "cifar100", "tiny-imagenet", "tiny-imagenet-200"):
            raise ValueError(f"Unsupported dataset: {dataset}")

    for ratio in args.keep_ratios:
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"Keep ratio must be in (0, 1], got {ratio}")


def maybe_load_or_extract_features(args, dataset_name: str, seed: int, device: torch.device):
    cache_path = Path(args.cache_dir) / f"{dataset_name}_seed{seed}_resnet18.pt"

    if not args.disable_cache and cache_path.exists():
        print(f"[Cache] Loading features from {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        return payload["features"], payload["labels"], payload["num_classes"]

    loader, dataset = build_train_loader(
        dataset_name=dataset_name,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = ResNet18FeatureExtractor(prefer_pretrained=not args.disable_pretrained)
    features, labels = extract_features(model=model, dataloader=loader, device=device, normalize_features=True)

    num_classes = len(dataset.classes)

    if not args.disable_cache:
        ensure_dir(cache_path.parent)
        torch.save(
            {
                "features": features,
                "labels": labels,
                "num_classes": num_classes,
            },
            cache_path,
        )
        print(f"[Cache] Saved features to {cache_path}")

    return features, labels, num_classes


def main():
    args = parse_args()
    validate_args(args)

    device = get_device()
    print(f"Using device: {device}")

    total_tasks = len(args.datasets) * len(args.seeds)
    tasks = [(d, s) for d in args.datasets for s in args.seeds]

    combo_progress = tqdm(tasks, total=total_tasks, desc="All dataset-seed combinations")
    for dataset_name, seed in combo_progress:
        combo_progress.set_postfix(dataset=dataset_name, seed=seed)
        set_seed(seed)

        features, labels, num_classes = maybe_load_or_extract_features(
            args=args,
            dataset_name=dataset_name,
            seed=seed,
            device=device,
        )

        generated = generate_masks_for_keep_ratios(
            features=features,
            labels=labels,
            keep_ratios=args.keep_ratios,
            num_classes=num_classes,
            dataset_name=dataset_name,
            seed=seed,
            output_root=args.output_root,
        )

        print(
            f"[Done] dataset={dataset_name} seed={seed} generated mask cut ratios: "
            f"{sorted(generated.keys())}"
        )


if __name__ == "__main__":
    main()
