from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import DATASETS
from .select import generate_masks_for_ratios, load_or_extract_features
from .train_base import TrainConfig, train_base_model
from .utils import resolve_device, set_deterministic, set_seed

DEFAULT_SEEDS = [22, 42, 96]
DEFAULT_KEEP_RATIOS = [20, 30, 40, 50, 60, 70, 80, 90]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Moderate-DS in this repository.")
    parser.add_argument("--dataset", default="all", choices=["cifar10", "cifar100", "tiny-imagenet", "all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--keep-ratios", nargs="+", type=int, default=DEFAULT_KEEP_RATIOS)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stage", choices=["train", "extract", "select", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def iter_datasets(dataset_arg: str):
    return DATASETS if dataset_arg == "all" else (dataset_arg,)


def main() -> None:
    args = parse_args()
    set_deterministic()
    device = resolve_device(args.device)

    for dataset in iter_datasets(args.dataset):
        for seed in args.seeds:
            set_seed(seed)
            ckpt_path = Path("MDS") / "ckpt" / dataset / str(seed) / "best.pth"

            if args.stage in ("train", "all"):
                config = TrainConfig(
                    dataset=dataset,
                    data_root=args.data_root,
                    seed=seed,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    epochs=args.epochs,
                )
                ckpt_path = train_base_model(config)
            elif not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found for dataset={dataset}, seed={seed}: {ckpt_path}")

            if args.stage in ("extract", "all"):
                load_or_extract_features(
                    dataset=dataset,
                    data_root=args.data_root,
                    seed=seed,
                    ckpt_path=ckpt_path,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    force_recompute=args.force_recompute,
                )

            if args.stage in ("select", "all"):
                features, labels = load_or_extract_features(
                    dataset=dataset,
                    data_root=args.data_root,
                    seed=seed,
                    ckpt_path=ckpt_path,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    force_recompute=args.force_recompute,
                )
                generate_masks_for_ratios(
                    dataset=dataset,
                    seed=seed,
                    keep_ratios=args.keep_ratios,
                    features=features,
                    labels=labels,
                    mask_root="mask/MDS",
                )


if __name__ == "__main__":
    main()
