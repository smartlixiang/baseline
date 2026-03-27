import argparse
import subprocess
import sys

from data_loader_with_index import DATASET_CHOICES

SEED_CHOICES = [22, 42, 96]


def main():
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper: delegate to generate_mask.py"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser.add_argument("--seed", type=int, required=True, choices=SEED_CHOICES)
    parser.add_argument("--keep_ratio", type=float, required=True)
    parser.add_argument("--clip_path", type=str, default="clip_model/ViT-B-32.pt")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "generate_mask.py",
        "--dataset",
        args.dataset,
        "--seed",
        str(args.seed),
        "--keep_ratio",
        str(args.keep_ratio),
        "--clip_path",
        args.clip_path,
        "--data_root",
        args.data_root,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
