import argparse
from pathlib import Path

import numpy as np

from data_loader_with_index import DATASET_CHOICES

SEED_CHOICES = [22, 42, 96]


def resolve_user_path(path_arg: str, script_dir: Path) -> Path:
    user_path = Path(path_arg)
    if user_path.is_absolute():
        return user_path.resolve()

    candidates = [
        (script_dir.parent / user_path).resolve(),
        (script_dir / user_path).resolve(),
        (Path.cwd() / user_path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return (script_dir / user_path).resolve()


def main():
    parser = argparse.ArgumentParser(description="Generate sample selection mask from scores")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser.add_argument("--seed", type=int, required=True, choices=SEED_CHOICES)
    parser.add_argument("--keep_ratio", type=float, required=True)
    parser.add_argument("--clip_path", type=str, default="clip_model/ViT-B-32.pt")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if not (0 < args.keep_ratio <= 1):
        raise ValueError("keep_ratio must be in (0, 1].")

    script_dir = Path(__file__).resolve().parent
    clip_path = resolve_user_path(args.clip_path, script_dir)
    if not clip_path.exists():
        raise FileNotFoundError(
            f"CLIP model file not found: {clip_path}. "
            "Please prepare local file `clip_model/ViT-B-32.pt`."
        )

    score_path = Path("Pruning_Scores") / args.dataset / str(args.seed) / "scores.npy"
    if not score_path.exists():
        raise FileNotFoundError(f"Score file not found: {score_path}")

    scores = np.load(score_path)
    if scores.ndim != 1:
        raise ValueError(f"scores.npy must be 1D, but got shape {scores.shape}")

    num_samples = scores.shape[0]
    keep_num = int(num_samples * args.keep_ratio)
    keep_num = max(1, min(num_samples, keep_num))

    topk_idx = np.argpartition(scores, -keep_num)[-keep_num:]

    mask = np.zeros(num_samples, dtype=np.uint8)
    mask[topk_idx] = 1

    save_path = Path("mask") / args.dataset / str(args.seed) / f"mask_{int(args.keep_ratio * 100)}.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(save_path, mask=mask)
    print(f"Saved mask to {save_path}")


if __name__ == "__main__":
    main()
