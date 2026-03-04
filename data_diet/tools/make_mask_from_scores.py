import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def build_one_mask(scores: np.ndarray, keep_ratio: float, keep_high: bool):
    n = scores.shape[0]
    keep_n = int(round(n * keep_ratio))
    keep_n = max(1, min(n, keep_n))

    order = np.argsort(scores)
    keep_idx = order[-keep_n:] if keep_high else order[:keep_n]
    keep_idx = np.sort(keep_idx).astype(np.int64)

    mask = np.zeros(n, dtype=np.bool_)
    mask[keep_idx] = True
    return keep_n, keep_idx, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=str, required=True, help="Path to 1D score .npy")
    ap.add_argument(
        "--keep_ratios",
        type=float,
        nargs="+",
        required=True,
        help="One or more ratios, e.g. 0.9 0.8 ... 0.2",
    )
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--name_prefix", type=str, required=True)
    ap.add_argument("--keep_high", action="store_true", help="Keep highest scores")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = np.load(Path(args.scores))
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got shape={scores.shape}")

    for r in args.keep_ratios:
        if not (0.0 < float(r) <= 1.0):
            raise ValueError(f"keep ratio must be in (0, 1], got {r}")

    ratios = sorted(set(float(r) for r in args.keep_ratios), reverse=True)
    bar = tqdm(ratios, desc=f"mask:{args.name_prefix}", leave=True)
    for ratio in bar:
        keep_n, keep_idx, mask = build_one_mask(scores, ratio, args.keep_high)
        ratio_tag = f"{int(round(ratio * 100)):02d}"
        name = f"{args.name_prefix}_keep{ratio_tag}"

        np.save(out_dir / f"{name}.mask.npy", mask)
        np.save(out_dir / f"{name}.idx.npy", keep_idx)
        with open(out_dir / f"{name}.idx.txt", "w", encoding="utf-8") as f:
            for i in keep_idx.tolist():
                f.write(f"{i}\n")

        bar.set_postfix(keep_n=keep_n, ratio=f"{ratio:.0%}")

    print(f"[OK] wrote {len(ratios)} mask groups to: {out_dir}")


if __name__ == "__main__":
    main()
