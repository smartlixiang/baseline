

# tools/make_mask_from_scores.py
import argparse
import numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=str, required=True, help="Path to scores .npy (shape [N])")
    ap.add_argument("--keep_ratio", type=float, required=True, help="e.g. 0.9 means keep top 90% by score")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--name", type=str, required=True, help="Prefix name for outputs")
    ap.add_argument("--keep_high", action="store_true", help="Keep highest scores (default True for EL2N/GraNd)")
    args = ap.parse_args()

    scores_path = Path(args.scores)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = np.load(scores_path)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got shape={scores.shape}")

    N = scores.shape[0]
    keep_ratio = float(args.keep_ratio)
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0, 1].")

    keep_n = int(round(N * keep_ratio))
    keep_n = max(1, min(N, keep_n))

    # keep highest scores by default
    order = np.argsort(scores)
    if args.keep_high:
        keep_idx = order[-keep_n:]
    else:
        keep_idx = order[:keep_n]

    keep_idx = np.sort(keep_idx).astype(np.int64)
    mask = np.zeros(N, dtype=np.bool_)
    mask[keep_idx] = True

    # save
    np.save(out_dir / f"{args.name}.mask.npy", mask)
    np.save(out_dir / f"{args.name}.idx.npy", keep_idx)

    # also save a readable txt
    with open(out_dir / f"{args.name}.idx.txt", "w", encoding="utf-8") as f:
        for i in keep_idx.tolist():
            f.write(f"{i}\n")

    print(f"[OK] N={N}, keep_n={keep_n} ({keep_ratio:.2%})")
    print(f"[OK] mask: {out_dir / (args.name + '.mask.npy')}")
    print(f"[OK] idx:  {out_dir / (args.name + '.idx.npy')}")

if __name__ == "__main__":
    main()
