#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
DATASET="tiny-imagenet"
MODEL="${MODEL:-resnet34_lowres}"
BASE_SEEDS_STR="${BASE_SEEDS:-22 42 96}"
KEEP_RATIOS_STR="${KEEP_RATIOS:-0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2}"
SCORE_EPOCH="${SCORE_EPOCH:-10}"
TRAIN_BATCH="${TRAIN_BATCH:-64}"
EL2N_SCORE_BATCH="${EL2N_SCORE_BATCH:-128}"
GRAND_SCORE_BATCH="${GRAND_SCORE_BATCH:-16}"

if [ "$MODEL" != "resnet34_lowres" ]; then
  echo "[ERR] tiny-imagenet mask pipeline requires MODEL=resnet34_lowres, got $MODEL"
  exit 1
fi

for required_ratio in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  if [[ " $KEEP_RATIOS_STR " != *" $required_ratio "* ]]; then
    echo "[ERR] KEEP_RATIOS must include $required_ratio (20/30/40/50/60/70/80/90%)"
    exit 1
  fi
done

count_train() {
  python - <<PY
import os
root=r"$ROOT/data/tiny-imagenet-200/train"
t=0
for c in os.listdir(root):
    d=os.path.join(root,c,'images')
    if os.path.isdir(d):
      t+=sum(1 for f in os.listdir(d) if os.path.isfile(os.path.join(d,f)))
print(t)
PY
}

NTRAIN="$(count_train)"
STEPS_PER_EPOCH=$((NTRAIN / TRAIN_BATCH))
SCORE_STEP=$((SCORE_EPOCH * STEPS_PER_EPOCH))
FINAL_STEP=$((90 * STEPS_PER_EPOCH))

base_seed_count=$(wc -w <<<"$BASE_SEEDS_STR")
if [ "$base_seed_count" -ne 3 ]; then
  echo "[WARN] expected 3 base seeds for final 3 mask groups, got $base_seed_count"
fi

echo "[INFO] score_epoch=$SCORE_EPOCH score_step=$SCORE_STEP final_step=$FINAL_STEP"
echo "[INFO] each base seed aggregates 3 independent runs for per-seed mean score"

for base_seed in $BASE_SEEDS_STR; do
  EXP_NAME="tiny_imagenet_${MODEL}_seed${base_seed}_E90_b${TRAIN_BATCH}_k3"
  EXP_DIR="$ROOT/exps/$EXP_NAME"
  echo "================ base_seed=$base_seed ================"

  for run_id in 0 1 2; do
    run_dir="$EXP_DIR/run_${run_id}"
    if [ ! -f "$run_dir/args.json" ]; then
      echo "[ERR] missing run args: $run_dir/args.json"
      exit 1
    fi

    el2n_file="$run_dir/error_l2_norm_scores/ckpt_${SCORE_STEP}.npy"
    grand_file="$run_dir/grad_norm_scores/ckpt_${SCORE_STEP}.npy"

    if [ ! -f "$el2n_file" ]; then
      echo "[EL2N] base_seed=$base_seed run=$run_id"
      python "$ROOT/scripts/get_run_score.py" "$ROOT" "$EXP_NAME" "$run_id" "$SCORE_STEP" "$EL2N_SCORE_BATCH" "l2_error"
    else
      echo "[SKIP EL2N] run=$run_id exists"
    fi

    if [ ! -f "$grand_file" ]; then
      echo "[GraNd] base_seed=$base_seed run=$run_id"
      python "$ROOT/scripts/get_run_score.py" "$ROOT" "$EXP_NAME" "$run_id" "$SCORE_STEP" "$GRAND_SCORE_BATCH" "grad_norm"
    else
      echo "[SKIP GraNd] run=$run_id exists"
    fi
  done

  [ -f "$EXP_DIR/error_l2_norm_scores/ckpt_${SCORE_STEP}.npy" ] || python "$ROOT/scripts/get_mean_score.py" "$ROOT" "$EXP_NAME" 3 "$SCORE_STEP" "l2_error"
  [ -f "$EXP_DIR/grad_norm_scores/ckpt_${SCORE_STEP}.npy" ] || python "$ROOT/scripts/get_mean_score.py" "$ROOT" "$EXP_NAME" 3 "$SCORE_STEP" "grad_norm"
  [ -f "$EXP_DIR/forget_scores/ckpt_${FINAL_STEP}.npy" ] || python "$ROOT/scripts/get_mean_score.py" "$ROOT" "$EXP_NAME" 3 "$FINAL_STEP" "forget"

  python - <<PY
import numpy as np
from pathlib import Path
from tqdm import tqdm

root=Path(r"$ROOT")
exp=Path(r"$EXP_DIR")
seed="$base_seed"
score_step=int($SCORE_STEP)
final_step=int($FINAL_STEP)
ratios=[float(x) for x in "$KEEP_RATIOS_STR".split()]
methods=[
    ("E2LN", exp/"error_l2_norm_scores"/f"ckpt_{score_step}.npy", True),
    ("GraNd", exp/"grad_norm_scores"/f"ckpt_{score_step}.npy", True),
    ("Forgetting", exp/"forget_scores"/f"ckpt_{final_step}.npy", False),
]
for method, path, keep_high in methods:
    if not path.exists():
        raise FileNotFoundError(f"missing mean score file: {path}")
    scores=np.load(path)
    out_dir=root/method/"tiny-imagenet"/seed
    out_dir.mkdir(parents=True, exist_ok=True)
    for ratio in tqdm(ratios, desc=f"{method}-seed{seed}"):
        out=out_dir/f"mask_{ratio}.npz"
        n=len(scores)
        k=max(1, min(n, int(round(n*ratio))))
        order=np.argsort(scores)
        idx=np.sort(order[-k:] if keep_high else order[:k]).astype(np.int64)
        mask=np.zeros(n, dtype=np.bool_)
        mask[idx]=True
        np.savez_compressed(out, mask=mask, idx=idx, keep_ratio=ratio, method=method, dataset="tiny-imagenet", seed=int(seed))

for method, _, _ in methods:
    for ratio in ratios:
        out=root/method/"tiny-imagenet"/seed/f"mask_{ratio}.npz"
        if not out.exists():
            raise FileNotFoundError(f"missing mask output: {out}")
print("[OK] mask export complete and paths verified")
PY

done

echo "[DONE] 9 proxy runs in total (3 seeds x 3 runs), output 3 mask groups (one per base seed)."
