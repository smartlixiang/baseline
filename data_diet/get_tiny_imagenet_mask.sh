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

echo "[INFO] score_epoch=$SCORE_EPOCH score_step=$SCORE_STEP"

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
  [ -f "$EXP_DIR/forget_scores/ckpt_$((90*STEPS_PER_EPOCH)).npy" ] || python "$ROOT/scripts/get_mean_score.py" "$ROOT" "$EXP_NAME" 3 "$((90*STEPS_PER_EPOCH))" "forget"

  python - <<PY
import numpy as np
from pathlib import Path
from tqdm import tqdm

root=Path(r"$ROOT")
exp=Path(r"$EXP_DIR")
seed="$base_seed"
score_step=int($SCORE_STEP)
final_step=int($((90*STEPS_PER_EPOCH)))
ratios=[float(x) for x in "$KEEP_RATIOS_STR".split()]
methods=[
    ("E2LN", exp/"error_l2_norm_scores"/f"ckpt_{score_step}.npy", True),
    ("GraNd", exp/"grad_norm_scores"/f"ckpt_{score_step}.npy", True),
    ("Forgetting", exp/"forget_scores"/f"ckpt_{final_step}.npy", False),
]
for method, path, keep_high in methods:
    scores=np.load(path)
    out_dir=root/method/"tiny-imagenet"/seed
    out_dir.mkdir(parents=True, exist_ok=True)
    for ratio in tqdm(ratios, desc=f"{method}-seed{seed}"):
        out=out_dir/f"mask_{ratio}.npz"
        if out.exists():
            continue
        n=len(scores)
        k=max(1, min(n, int(round(n*ratio))))
        order=np.argsort(scores)
        idx=np.sort(order[-k:] if keep_high else order[:k]).astype(np.int64)
        mask=np.zeros(n, dtype=np.bool_)
        mask[idx]=True
        np.savez_compressed(out, mask=mask, idx=idx, keep_ratio=ratio, method=method, dataset="tiny-imagenet", seed=int(seed))
print("[OK] mask export complete")
PY

done

