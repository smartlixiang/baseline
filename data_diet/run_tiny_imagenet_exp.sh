#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
BASE_SEED="${BASE_SEED:-${1:-}}"
if [ -z "$BASE_SEED" ]; then
  echo "Usage: bash run_tiny_imagenet_exp.sh <base_seed>"
  echo "   or: BASE_SEED=<base_seed> bash run_tiny_imagenet_exp.sh"
  exit 1
fi

DATASET="tiny-imagenet"
MODEL="resnet34_lowres"
EPOCHS="${EPOCHS:-90}"
TRAIN_BATCH="${TRAIN_BATCH:-64}"
TEST_BATCH="${TEST_BATCH:-256}"
SCORE_EPOCH="${SCORE_EPOCH:-10}"
K_RUNS="${K_RUNS:-3}"

if [ "$K_RUNS" -ne 3 ]; then
  echo "[ERR] Tiny-ImageNet pipeline requires K_RUNS=3 per base seed to keep one mask group per base seed."
  exit 1
fi

EXP_NAME="tiny_imagenet_${MODEL}_seed${BASE_SEED}_E${EPOCHS}_b${TRAIN_BATCH}_k3"
EXP_DIR="$ROOT/exps/$EXP_NAME"
mkdir -p "$EXP_DIR"

echo "[INFO] base_seed=$BASE_SEED dataset=$DATASET model=$MODEL"
echo "[INFO] exp_dir=$EXP_DIR"

for run_id in $(seq 0 $((K_RUNS-1))); do
  real_seed=$((BASE_SEED * (run_id + 1)))
  run_dir="$EXP_DIR/run_${run_id}"

  if [ -f "$run_dir/args.json" ]; then
    echo "[SKIP TRAIN] run=$run_id seed=$real_seed already done"
    continue
  fi

  echo "[RUN] run=$run_id seed=$real_seed"
  python "$ROOT/scripts/run_full_data_tiny_imagenet.py" "$ROOT" "$EXP_NAME" "$run_id" "$real_seed" "$SCORE_EPOCH"
  python - <<PY
import json
from pathlib import Path
p=Path(r"$run_dir")/"args.json"
a=json.loads(p.read_text())
a["base_seed"]=$BASE_SEED
a["run_id"]=$run_id
a["real_seed"]=$real_seed
a["score_epoch"]=$SCORE_EPOCH
p.write_text(json.dumps(a, indent=2))
print(f"[META] updated {p}")
PY

done

echo "[DONE] training complete for base_seed=$BASE_SEED (k=$K_RUNS)"
