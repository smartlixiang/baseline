#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-cifar100}
SEED=${2:-22}
DEVICE=${3:-0}
if [ "$#" -ge 3 ]; then
  shift 3
else
  shift "$#"
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$SCRIPT_DIR"

for KR in 20 30 40 50 60 70 80 90; do
  echo "Running RLSelector: dataset=${DATASET}, seed=${SEED}, keep_ratio=${KR}, device=${DEVICE}"
  python train.py \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --keep_ratio "$KR" \
    --root "$REPO_ROOT/data" \
    --output_root "$REPO_ROOT/mask" \
    --device "$DEVICE" \
    "$@"
done
