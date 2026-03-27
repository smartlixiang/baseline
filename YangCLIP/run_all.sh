#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CLIP_MODEL_PATH="clip_model/ViT-B-32.pt"
DATA_ROOT_PATH="data"
KEEP_RATIOS=(0.2 0.3 0.4 0.6 0.7 0.8 0.9)

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is required but not found in PATH."
  exit 1
fi

if [ ! -f "$CLIP_MODEL_PATH" ]; then
  echo "Error: CLIP model not found at $CLIP_MODEL_PATH"
  exit 1
fi

run_dataset_job () {
  local session_name="$1"
  local gpu_id="$2"
  local dataset_name="$3"
  local batch_size="$4"

  tmux kill-session -t "$session_name" >/dev/null 2>&1 || true

  tmux new-session -d -s "$session_name" "bash -lc '
    conda activate shampoo
    for seed in 22 42 96; do
      CUDA_VISIBLE_DEVICES=$gpu_id python train_adapter.py \
        --dataset $dataset_name \
        --seed \$seed \
        --batch_size $batch_size \
        --epochs 30 \
        --lr 1e-4 \
        --clip_path \"$CLIP_MODEL_PATH\" \
        --data_root \"$DATA_ROOT_PATH\"

      CUDA_VISIBLE_DEVICES=$gpu_id python sample_scoring.py \
        --dataset $dataset_name \
        --seed \$seed \
        --batch_size $batch_size \
        --clip_path \"$CLIP_MODEL_PATH\" \
        --data_root \"$DATA_ROOT_PATH\"

      for keep_ratio in ${KEEP_RATIOS[*]}; do
        CUDA_VISIBLE_DEVICES=$gpu_id python generate_mask.py \
          --dataset $dataset_name \
          --seed \$seed \
          --keep_ratio \$keep_ratio \
          --batch_size $batch_size \
          --clip_path \"$CLIP_MODEL_PATH\" \
          --data_root \"$DATA_ROOT_PATH\"
      done
    done
  '"
}

run_dataset_job "000" "0" "cifar10" "256"
run_dataset_job "111" "1" "cifar100" "256"
run_dataset_job "222" "2" "tiny-imagenet" "64"

echo "Started tmux sessions:"
echo "  000 -> GPU0 / cifar10"
echo "  111 -> GPU1 / cifar100"
echo "  222 -> GPU2 / tiny-imagenet"
echo "Use 'tmux attach -t 000|111|222' to monitor."
