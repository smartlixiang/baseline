#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$(pwd)" != "$SCRIPT_DIR" ]; then
  echo "Error: please run run_all.sh inside YangCLIP directory: $SCRIPT_DIR"
  exit 1
fi

ROOT_DIR="$SCRIPT_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is required but not found in PATH."
  exit 1
fi

CLIP_MODEL_PATH="$ROOT_DIR/clip_model/ViT-B-32.pt"
DATA_ROOT_PATH="$ROOT_DIR/data"

if [ ! -f "$CLIP_MODEL_PATH" ]; then
  echo "Error: CLIP model not found at $CLIP_MODEL_PATH"
  exit 1
fi

for ds_dir in cifar-100-python cifar-10-batches-py tiny-imagenet-200; do
  if [ ! -d "$DATA_ROOT_PATH/$ds_dir" ]; then
    echo "Error: dataset directory not found: $DATA_ROOT_PATH/$ds_dir"
    exit 1
  fi
done

run_dataset_job () {
  local session_name="$1"
  local gpu_id="$2"
  local dataset_name="$3"

  tmux new-session -d -s "$session_name" "bash -lc '
    conda activate shampoo
    for seed in 22 42 96; do
      CUDA_VISIBLE_DEVICES=$gpu_id python train_adapter.py \
        --dataset $dataset_name \
        --seed \$seed \
        --epochs 50 \
        --batch_size 128 \
        --lr 1e-3 \
        --clip_path \"$CLIP_MODEL_PATH\" \
        --data_root \"$DATA_ROOT_PATH\"

      CUDA_VISIBLE_DEVICES=$gpu_id python generate_mask.py \
        --dataset $dataset_name \
        --seed \$seed \
        --keep_ratio 0.9 \
        --clip_path \"$CLIP_MODEL_PATH\" \
        --data_root \"$DATA_ROOT_PATH\"
    done
  '"
}

run_dataset_job "000" "0" "cifar10"
run_dataset_job "111" "1" "cifar100"
run_dataset_job "222" "2" "tiny-imagenet"

echo "Started tmux sessions: 000 (GPU0/cifar10), 111 (GPU1/cifar100), 222 (GPU2/tiny-imagenet)."
