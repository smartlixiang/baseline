#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_NAME="tiny-imagenet"
DATA_ROOT="${BASE_DIR}/data"
CLIP_MODEL_PATH="${BASE_DIR}/clip_model/ViT-B-32.pt"
KEEP_RATIOS="20,30,40,50,60,70,80,90"
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"

SESSIONS=("000" "111" "222")
GPUS=("0" "1" "2")
SEEDS=("22" "42" "96")

launch_seed_session() {
  local session_name="$1"
  local gpu_id="$2"
  local seed="$3"

  tmux kill-session -t "${session_name}" 2>/dev/null || true

  local cmd="cd '${BASE_DIR}' && \
source '${CONDA_SH}' && \
conda activate shampoo && \
export CUDA_VISIBLE_DEVICES=${gpu_id} && \
rm -f 'adapter_ckpt/${DATASET_NAME}/adapter_seed_${seed}.pt' && \
rm -rf 'scores/${DATASET_NAME}/seed_${seed}' && \
rm -rf 'mask/${DATASET_NAME}/${seed}' && \
python train_adapter.py --dataset ${DATASET_NAME} --data_root '${DATA_ROOT}' --clip_model_path '${CLIP_MODEL_PATH}' --seed ${seed} && \
python sample_scoring.py --dataset ${DATASET_NAME} --data_root '${DATA_ROOT}' --clip_model_path '${CLIP_MODEL_PATH}' --seed ${seed} && \
python optimize_selection.py --dataset ${DATASET_NAME} --seed ${seed} --keep_ratios '${KEEP_RATIOS}'"

  tmux new-session -d -s "${session_name}" "bash -lc \"${cmd}\""
}

for i in "${!SESSIONS[@]}"; do
  launch_seed_session "${SESSIONS[$i]}" "${GPUS[$i]}" "${SEEDS[$i]}"
done

echo "tmux sessions started (tiny-imagenet rerun):"
echo "  000 -> GPU 0 -> tiny-imagenet -> seed 22"
echo "  111 -> GPU 1 -> tiny-imagenet -> seed 42"
echo "  222 -> GPU 2 -> tiny-imagenet -> seed 96"
echo "use 'tmux ls' to check and 'tmux attach -t 000|111|222' to monitor."
