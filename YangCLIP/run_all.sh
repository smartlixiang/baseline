#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
SEEDS=(22 42 96)
DATA_ROOT="${BASE_DIR}/data"
CLIP_MODEL_PATH="${BASE_DIR}/clip_model/ViT-B-32.pt"
KEEP_RATIOS="20,30,40,50,60,70,80,90"
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"

launch_session() {
  local session_name="$1"
  local gpu_id="$2"
  local dataset_name="$3"

  tmux kill-session -t "${session_name}" 2>/dev/null || true

  local pipeline_cmd=""
  for seed in "${SEEDS[@]}"; do
    local adapter_path="${BASE_DIR}/adapter_ckpt/${dataset_name}/adapter_seed_${seed}.pt"

    pipeline_cmd+="if [ ! -f '${adapter_path}' ]; then "
    pipeline_cmd+="python train_adapter.py --dataset ${dataset_name} --data_root '${DATA_ROOT}' --clip_model_path '${CLIP_MODEL_PATH}' --seed ${seed}; "
    pipeline_cmd+="else "
    pipeline_cmd+="echo '[run_all] adapter exists, skip training: ${adapter_path}'; "
    pipeline_cmd+="fi && "

    pipeline_cmd+="python sample_scoring.py --dataset ${dataset_name} --data_root '${DATA_ROOT}' --clip_model_path '${CLIP_MODEL_PATH}' --seed ${seed} && "
    pipeline_cmd+="python optimize_selection.py --dataset ${dataset_name} --seed ${seed} --keep_ratios '${KEEP_RATIOS}' && "
  done

  pipeline_cmd="${pipeline_cmd%&& }"

  local cmd="cd '${BASE_DIR}' && \
source '${CONDA_SH}' && \
conda activate shampoo && \
export CUDA_VISIBLE_DEVICES=${gpu_id} && \
${pipeline_cmd}"

  tmux new-session -d -s "${session_name}" "bash -lc \"${cmd}\""
}

launch_session "000" "0" "cifar10"
launch_session "111" "1" "cifar100"
launch_session "222" "2" "tiny-imagenet"

echo "tmux sessions started:"
echo "  000 -> GPU 0 -> cifar10   -> seeds: ${SEEDS[*]}"
echo "  111 -> GPU 1 -> cifar100  -> seeds: ${SEEDS[*]}"
echo "  222 -> GPU 2 -> tiny-imagenet -> seeds: ${SEEDS[*]}"