#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
SEED=22
DATA_ROOT="${BASE_DIR}/data"
CLIP_MODEL_PATH="${BASE_DIR}/clip_model/ViT-B-32.pt"
KEEP_RATIOS="20,30,40,50,60,70,80,90"
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"

all_adapters_exist=1
for ds in cifar10 cifar100 tiny-imagenet; do
  if [ ! -f "${BASE_DIR}/adapter_ckpt/${ds}/adapter_seed_${SEED}.pt" ]; then
    all_adapters_exist=0
    break
  fi
done

launch_session() {
  local session_name="$1"
  local gpu_id="$2"
  local dataset_name="$3"

  local adapter_path="${BASE_DIR}/adapter_ckpt/${dataset_name}/adapter_seed_${SEED}.pt"

  tmux kill-session -t "${session_name}" 2>/dev/null || true

  local maybe_train_cmd=""
  if [ "${all_adapters_exist}" -eq 0 ]; then
    maybe_train_cmd="if [ ! -f '${adapter_path}' ]; then python train_adapter.py --dataset ${dataset_name} --data_root '${DATA_ROOT}' --clip_model_path '${CLIP_MODEL_PATH}' --seed ${SEED}; fi &&"
  fi

  local cmd="cd '${BASE_DIR}' && \
source '${CONDA_SH}' && \
conda activate shampoo && \
export CUDA_VISIBLE_DEVICES=${gpu_id} && \
${maybe_train_cmd} \
python sample_scoring.py --dataset ${dataset_name} --data_root '${DATA_ROOT}' --clip_model_path '${CLIP_MODEL_PATH}' --seed ${SEED} && \
python optimize_selection.py --dataset ${dataset_name} --seed ${SEED} --keep_ratios '${KEEP_RATIOS}'"

  tmux new-session -d -s "${session_name}" "bash -lc \"${cmd}\""
}

launch_session "000" "0" "cifar10"
launch_session "111" "1" "cifar100"
launch_session "222" "2" "tiny-imagenet"

if [ "${all_adapters_exist}" -eq 1 ]; then
  echo "all adapter checkpoints exist; skipped adapter training stage"
fi
echo "tmux sessions started: 000(cifar10), 111(cifar100), 222(tiny-imagenet)"
