#!/usr/bin/env bash
set -euo pipefail

# One-place seed variable for future switch (22/42/96 ...).
SEED=22
BASE_DIR="/workspace/baseline/YangCLIP"
DATA_ROOT="${BASE_DIR}/data"
CLIP_MODEL_PATH="${BASE_DIR}/clip_model/ViT-B-32.pt"
KEEP_RATIOS="20,30,40,50,60,70,80,90"

SESSION_000="000"
SESSION_111="111"
SESSION_222="222"

DATASET_000="cifar10"
DATASET_111="cifar100"
DATASET_222="tiny-imagenet"

GPU_000="0"
GPU_111="1"
GPU_222="2"

launch_session() {
  local session_name="$1"
  local gpu_id="$2"
  local dataset_name="$3"

  local adapter_path="${BASE_DIR}/adapter_ckpt/${dataset_name}/adapter_seed_${SEED}.pt"

  if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "[run_all] tmux session ${session_name} already exists, killing and recreating..."
    tmux kill-session -t "${session_name}"
  fi

  # Note: explicit conda init/activate before all python commands.
  local cmd="cd ${BASE_DIR} && \
conda init && \
conda activate shampoo && \
export CUDA_VISIBLE_DEVICES=${gpu_id} && \
if [ ! -f '${adapter_path}' ]; then \
  python ${BASE_DIR}/train_adapter.py --dataset ${dataset_name} --data_root ${DATA_ROOT} --clip_model_path ${CLIP_MODEL_PATH} --seed ${SEED}; \
fi && \
python ${BASE_DIR}/sample_scoring.py --dataset ${dataset_name} --data_root ${DATA_ROOT} --clip_model_path ${CLIP_MODEL_PATH} --seed ${SEED} && \
python ${BASE_DIR}/optimize_selection.py --dataset ${dataset_name} --data_root ${DATA_ROOT} --clip_model_path ${CLIP_MODEL_PATH} --seed ${SEED} --keep_ratios ${KEEP_RATIOS}"

  tmux new-session -d -s "${session_name}" "${cmd}"
}

launch_session "${SESSION_000}" "${GPU_000}" "${DATASET_000}"
launch_session "${SESSION_111}" "${GPU_111}" "${DATASET_111}"
launch_session "${SESSION_222}" "${GPU_222}" "${DATASET_222}"

echo "[run_all] Started tmux sessions:"
echo "  session ${SESSION_000} -> GPU ${GPU_000} -> ${DATASET_000}"
echo "  session ${SESSION_111} -> GPU ${GPU_111} -> ${DATASET_111}"
echo "  session ${SESSION_222} -> GPU ${GPU_222} -> ${DATASET_222}"
