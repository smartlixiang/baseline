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
rm -rf 'scores/${DATASET_NAME}/seed_${seed}' && \
rm -rf 'mask/${DATASET_NAME}/${seed}' && \
python sample_scoring.py --dataset ${DATASET_NAME} --data_root '${DATA_ROOT}' --clip_model_path '${CLIP_MODEL_PATH}' --seed ${seed} && \
python optimize_selection.py --dataset ${DATASET_NAME} --seed ${seed} --keep_ratios '${KEEP_RATIOS}'"

  tmux new-session -d -s "${session_name}" "bash -lc \"${cmd}\""
}

for i in "${!SESSIONS[@]}"; do
  launch_seed_session "${SESSIONS[$i]}" "${GPUS[$i]}" "${SEEDS[$i]}"
done

echo "waiting for all tmux sessions to finish..."
while true; do
  all_done=1
  for session_name in "${SESSIONS[@]}"; do
    if tmux has-session -t "${session_name}" 2>/dev/null; then
      all_done=0
      break
    fi
  done
  if [ "${all_done}" -eq 1 ]; then
    break
  fi
  sleep 15
done

python - <<'PY'
import csv
import os
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset = "tiny-imagenet"
mask_root = os.path.join(base_dir, "mask", dataset)
seeds = [22, 42, 96]
keep_ratios = [20, 30, 40, 50, 60, 70, 80, 90]

out_path = os.path.join(base_dir, "check.csv")

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["kr"] + [str(seed) for seed in seeds])
    for kr in keep_ratios:
        row = [kr]
        for seed in seeds:
            mask_path = os.path.join(mask_root, str(seed), f"mask_{kr}.npz")
            if not os.path.isfile(mask_path):
                row.append("MISSING")
                continue
            data = np.load(mask_path)
            if "indices" in data:
                selected = int(len(data["indices"]))
            elif "mask" in data:
                selected = int(np.asarray(data["mask"]).sum())
            else:
                selected = "INVALID"
            row.append(selected)
        writer.writerow(row)

print(f"[rerun_tiny] wrote mask count table to: {out_path}")
PY

echo "tmux sessions started (tiny-imagenet rerun):"
echo "  000 -> GPU 0 -> tiny-imagenet -> seed 22"
echo "  111 -> GPU 1 -> tiny-imagenet -> seed 42"
echo "  222 -> GPU 2 -> tiny-imagenet -> seed 96"
echo "all sessions completed and check.csv generated."
