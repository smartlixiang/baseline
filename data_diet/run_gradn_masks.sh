#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

DATASETS=("cifar10" "cifar100")
SEEDS=(22 42 96)
KEEP_RATIOS=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2)

MODEL="resnet18_lowres"
LR=0.1
BETA=0.9
WEIGHT_DECAY=0.0005
NESTEROV=true

SCORE_EPOCH=10
N_AVG="${N_AVG:-1}"

# GraNd score 的 batch：必须极小（你已验证 5 能跑通）
SCORE_BATCH="${GRAND_SCORE_BATCH:-5}"

TRAIN_BATCH="${TRAIN_BATCH:-32}"
TEST_BATCH="${TEST_BATCH:-128}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0"

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore::FutureWarning,ignore::DeprecationWarning"

EP_STEPS=390
SCORE_STEP=$((EP_STEPS * SCORE_EPOCH))

mkdir -p "$ROOT/exps" "$ROOT/masks"

for dataset in "${DATASETS[@]}"; do
  for base_seed in "${SEEDS[@]}"; do

    EXP="grand_${dataset}_seed${base_seed}_epoch${SCORE_EPOCH}"
    RUN_DIR="$ROOT/exps/${EXP}"
    mkdir -p "$RUN_DIR"

    echo "=============================="
    echo "[GraNd] dataset=$dataset seed=$base_seed N_AVG=$N_AVG keep=${KEEP_RATIOS[*]}"
    echo "=============================="

    SCORE_FILES=()
    for k in $(seq 0 $((N_AVG-1))); do
      run_seed=$((base_seed + k * 1000003))
      run_id=$k

      one_exp="${EXP}/avg${N_AVG}/run_${run_id}"
      save_dir="$ROOT/exps/${one_exp}"

      score_path="$save_dir/grad_norm_scores/ckpt_${SCORE_STEP}.npy"
      if [ -f "$score_path" ]; then
        echo "[SKIP] score exists: $score_path"
        SCORE_FILES+=("$score_path")
        continue
      fi

      export CUDA_VISIBLE_DEVICES=$((k % 8))

      echo "[1] Train proxy (run=$run_id seed=$run_seed) -> step=$SCORE_STEP"
      python - <<PY
from data_diet.train import train
from types import SimpleNamespace

ROOT=r"$ROOT"
dataset=r"$dataset"
save_dir=r"$save_dir"

EP_STEPS=$EP_STEPS
num_steps=$SCORE_STEP

args=SimpleNamespace()
args.data_dir = ROOT + "/data"
args.dataset = dataset

args.subset=None
args.subset_size=None
args.scores_path=None
args.subset_offset=None
args.random_subset_seed=None

args.model=r"$MODEL"
args.model_seed=int($run_seed)
args.load_dir=None
args.ckpt=0

args.lr=float($LR)
args.beta=float($BETA)
args.weight_decay=float($WEIGHT_DECAY)
args.nesterov=bool($NESTEROV)
args.lr_vitaly=False
args.decay_factor=0.2
args.decay_steps=[60*EP_STEPS, 120*EP_STEPS, 160*EP_STEPS]

args.num_steps=int(num_steps)
args.train_seed=int($run_seed)
args.train_batch_size=int($TRAIN_BATCH)
args.test_batch_size=int($TEST_BATCH)
args.augment=True
args.track_forgetting=False

args.save_dir=save_dir
args.log_steps=EP_STEPS
args.early_step=0
args.early_save_steps=None
args.save_steps=EP_STEPS

train(args)
PY

      echo "[2] Compute GraNd score (run=$run_id) batch=$SCORE_BATCH"
      python scripts/get_run_score.py "$ROOT" "${EXP}/avg${N_AVG}" "$run_id" "$SCORE_STEP" "$SCORE_BATCH" grad_norm

      if [ ! -f "$score_path" ]; then
        echo "[ERR] score not found: $score_path"
        exit 1
      fi
      SCORE_FILES+=("$score_path")
    done

    # 平均（可选）
    if [ "$N_AVG" -eq 1 ]; then
      AVG_SCORE="${SCORE_FILES[0]}"
    else
      AVG_SCORE="$RUN_DIR/avg${N_AVG}/avg_scores/ckpt_${SCORE_STEP}.npy"
      python - <<PY
import numpy as np
from pathlib import Path
out_path=Path(r"$AVG_SCORE")
in_paths=[Path(p) for p in r"""${SCORE_FILES[*]}""".split()]
arrs=[np.load(str(p)) for p in in_paths]
m=np.mean(np.stack(arrs,0),0)
out_path.parent.mkdir(parents=True, exist_ok=True)
np.save(str(out_path), m)
print("[OK] averaged", len(arrs), "->", out_path)
PY
    fi

    # 生成 mask
    for kr in "${KEEP_RATIOS[@]}"; do
      tag="GraNd_${dataset}_seed${base_seed}_epoch${SCORE_EPOCH}_keep${kr}"
      out_dir="$ROOT/masks/${dataset}/GraNd/seed${base_seed}/epoch${SCORE_EPOCH}"
      python tools/make_mask_from_scores.py \
        --scores "$AVG_SCORE" \
        --keep_ratio "$kr" \
        --out_dir "$out_dir" \
        --name "$tag" \
        --keep_high
    done

    echo "[DONE] masks in: $ROOT/masks/${dataset}/GraNd/seed${base_seed}/epoch${SCORE_EPOCH}"
  done
done
