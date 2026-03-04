#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# =========================
# Experiment grid
# =========================
DATASETS=("cifar10" "cifar100")          # 如需别的数据集，后续再加
BASE_SEEDS=(22 42 96)
KEEP_RATIOS=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2)

# =========================
# Paper-default training (except batch-related)
# =========================
MODEL="resnet18_lowres"
LR=0.1
BETA=0.9
WEIGHT_DECAY=0.0005
NESTEROV=true
AUGMENT=true

# Paper uses 200 epochs for proxy training
TRAIN_EPOCHS="${TRAIN_EPOCHS:-200}"

# EL2N/GraNd are evaluated in early epochs; paper scans 0-20.
# Default: epoch 20 (most common choice for producing masks).
SCORE_EPOCHS=(${SCORE_EPOCHS:-20})  # e.g. SCORE_EPOCHS="0 4 8 12 16 20"

# Independent runs averaged: paper uses 10; you use 8 GPUs => 8 runs.
N_RUNS="${N_RUNS:-8}"              # default 8

# =========================
# Batch settings (allowed to differ)
# =========================
TRAIN_BATCH="${TRAIN_BATCH:-32}"
TEST_BATCH="${TEST_BATCH:-128}"

# Score batch must divide N_train (CIFAR train N=50000) because repo uses np.split
EL2N_SCORE_BATCH="${EL2N_SCORE_BATCH:-100}"     # 50000/100=500 batches
GRAND_SCORE_BATCH="${GRAND_SCORE_BATCH:-5}"     # 50000/5=10000 batches (you validated)

# =========================
# Seed policy for 8 independent runs
# =========================
SEED_INCR=1000003
# run_seed = base_seed + run_id * SEED_INCR  (reproducible & independent)

# =========================
# JAX/XLA stability flags (you validated needed)
# =========================
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0"

# Reduce log noise (optional)
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore::FutureWarning,ignore::DeprecationWarning"

# =========================
# Output roots
# =========================
EXP_ROOT="$ROOT/exps_formal"
MASK_ROOT="$ROOT/masks_formal"
mkdir -p "$EXP_ROOT" "$MASK_ROOT"

if [ ! -f "$ROOT/tools/make_mask_from_scores.py" ]; then
  echo "[ERR] missing tools/make_mask_from_scores.py"
  echo "      Please create it (used to convert mean scores -> masks)."
  exit 1
fi

# =========================
# Dataset sizes (classic CIFAR; keep stable)
# If you add other datasets later, extend this mapping.
# =========================
num_train_examples () {
  case "$1" in
    cifar10) echo 50000 ;;
    cifar100) echo 50000 ;;
    *) echo "" ;;
  esac
}

steps_per_epoch () {
  local n="$1"
  local b="$2"
  python - <<PY
import math
n=int("$n"); b=int("$b")
print((n + b - 1)//b)
PY
}

# =========================
# Train one run to 200 epochs with forgetting tracking
# =========================
train_one () {
  local dataset="$1"
  local run_seed="$2"
  local save_dir="$3"
  local spe="$4"        # steps per epoch
  local num_steps="$5"  # total steps
  python - <<PY
from data_diet.train import train
from types import SimpleNamespace

ROOT=r"$ROOT"
dataset=r"$dataset"
save_dir=r"$save_dir"

EP_STEPS=int($spe)
NUM_STEPS=int($num_steps)

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
args.nesterov=True
args.lr_vitaly=False
args.decay_factor=0.2
# LR decay at epochs 60/120/160 (paper default); convert to steps (batch-related)
args.decay_steps=[60*EP_STEPS, 120*EP_STEPS, 160*EP_STEPS]

args.num_steps=int(NUM_STEPS)
args.train_seed=int($run_seed)
args.train_batch_size=int($TRAIN_BATCH)
args.test_batch_size=int($TEST_BATCH)
args.augment=True

# Forgetting needs full training trajectory tracking
args.track_forgetting=True

args.save_dir=save_dir
args.log_steps=EP_STEPS
args.early_step=0
args.early_save_steps=None
# Save every epoch so we can score at epoch=20 (or any in SCORE_EPOCHS)
args.save_steps=EP_STEPS

train(args)
PY
}

# =========================
# Compute per-run scores at a given epoch checkpoint
# =========================
score_one () {
  local exp_name="$1"
  local run_id="$2"
  local score_step="$3"
  local batch_sz="$4"
  local type="$5"
  python scripts/get_run_score.py "$ROOT" "$exp_name" "$run_id" "$score_step" "$batch_sz" "$type"
}

# =========================
# Mean score across runs at a given step
# type in {l2_error, grad_norm, forget}
# =========================
mean_score () {
  local exp_name="$1"
  local num_runs="$2"
  local score_step="$3"
  local type="$4"
  python scripts/get_mean_score.py "$ROOT" "$exp_name" "$num_runs" "$score_step" "$type"
}

# Robustly find mean score file for a given type+step
find_mean_score_path () {
  local exp_dir="$1"
  local type="$2"
  local step="$3"
  find "$exp_dir" -maxdepth 4 -type f -name "ckpt_${step}.npy" -path "*${type}*" | head -n 1 || true
}

# =========================
# Main loop
# =========================
for dataset in "${DATASETS[@]}"; do
  ntrain="$(num_train_examples "$dataset")"
  if [ -z "$ntrain" ]; then
    echo "[ERR] unknown dataset=$dataset for num_train_examples mapping."
    echo "      Please extend num_train_examples() in script."
    exit 1
  fi

  spe="$(steps_per_epoch "$ntrain" "$TRAIN_BATCH")"
  total_steps=$((TRAIN_EPOCHS * spe))

  # sanity: score batch must divide ntrain
  if [ $((ntrain % EL2N_SCORE_BATCH)) -ne 0 ]; then
    echo "[ERR] EL2N_SCORE_BATCH=$EL2N_SCORE_BATCH does not divide N_train=$ntrain"
    exit 1
  fi
  if [ $((ntrain % GRAND_SCORE_BATCH)) -ne 0 ]; then
    echo "[ERR] GRAND_SCORE_BATCH=$GRAND_SCORE_BATCH does not divide N_train=$ntrain"
    exit 1
  fi

  for base_seed in "${BASE_SEEDS[@]}"; do
    EXP_NAME="formal_${dataset}_seed${base_seed}_E${TRAIN_EPOCHS}_b${TRAIN_BATCH}"
    EXP_DIR="$EXP_ROOT/$EXP_NAME"
    mkdir -p "$EXP_DIR"

    echo "===================================================="
    echo "[START] dataset=$dataset N_train=$ntrain train_batch=$TRAIN_BATCH steps/epoch=$spe"
    echo "        base_seed=$base_seed N_RUNS=$N_RUNS TRAIN_EPOCHS=$TRAIN_EPOCHS total_steps=$total_steps"
    echo "        SCORE_EPOCHS=(${SCORE_EPOCHS[*]})"
    echo "===================================================="

    # 1) Train N_RUNS in parallel (up to 8 GPUs)
    pids=()
    for run_id in $(seq 0 $((N_RUNS-1))); do
      run_seed=$((base_seed + run_id * SEED_INCR))
      save_dir="$EXP_DIR/run_${run_id}"

      if [ -f "$save_dir/args.json" ]; then
        echo "[SKIP TRAIN] exists: $save_dir"
        continue
      fi

      (
        export CUDA_VISIBLE_DEVICES="$run_id"
        echo "[TRAIN] run=$run_id gpu=$CUDA_VISIBLE_DEVICES seed=$run_seed"
        train_one "$dataset" "$run_seed" "$save_dir" "$spe" "$total_steps"
      ) &
      pids+=($!)
    done

    if [ "${#pids[@]}" -gt 0 ]; then
      echo "[WAIT] training jobs: ${#pids[@]}"
      wait "${pids[@]}"
    fi

    # 2) For each score epoch -> step, compute per-run EL2N & GraNd scores (parallel by run)
    for e in "${SCORE_EPOCHS[@]}"; do
      score_step=$((e * spe))

      echo "------------------------------"
      echo "[SCORE] epoch=$e score_step=$score_step"
      echo "------------------------------"

      # EL2N
      pids=()
      for run_id in $(seq 0 $((N_RUNS-1))); do
        (
          export CUDA_VISIBLE_DEVICES="$run_id"
          echo "[EL2N] run=$run_id gpu=$CUDA_VISIBLE_DEVICES batch=$EL2N_SCORE_BATCH"
          score_one "$EXP_NAME" "$run_id" "$score_step" "$EL2N_SCORE_BATCH" "l2_error"
        ) &
        pids+=($!)
      done
      wait "${pids[@]}"

      # GraNd
      pids=()
      for run_id in $(seq 0 $((N_RUNS-1))); do
        (
          export CUDA_VISIBLE_DEVICES="$run_id"
          echo "[GraNd] run=$run_id gpu=$CUDA_VISIBLE_DEVICES batch=$GRAND_SCORE_BATCH"
          score_one "$EXP_NAME" "$run_id" "$score_step" "$GRAND_SCORE_BATCH" "grad_norm"
        ) &
        pids+=($!)
      done
      wait "${pids[@]}"

      # 3) Mean EL2N & GraNd at this step (across runs)
      echo "[MEAN] l2_error at step=$score_step"
      mean_score "$EXP_NAME" "$N_RUNS" "$score_step" "l2_error"
      echo "[MEAN] grad_norm at step=$score_step"
      mean_score "$EXP_NAME" "$N_RUNS" "$score_step" "grad_norm"

      # 4) Generate masks from mean EL2N/GraNd scores for each keep ratio
      for METHOD in EL2N GraNd; do
        if [ "$METHOD" = "EL2N" ]; then
          type="l2_error"
        else
          type="grad_norm"
        fi
        mean_path="$(find_mean_score_path "$EXP_DIR" "$type" "$score_step")"
        if [ -z "$mean_path" ]; then
          echo "[ERR] cannot find mean score for $METHOD (type=$type) at step=$score_step under $EXP_DIR"
          exit 1
        fi

        for kr in "${KEEP_RATIOS[@]}"; do
          out_dir="$MASK_ROOT/${dataset}/${METHOD}/seed${base_seed}/E${TRAIN_EPOCHS}/scoreE${e}"
          name="${METHOD}_${dataset}_seed${base_seed}_trainE${TRAIN_EPOCHS}_scoreE${e}_keep${kr}"
          python "$ROOT/tools/make_mask_from_scores.py" \
            --scores "$mean_path" \
            --keep_ratio "$kr" \
            --out_dir "$out_dir" \
            --name "$name" \
            --keep_high
        done
        echo "[DONE] $METHOD masks -> $MASK_ROOT/${dataset}/${METHOD}/seed${base_seed}/E${TRAIN_EPOCHS}/scoreE${e}"
      done
    done

    # 5) Forgetting mean score: should be computed from full training (200 epochs) tracking.
    # Use last step = TRAIN_EPOCHS*steps_per_epoch.
    final_step="$total_steps"
    echo "------------------------------"
    echo "[FORGET] mean forgetting at final_step=$final_step (trainE=$TRAIN_EPOCHS)"
    echo "------------------------------"
    mean_score "$EXP_NAME" "$N_RUNS" "$final_step" "forget"

    # Generate masks from mean forgetting score (one set; no scoreE scan; forgetting is defined over full training)
    mean_forget_path="$(find_mean_score_path "$EXP_DIR" "forget" "$final_step")"
    if [ -z "$mean_forget_path" ]; then
      echo "[ERR] cannot find mean forget score at step=$final_step under $EXP_DIR"
      exit 1
    fi

    for kr in "${KEEP_RATIOS[@]}"; do
      out_dir="$MASK_ROOT/${dataset}/Forget/seed${base_seed}/E${TRAIN_EPOCHS}/final"
      name="Forget_${dataset}_seed${base_seed}_trainE${TRAIN_EPOCHS}_keep${kr}"
      python "$ROOT/tools/make_mask_from_scores.py" \
        --scores "$mean_forget_path" \
        --keep_ratio "$kr" \
        --out_dir "$out_dir" \
        --name "$name" \
        --keep_high
    done
    echo "[DONE] Forget masks -> $MASK_ROOT/${dataset}/Forget/seed${base_seed}/E${TRAIN_EPOCHS}/final"

    echo "[FINISH] dataset=$dataset base_seed=$base_seed"
  done
done

echo "ALL DONE. masks root: $MASK_ROOT"
