#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# =========================
# Experiment grid
# =========================
DATASETS=("cifar10" "cifar100")
BASE_SEEDS=(22 42 96)
KEEP_RATIOS=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2)

# =========================
# Core settings
# =========================
MODEL="resnet18_lowres"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-200}"
SCORE_EPOCHS=(${SCORE_EPOCHS:-20})

# Paper uses 10 independent runs; here default k=8 for 8 GPUs.
K_RUNS="${K_RUNS:-8}"
MAX_GPUS=8
if [ "$K_RUNS" -lt 1 ] || [ "$K_RUNS" -gt "$MAX_GPUS" ]; then
  echo "[ERR] K_RUNS must be in [1, ${MAX_GPUS}], got ${K_RUNS}"
  exit 1
fi

# Host RAM can be the bottleneck before VRAM when each worker loads JAX/TFDS.
# Use train waves to avoid Linux OOM-killer taking down random workers.
TRAIN_PARALLEL="${TRAIN_PARALLEL:-4}"
if [ "$TRAIN_PARALLEL" -lt 1 ] || [ "$TRAIN_PARALLEL" -gt "$K_RUNS" ]; then
  echo "[ERR] TRAIN_PARALLEL must be in [1, ${K_RUNS}], got ${TRAIN_PARALLEL}"
  exit 1
fi

# NOTE: Batch sizes are intentionally conservative due to 2080Ti VRAM limits;
#       they are expected to differ from paper defaults to avoid OOM.
TRAIN_BATCH="${TRAIN_BATCH:-32}"
TEST_BATCH="${TEST_BATCH:-128}"
EL2N_SCORE_BATCH="${EL2N_SCORE_BATCH:-100}"
GRAND_SCORE_BATCH="${GRAND_SCORE_BATCH:-5}"

LR="${LR:-0.1}"
BETA="${BETA:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"

# JAX/XLA runtime knobs
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore::FutureWarning,ignore::DeprecationWarning"

# Use official exps/ for compatibility with scripts/get_run_score.py + get_mean_score.py.
EXP_ROOT="$ROOT/exps"
MASK_ROOT="$ROOT/mask"
LOG_ROOT="$ROOT/logs_formal_masks"
mkdir -p "$EXP_ROOT" "$MASK_ROOT" "$LOG_ROOT"

if [ ! -f "$ROOT/tools/make_mask_from_scores.py" ]; then
  echo "[ERR] missing tools/make_mask_from_scores.py"
  exit 1
fi

num_train_examples () {
  case "$1" in
    cifar10) echo 50000 ;;
    cifar100) echo 50000 ;;
    cinic10) echo 90000 ;;
    *) echo "" ;;
  esac
}

steps_per_epoch () {
  local n="$1"
  local b="$2"
  python - <<PY
import math
print((int("$n") + int("$b") - 1)//int("$b"))
PY
}

train_one () {
  local dataset="$1"
  local run_seed="$2"
  local save_dir="$3"
  local spe="$4"
  local num_steps="$5"

  python - <<PY
from data_diet.train import train
from types import SimpleNamespace

ROOT=r"$ROOT"
args=SimpleNamespace()
args.data_dir = ROOT + "/data"
args.dataset = r"$dataset"

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
args.decay_steps=[60*int($spe), 120*int($spe), 160*int($spe)]

args.num_steps=int($num_steps)
args.train_seed=int($run_seed)
args.train_batch_size=int($TRAIN_BATCH)
args.test_batch_size=int($TEST_BATCH)
args.augment=True

args.track_forgetting=True

args.save_dir=r"$save_dir"
args.log_steps=int($spe)
args.early_step=0
args.early_save_steps=None
args.save_steps=int($spe)

train(args)
PY
}

find_mean_score_path () {
  local exp_dir="$1"
  local type="$2"
  local step="$3"
  find "$exp_dir" -maxdepth 4 -type f -name "ckpt_${step}.npy" -path "*${type}*" | head -n 1 || true
}

wait_for_pids () {
  local -n _pids=$1
  local fail=0
  for pid in "${_pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if [ "$fail" -ne 0 ]; then
    echo "[ERR] at least one parallel subprocess failed"
    exit 1
  fi
}

echo "[INFO] Multi-GPU progress note: to avoid tqdm bars being garbled by parallel workers,"
echo "       worker stdout/stderr is redirected to per-run log files under: $LOG_ROOT"
echo "[INFO] train worker parallelism per wave: TRAIN_PARALLEL=$TRAIN_PARALLEL (K_RUNS=$K_RUNS)"

for dataset in "${DATASETS[@]}"; do
  ntrain="$(num_train_examples "$dataset")"
  if [ -z "$ntrain" ]; then
    echo "[ERR] unknown dataset=$dataset"
    exit 1
  fi

  spe="$(steps_per_epoch "$ntrain" "$TRAIN_BATCH")"
  total_steps=$((TRAIN_EPOCHS * spe))

  if [ $((ntrain % EL2N_SCORE_BATCH)) -ne 0 ]; then
    echo "[ERR] EL2N_SCORE_BATCH=$EL2N_SCORE_BATCH does not divide N_train=$ntrain"
    exit 1
  fi
  if [ $((ntrain % GRAND_SCORE_BATCH)) -ne 0 ]; then
    echo "[ERR] GRAND_SCORE_BATCH=$GRAND_SCORE_BATCH does not divide N_train=$ntrain"
    exit 1
  fi

  for base_seed in "${BASE_SEEDS[@]}"; do
    exp_name="formal_${dataset}_seed${base_seed}_E${TRAIN_EPOCHS}_b${TRAIN_BATCH}_k${K_RUNS}"
    exp_dir="$EXP_ROOT/$exp_name"
    mkdir -p "$exp_dir"

    echo "===================================================="
    echo "[START] dataset=$dataset base_seed=$base_seed k=$K_RUNS"
    echo "        train_batch=$TRAIN_BATCH, steps/epoch=$spe, total_steps=$total_steps"
    echo "        score_epochs=(${SCORE_EPOCHS[*]})"
    echo "        run_seed formula: run_seed = base_seed * (run_id + 1)"
    echo "===================================================="

    pids=()
    launched_in_wave=0
    for run_id in $(seq 0 $((K_RUNS-1))); do
      run_seed=$((base_seed * (run_id + 1)))
      run_dir="$exp_dir/run_${run_id}"
      log_file="$LOG_ROOT/${exp_name}.run_${run_id}.train.log"

      if [ -f "$run_dir/args.json" ]; then
        echo "[SKIP TRAIN] run=$run_id exists: $run_dir"
        continue
      fi

      (
        export CUDA_VISIBLE_DEVICES="$run_id"
        train_one "$dataset" "$run_seed" "$run_dir" "$spe" "$total_steps"
      ) >"$log_file" 2>&1 &
      pids+=($!)
      launched_in_wave=$((launched_in_wave + 1))
      echo "[TRAIN] launched run=$run_id gpu=$run_id seed=$run_seed log=$log_file"

      if [ "$launched_in_wave" -ge "$TRAIN_PARALLEL" ]; then
        wait_for_pids pids
        echo "[TRAIN] finished one wave of $launched_in_wave workers"
        pids=()
        launched_in_wave=0
      fi
    done
    if [ "${#pids[@]}" -gt 0 ]; then
      wait_for_pids pids
      echo "[TRAIN] finished final wave of $launched_in_wave workers"
    fi
    echo "[TRAIN] all training workers finished"

    for e in "${SCORE_EPOCHS[@]}"; do
      score_step=$((e * spe))
      echo "[SCORE] epoch=$e step=$score_step"

      # Parallel over runs for EL2N.
      pids=()
      for run_id in $(seq 0 $((K_RUNS-1))); do
        log_file="$LOG_ROOT/${exp_name}.run_${run_id}.el2n.step_${score_step}.log"
        (
          export CUDA_VISIBLE_DEVICES="$run_id"
          python "$ROOT/scripts/get_run_score.py" "$ROOT" "$exp_name" "$run_id" "$score_step" "$EL2N_SCORE_BATCH" "l2_error"
        ) >"$log_file" 2>&1 &
        pids+=($!)
      done
      wait_for_pids pids
      echo "[SCORE] EL2N per-run finished"

      # Parallel over runs for GraNd.
      pids=()
      for run_id in $(seq 0 $((K_RUNS-1))); do
        log_file="$LOG_ROOT/${exp_name}.run_${run_id}.grand.step_${score_step}.log"
        (
          export CUDA_VISIBLE_DEVICES="$run_id"
          python "$ROOT/scripts/get_run_score.py" "$ROOT" "$exp_name" "$run_id" "$score_step" "$GRAND_SCORE_BATCH" "grad_norm"
        ) >"$log_file" 2>&1 &
        pids+=($!)
      done
      wait_for_pids pids
      echo "[SCORE] GraNd per-run finished"

      # Mean aggregation is lightweight; run serially for determinism.
      python "$ROOT/scripts/get_mean_score.py" "$ROOT" "$exp_name" "$K_RUNS" "$score_step" "l2_error"
      python "$ROOT/scripts/get_mean_score.py" "$ROOT" "$exp_name" "$K_RUNS" "$score_step" "grad_norm"

      # Save masks to mask/[dataset]/[seed]/...
      seed_mask_root="$MASK_ROOT/$dataset/$base_seed"
      mkdir -p "$seed_mask_root"

      el2n_mean="$(find_mean_score_path "$exp_dir" "error_l2_norm_scores" "$score_step")"
      grand_mean="$(find_mean_score_path "$exp_dir" "grad_norm_scores" "$score_step")"

      if [ -z "$el2n_mean" ] || [ -z "$grand_mean" ]; then
        echo "[ERR] missing mean score file at step=$score_step under $exp_dir"
        exit 1
      fi

      python "$ROOT/tools/make_mask_from_scores.py" \
        --scores "$el2n_mean" \
        --keep_ratios "${KEEP_RATIOS[@]}" \
        --out_dir "$seed_mask_root/EL2N/scoreE${e}" \
        --name_prefix "EL2N_${dataset}_seed${base_seed}_scoreE${e}" \
        --keep_high

      python "$ROOT/tools/make_mask_from_scores.py" \
        --scores "$grand_mean" \
        --keep_ratios "${KEEP_RATIOS[@]}" \
        --out_dir "$seed_mask_root/GraNd/scoreE${e}" \
        --name_prefix "GraNd_${dataset}_seed${base_seed}_scoreE${e}" \
        --keep_high
    done

    # Forgetting scores are generated from full-training trajectories and averaged across runs.
    final_step="$total_steps"
    python "$ROOT/scripts/get_mean_score.py" "$ROOT" "$exp_name" "$K_RUNS" "$final_step" "forget"
    forget_mean="$(find_mean_score_path "$exp_dir" "forget_scores" "$final_step")"
    if [ -z "$forget_mean" ]; then
      echo "[ERR] missing mean forgetting score at step=$final_step under $exp_dir"
      exit 1
    fi

    seed_mask_root="$MASK_ROOT/$dataset/$base_seed"
    python "$ROOT/tools/make_mask_from_scores.py" \
      --scores "$forget_mean" \
      --keep_ratios "${KEEP_RATIOS[@]}" \
      --out_dir "$seed_mask_root/forgetting/final" \
      --name_prefix "forgetting_${dataset}_seed${base_seed}_final" \
      --keep_high

    echo "[DONE] dataset=$dataset base_seed=$base_seed masks at $seed_mask_root"
  done
done

echo "[ALL DONE] mask root: $MASK_ROOT"
