#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

DATASET="tiny-imagenet"
MODEL="resnet34_lowres"
BASE_SEEDS=(22 42 96)
K_RUNS="${K_RUNS:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-90}"
SCORE_EPOCHS=(${SCORE_EPOCHS:-10})
KEEP_RATIOS=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2)

TRAIN_BATCH="${TRAIN_BATCH:-64}"
TEST_BATCH="${TEST_BATCH:-256}"
EL2N_SCORE_BATCH="${EL2N_SCORE_BATCH:-100}"
GRAND_SCORE_BATCH="${GRAND_SCORE_BATCH:-20}"

if [ -n "${GPU_IDS:-}" ]; then
  # shellcheck disable=SC2206
  GPU_ID_LIST=(${GPU_IDS})
else
  GPU_ID_LIST=(0)
fi
GPU_COUNT="${#GPU_ID_LIST[@]}"
TRAIN_PARALLEL="${TRAIN_PARALLEL:-$GPU_COUNT}"
SCORE_PARALLEL="${SCORE_PARALLEL:-$GPU_COUNT}"

EXP_ROOT="$ROOT/exps"
MASK_ROOT="$ROOT/mask"
LOG_ROOT="$ROOT/logs_tiny_imagenet_masks"
mkdir -p "$EXP_ROOT" "$MASK_ROOT" "$LOG_ROOT"

count_tiny_train_examples () {
  python - <<PY
import os
root = r"$ROOT/data/tiny-imagenet-200/train"
total = 0
for cls_name in os.listdir(root):
  img_dir = os.path.join(root, cls_name, 'images')
  if os.path.isdir(img_dir):
    total += sum(1 for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)))
print(total)
PY
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
    echo "[ERR] parallel subprocess failed"
    exit 1
  fi
}

score_exists () {
  local run_dir="$1"
  local step="$2"
  local type="$3"
  local path_name=""
  case "$type" in
    l2_error) path_name="error_l2_norm_scores" ;;
    grad_norm) path_name="grad_norm_scores" ;;
    *) echo "[ERR] unsupported score type: $type"; exit 1 ;;
  esac
  [ -f "$run_dir/$path_name/ckpt_${step}.npy" ]
}

ntrain="$(count_tiny_train_examples)"
steps_per_epoch=$((ntrain / TRAIN_BATCH))
total_steps=$((TRAIN_EPOCHS * steps_per_epoch))

echo "[INFO] dataset=$DATASET"
echo "[INFO] steps_per_epoch=$steps_per_epoch"
echo "[INFO] total_steps=$total_steps"

auto_score_batch () {
  local preferred=$1
  local b=$preferred
  while [ "$b" -gt 1 ]; do
    if [ $((ntrain % b)) -eq 0 ]; then
      echo "$b"
      return
    fi
    b=$((b - 1))
  done
  echo "1"
}

EL2N_SCORE_BATCH="$(auto_score_batch "$EL2N_SCORE_BATCH")"
GRAND_SCORE_BATCH="$(auto_score_batch "$GRAND_SCORE_BATCH")"
echo "[INFO] EL2N score batch=$EL2N_SCORE_BATCH"
echo "[INFO] GraNd score batch=$GRAND_SCORE_BATCH"

for base_seed in "${BASE_SEEDS[@]}"; do
  exp_name="formal_${DATASET}_${MODEL}_seed${base_seed}_E${TRAIN_EPOCHS}_b${TRAIN_BATCH}_k${K_RUNS}"
  exp_dir="$EXP_ROOT/$exp_name"
  mkdir -p "$exp_dir"

  echo "===================================================="
  echo "[START] base_seed=$base_seed exp=$exp_name"
  echo "[INFO] output_dir=$exp_dir"
  for e in "${SCORE_EPOCHS[@]}"; do
    echo "[INFO] score_epoch=$e -> score_step=$((e * steps_per_epoch))"
  done
  echo "===================================================="

  pids=()
  launched=0
  for run_id in $(seq 0 $((K_RUNS-1))); do
    run_dir="$exp_dir/run_${run_id}"
    log_file="$LOG_ROOT/${exp_name}.run_${run_id}.train.log"
    gpu_slot=$((run_id % GPU_COUNT))
    gpu_id="${GPU_ID_LIST[$gpu_slot]}"

    if [ -f "$run_dir/args.json" ]; then
      echo "[SKIP TRAIN] run=${run_id} exists"
      continue
    fi

    (
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      python "$ROOT/scripts/run_full_data_tiny_imagenet.py" "$ROOT" "$exp_name" "$run_id" "$base_seed"
    ) >"$log_file" 2>&1 &
    pids+=($!)
    launched=$((launched + 1))
    echo "[TRAIN] launched run=${run_id} gpu=${gpu_id}"

    if [ "$launched" -ge "$TRAIN_PARALLEL" ]; then
      wait_for_pids pids
      pids=()
      launched=0
    fi
  done
  if [ "${#pids[@]}" -gt 0 ]; then
    wait_for_pids pids
  fi

  for e in "${SCORE_EPOCHS[@]}"; do
    score_step=$((e * steps_per_epoch))

    pids=()
    launched=0
    for run_id in $(seq 0 $((K_RUNS-1))); do
      run_dir="$exp_dir/run_${run_id}"
      log_file="$LOG_ROOT/${exp_name}.run_${run_id}.el2n.step_${score_step}.log"
      gpu_slot=$((run_id % GPU_COUNT))
      gpu_id="${GPU_ID_LIST[$gpu_slot]}"
      if score_exists "$run_dir" "$score_step" "l2_error"; then
        echo "[SKIP SCORE] EL2N run=${run_id} step=${score_step} exists"
        continue
      fi
      (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        python "$ROOT/scripts/get_run_score.py" "$ROOT" "$exp_name" "$run_id" "$score_step" "$EL2N_SCORE_BATCH" "l2_error"
      ) >"$log_file" 2>&1 &
      pids+=($!)
      launched=$((launched + 1))
      if [ "$launched" -ge "$SCORE_PARALLEL" ]; then
        wait_for_pids pids
        pids=()
        launched=0
      fi
    done
    if [ "${#pids[@]}" -gt 0 ]; then
      wait_for_pids pids
    fi

    pids=()
    launched=0
    for run_id in $(seq 0 $((K_RUNS-1))); do
      run_dir="$exp_dir/run_${run_id}"
      log_file="$LOG_ROOT/${exp_name}.run_${run_id}.grand.step_${score_step}.log"
      gpu_slot=$((run_id % GPU_COUNT))
      gpu_id="${GPU_ID_LIST[$gpu_slot]}"
      if score_exists "$run_dir" "$score_step" "grad_norm"; then
        echo "[SKIP SCORE] GraNd run=${run_id} step=${score_step} exists"
        continue
      fi
      (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        python "$ROOT/scripts/get_run_score.py" "$ROOT" "$exp_name" "$run_id" "$score_step" "$GRAND_SCORE_BATCH" "grad_norm"
      ) >"$log_file" 2>&1 &
      pids+=($!)
      launched=$((launched + 1))
      if [ "$launched" -ge "$SCORE_PARALLEL" ]; then
        wait_for_pids pids
        pids=()
        launched=0
      fi
    done
    if [ "${#pids[@]}" -gt 0 ]; then
      wait_for_pids pids
    fi

    mean_el2n="$exp_dir/error_l2_norm_scores/ckpt_${score_step}.npy"
    mean_grand="$exp_dir/grad_norm_scores/ckpt_${score_step}.npy"

    if [ ! -f "$mean_el2n" ]; then
      python "$ROOT/scripts/get_mean_score.py" "$ROOT" "$exp_name" "$K_RUNS" "$score_step" "l2_error"
    fi
    if [ ! -f "$mean_grand" ]; then
      python "$ROOT/scripts/get_mean_score.py" "$ROOT" "$exp_name" "$K_RUNS" "$score_step" "grad_norm"
    fi

    seed_mask_root="$MASK_ROOT/$DATASET/$base_seed"
    mkdir -p "$seed_mask_root"

    if [ -f "$seed_mask_root/EL2N/scoreE${e}/EL2N_${DATASET}_seed${base_seed}_scoreE${e}_keep90.idx.npy" ]; then
      echo "[SKIP MASK] EL2N scoreE${e} already exists"
    else
      python "$ROOT/tools/make_mask_from_scores.py" \
        --scores "$mean_el2n" \
        --keep_ratios "${KEEP_RATIOS[@]}" \
        --out_dir "$seed_mask_root/EL2N/scoreE${e}" \
        --name_prefix "EL2N_${DATASET}_seed${base_seed}_scoreE${e}" \
        --keep_high
    fi

    if [ -f "$seed_mask_root/GraNd/scoreE${e}/GraNd_${DATASET}_seed${base_seed}_scoreE${e}_keep90.idx.npy" ]; then
      echo "[SKIP MASK] GraNd scoreE${e} already exists"
    else
      python "$ROOT/tools/make_mask_from_scores.py" \
        --scores "$mean_grand" \
        --keep_ratios "${KEEP_RATIOS[@]}" \
        --out_dir "$seed_mask_root/GraNd/scoreE${e}" \
        --name_prefix "GraNd_${DATASET}_seed${base_seed}_scoreE${e}" \
        --keep_high
    fi
  done

  mean_forget="$exp_dir/forget_scores/ckpt_${total_steps}.npy"
  if [ ! -f "$mean_forget" ]; then
    python "$ROOT/scripts/get_mean_score.py" "$ROOT" "$exp_name" "$K_RUNS" "$total_steps" "forget"
  fi

  seed_mask_root="$MASK_ROOT/$DATASET/$base_seed"
  if [ -f "$seed_mask_root/forgetting/final/forgetting_${DATASET}_seed${base_seed}_final_keep90.idx.npy" ]; then
    echo "[SKIP MASK] forgetting final already exists"
  else
    python "$ROOT/tools/make_mask_from_scores.py" \
      --scores "$mean_forget" \
      --keep_ratios "${KEEP_RATIOS[@]}" \
      --out_dir "$seed_mask_root/forgetting/final" \
      --name_prefix "forgetting_${DATASET}_seed${base_seed}_final" \
      --keep_high
  fi

  echo "[DONE] base_seed=$base_seed masks at $seed_mask_root"
done

echo "[ALL DONE] tiny-imagenet masks at $MASK_ROOT/$DATASET"
