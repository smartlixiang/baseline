 #!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ======= 你常改的配置 =======
DATASETS=("cifar10" "cifar100")
SEEDS=(22 42 96)
KEEP_RATIOS=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2)

MODEL="resnet18_lowres"
LR=0.1
BETA=0.9
WEIGHT_DECAY=0.0005
NESTEROV=true

# 论文里常用在 epoch 10 计算 EL2N（示例图/附录里频繁出现）:contentReference[oaicite:2]{index=2}
SCORE_EPOCH=10

# 近似论文的“多次平均”（论文写的是 10 次平均）:contentReference[oaicite:3]{index=3}
# 你现在如果只想每 seed 跑一次：N_AVG=1
# 你有 8 张卡想更接近论文：N_AVG=8（会并行跑 8 次再对 scores 求均值）
N_AVG="${N_AVG:-1}"

# score 计算 batch：EL2N 通常可以较大；要求整除 50000（CIFAR10/100 训练集）
SCORE_BATCH="${SCORE_BATCH:-100}"

# 训练 batch：你当前机器稳定跑通的是 32（官方默认 128，但你显存/旧版XLA需降）:contentReference[oaicite:4]{index=4}
TRAIN_BATCH="${TRAIN_BATCH:-32}"
TEST_BATCH="${TEST_BATCH:-128}"

# ======= 稳定性环境变量（你已经验证过能救活） =======
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0"

# 降低刷屏（可选）
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore::FutureWarning,ignore::DeprecationWarning"

# steps/epoch：CIFAR10/100 的训练集都是 50000，官方脚本里 EP_STEPS=390（50000/128≈390）；
# 这里我们直接用 390 来算 score_step=epoch*390，保持与仓库脚本一致。
EP_STEPS=390
SCORE_STEP=$((EP_STEPS * SCORE_EPOCH))

mkdir -p "$ROOT/exps" "$ROOT/masks"

# ---- helper: average multiple score runs ----
avg_scores_py=$(cat <<'PY'
import argparse, numpy as np
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--inputs", nargs="+", required=True)
args=ap.parse_args()
arrs=[np.load(p) for p in args.inputs]
m=np.mean(np.stack(arrs,0),0)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.save(args.out, m)
print("[OK] averaged", len(arrs), "->", args.out, "shape", m.shape)
PY
)

for dataset in "${DATASETS[@]}"; do
  for base_seed in "${SEEDS[@]}"; do

    EXP="el2n_${dataset}_seed${base_seed}_epoch${SCORE_EPOCH}"
    RUN_DIR="$ROOT/exps/${EXP}"
    mkdir -p "$RUN_DIR"

    echo "=============================="
    echo "[EL2N] dataset=$dataset seed=$base_seed N_AVG=$N_AVG keep=${KEEP_RATIOS[*]}"
    echo "=============================="

    # 1) 训练+算分（可能多次）
    SCORE_FILES=()
    for k in $(seq 0 $((N_AVG-1))); do
      # 派生 seed（保证同一 base_seed 下可复现、不同 k 不同）
      run_seed=$((base_seed + k * 1000003))
      run_id=$k

      one_exp="${EXP}/avg${N_AVG}/run_${run_id}"
      save_dir="$ROOT/exps/${one_exp}"

      # 如果已存在 score 则跳过
      score_path="$save_dir/error_l2_norm_scores/ckpt_${SCORE_STEP}.npy"
      if [ -f "$score_path" ]; then
        echo "[SKIP] score exists: $score_path"
        SCORE_FILES+=("$score_path")
        continue
      fi

      # 指定 GPU（k=0..7 映射到 8 张卡；若 N_AVG=1 就用 0）
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

      echo "[2] Compute EL2N score (run=$run_id) batch=$SCORE_BATCH"
      python scripts/get_run_score.py "$ROOT" "${EXP}/avg${N_AVG}" "$run_id" "$SCORE_STEP" "$SCORE_BATCH" l2_error

      if [ ! -f "$score_path" ]; then
        echo "[ERR] score not found: $score_path"
        exit 1
      fi
      SCORE_FILES+=("$score_path")
    done

    # 2) 平均（如果 N_AVG>1）
    if [ "$N_AVG" -eq 1 ]; then
      AVG_SCORE="${SCORE_FILES[0]}"
    else
      AVG_SCORE="$RUN_DIR/avg${N_AVG}/avg_scores/ckpt_${SCORE_STEP}.npy"
      if [ ! -f "$AVG_SCORE" ]; then
        python - <<PY
$avg_scores_py
PY
        python - <<PY
import numpy as np
inputs=${SCORE_FILES[@]}
PY
      fi
      python - <<PY
import numpy as np
from pathlib import Path
outs=r"$AVG_SCORE"
inputs=${SCORE_FILES[@]}
PY
      # 上面两段为了避免 bash/py 混入复杂转义，下面用更稳的方式再做一次：
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

    # 3) 生成 8 个 keep ratio 的 mask（只从 AVG_SCORE 派生，不再训练）
    for kr in "${KEEP_RATIOS[@]}"; do
      tag="EL2N_${dataset}_seed${base_seed}_epoch${SCORE_EPOCH}_keep${kr}"
      out_dir="$ROOT/masks/${dataset}/EL2N/seed${base_seed}/epoch${SCORE_EPOCH}"
      python tools/make_mask_from_scores.py \
        --scores "$AVG_SCORE" \
        --keep_ratio "$kr" \
        --out_dir "$out_dir" \
        --name "$tag" \
        --keep_high
    done

    echo "[DONE] masks in: $ROOT/masks/${dataset}/EL2N/seed${base_seed}/epoch${SCORE_EPOCH}"
  done
done
