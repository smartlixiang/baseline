#!/usr/bin/env bash
set -e

# 说明：
# 1. 请在 MoSo/ 目录下执行本脚本
# 2. 默认数据目录为 ./data，mask 输出目录为 ./mask
# 3. tmux 会话：
#    222 -> GPU 2 -> cifar10
#    333 -> GPU 3 -> cifar100
#    444 -> GPU 4 -> tiny
# 4. 每个数据集只训练一次、打分一次
# 5. 然后将同一套 score 导出到 seed=22/42/96 三个路径下
# 6. keep_ratio 取值：20 30 40 50 60 70 80 90

KEEP_RATIOS=(20 30 40 50 60 70 80 90)
SEEDS=(22 42 96)

# ===== 你需要按自己的 conda 安装位置修改这里 =====
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
  CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# 如果还是不存在，请手动改成你机器上的 conda.sh 路径
if [ ! -f "$CONDA_SH" ]; then
  echo "Cannot find conda.sh. Please edit CONDA_SH in this script."
  exit 1
fi

WORKDIR="$(pwd)"

# 跑单个数据集的完整流程
# 参数：
#   $1 = dataset 名称: cifar10 / cifar100 / tiny
#   $2 = gpu id
#   $3 = tmux session name
run_dataset_job() {
  local DATASET="$1"
  local GPU="$2"
  local SESSION="$3"
  local EXP_PATH="runs/${DATASET}"

  local CMD="
set -e
cd \"$WORKDIR\"
source \"$CONDA_SH\"
conda activate shampoo

echo \"[INFO] session=${SESSION}, gpu=${GPU}, dataset=${DATASET}\"
echo \"[INFO] working dir: \$(pwd)\"
echo \"[INFO] start time: \$(date)\"

# tiny-imagenet 先做一次 val 目录整理
if [ \"$DATASET\" = \"tiny\" ]; then
  echo \"[INFO] preparing tiny-imagenet val directory...\"
  CUDA_VISIBLE_DEVICES=${GPU} python prepare_tiny_imagenet.py --data_root ./data
fi

echo \"[INFO] surrogate training: ${DATASET}\"
CUDA_VISIBLE_DEVICES=${GPU} python surrogate_training.py \\
  --dataset ${DATASET} \\
  --model resnet50 \\
  --path ${EXP_PATH} \\
  --data_root ./data \\
  --maxepoch 50 \\
  --num_trails 8 \\
  --trainaug 0

echo \"[INFO] scoring: ${DATASET}\"
CUDA_VISIBLE_DEVICES=${GPU} python scoring.py \\
  --dataset ${DATASET} \\
  --model resnet50 \\
  --path ${EXP_PATH} \\
  --data_root ./data \\
  --maxepoch 50 \\
  --num_trails 8 \\
  --samples 10 \\
  --trainaug 0

echo \"[INFO] exporting masks: ${DATASET}\"
for SEED in ${SEEDS[*]}; do
  for KR in ${KEEP_RATIOS[*]}; do
    echo \"[INFO] export mask dataset=${DATASET} seed=\${SEED} keep_ratio=\${KR}\"
    python export_mask.py \\
      --dataset ${DATASET} \\
      --seed \${SEED} \\
      --keep_ratio \${KR} \\
      --path ${EXP_PATH} \\
      --data_root ./data \\
      --mask_root ./mask
  done
done

echo \"[INFO] finished dataset=${DATASET} at \$(date)\"
"

  tmux new-session -d -s "$SESSION"
  tmux send-keys -t "$SESSION" "$CMD" C-m
}

# 创建三个 tmux 会话并分别跑三个数据集
run_dataset_job cifar10 2 222
run_dataset_job cifar100 3 333
run_dataset_job tiny 4 444

echo "Started tmux sessions:"
echo "  session 222 -> GPU 2 -> cifar10"
echo "  session 333 -> GPU 3 -> cifar100"
echo "  session 444 -> GPU 4 -> tiny"
echo
echo "Useful commands:"
echo "  tmux attach -t 222"
echo "  tmux attach -t 333"
echo "  tmux attach -t 444"
echo
echo "Mask outputs will be saved under:"
echo "  ./mask/cifar10/{22,42,96}/mask_{20,30,40,50,60,70,80,90}.npz"
echo "  ./mask/cifar100/{22,42,96}/mask_{20,30,40,50,60,70,80,90}.npz"
echo "  ./mask/tiny/{22,42,96}/mask_{20,30,40,50,60,70,80,90}.npz"