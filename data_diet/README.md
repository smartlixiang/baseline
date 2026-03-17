# Data Diet (PyTorch)

PyTorch-only implementation of EL2N / GraNd / Forgetting scoring and mask generation.

## Setup

Core dependencies: `torch`, `torchvision`, `numpy`, `tqdm`, `pillow`.

```bash
mkdir -p data exps
```

Datasets are read from `<ROOT>/data`.

## Reproducibility

Training/scoring set unified seeds for `python`, `numpy`, `torch` (CPU/GPU). cuDNN is set to deterministic mode (`torch.backends.cudnn.deterministic=True`, `benchmark=False`). This improves reproducibility but can reduce speed.

## Main workflow

### 1) Train one run

```bash
python scripts/run_full_data.py <ROOT> <EXP_NAME> <RUN_ID> [DATASET]
```

- Supported datasets: `cifar10`, `cifar100`, `cinic10`, `tiny-imagenet`.
- Saves checkpoints to `<ROOT>/exps/<EXP_NAME>/run_<RUN_ID>/ckpts/checkpoint_<STEP>.pt`.
- Forgetting scores are written during checkpointing to `forget_scores/ckpt_<STEP>.npy`.

### 2) Compute per-run EL2N / GraNd at a checkpoint

```bash
python scripts/get_run_score.py <ROOT> <EXP_NAME> <RUN_ID> <STEP> <BATCH_SZ> l2_error
python scripts/get_run_score.py <ROOT> <EXP_NAME> <RUN_ID> <STEP> <BATCH_SZ> grad_norm
```

- EL2N: `error_l2_norm_scores/ckpt_<STEP>.npy`.
- GraNd: `grad_norm_scores/ckpt_<STEP>.npy`.

### 3) Aggregate mean score across runs

```bash
python scripts/get_mean_score.py <ROOT> <EXP_NAME> <N_RUNS> <STEP> l2_error
python scripts/get_mean_score.py <ROOT> <EXP_NAME> <N_RUNS> <STEP> grad_norm
python scripts/get_mean_score.py <ROOT> <EXP_NAME> <N_RUNS> <STEP> forget
```

### 4) Build masks from mean scores

```bash
python tools/make_mask_from_scores.py \
  --scores <MEAN_SCORE.npy> \
  --keep_ratios 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 \
  --out_dir <OUT_DIR> \
  --name_prefix <PREFIX> \
  --keep_high
```

### End-to-end launcher

```bash
bash run_formal_masks.sh
```

This script trains K runs, computes EL2N/GraNd at selected epochs, averages scores, averages forgetting at final step, and writes masks under `mask/`.
