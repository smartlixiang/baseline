# Data Diet (PyTorch)

PyTorch-only implementation of EL2N / GraNd / Forgetting scoring and mask generation.

## Tiny-ImageNet two-stage workflow

### Stage 1: run training for **one** base seed (3 runs)

```bash
cd data_diet
bash run_tiny_imagenet_exp.sh 22
# or
BASE_SEED=22 bash run_tiny_imagenet_exp.sh
```

For each `base_seed`, this script launches exactly 3 runs with real seeds:

- run_0: `seed = base_seed * 1`
- run_1: `seed = base_seed * 2`
- run_2: `seed = base_seed * 3`

It only trains and stores intermediate artifacts (args, checkpoints, forget scores).

### Stage 2: compute EL2N / GraNd / Forgetting means and export masks

After all desired base seeds are trained (default: `22 42 96`):

```bash
cd data_diet
bash get_tiny_imagenet_mask.sh
```

Optional overrides:

```bash
BASE_SEEDS="22 42 96" SCORE_EPOCH=10 KEEP_RATIOS="0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2" bash get_tiny_imagenet_mask.sh
```

Mask output paths follow:

- `E2LN/tiny-imagenet/<seed>/mask_<keep_ratio>.npz`
- `GraNd/tiny-imagenet/<seed>/mask_<keep_ratio>.npz`
- `Forgetting/tiny-imagenet/<seed>/mask_<keep_ratio>.npz`

## Notes

- Tiny-ImageNet uses `resnet34_lowres`.
- Training augmentation keeps **random horizontal flip only** for Tiny-ImageNet.
- Default Tiny-ImageNet setup: `epochs=90`, `train_batch=64`, no grad accumulation, `lr=0.025`, LR drops at epochs 30/60 by 10x.
- EL2N/GraNd default score epoch is 10.
