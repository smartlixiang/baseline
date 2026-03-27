for dataset in cifar10 cifar100 tiny-imagenet
do
  for seed in 22 42 96
  do
    python train_adapter.py \
      --dataset $dataset \
      --seed $seed \
      --epochs 50 \
      --batch_size 128 \
      --lr 1e-3 \
      --clip_path ../clip_model

    python generate_mask.py \
      --dataset $dataset \
      --seed $seed \
      --keep_ratio 0.9 \
      --clip_path ../clip_model
  done
done
