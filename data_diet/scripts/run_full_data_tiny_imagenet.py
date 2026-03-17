# python run_full_data_tiny_imagenet.py <ROOT:str> <EXP:str> <RUN:int> <BASE_SEED:int>

import os
import sys
from types import SimpleNamespace

from data_diet.train import train


def count_tiny_imagenet_train_examples(data_dir):
  train_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'train')
  n = 0
  for cls_name in os.listdir(train_dir):
    img_dir = os.path.join(train_dir, cls_name, 'images')
    if not os.path.isdir(img_dir):
      continue
    n += sum(1 for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)))
  return n


ROOT = sys.argv[1]
EXP = sys.argv[2]
RUN = int(sys.argv[3])
BASE_SEED = int(sys.argv[4])

DATA_DIR = ROOT + '/data'
EXPS_DIR = ROOT + '/exps'

# Tiny-ImageNet configuration
EPOCHS = 90
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 256
SCORE_EPOCH = 10

# Current training loop does not safely support gradient accumulation with minimal changes.
# We therefore keep batch_size=64 and linearly scale learning rate from 0.1*(64/256)=0.025.
BASE_LR = 0.025
META_MODEL_SEED, META_TRAIN_SEED, SEED_INCR = 42, 4242, 424242

num_train_examples = count_tiny_imagenet_train_examples(DATA_DIR)
steps_per_epoch = num_train_examples // TRAIN_BATCH_SIZE
num_steps = EPOCHS * steps_per_epoch
score_step = SCORE_EPOCH * steps_per_epoch

model_seed = META_MODEL_SEED + BASE_SEED * 1000000 + RUN * SEED_INCR
train_seed = META_TRAIN_SEED + BASE_SEED * 1000000 + RUN * SEED_INCR

print(f'[tiny-imagenet] steps_per_epoch={steps_per_epoch}')
print(f'[tiny-imagenet] total_steps={num_steps}')
print(f'[tiny-imagenet] score_epoch={SCORE_EPOCH}, score_step={score_step}')
print(f'[tiny-imagenet] save_dir={EXPS_DIR}/{EXP}/run_{RUN}')
print(f'[tiny-imagenet] model_seed={model_seed}, train_seed={train_seed}')

args = SimpleNamespace()
# data
args.data_dir = DATA_DIR
args.dataset = 'tiny-imagenet'
# subsets
args.subset = None
args.subset_size = None
args.scores_path = None
args.subset_offset = None
args.random_subset_seed = None
# model
args.model = 'resnet34_lowres'
args.model_seed = model_seed
args.load_dir = None
args.ckpt = 0
# optimizer
args.lr = BASE_LR
args.beta = 0.9
args.weight_decay = 1e-4
args.nesterov = True
args.lr_vitaly = False
args.decay_factor = 0.1
args.decay_steps = [30 * steps_per_epoch, 60 * steps_per_epoch]
# training
args.num_steps = num_steps
args.train_seed = train_seed
args.train_batch_size = TRAIN_BATCH_SIZE
args.test_batch_size = TEST_BATCH_SIZE
args.augment = True
args.track_forgetting = True
# checkpoints
args.save_dir = EXPS_DIR + f'/{EXP}/run_{RUN}'
args.log_steps = steps_per_epoch
args.early_step = 0
args.early_save_steps = None
args.save_steps = steps_per_epoch

train(args)
