# python run_full_data_tiny_imagenet.py <ROOT:str> <EXP:str> <RUN:int> <SEED:int> [SCORE_EPOCH:int]

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from types import SimpleNamespace

from data_diet.train import train


def count_tiny_imagenet_train_examples(data_dir):
    train_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'train')
    total = 0
    for cls_name in os.listdir(train_dir):
        img_dir = os.path.join(train_dir, cls_name, 'images')
        if os.path.isdir(img_dir):
            total += sum(1 for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)))
    return total


ROOT = sys.argv[1]
EXP = sys.argv[2]
RUN = int(sys.argv[3])
SEED = int(sys.argv[4])
SCORE_EPOCH = int(sys.argv[5]) if len(sys.argv) > 5 else 10

DATA_DIR = ROOT + '/data'
EXPS_DIR = ROOT + '/exps'

EPOCHS = 90
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 256
BASE_LR = 0.025

num_train_examples = count_tiny_imagenet_train_examples(DATA_DIR)
steps_per_epoch = max(1, num_train_examples // TRAIN_BATCH_SIZE)
num_steps = EPOCHS * steps_per_epoch
score_step = max(1, min(num_steps, SCORE_EPOCH * steps_per_epoch))

args = SimpleNamespace()
args.data_dir = DATA_DIR
args.dataset = 'tiny-imagenet'
args.subset = None
args.subset_size = None
args.scores_path = None
args.subset_offset = None
args.random_subset_seed = None
args.model = 'resnet34_lowres'
args.model_seed = SEED
args.load_dir = None
args.ckpt = 0
args.lr = BASE_LR
args.beta = 0.9
args.weight_decay = 1e-4
args.nesterov = True
args.lr_vitaly = False
args.decay_factor = 0.1
args.decay_steps = [30 * steps_per_epoch, 60 * steps_per_epoch]
args.num_steps = num_steps
args.train_seed = SEED
args.train_batch_size = TRAIN_BATCH_SIZE
args.test_batch_size = TEST_BATCH_SIZE
args.augment = True
args.track_forgetting = True
args.save_dir = EXPS_DIR + f'/{EXP}/run_{RUN}'
args.log_steps = steps_per_epoch
args.early_step = score_step
args.early_save_steps = score_step
args.save_steps = num_steps

train(args)
