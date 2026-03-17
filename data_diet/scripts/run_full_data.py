# python run_full_data.py <ROOT:str> <EXP:str> <RUN:int> [DATASET:str] [SEED:int]

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from types import SimpleNamespace

from data_diet.train import train


def count_train_examples(root, dataset):
    if dataset in {'cifar10', 'cifar100'}:
        return 50000
    if dataset == 'cinic10':
        return 90000
    if dataset == 'tiny-imagenet':
        train_root = os.path.join(root, 'data', 'tiny-imagenet-200', 'train')
        total = 0
        for cls_name in os.listdir(train_root):
            img_dir = os.path.join(train_root, cls_name, 'images')
            if os.path.isdir(img_dir):
                total += sum(1 for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)))
        return total
    raise ValueError(f'unsupported dataset={dataset}')


ROOT = sys.argv[1]
EXP = sys.argv[2]
RUN = int(sys.argv[3])
DATASET = sys.argv[4] if len(sys.argv) > 4 else 'cifar10'
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else RUN

DATA_DIR = ROOT + '/data'
EXPS_DIR = ROOT + '/exps'

ntrain = count_train_examples(ROOT, DATASET)
train_batch = 64 if DATASET == 'tiny-imagenet' else 128
epochs = 90 if DATASET == 'tiny-imagenet' else 200
steps_per_epoch = max(1, ntrain // train_batch)

args = SimpleNamespace()
args.data_dir = DATA_DIR
args.dataset = DATASET
args.subset = None
args.subset_size = None
args.scores_path = None
args.subset_offset = None
args.random_subset_seed = None
args.model = 'resnet34_lowres' if DATASET == 'tiny-imagenet' else 'resnet18_lowres'
args.model_seed = SEED
args.load_dir = None
args.ckpt = 0
args.lr = 0.025 if DATASET == 'tiny-imagenet' else 0.1
args.beta = 0.9
args.weight_decay = 1e-4 if DATASET == 'tiny-imagenet' else 5e-4
args.nesterov = True
args.lr_vitaly = False
args.decay_factor = 0.1 if DATASET == 'tiny-imagenet' else 0.2
args.decay_steps = [30 * steps_per_epoch, 60 * steps_per_epoch] if DATASET == 'tiny-imagenet' else [60 * steps_per_epoch, 120 * steps_per_epoch, 160 * steps_per_epoch]
args.num_steps = epochs * steps_per_epoch
args.train_seed = SEED
args.train_batch_size = train_batch
args.test_batch_size = 256 if DATASET == 'tiny-imagenet' else 1024
args.augment = True
args.track_forgetting = True
args.save_dir = EXPS_DIR + f'/{EXP}/run_{RUN}'
args.log_steps = steps_per_epoch
args.early_step = 0
args.early_save_steps = None
args.save_steps = steps_per_epoch

train(args)
