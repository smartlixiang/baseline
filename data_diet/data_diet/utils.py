import json
import os
import random
import shutil
from types import SimpleNamespace

import numpy as np
import torch

from .train_state import get_train_state


def save_args(args, save_dir, verbose=True):
    path = save_dir + '/args.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)
    if verbose:
        print(f'Save args to {path}')
    return path


def load_args(load_dir, verbose=True):
    path = load_dir + '/args.json'
    with open(path, 'r', encoding='utf-8') as f:
        args = SimpleNamespace(**json.load(f))
    if verbose:
        print(f'Load args from {path}')
    return args


def print_args(args):
    print(json.dumps(vars(args), indent=4))


def make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def load_model_for_scoring(args):
    state, args = get_train_state(args)
    device = state.model.fc.weight.device if hasattr(state.model, 'fc') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return state.model, device, args


def set_global_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
