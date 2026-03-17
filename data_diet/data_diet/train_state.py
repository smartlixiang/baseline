from dataclasses import dataclass
import os
import time

import torch

from .models import get_model, get_num_params


@dataclass
class TrainState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer


def build_optimizer(args, model):
    return torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.beta,
        weight_decay=args.weight_decay,
        nesterov=getattr(args, 'nesterov', False),
    )


def get_device(args):
    return torch.device(getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'))


def checkpoint_path(load_dir, ckpt):
    return os.path.join(load_dir, 'ckpts', f'checkpoint_{ckpt}.pt')


def get_train_state(args):
    t0 = time.time()
    print('get train state... ', end='')
    device = get_device(args)
    model = get_model(args).to(device)
    optimizer = build_optimizer(args, model)
    state = TrainState(model=model, optimizer=optimizer)
    if args.load_dir:
        path = checkpoint_path(args.load_dir, args.ckpt)
        print(f'load from {path}... ', end='')
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
    args.num_params = get_num_params(model)
    print(f'{int(time.time()-t0)}s')
    return state, args
