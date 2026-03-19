from __future__ import annotations

import argparse
from pathlib import Path

from export_mask import export_mask, resolve_score_path
from dataset_utils import DEFAULT_DATA_ROOT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check a MoSo score file and export the corresponding mask.')
    parser.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100', 'tiny'])
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--keep_ratio', required=True, type=int)
    parser.add_argument('--data_root', default=str(DEFAULT_DATA_ROOT), type=str)
    parser.add_argument('--score_root', default=None, type=str)
    parser.add_argument('--mask_root', default=None, type=str)
    parser.add_argument('--path', default=None, type=str)
    parser.add_argument('--random', default=0, type=int)
    args = parser.parse_args()

    score_path = resolve_score_path(args.path, args.score_root)
    if not Path(score_path).exists():
        raise FileNotFoundError(f'Score file not found: {score_path}. Please run `python scoring.py ...` from the MoSo directory first.')

    export_mask(
        dataset=args.dataset,
        seed=args.seed,
        keep_ratio=args.keep_ratio,
        data_root=args.data_root,
        path=args.path,
        score_root=args.score_root,
        mask_root=args.mask_root,
        random_mode=bool(args.random),
    )
