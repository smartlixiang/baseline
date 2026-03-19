from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dataset_utils import DEFAULT_DATA_ROOT, DEFAULT_MASK_ROOT, build_eval_train_dataset, get_dataset_targets, resolve_moso_path
from selection_utils import load_score_file, make_binary_mask, select_indices_from_scores


def resolve_score_path(path: str | None, score_root: str | None) -> Path:
    if score_root:
        return resolve_moso_path(score_root) / 'moso_score.pth'
    if not path:
        raise ValueError('Either --path or --score_root must be provided to locate moso_score.pth.')
    return resolve_moso_path(path) / 'score' / 'moso_score.pth'


def export_mask(dataset: str, seed: int, keep_ratio: int, data_root: str, path: str | None, score_root: str | None, mask_root: str | None, random_mode: bool = False) -> Path:
    keep_fraction = keep_ratio / 100.0
    score_path = resolve_score_path(path, score_root)
    if not score_path.exists():
        raise FileNotFoundError(f'Score file not found: {score_path}')

    train_dataset = build_eval_train_dataset(dataset, data_root)
    targets = get_dataset_targets(train_dataset)
    scores = load_score_file(score_path)

    assert len(scores) == len(train_dataset), f'len(score)={len(scores)} != len(train_dataset)={len(train_dataset)}'
    selected_index = select_indices_from_scores(scores, keep_fraction, targets, random_mode=random_mode)
    mask = make_binary_mask(len(train_dataset), selected_index)
    assert len(mask) == len(train_dataset), f'len(mask)={len(mask)} != len(train_dataset)={len(train_dataset)}'

    mask_base = resolve_moso_path(mask_root) if mask_root else DEFAULT_MASK_ROOT.resolve()
    save_dir = mask_base / dataset / str(seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'mask_{keep_ratio}.npz'
    np.savez(save_path, mask=mask, selected_index=selected_index)
    print(f'Saved mask to {save_path}')
    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a MoSo mask in repository-wide NPZ format.')
    parser.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100', 'tiny'])
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--keep_ratio', required=True, type=int, help='Integer keep ratio percentage, e.g. 30 for 30%%.')
    parser.add_argument('--data_root', default=str(DEFAULT_DATA_ROOT), type=str)
    parser.add_argument('--score_root', default=None, type=str, help='Optional directory containing moso_score.pth.')
    parser.add_argument('--mask_root', default=None, type=str, help='Optional mask root, defaults to MoSo/mask.')
    parser.add_argument('--path', default=None, type=str, help='MoSo experiment directory containing score/moso_score.pth.')
    parser.add_argument('--random', default=0, type=int, help='Use retraining random selection logic instead of MoSo selection.')
    args = parser.parse_args()
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
