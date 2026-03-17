import os
import time
from typing import Tuple

import numpy as np
from PIL import Image
from torchvision import datasets


def one_hot(labels, num_classes, dtype=np.float32):
    return (labels[:, None] == np.arange(num_classes)).astype(dtype)


def _norm(x, mean, std):
    mean_rgb = np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3) * 255.0
    std_rgb = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3) * 255.0
    return (x.astype(np.float32) - mean_rgb) / std_rgb


def normalize_cifar10_images(x): return _norm(x, [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
def normalize_cifar100_images(x): return normalize_cifar10_images(x)
def normalize_cinic10_images(x): return _norm(x, [0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835])
def normalize_tiny_imagenet_images(x): return _norm(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def sort_by_class(X, Y):
    order = Y.argmax(1).argsort()
    return X[order], Y[order]


def update_data_args(args, X_train, Y_train, X_test, Y_test):
    args.image_shape = tuple(X_train.shape[1:])
    args.num_classes = int(Y_train.shape[1])
    args.num_train_examples = int(X_train.shape[0])
    args.num_test_examples = int(X_test.shape[0])
    args.steps_per_epoch = max(1, int(args.num_train_examples // args.train_batch_size))
    args.steps_per_test = int(np.ceil(args.num_test_examples / args.test_batch_size))
    return args


def _load_torchvision_set(ds_cls, root, train):
    ds = ds_cls(root=root, train=train, download=True)
    return np.asarray(ds.data), np.asarray(ds.targets)


def load_cifar10(args):
    print('load cifar10... ', end='')
    t0 = time.time()
    X_train, Y_train = _load_torchvision_set(datasets.CIFAR10, args.data_dir, True)
    X_test, Y_test = _load_torchvision_set(datasets.CIFAR10, args.data_dir, False)
    print(f'{int(time.time()-t0)}s')
    X_train, X_test = normalize_cifar10_images(X_train), normalize_cifar10_images(X_test)
    Y_train, Y_test = one_hot(Y_train, 10), one_hot(Y_test, 10)
    X_train, Y_train = sort_by_class(X_train, Y_train)
    X_test, Y_test = sort_by_class(X_test, Y_test)
    return X_train, Y_train, X_test, Y_test, update_data_args(args, X_train, Y_train, X_test, Y_test)


def load_cifar100(args):
    print('load cifar100... ', end='')
    t0 = time.time()
    X_train, Y_train = _load_torchvision_set(datasets.CIFAR100, args.data_dir, True)
    X_test, Y_test = _load_torchvision_set(datasets.CIFAR100, args.data_dir, False)
    print(f'{int(time.time()-t0)}s')
    X_train, X_test = normalize_cifar100_images(X_train), normalize_cifar100_images(X_test)
    Y_train, Y_test = one_hot(Y_train, 100), one_hot(Y_test, 100)
    X_train, Y_train = sort_by_class(X_train, Y_train)
    X_test, Y_test = sort_by_class(X_test, Y_test)
    return X_train, Y_train, X_test, Y_test, update_data_args(args, X_train, Y_train, X_test, Y_test)


def load_cinic10(args):
    print('load cinic10... ', end='')
    t0 = time.time()
    p = os.path.join(args.data_dir, 'cinic10')
    X_train, Y_train = np.load(p + '/X_train.npy'), np.load(p + '/Y_train.npy')
    X_valid, Y_valid = np.load(p + '/X_valid.npy'), np.load(p + '/Y_valid.npy')
    X_test, Y_test = np.load(p + '/X_test.npy'), np.load(p + '/Y_test.npy')
    X_train, Y_train = np.concatenate((X_train, X_valid)), np.concatenate((Y_train, Y_valid))
    X_train, X_test = normalize_cinic10_images(X_train), normalize_cinic10_images(X_test)
    Y_train, Y_test = one_hot(Y_train, 10), one_hot(Y_test, 10)
    X_train, Y_train = sort_by_class(X_train, Y_train)
    X_test, Y_test = sort_by_class(X_test, Y_test)
    print(f'{int(time.time()-t0)}s')
    return X_train, Y_train, X_test, Y_test, update_data_args(args, X_train, Y_train, X_test, Y_test)


def _load_tiny_imagenet(args):
    root = os.path.join(args.data_dir, 'tiny-imagenet-200')
    train_dir, val_dir = os.path.join(root, 'train'), os.path.join(root, 'val')
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    class_to_idx = {n: i for i, n in enumerate(class_names)}

    def load_split(split_dir):
        xs, ys = [], []
        for cls in class_names:
            img_dir = os.path.join(split_dir, cls, 'images')
            if not os.path.isdir(img_dir):
                continue
            for fname in sorted(os.listdir(img_dir)):
                path = os.path.join(img_dir, fname)
                with Image.open(path) as im:
                    xs.append(np.array(im.convert('RGB')))
                ys.append(class_to_idx[cls])
        return np.stack(xs), np.array(ys)

    ann = os.path.join(val_dir, 'val_annotations.txt')
    xs, ys = [], []
    with open(ann, 'r', encoding='utf-8') as f:
        for line in f:
            toks = line.strip().split('\t')
            if len(toks) < 2:
                continue
            fn, cls = toks[0], toks[1]
            with Image.open(os.path.join(val_dir, 'images', fn)) as im:
                xs.append(np.array(im.convert('RGB')))
            ys.append(class_to_idx[cls])
    X_train, Y_train = load_split(train_dir)
    X_test, Y_test = np.stack(xs), np.array(ys)
    X_train, X_test = normalize_tiny_imagenet_images(X_train), normalize_tiny_imagenet_images(X_test)
    Y_train, Y_test = one_hot(Y_train, 200), one_hot(Y_test, 200)
    X_train, Y_train = sort_by_class(X_train, Y_train)
    X_test, Y_test = sort_by_class(X_test, Y_test)
    return X_train, Y_train, X_test, Y_test, update_data_args(args, X_train, Y_train, X_test, Y_test)


def load_dataset(args):
    if args.dataset == 'cifar10':
        return load_cifar10(args)
    if args.dataset == 'cifar100':
        return load_cifar100(args)
    if args.dataset == 'cinic10':
        return load_cinic10(args)
    if args.dataset == 'tiny-imagenet':
        return _load_tiny_imagenet(args)
    raise NotImplementedError


def update_train_data_args(args, I):
    args.num_train_examples = int(I.shape[0])
    args.steps_per_epoch = max(1, int(args.num_train_examples // args.train_batch_size))
    return args


def subset_train_idxs(I, args):
    if args.subset == 'random':
        rng = np.random.RandomState(args.random_subset_seed)
        I = np.sort(rng.choice(I.shape[0], args.subset_size, replace=False)).astype(np.int32)
    else:
        scores = np.load(args.scores_path)
        if args.subset == 'offset':
            idxs = scores.argsort()[args.subset_offset: args.subset_offset + args.subset_size]
        elif args.subset == 'keep_min_scores':
            idxs = scores.argsort()[:args.subset_size]
        elif args.subset == 'keep_max_scores':
            idxs = scores.argsort()[-args.subset_size:]
        elif args.subset == 'keep_min_abs_scores':
            idxs = np.abs(scores).argsort()[:args.subset_size]
        elif args.subset == 'keep_max_abs_scores':
            idxs = np.abs(scores).argsort()[-args.subset_size:]
        else:
            raise NotImplementedError
        I = np.sort(idxs).astype(np.int32)
    return I, update_train_data_args(args, I)


def load_data(args):
    X_train, Y_train, X_test, Y_test, args = load_dataset(args)
    if not hasattr(args, 'random_label_fraction'):
        args.random_label_fraction = 0
    I_train = np.arange(X_train.shape[0], dtype=np.int32)
    if args.subset:
        I_train, args = subset_train_idxs(I_train, args)
    return I_train, X_train, Y_train, X_test, Y_test, args


def _augment_batch(X, rng, do_crop=True):
    if do_crop:
        Xp = np.pad(X, ((0, 0), (4, 4), (4, 4), (0, 0)), mode='reflect')
        outs = np.empty_like(X)
        for i in range(X.shape[0]):
            top = rng.randint(0, 9)
            left = rng.randint(0, 9)
            outs[i] = Xp[i, top:top + X.shape[1], left:left + X.shape[2], :]
        X = outs
    flip = rng.rand(X.shape[0]) < 0.5
    X[flip] = X[flip, :, ::-1, :]
    return X


def train_batches(I, X, Y, args):
    num_examples = I.shape[0]
    rng = np.random.RandomState(args.train_seed)
    order = rng.permutation(I)
    curr_step, start_idx = args.ckpt + 1, 0
    while curr_step <= args.num_steps:
        end_idx = start_idx + args.train_batch_size
        if end_idx > num_examples:
            order = rng.permutation(I)
            start_idx = 0
            continue
        idxs = order[start_idx:end_idx]
        xb = X[idxs].copy()
        if args.augment:
            xb = _augment_batch(xb, rng, do_crop=args.dataset in {'cifar10', 'cifar100', 'cinic10'})
        yb = Y[idxs]
        yield curr_step, idxs, xb, yb
        curr_step += 1
        start_idx = end_idx


def test_batches(X, Y, batch_size):
    start = 0
    while start < X.shape[0]:
        end = min(start + batch_size, X.shape[0])
        yield end - start, X[start:end], Y[start:end]
        start = end


def get_class_balanced_random_subset(X, Y, cls_smpl_sz, seed):
    n_cls = Y.shape[1]
    X_c, Y_c = np.stack(np.split(X, n_cls)), np.stack(np.split(Y, n_cls))
    rng = np.random.RandomState(seed)
    idxs = [rng.choice(X_c.shape[1], cls_smpl_sz, replace=False) for _ in range(n_cls)]
    X = np.concatenate([X_c[c, idxs[c]] for c in range(n_cls)])
    Y = np.concatenate([Y_c[c, idxs[c]] for c in range(n_cls)])
    return X, Y
