import os
import time

import torch

from .data import load_data, train_batches
from .forgetting import init_forget_stats, save_forget_scores, update_forget_stats
from .metrics import accuracy, correct, cross_entropy_loss
from .recorder import init_recorder, record_ckpt, record_test_stats, record_train_stats, save_recorder
from .test import test
from .train_state import get_device, get_train_state
from .utils import make_dir, print_args, save_args, set_global_seed


def get_lr(args, step):
    if getattr(args, 'lr_vitaly', False):
        base_lr, top, total = 0.2, 4680, 31200
        if step <= top:
            return base_lr * step / top
        return base_lr - base_lr * (step - top) / (total - top)
    if getattr(args, 'decay_steps', None):
        m = 1.0
        for i, s in enumerate(args.decay_steps):
            if step >= s:
                m = args.decay_factor ** (i + 1)
        return args.lr * m
    return args.lr


def _make_dirs(args):
    make_dir(args.save_dir)
    os.makedirs(args.save_dir + '/ckpts', exist_ok=True)
    if args.track_forgetting:
        os.makedirs(args.save_dir + '/forget_scores', exist_ok=True)


def _save_checkpoint(args, step, state, rec, forget_stats=None):
    torch.save({'step': step, 'model': state.model.state_dict(), 'optimizer': state.optimizer.state_dict()},
               f'{args.save_dir}/ckpts/checkpoint_{step}.pt')
    if forget_stats is not None:
        save_forget_scores(args.save_dir, step, forget_stats)
    return record_ckpt(rec, step)


def train(args):
    set_global_seed(getattr(args, 'train_seed', 0))
    _make_dirs(args)
    I_train, X_train, Y_train, X_test, Y_test, args = load_data(args)
    state, args = get_train_state(args)
    device = get_device(args)

    print('train args:')
    print_args(args)
    save_args(args, args.save_dir, verbose=True)

    rec = init_recorder()
    forget_stats = init_forget_stats(args) if args.track_forgetting else None
    t_start = t_prev = time.time()

    test_loss, test_acc = test(state, X_test, Y_test, args.test_batch_size, device)
    rec = record_test_stats(rec, args.ckpt, test_loss, test_acc)
    print(f"  0.00% | time: {0.0:5.1f}s ({0.0:5.1f}m) | step: {args.ckpt:6d} | lr:  init | train acc:  init | test acc: {test_acc:.3f}")
    rec = _save_checkpoint(args, args.ckpt, state, rec, forget_stats)

    for t, idxs, x, y in train_batches(I_train, X_train, Y_train, args):
        state.model.train()
        xb = torch.from_numpy(x).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
        yb = torch.from_numpy(y).to(device=device, dtype=torch.float32)
        lr = get_lr(args, t)
        for g in state.optimizer.param_groups:
            g['lr'] = lr
        state.optimizer.zero_grad(set_to_none=True)
        logits = state.model(xb)
        loss = cross_entropy_loss(logits, yb)
        loss.backward()
        state.optimizer.step()
        acc = accuracy(logits.detach(), yb)
        if args.track_forgetting:
            batch_accs = correct(logits.detach(), yb).int().cpu().numpy()
            forget_stats = update_forget_stats(forget_stats, idxs, batch_accs)
        rec = record_train_stats(rec, t - 1, loss.item(), acc.item(), lr)

        if t % args.log_steps == 0:
            test_loss, test_acc = test(state, X_test, Y_test, args.test_batch_size, device)
            t_now = time.time()
            print(f"{t/args.num_steps*100:6.2f}% | time: {t_now-t_prev:5.1f}s ({(t_now-t_start)/60:5.1f}m) | step: {t:6d} | lr: {lr:.4f} | train acc: {acc.item():.3f} | test acc: {test_acc:.3f}")
            t_prev = t_now
            rec = record_test_stats(rec, t, test_loss, test_acc)

        if ((t <= args.early_step and args.early_save_steps and t % args.early_save_steps == 0) or
            (t > args.early_step and t % args.save_steps == 0) or
                (t == args.num_steps)):
            if t % args.log_steps != 0:
                test_loss, test_acc = test(state, X_test, Y_test, args.test_batch_size, device)
                rec = record_test_stats(rec, t, test_loss, test_acc)
            rec = _save_checkpoint(args, t, state, rec, forget_stats)

    save_recorder(args.save_dir, rec)
