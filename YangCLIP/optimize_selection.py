import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import get_dataset_subdir, normalize_dataset_name

DEFAULT_CLIP_MODEL_PATH = "clip_model/ViT-B-32.pt"
DEFAULT_KEEP_RATIOS = [20, 30, 40, 50, 60, 70, 80, 90]
BETA = 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_keep_ratios(keep_ratios):
    if isinstance(keep_ratios, str):
        vals = [int(x.strip()) for x in keep_ratios.split(",") if x.strip()]
        return vals
    return keep_ratios


def optimize_mask(similarity_scores, dis_loss, sr, theta=5e-4, lr=1e-3, momentum=0.9, max_iters=100000):
    """Official-style optimization on sample variable d (w), then strict binarization.

    Keeps original idea: optimize continuous logits, enforce ratio with penalty,
    then threshold (strict binary) at the end.
    """
    device = similarity_scores.device
    n = len(similarity_scores)
    k = int(round(n * sr))

    w = nn.Parameter(0.01 * torch.ones(n, device=device))
    optimizer = optim.SGD([w], lr=lr, momentum=momentum)
    scale_factor = 100.0

    progress = tqdm(range(max_iters), desc=f"Optimize keep_ratio={int(sr * 100)}", leave=False)
    for step in progress:
        x = torch.sigmoid(scale_factor * w)
        loss1 = -torch.mean(x * (similarity_scores / (similarity_scores.mean() + 1e-12)))
        loss2 = -torch.mean(x * (dis_loss / (dis_loss.mean() + 1e-12))) * 0.1
        # Paper/official idea: constrain cardinality during optimization.
        loss3 = torch.sqrt(((((x > 0.5).float() - x.detach() + x).sum() - k) / n) ** 2) * BETA
        loss = loss1 + loss2 + loss3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            cur_sr = float((torch.sigmoid(scale_factor * w) > 0.5).float().mean().item())
            progress.set_postfix(loss=f"{loss.item():.5f}", sr=f"{cur_sr:.4f}")

        if loss3.item() < theta:
            break

    scores = torch.sigmoid(scale_factor * w).detach().cpu().numpy()
    mask = (scores > 0.5).astype(np.uint8)

    # Keep count as close as possible to target if optimization settles off-target.
    target = k
    current = int(mask.sum())
    if current != target:
        order = np.argsort(scores)
        if current < target:
            need = target - current
            add_idx = order[::-1][mask[order[::-1]] == 0][:need]
            mask[add_idx] = 1
        else:
            need = current - target
            rm_idx = order[mask[order] == 1][:need]
            mask[rm_idx] = 0

    indices = np.where(mask == 1)[0].astype(np.int64)
    return mask, indices


def main():
    parser = argparse.ArgumentParser(description="YangCLIP selection optimization to output binary masks.")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100", "tiny-imagenet"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--clip_model_path", type=str, default=DEFAULT_CLIP_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--score_dir", type=str, default="scores")
    parser.add_argument("--mask_root", type=str, default="mask")
    parser.add_argument("--keep_ratios", type=str, default=",".join(map(str, DEFAULT_KEEP_RATIOS)))
    parser.add_argument("--theta", type=float, default=5e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--max_iters", type=int, default=100000)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_name = normalize_dataset_name(args.dataset)

    # Keep parameter for compatibility and explicit local-path requirement; no download attempt.
    if not os.path.isfile(args.clip_model_path):
        raise FileNotFoundError(
            f"CLIP checkpoint not found at '{args.clip_model_path}'. Please place local weights there."
        )

    _ = os.path.join(args.data_root, get_dataset_subdir(dataset_name))  # normalized path hook for consistency.

    score_root = os.path.join(args.score_dir, dataset_name, f"seed_{args.seed}")
    sa_norm_path = os.path.join(score_root, "sa_norm.npy")
    sd_norm_path = os.path.join(score_root, "sd_norm.npy")
    if not os.path.isfile(sa_norm_path) or not os.path.isfile(sd_norm_path):
        raise FileNotFoundError(f"Missing score files under: {score_root}. Please run sample_scoring.py first.")

    similarity_scores = torch.tensor(np.load(sa_norm_path), dtype=torch.float32)
    dis_loss = torch.tensor(np.load(sd_norm_path), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    similarity_scores = similarity_scores.to(device)
    dis_loss = dis_loss.to(device)

    keep_ratios = parse_keep_ratios(args.keep_ratios)
    mask_dir = os.path.join(args.mask_root, dataset_name, str(args.seed))
    os.makedirs(mask_dir, exist_ok=True)

    for keep_ratio in tqdm(keep_ratios, desc="keep_ratio list"):
        sr = keep_ratio / 100.0  # alpha = keep_ratio selection ratio per paper.
        mask, indices = optimize_mask(
            similarity_scores=similarity_scores,
            dis_loss=dis_loss,
            sr=sr,
            theta=args.theta,
            lr=args.lr,
            momentum=args.momentum,
            max_iters=args.max_iters,
        )

        out_path = os.path.join(mask_dir, f"mask_{keep_ratio}.npz")
        np.savez(
            out_path,
            mask=mask.astype(np.uint8),
            indices=indices,
            keep_ratio=np.int32(keep_ratio),
            dataset=np.array(dataset_name),
            seed=np.int32(args.seed),
        )
        print(
            f"[optimize_selection] keep_ratio={keep_ratio} "
            f"selected={int(mask.sum())}/{len(mask)} saved={out_path}"
        )


if __name__ == "__main__":
    main()
