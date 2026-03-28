import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

DEFAULT_KEEP_RATIOS = [20, 30, 40, 50, 60, 70, 80, 90]


def parse_keep_ratios(value: str) -> List[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    keep_ratios = [int(p) for p in parts]
    for k in keep_ratios:
        if k <= 0 or k > 100:
            raise ValueError(f"Invalid keep ratio: {k}. Expected integers in (0, 100].")
    return keep_ratios


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    denom = float(arr.max() - arr.min())
    if denom < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - arr.min()) / denom


def load_intermediate_scores(score_root: str, dataset: str, seed: int):
    score_dir = os.path.join(score_root, dataset, f"seed_{seed}")
    sa_path = os.path.join(score_dir, "sa_scores.npy")
    sd_path = os.path.join(score_dir, "sd_scores.npy")

    if not os.path.isfile(sa_path):
        raise FileNotFoundError(f"Missing semantic-alignment score file: {sa_path}")
    if not os.path.isfile(sd_path):
        raise FileNotFoundError(f"Missing sample-diversity score file: {sd_path}")

    sa_scores = np.load(sa_path).astype(np.float32)
    sd_scores = np.load(sd_path).astype(np.float32)

    if sa_scores.shape[0] != sd_scores.shape[0]:
        raise ValueError(
            f"Shape mismatch between sa_scores ({sa_scores.shape}) and sd_scores ({sd_scores.shape})."
        )

    return sa_scores, sd_scores


def optimize_mask(
    similarity_scores: torch.Tensor,
    diversity_scores: torch.Tensor,
    keep_ratio: float,
    beta_: float,
    learning_rate: float,
    momentum: float,
    num_epochs: int,
    theta: float,
    scale_factor: float,
) -> np.ndarray:
    """
    Paper-consistent optimization objective:

        L = Lsa + alpha * Lsd + beta * Ls

    where:
        alpha = sr (paper says alpha is proportional to expected selection ratio and
                    can be set equivalent to sr),
        beta = 2,
        Lsa = -mean(sigmoid(d) * SA),
        Lsd = -mean(sigmoid(d) * SD),
        Ls  = sqrt(( STE[I(sigmoid(d)>0.5)] mean - sr )^2)

    We keep the released-code sharpening factor (scale_factor=100) as an engineering
    detail for stable near-binary optimization, but the objective itself follows the paper.
    """
    n = len(similarity_scores)
    sr = float(keep_ratio)
    alpha_ = sr

    # Paper: "Initially, d is initialized with all 1s."
    d = nn.Parameter(torch.ones(n, device=similarity_scores.device, dtype=torch.float32))
    optimizer = optim.SGD([d], lr=learning_rate, momentum=momentum)

    pbar = tqdm(range(num_epochs), desc=f"optimize keep={int(sr * 100)}", leave=False)
    for epoch in pbar:
        x = torch.sigmoid(scale_factor * d)

        loss_sa = -torch.mean(x * similarity_scores)
        loss_sd = -torch.mean(x * diversity_scores)

        hard_x = (x > 0.5).float()
        st_x = hard_x - x.detach() + x

        # Paper Eq. (5): ratio-level selection loss with STE
        loss_sel = torch.sqrt((st_x.mean() - sr) ** 2)

        loss = loss_sa + alpha_ * loss_sd + beta_ * loss_sel

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            selected_ratio = float(hard_x.mean().item())
            pbar.set_postfix(
                loss=f"{loss.item():.6f}",
                lsa=f"{loss_sa.item():.6f}",
                lsd=f"{loss_sd.item():.6f}",
                ls=f"{loss_sel.item():.6f}",
                ratio=f"{selected_ratio:.4f}",
            )

        # Paper: theta = 5e-4 on Ls
        if loss_sel.item() < theta:
            break

    final_scores = torch.sigmoid(scale_factor * d).detach().cpu().numpy()
    binary_mask = (final_scores > 0.5).astype(np.uint8)
    return binary_mask


def save_mask(mask_root: str, dataset: str, seed: int, keep_ratio: int, mask: np.ndarray):
    mask = mask.astype(np.uint8).reshape(-1)
    indices = np.where(mask == 1)[0].astype(np.int64)

    out_dir = os.path.join(mask_root, dataset, str(seed))
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"mask_{keep_ratio}.npz")
    np.savez(
        out_path,
        mask=mask,
        indices=indices,
        keep_ratio=np.int64(keep_ratio),
        dataset=np.array(dataset),
        seed=np.int64(seed),
    )
    print(f"[optimize_selection] saved: {out_path} (selected={len(indices)}/{len(mask)})")


def main():
    parser = argparse.ArgumentParser(description="Run YangCLIP selection optimization.")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "tiny-imagenet"])
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--score_root", type=str, default="scores")
    parser.add_argument("--mask_root", type=str, default="mask")
    parser.add_argument("--keep_ratios", type=str, default=",".join(map(str, DEFAULT_KEEP_RATIOS)))

    # Fixed to paper / released-code defaults
    parser.add_argument("--beta_", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_epochs", type=int, default=100000)
    parser.add_argument("--theta", type=float, default=5e-4)
    parser.add_argument("--scale_factor", type=float, default=100.0)
    args = parser.parse_args()

    keep_ratios = parse_keep_ratios(args.keep_ratios)
    sa_scores, sd_scores = load_intermediate_scores(args.score_root, args.dataset, args.seed)

    # Released code normalizes similarity and diversity terms before optimization.
    sa_norm = minmax_norm(sa_scores)
    sd_norm = minmax_norm(sd_scores)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    similarity_scores = torch.tensor(sa_norm, dtype=torch.float32, device=device)
    diversity_scores = torch.tensor(sd_norm, dtype=torch.float32, device=device)

    for keep_ratio in tqdm(keep_ratios, desc=f"{args.dataset} keep ratios"):
        binary_mask = optimize_mask(
            similarity_scores=similarity_scores,
            diversity_scores=diversity_scores,
            keep_ratio=keep_ratio / 100.0,
            beta_=args.beta_,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            num_epochs=args.num_epochs,
            theta=args.theta,
            scale_factor=args.scale_factor,
        )
        save_mask(args.mask_root, args.dataset, args.seed, keep_ratio, binary_mask)


if __name__ == "__main__":
    main()
