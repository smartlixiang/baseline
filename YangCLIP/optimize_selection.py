import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data_loader_with_index import DATASET_CHOICES

SEED_CHOICES = [22, 42, 96]


def resolve_user_path(path_arg: str, script_dir: Path) -> Path:
    user_path = Path(path_arg)
    if user_path.is_absolute():
        return user_path.resolve()

    candidates = [
        (script_dir.parent / user_path).resolve(),
        (script_dir / user_path).resolve(),
        (Path.cwd() / user_path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return (script_dir / user_path).resolve()


def build_ste_mask(logits: torch.Tensor, temperature: float):
    soft_mask = torch.sigmoid(logits / temperature)
    hard_mask = (soft_mask >= 0.5).float()
    ste_mask = hard_mask.detach() - soft_mask.detach() + soft_mask
    return soft_mask, ste_mask


def optimize_selection(scores: np.ndarray, keep_ratio: float, steps: int, lr: float, weight_sa: float, weight_sds: float, weight_budget: float, temperature: float) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    score_tensor = torch.from_numpy(scores.astype(np.float32)).to(device)
    score_tensor = (score_tensor - score_tensor.mean()) / (score_tensor.std() + 1e-6)

    num_samples = score_tensor.numel()
    keep_num = int(num_samples * keep_ratio)
    keep_num = max(1, min(num_samples, keep_num))

    logits = torch.zeros(num_samples, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=lr, momentum=0.9)

    step_bar = tqdm(range(steps), desc="Selection optimization")
    for step in step_bar:
        soft_mask, ste_mask = build_ste_mask(logits, temperature=temperature)

        # SA: score-aware objective, prioritize high-score samples.
        sa_term = -(score_tensor * ste_mask).mean()
        # SDS: sharpen mask distribution towards binary 0/1 selections.
        sds_term = (soft_mask * (1.0 - soft_mask)).mean()
        # Budget constraint: enforce target keep ratio.
        budget_term = (soft_mask.mean() - keep_ratio).pow(2)

        loss = weight_sa * sa_term + weight_sds * sds_term + weight_budget * budget_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == steps - 1:
            selected_est = int((soft_mask >= 0.5).sum().item())
            step_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                sa=f"{sa_term.item():.4f}",
                sds=f"{sds_term.item():.4f}",
                budget=f"{budget_term.item():.4f}",
                selected=f"{selected_est}/{keep_num}",
            )

    with torch.no_grad():
        final_soft_mask = torch.sigmoid(logits / temperature)
        topk_idx = torch.topk(final_soft_mask, k=keep_num, largest=True).indices
        final_mask = torch.zeros(num_samples, device=device, dtype=torch.uint8)
        final_mask[topk_idx] = 1

    return final_mask.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Optimize sample selection mask from scores")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser.add_argument("--seed", type=int, required=True, choices=SEED_CHOICES)
    parser.add_argument("--keep_ratio", type=float, required=True)
    parser.add_argument("--clip_path", type=str, default="clip_model/ViT-B-32.pt")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--weight_sa", type=float, default=1.0)
    parser.add_argument("--weight_sds", type=float, default=0.5)
    parser.add_argument("--weight_budget", type=float, default=20.0)
    args = parser.parse_args()

    if not (0 < args.keep_ratio <= 1):
        raise ValueError("keep_ratio must be in (0, 1].")

    script_dir = Path(__file__).resolve().parent
    clip_path = resolve_user_path(args.clip_path, script_dir)
    if not clip_path.exists():
        raise FileNotFoundError(
            f"CLIP model file not found: {clip_path}. "
            "Please prepare local file `clip_model/ViT-B-32.pt`."
        )

    score_path = Path("Pruning_Scores") / args.dataset / str(args.seed) / "scores.npy"
    if not score_path.exists():
        raise FileNotFoundError(f"Score file not found: {score_path}")

    scores = np.load(score_path)
    if scores.ndim != 1:
        raise ValueError(f"scores.npy must be 1D, but got shape {scores.shape}")

    mask = optimize_selection(
        scores=scores,
        keep_ratio=args.keep_ratio,
        steps=args.steps,
        lr=args.lr,
        weight_sa=args.weight_sa,
        weight_sds=args.weight_sds,
        weight_budget=args.weight_budget,
        temperature=args.temperature,
    )

    save_path = Path("mask") / args.dataset / str(args.seed) / f"mask_{int(args.keep_ratio * 100)}.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, mask=mask)
    print(f"Saved mask to {save_path}")


if __name__ == "__main__":
    main()
