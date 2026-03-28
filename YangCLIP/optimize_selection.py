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


def build_ste_mask(w: torch.Tensor):
    soft_mask = torch.sigmoid(100.0 * w)
    hard_mask = (soft_mask >= 0.5).float()
    ste_mask = hard_mask.detach() - soft_mask.detach() + soft_mask
    return soft_mask, ste_mask


def optimize_selection(
    scores: np.ndarray,
    density: np.ndarray,
    keep_ratio: float,
    steps: int,
    lr: float,
    weight_align: float,
    weight_div: float,
    weight_budget: float,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    score_tensor = torch.from_numpy(scores.astype(np.float32)).to(device)
    density_tensor = torch.from_numpy(density.astype(np.float32)).to(device)
    score_tensor = (score_tensor - score_tensor.mean()) / (score_tensor.std() + 1e-6)
    density_tensor = (density_tensor - density_tensor.mean()) / (density_tensor.std() + 1e-6)

    if score_tensor.shape != density_tensor.shape:
        raise ValueError(
            "scores and density must have same shape, "
            f"got {score_tensor.shape} and {density_tensor.shape}"
        )
    num_samples = score_tensor.numel()

    w = torch.zeros(num_samples, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([w], lr=lr, momentum=0.9)

    step_bar = tqdm(range(steps), desc="Selection optimization")
    for step in step_bar:
        mask_soft, mask_ste = build_ste_mask(w)

        loss_align = -(mask_ste * score_tensor).sum()
        # Higher density -> stronger penalty, encouraging dispersed selections.
        loss_div = (mask_ste * density_tensor).sum()
        loss_budget = (mask_soft.mean() - keep_ratio).pow(2)

        loss = (
            weight_align * loss_align
            + weight_div * loss_div
            + weight_budget * loss_budget
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == steps - 1:
            selected_est = int((mask_soft >= 0.5).sum().item())
            step_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                align=f"{loss_align.item():.4f}",
                div=f"{loss_div.item():.4f}",
                budget=f"{loss_budget.item():.4f}",
                selected=f"{selected_est}/{num_samples}",
            )

    with torch.no_grad():
        final_mask = (torch.sigmoid(100.0 * w) > 0.5).to(torch.uint8)

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
    parser.add_argument("--weight_align", type=float, default=1.0)
    parser.add_argument("--weight_div", type=float, default=0.5)
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
    density_path = Path("Pruning_Scores") / args.dataset / str(args.seed) / "density.npy"
    if not score_path.exists():
        raise FileNotFoundError(f"Score file not found: {score_path}")
    if not density_path.exists():
        raise FileNotFoundError(f"Density file not found: {density_path}")

    scores = np.load(score_path)
    density = np.load(density_path)
    if scores.ndim != 1:
        raise ValueError(f"scores.npy must be 1D, but got shape {scores.shape}")
    if density.ndim != 1:
        raise ValueError(f"density.npy must be 1D, but got shape {density.shape}")

    mask = optimize_selection(
        scores=scores,
        density=density,
        keep_ratio=args.keep_ratio,
        steps=args.steps,
        lr=args.lr,
        weight_align=args.weight_align,
        weight_div=args.weight_div,
        weight_budget=args.weight_budget,
    )

    save_dir = Path("mask") / args.dataset / str(args.seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    ratio_tag = int(args.keep_ratio * 100)
    save_path = save_dir / f"mask_{ratio_tag}.npy"
    canonical_path = save_dir / "mask.npy"
    np.save(save_path, mask)
    np.save(canonical_path, mask)
    print(f"Saved mask to {save_path}")
    print(f"Saved canonical mask to {canonical_path}")


if __name__ == "__main__":
    main()
