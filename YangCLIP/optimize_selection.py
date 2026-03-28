import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data_loader_with_index import DATASET_CHOICES

SEED_CHOICES = [22, 42, 96]
SCALE_FACTOR = 100.0


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


def min_max_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def get_knn_k(dataset: str, class_size: int) -> int:
    if class_size <= 1:
        return 0
    if dataset == "cifar10":
        return min(100, class_size - 1)
    if dataset == "cifar100":
        return min(50, class_size - 1)
    if dataset == "tiny-imagenet":
        k = max(1, int(class_size * 0.1))
        return min(k, class_size - 1)
    raise ValueError(f"Unsupported dataset for KNN setup: {dataset}")


def compute_dis_loss(dataset: str, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"image_features must be 2D, got shape {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "image_features and labels must have same number of samples, "
            f"got {features.shape[0]} and {labels.shape[0]}"
        )

    # Normalize to unit features before cosine KNN.
    feat = features.astype(np.float32)
    feat = feat / np.clip(np.linalg.norm(feat, axis=1, keepdims=True), a_min=1e-12, a_max=None)

    dis_loss = np.zeros(feat.shape[0], dtype=np.float32)
    unique_labels = np.unique(labels)
    for cls in tqdm(unique_labels, desc="Computing class-wise dis_loss"):
        cls_indices = np.where(labels == cls)[0]
        cls_feats = feat[cls_indices]
        cls_size = cls_feats.shape[0]

        k = get_knn_k(dataset, cls_size)
        if k <= 0:
            dis_loss[cls_indices] = 0.0
            continue

        # Cosine similarity matrix within same class.
        sim = cls_feats @ cls_feats.T
        np.fill_diagonal(sim, -np.inf)

        # K nearest neighbors in cosine distance <=> K largest cosine similarities.
        topk_sim = np.partition(sim, kth=cls_size - k, axis=1)[:, -k:]
        mean_knn_sim = topk_sim.mean(axis=1)
        cls_dis_loss = 1.0 - mean_knn_sim
        dis_loss[cls_indices] = cls_dis_loss.astype(np.float32)

    return dis_loss


def optimize_mask(
    similarity_scores: np.ndarray,
    dis_loss: np.ndarray,
    keep_ratio: float,
    total_steps: int,
) -> np.ndarray:
    if similarity_scores.ndim != 1:
        raise ValueError(
            f"similarity_scores must be 1D, but got shape {similarity_scores.shape}"
        )
    if dis_loss.ndim != 1:
        raise ValueError(f"dis_loss must be 1D, but got shape {dis_loss.shape}")
    if similarity_scores.shape[0] != dis_loss.shape[0]:
        raise ValueError(
            "similarity_scores and dis_loss length mismatch, "
            f"got {similarity_scores.shape[0]} and {dis_loss.shape[0]}"
        )

    n = int(similarity_scores.shape[0])
    k = int(round(n * keep_ratio))
    k = max(1, min(k, n))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim = torch.from_numpy(similarity_scores.astype(np.float32)).to(device)
    dis = torch.from_numpy(dis_loss.astype(np.float32)).to(device)

    sim_mean = sim.mean().clamp_min(1e-12)
    dis_mean = dis.mean().clamp_min(1e-12)

    w = (0.01 * torch.ones(n, dtype=torch.float32, device=device)).requires_grad_(True)
    optimizer = torch.optim.SGD([w], lr=1e-3, momentum=0.9)

    progress = tqdm(range(total_steps), desc="Selection optimization")
    for step in progress:
        x = torch.sigmoid(SCALE_FACTOR * w)
        x_hard = (x > 0.5).float()
        x_ste = x_hard - x.detach() + x

        loss1 = -torch.mean(x_ste * (sim / sim_mean))
        loss2 = -torch.mean(x_ste * (dis / dis_mean)) * 0.1
        loss3 = torch.sqrt(((((x_hard - x.detach() + x).sum() - k) / n) ** 2)) * 2.0
        loss = loss1 + loss2 + loss3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == total_steps - 1:
            selected = int((x > 0.5).sum().item())
            progress.set_postfix(
                loss=f"{loss.item():.6f}",
                loss1=f"{loss1.item():.6f}",
                loss2=f"{loss2.item():.6f}",
                loss3=f"{loss3.item():.6f}",
                selected=f"{selected}/{n}",
            )

        if loss3.item() < 0.001:
            progress.set_postfix(
                loss=f"{loss.item():.6f}",
                loss1=f"{loss1.item():.6f}",
                loss2=f"{loss2.item():.6f}",
                loss3=f"{loss3.item():.6f}",
                early_stop="yes",
            )
            break

    with torch.no_grad():
        binary_mask = (torch.sigmoid(SCALE_FACTOR * w) > 0.5).to(torch.uint8)
    return binary_mask.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Official-style selection optimization for YangCLIP")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser.add_argument("--seed", type=int, required=True, choices=SEED_CHOICES)
    parser.add_argument("--keep_ratio", type=float, required=True)
    parser.add_argument("--clip_path", type=str, default="clip_model/ViT-B-32.pt")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100000)
    args = parser.parse_args()

    if not (0 < args.keep_ratio <= 1):
        raise ValueError("keep_ratio must be in (0, 1].")

    # Keep these argument checks for path compatibility with existing scripts.
    script_dir = Path(__file__).resolve().parent
    clip_path = resolve_user_path(args.clip_path, script_dir)
    data_root = resolve_user_path(args.data_root, script_dir)
    if not clip_path.exists():
        raise FileNotFoundError(
            f"CLIP model file not found: {clip_path}. "
            "Please prepare local file `clip_model/ViT-B-32.pt`."
        )
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    score_dir = Path("Pruning_Scores") / args.dataset / str(args.seed)
    score_path = score_dir / "scores.npy"
    features_path = score_dir / "image_features.npy"
    labels_path = score_dir / "labels.npy"
    dis_loss_path = score_dir / "dis_loss.npy"

    if not score_path.exists():
        raise FileNotFoundError(f"Score file not found: {score_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    similarity_scores = np.load(score_path).astype(np.float32)
    similarity_scores = min_max_norm(similarity_scores)
    image_features = np.load(features_path).astype(np.float32)
    labels = np.load(labels_path).astype(np.int64)

    if dis_loss_path.exists():
        dis_loss = np.load(dis_loss_path).astype(np.float32)
        if dis_loss.ndim != 1 or dis_loss.shape[0] != similarity_scores.shape[0]:
            raise ValueError(
                f"Invalid dis_loss shape {dis_loss.shape}, expected ({similarity_scores.shape[0]},)"
            )
        print(f"Loaded dis_loss from {dis_loss_path}")
    else:
        dis_loss = compute_dis_loss(args.dataset, image_features, labels)
        score_dir.mkdir(parents=True, exist_ok=True)
        np.save(dis_loss_path, dis_loss)
        print(f"Saved dis_loss to {dis_loss_path}")

    mask = optimize_mask(
        similarity_scores=similarity_scores,
        dis_loss=dis_loss,
        keep_ratio=args.keep_ratio,
        total_steps=args.steps,
    )

    save_dir = Path("mask") / args.dataset / str(args.seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    ratio_tag = int(args.keep_ratio * 100)
    save_path = save_dir / f"mask_{ratio_tag}.npz"
    np.savez(save_path, mask.astype(np.uint8))
    print(f"Saved mask to {save_path}")


if __name__ == "__main__":
    main()
