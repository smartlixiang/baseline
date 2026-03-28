import os
import argparse
import numpy as np
from tqdm import tqdm

KEEP_RATIOS = [20, 30, 40, 50, 60, 70, 80, 90]
SEED = 22
SCORE_ROOT = "scores"
MASK_ROOT = "mask"


def load_scores(dataset):
    score_dir = os.path.join(SCORE_ROOT, dataset, f"seed_{SEED}")
    score_path = os.path.join(score_dir, "scores.npz")
    if not os.path.exists(score_path):
        raise FileNotFoundError(f"Score file not found: {score_path}")
    data = np.load(score_path, allow_pickle=True)
    if "score" in data:
        score = data["score"]
    elif "scores" in data:
        score = data["scores"]
    elif "final_score" in data:
        score = data["final_score"]
    else:
        raise KeyError(f"Cannot find score array in {score_path}. Keys: {data.files}")
    return score


def save_mask(dataset, keep_ratio, selected_indices, total_num):
    mask_dir = os.path.join(MASK_ROOT, dataset, str(SEED))
    os.makedirs(mask_dir, exist_ok=True)

    mask = np.zeros(total_num, dtype=np.uint8)
    mask[selected_indices] = 1

    save_path = os.path.join(mask_dir, f"mask_{keep_ratio}.npz")
    np.savez(
        save_path,
        mask=mask,
        indices=np.array(selected_indices, dtype=np.int64),
        keep_ratio=np.int64(keep_ratio),
        dataset=np.array(dataset),
        seed=np.int64(SEED),
    )
    print(f"saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "tiny-imagenet"])
    args = parser.parse_args()

    dataset = args.dataset
    score = load_scores(dataset)
    score = np.asarray(score).reshape(-1)
    total_num = len(score)

    order = np.argsort(-score)

    for keep_ratio in tqdm(KEEP_RATIOS, desc=f"{dataset} keep_ratios"):
        keep_num = int(round(total_num * keep_ratio / 100.0))
        keep_num = max(1, min(keep_num, total_num))
        selected_indices = order[:keep_num]
        save_mask(dataset, keep_ratio, selected_indices, total_num)


if __name__ == "__main__":
    main()
