#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click label-noise experiment script for YangCLIP.

Run from the baseline repository root, e.g.:
    CUDA_VISIBLE_DEVICES=0 python select_from_noise_yangclip.py --dataset cifar10 --seed 22

This script trains YangCLIP image/text adapters on noisy labels, computes semantic
alignment and sample diversity scores under noisy labels, runs optimization-based
selection for keep ratios 30/50/70, and records how many selected samples are
injected-noise samples.

It expects a local CLIP checkpoint at:
    baseline/YangCLIP/clip_model/ViT-B-32.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm

DATASETS = ("cifar10", "cifar100")
SEEDS = (22, 42, 96)
KEEP_RATIOS = (30, 50, 70)
NOISE_RATE = 0.20
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


@dataclass
class Config:
    dataset: str
    seed: int
    data_root: str
    noise_root: str
    exp_root: str
    mask_root: str
    clip_model_path: str
    adapter_epochs: int = 30
    batch_size: int = 256
    num_workers: int = 8
    adapter_lr: float = 1e-4
    lambda_: float = 0.1
    beta_: float = 2.0
    selection_lr: float = 1e-3
    selection_epochs: int = 100000
    scale_factor: float = 100.0
    force: bool = False

    @property
    def noise_list_path(self) -> Path:
        return Path(self.noise_root) / self.dataset / f"noise_list_{self.seed}.txt"

    @property
    def run_dir(self) -> Path:
        return Path(self.exp_root) / self.dataset / f"seed_{self.seed}"

    @property
    def adapter_path(self) -> Path:
        return self.run_dir / "adapter.pt"

    @property
    def score_dir(self) -> Path:
        return self.run_dir / "scores"

    @property
    def score_npz_path(self) -> Path:
        return self.score_dir / "scores.npz"

    @property
    def feature_path(self) -> Path:
        return self.score_dir / "image_features.pt"

    @property
    def run_summary_path(self) -> Path:
        return Path(self.mask_root) / self.dataset / str(self.seed) / "summary.csv"

    @property
    def global_summary_path(self) -> Path:
        return Path(self.mask_root) / "summary.csv"


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        return index, image, target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_noise_list(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Noise list not found: {path}")
    arr = np.loadtxt(path, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Noise list must have two columns, got shape={arr.shape}: {path}")
    return arr


def apply_noise(clean_targets: np.ndarray, noise_list: np.ndarray, num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_targets = np.asarray(clean_targets, dtype=np.int64)
    noisy_targets = clean_targets.copy()
    ids = noise_list[:, 0].astype(np.int64)
    labels = noise_list[:, 1].astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate sample ids in noise list.")
    if np.any(ids < 0) or np.any(ids >= len(clean_targets)):
        raise ValueError("Noise sample id out of range.")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("Noisy label out of range.")
    if np.any(labels == clean_targets[ids]):
        raise ValueError("Noise list contains unchanged labels.")
    noisy_targets[ids] = labels
    is_noisy = np.zeros(len(clean_targets), dtype=bool)
    is_noisy[ids] = True
    if abs(float(is_noisy.mean()) - NOISE_RATE) > 1e-6:
        raise ValueError(f"Noise rate mismatch: {is_noisy.mean()} vs expected {NOISE_RATE}")
    return noisy_targets, ids, is_noisy


def build_noisy_train_dataset(cfg: Config, transform):
    if cfg.dataset == "cifar10":
        ds = datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=transform)
    elif cfg.dataset == "cifar100":
        ds = datasets.CIFAR100(root=cfg.data_root, train=True, download=True, transform=transform)
    else:
        raise ValueError(cfg.dataset)
    clean_targets = np.asarray(ds.targets, dtype=np.int64)
    noisy_targets, noisy_ids, is_noisy = apply_noise(clean_targets, read_noise_list(cfg.noise_list_path), NUM_CLASSES[cfg.dataset])
    ds.targets = noisy_targets.tolist()
    return ds, clean_targets, noisy_targets, noisy_ids, is_noisy


def get_yangclip_classnames(dataset: str) -> List[str]:
    yangclip_dir = Path.cwd() / "YangCLIP"
    if str(yangclip_dir) not in sys.path:
        sys.path.insert(0, str(yangclip_dir))
    try:
        from utils import obtain_classnames  # type: ignore
        return list(obtain_classnames(dataset))
    except Exception as exc:
        raise RuntimeError(
            "Failed to load CIFAR class names from YangCLIP/utils.py. "
            f"Please check the YangCLIP directory. Reason: {exc}"
        ) from exc


def load_clip_model(cfg: Config, device: torch.device):
    if not Path(cfg.clip_model_path).is_file():
        raise FileNotFoundError(
            f"CLIP checkpoint not found: {cfg.clip_model_path}\n"
            "Place ViT-B-32.pt there, or edit clip_model_path in this script."
        )
    import clip  # type: ignore
    model, preprocess = clip.load(cfg.clip_model_path, device=device)
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, preprocess, clip


def encode_text_features(model, clip_module, class_names: List[str], device: torch.device) -> torch.Tensor:
    prompts = [f"a photo of a {c}." for c in class_names]
    text_inputs = torch.cat([clip_module.tokenize(p) for p in prompts]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs).float()
    return text_features


def train_adapters(cfg: Config, device: torch.device):
    model, preprocess, clip_module = load_clip_model(cfg, device)
    input_dim = model.text_projection.shape[1]
    adapter_img = nn.Linear(input_dim, input_dim).to(device)
    adapter_txt = nn.Linear(input_dim, input_dim).to(device)

    if cfg.adapter_path.exists() and not cfg.force:
        print(f"[adapter] load existing {cfg.adapter_path}")
        ckpt = torch.load(cfg.adapter_path, map_location=device)
        adapter_img.load_state_dict(ckpt["adapter_img"])
        adapter_txt.load_state_dict(ckpt["adapter_text"])
        ds, clean_targets, noisy_targets, noisy_ids, is_noisy = build_noisy_train_dataset(cfg, preprocess)
        return model, preprocess, clip_module, adapter_img, adapter_txt, clean_targets, noisy_targets, noisy_ids, is_noisy

    ds, clean_targets, noisy_targets, noisy_ids, is_noisy = build_noisy_train_dataset(cfg, preprocess)
    loader = DataLoader(IndexedDataset(ds), batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    class_names = get_yangclip_classnames(cfg.dataset)
    text_features = encode_text_features(model, clip_module, class_names, device)
    optimizer = torch.optim.Adam(list(adapter_img.parameters()) + list(adapter_txt.parameters()), lr=cfg.adapter_lr)
    cfg.adapter_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in tqdm(range(1, cfg.adapter_epochs + 1), desc=f"Train YangCLIP adapters {cfg.dataset} seed={cfg.seed}"):
        adapter_img.train(); adapter_txt.train()
        running = total = 0
        pbar = tqdm(loader, desc=f"adapter epoch {epoch}/{cfg.adapter_epochs}", leave=False, dynamic_ncols=True)
        for _, images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.no_grad():
                image_feat = model.encode_image(images).float()
            img_out = F.normalize(adapter_img(image_feat), dim=-1)
            txt_out = F.normalize(adapter_txt(text_features[targets]), dim=-1)
            logits = img_out @ txt_out.t()
            labels = torch.arange(images.size(0), device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * images.size(0)
            total += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"[adapter] epoch={epoch} avg_loss={running / max(1, total):.4f}")
    torch.save({"adapter_img": adapter_img.state_dict(), "adapter_text": adapter_txt.state_dict(),
                "dataset": cfg.dataset, "seed": cfg.seed, "epochs": cfg.adapter_epochs}, cfg.adapter_path)
    print(f"[adapter] saved {cfg.adapter_path}")
    return model, preprocess, clip_module, adapter_img, adapter_txt, clean_targets, noisy_targets, noisy_ids, is_noisy


def minmax_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = float(x.max() - x.min())
    if denom < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x.min()) / denom


@torch.no_grad()
def compute_scores(cfg: Config, model, preprocess, clip_module, adapter_img, adapter_txt, device: torch.device,
                   clean_targets, noisy_targets, noisy_ids, is_noisy):
    if cfg.score_npz_path.exists() and cfg.feature_path.exists() and not cfg.force:
        print(f"[score] load {cfg.score_npz_path}")
        arr = np.load(cfg.score_npz_path)
        return arr["sa_scores"], arr["sd_scores"], arr["sa_norm"], arr["sd_norm"]

    ds, *_ = build_noisy_train_dataset(cfg, preprocess)
    loader = DataLoader(IndexedDataset(ds), batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    class_names = get_yangclip_classnames(cfg.dataset)
    text_features = encode_text_features(model, clip_module, class_names, device)
    adapter_img.eval(); adapter_txt.eval(); model.eval()
    ft_text_features = F.normalize(adapter_txt(text_features), dim=-1)
    n = len(ds)
    input_dim = model.text_projection.shape[1]
    image_features = torch.zeros((n, input_dim), dtype=torch.float32)
    sa_scores = torch.full((n,), -1.0, dtype=torch.float32)
    for indices, images, targets in tqdm(loader, desc="YangCLIP feature/SA scoring", dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        image_feat = model.encode_image(images).float()
        img_out = F.normalize(adapter_img(image_feat), dim=-1)
        txt_out = ft_text_features[targets]
        matchness = F.cosine_similarity(img_out, txt_out, dim=-1)
        image_features[indices] = img_out.detach().cpu()
        sa_scores[indices] = matchness.detach().cpu()

    sd_scores = torch.zeros(n, dtype=torch.float32)
    targets_np = np.asarray(noisy_targets, dtype=np.int64)
    k = 100 if cfg.dataset == "cifar10" else 50
    feats_np = image_features.numpy()
    for cls in tqdm(np.unique(targets_np), desc="YangCLIP SD/KNN", dynamic_ncols=True):
        cls_idx = np.where(targets_np == cls)[0]
        cls_feats = feats_np[cls_idx]
        n_neighbors = min(k, len(cls_idx))
        if n_neighbors <= 1:
            continue
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(cls_feats)
        distances, _ = nbrs.kneighbors(cls_feats)
        nearest = distances[:, 1:] if distances.shape[1] > 1 else distances
        sd_scores[cls_idx] = torch.tensor(nearest.mean(axis=1), dtype=torch.float32)

    sa_np = sa_scores.numpy().astype(np.float32)
    sd_np = sd_scores.numpy().astype(np.float32)
    sa_norm = minmax_norm(sa_np)
    sd_norm = minmax_norm(sd_np)
    cfg.score_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.score_npz_path, sa_scores=sa_np, sd_scores=sd_np, sa_norm=sa_norm, sd_norm=sd_norm,
             clean_targets=clean_targets, noisy_targets=noisy_targets, noisy_sample_ids=noisy_ids,
             is_noisy=is_noisy.astype(np.uint8))
    np.save(cfg.score_dir / "targets.npy", noisy_targets)
    torch.save(image_features, cfg.feature_path)
    print(f"[score] saved {cfg.score_npz_path}")
    return sa_np, sd_np, sa_norm, sd_norm


def optimize_selection_scores(similarity_scores: np.ndarray, diversity_scores: np.ndarray, keep_ratio: int, cfg: Config, device: torch.device) -> np.ndarray:
    sim = torch.tensor(similarity_scores, dtype=torch.float32, device=device)
    div = torch.tensor(diversity_scores, dtype=torch.float32, device=device)
    n = len(sim)
    k = int(round(n * keep_ratio / 100.0))
    w = nn.Parameter(0.01 * torch.ones(n, device=device, requires_grad=True))
    optimizer = torch.optim.SGD([w], lr=cfg.selection_lr, momentum=0.9)
    pbar = tqdm(range(cfg.selection_epochs), desc=f"YangCLIP optimize keep={keep_ratio}%", dynamic_ncols=True)
    for epoch in pbar:
        x = torch.sigmoid(cfg.scale_factor * w)
        loss1 = -torch.mean(x * (sim / sim.mean().clamp(min=1e-12)))
        loss2 = -torch.mean(x * (div / div.mean().clamp(min=1e-12))) * cfg.lambda_
        hard_x = (x > 0.5).float()
        st_x = hard_x - x.detach() + x
        loss3 = torch.sqrt((((st_x.sum() - k) / n) ** 2)) * cfg.beta_
        loss = loss1 + loss2 + loss3
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", l3=f"{loss3.item():.4f}", ratio=f"{float((x > 0.5).float().mean().item()):.4f}")
        if loss3.item() < 1e-3:
            break
    final_scores = torch.sigmoid(cfg.scale_factor * w).detach().cpu().numpy().astype(np.float32)
    return final_scores


def mask_from_optimized_scores(selection_scores: np.ndarray, keep_ratio: int) -> np.ndarray:
    n = len(selection_scores)
    k = int(round(n * keep_ratio / 100.0))
    selected = np.argsort(-selection_scores, kind="mergesort")[:k]
    mask = np.zeros(n, dtype=np.uint8)
    mask[selected] = 1
    return mask


def save_mask(cfg: Config, keep_ratio: int, mask: np.ndarray, selection_scores: np.ndarray, sa_scores, sd_scores,
              clean_targets, noisy_targets, noisy_ids, is_noisy) -> dict:
    selected = np.where(mask == 1)[0].astype(np.int64)
    num_noisy_selected = int(is_noisy[selected].sum())
    ratio = float(num_noisy_selected / max(1, len(selected)))
    out_dir = Path(cfg.mask_root) / cfg.dataset / str(cfg.seed) / "YangCLIP"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mask_{keep_ratio}.npz"
    np.savez(out_path, mask=mask.astype(np.uint8), selected_indices=selected,
             selection_scores=selection_scores.astype(np.float32), sa_scores=sa_scores.astype(np.float32), sd_scores=sd_scores.astype(np.float32),
             clean_targets=clean_targets.astype(np.int64), noisy_targets=noisy_targets.astype(np.int64),
             noisy_sample_ids=noisy_ids.astype(np.int64), is_noisy=is_noisy.astype(np.uint8),
             dataset=np.array(cfg.dataset), seed=np.array(cfg.seed), method=np.array("YangCLIP"), keep_ratio=np.array(keep_ratio),
             num_selected=np.array(len(selected)), num_noisy_selected=np.array(num_noisy_selected), noise_ratio_in_mask=np.array(ratio))
    print(f"[mask] kr={keep_ratio} selected={len(selected)} noisy_selected={num_noisy_selected} ratio={ratio:.4f} -> {out_path}")
    return {"dataset": cfg.dataset, "seed": cfg.seed, "method": "YangCLIP", "keep_ratio": keep_ratio,
            "num_selected": len(selected), "num_noisy_selected": num_noisy_selected,
            "noise_ratio_in_mask": ratio, "mask_path": str(out_path)}


def write_summary(path: Path, rows: List[dict]) -> None:
    fields = ["dataset", "seed", "method", "keep_ratio", "num_selected", "num_noisy_selected", "noise_ratio_in_mask", "mask_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"[summary] {path}")


def append_global_summary(path: Path, rows: List[dict]) -> None:
    fields = ["dataset", "seed", "method", "keep_ratio", "num_selected", "num_noisy_selected", "noise_ratio_in_mask", "mask_path"]
    old = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
    keys = {(str(r["dataset"]), str(r["seed"]), str(r["method"]), str(r["keep_ratio"])) for r in rows}
    old = [r for r in old if (str(r["dataset"]), str(r["seed"]), str(r["method"]), str(r["keep_ratio"])) not in keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(old + rows)


def parse_args():
    p = argparse.ArgumentParser(description="YangCLIP selection under fixed 20% label noise.")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    cfg = Config(dataset=args.dataset, seed=args.seed, data_root=str(root / "data"), noise_root=str(root / "noise"),
                 exp_root=str(root / "YangCLIP" / "noise_exps"), mask_root=str(root / "YangCLIP" / "noise_masks"),
                 clip_model_path=str(root / "YangCLIP" / "clip_model" / "ViT-B-32.pt"), force=args.force)
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, clip_module, adapter_img, adapter_txt, clean_targets, noisy_targets, noisy_ids, is_noisy = train_adapters(cfg, device)
    sa_scores, sd_scores, sa_norm, sd_norm = compute_scores(cfg, model, preprocess, clip_module, adapter_img, adapter_txt,
                                                            device, clean_targets, noisy_targets, noisy_ids, is_noisy)
    rows = []
    for kr in KEEP_RATIOS:
        sel_scores = optimize_selection_scores(sa_norm, sd_norm, kr, cfg, device)
        mask = mask_from_optimized_scores(sel_scores, kr)
        rows.append(save_mask(cfg, kr, mask, sel_scores, sa_scores, sd_scores, clean_targets, noisy_targets, noisy_ids, is_noisy))
    write_summary(cfg.run_summary_path, rows)
    append_global_summary(cfg.global_summary_path, rows)
    print("[done]")


if __name__ == "__main__":
    main()
