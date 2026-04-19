from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import NUM_CLASSES, build_test_dataset, build_train_dataset
from .model import build_resnet50
from .utils import ensure_dir


@dataclass
class TrainConfig:
    dataset: str
    data_root: str
    seed: int
    device: torch.device
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 200
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / max(total, 1)


def train_base_model(config: TrainConfig) -> Path:
    train_dataset = build_train_dataset(config.dataset, config.data_root, for_feature=False)
    test_dataset = build_test_dataset(config.dataset, config.data_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    model = build_resnet50(NUM_CLASSES[config.dataset]).to(config.device)
    optimizer = SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = torch.nn.CrossEntropyLoss()

    ckpt_dir = ensure_dir(Path("MDS") / "ckpt" / config.dataset / str(config.seed))
    ckpt_path = ckpt_dir / "best.pth"

    best_acc = -1.0
    for epoch in range(config.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {config.dataset} seed={config.seed} epoch={epoch + 1}/{config.epochs}", leave=False)
        for _, images, targets in pbar:
            images = images.to(config.device, non_blocking=True)
            targets = targets.to(config.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        acc = evaluate(model, test_loader, config.device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt_path)
        print(f"[train] dataset={config.dataset} seed={config.seed} epoch={epoch + 1}/{config.epochs} val_acc={acc:.4f} best={best_acc:.4f}")

    return ckpt_path
