import argparse
import random
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader_with_index import DATASET_CHOICES, build_train_dataset


SEED_CHOICES = [22, 42, 96]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(device: torch.device, clip_path: str):
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=clip_path)
    model = model.float()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    image_adapter = nn.Linear(512, 512).to(device)
    text_adapter = nn.Linear(512, 512).to(device)
    return model, preprocess, image_adapter, text_adapter


def train_one_epoch(
    model,
    image_adapter,
    text_adapter,
    train_loader,
    class_text_features,
    optimizer,
    device,
):
    image_adapter.train()
    text_adapter.train()

    total_loss = 0.0
    total_num = 0

    for images, labels, _ in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            image_features = model.encode_image(images).float()

        image_features = F.normalize(image_adapter(image_features), dim=-1)

        adapted_text_features = F.normalize(text_adapter(class_text_features), dim=-1)
        logits = image_features @ adapted_text_features.t()

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_num += batch_size

    return total_loss / max(total_num, 1)


def main():
    parser = argparse.ArgumentParser(description="Train YangCLIP adapters")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser.add_argument("--seed", type=int, required=True, choices=SEED_CHOICES)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_path", type=str, default="../clip_model")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, preprocess, image_adapter, text_adapter = build_model(device, args.clip_path)
    data_root = (Path(__file__).resolve().parent / args.data_root).resolve()

    train_dataset, class_names = build_train_dataset(args.dataset, preprocess, data_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    prompts = [f"A photo of a {name}" for name in class_names]
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        class_text_features = model.encode_text(text_tokens).float()

    optimizer = torch.optim.Adam(
        list(image_adapter.parameters()) + list(text_adapter.parameters()),
        lr=args.lr,
    )

    epoch_bar = tqdm(range(1, args.epochs + 1))
    for _ in epoch_bar:
        avg_loss = train_one_epoch(
            model=model,
            image_adapter=image_adapter,
            text_adapter=text_adapter,
            train_loader=train_loader,
            class_text_features=class_text_features,
            optimizer=optimizer,
            device=device,
        )
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}")

    save_path = Path("checkpoints") / args.dataset / f"seed_{args.seed}" / "adapter.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_adapter": image_adapter.state_dict(),
            "text_adapter": text_adapter.state_dict(),
            "dataset": args.dataset,
            "seed": args.seed,
        },
        save_path,
    )
    print(f"Saved adapter checkpoint to {save_path}")


if __name__ == "__main__":
    main()
