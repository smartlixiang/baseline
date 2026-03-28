import argparse
import os
import random

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset as dataset_lib
from utils import normalize_dataset_name, obtain_classnames

DEFAULT_CLIP_MODEL_PATH = "clip_model/ViT-B-32.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_text_features(model, dataset_name: str, device: torch.device, class_names=None):
    if class_names is None:
        class_names = obtain_classnames(dataset_name)
    text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}.") for c in class_names]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs).float()
    return text_features


def build_loader(args, transform):
    import dataset as dataset_lib
    from torch.utils.data import DataLoader

    train_dataset = dataset_lib.build_dataset(
        dataset_name=args.dataset,
        data_root="data",
        train=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return train_dataset, train_loader


def main():
    parser = argparse.ArgumentParser(description="Train YangCLIP image/text adapters (fixed default setting).")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "tiny-imagenet"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--clip_model_path", type=str, default=DEFAULT_CLIP_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="adapter_ckpt")
    args = parser.parse_args()

    dataset_name = normalize_dataset_name(args.dataset)
    if args.batch_size is None:
        args.batch_size = 64 if dataset_name == "tiny-imagenet" else 256

    set_seed(args.seed)

    if not os.path.isfile(args.clip_model_path):
        raise FileNotFoundError(
            f"CLIP checkpoint not found at '{args.clip_model_path}'. "
            f"Please place local weights there."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, preprocess = clip.load(args.clip_model_path, device=device)
    model = model.float()
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    input_dim = model.text_projection.shape[1]
    adapter_img = nn.Linear(input_dim, input_dim).to(device)
    adapter_txt = nn.Linear(input_dim, input_dim).to(device)

    optimizer = torch.optim.Adam(
        list(adapter_img.parameters()) + list(adapter_txt.parameters()),
        lr=args.lr,
    )

    train_dataset, train_loader = build_loader(args, preprocess)

    class_names = train_dataset.classes if dataset_name == "tiny-imagenet" else obtain_classnames(dataset_name)
    text_features = get_text_features(model, dataset_name, device, class_names=class_names)

    save_dir = os.path.join(args.save_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"adapter_seed_{args.seed}.pt")

    epoch_bar = tqdm(range(args.epochs), desc=f"Train adapters ({dataset_name})", leave=True)
    for epoch in epoch_bar:
        adapter_img.train()
        adapter_txt.train()

        running_loss = 0.0
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)

        for _, images, target in batch_bar:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.no_grad():
                image_feat = model.encode_image(images).float()

            img_out = F.normalize(adapter_img(image_feat), dim=-1)
            txt_out = F.normalize(adapter_txt(text_features[target]), dim=-1)

            logits = img_out @ txt_out.t()
            labels = torch.arange(images.size(0), device=device)

            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.t(), labels)
            loss = 0.5 * (loss_i + loss_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_dataset)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")

        torch.save(
            {
                "adapter_img": adapter_img.state_dict(),
                "adapter_text": adapter_txt.state_dict(),
                "dataset": dataset_name,
                "seed": args.seed,
                "epochs": args.epochs,
                "lr": args.lr,
            },
            save_path,
        )

    print(f"[train_adapter] saved: {save_path}")


if __name__ == "__main__":
    main()
