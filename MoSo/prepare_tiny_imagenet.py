from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dataset_utils import DEFAULT_DATA_ROOT, resolve_dataset_path


def prepare_tiny_val(data_root: str) -> None:
    tiny_root = resolve_dataset_path('tiny', data_root)
    val_root = tiny_root / 'val'
    images_dir = val_root / 'images'
    annotation_path = val_root / 'val_annotations.txt'

    if not val_root.exists():
        raise FileNotFoundError(f'Tiny-ImageNet val directory not found: {val_root}')
    if not annotation_path.exists():
        raise FileNotFoundError(f'Tiny-ImageNet annotation file not found: {annotation_path}')

    if images_dir.exists():
        moved_count = 0
        with annotation_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    parts = line.strip().split()
                if len(parts) < 2:
                    continue
                image_name, class_name = parts[0], parts[1]
                src = images_dir / image_name
                dst_dir = val_root / class_name
                dst = dst_dir / image_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    continue
                if not src.exists():
                    continue
                shutil.move(str(src), str(dst))
                moved_count += 1
        if images_dir.exists() and not any(images_dir.iterdir()):
            images_dir.rmdir()
        print(f'Prepared Tiny-ImageNet val set at {val_root}. Moved {moved_count} images.')
        return

    class_dirs = [path for path in val_root.iterdir() if path.is_dir()]
    if class_dirs:
        print(f'Tiny-ImageNet val set already prepared at {val_root}.')
        return

    raise RuntimeError(
        f'Cannot prepare Tiny-ImageNet val at {val_root}: neither `images/` nor class folders were found.'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare Tiny-ImageNet val split for ImageFolder loading.')
    parser.add_argument('--data_root', default=str(DEFAULT_DATA_ROOT), type=str, help='Root data directory containing tiny-imagenet-200.')
    args = parser.parse_args()
    prepare_tiny_val(args.data_root)
