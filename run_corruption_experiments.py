"""Unified Tiny-ImageNet image-corruption mask generation.

This runner only schedules Tiny-ImageNet experiments. It reads corruption
specifications from::

    corruption_data/tiny-imagenet/corruption_list_<seed>.txt

The five image corruptions are loaded at runtime from
``corruption_data/corruption_opt.py``; this script does not copy or reimplement
those operations. For training samples, images are loaded in canonical
Tiny-ImageNet order, deterministic corruption is applied before each method's
resize/crop/augmentation/ToTensor/Normalize transform, and the original clean
label is returned unchanged. Validation/test splits are always kept clean.

Methods: EL2N, GraNd, Forgetting, herding, MDS, MoSo, YangCLIP, RLSelector.
Keep ratios are fixed to (30, 50, 70), seeds are fixed to (22, 42, 96),
data_diet runs once per seed, and MoSo keeps four disjoint proxy trials.

Unified masks are written to::

    corruption_<method>/tiny-imagenet/<seed>/mask_<keep_ratio>.npz

with the historical EL2N directory spelling ``corruption_E2LN``. Method-internal
outputs mirror ``run_noise_experiments.py`` with newly generated ``noise`` path
components replaced by ``corruption``. Worker subprocesses inherit stdout/stderr
so internal tqdm bars remain visible; existing valid masks are skipped unless
``--force`` is used; CUDA OOM/SIGKILL/nonzero worker failures are recorded and
later tasks continue.

Examples::

    python run_corruption_experiments.py --device 0
    python run_corruption_experiments.py --seeds 22 --device 0
    python run_corruption_experiments.py --seeds 22,42,96 --device 0 --force
    python run_corruption_experiments.py --scan-only
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import importlib.util
import os
import runpy
import shutil
import signal
import subprocess
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

try:
    import numpy as np
except ModuleNotFoundError:  # Allow --help/--scan-only in minimal environments.
    np = None  # type: ignore[assignment]
try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - real experiments require Pillow.
    Image = None  # type: ignore[assignment]
try:
    from torch.utils.data import Dataset, Subset
except ModuleNotFoundError:  # pragma: no cover - real experiments require torch.
    class Dataset:  # type: ignore[no-redef]
        pass
    class Subset:  # type: ignore[no-redef]
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - compact fallback for checks.
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else [])
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def update(self, n=1):
            return None
        def set_postfix(self, **kwargs):
            return None
        @staticmethod
        def write(message):
            print(message)


REPO_ROOT = Path(__file__).resolve().parent
SEEDS = (22, 42, 96)
KEEP_RATIOS = (30, 50, 70)
CORRUPTION_RATE = 0.20
TINY_DATASET = "tiny-imagenet"
TINY_NUM_CLASSES = 200
TINY_TRAIN_SIZE = 100000
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp")

# Keep the existing repository naming convention. In particular, the historical
# unified directory is corruption_E2LN even though the method is now called EL2N.
UNIFIED_DIR = {
    "EL2N": "corruption_E2LN",
    "GraNd": "corruption_GraNd",
    "Forgetting": "corruption_Forgetting",
    "herding": "corruption_herding",
    "MDS": "corruption_MDS",
    "MoSo": "corruption_MoSo",
    "YangCLIP": "corruption_YangCLIP",
    "RLSelector": "corruption_RLSelector",
}

OOM_EXIT_CODE = 75


def stage(message: str) -> None:
    """Print a concise stage boundary without breaking active tqdm bars."""
    tqdm.write(f"\n[STAGE] {message}")


def cleanup_cuda() -> None:
    gc.collect()
    with contextlib.suppress(Exception):
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def is_oom_exception(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "cuda out of memory",
        "outofmemoryerror",
        "cudnn_status_alloc_failed",
        "can't allocate memory",
        "cannot allocate memory",
        "std::bad_alloc",
    )
    return any(marker in text for marker in markers)


@dataclass(frozen=True)
class ExpectedMask:
    method: str
    dataset: str
    seed: int
    keep_ratio: int

    @property
    def path(self) -> Path:
        return (
            REPO_ROOT
            / UNIFIED_DIR[self.method]
            / self.dataset
            / str(self.seed)
            / f"mask_{self.keep_ratio}.npz"
        )

    @property
    def num_samples(self) -> int:
        if self.dataset == TINY_DATASET:
            return TINY_TRAIN_SIZE
        raise ValueError(self.dataset)


@dataclass(frozen=True)
class Task:
    name: str
    worker: str
    expected: Tuple[ExpectedMask, ...]
    dataset: str = TINY_DATASET


@dataclass(frozen=True)
class CorruptionInfo:
    dataset: str
    seed: int
    list_path: Path
    corruption_types: np.ndarray
    is_corrupted: np.ndarray
    corrupted_indices: np.ndarray


def require_numpy():
    if np is None:
        raise RuntimeError("NumPy is required for running corruption experiments")


def require_pillow():
    if Image is None:
        raise RuntimeError("Pillow is required for running corruption experiments")


def load_corruption_module():
    require_numpy()
    require_pillow()
    return load_module("corruption_opt_runtime", REPO_ROOT / "corruption_data" / "corruption_opt.py")


def apply_image_corruption(image: Image.Image, info: CorruptionInfo, sample_index: int) -> Image.Image:
    corruption_type = int(info.corruption_types[int(sample_index)])
    if corruption_type < 0:
        return image
    seed_sequence = np.random.SeedSequence([
        int(info.seed), int(sample_index), int(corruption_type), 200
    ])
    rng = np.random.default_rng(seed_sequence)
    return load_corruption_module().apply_corruption(image, corruption_type, rng=rng)


class CanonicalTinyImageNet(Dataset):
    """Canonical Tiny-ImageNet loader that can corrupt training images pre-transform."""

    def __init__(self, root: Union[str, Path], train: bool = True, transform=None, corruption_info: CorruptionInfo | None = None):
        self.root = resolve_tiny_root(Path(root))
        self.train = bool(train)
        self.transform = transform
        self.corruption_info = corruption_info if self.train else None
        split_root = self.root / ("train" if self.train else "val")
        if not split_root.is_dir():
            raise FileNotFoundError(f"Tiny-ImageNet split not found: {split_root}")
        self.classes = sorted(p.name for p in split_root.iterdir() if p.is_dir())
        if len(self.classes) != TINY_NUM_CLASSES:
            raise RuntimeError(f"Tiny-ImageNet should contain {TINY_NUM_CLASSES} class directories under {split_root}, found {len(self.classes)}.")
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples: List[Tuple[str, int]] = []
        self.targets: List[int] = []
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            paths = sorted(p for p in (split_root / class_name).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            for path in paths:
                self.samples.append((str(path), class_idx))
                self.targets.append(class_idx)
        if self.train and len(self.samples) != TINY_TRAIN_SIZE:
            raise RuntimeError(f"Tiny-ImageNet train set should contain {TINY_TRAIN_SIZE} images, found {len(self.samples)}.")
        if self.corruption_info is not None and len(self.corruption_info.corruption_types) != len(self.samples):
            raise ValueError("corruption_info length mismatch")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        target = int(self.targets[index])
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.corruption_info is not None:
            image = apply_image_corruption(image, self.corruption_info, index)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def resolve_tiny_root(root: Path) -> Path:
    candidates = (
        root,
        root / "tiny-imagenet-200",
        root / "tiny-imagenet",
        root / "tinyimagenet",
        root / "Tiny-ImageNet",
        root / "Tiny-Imagenet",
    )
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot locate Tiny-ImageNet under {root}. Expected data/tiny-imagenet-200/train and val."
    )


def corruption_list_path(dataset: str, seed: int) -> Path:
    path = REPO_ROOT / "corruption_data" / dataset / f"corruption_list_{seed}.txt"
    if path.is_file():
        return path
    raise FileNotFoundError(f"Corruption list not found: {path}")


def read_corruption_list(dataset: str, seed: int) -> np.ndarray:
    path = corruption_list_path(dataset, seed)
    arr = np.loadtxt(path, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Corruption list must have two integer columns, got {arr.shape}: {path}")
    return arr


def load_corruption_info(dataset: str, seed: int, num_samples: int = TINY_TRAIN_SIZE) -> CorruptionInfo:
    if dataset != TINY_DATASET:
        raise ValueError(f"Only {TINY_DATASET} is supported, got {dataset}")
    path = corruption_list_path(dataset, seed)
    arr = read_corruption_list(dataset, seed)
    ids = arr[:, 0].astype(np.int64)
    types = arr[:, 1].astype(np.int64)
    if len(ids) != int(round(num_samples * CORRUPTION_RATE)):
        raise ValueError(f"Corruption count mismatch: got {len(ids)}, expected {int(round(num_samples * CORRUPTION_RATE))}")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"Duplicate sample ids in {path}")
    if np.any(ids < 0) or np.any(ids >= num_samples):
        raise ValueError(f"Corruption sample id out of range [0,{num_samples}): {path}")
    opt = load_corruption_module()
    n_types = int(getattr(opt, "NUM_CORRUPTION_TYPES", 5))
    if np.any(types < 0) or np.any(types >= n_types):
        raise ValueError(f"Corruption type must be in [0,{n_types}): {path}")
    counts = np.bincount(types, minlength=n_types)
    if n_types != 5 or not np.array_equal(counts[:5], np.full(5, 4000)):
        raise ValueError(f"Expected five corruption types with 4000 rows each, got {counts.tolist()}: {path}")
    corruption_types = np.full(num_samples, -1, dtype=np.int16)
    corruption_types[ids] = types.astype(np.int16)
    is_corrupted = np.zeros(num_samples, dtype=bool)
    is_corrupted[ids] = True
    return CorruptionInfo(dataset, seed, path, corruption_types, is_corrupted, ids)


def build_corrupted_tiny(seed: int, transform=None):
    info = load_corruption_info(TINY_DATASET, seed)
    dataset = build_tiny_dataset(train=True, transform=transform, corruption_info=info)
    clean = np.asarray(dataset.targets, dtype=np.int64)
    # Compatibility with old noisy APIs: labels remain clean, only ids/is_noisy mean corrupted.
    return dataset, clean, clean.copy(), info.corrupted_indices, info.is_corrupted


def set_dataset_targets(dataset: Dataset, targets: Sequence[int]) -> None:
    values = np.asarray(targets, dtype=np.int64).tolist()
    if len(values) != len(dataset):
        raise ValueError(f"Target length mismatch: {len(values)} vs {len(dataset)}")
    dataset.targets = values  # type: ignore[attr-defined]
    if hasattr(dataset, "samples"):
        samples = list(getattr(dataset, "samples"))
        if len(samples) == len(values):
            dataset.samples = [(path, int(values[i])) for i, (path, _) in enumerate(samples)]  # type: ignore[attr-defined]


def load_module(module_name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def copy_mask(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def validate_mask(expected: ExpectedMask) -> Tuple[bool, str]:
    path = expected.path
    if not path.is_file():
        return False, "missing"
    if np is None:
        return False, "numpy unavailable"
    try:
        with np.load(path, allow_pickle=False) as payload:
            if "mask" not in payload:
                return False, "no mask key"
            mask = np.asarray(payload["mask"]).reshape(-1)
    except Exception as exc:
        return False, f"unreadable: {exc}"

    if len(mask) != expected.num_samples:
        return False, f"length={len(mask)}, expected={expected.num_samples}"
    selected = int(np.count_nonzero(mask))
    target_selected = int(round(expected.num_samples * expected.keep_ratio / 100.0))
    if selected != target_selected:
        return False, f"selected={selected}, expected={target_selected}"
    return True, "ok"



def report_corruption_ratio(method: str, seed: int, keep_ratio: int) -> None:
    expected = ExpectedMask(method, TINY_DATASET, seed, keep_ratio)
    ok, reason = validate_mask(expected)
    if not ok:
        tqdm.write(f"method={method} seed={seed} kr={keep_ratio} invalid={reason}")
        return
    info = load_corruption_info(TINY_DATASET, seed)
    with np.load(expected.path, allow_pickle=False) as payload:
        mask = np.asarray(payload["mask"]).reshape(-1)
    selected_indices = np.flatnonzero(mask)
    corrupted = int(info.is_corrupted[selected_indices].sum())
    ratio = corrupted / max(1, len(selected_indices))
    tqdm.write(f"method={method} seed={seed} kr={keep_ratio} corrupted={corrupted}/{len(selected_indices)} ratio={ratio:.4f}")

def expected_for_seed(seed: int) -> List[ExpectedMask]:
    masks: List[ExpectedMask] = []
    for method in ("EL2N", "GraNd", "Forgetting", "herding", "MDS", "MoSo", "YangCLIP", "RLSelector"):
        for kr in KEEP_RATIOS:
            masks.append(ExpectedMask(method, TINY_DATASET, seed, kr))
    return masks


def build_tasks(seed: int) -> List[Task]:
    def masks(methods: Sequence[str], dataset: str = TINY_DATASET) -> Tuple[ExpectedMask, ...]:
        return tuple(ExpectedMask(m, dataset, seed, kr) for m in methods for kr in KEEP_RATIOS)

    return [
        Task("EL2N/GraNd/Forgetting", "data_diet", masks(("EL2N", "GraNd", "Forgetting"))),
        Task("herding", "herding", masks(("herding",))),
        Task("MDS", "mds", masks(("MDS",))),
        Task("MoSo", "moso", masks(("MoSo",))),
        Task("YangCLIP", "yangclip", masks(("YangCLIP",))),
        Task("RL-Selector", "rlselector", masks(("RLSelector",)), TINY_DATASET),
    ]


def torchvision_transforms():
    from torchvision import transforms
    return transforms


def tiny_eval_transform():
    transforms = torchvision_transforms()
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def mds_tiny_transform():
    transforms = torchvision_transforms()
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])


def moso_tiny_train_transform():
    transforms = torchvision_transforms()
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2302, 0.2265, 0.2262)
    return transforms.Compose([
        transforms.RandomResizedCrop(55),
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def moso_tiny_eval_transform():
    transforms = torchvision_transforms()
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2302, 0.2265, 0.2262)
    return transforms.Compose([
        transforms.Resize(int(64 / 0.875)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_tiny_dataset(train: bool, transform=None, corruption_info: CorruptionInfo | None = None) -> CanonicalTinyImageNet:
    return CanonicalTinyImageNet(REPO_ROOT / "data", train=train, transform=transform, corruption_info=corruption_info)


def load_tiny_numpy(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, CorruptionInfo]:
    info = load_corruption_info(TINY_DATASET, seed)
    train = build_tiny_dataset(train=True, transform=None, corruption_info=info)
    val = build_tiny_dataset(train=False, transform=None)

    def materialize(dataset: CanonicalTinyImageNet, desc: str) -> np.ndarray:
        array = np.empty((len(dataset), 64, 64, 3), dtype=np.uint8)
        for i in tqdm(range(len(dataset)), desc=desc, dynamic_ncols=True):
            image, _ = dataset[i]
            array[i] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return array

    clean_targets = np.asarray(train.targets, dtype=np.int64)
    return (
        materialize(train, "materialize corrupted Tiny train"),
        clean_targets,
        materialize(val, "materialize clean Tiny val"),
        np.asarray(val.targets, dtype=np.int64),
        TINY_NUM_CLASSES,
        info,
    )


def normalize_tiny_numpy(x: np.ndarray) -> np.ndarray:
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 1, 1, 3) * 255.0
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 1, 1, 3) * 255.0
    return (x.astype(np.float32) - mean) / std


def augment_tiny_numpy(x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    # Match data_diet/data_diet/data.py: Tiny-ImageNet uses horizontal flip,
    # while random cropping is restricted to the CIFAR/CINIC loaders.
    out = x.copy()
    flip = rng.rand(x.shape[0]) < 0.5
    out[flip] = out[flip, :, ::-1]
    return out


def _run_data_diet(seed: int, force: bool) -> None:
    """Run Tiny-ImageNet data_diet once; labels remain clean."""
    module = load_module("corruption_data_diet_runtime", REPO_ROOT / "data_diet" / "select_from_noise.py")
    original_build_config = module.build_config
    module.SUPPORTED_DATASETS = set(module.SUPPORTED_DATASETS) | {TINY_DATASET}

    stage("data_diet: materializing corrupted Tiny-ImageNet into memory")
    x_train_raw, clean_targets, x_test_raw, y_test_clean, num_classes, info = load_tiny_numpy(seed)

    def patched_build_config(args):
        cfg = original_build_config(args)
        cfg.model = "resnet34_lowres"
        cfg.epochs = 90
        cfg.score_epoch = 10
        cfg.train_batch_size = 64
        cfg.test_batch_size = 256
        cfg.lr = 0.025
        cfg.weight_decay = 1e-4
        cfg.decay_factor = 0.1
        cfg.decay_epochs = (30, 60)
        cfg.keep_ratios = KEEP_RATIOS
        cfg.el2n_batch_size = 256
        cfg.grand_batch_size = 8
        cfg.eval_every_epochs = 10
        return cfg

    def patched_load_noisy_data(cfg):
        x_train = normalize_tiny_numpy(x_train_raw)
        x_test = normalize_tiny_numpy(x_test_raw)
        train_sort_order = np.argsort(clean_targets, kind="mergesort")
        orig_ids_sorted = np.arange(len(clean_targets), dtype=np.int64)[train_sort_order]
        test_sort_order = np.argsort(y_test_clean, kind="mergesort")
        return module.LoadedData(
            x_train_sorted=x_train[train_sort_order],
            y_train_sorted=module.one_hot(clean_targets[train_sort_order], num_classes),
            x_test_sorted=x_test[test_sort_order],
            y_test_sorted=module.one_hot(y_test_clean[test_sort_order], num_classes),
            clean_targets_orig=clean_targets,
            noisy_targets_orig=clean_targets.copy(),
            noisy_sample_ids_orig=info.corrupted_indices,
            is_noisy_orig=info.is_corrupted,
            orig_ids_sorted=orig_ids_sorted,
            train_sort_order=train_sort_order,
            num_classes=num_classes,
            num_train=len(clean_targets),
            num_test=len(y_test_clean),
            steps_per_epoch=max(1, int(len(clean_targets) // cfg.train_batch_size)),
        )

    module.normalize_cifar_images = normalize_tiny_numpy
    module.augment_cifar_batch = augment_tiny_numpy
    module.build_config = patched_build_config
    module.load_noisy_data = patched_load_noisy_data

    args = argparse.Namespace(dataset=TINY_DATASET, seed=seed, force=force)
    cfg = module.build_config(args)
    cfg.seed = seed
    cfg.output_root = str(REPO_ROOT / "data_diet" / "corruption_exps_multirun" / f"base_seed_{seed}")
    cfg.noise_root = str(REPO_ROOT / "corruption_data")
    cfg.mask_root = str(REPO_ROOT / "data_diet" / "corruption_masks")
    cfg.force = force

    module.validate_runtime_environment(cfg)
    module.ensure_clean_dir(cfg.run_dir, force=cfg.force)
    module.ensure_clean_dir(cfg.ckpt_dir, force=False)
    module.ensure_clean_dir(cfg.score_dir, force=False)
    module.ensure_clean_dir(cfg.meta_dir, force=False)
    Path(cfg.mask_root).mkdir(parents=True, exist_ok=True)
    module.save_json(cfg.run_dir / "config.json", module.asdict(cfg))

    module.set_all_seeds(cfg.seed)
    device = module.get_device()
    data = module.load_noisy_data(cfg)
    module.save_data_meta(cfg, data)
    stage("data_diet: single proxy training, epochs=90, score_epoch=10")
    module.train_proxy_model(cfg, data, device)
    stage("data_diet: single EL2N/GraNd/Forgetting scoring pass")
    scores = module.compute_all_scores(cfg, data, device)
    cfg.output_root = str(REPO_ROOT / "data_diet" / "corruption_exps_aggregated")
    cfg.mask_root = str(REPO_ROOT / "data_diet" / "corruption_masks")
    module.generate_masks(cfg, data, scores)

    for method in ("EL2N", "GraNd", "Forgetting"):
        for kr in KEEP_RATIOS:
            source = REPO_ROOT / "data_diet" / "corruption_masks" / TINY_DATASET / str(seed) / method / f"mask_{kr}.npz"
            copy_mask(source, ExpectedMask(method, TINY_DATASET, seed, kr).path)
            report_corruption_ratio(method, seed, kr)
    cleanup_cuda()


def _run_herding(seed: int, force: bool) -> None:
    path = REPO_ROOT / "herding" / "select_from_noise_herding.py"
    module = load_module("corruption_herding_runtime", path)
    module.DATASETS = tuple(module.DATASETS) + (TINY_DATASET,)
    module.KEEP_RATIOS = KEEP_RATIOS
    module.NUM_CLASSES = {**module.NUM_CLASSES, TINY_DATASET: TINY_NUM_CLASSES}
    module.CIFAR_STATS = {**module.CIFAR_STATS, TINY_DATASET: ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))}

    def patched_build(cfg):
        return build_corrupted_tiny(cfg.seed, tiny_eval_transform())

    module.build_noisy_train_dataset = patched_build
    argv = [str(path), "--dataset", TINY_DATASET, "--seed", str(seed)]
    if force:
        argv.append("--force")
    stage("herding: extract pretrained ResNet-18 features, then run class-wise herding")
    with patched_argv(argv):
        module.main()

    for kr in KEEP_RATIOS:
        source = REPO_ROOT / "herding" / "corruption_masks" / TINY_DATASET / str(seed) / "herding" / f"mask_{kr}.npz"
        copy_mask(source, ExpectedMask("herding", TINY_DATASET, seed, kr).path)
        report_corruption_ratio("herding", seed, kr)


def _run_mds(seed: int, force: bool) -> None:
    path = REPO_ROOT / "MDS" / "select_from_noise_mds.py"
    module = load_module("corruption_mds_runtime", path)
    module.DATASETS = tuple(module.DATASETS) + (TINY_DATASET,)
    module.KEEP_RATIOS = KEEP_RATIOS
    module.NUM_CLASSES = {**module.NUM_CLASSES, TINY_DATASET: TINY_NUM_CLASSES}

    def patched_transform():
        return mds_tiny_transform()

    def patched_build(cfg, train: bool, noisy: bool = False):
        if cfg.dataset != TINY_DATASET:
            raise ValueError(cfg.dataset)
        dataset = build_tiny_dataset(train=train, transform=patched_transform(), corruption_info=load_corruption_info(TINY_DATASET, cfg.seed) if train and noisy else None)
        clean = np.asarray(dataset.targets, dtype=np.int64)
        if train and noisy:
            noisy_targets, ids, is_noisy = (clean.copy(), load_corruption_info(TINY_DATASET, cfg.seed).corrupted_indices, load_corruption_info(TINY_DATASET, cfg.seed).is_corrupted)
            set_dataset_targets(dataset, noisy_targets)
            return dataset, clean, noisy_targets, ids, is_noisy
        return dataset

    module.cifar_transform = patched_transform
    module.build_cifar = patched_build

    cfg = module.Config(
        dataset=TINY_DATASET,
        seed=seed,
        data_root=str(REPO_ROOT / "data"),
        noise_root=str(REPO_ROOT / "corruption_data"),
        exp_root=str(REPO_ROOT / "MDS" / "corruption_exps"),
        mask_root=str(REPO_ROOT / "MDS" / "corruption_masks"),
        epochs=200,
        batch_size=128,
        num_workers=4,
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        force=force,
    )
    module.set_seed(seed)
    device = module.torch.device("cuda" if module.torch.cuda.is_available() else "cpu")

    stage("MDS: training ResNet-50 proxy, epochs=200")
    clean_targets, noisy_targets, noisy_ids, is_noisy = module.train_base_model(cfg, device)
    stage("MDS: extracting train-set features")
    features, labels = module.extract_features(
        cfg, device, clean_targets, noisy_targets, noisy_ids, is_noisy
    )
    stage("MDS: computing class-median distances and exporting masks")
    distances = module.compute_distances(cfg, features, labels)
    rows = []
    for kr in KEEP_RATIOS:
        mask = module.select_middle_band(distances, kr)
        rows.append(
            module.save_mask(
                cfg, kr, mask, distances,
                clean_targets, noisy_targets, noisy_ids, is_noisy,
            )
        )
    module.write_summary(cfg.run_summary_path, rows)
    module.append_global_summary(cfg.global_summary_path, rows)

    for kr in KEEP_RATIOS:
        source = (
            REPO_ROOT / "MDS" / "corruption_masks" / TINY_DATASET
            / str(seed) / "MDS" / f"mask_{kr}.npz"
        )
        copy_mask(source, ExpectedMask("MDS", TINY_DATASET, seed, kr).path)
        report_corruption_ratio("MDS", seed, kr)
    cleanup_cuda()

MOSO_TRIALS = 4

def _run_moso(seed: int, force: bool, num_trials: int = MOSO_TRIALS) -> None:
    """Run a four-trial Tiny-ImageNet MoSo-P approximation.

    The original implementation partitions the training set into disjoint support
    subsets and trains one proxy per trial. Here each sample is scored by the best
    checkpoint of the trial containing it. This keeps the four-proxy structure while
    avoiding the original ten-checkpoint-per-trial scoring multiplier.
    """
    path = REPO_ROOT / "MoSo" / "select_from_noise_moso.py"
    module = load_module("corruption_moso_runtime", path)
    module.DATASETS = tuple(module.DATASETS) + (TINY_DATASET,)
    module.KEEP_RATIOS = KEEP_RATIOS
    module.NUM_CLASSES = {**module.NUM_CLASSES, TINY_DATASET: TINY_NUM_CLASSES}
    module.train_transform = moso_tiny_train_transform
    module.eval_transform = moso_tiny_eval_transform

    def patched_build(cfg, train: bool, noisy: bool, transform):
        if cfg.dataset != TINY_DATASET:
            raise ValueError(cfg.dataset)
        dataset = build_tiny_dataset(train=train, transform=transform, corruption_info=load_corruption_info(TINY_DATASET, cfg.seed) if train and noisy else None)
        clean = np.asarray(dataset.targets, dtype=np.int64)
        if train and noisy:
            noisy_targets, ids, is_noisy = (clean.copy(), load_corruption_info(TINY_DATASET, cfg.seed).corrupted_indices, load_corruption_info(TINY_DATASET, cfg.seed).is_corrupted)
            set_dataset_targets(dataset, noisy_targets)
            return dataset, clean, noisy_targets, ids, is_noisy
        return dataset

    def build_tiny_resnet50(cfg):
        model = module.tv_resnet50(weights=None)
        model.fc = module.nn.Linear(model.fc.in_features, TINY_NUM_CLASSES)
        return model

    module.build_cifar = patched_build
    module.build_moso_model = build_tiny_resnet50

    cfg = module.Config(
        dataset=TINY_DATASET,
        seed=seed,
        data_root=str(REPO_ROOT / "data"),
        noise_root=str(REPO_ROOT / "corruption_data"),
        exp_root=str(REPO_ROOT / "MoSo" / "corruption_exps"),
        mask_root=str(REPO_ROOT / "MoSo" / "corruption_masks"),
        model="resnet50",
        epochs=50,
        batch_size=256,
        score_batch_size=1,
        num_workers=4,
        lr=0.1,
        momentum=0.9,
        weight_decay=2e-4,
        force=force,
    )
    module.set_seed(seed)
    device = module.torch.device("cuda" if module.torch.cuda.is_available() else "cpu")

    train_ds, clean_targets, noisy_targets, noisy_ids, is_noisy = module.build_cifar(
        cfg, True, True, module.train_transform()
    )
    eval_train_ds, *_ = module.build_cifar(cfg, True, True, module.eval_transform())
    test_ds = module.build_cifar(cfg, False, False, module.eval_transform())
    test_loader = module.DataLoader(
        test_ds, batch_size=100, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=module.torch.cuda.is_available(),
    )

    support_splits = np.array_split(np.arange(len(train_ds), dtype=np.int64), num_trials)
    combined_scores = np.zeros(len(train_ds), dtype=np.float32)
    trial_root = cfg.run_dir / f"trials_{num_trials}"
    trial_root.mkdir(parents=True, exist_ok=True)

    for trial_index, support_indices in enumerate(support_splits):
        trial_no = trial_index + 1
        trial_dir = trial_root / f"trial_{trial_index}"
        ckpt_path = trial_dir / "best.pth"
        score_path = trial_dir / "scores.npz"
        trial_dir.mkdir(parents=True, exist_ok=True)

        if force:
            ckpt_path.unlink(missing_ok=True)
            score_path.unlink(missing_ok=True)

        if not ckpt_path.exists():
            stage(
                f"MoSo: proxy trial {trial_no}/{num_trials}, "
                f"support={len(support_indices)}, epochs={cfg.epochs}"
            )
            module.set_seed(seed + trial_index)
            model = module.build_moso_model(cfg).to(device)
            criterion = module.nn.CrossEntropyLoss()
            optimizer = module.SGD(
                model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                weight_decay=cfg.weight_decay, nesterov=True,
            )
            scheduler = module.torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs, eta_min=1e-4
            )
            trial_loader = module.DataLoader(
                Subset(train_ds, support_indices.tolist()),
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=module.torch.cuda.is_available(),
            )
            best_acc = -1.0
            epoch_bar = tqdm(
                range(1, cfg.epochs + 1),
                desc=f"MoSo trial {trial_no}/{num_trials} epochs",
                unit="epoch",
                dynamic_ncols=True,
                leave=True,
            )
            for epoch in epoch_bar:
                model.train()
                running_loss = 0.0
                seen = 0
                correct = 0
                batch_bar = tqdm(
                    trial_loader,
                    desc=f"trial {trial_no} epoch {epoch}/{cfg.epochs}",
                    unit="batch",
                    dynamic_ncols=True,
                    leave=False,
                )
                for images, targets in batch_bar:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = criterion(logits, targets)
                    loss.backward()
                    optimizer.step()
                    batch_n = int(targets.numel())
                    running_loss += float(loss.item()) * batch_n
                    seen += batch_n
                    correct += int((logits.argmax(dim=1) == targets).sum().item())
                    batch_bar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        acc=f"{correct / max(1, seen):.3f}",
                    )
                val_acc = module.evaluate(model, test_loader, device)
                lr = float(optimizer.param_groups[0]["lr"])
                epoch_bar.set_postfix(
                    loss=f"{running_loss / max(1, seen):.4f}",
                    val=f"{val_acc:.4f}",
                    best=f"{max(best_acc, val_acc):.4f}",
                    lr=f"{lr:.5f}",
                )
                if val_acc > best_acc:
                    best_acc = val_acc
                    module.torch.save(
                        {
                            "net": model.state_dict(),
                            "acc": val_acc,
                            "epoch": epoch,
                            "lr": lr,
                            "trial": trial_index,
                        },
                        ckpt_path,
                    )
                scheduler.step()
            del model, optimizer, scheduler
            cleanup_cuda()
        else:
            stage(f"MoSo: proxy trial {trial_no}/{num_trials} checkpoint exists; skip training")

        if score_path.exists() and not force:
            payload = np.load(score_path)
            saved_indices = payload["indices"].astype(np.int64)
            saved_scores = payload["scores"].astype(np.float32)
            if not np.array_equal(saved_indices, support_indices):
                raise RuntimeError(f"MoSo cached score indices mismatch: {score_path}")
            combined_scores[support_indices] = saved_scores
            stage(f"MoSo: trial {trial_no}/{num_trials} scores exist; loaded")
            continue

        stage(f"MoSo: exact gradient scoring for trial {trial_no}/{num_trials}")
        ckpt = module.torch.load(ckpt_path, map_location=device)
        model = module.build_moso_model(cfg).to(device)
        model.load_state_dict(ckpt["net"])
        model.eval()
        criterion = module.nn.CrossEntropyLoss()
        support_loader = module.DataLoader(
            Subset(module.IndexedDataset(eval_train_ds, return_index=True), support_indices.tolist()),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        overall_grad = None
        count = 0
        for _, images, targets in tqdm(
            support_loader,
            total=len(support_indices),
            desc=f"MoSo trial {trial_no}/{num_trials} average gradient",
            unit="sample",
            dynamic_ncols=True,
        ):
            images = images.to(device)
            targets = targets.to(device)
            g = module.grad_vector_for_sample(model, images, targets, criterion)
            count += 1
            if overall_grad is None:
                overall_grad = g.clone()
            else:
                overall_grad.mul_((count - 1) / count).add_(g, alpha=1.0 / count)
            del g

        if overall_grad is None or count <= 1:
            raise RuntimeError(f"MoSo trial {trial_no} has insufficient support samples")

        trial_scores = np.zeros(len(support_indices), dtype=np.float32)
        n_support = float(count)
        lr_scale = float(ckpt.get("lr", 1.0))
        for local_pos, (_, images, targets) in enumerate(tqdm(
            support_loader,
            total=len(support_indices),
            desc=f"MoSo trial {trial_no}/{num_trials} sample scores",
            unit="sample",
            dynamic_ncols=True,
        )):
            images = images.to(device)
            targets = targets.to(device)
            g = module.grad_vector_for_sample(model, images, targets, criterion)
            term1 = (2 * n_support - 3) / ((n_support - 1) ** 2) * (overall_grad * overall_grad).sum()
            term2 = -1 / ((n_support - 1) ** 2) * (g * g).sum()
            term3 = (2 * n_support - 4) / ((n_support - 1) ** 2) * (
                (overall_grad - g / n_support) * g
            ).sum()
            trial_scores[local_pos] = float(((term1 + term2 + term3) * lr_scale).detach().cpu().item())
            del g

        combined_scores[support_indices] = trial_scores
        np.savez(score_path, indices=support_indices, scores=trial_scores)
        del model, overall_grad
        cleanup_cuda()

    cfg.score_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cfg.score_path, combined_scores)
    module.torch.save(module.torch.tensor(combined_scores), cfg.run_dir / "score" / "moso_score.pth")

    stage("MoSo: exporting 30%, 50%, and 70% masks")
    rows = []
    for kr in KEEP_RATIOS:
        mask = module.exact_class_top_mask(
            combined_scores, noisy_targets, kr, TINY_NUM_CLASSES
        )
        rows.append(
            module.save_mask(
                cfg, kr, mask, combined_scores,
                clean_targets, noisy_targets, noisy_ids, is_noisy,
            )
        )
    module.write_summary(cfg.run_summary_path, rows)
    module.append_global_summary(cfg.global_summary_path, rows)

    for kr in KEEP_RATIOS:
        source = (
            REPO_ROOT / "MoSo" / "corruption_masks" / TINY_DATASET
            / str(seed) / "MoSo" / f"mask_{kr}.npz"
        )
        copy_mask(source, ExpectedMask("MoSo", TINY_DATASET, seed, kr).path)
        report_corruption_ratio("MoSo", seed, kr)
    cleanup_cuda()

def _tiny_prompt_names() -> List[str]:
    dataset = build_tiny_dataset(train=True, transform=None)
    utils_path = REPO_ROOT / "YangCLIP" / "utils.py"
    utils = load_module("yangclip_utils_runtime", utils_path)
    return list(utils.resolve_class_names(TINY_DATASET, str(REPO_ROOT / "data"), dataset.classes))


def _run_yangclip(seed: int, force: bool) -> None:
    path = REPO_ROOT / "YangCLIP" / "select_from_noise_yangclip.py"
    module = load_module("corruption_yangclip_runtime", path)
    module.DATASETS = tuple(module.DATASETS) + (TINY_DATASET,)
    module.KEEP_RATIOS = KEEP_RATIOS
    module.NUM_CLASSES = {**module.NUM_CLASSES, TINY_DATASET: TINY_NUM_CLASSES}

    def patched_build(cfg, transform):
        return build_corrupted_tiny(cfg.seed, transform)

    prompt_names = _tiny_prompt_names()
    original_get_classnames = module.get_yangclip_classnames
    module.build_noisy_train_dataset = patched_build
    module.get_yangclip_classnames = (
        lambda dataset: prompt_names if dataset == TINY_DATASET else original_get_classnames(dataset)
    )

    cfg = module.Config(
        dataset=TINY_DATASET,
        seed=seed,
        data_root=str(REPO_ROOT / "data"),
        noise_root=str(REPO_ROOT / "corruption_data"),
        exp_root=str(REPO_ROOT / "YangCLIP" / "corruption_exps"),
        mask_root=str(REPO_ROOT / "YangCLIP" / "corruption_masks"),
        clip_model_path=str(REPO_ROOT / "YangCLIP" / "clip_model" / "ViT-B-32.pt"),
        adapter_epochs=30,
        batch_size=256,
        num_workers=8,
        adapter_lr=1e-4,
        lambda_=0.1,
        beta_=2.0,
        selection_lr=1e-3,
        selection_epochs=100000,
        scale_factor=100.0,
        force=force,
    )
    module.set_seed(seed)
    device = module.torch.device("cuda" if module.torch.cuda.is_available() else "cpu")

    stage("YangCLIP: training image/text adapters, epochs=30")
    result = module.train_adapters(cfg, device)
    model, preprocess, clip_module, adapter_img, adapter_txt = result[:5]
    clean_targets, noisy_targets, noisy_ids, is_noisy = result[5:]
    stage("YangCLIP: extracting CLIP features and computing SA/SD scores")
    sa_scores, sd_scores, sa_norm, sd_norm = module.compute_scores(
        cfg, model, preprocess, clip_module, adapter_img, adapter_txt,
        device, clean_targets, noisy_targets, noisy_ids, is_noisy,
    )
    rows = []
    for kr in KEEP_RATIOS:
        stage(f"YangCLIP: optimizing keep_ratio={kr}%")
        sel_scores = module.optimize_selection_scores(sa_norm, sd_norm, kr, cfg, device)
        mask = module.mask_from_optimized_scores(sel_scores, kr)
        rows.append(
            module.save_mask(
                cfg, kr, mask, sel_scores, sa_scores, sd_scores,
                clean_targets, noisy_targets, noisy_ids, is_noisy,
            )
        )
    module.write_summary(cfg.run_summary_path, rows)
    module.append_global_summary(cfg.global_summary_path, rows)

    for kr in KEEP_RATIOS:
        source = (
            REPO_ROOT / "YangCLIP" / "corruption_masks" / TINY_DATASET
            / str(seed) / "YangCLIP" / f"mask_{kr}.npz"
        )
        copy_mask(source, ExpectedMask("YangCLIP", TINY_DATASET, seed, kr).path)
        report_corruption_ratio("YangCLIP", seed, kr)
    cleanup_cuda()

def _patch_rl_dataset_module(seed: int, dataset_name: str):
    rl_dir = REPO_ROOT / "RLSelector"
    if str(rl_dir) not in sys.path:
        sys.path.insert(0, str(rl_dir))
    module = load_module("Dataset", rl_dir / "Dataset.py")
    info = load_corruption_info(TINY_DATASET, seed)

    class RuntimeTinyImageNet(CanonicalTinyImageNet):
        def __init__(self, root, train=True, transform=None, download=False):
            super().__init__(root, train=train, transform=transform, corruption_info=info if train else None)

    module.TinyImageNet = RuntimeTinyImageNet
    return module


def _run_rlselector_single(seed: int, device: str, dataset: str, keep_ratio: int, force: bool) -> None:
    expected = ExpectedMask("RLSelector", dataset, seed, keep_ratio)
    if validate_mask(expected)[0] and not force:
        stage(f"RL-Selector {dataset} keep={keep_ratio}%: valid mask exists; skip")
        return
    rl_dir = REPO_ROOT / "RLSelector"
    _patch_rl_dataset_module(seed, dataset)
    temp_root = REPO_ROOT / ".corruption_rlselector_raw"
    internal_dataset = "tiny-imagenet"
    internal_output = "tinyimagenet"
    source = temp_root / internal_output / str(seed) / f"mask_{keep_ratio}.npz"
    target = ExpectedMask("RLSelector", dataset, seed, keep_ratio).path
    if force:
        source.unlink(missing_ok=True)
        target.unlink(missing_ok=True)

    argv = [
        str(rl_dir / "train.py"),
        "--dataset", internal_dataset,
        "--seed", str(seed),
        "--keep_ratio", str(keep_ratio),
        "--root", str(REPO_ROOT / "data"),
        "--output_root", str(temp_root),
        "--device", str(device),
    ]
    stage(
        f"RL-Selector: dataset={dataset}, keep_ratio={keep_ratio}%, "
        f"epochs={'90' if dataset == TINY_DATASET else '200'}"
    )
    old_cwd = Path.cwd()
    try:
        os.chdir(rl_dir)
        with concise_rl_logging(dataset, keep_ratio), patched_argv(argv):
            runpy.run_path(str(rl_dir / "train.py"), run_name="__main__")
    finally:
        os.chdir(old_cwd)

    if not source.is_file():
        raise FileNotFoundError(f"RL-Selector did not generate {source}")
    with np.load(source, allow_pickle=False) as payload:
        if "mask" not in payload:
            raise ValueError(f"RL-Selector mask has no 'mask' key: {source}")
        mask = np.asarray(payload["mask"], dtype=np.uint8).reshape(-1)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, mask=mask)
    report_corruption_ratio("RLSelector", seed, keep_ratio)


def _run_rlselector(seed: int, device: str, dataset: str, force: bool) -> None:
    for kr in KEEP_RATIOS:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker", "rlselector-single",
            "--seed", str(seed),
            "--device", str(device),
            "--dataset", dataset,
            "--keep-ratio", str(kr),
        ]
        if force:
            command.append("--force")
        completed = subprocess.run(command, cwd=REPO_ROOT)
        if completed.returncode == OOM_EXIT_CODE or completed.returncode in (-signal.SIGKILL, 137):
            raise RuntimeError(
                f"CUDA out of memory or process killed in RL-Selector: "
                f"dataset={dataset}, keep_ratio={kr}"
            )
        if completed.returncode != 0:
            raise RuntimeError(f"RL-Selector failed for {dataset}, keep_ratio={kr}")


@contextlib.contextmanager
def patched_argv(argv: Sequence[str]):
    old = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old


@contextlib.contextmanager
def concise_rl_logging(dataset: str, keep_ratio: int):
    """Route RL-Selector's useful epoch metrics to tqdm and disable file logs."""
    import logging

    original_basic_config = logging.basicConfig
    original_info = logging.info
    state = {"cr": "-", "acc": "-"}

    def no_file_basic_config(*args, **kwargs):
        # RLSelector/train.py calls basicConfig(filename=...).  Suppress it so this
        # unified runner does not create a second, hidden progress channel.
        return None

    def strip_equals(value: str) -> str:
        return value.strip().strip("=").strip()

    def concise_info(message, *args, **kwargs):
        try:
            text = str(message) % args if args else str(message)
        except Exception:
            text = str(message)

        if text.startswith("Loaded dataset"):
            tqdm.write(f"[RL] {text}")
            return
        if "Saving Masks:" in text or text.startswith("Saved standard mask"):
            tqdm.write(f"[RL] {text}")
            return
        if text.startswith("EPOCH:") and "CR:" in text:
            state["cr"] = strip_equals(text.split("CR:", 1)[1])
            return
        if text.startswith("EPOCH:") and "ACC:" in text:
            state["acc"] = strip_equals(text.split("ACC:", 1)[1])
            return
        if text.startswith("BEST EPOCH:") and "BEST ACC:" in text:
            left, best = text.split(",BEST ACC:", 1)
            epoch = left.split("BEST EPOCH:", 1)[1].strip()
            tqdm.write(
                f"[RL] dataset={dataset} keep={keep_ratio}% epoch={epoch} "
                f"acc={state['acc']} CR={state['cr']} best={best.strip()}"
            )

    logging.basicConfig = no_file_basic_config
    logging.info = concise_info
    try:
        yield
    finally:
        logging.basicConfig = original_basic_config
        logging.info = original_info


def run_worker(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    if args.worker == "data_diet":
        _run_data_diet(args.seed, args.force)
    elif args.worker == "herding":
        _run_herding(args.seed, args.force)
    elif args.worker == "mds":
        _run_mds(args.seed, args.force)
    elif args.worker == "moso":
        _run_moso(args.seed, args.force, MOSO_TRIALS)
    elif args.worker == "yangclip":
        _run_yangclip(args.seed, args.force)
    elif args.worker == "rlselector":
        _run_rlselector(args.seed, args.device, args.dataset, args.force)
    elif args.worker == "rlselector-single":
        if args.keep_ratio not in KEEP_RATIOS:
            raise ValueError(f"Invalid keep ratio: {args.keep_ratio}")
        _run_rlselector_single(args.seed, args.device, args.dataset, args.keep_ratio, args.force)
    else:
        raise ValueError(args.worker)


def task_is_complete(task: Task) -> bool:
    return all(validate_mask(mask)[0] for mask in task.expected)


def run_task(task: Task, args: argparse.Namespace, seed: int) -> Tuple[bool, str]:
    if task_is_complete(task) and not args.force:
        return True, "existing"

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker", task.worker,
        "--seed", str(seed),
        "--device", str(args.device),
        "--dataset", task.dataset,
    ]
    if args.force:
        command.append("--force")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    env["PYTHONUNBUFFERED"] = "1"

    # stdout/stderr are deliberately inherited so every method's own tqdm bars
    # remain visible in the tmux terminal. Nothing is redirected to a log file.
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    if completed.returncode == 0 and task_is_complete(task):
        return True, "generated"
    if completed.returncode in (OOM_EXIT_CODE, -signal.SIGKILL, 137):
        return False, "oom"
    return False, "failed"

def scan_expected(seeds: Iterable[int]) -> List[Tuple[ExpectedMask, str]]:
    missing: List[Tuple[ExpectedMask, str]] = []
    for seed in seeds:
        for expected in expected_for_seed(seed):
            ok, reason = validate_mask(expected)
            if not ok:
                missing.append((expected, reason))
    return missing


def print_scan(missing: List[Tuple[ExpectedMask, str]]) -> None:
    if not missing:
        print("[SCAN] all expected masks are present and valid")
        return
    print(f"[SCAN] missing or invalid masks: {len(missing)}")
    for expected, reason in missing:
        print(
            f"  {expected.method:10s} dataset={expected.dataset:14s} "
            f"seed={expected.seed} kr={expected.keep_ratio}: {reason}"
        )


def parse_seeds(value: str) -> Tuple[int, ...]:
    seen = set()
    parsed: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        seed = int(part)
        if seed not in SEEDS:
            raise argparse.ArgumentTypeError(f"--seeds may only contain {SEEDS}, got {seed}")
        if seed not in seen:
            parsed.append(seed)
            seen.add(seed)
    if not parsed:
        raise argparse.ArgumentTypeError("--seeds cannot be empty")
    return tuple(parsed)


def run_parent(args: argparse.Namespace) -> None:
    for seed in args.seeds:
        corruption_list_path(TINY_DATASET, seed)
    resolve_tiny_root(REPO_ROOT / "data")

    statuses: List[Tuple[int, str, str]] = []
    stage(f"run: seeds={args.seeds}, GPU={args.device}, keep_ratios={KEEP_RATIOS}, moso_trials={MOSO_TRIALS}")
    with tqdm(total=len(args.seeds) * 6, desc="corruption experiments", unit="task", dynamic_ncols=True) as bar:
        for seed in args.seeds:
            for task in build_tasks(seed):
                bar.set_postfix(seed=seed, task=task.name)
                ok, status = run_task(task, args, seed)
                statuses.append((seed, task.name, status))
                if ok:
                    tqdm.write(f"[OK] seed={seed} {task.name}: {status}")
                else:
                    tag = "OOM" if status == "oom" else "FAIL"
                    tqdm.write(f"[{tag}] seed={seed} {task.name}; skipped and continuing")
                cleanup_cuda()
                bar.update(1)

    failed = [(seed, name, status) for seed, name, status in statuses if status in {"oom", "failed"}]
    if failed:
        tqdm.write("[RUN] failed tasks: " + ", ".join(f"seed={seed} {name}({status})" for seed, name, status in failed))
    else:
        tqdm.write("[RUN] all scheduled tasks completed or already existed")
    print_scan(scan_expected(args.seeds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tiny-ImageNet image-corruption data-selection experiments.")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("22,42,96"), help="Comma-separated subset of 22,42,96.")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--force", action="store_true", help="Rerun methods and overwrite generated outputs.")
    parser.add_argument("--scan-only", action="store_true", help="Only scan selected seeds without launching workers.")
    parser.add_argument("--worker", choices=("data_diet", "herding", "mds", "moso", "yangclip", "rlselector", "rlselector-single"), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=(TINY_DATASET,), default=TINY_DATASET, help=argparse.SUPPRESS)
    parser.add_argument("--keep-ratio", type=int, choices=KEEP_RATIOS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker and args.seed is None:
        parser.error("--seed is required for worker mode")
    return args

def main() -> None:
    args = parse_args()
    if args.worker:
        try:
            run_worker(args)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            if is_oom_exception(exc):
                print(f"[OOM] {type(exc).__name__}: {exc}")
                cleanup_cuda()
                raise SystemExit(OOM_EXIT_CODE)
            traceback.print_exc()
            raise SystemExit(1)
        return
    if args.scan_only:
        print_scan(scan_expected(args.seeds))
        return
    run_parent(args)


if __name__ == "__main__":
    main()