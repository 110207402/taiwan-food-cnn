"""Dataset, transforms, and dataloader construction."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(img_size: int, augment_cfg: Dict) -> transforms.Compose:
    rc_scale = tuple(augment_cfg.get("random_crop_scale", [0.6, 1.0]))
    cj = augment_cfg.get("color_jitter", 0.2)
    rot = augment_cfg.get("rotation_degrees", 15)
    n = augment_cfg.get("rand_augment_n", 2)
    m = augment_cfg.get("rand_augment_m", 9)
    hp = augment_cfg.get("horizontal_flip_p", 0.5)

    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=rc_scale, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=hp),
        transforms.RandomRotation(rot, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=cj * 0.25),
        transforms.RandAugment(num_ops=n, magnitude=m),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])


def build_eval_transform(img_size: int) -> transforms.Compose:
    resize = int(round(img_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load_class_mapping(csv_path: Path) -> Dict[int, str]:
    """Read class_mapping.csv as {label_int: class_name}. Handles UTF-8 BOM."""
    mapping: Dict[int, str] = {}
    # utf-8-sig transparently strips a BOM if present (Windows-saved CSVs often have one).
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row["label"])] = row["class_name"]
    return mapping


def _verify_class_alignment(image_folder: ImageFolder, mapping_csv: Path) -> None:
    """Ensure ImageFolder's class ordering matches the official class_mapping.csv."""
    if not mapping_csv.exists():
        return
    expected = _load_class_mapping(mapping_csv)
    for cls, idx in image_folder.class_to_idx.items():
        if expected.get(idx) != cls:
            raise RuntimeError(
                f"Class index mismatch: ImageFolder maps '{cls}'->{idx} but "
                f"class_mapping.csv expects label {idx} -> '{expected.get(idx)}'. "
                f"Make sure the dataset directory hasn't been renamed."
            )


def build_datasets(data_root: str | Path, img_size: int, augment_cfg: Dict):
    """Returns (train_ds, val_ds, test_ds, class_names)."""
    data_root = Path(data_root)
    train_tf = build_train_transform(img_size, augment_cfg)
    eval_tf = build_eval_transform(img_size)

    train_ds = ImageFolder(str(data_root / "train"), transform=train_tf)
    val_ds = ImageFolder(str(data_root / "val"), transform=eval_tf)
    test_ds = ImageFolder(str(data_root / "test"), transform=eval_tf)

    mapping_csv = data_root / "class_mapping.csv"
    _verify_class_alignment(train_ds, mapping_csv)

    if train_ds.class_to_idx != val_ds.class_to_idx or train_ds.class_to_idx != test_ds.class_to_idx:
        raise RuntimeError("class_to_idx differs between splits; check your data folder.")

    class_names: List[str] = [c for c, _ in sorted(train_ds.class_to_idx.items(), key=lambda x: x[1])]
    return train_ds, val_ds, test_ds, class_names


def build_dataloaders(
    data_root: str | Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
    augment_cfg: Dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_ds, val_ds, test_ds, class_names = build_datasets(data_root, img_size, augment_cfg)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, test_loader, class_names


def get_test_samples(data_root: str | Path) -> List[Tuple[str, int]]:
    """Returns (image_path, label_idx) for every test image, in ImageFolder order."""
    test_ds = ImageFolder(str(Path(data_root) / "test"))
    return list(test_ds.samples)
