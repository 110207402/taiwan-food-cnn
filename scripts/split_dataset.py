"""Create reproducible train/validation/test splits from class directories.

Expected input:
    source/<class_name>/*.{jpg,jpeg,png}

Example:
    python -m scripts.split_dataset --source raw_food101 --output data
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_dataset(
    source: Path,
    output: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Copy images into deterministic, per-class stratified splits."""
    if not source.is_dir():
        raise ValueError(f"source is not a directory: {source}")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train, validation, and test ratios must sum to 1")
    if any(ratio <= 0 for ratio in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("all split ratios must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")

    class_dirs = sorted(path for path in source.iterdir() if path.is_dir() and not path.name.startswith("."))
    if not class_dirs:
        raise ValueError("source contains no class directories")

    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    split_rows: list[dict] = []
    summary_rows: list[dict] = []
    mapping_rows: list[dict] = []
    totals = {"train": 0, "val": 0, "test": 0}

    for label, class_dir in enumerate(class_dirs):
        images = sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if len(images) < 3:
            raise ValueError(f"class '{class_dir.name}' needs at least 3 images")
        rng.shuffle(images)

        train_end = max(1, int(len(images) * train_ratio))
        val_end = train_end + max(1, int(len(images) * val_ratio))
        if val_end >= len(images):
            val_end = len(images) - 1

        assignments = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }
        mapping_rows.append({"label": label, "class_name": class_dir.name})
        summary = {"class_name": class_dir.name}

        for split_name, split_images in assignments.items():
            destination_dir = output / split_name / class_dir.name
            destination_dir.mkdir(parents=True, exist_ok=True)
            summary[split_name] = len(split_images)
            totals[split_name] += len(split_images)

            for image in split_images:
                destination = destination_dir / image.name
                shutil.copy2(image, destination)
                split_rows.append(
                    {
                        "path": destination.relative_to(output).as_posix(),
                        "split": split_name,
                        "label": label,
                        "class_name": class_dir.name,
                    }
                )
        summary_rows.append(summary)

    _write_csv(output / "class_mapping.csv", ["label", "class_name"], mapping_rows)
    _write_csv(output / "dataset_split.csv", ["path", "split", "label", "class_name"], split_rows)
    _write_csv(output / "dataset_summary.csv", ["class_name", "train", "val", "test"], summary_rows)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Directory containing one folder per class")
    parser.add_argument("--output", required=True, type=Path, help="Empty destination directory")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    totals = split_dataset(
        args.source,
        args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print("Created split:", ", ".join(f"{name}={count}" for name, count in totals.items()))


if __name__ == "__main__":
    main()
