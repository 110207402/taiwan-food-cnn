"""End-to-end: train one model, then evaluate + run error analysis on test set.

Usage:
    python -m scripts.run_train --config configs/convnext_tiny.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python -m scripts.run_train` OR `python scripts/run_train.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.analysis import (
    per_class_metrics,
    plot_confusion_matrix,
    save_metrics_table,
    top_confused_pairs,
    visualize_misclassified,
)
from src.data import build_dataloaders, get_test_samples
from src.evaluate import compute_basic_metrics, predict
from src.models import build_model
from src.train import train
from src.utils import load_config, save_json, setup_logger


def run_test_and_analysis(cfg):
    out_dir = Path(cfg["output"]["dir"])
    logger = setup_logger(cfg["name"] + ".test", log_file=out_dir / "test.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild loaders (need test_loader and class_names).
    _, _, test_loader, class_names = build_dataloaders(
        data_root=cfg["data"]["root"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment_cfg=cfg["augment"],
    )

    # Build model and load best checkpoint
    model = build_model(cfg["model"]["name"], cfg["data"]["num_classes"], pretrained=False)
    ckpt = torch.load(out_dir / "best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch')} (val_top1={ckpt.get('val_top1'):.2f})")

    tta = cfg.get("eval", {}).get("tta_hflip", True)
    logger.info(f"TTA hflip: {tta}")

    pred = predict(model, test_loader, device, tta_hflip=tta)
    metrics = compute_basic_metrics(pred["probs"], pred["y_true"], pred["y_pred"])
    logger.info(
        f"TEST  top1={metrics['top1']:.2f}  top5={metrics['top5']:.2f}  "
        f"macro_f1={metrics['macro_f1']:.2f}  weighted_f1={metrics['weighted_f1']:.2f}"
    )

    save_json(metrics, out_dir / "test_metrics.json")
    save_metrics_table(metrics, out_dir / "test_metrics.csv")

    pcm = per_class_metrics(pred["y_true"], pred["y_pred"], class_names)
    pcm.to_csv(out_dir / "per_class_metrics.csv", index=False)
    logger.info("Worst 5 classes by F1:")
    for _, row in pcm.head(5).iterrows():
        logger.info(f"  {row['class']:40s}  P={row['precision']:.3f}  R={row['recall']:.3f}  F1={row['f1']:.3f}  n={row['support']}")

    cm = plot_confusion_matrix(
        pred["y_true"], pred["y_pred"], class_names,
        save_path=out_dir / "confusion_matrix.png",
        normalize=True, title=f"{cfg['name']} confusion matrix",
    )
    plot_confusion_matrix(
        pred["y_true"], pred["y_pred"], class_names,
        save_path=out_dir / "confusion_matrix_counts.png",
        normalize=False, title=f"{cfg['name']} confusion matrix",
    )

    pairs = top_confused_pairs(cm, class_names, top_k=10)
    pairs.to_csv(out_dir / "top_confused_pairs.csv", index=False)
    logger.info("Top 5 confused pairs:")
    for _, row in pairs.head(5).iterrows():
        logger.info(f"  {row['class_a']} ↔ {row['class_b']}  total={row['total']}  "
                    f"(a→b={row['a_to_b']}, b→a={row['b_to_a']})")

    # Misclassified visualization needs file paths in test-loader order.
    samples = get_test_samples(cfg["data"]["root"])
    image_paths = [p for p, _ in samples]
    visualize_misclassified(
        image_paths=image_paths,
        y_true=pred["y_true"], y_pred=pred["y_pred"], probs=pred["probs"],
        class_names=class_names,
        save_path=out_dir / "misclassified_samples.png",
        worst_n_classes=5, samples_per_class=8,
        title=f"{cfg['name']} — worst-class misclassifications",
    )

    # Save raw predictions for later analysis
    import numpy as np
    np.savez_compressed(
        out_dir / "predictions.npz",
        probs=pred["probs"].astype("float32"),
        y_true=pred["y_true"].astype("int32"),
        y_pred=pred["y_pred"].astype("int32"),
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--data-root", default=None, help="Override data.root in config")
    parser.add_argument("--skip-train", action="store_true", help="Only run test+analysis (assumes best.pth exists)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        summary = train(cfg)
        save_json(summary, out_dir / "train_summary.json")

    metrics = run_test_and_analysis(cfg)

    # Combined summary file
    combined = {"config": cfg, "test_metrics": metrics}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print("\n=== Done ===")
    print(f"Outputs at: {out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
