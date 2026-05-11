"""Two-stage transfer-learning trainer with early stopping."""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from .data import build_dataloaders
from .engine import evaluate, train_one_epoch
from .models import build_model, freeze_backbone, head_only_param_groups, split_param_groups, unfreeze_all
from .utils import count_params, device_info, save_checkpoint, save_json, set_seed, setup_logger


def _cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda=fn)


def _plot_curves(log_rows, save_path: Path) -> None:
    if not log_rows:
        return
    epochs = [r["epoch"] for r in log_rows]
    tr_loss = [r["train_loss"] for r in log_rows]
    va_loss = [r["val_loss"] for r in log_rows]
    tr_acc = [r["train_top1"] for r in log_rows]
    va_acc = [r["val_top1"] for r in log_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, tr_loss, label="train")
    axes[0].plot(epochs, va_loss, label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, tr_acc, label="train")
    axes[1].plot(epochs, va_acc, label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("top-1 (%)"); axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _write_log_csv(log_rows, path: Path) -> None:
    if not log_rows:
        return
    fieldnames = list(log_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(log_rows)


def train(cfg: Dict) -> Dict:
    """Run the full two-stage training pipeline. Returns a summary dict."""
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(cfg["name"], log_file=out_dir / "train.log")
    logger.info(f"Device: {device_info()}")

    set_seed(cfg["train"]["seed"])

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_root=cfg["data"]["root"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment_cfg=cfg["augment"],
    )
    logger.info(f"train batches={len(train_loader)}  val={len(val_loader)}  test={len(test_loader)}")
    logger.info(f"classes={len(class_names)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]["name"], cfg["data"]["num_classes"], cfg["model"]["pretrained"])
    model.to(device)
    pc = count_params(model)
    logger.info(f"model={cfg['model']['name']}  params total={pc['total']:,}  trainable={pc['trainable']:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])

    log_rows = []
    best_val_top1 = -1.0
    best_epoch = -1
    epochs_since_best = 0
    global_epoch = 0

    # -------- Stage 1: head only --------
    s1_epochs = cfg["train"]["stage1_epochs"]
    if s1_epochs > 0:
        logger.info(f"=== Stage 1: head only, {s1_epochs} epochs ===")
        freeze_backbone(model)
        pc = count_params(model)
        logger.info(f"  trainable params: {pc['trainable']:,}")

        groups = head_only_param_groups(
            model, head_lr=cfg["train"]["stage1_lr_head"], weight_decay=cfg["train"]["weight_decay"]
        )
        opt1 = torch.optim.AdamW(groups)
        total_steps = s1_epochs * len(train_loader)
        sched1 = _cosine_with_warmup(opt1, warmup_steps=max(1, len(train_loader)), total_steps=total_steps)

        for e in range(1, s1_epochs + 1):
            global_epoch += 1
            t0 = time.time()
            tl, ttop1, ttop5 = train_one_epoch(model, train_loader, opt1, criterion, device,
                                               scheduler=sched1, grad_clip=cfg["train"].get("grad_clip"),
                                               epoch_label=f"s1 {e}/{s1_epochs}")
            vl, vtop1, vtop5 = evaluate(model, val_loader, criterion, device, desc="val")
            dt = time.time() - t0
            lr_now = opt1.param_groups[0]["lr"]
            logger.info(
                f"[s1 e{e}/{s1_epochs}] train_loss={tl:.4f} top1={ttop1:.2f}  "
                f"val_loss={vl:.4f} top1={vtop1:.2f} top5={vtop5:.2f}  lr={lr_now:.2e}  {dt:.1f}s"
            )
            log_rows.append({
                "epoch": global_epoch, "stage": 1,
                "train_loss": tl, "train_top1": ttop1, "train_top5": ttop5,
                "val_loss": vl, "val_top1": vtop1, "val_top5": vtop5,
                "lr": lr_now, "epoch_time_s": dt,
            })

            if vtop1 > best_val_top1:
                best_val_top1 = vtop1
                best_epoch = global_epoch
                epochs_since_best = 0
                save_checkpoint(
                    {"model": model.state_dict(), "class_names": class_names,
                     "cfg": cfg, "epoch": global_epoch, "val_top1": vtop1},
                    out_dir / "best.pth",
                )
                logger.info(f"  ✓ new best (val_top1={vtop1:.2f})")
            else:
                epochs_since_best += 1

    # -------- Stage 2: full fine-tune --------
    s2_epochs = cfg["train"]["stage2_epochs"]
    logger.info(f"=== Stage 2: full fine-tune, {s2_epochs} epochs ===")
    unfreeze_all(model)
    pc = count_params(model)
    logger.info(f"  trainable params: {pc['trainable']:,}")

    groups2 = split_param_groups(
        model,
        head_lr=cfg["train"]["stage2_lr_head"],
        backbone_lr=cfg["train"]["stage2_lr_backbone"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    opt2 = torch.optim.AdamW(groups2)
    total_steps2 = s2_epochs * len(train_loader)
    warmup_steps2 = cfg["train"]["warmup_epochs"] * len(train_loader)
    sched2 = _cosine_with_warmup(opt2, warmup_steps=warmup_steps2, total_steps=total_steps2)

    patience = cfg["train"]["early_stop_patience"]
    early_stopped = False

    for e in range(1, s2_epochs + 1):
        global_epoch += 1
        t0 = time.time()
        tl, ttop1, ttop5 = train_one_epoch(model, train_loader, opt2, criterion, device,
                                           scheduler=sched2, grad_clip=cfg["train"].get("grad_clip"),
                                           epoch_label=f"s2 {e}/{s2_epochs}")
        vl, vtop1, vtop5 = evaluate(model, val_loader, criterion, device, desc="val")
        dt = time.time() - t0
        lr_head = opt2.param_groups[0]["lr"]
        logger.info(
            f"[s2 e{e}/{s2_epochs}] train_loss={tl:.4f} top1={ttop1:.2f}  "
            f"val_loss={vl:.4f} top1={vtop1:.2f} top5={vtop5:.2f}  lr_head={lr_head:.2e}  {dt:.1f}s"
        )
        log_rows.append({
            "epoch": global_epoch, "stage": 2,
            "train_loss": tl, "train_top1": ttop1, "train_top5": ttop5,
            "val_loss": vl, "val_top1": vtop1, "val_top5": vtop5,
            "lr": lr_head, "epoch_time_s": dt,
        })

        if vtop1 > best_val_top1:
            best_val_top1 = vtop1
            best_epoch = global_epoch
            epochs_since_best = 0
            save_checkpoint(
                {"model": model.state_dict(), "class_names": class_names,
                 "cfg": cfg, "epoch": global_epoch, "val_top1": vtop1},
                out_dir / "best.pth",
            )
            logger.info(f"  ✓ new best (val_top1={vtop1:.2f})")
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                logger.info(f"  ✗ early stopping (no improvement for {patience} epochs)")
                early_stopped = True
                break

    _write_log_csv(log_rows, out_dir / "train_log.csv")
    _plot_curves(log_rows, out_dir / "train_curves.png")

    summary = {
        "best_val_top1": best_val_top1,
        "best_epoch": best_epoch,
        "total_epochs_run": global_epoch,
        "early_stopped": early_stopped,
        "checkpoint": str(out_dir / "best.pth"),
        "model": cfg["model"]["name"],
        "params_total": pc["total"],
    }
    save_json(summary, out_dir / "train_summary.json")
    logger.info(f"DONE. best val_top1={best_val_top1:.2f} @ epoch {best_epoch}")
    return summary
