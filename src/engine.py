"""Per-epoch train / eval inner loops with bf16 autocast."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import AverageMeter, accuracy


def _autocast_ctx(device: torch.device):
    """bf16 autocast on CUDA (A100 supports it natively; no GradScaler needed).

    Falls back to no-op on CPU/MPS.
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    class _NullCtx:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    return _NullCtx()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    grad_clip: Optional[float] = None,
    epoch_label: str = "",
) -> Tuple[float, float, float]:
    """Returns (avg_loss, top1_acc, top5_acc) for the epoch."""
    model.train()
    loss_m, top1_m, top5_m = AverageMeter(), AverageMeter(), AverageMeter()

    pbar = tqdm(loader, desc=f"train {epoch_label}", leave=False, dynamic_ncols=True)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device):
            logits = model(images)
            loss = criterion(logits, targets)

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], grad_clip
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = targets.size(0)
        top1, top5 = accuracy(logits.float().detach(), targets, topk=(1, 5))
        loss_m.update(loss.item(), bs)
        top1_m.update(top1, bs)
        top5_m.update(top5, bs)

        pbar.set_postfix(loss=f"{loss_m.avg:.4f}", top1=f"{top1_m.avg:.2f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return loss_m.avg, top1_m.avg, top5_m.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "val",
) -> Tuple[float, float, float]:
    """Returns (avg_loss, top1_acc, top5_acc)."""
    model.eval()
    loss_m, top1_m, top5_m = AverageMeter(), AverageMeter(), AverageMeter()

    for images, targets in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with _autocast_ctx(device):
            logits = model(images)
            loss = criterion(logits, targets)

        bs = targets.size(0)
        top1, top5 = accuracy(logits.float(), targets, topk=(1, 5))
        loss_m.update(loss.item(), bs)
        top1_m.update(top1, bs)
        top5_m.update(top5, bs)

    return loss_m.avg, top1_m.avg, top5_m.avg
