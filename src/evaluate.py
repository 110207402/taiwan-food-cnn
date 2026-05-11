"""Test-set evaluation with optional horizontal-flip TTA."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def _autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    class _NullCtx:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    return _NullCtx()


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device, tta_hflip: bool = False) -> Dict:
    """Run inference and return predictions + probabilities + targets."""
    model.eval()
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for images, targets in tqdm(loader, desc="test", leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        targets_np = targets.numpy()

        with _autocast_ctx(device):
            logits = model(images).float()
            if tta_hflip:
                logits_flip = model(torch.flip(images, dims=[3])).float()
                logits = (logits + logits_flip) / 2.0

        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets_np)

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = probs.argmax(axis=1)
    return {"probs": probs, "y_true": y_true, "y_pred": y_pred}


def topk_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int = 1) -> float:
    topk_pred = np.argsort(-probs, axis=1)[:, :k]
    hits = (topk_pred == y_true[:, None]).any(axis=1).mean()
    return float(hits) * 100.0


def compute_basic_metrics(probs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score
    return {
        "top1": topk_accuracy(probs, y_true, 1),
        "top5": topk_accuracy(probs, y_true, 5),
        "macro_f1": f1_score(y_true, y_pred, average="macro") * 100.0,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted") * 100.0,
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0,
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0,
    }
