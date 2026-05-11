"""Error analysis: confusion matrix, per-class metrics, misclassified samples."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0, digits=4,
    )
    rows = []
    for cls in class_names:
        r = report[cls]
        rows.append({
            "class": cls,
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1-score"],
            "support": int(r["support"]),
        })
    df = pd.DataFrame(rows).sort_values("f1", ascending=True).reset_index(drop=True)
    return df


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Path,
    normalize: bool = True,
    title: str = "Confusion matrix",
) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
    else:
        cm_norm = cm.astype(float)

    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(
        cm_norm, annot=False, cmap="Blues", vmin=0, vmax=1 if normalize else None,
        xticklabels=class_names, yticklabels=class_names, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title} ({'row-normalized' if normalize else 'counts'})")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return cm


def top_confused_pairs(cm: np.ndarray, class_names: List[str], top_k: int = 10) -> pd.DataFrame:
    """Find the top-k most-confused (unordered) class pairs by total off-diagonal mass."""
    n = cm.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score = int(cm[i, j] + cm[j, i])
            if score > 0:
                pairs.append({
                    "class_a": class_names[i],
                    "class_b": class_names[j],
                    "a_to_b": int(cm[i, j]),
                    "b_to_a": int(cm[j, i]),
                    "total": score,
                })
    df = pd.DataFrame(pairs).sort_values("total", ascending=False).head(top_k).reset_index(drop=True)
    return df


def visualize_misclassified(
    image_paths: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    save_path: Path,
    worst_n_classes: int = 5,
    samples_per_class: int = 8,
    title: str = "",
) -> None:
    """For each of the N classes with the lowest recall, show up to K misclassified examples."""
    n_classes = len(class_names)
    per_class_recall = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            per_class_recall.append((c, 1.0, 0))
            continue
        recall = (y_pred[mask] == c).mean()
        per_class_recall.append((c, float(recall), int(mask.sum())))

    per_class_recall.sort(key=lambda x: x[1])
    worst = [x for x in per_class_recall if x[2] > 0][:worst_n_classes]

    fig, axes = plt.subplots(len(worst), samples_per_class, figsize=(2.0 * samples_per_class, 2.2 * len(worst)))
    if len(worst) == 1:
        axes = np.array([axes])

    for row_idx, (cls_idx, recall, support) in enumerate(worst):
        mis_mask = (y_true == cls_idx) & (y_pred != cls_idx)
        mis_indices = np.where(mis_mask)[0]
        # show the ones the model was most-confident-wrong about first
        if len(mis_indices) > 0:
            wrong_conf = probs[mis_indices, y_pred[mis_indices]]
            order = np.argsort(-wrong_conf)
            mis_indices = mis_indices[order]

        for col in range(samples_per_class):
            ax = axes[row_idx, col]
            ax.axis("off")
            if col == 0:
                ax.text(
                    -0.1, 0.5, f"{class_names[cls_idx]}\nrecall={recall*100:.1f}%  n={support}",
                    transform=ax.transAxes, ha="right", va="center", fontsize=9,
                )
            if col < len(mis_indices):
                idx = mis_indices[col]
                try:
                    img = Image.open(image_paths[idx]).convert("RGB")
                    ax.imshow(img)
                    pred_cls = class_names[y_pred[idx]]
                    conf = probs[idx, y_pred[idx]] * 100
                    ax.set_title(f"→ {pred_cls}\n({conf:.0f}%)", fontsize=8, color="crimson")
                except Exception:
                    pass

    fig.suptitle(title or f"Worst {len(worst)} classes — misclassified samples", fontsize=12)
    fig.tight_layout(rect=(0.04, 0, 1, 0.97))
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_metrics_table(metrics: Dict[str, float], path: Path) -> None:
    df = pd.DataFrame([metrics]).T
    df.columns = ["value"]
    df.index.name = "metric"
    df.to_csv(path)
