"""Model factory + freeze / param-group helpers (timm-based)."""
from __future__ import annotations

from typing import Iterable, List

import timm
import torch.nn as nn


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a CNN backbone with a fresh head sized to num_classes."""
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    return model


def _classifier_module(model: nn.Module) -> nn.Module:
    """Return the classifier submodule (timm exposes get_classifier())."""
    if hasattr(model, "get_classifier"):
        return model.get_classifier()
    if hasattr(model, "fc"):
        return model.fc
    if hasattr(model, "classifier"):
        return model.classifier
    raise AttributeError("Could not locate classifier head on the model")


def freeze_backbone(model: nn.Module) -> None:
    """Freeze everything except the classifier head."""
    head = _classifier_module(model)
    head_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def head_param_ids(model: nn.Module) -> set:
    head = _classifier_module(model)
    return {id(p) for p in head.parameters()}


def split_param_groups(model: nn.Module, head_lr: float, backbone_lr: float, weight_decay: float) -> List[dict]:
    """Differential LR: head_lr for the classifier, backbone_lr for the rest.

    Excludes norm layers and biases from weight decay (standard practice).
    """
    head_ids = head_param_ids(model)
    head_decay, head_nodecay, bb_decay, bb_nodecay = [], [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = id(p) in head_ids
        # No weight decay for biases or norm parameters (1D tensors / typical norm shapes)
        no_decay = p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower()
        if is_head and no_decay:
            head_nodecay.append(p)
        elif is_head:
            head_decay.append(p)
        elif no_decay:
            bb_nodecay.append(p)
        else:
            bb_decay.append(p)

    groups = []
    if head_decay:
        groups.append({"params": head_decay, "lr": head_lr, "weight_decay": weight_decay, "tag": "head_decay"})
    if head_nodecay:
        groups.append({"params": head_nodecay, "lr": head_lr, "weight_decay": 0.0, "tag": "head_nodecay"})
    if bb_decay:
        groups.append({"params": bb_decay, "lr": backbone_lr, "weight_decay": weight_decay, "tag": "backbone_decay"})
    if bb_nodecay:
        groups.append({"params": bb_nodecay, "lr": backbone_lr, "weight_decay": 0.0, "tag": "backbone_nodecay"})
    return groups


def head_only_param_groups(model: nn.Module, head_lr: float, weight_decay: float) -> List[dict]:
    """Param groups for stage-1 (head only)."""
    head = _classifier_module(model)
    decay, nodecay = [], []
    for name, p in head.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1 or name.endswith("bias")
        (nodecay if no_decay else decay).append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": head_lr, "weight_decay": weight_decay, "tag": "head_decay"})
    if nodecay:
        groups.append({"params": nodecay, "lr": head_lr, "weight_decay": 0.0, "tag": "head_nodecay"})
    return groups
