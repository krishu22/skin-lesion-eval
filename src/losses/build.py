import torch
import torch.nn as nn

from src.losses.focal import FocalLoss


def _class_balanced_weights(class_counts, beta):
    """Effective-number-of-samples weights from the Class-Balanced Loss paper.

    weight_c = (1 - beta) / (1 - beta^n_c), rescaled to sum to num_classes.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    effective_num = 1.0 - torch.pow(torch.tensor(float(beta)), counts)
    weights = (1.0 - beta) / effective_num
    return weights / weights.sum() * len(counts)


def build_loss(loss_cfg, class_counts=None):
    name = loss_cfg["name"]
    params = dict(loss_cfg.get("params", {}))

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(**params)

    if name == "focal":
        return FocalLoss(**params)

    if name == "cb_focal":
        if class_counts is None:
            raise ValueError("cb_focal loss requires class_counts")
        beta = params.pop("beta", 0.999)
        weights = _class_balanced_weights(class_counts, beta)
        return FocalLoss(alpha=weights, **params)

    raise ValueError(f"Unknown loss name: '{name}'")