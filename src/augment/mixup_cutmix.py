import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def mixup_data(images, targets, alpha, device):
    """Standard MixUp (image-only, no metadata branch).

    lam ~ Beta(alpha, alpha); mixed = lam * images + (1 - lam) * images[perm]
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(images.size(0), device=device)

    mixed_images = lam * images + (1 - lam) * images[index]
    targets_a, targets_b = targets, targets[index]
    return mixed_images, targets_a, targets_b, lam


def _rand_bbox(width, height, lam):
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = int(np.clip(cx - cut_w // 2, 0, width))
    y1 = int(np.clip(cy - cut_h // 2, 0, height))
    x2 = int(np.clip(cx + cut_w // 2, 0, width))
    y2 = int(np.clip(cy + cut_h // 2, 0, height))
    return x1, y1, x2, y2


def cutmix_data(images, targets, alpha, device):
    """CutMix. lam is recomputed from the actual pasted box area, not the raw sampled lam."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(images.size(0), device=device)

    _, _, height, width = images.shape
    x1, y1, x2, y2 = _rand_bbox(width, height, lam)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    box_area = (x2 - x1) * (y2 - y1)
    lam_adjusted = 1.0 - (box_area / (width * height))

    targets_a, targets_b = targets, targets[index]
    return mixed_images, targets_a, targets_b, lam_adjusted


def mixup_cutmix_criterion(criterion, logits, targets_a, targets_b, lam):
    """Generic loss-blending wrapper: works with any criterion(logits, targets) -> scalar.

    Uses only its own arguments (targets_a, targets_b, lam) — never enclosing-scope
    variables of the same name — to avoid the mixed-batch label bug this project hit before.
    """
    return lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)


def select_batch_mode(mixup_cfg, cutmix_cfg):
    """Pick at most one of {mixup, cutmix} for this batch, per their batch_prob.

    batch_prob is treated as this batch's share of the [0, 1) draw, so the two
    are mutually exclusive by construction and never both fire on one batch.
    """
    mixup_enabled = bool(mixup_cfg and mixup_cfg.get("enabled", False))
    cutmix_enabled = bool(cutmix_cfg and cutmix_cfg.get("enabled", False))

    if not mixup_enabled and not cutmix_enabled:
        return "none"

    mixup_prob = mixup_cfg.get("batch_prob", 0.0) if mixup_enabled else 0.0
    cutmix_prob = cutmix_cfg.get("batch_prob", 0.0) if cutmix_enabled else 0.0

    r = random.random()
    if r < mixup_prob:
        return "mixup"
    if r < mixup_prob + cutmix_prob:
        return "cutmix"
    return "none"


def apply_batch_augmentation(images, labels, mixup_cfg, cutmix_cfg, device):
    """Freshly sample and apply MixUp/CutMix (or neither) for one training batch.

    Returns (aug_images, targets_a, targets_b, lam, mode). targets_a is always the
    batch's original (unpermuted) labels, so callers can keep using it for anything
    that needs the true hard labels (e.g. accuracy bookkeeping) unchanged.
    """
    mode = select_batch_mode(mixup_cfg, cutmix_cfg)

    if mode == "mixup":
        aug_images, targets_a, targets_b, lam = mixup_data(images, labels, mixup_cfg["alpha"], device)
    elif mode == "cutmix":
        aug_images, targets_a, targets_b, lam = cutmix_data(images, labels, cutmix_cfg["alpha"], device)
    else:
        aug_images, targets_a, targets_b, lam = images, labels, labels, 1.0

    logger.debug("batch augmentation=%s lam=%.4f", mode, lam)
    return aug_images, targets_a, targets_b, lam, mode
