"""Quick sanity check for MixUp/CutMix + generic loss-blending wrapper.

Runs a few dummy batches through mixup_data, cutmix_data, and
mixup_cutmix_criterion to confirm output shapes and gradient flow,
without needing the real dataset/model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from src.augment.mixup_cutmix import (
    mixup_data,
    cutmix_data,
    mixup_cutmix_criterion,
    apply_batch_augmentation,
    select_batch_mode,
)
from src.losses.focal import FocalLoss
from src.losses.confusion_aware import ConfusionAwareCELoss
from src.losses.logit_adjusted import LogitAdjustedCrossEntropy

torch.manual_seed(0)

BATCH_SIZE, NUM_CLASSES, IMG_SIZE = 8, 7, 224
device = torch.device("cpu")

toy_model = nn.Sequential(
    nn.Conv2d(3, 4, kernel_size=7, stride=4),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(4, NUM_CLASSES),
)


def make_batch():
    imgs = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, requires_grad=False)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    return imgs, labels


def check(name, mixed_images, targets_a, targets_b, lam, criterion):
    assert mixed_images.shape == (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE), f"{name}: bad image shape {mixed_images.shape}"
    assert targets_a.shape == (BATCH_SIZE,) and targets_b.shape == (BATCH_SIZE,), f"{name}: bad target shape"
    assert 0.0 <= lam <= 1.0, f"{name}: lam out of range: {lam}"

    toy_model.zero_grad()
    logits = toy_model(mixed_images)
    loss = mixup_cutmix_criterion(criterion, logits, targets_a, targets_b, lam)
    assert loss.dim() == 0, f"{name}: loss is not scalar"
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in toy_model.parameters() if p.grad is not None]
    assert all(g == g for g in grad_norms), f"{name}: NaN gradient"
    assert sum(grad_norms) > 0, f"{name}: no gradient flowed"

    print(f"[OK] {name:8s} lam={lam:.4f} loss={loss.item():.4f} grad_norm_sum={sum(grad_norms):.4f}")


criterion = nn.CrossEntropyLoss()

print("== mixup_data ==")
for _ in range(3):
    imgs, labels = make_batch()
    mixed, ta, tb, lam = mixup_data(imgs, labels, alpha=0.4, device=device)
    check("mixup", mixed, ta, tb, lam, criterion)

print("\n== cutmix_data ==")
for _ in range(3):
    imgs, labels = make_batch()
    mixed, ta, tb, lam = cutmix_data(imgs, labels, alpha=1.0, device=device)
    check("cutmix", mixed, ta, tb, lam, criterion)

print("\n== alpha<=0 -> lam==1.0 (no-op blend) ==")
imgs, labels = make_batch()
mixed, ta, tb, lam = mixup_data(imgs, labels, alpha=0.0, device=device)
assert lam == 1.0
assert torch.equal(mixed, imgs)
print("[OK] mixup alpha=0 leaves images/lam unchanged")

print("\n== select_batch_mode: mutual exclusivity + batch_prob split over many draws ==")
mixup_cfg = {"enabled": True, "alpha": 0.2, "batch_prob": 0.3}
cutmix_cfg = {"enabled": True, "alpha": 1.0, "batch_prob": 0.3}
counts = {"none": 0, "mixup": 0, "cutmix": 0}
N = 20000
for _ in range(N):
    counts[select_batch_mode(mixup_cfg, cutmix_cfg)] += 1
print(f"counts over {N} draws: {counts} (expect ~30% mixup, ~30% cutmix, ~40% none)")
assert abs(counts["mixup"] / N - 0.3) < 0.03
assert abs(counts["cutmix"] / N - 0.3) < 0.03

print("\n== apply_batch_augmentation end-to-end (mimics trainer.run_epoch call site) ==")
for _ in range(5):
    imgs, labels = make_batch()
    aug_imgs, ta, tb, lam, mode = apply_batch_augmentation(imgs, labels, mixup_cfg, cutmix_cfg, device)
    assert mode in ("none", "mixup", "cutmix")
    assert torch.equal(ta, labels), "targets_a must always be the original hard labels"
    check(f"e2e:{mode}", aug_imgs, ta, tb, lam, criterion)

print("\n== both disabled -> always mode='none', loss == plain criterion(logits, labels) ==")
imgs, labels = make_batch()
off_cfg_mixup = {"enabled": False, "alpha": 0.2, "batch_prob": 1.0}
off_cfg_cutmix = {"enabled": False, "alpha": 1.0, "batch_prob": 1.0}
aug_imgs, ta, tb, lam, mode = apply_batch_augmentation(imgs, labels, off_cfg_mixup, off_cfg_cutmix, device)
assert mode == "none" and lam == 1.0
assert torch.equal(aug_imgs, imgs)
logits = toy_model(aug_imgs)
blended = mixup_cutmix_criterion(criterion, logits, ta, tb, lam)
plain = criterion(logits, labels)
assert torch.allclose(blended, plain), "disabled case must equal plain criterion(logits, labels)"
print("[OK] disabled mixup+cutmix reduces exactly to plain criterion(logits, labels)")

print("\n== all existing loss classes work unmodified through the wrapper ==")
class_counts = [10] * NUM_CLASSES
real_criteria = {
    "focal": FocalLoss(gamma=2.0),
    "confusion_aware_ce": ConfusionAwareCELoss(num_classes=NUM_CLASSES),
    "logit_adjusted_ce": LogitAdjustedCrossEntropy(class_counts=class_counts, tau=1.0),
}
for loss_name, crit in real_criteria.items():
    imgs, labels = make_batch()
    mixed, ta, tb, lam = mixup_data(imgs, labels, alpha=0.4, device=device)
    check(f"loss:{loss_name}", mixed, ta, tb, lam, crit)

print("\nAll sanity checks passed.")
