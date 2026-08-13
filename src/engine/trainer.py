import os
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from tqdm.auto import tqdm
from math import ceil

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.logger import log


def build_optimizer(model, optimizer_cfg):
    name = optimizer_cfg["name"]
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_cfg["lr"],
            weight_decay=optimizer_cfg["weight_decay"],
        )
    raise ValueError(f"Unknown optimizer name: '{name}'")


def _wrap_with_warmup(optimizer, base_sched, scheduler_cfg, steps_per_epoch):
    # Optional linear warmup (specified in steps in config). If provided,
    # convert warmup steps to whole epochs using steps_per_epoch and
    # prepend a LambdaLR warmup using SequentialLR.
    warmup_steps = scheduler_cfg.get("warmup_steps", 0)
    if not (warmup_steps and steps_per_epoch):
        return base_sched

    warmup_epochs = max(1, ceil(warmup_steps / float(steps_per_epoch)))

    from torch.optim.lr_scheduler import LambdaLR, SequentialLR

    def _warmup_lambda(epoch):
        return float(epoch + 1) / float(warmup_epochs) if epoch < warmup_epochs else 1.0

    warmup_sched = LambdaLR(optimizer, lr_lambda=_warmup_lambda)
    return SequentialLR(optimizer, schedulers=[warmup_sched, base_sched], milestones=[warmup_epochs])


def build_scheduler(optimizer, scheduler_cfg, steps_per_epoch=None, max_epochs=None):
    name = scheduler_cfg["name"]

    if name == "cosine_warm_restarts":
        base_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_cfg["T_0"],
            T_mult=scheduler_cfg["T_mult"],
        )
        return _wrap_with_warmup(optimizer, base_sched, scheduler_cfg, steps_per_epoch)

    if name == "cosine":
        # Single smooth decay with no restarts — avoids the periodic LR-jump
        # that cosine_warm_restarts causes, which can trip early stopping.
        t_max = scheduler_cfg.get("T_max", max_epochs)
        base_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return _wrap_with_warmup(optimizer, base_sched, scheduler_cfg, steps_per_epoch)

    raise ValueError(f"Unknown scheduler name: '{name}'")


def run_epoch(model, loader, criterion, optimizer, device, train_mode, epoch_num):
    model.train() if train_mode else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    desc = f"Epoch {epoch_num} [train]" if train_mode else f"Epoch {epoch_num} [val]  "
    pbar = tqdm(loader, desc=desc, leave=False)

    with torch.set_grad_enabled(train_mode):
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            if train_mode:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    # Confusion-aware losses rebuild their cost matrix from the latest
    # validation confusion matrix so next epoch's training reflects it.
    if not train_mode and hasattr(criterion, "update_from_confusion_matrix"):
        cm = confusion_matrix(all_labels, all_preds, labels=range(criterion.num_classes))
        criterion.update_from_confusion_matrix(cm)

    return avg_loss, bal_acc


def train_model(model, train_loader, val_loader, criterion, train_cfg, device, checkpoint_path):
    max_epochs = train_cfg["max_epochs"]
    patience = train_cfg["patience"]

    optimizer = build_optimizer(model, train_cfg["optimizer"])
    # provide steps per epoch so warmup_steps in config can be converted to epochs
    scheduler = build_scheduler(
        optimizer, train_cfg["scheduler"], steps_per_epoch=len(train_loader), max_epochs=max_epochs
    )

    best_bal_acc = -1.0
    epochs_no_improve = 0

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        train_loss, train_bal_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train_mode=True, epoch_num=epoch
        )
        val_loss, val_bal_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train_mode=False, epoch_num=epoch
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_bal_acc": train_bal_acc,
            "val_loss": val_loss,
            "val_bal_acc": val_bal_acc,
            "lr": current_lr,
        }, step=epoch)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_bal_acc={train_bal_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_bal_acc={val_bal_acc:.4f}")

        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best val balanced accuracy: {best_bal_acc:.4f} (checkpoint saved)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    return best_bal_acc