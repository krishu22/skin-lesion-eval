import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, log_loss
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.logger import log


def _expected_calibration_error(probs, preds, labels, n_bins=15):
    confidences = probs.max(axis=1)
    accuracies = (preds == labels).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return ece


def _brier_score(probs, labels, num_classes):
    one_hot_labels = np.eye(num_classes)[labels]
    return np.mean(np.sum((probs - one_hot_labels) ** 2, axis=1))


@torch.no_grad()
def _run_inference(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


def compute_metrics(all_probs, all_preds, all_labels, classes, output_dir=None, prefix="test"):
    num_classes = len(classes)

    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(num_classes), zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    specificity = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    try:
        macro_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError as e:
        macro_auc = float("nan")
        print("Could not compute macro-AUC (likely a class missing from test set):", e)

    nll = log_loss(all_labels, all_probs, labels=range(num_classes))
    brier = _brier_score(all_probs, all_labels, num_classes)
    ece = _expected_calibration_error(all_probs, all_preds, all_labels)

    print(f"\n===== {prefix.upper()} SET RESULTS =====")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Macro-AUC: {macro_auc:.4f}")
    print(f"NLL: {nll:.4f}")
    print(f"Brier score: {brier:.4f}")
    print(f"ECE: {ece:.4f}")

    per_class_df = pd.DataFrame({
        "class": classes,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "support": support,
    })
    print("\nPer-class metrics:\n", per_class_df.to_string(index=False))

    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print("\nConfusion matrix (rows=true, cols=predicted):\n", cm_df)

    summary = {
        f"{prefix}_balanced_accuracy": bal_acc,
        f"{prefix}_macro_f1": macro_f1,
        f"{prefix}_macro_auc": macro_auc,
        f"{prefix}_nll": nll,
        f"{prefix}_brier": brier,
        f"{prefix}_ece": ece,
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        per_class_df.to_csv(os.path.join(output_dir, f"{prefix}_per_class_metrics.csv"), index=False)
        cm_df.to_csv(os.path.join(output_dir, f"{prefix}_confusion_matrix.csv"))
        pd.DataFrame([summary]).to_csv(os.path.join(output_dir, f"{prefix}_summary.csv"), index=False)

    return summary


def evaluate_model(model, test_loader, device, checkpoint_path, classes, output_dir):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    all_probs, all_preds, all_labels = _run_inference(model, test_loader, device)
    summary = compute_metrics(all_probs, all_preds, all_labels, classes, output_dir=output_dir, prefix="test")

    log(summary)

    return summary