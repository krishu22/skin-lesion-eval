"""
One-time external validation: evaluate an already-trained HAM10000
checkpoint on the ISIC2019 external set prepared by
scripts/prepare_isic2019_external_val.py (outputs/isic2019_external_val.csv).

Pure inference — no training, no tuning on the result. Reuses the exact
TTA variants/transforms from scripts/run_tta.py and the same metric suite
as the HAM10000 test-set evaluation (src/engine/evaluator.py).

Safe to run multiple times in the same session with different
--checkpoint values (e.g. one per ablation run) — each run's metrics are
written to their own output_dir keyed by the checkpoint path, and nothing
is overwritten between runs.

Prerequisite: download the images with
  bash scripts/download_dataset.sh isic2019
(mirrors `bash scripts/download_dataset.sh ham10000`; populates ./isic2019/
with one folder per class — AK/, BCC/, BKL/, DF/, MEL/, NV/, VASC/, plus
the excluded SCC/, UNK/). --data-root defaults to "isic2019" to match.

Usage:
  python scripts/eval_isic2019_external.py \\
      --config configs/experiment.yaml \\
      --checkpoint outputs/<run_name>/best_model.pth \\
      [--data-root isic2019] \\
      [data=ham10000_segmented ...same overrides used to train the checkpoint...]
"""

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.dataset import HAM10000Dataset
from src.models.build import build_model, get_device
from src.engine.evaluator import _run_inference, compute_metrics
from scripts.run_tta import TTA_VARIANTS, _build_tta_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to top-level yaml config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved best_model.pth")
    parser.add_argument(
        "--csv", type=str, default="outputs/isic2019_external_val.csv",
        help="External val CSV with columns image_id, filepath, label",
    )
    parser.add_argument(
        "--data-root", type=str, default="isic2019",
        help="Directory the CSV's filepath column is relative to (the ISIC2019 dataset root, "
             "as populated by `bash scripts/download_dataset.sh isic2019`)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Same data/loss/model overrides used to train the checkpoint, e.g. data=ham10000_segmented",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)

    print(f"Loading checkpoint: {args.checkpoint}")

    df = pd.read_csv(args.csv)
    data_root = Path(args.data_root)
    df["filepath"] = df["filepath"].apply(lambda p: str(data_root / p))

    device = get_device()
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    batch_size = cfg["train"]["batch_size"]
    classes = cfg["data"]["classes"]

    variant_probs = []
    all_labels = None
    for name, extra_ops in TTA_VARIANTS:
        transform = _build_tta_transform(cfg["data"], extra_ops)
        loader = DataLoader(
            HAM10000Dataset(df, transform),
            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        )
        probs, _, labels = _run_inference(model, loader, device)
        variant_probs.append(probs)
        if all_labels is None:
            all_labels = labels
        print(f"[TTA] variant={name} done ({len(probs)} images)")

    avg_probs = np.mean(variant_probs, axis=0)
    tta_preds = avg_probs.argmax(axis=1)

    checkpoint_key = Path(args.checkpoint).parent.name or Path(args.checkpoint).stem
    output_dir = f"outputs/isic2019_external/{checkpoint_key}"
    summary = compute_metrics(
        avg_probs, tta_preds, all_labels, classes,
        output_dir=output_dir, prefix="isic2019_external",
    )
    print("\nISIC2019 external validation complete. Summary:", summary)


if __name__ == "__main__":
    main()
