"""
One-time external validation: evaluate an already-trained HAM10000
checkpoint on the ISIC2019 external set prepared by
scripts/prepare_isic2019_external_val.py and
scripts/merge_isic2019_metadata_features.py (outputs/isic2019_metadata.csv
— image_id, filepath, label, and the 13 metadata feature columns, all in
one file).

Pure inference — no training, no tuning on the result. TTA is off by
default (single center-cropped pass, via the same build_eval_transform
used for the HAM10000 test set); pass tta.enabled=true to instead average
over the exact same 8 TTA_VARIANTS/transforms/averaging locked in on
HAM10000 val by scripts/run_tta.py — this script does not retune or add
ISIC2019-specific augmentations. Metrics use the same suite as the
HAM10000 test-set evaluation (src/engine/evaluator.py).

When the checkpoint's config has metadata.use_metadata=true, TTA only
augments the image side: a fresh HAM10000Dataset is built per TTA variant
from the same (df, isic_meta) pair, so every variant looks up the same
metadata row per image_id — the metadata vector for a given image is
identical across all of its augmented copies, never duplicated or
misaligned relative to the image batch.

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
      [tta.enabled=true] \\
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
from src.data.metadata import load_metadata_features_from_path
from src.data.transforms import build_eval_transform
from src.models.build import build_model, get_device
from src.engine.evaluator import _run_inference, compute_metrics
from scripts.run_tta import TTA_VARIANTS, _build_tta_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to top-level yaml config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved best_model.pth")
    parser.add_argument(
        "--csv", type=str, default="outputs/isic2019_metadata.csv",
        help="External val CSV with columns image_id, filepath, label, and (when the "
             "checkpoint's config has metadata.use_metadata=true) the same 13 feature "
             "columns as outputs/splits/{train,val,test}.csv.",
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

    metadata_cfg = cfg.get("metadata", {})
    use_metadata = metadata_cfg.get("use_metadata", False)

    isic_meta = None
    if use_metadata:
        csv_path = Path(args.csv)
        try:
            isic_meta = load_metadata_features_from_path(csv_path, id_column="image_id")
        except ValueError as e:
            raise RuntimeError(
                f"cfg.metadata.use_metadata=true (fusion_type={metadata_cfg.get('fusion_type')}), "
                f"but '{csv_path}' doesn't have the expected metadata columns ({e}). Build it "
                "with scripts/add_isic2019_metadata_features.py / "
                "scripts/merge_isic2019_metadata_features.py, or re-run this checkpoint's "
                "training with metadata=none if you need an image-only checkpoint to evaluate "
                "here instead."
            ) from e

    device = get_device()
    model = build_model(cfg["model"], metadata_cfg=metadata_cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    batch_size = cfg["train"]["batch_size"]
    classes = cfg["data"]["classes"]

    tta_cfg = cfg.get("tta", {})
    use_tta = tta_cfg.get("enabled", False)
    variants = TTA_VARIANTS if use_tta else [("center_crop", None)]

    variant_probs = []
    all_labels = None
    for name, extra_ops in variants:
        transform = _build_tta_transform(cfg["data"], extra_ops) if use_tta else build_eval_transform(cfg["data"])
        loader = DataLoader(
            HAM10000Dataset(df, transform, metadata_df=isic_meta),
            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        )
        probs, _, labels = _run_inference(model, loader, device, use_metadata=use_metadata)
        variant_probs.append(probs)
        if all_labels is None:
            all_labels = labels
        print(f"[{'TTA' if use_tta else 'single-pass'}] variant={name} done ({len(probs)} images)")

    avg_probs = np.mean(variant_probs, axis=0)
    tta_preds = avg_probs.argmax(axis=1)

    checkpoint_key = Path(args.checkpoint).parent.name or Path(args.checkpoint).stem
    output_dir = f"outputs/isic2019_external/{checkpoint_key}"
    summary = compute_metrics(
        avg_probs, tta_preds, all_labels, classes,
        output_dir=output_dir, prefix="isic2019_external",
    )
    print(f"\nISIC2019 external validation complete (tta.enabled={use_tta}). Summary:", summary)


if __name__ == "__main__":
    main()
