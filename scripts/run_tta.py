import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.splits import get_lesion_level_splits
from src.data.dataset import HAM10000Dataset
from src.data.hair_removal import DullRazor
from src.models.build import build_model, get_device
from src.engine.evaluator import _run_inference, compute_metrics

# Each entry is (name, extra deterministic ops applied after resize, before
# ToTensor/Normalize) — same 8 views for every test image, no random sampling.
TTA_VARIANTS = [
    ("center_crop", []),
    ("h_flip", [transforms.RandomHorizontalFlip(p=1.0)]),
    ("v_flip", [transforms.RandomVerticalFlip(p=1.0)]),
    ("rotate_2deg", [transforms.RandomRotation((2, 2))]),
    ("rotate_3deg", [transforms.RandomRotation((3, 3))]),
    ("color_jitter", [transforms.ColorJitter(brightness=0.2, contrast=0.2)]),
    ("h_flip_color_jitter", [transforms.RandomHorizontalFlip(p=1.0), transforms.ColorJitter(brightness=0.2, contrast=0.2)]),
    ("v_flip_color_jitter", [transforms.RandomVerticalFlip(p=1.0), transforms.ColorJitter(brightness=0.2, contrast=0.2)]),
]


def _build_tta_transform(data_cfg, extra_ops):
    image_size = data_cfg["image_size"]
    norm = data_cfg["normalize"]

    transform_list = []
    if data_cfg.get("dullrazor", False):
        transform_list.append(DullRazor())

    transform_list.append(transforms.Resize((image_size, image_size)))
    transform_list.extend(extra_ops)
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=norm["mean"], std=norm["std"]))

    return transforms.Compose(transform_list)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to top-level yaml config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a saved best_model.pth")
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="If --checkpoint is omitted, load outputs/<run-name>/best_model.pth",
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

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        run_name = args.run_name or cfg["run_name"]
        checkpoint_path = f"outputs/{run_name}/best_model.pth"

    print(f"Loading checkpoint: {checkpoint_path}")

    _, _, test_df = get_lesion_level_splits(cfg["data"], save_dir=cfg["splits_dir"])

    device = get_device()
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    batch_size = cfg["train"]["batch_size"]
    classes = cfg["data"]["classes"]

    variant_probs = []
    all_labels = None
    for name, extra_ops in TTA_VARIANTS:
        transform = _build_tta_transform(cfg["data"], extra_ops)
        loader = DataLoader(
            HAM10000Dataset(test_df, transform),
            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        )
        probs, _, labels = _run_inference(model, loader, device)
        variant_probs.append(probs)
        if all_labels is None:
            all_labels = labels
        print(f"[TTA] variant={name} done ({len(probs)} images)")

    avg_probs = np.mean(variant_probs, axis=0)
    tta_preds = avg_probs.argmax(axis=1)

    output_dir = f"{cfg['output_dir']}/tta_metrics"
    summary = compute_metrics(avg_probs, tta_preds, all_labels, classes, output_dir=output_dir, prefix="tta")
    print("\nTTA run complete. Summary:", summary)


if __name__ == "__main__":
    main()
