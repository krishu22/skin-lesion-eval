import argparse
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils.seed import set_seed
from src.data.splits import get_lesion_level_splits
from src.data.transforms import build_train_transform, build_eval_transform
from src.data.dataset import HAM10000Dataset
from src.models.build import build_model, get_device
from src.losses.build import build_loss
from src.engine.trainer import train_model
from src.engine.evaluator import evaluate_model
from src.utils.logger import init_run, finish


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to top-level yaml config")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["train"]["seed"], deterministic=cfg["train"].get("deterministic", False))

    train_df, val_df, test_df = get_lesion_level_splits(
        cfg["data"], save_dir=f"{cfg['output_dir']}/splits"
    )

    train_transform = build_train_transform(cfg["data"])
    eval_transform = build_eval_transform(cfg["data"])

    train_ds = HAM10000Dataset(train_df, train_transform)
    val_ds = HAM10000Dataset(val_df, eval_transform)
    test_ds = HAM10000Dataset(test_df, eval_transform)

    batch_size = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(cfg["model"])
    device = get_device()
    criterion = build_loss(cfg["loss"])

    checkpoint_path = f"{cfg['output_dir']}/checkpoints/best.pt"
    metrics_dir = f"{cfg['output_dir']}/metrics"

    init_run(cfg)

    print(f"\nStarting training on device: {device}\n")
    best_val_bal_acc = train_model(
        model, train_loader, val_loader, criterion, cfg["train"], device, checkpoint_path
    )
    print(f"\nTraining complete. Best val balanced accuracy: {best_val_bal_acc:.4f}")

    print("\nRunning final evaluation on test set...\n")
    test_summary = evaluate_model(
        model, test_loader, device, checkpoint_path, cfg["data"]["classes"], metrics_dir
    )

    finish()
    print("\nRun complete. Test summary:", test_summary)


if __name__ == "__main__":
    main()