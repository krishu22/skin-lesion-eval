import argparse
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils.seed import set_seed
from src.data.splits import get_lesion_level_splits, get_cv_fold
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
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotlist overrides, e.g. loss=focal train.mixup.enabled=true run_name=my_run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)

    set_seed(cfg["train"]["seed"], deterministic=cfg["train"].get("deterministic", False))

    cv_cfg = cfg.get("cv", {})
    cv_enabled = cv_cfg.get("enabled", False)

    if cv_enabled:
        fold_index = cv_cfg["fold_index"]
        train_df, val_df, test_df = get_cv_fold(
            cfg["data"], cv_cfg, save_dir=cfg["splits_dir"]
        )
        cfg["run_name"] = f"{cfg['run_name']}_fold{fold_index}"
        cfg["output_dir"] = f"{cfg['output_dir']}_fold{fold_index}"
    else:
        train_df, val_df, test_df = get_lesion_level_splits(
            cfg["data"], save_dir=cfg["splits_dir"]
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

    num_classes = cfg["model"]["num_classes"]
    class_counts = [int((train_df["label"] == c).sum()) for c in range(num_classes)]
    criterion = build_loss(cfg["loss"], class_counts=class_counts)

    checkpoint_path = f"{cfg['output_dir']}/best_model.pth"
    metrics_dir = f"{cfg['output_dir']}/metrics"

    checkpoint_meta = {
        "loss_type": cfg["loss"]["name"],
        "mixup_enabled": cfg["train"].get("mixup", {}).get("enabled", False),
        "cutmix_enabled": cfg["train"].get("cutmix", {}).get("enabled", False),
        "segmentation_enabled": cfg["data"].get("use_segmented", False),
        "dullrazor_enabled": cfg["data"].get("dullrazor", False),
    }

    init_run(cfg)

    metric_for_best = cfg["train"].get("metric_for_best", "balanced_accuracy")
    print(f"\nStarting training on device: {device}\n")
    best_metric = train_model(
        model, train_loader, val_loader, criterion, cfg["train"], device, checkpoint_path,
        checkpoint_meta=checkpoint_meta,
    )
    print(f"\nTraining complete. Best val {metric_for_best}: {best_metric:.4f}")

    print("\nRunning final evaluation on test set...\n")
    test_summary = evaluate_model(
        model, test_loader, device, checkpoint_path, cfg["data"]["classes"], metrics_dir
    )

    finish()
    print("\nRun complete. Test summary:", test_summary)

    if cv_enabled:
        print(
            f"[CV] fold={fold_index} "
            f"test_balanced_accuracy={test_summary['test_balanced_accuracy']:.4f} "
            f"test_macro_f1={test_summary['test_macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()