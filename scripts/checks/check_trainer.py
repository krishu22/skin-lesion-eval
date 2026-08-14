import torch
from torch.utils.data import DataLoader, Subset

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.splits import get_lesion_level_splits
from src.data.transforms import build_train_transform, build_eval_transform
from src.data.dataset import HAM10000Dataset
from src.models.build import build_model, get_device
from src.losses.build import build_loss
from src.engine.trainer import train_model
from src.utils.logger import init_run, finish

cfg = load_config("configs/experiment.yaml")

train_df, val_df, test_df = get_lesion_level_splits(
    cfg["data"], save_dir=cfg["splits_dir"]
)

train_ds = HAM10000Dataset(train_df, build_train_transform(cfg["data"]))
val_ds = HAM10000Dataset(val_df, build_eval_transform(cfg["data"]))

# tiny subsets so this finishes in a couple minutes on CPU
train_ds_small = Subset(train_ds, range(32))
val_ds_small = Subset(val_ds, range(16))

train_loader = DataLoader(train_ds_small, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds_small, batch_size=8, shuffle=False)

model = build_model(cfg["model"])
device = get_device()
criterion = build_loss(cfg["loss"])

# override config for a fast smoke test only
cfg["train"]["max_epochs"] = 2
cfg["train"]["patience"] = 2

init_run(cfg)
best_bal_acc = train_model(
    model, train_loader, val_loader, criterion, cfg["train"], device,
    checkpoint_path="outputs/baseline_smoke_test/checkpoints/smoke_test.pt"
)
finish()

print("Smoke test complete. Best val balanced accuracy:", best_bal_acc)