import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.splits import get_lesion_level_splits
from src.data.transforms import build_train_transform, build_eval_transform
from src.data.dataset import HAM10000Dataset

cfg = load_config("configs/experiment.yaml")

train_df, val_df, test_df = get_lesion_level_splits(
    cfg["data"], save_dir=cfg["splits_dir"]
)

train_ds = HAM10000Dataset(train_df, build_train_transform(cfg["data"]))
val_ds = HAM10000Dataset(val_df, build_eval_transform(cfg["data"]))

print("Train dataset size:", len(train_ds))
print("Val dataset size:", len(val_ds))

img, label = train_ds[0]
print("Sample image tensor shape:", img.shape)
print("Sample label:", label)

# confirm same index gives different tensor each call (train = augmented, random)
img_a, _ = train_ds[0]
img_b, _ = train_ds[0]
print("Same index, train set — tensors identical?", (img_a == img_b).all().item())

# confirm eval set gives IDENTICAL tensor each call (deterministic)
img_c, _ = val_ds[0]
img_d, _ = val_ds[0]
print("Same index, val set — tensors identical?", (img_c == img_d).all().item())