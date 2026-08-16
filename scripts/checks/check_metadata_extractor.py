import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch

from src.config import load_config
from src.data.splits import get_lesion_level_splits
from src.data.transforms import build_train_transform, build_eval_transform
from src.data.dataset import HAM10000Dataset
from src.data.metadata import load_metadata_features, METADATA_FEATURE_COLUMNS
from src.models.build import get_device
from src.models.metadata_extractor import build_metadata_extractor, METADATA_INPUT_DIM

cfg = load_config("configs/experiment.yaml")
splits_dir = cfg["splits_dir"]

train_df, val_df, test_df = get_lesion_level_splits(cfg["data"], save_dir=splits_dir)

train_meta = load_metadata_features("train", splits_dir)
val_meta = load_metadata_features("val", splits_dir)

print("Metadata feature columns:", METADATA_FEATURE_COLUMNS)
print("Metadata input dim:", METADATA_INPUT_DIM)

train_ds = HAM10000Dataset(train_df, build_train_transform(cfg["data"]), metadata_df=train_meta)
val_ds = HAM10000Dataset(val_df, build_eval_transform(cfg["data"]), metadata_df=val_meta)

img, metadata, label = train_ds[0]
print("\nSample image tensor shape:", img.shape)
print("Sample metadata tensor shape:", metadata.shape)
print("Sample label:", label)

# Alignment check: pull metadata straight from the CSV by image_id for a few
# random rows and confirm it exactly matches what the dataset returned.
mismatches = 0
for idx in [0, 5, len(val_ds) // 2, len(val_ds) - 1]:
    row = val_ds.df.iloc[idx]
    _, metadata_from_ds, _ = val_ds[idx]
    expected = torch.tensor(val_meta.loc[row["image_id"]].values, dtype=torch.float32)
    if not torch.equal(metadata_from_ds, expected):
        mismatches += 1
        print(f"MISMATCH at idx={idx}, image_id={row['image_id']}")
print(f"\nAlignment check ({4} samples): {'OK' if mismatches == 0 else f'{mismatches} MISMATCHES'}")

# Forward pass through the metadata extraction module on a real batch.
device = get_device()
extractor = build_metadata_extractor(device)
extractor.eval()

batch_size = 4
metadata_batch = torch.stack([val_ds[i][1] for i in range(batch_size)]).to(device)
with torch.no_grad():
    out = extractor(metadata_batch)

print("\nMetadata extractor input shape:", metadata_batch.shape)
print("Metadata extractor output shape:", out.shape)  # should be (4, 256)

# Dataset without metadata_df should still behave exactly like before (image-only).
plain_ds = HAM10000Dataset(val_df, build_eval_transform(cfg["data"]))
plain_img, plain_label = plain_ds[0]
print("\nBackward-compat (no metadata_df) sample:", plain_img.shape, plain_label)
