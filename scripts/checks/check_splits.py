import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.splits import get_lesion_level_splits

cfg = load_config("configs/experiment.yaml")
train_df, val_df, test_df = get_lesion_level_splits(
    cfg["data"], save_dir=cfg["splits_dir"]
)

group_col = cfg["data"]["split"]["group_col"]

train_lesions = set(train_df[group_col])
val_lesions = set(val_df[group_col])
test_lesions = set(test_df[group_col])

print("Train ∩ Val:", len(train_lesions & val_lesions))
print("Train ∩ Test:", len(train_lesions & test_lesions))
print("Val ∩ Test:", len(val_lesions & test_lesions))