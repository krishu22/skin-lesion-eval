from torch.utils.data import DataLoader, Subset

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.splits import get_lesion_level_splits
from src.data.transforms import build_eval_transform
from src.data.dataset import HAM10000Dataset
from src.models.build import build_model, get_device
from src.engine.evaluator import evaluate_model
from src.utils.logger import init_run, finish

cfg = load_config("configs/experiment.yaml")

train_df, val_df, test_df = get_lesion_level_splits(
    cfg["data"], save_dir="outputs/baseline_smoke_test/splits"
)

test_ds = HAM10000Dataset(test_df, build_eval_transform(cfg["data"]))
test_ds_small = Subset(test_ds, range(16))
test_loader = DataLoader(test_ds_small, batch_size=8, shuffle=False)

model = build_model(cfg["model"])
device = get_device()

init_run(cfg)
summary = evaluate_model(
    model, test_loader, device,
    checkpoint_path="outputs/baseline_smoke_test/checkpoints/smoke_test.pt",
    classes=cfg["data"]["classes"],
    output_dir="outputs/baseline_smoke_test/metrics_smoke_test",
)
finish()

print("\nEvaluator smoke test complete.")
print(summary)