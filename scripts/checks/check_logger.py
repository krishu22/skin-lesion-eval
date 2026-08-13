import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils.logger import init_run, log, finish

cfg = load_config("configs/experiment.yaml")

run = init_run(cfg)
print("W&B run initialized:", run.name)

# simulate a few fake epochs of logging
for epoch in range(1, 4):
    fake_train_loss = 1.0 / epoch
    fake_val_bal_acc = 0.5 + 0.1 * epoch
    log({"train_loss": fake_train_loss, "val_bal_acc": fake_val_bal_acc}, step=epoch)
    print(f"Logged fake epoch {epoch}")
    time.sleep(1)

finish()
print("Run finished — check your W&B dashboard.")