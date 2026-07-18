import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config

cfg = load_config("configs/stage1_baseline.yaml")
print(cfg["data"]["image_size"])       # should print 224
print(cfg["train"]["optimizer"]["lr"]) # should print 0.0001
print(cfg["run_name"])                 # should print stage1_baseline_ce_swin