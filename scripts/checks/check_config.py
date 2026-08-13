import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config

cfg = load_config("configs/experiment.yaml")
print(cfg["data"]["image_size"])       # should print 224
print(cfg["train"]["optimizer"]["lr"]) # should print 2e-05
print(cfg["run_name"])                 # should print baseline