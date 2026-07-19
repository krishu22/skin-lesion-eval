import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from src.config import load_config
from src.models.build import build_model, get_device

cfg = load_config("configs/stage1_baseline.yaml")

model = build_model(cfg["model"])
device = get_device()

print("Device:", device)
print("Model type:", type(model).__name__)

# Fake a batch of 2 images, exactly the shape the real pipeline produces
dummy_input = torch.randn(2, 3, 224, 224).to(device)

model.eval()
with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)  # should be torch.Size([2, 7])