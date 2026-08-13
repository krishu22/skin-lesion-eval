import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.losses.build import build_loss

cfg = load_config("configs/experiment.yaml")
criterion = build_loss(cfg["loss"])

print("Loss type:", type(criterion).__name__)

# Fake a batch: 4 samples, 7 class logits each, plus fake true labels
dummy_logits = torch.randn(4, 7)
dummy_labels = torch.tensor([0, 3, 6, 1])

loss_value = criterion(dummy_logits, dummy_labels)
print("Loss value:", loss_value.item())