import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import load_config
from src.data.transforms import build_train_transform, build_eval_transform
from PIL import Image

cfg = load_config("configs/stage1_baseline.yaml")
train_t = build_train_transform(cfg["data"])
eval_t = build_eval_transform(cfg["data"])

img = Image.open("ham10000/HAM10000_images_part_1/ISIC_0024306.jpg").convert("RGB")

train_tensor = train_t(img)
eval_tensor = eval_t(img)

print(train_tensor)
print(eval_tensor)

print()

print(train_tensor.shape)
print(eval_tensor.shape)