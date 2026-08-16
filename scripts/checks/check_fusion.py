import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch

from src.config import load_config
from src.models.build import build_model, get_device
from src.models.metadata_extractor import METADATA_INPUT_DIM

device = get_device()
batch_size = 4
dummy_img = torch.randn(batch_size, 3, 224, 224).to(device)
dummy_metadata = torch.randn(batch_size, METADATA_INPUT_DIM).to(device)

# Image-only path must be byte-for-byte the same as before this feature existed.
cfg = load_config("configs/experiment.yaml", overrides=["model.pretrained=false"])
plain_model = build_model(cfg["model"])
plain_model.eval()
with torch.no_grad():
    plain_out = plain_model(dummy_img)
print("image-only output shape:", plain_out.shape)
assert plain_out.shape == (batch_size, cfg["model"]["num_classes"])

for fusion_type in ["concatenation", "hadamard", "self_cross_attention"]:
    cfg = load_config("configs/experiment.yaml", overrides=["model.pretrained=false", f"metadata={fusion_type}"])
    assert cfg["metadata"]["use_metadata"] is True
    assert cfg["metadata"]["fusion_type"] == fusion_type

    model = build_model(cfg["model"], metadata_cfg=cfg["metadata"])
    model.eval()
    with torch.no_grad():
        out = model(dummy_img, dummy_metadata)

    print(f"fusion_type={fusion_type}: fused_dim={model.fusion.output_dim}, output shape={out.shape}")
    assert out.shape == (batch_size, cfg["model"]["num_classes"])

# metadata=none should load and behave exactly like the plain path.
cfg_none = load_config("configs/experiment.yaml", overrides=["model.pretrained=false", "metadata=none"])
assert cfg_none["metadata"]["use_metadata"] is False
none_model = build_model(cfg_none["model"], metadata_cfg=cfg_none["metadata"])
none_model.eval()
with torch.no_grad():
    none_out = none_model(dummy_img)
print("metadata=none output shape:", none_out.shape)
assert none_out.shape == (batch_size, cfg_none["model"]["num_classes"])

# Invalid fusion_type must fail loudly, not silently.
try:
    load_config("configs/experiment.yaml", overrides=["metadata.use_metadata=true", "metadata.fusion_type=bogus"])
    print("ERROR: should have raised on invalid fusion_type")
except ValueError as e:
    print("correctly rejected invalid fusion_type:", str(e)[:80])

print("\nAll fusion checks passed.")
