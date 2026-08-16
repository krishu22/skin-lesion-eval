import timm
import torch.nn as nn

from src.models.metadata_extractor import (
    MetadataFeatureExtraction,
    METADATA_INPUT_DIM,
    METADATA_OUTPUT_DIM,
)
from src.models.fusion import FuseImageAndMetadata


class MultimodalLesionClassifier(nn.Module):
    """
    Swin image backbone + the existing MetadataFeatureExtraction module,
    combined via a configurable fusion module, feeding a linear classifier head.

        image (B,3,H,W) --backbone-->        (B, image_dim)      [768 for swin_tiny]
        metadata (B,13) --MetadataFeatureExtraction--> (B, 256)
        (img_feat, meta_feat) --FuseImageAndMetadata(fusion_type)--> (B, fused_dim)
        fused --Dropout+Linear--> (B, num_classes)

    fused_dim depends on fusion_type (see FuseImageAndMetadata's docstring)
    and is read off the fusion module rather than hardcoded here, so the
    classifier head adapts automatically to whichever backbone/fusion_type
    is configured.
    """

    def __init__(self, model_cfg, fusion_type):
        super().__init__()
        self.backbone = timm.create_model(
            model_cfg["name"],
            pretrained=model_cfg["pretrained"],
            num_classes=0,  # drop the classification head; forward() returns pooled features
            drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        )
        image_dim = self.backbone.num_features

        self.metadata_encoder = MetadataFeatureExtraction(input_dim=METADATA_INPUT_DIM)
        self.fusion = FuseImageAndMetadata(image_dim, METADATA_OUTPUT_DIM, fusion_type)

        dropout = model_cfg.get("dropout", 0.0)
        head = nn.Linear(self.fusion.output_dim, model_cfg["num_classes"])
        self.classifier = nn.Sequential(nn.Dropout(dropout), head) if dropout > 0 else head

    def forward(self, img, metadata):
        img_features = self.backbone(img)  # (B, image_dim)
        meta_features = self.metadata_encoder(metadata)  # (B, 256)
        fused = self.fusion(img_features, meta_features)
        return self.classifier(fused)
