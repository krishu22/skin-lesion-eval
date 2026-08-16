import torch.nn as nn
import torch.nn.functional as F

from src.data.metadata import METADATA_FEATURE_COLUMNS

METADATA_INPUT_DIM = len(METADATA_FEATURE_COLUMNS)  # 13: 2 age + 2 sex + 9 location
METADATA_OUTPUT_DIM = 256  # width of fc2 below; consumed by src/models/fusion.py


class MetadataFeatureExtraction(nn.Module):
    def __init__(self, input_dim=METADATA_INPUT_DIM, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.d1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.d2 = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, input_dim)
        x = self.fc1(x)  # (B, 128)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.d1(x)

        x = self.fc2(x)  # (B, 256)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.d2(x)

        return x  # (B, 256)


def build_metadata_extractor(device, input_dim=METADATA_INPUT_DIM, dropout=0.3):
    model = MetadataFeatureExtraction(input_dim=input_dim, dropout=dropout)
    return model.to(device)
