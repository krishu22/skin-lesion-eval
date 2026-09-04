import torch
import torch.nn as nn


FUSION_TYPES = ("concatenation","hadamard","attention")


class MultimodalAttentionFusion(nn.Module):

    def __init__(self,image_dim,meta_dim,attention_dim=256,num_heads=8,dropout=0.1):
        super().__init__()

        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")

        self.image_projection = nn.Linear(image_dim,attention_dim)

        self.meta_projection = nn.Linear(meta_dim,attention_dim)

        self.attention = nn.MultiheadAttention(embed_dim=attention_dim,num_heads=num_heads,dropout=dropout,batch_first=True)

        self.norm1 = nn.LayerNorm(attention_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(attention_dim,attention_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim * 4,attention_dim),
            nn.Dropout(dropout),
        )

        self.norm2 = nn.LayerNorm(attention_dim)

        self.output_dim = attention_dim * 2

    def forward(self,image_features,meta_features):
        # image_features: (B, image_dim)
        # meta_features:  (B, meta_dim)

        image_token = self.image_projection(image_features)

        meta_token = self.meta_projection(meta_features)

        # (B, 2, attention_dim)
        tokens = torch.stack([image_token, meta_token],dim=1,)

        # Each token attends over both image and metadata tokens
        attended_tokens, _ = self.attention(query=tokens,key=tokens,value=tokens,need_weights=False,)

        # Attention residual connection
        tokens = self.norm1(tokens + attended_tokens)

        # Feed-forward residual connection
        tokens = self.norm2(tokens + self.feed_forward(tokens))

        # (B, 2, attention_dim)
        #       ↓
        # (B, 2 * attention_dim)
        return tokens.flatten(start_dim=1)


class FuseImageAndMetadata(nn.Module):
    """
    Fuses a Swin image embedding with a metadata embedding.

    Default dimension flow for image_dim=768 and meta_dim=256:

      concatenation:
          [image | metadata] -> 1024 dimensions

      hadamard:
          project image 768 -> 256
          elementwise multiply with metadata
          -> 256 dimensions

      attention:
          project image 768 -> 256
          project metadata 256 -> 256
          create two modality tokens
          apply multi-head attention
          concatenate the two attended tokens
          -> 512 dimensions
    """

    def __init__(self,image_dim,meta_dim,fusion_type,attention_dim=256,num_heads=8,attention_dropout=0.1):
        super().__init__()

        if fusion_type not in FUSION_TYPES:
            raise ValueError(
                f"Invalid fusion_type '{fusion_type}'. "
                f"Expected one of {FUSION_TYPES}."
            )

        self.fusion_type = fusion_type
        self.image_dim = image_dim
        self.meta_dim = meta_dim

        if fusion_type == "concatenation":
            self.output_dim = image_dim + meta_dim

        elif fusion_type == "hadamard":
            self.image_proj_to_meta = nn.Linear(image_dim,meta_dim)
            self.output_dim = meta_dim

        elif fusion_type == "attention":
            self.attention_fusion = (
                MultimodalAttentionFusion(
                    image_dim=image_dim,
                    meta_dim=meta_dim,
                    attention_dim=attention_dim,
                    num_heads=num_heads,
                    dropout=attention_dropout,
                )
            )

            self.output_dim = (self.attention_fusion.output_dim)

    def forward(self,img_features,meta_features):
        # img_features:  (B, image_dim)
        # meta_features: (B, meta_dim)

        if self.fusion_type == "concatenation":
            return torch.cat([img_features, meta_features],dim=1)

        if self.fusion_type == "hadamard":
            image_projected = self.image_proj_to_meta(img_features)
            return image_projected * meta_features

        return self.attention_fusion(img_features,meta_features)