import torch
import torch.nn as nn

FUSION_TYPES = ("concatenation", "hadamard", "self_cross_attention")


class SelfAttention(nn.Module):
    """Scaled dot-product self-attention with a residual + LayerNorm.

    Each sample's feature vector is treated as a single token (there is no
    sequence dimension here — one image embedding, one metadata embedding
    per sample), so the attention score is one scalar per sample and its
    softmax is always 1. In effect this block is a learned residual
    transform of the input (Q/K/V projections, then residual + norm). This
    mirrors the fusion design this module is ported from; embed_dim is
    whatever width the caller passes in (768 for image features, 256 for
    metadata features in this repo).
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # x: (B, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_scores = torch.matmul(Q.unsqueeze(1), K.unsqueeze(2)) / (x.size(-1) ** 0.5)  # (B, 1, 1)
        attn_weights = self.softmax(attn_scores)  # (B, 1, 1)
        attn_output = attn_weights.squeeze(-1) * V  # (B, embed_dim)
        return self.norm(attn_output + x)  # (B, embed_dim)


class CrossModalAttention(nn.Module):
    """Cross-modal attention: query from one modality, key/value from the other.

    dim_query and dim_keyvalue may differ (768 for image, 256 for metadata
    here) — the key/value projections map the other modality's features into
    the query's space, so the two modalities never need a shared embedding
    width.
    """

    def __init__(self, dim_query, dim_keyvalue):
        super().__init__()
        self.query = nn.Linear(dim_query, dim_query)
        self.key = nn.Linear(dim_keyvalue, dim_query)
        self.value = nn.Linear(dim_keyvalue, dim_query)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(dim_query)

    def forward(self, query, keyvalue):
        Q = self.query(query)     # (B, dim_query)
        K = self.key(keyvalue)    # (B, dim_query)
        V = self.value(keyvalue)  # (B, dim_query)

        attn_scores = torch.matmul(Q.unsqueeze(1), K.unsqueeze(2)) / (Q.size(-1) ** 0.5)  # (B, 1, 1)
        attn_weights = self.softmax(attn_scores)
        attn_output = attn_weights.squeeze(-1) * V  # (B, dim_query)
        return self.norm(attn_output + query)  # (B, dim_query)


class FuseImageAndMetadata(nn.Module):
    """
    Fuses a Swin image embedding (B, image_dim) with the metadata embedding
    (B, meta_dim) produced by the existing MetadataFeatureExtraction module.

    Dimension flow per fusion_type (image_dim=768, meta_dim=256 in this repo):
      - concatenation:        [img | meta]                             -> image_dim + meta_dim (1024)
      - hadamard:             proj(img: image_dim->meta_dim) * meta    -> meta_dim              (256)
      - self_cross_attention: each modality self-attends, then cross-attends
                               to the other; [cross(img->meta) | cross(meta->img)]
                                                                        -> image_dim + meta_dim (1024)

    `output_dim` is computed here (not by the caller) so the classifier head
    can size itself dynamically instead of hardcoding a fused width.
    """

    def __init__(self, image_dim, meta_dim, fusion_type):
        super().__init__()
        if fusion_type not in FUSION_TYPES:
            raise ValueError(f"invalid fusion_type '{fusion_type}', must be one of {FUSION_TYPES}")

        self.fusion_type = fusion_type
        self.image_dim = image_dim
        self.meta_dim = meta_dim

        if fusion_type == "hadamard":
            self.image_proj_to_meta = nn.Linear(image_dim, meta_dim)
            self.output_dim = meta_dim
        elif fusion_type == "concatenation":
            self.output_dim = image_dim + meta_dim
        else:  # self_cross_attention
            self.image_self = SelfAttention(image_dim)
            self.meta_self = SelfAttention(meta_dim)
            self.image_cross = CrossModalAttention(image_dim, meta_dim)
            self.meta_cross = CrossModalAttention(meta_dim, image_dim)
            self.output_dim = image_dim + meta_dim

    def forward(self, img_features, meta_features):  # (B, image_dim), (B, meta_dim)
        if self.fusion_type == "hadamard":
            image_projected = self.image_proj_to_meta(img_features)
            return image_projected * meta_features  # (B, meta_dim)

        if self.fusion_type == "concatenation":
            return torch.cat([img_features, meta_features], dim=1)  # (B, image_dim + meta_dim)

        img_self = self.image_self(img_features)
        meta_self = self.meta_self(meta_features)
        img_cross = self.image_cross(img_self, meta_self)  # (B, image_dim)
        meta_cross = self.meta_cross(meta_self, img_self)  # (B, meta_dim)
        return torch.cat([img_cross, meta_cross], dim=1)  # (B, image_dim + meta_dim)
