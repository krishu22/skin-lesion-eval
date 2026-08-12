import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    alpha, if given, is a per-class weight tensor (e.g. class-balanced weights).
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha)

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[targets]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
