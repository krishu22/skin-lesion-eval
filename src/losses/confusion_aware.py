import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfusionAwareCELoss(nn.Module):
    """Cross-entropy weighted by the cost of the mistake actually made.

    L = -1/N * sum_i C[y_i, pred_i] * log p(y_i)

    C is a KxK cost matrix rebuilt each epoch from the validation confusion
    matrix (see update_from_confusion_matrix): C[y,y] = 1, and for k != y,
    C[y,k] = 1 + lam * M[y,k] / (sum of row y's off-diagonal counts + eps),
    so predicted classes the model frequently confuses with the true class
    get an amplified gradient. Starts as all-ones (plain CE) until the first
    validation pass produces a confusion matrix.
    """

    def __init__(self, num_classes, lam=1.0, eps=1e-6, reduction="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.lam = lam
        self.eps = eps
        self.reduction = reduction
        self.register_buffer("cost_matrix", torch.ones(num_classes, num_classes))

    def update_from_confusion_matrix(self, confusion_matrix):
        cm = torch.as_tensor(confusion_matrix, dtype=torch.float32)
        off_diag_row_sums = cm.sum(dim=1) - torch.diagonal(cm)
        cost = 1.0 + self.lam * cm / (off_diag_row_sums.unsqueeze(1) + self.eps)
        cost.fill_diagonal_(1.0)
        self.cost_matrix = cost.to(self.cost_matrix.device)

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        preds = logits.argmax(dim=1).detach()
        weight = self.cost_matrix.to(logits.device)[targets, preds]
        loss = -weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
