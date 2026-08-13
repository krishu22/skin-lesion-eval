import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdjustedCrossEntropy(nn.Module):
    """Logit-Adjusted Cross-Entropy (Menon et al., 2021 convention).

    L_LA(z, y) = CE(z + tau * log(pi), y), where pi_k is the training-fold
    class prior (n_k / N). This is the plus-sign training-time convention
    (penalizes majority classes during training), not the inference-time
    posterior-correction minus-sign variant. Evaluation in this codebase
    uses raw (unadjusted) logits for every loss, so the adjustment here is
    train-only — consistent with how cb_focal's class weights are also
    training-only.

    tau=0 -> ordinary cross-entropy; tau=1 -> full prior-logit adjustment
    (equivalent to Balanced Softmax up to a class-independent constant).
    """

    def __init__(self, class_counts, tau=1.0, reduction="mean"):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        pi = counts / counts.sum()
        self.register_buffer("log_pi", torch.log(pi))
        self.tau = tau
        self.reduction = reduction

    def forward(self, logits, targets):
        adjusted_logits = logits + self.tau * self.log_pi.to(logits.device).unsqueeze(0)
        return F.cross_entropy(adjusted_logits, targets, reduction=self.reduction)
