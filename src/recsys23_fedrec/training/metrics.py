"""Binary classification metrics for Stage 6 (log-loss + ROC-AUC).

Implemented directly on logits/probabilities in torch with no sklearn
dependency, so the metric code is auditable and identical across later
federated stages. Both functions accept raw logits (sigmoid is applied here
for evaluation only — never before BCEWithLogitsLoss in the loss path).
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["binary_logloss", "roc_auc"]


def binary_logloss(logits: Tensor, targets: Tensor, *, eps: float = 1e-7) -> Tensor:
    """Numerically stable binary log-loss from raw logits.

    Uses the fused form ``log(1 + exp(-z)) + (1 - y) * z`` which never
    exponentiates a large positive number. Returns a scalar float tensor.
    """
    logits = logits.detach().float().reshape(-1)
    targets = targets.detach().float().reshape(-1)
    if logits.numel() == 0:
        raise ValueError("binary_logloss received empty tensors")
    if not torch.isfinite(logits).all():
        raise ValueError("binary_logloss received non-finite logits")
    losses = torch.nn.functional.softplus(-logits) + (1.0 - targets) * logits
    return losses.mean()


def _rank_auc(scores: Tensor, targets: Tensor) -> Tensor:
    """Exact ROC-AUC via the probabilistic interpretation.

    AUC = P(score_positive > score_negative) + 0.5 * P(tie), computed with a
    sort + linear rank scan. Handles ties exactly (average ranks) and is
    O(n log n). No sklearn dependency.
    """
    scores = scores.detach().float().reshape(-1)
    targets = targets.detach().float().reshape(-1)
    n = scores.numel()
    if n == 0:
        raise ValueError("roc_auc received empty tensors")
    if not torch.isfinite(scores).all():
        raise ValueError("roc_auc received non-finite scores")
    n_pos = int(targets.sum().item())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"roc_auc undefined: labels contain only one class "
            f"(positives={n_pos}, negatives={n_neg})"
        )

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_targets = targets[order]

    # Average ranks (1-based) with ties handled: ties share the mean rank.
    ranks = torch.arange(1, n + 1, dtype=torch.float64)
    # Find groups of equal scores; assign mean rank per group.
    group_start = torch.zeros(n, dtype=torch.bool)
    group_start[0] = True
    group_start[1:] = sorted_scores[1:] != sorted_scores[:-1]
    group_ids = torch.cumsum(group_start.long(), dim=0) - 1  # [n]
    counts = torch.bincount(group_ids)
    cum = torch.cumsum(counts, dim=0)
    starts = cum - counts
    mean_rank_per_group = starts.float() + (counts.float() + 1.0) / 2.0  # 1-based avg
    avg_ranks = mean_rank_per_group[group_ids]

    rank_sum_pos = avg_ranks[sorted_targets == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc.float()


def roc_auc(logits_or_scores: Tensor, targets: Tensor) -> Tensor:
    """ROC-AUC from raw logits (monotone in probability) or direct scores."""
    return _rank_auc(logits_or_scores, targets)
