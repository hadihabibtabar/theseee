"""Stage 6 metric tests (spec 21b): known examples + properties."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
for path in (str(SRC), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recsys23_fedrec.training.metrics import binary_logloss, roc_auc  # noqa: E402


# ----------------------------------------------------------------------
# Log-loss
# ----------------------------------------------------------------------
def test_logloss_perfect_predictions_near_zero():
    logits = torch.tensor([20.0, -20.0, 15.0, -15.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert binary_logloss(logits, targets).item() < 1e-5


def test_logloss_worst_predictions():
    logits = torch.tensor([20.0, -20.0])
    targets = torch.tensor([0.0, 1.0])
    assert binary_logloss(logits, targets).item() > 19  # ~ -log(eps)


def test_logloss_uniform_half_equals_log2():
    logits = torch.zeros(1000)
    targets = torch.tensor([1.0, 0.0] * 500)
    assert math.isclose(binary_logloss(logits, targets).item(), math.log(2), rel_tol=1e-6)


def test_logloss_matches_bcewithlogits():
    torch.manual_seed(0)
    logits = torch.randn(500)
    targets = torch.randint(0, 2, (500,)).float()
    ref = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(binary_logloss(logits, targets), ref, atol=1e-6)


def test_logloss_numerically_stable_for_extreme_logits():
    logits = torch.tensor([1e4, -1e4, 5e3, -5e3])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    value = binary_logloss(logits, targets).item()
    assert math.isfinite(value) and value >= 0


def test_logloss_scalar_and_empty_rejected():
    assert binary_logloss(torch.tensor([1.0]), torch.tensor([1.0])).dim() == 0
    try:
        binary_logloss(torch.tensor([]), torch.tensor([]))
        raise AssertionError("expected ValueError for empty input")
    except ValueError:
        pass


# ----------------------------------------------------------------------
# ROC-AUC
# ----------------------------------------------------------------------
def test_auc_perfect_separation():
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert roc_auc(scores, targets).item() == 1.0


def test_auc_perfectly_wrong():
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert roc_auc(scores, targets).item() == 0.0


def test_auc_interleaved_scores():
    # Positives/negatives arranged so exactly half the pairs are wins:
    # pos scores {0.3, 0.7}, neg scores {0.1, 0.5} -> wins: (0.3>0.1)+(0.3>0.5)*0.5...
    # pairwise: (0.3 vs 0.1) W, (0.3 vs 0.5) L, (0.7 vs 0.1) W, (0.7 vs 0.5) W = 0.75
    scores = torch.tensor([0.3, 0.7, 0.1, 0.5])
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert roc_auc(scores, targets).item() == 0.75


def test_auc_half_when_scores_cross_evenly():
    # pos {0.1, 0.5}, neg {0.3, 0.7}: (0.1>0.3) L, (0.1>0.7) L, (0.5>0.3) W, (0.5>0.7) L -> 0.25
    scores = torch.tensor([0.1, 0.5, 0.3, 0.7])
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert roc_auc(scores, targets).item() == 0.25


def test_auc_ties_handled_exactly():
    # All scores tied -> AUC 0.5 regardless of labels.
    scores = torch.zeros(10)
    targets = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]).float()
    assert roc_auc(scores, targets).item() == 0.5


def test_auc_matches_manual_pairwise_count():
    scores = torch.tensor([0.2, 0.4, 0.35, 0.8, 0.65])
    targets = torch.tensor([0, 1, 0, 1, 0]).float()
    pos = scores[targets == 1]
    neg = scores[targets == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    expected = wins.item() / (len(pos) * len(neg))
    assert math.isclose(roc_auc(scores, targets).item(), expected, rel_tol=1e-6)


def test_auc_matches_sklearn_on_realistic_sample():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(42)
    scores = torch.randn(4096)
    targets = (torch.rand(4096) < 0.3).float()
    ours = roc_auc(scores, targets).item()
    ref = roc_auc_score(targets.numpy(), scores.numpy())
    # float32 (ours) vs float64 (sklearn) reduction noise; identical ranking.
    assert math.isclose(ours, ref, rel_tol=1e-6)


def test_auc_single_class_rejected():
    try:
        roc_auc(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_metrics_accept_logits_not_probabilities():
    """Scores must be monotone-equivalent under sigmoid (same AUC)."""
    torch.manual_seed(1)
    logits = torch.randn(2000)
    targets = (torch.rand(2000) < 0.5).float()
    a = roc_auc(logits, targets).item()
    b = roc_auc(torch.sigmoid(logits), targets).item()
    assert math.isclose(a, b, rel_tol=1e-9)
