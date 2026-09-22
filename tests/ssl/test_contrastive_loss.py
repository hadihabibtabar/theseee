"""Stage 7 contrastive-loss tests: NT-Xent / InfoNCE (spec 7.8)."""

from __future__ import annotations

import math

import pytest
import torch

from recsys23_fedrec.ssl import (
    DEFAULT_TEMPERATURE,
    ContrastiveLossError,
    l2_normalize,
    nt_xent_loss,
)

# ---------------------------------------------------------------------------
# Hand-checkable synthetic examples
# ---------------------------------------------------------------------------


def test_b2_hand_computable():
    """B=2; verifies the loss against a fully manual NT-Xent computation
    at tau=0.2 (independent loop-based reference)."""
    tau = 0.2
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z2 = torch.tensor([[0.99, 0.141], [0.141, 0.99]])
    z1n, z2n = l2_normalize(z1), l2_normalize(z2)
    z = torch.cat([z1n, z2n], dim=0)
    sim = (z @ z.T) / tau
    eye = torch.eye(4, dtype=torch.bool)
    log_prob = torch.log_softmax(sim.masked_fill(eye, float("-inf")), dim=1)
    idx = torch.arange(4)
    pos = (idx + 2) % 4
    expected = -log_prob[idx, pos].mean().item()
    loss = nt_xent_loss(z1, z2, temperature=tau)
    assert math.isclose(float(loss), expected, rel_tol=1e-6)


def test_identical_pairs_yield_low_loss_perfect_predictions():
    """Perfectly aligned positive pairs, negatives orthogonal: loss should be
    strictly lower than for shuffled pairs (positive-pair indexing check)."""
    torch.manual_seed(0)
    base = torch.randn(8, 16)
    z1 = l2_normalize(base)
    z2 = l2_normalize(base + 0.01 * torch.randn(8, 16))  # nearly identical
    aligned = float(nt_xent_loss(z1, z2))

    perm = torch.randperm(8)
    z2_shuffled = z2[perm]
    # ensure the permutation actually breaks alignment
    assert not torch.equal(z2_shuffled, z2)
    shuffled = float(nt_xent_loss(z1, z2_shuffled))
    assert aligned < shuffled


def test_b4_loss_value_matches_brute_force():
    """Reference implementation written independently (loops, no tricks)."""
    torch.manual_seed(3)
    tau = 0.2
    z1 = torch.randn(4, 8)
    z2 = torch.randn(4, 8)

    z = torch.cat([l2_normalize(z1), l2_normalize(z2)], dim=0)
    n = 8
    total = 0.0
    for i in range(n):
        pos = (i + 4) % n
        num = math.exp(float(z[i] @ z[pos]) / tau)
        denom = sum(
            math.exp(float(z[i] @ z[k]) / tau) for k in range(n) if k != i
        )
        total += -math.log(num / denom)
    expected = total / n
    loss = nt_xent_loss(z1, z2, temperature=tau)
    assert math.isclose(float(loss), expected, rel_tol=1e-5)


def test_symmetric_loss_view_order_invariant():
    """Swapping (z1, z2) -> (z2, z1) must leave the symmetric loss unchanged."""
    torch.manual_seed(5)
    z1, z2 = torch.randn(6, 32), torch.randn(6, 32)
    a = float(nt_xent_loss(z1, z2))
    b = float(nt_xent_loss(z2, z1))
    assert math.isclose(a, b, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Requirements from spec 7.3
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("b", [2, 4, 8, 32])
def test_supports_arbitrary_batch_sizes(b):
    z1, z2 = torch.randn(b, 16), torch.randn(b, 16)
    loss = nt_xent_loss(z1, z2)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("dim", [1, 3, 64, 128])
def test_arbitrary_projection_dimension(dim):
    z1, z2 = torch.randn(4, dim), torch.randn(4, dim)
    loss = nt_xent_loss(z1, z2)
    assert torch.isfinite(loss)


def test_works_on_unnormalized_embeddings():
    """Projections of any magnitude are defensively normalized inside."""
    torch.manual_seed(7)
    z1 = torch.randn(4, 8) * 100.0
    z2 = torch.randn(4, 8) * 0.001
    loss = nt_xent_loss(z1, z2)
    assert torch.isfinite(loss)


def test_self_similarity_excluded_numerically():
    """Loss must never treat a view as its own negative.

    Hand-checkable construction (B=2, 4-D orthonormal basis): every view's
    positive AND both negatives are mutually orthogonal (sim 0). With the
    self entry excluded, each row's denominator is exactly 3, so the loss
    is exactly log(3). If the diagonal leaked in, each row would also carry
    its own sim(1)/tau = 5 term and the loss would be log(3 + e^5) != log(3).
    """
    tau = 0.2
    z1 = torch.eye(4)[:2]  # e1, e2
    z2 = torch.eye(4)[2:]  # e3, e4
    loss = float(nt_xent_loss(z1, z2, temperature=tau))
    assert math.isclose(loss, math.log(3.0), rel_tol=1e-6)
    leak_value = math.log(3.0 + math.exp(1.0 / tau))
    assert not math.isclose(loss, leak_value, rel_tol=1e-3)


def test_degenerate_all_equal_views_hand_value():
    """Degenerate repeated vectors with a hand-computed value.

    z1 block = e1 repeated B times, z2 block = e2 repeated B times (B=4,
    tau=0.2). For any view: positive sim 0; negatives = (B-1) same-direction
    (sim 1) + (B-1) orthogonal (sim 0). Per-view loss =
    log(1 + (B-1)e^{1/tau} + (B-1)) = log(B + (B-1)e^{1/tau}).
    Also proves stability: negatives dominate (e^5 terms) yet no overflow.
    """
    b, tau = 4, 0.2
    same = torch.zeros(b, 2); same[:, 0] = 1.0   # e1 x b
    other = torch.zeros(b, 2); other[:, 1] = 1.0  # e2 x b
    loss = float(nt_xent_loss(same, other, temperature=tau))
    expected = math.log(b + (b - 1) * math.exp(1.0 / tau))
    assert math.isclose(loss, expected, rel_tol=1e-5)


def test_finite_loss_and_no_nan_inf():
    torch.manual_seed(11)
    for _ in range(20):
        b = torch.randint(2, 64, (1,)).item()
        d = torch.randint(1, 128, (1,)).item()
        z1 = torch.randn(b, d) * 10
        z2 = torch.randn(b, d) * 10
        loss = nt_xent_loss(z1, z2)
        assert torch.isfinite(loss)


def test_extreme_similarity_does_not_overflow():
    """Cosine 1.0 / tau=0.2 -> exp(5) is safe; check large-but-valid inputs."""
    z = torch.ones(4, 8)
    loss = nt_xent_loss(z, z.clone(), temperature=0.2)
    assert torch.isfinite(loss)
    # degenerate tau=0.05: exp(20) still fine in float32 via log_softmax
    loss = nt_xent_loss(z, z.clone(), temperature=0.05)
    assert torch.isfinite(loss)


def test_gradients_non_zero_and_finite():
    torch.manual_seed(13)
    z1 = torch.randn(8, 16, requires_grad=True)
    z2 = torch.randn(8, 16, requires_grad=True)
    loss = nt_xent_loss(z1, z2)
    loss.backward()
    assert z1.grad is not None and z2.grad is not None
    assert torch.isfinite(z1.grad).all() and torch.isfinite(z2.grad).all()
    assert z1.grad.abs().sum() > 0 and z2.grad.abs().sum() > 0


def test_gradients_flow_to_both_views_symmetrically():
    """Symmetry: gradient magnitudes for z1 and z2 should match closely when
    the configuration is mirrored."""
    torch.manual_seed(17)
    z1 = torch.randn(4, 8)
    z2 = torch.randn(4, 8)
    a1 = z1.clone().requires_grad_(True)
    a2 = z2.clone().requires_grad_(True)
    nt_xent_loss(a1, a2).backward()
    b1 = z2.clone().requires_grad_(True)
    b2 = z1.clone().requires_grad_(True)
    nt_xent_loss(b1, b2).backward()
    assert torch.allclose(a1.grad, b2.grad, atol=1e-6)
    assert torch.allclose(a2.grad, b1.grad, atol=1e-6)


def test_temperature_changes_loss_magnitude():
    """For perfectly aligned pairs (pos sim = 1 > every negative sim), the
    loss is monotonically decreasing in sharper temperature: cold < hot."""
    torch.manual_seed(19)
    z1 = torch.randn(8, 16)
    z2 = z1.clone()  # perfectly aligned positives
    hot = float(nt_xent_loss(z1, z2, temperature=1.0))
    cold = float(nt_xent_loss(z1, z2, temperature=0.2))
    assert cold < hot


def test_default_temperature_is_spec_value():
    assert DEFAULT_TEMPERATURE == 0.2


def test_l2_normalize_unit_rows_and_zero_guard():
    z = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    z = l2_normalize(z)
    assert torch.allclose(z[0].norm(), torch.tensor(1.0))
    assert torch.isfinite(z).all()  # zero row must not produce NaN


def test_invalid_inputs_rejected():
    with pytest.raises(ContrastiveLossError):
        nt_xent_loss(torch.randn(1, 8), torch.randn(1, 8))  # B < 2
    with pytest.raises(ContrastiveLossError):
        nt_xent_loss(torch.randn(4, 8), torch.randn(4, 9))  # shape mismatch
    with pytest.raises(ContrastiveLossError):
        nt_xent_loss(torch.randn(4, 8), torch.randn(4, 8), temperature=0.0)
    with pytest.raises(ContrastiveLossError):
        nt_xent_loss(torch.randn(4), torch.randn(4))  # not 2-D


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu():
    torch.manual_seed(23)
    z1, z2 = torch.randn(8, 32), torch.randn(8, 32)
    cpu = float(nt_xent_loss(z1, z2))
    cuda = float(nt_xent_loss(z1.cuda(), z2.cuda()))
    assert math.isclose(cpu, cuda, rel_tol=1e-5)
