"""Stage 5 MMoE tests (spec sections A-G).

Synthetic-fixture tests run on the preprocessed 300-row mini-dataset; the
real Stage 2 artifact is used (read-only) for construction fidelity checks.
Real-data tests live in scripts/test_mmoe.py; here the synthetic fixture
exercises identical code paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
for path in (str(SRC), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recsys23_fedrec.models import (  # noqa: E402
    MMoEConfig,
    TransformerConfig,
    TransformerMMoE,
    mmoe_loss,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture()
def small_mmoe(feature_set):
    """Compact MMoE over the synthetic fixture schema (fast tests)."""
    torch.manual_seed(42)
    return TransformerMMoE(
        feature_set,
        backbone_config=TransformerConfig(d_model=32, nhead=4, num_layers=2, dim_feedforward=32),
        mmoe_config=MMoEConfig(num_experts=8, dropout=0.1),
    )


# ----------------------------------------------------------------------
# A. Model construction from the real Stage 2 schema
# ----------------------------------------------------------------------
def test_construction_from_real_schema(real_feature_set):
    torch.manual_seed(42)
    model = TransformerMMoE(real_feature_set)
    assert model.backbone.config.d_model == 128
    assert model.mmoe_config.num_experts == 8
    assert model.mmoe_config.num_tasks == 2
    assert model.backbone.num_feature_tokens == 92
    assert model.backbone.sequence_length == 93


def test_vocabularies_unchanged_from_stage4(real_feature_set):
    torch.manual_seed(42)
    model = TransformerMMoE(real_feature_set)
    expected = [int(v) for v in real_feature_set.vocabulary_sizes]
    assert [e.num_embeddings for e in model.backbone.embeddings] == expected


def test_transformer_config_unchanged(real_feature_set):
    torch.manual_seed(42)
    model = TransformerMMoE(real_feature_set)
    cfg = model.backbone.config
    assert (cfg.d_model, cfg.nhead, cfg.num_layers, cfg.dim_feedforward, cfg.dropout) == (
        128, 8, 6, 128, 0.1,
    )


def test_invalid_mmoe_config_rejected(real_feature_set):
    with pytest.raises(Exception):
        TransformerMMoE(real_feature_set, mmoe_config=MMoEConfig(num_experts=0))


# ----------------------------------------------------------------------
# B. Shape tests (synthetic batch)
# ----------------------------------------------------------------------
def test_output_shapes(small_mmoe, synthetic_batch, feature_set):
    small_mmoe.eval()
    b = synthetic_batch["categorical"].shape[0]
    d = 32
    with torch.no_grad():
        out = small_mmoe(synthetic_batch)
    assert out["cls"].shape == (b, d)
    assert out["expert_outputs"].shape == (b, 8, d)
    assert out["click_gate"].shape == (b, 8)
    assert out["install_gate"].shape == (b, 8)
    assert out["click_repr"].shape == (b, d)
    assert out["install_repr"].shape == (b, d)
    assert out["click_logit"].shape == (b,)
    assert out["install_logit"].shape == (b,)


def test_encoder_sequence_remains_93(small_mmoe, synthetic_batch):
    """Sequence = 1 + num_feature_tokens, derived from the schema (93 on real data)."""
    small_mmoe.eval()
    with torch.no_grad():
        encoded = small_mmoe.backbone.encode(synthetic_batch)
    b = synthetic_batch["categorical"].shape[0]
    assert encoded.shape == (b, small_mmoe.backbone.sequence_length, 32)
    assert small_mmoe.backbone.sequence_length == 1 + small_mmoe.backbone.num_feature_tokens


def test_diagnostics_can_be_disabled(small_mmoe, synthetic_batch):
    small_mmoe.eval()
    with torch.no_grad():
        out = small_mmoe(synthetic_batch, return_diagnostics=False)
    assert set(out) == {"click_logit", "install_logit"}


def test_outputs_finite(small_mmoe, synthetic_batch):
    small_mmoe.eval()
    with torch.no_grad():
        out = small_mmoe(synthetic_batch)
    for key, value in out.items():
        assert torch.isfinite(value).all(), f"{key} contains non-finite values"


# ----------------------------------------------------------------------
# 12. Gate correctness (per-sample, not batch average)
# ----------------------------------------------------------------------
def test_gates_sum_to_one_per_sample(small_mmoe, synthetic_batch):
    small_mmoe.eval()
    with torch.no_grad():
        out = small_mmoe(synthetic_batch)
    b = out["click_gate"].shape[0]
    for key in ("click_gate", "install_gate"):
        sums = out[key].sum(dim=-1)
        assert sums.shape == (b,)
        assert torch.allclose(sums, torch.ones(b), atol=1e-6), f"{key} rows must each sum to 1"
        assert bool(((out[key] >= -1e-6) & (out[key] <= 1 + 1e-6)).all()), (
            f"{key} weights must be within [0, 1]"
        )


def test_gates_are_sample_specific(small_mmoe):
    """Gates must be functions of the input, not global constants (spec 13)."""
    d = small_mmoe.backbone.config.d_model
    cls_a = torch.randn(1, d)
    cls_b = torch.randn(1, d)
    # Make the two inputs clearly distinct.
    cls_b = -cls_a
    small_mmoe.eval()
    with torch.no_grad():
        ga = small_mmoe.compute_gate(0, cls_a)
        gb = small_mmoe.compute_gate(0, cls_b)
        ga1 = small_mmoe.compute_gate(1, cls_a)
        gb1 = small_mmoe.compute_gate(1, cls_b)
    assert not torch.allclose(ga, gb, atol=1e-4), "click gate must vary with input"
    assert not torch.allclose(ga1, gb1, atol=1e-4), "install gate must vary with input"
    # Row-wise: each row is its own distribution (no broadcast constant).
    cls_batch = torch.randn(4, d)
    with torch.no_grad():
        g = small_mmoe.compute_gate(0, cls_batch)
    assert g.shape == (4, 8)
    assert not torch.allclose(g[0], g[1], atol=1e-4)


# ----------------------------------------------------------------------
# 14. Expert sharing
# ----------------------------------------------------------------------
def test_single_shared_expert_set(small_mmoe):
    assert len(small_mmoe.experts) == 8
    expert_ids = {id(e) for e in small_mmoe.experts}
    assert len(expert_ids) == 8, "experts must be distinct modules"
    # One gate + one tower per task, but experts live outside tasks.
    assert len(small_mmoe.gates) == 2
    assert len(small_mmoe.towers) == 2
    assert not hasattr(small_mmoe, "click_experts")
    assert not hasattr(small_mmoe, "install_experts")


def test_both_tasks_route_through_same_experts(small_mmoe, synthetic_batch):
    """Perturbing an expert must change BOTH task representations."""
    small_mmoe.eval()
    with torch.no_grad():
        out_before = small_mmoe(synthetic_batch)
        saved = small_mmoe.experts[3][0].weight.clone()
        with torch.no_grad():
            # Multiplicative perturbation: an additive all-ones delta would
            # only interact with sum(cls) ~ 0 (post-LayerNorm) and vanish.
            small_mmoe.experts[3][0].weight.mul_(3.0)
        out_after = small_mmoe(synthetic_batch)
        small_mmoe.experts[3][0].weight.copy_(saved)
    assert not torch.allclose(out_before["click_repr"], out_after["click_repr"], atol=1e-6), (
        "click must route through the shared experts"
    )
    assert not torch.allclose(out_before["install_repr"], out_after["install_repr"], atol=1e-6), (
        "install must route through the shared experts"
    )


def test_expert_aggregation_is_weighted_sum(small_mmoe, synthetic_batch):
    """einsum aggregation must equal the manual weighted sum."""
    small_mmoe.eval()
    with torch.no_grad():
        out = small_mmoe(synthetic_batch)
    manual = (out["install_gate"].unsqueeze(-1) * out["expert_outputs"]).sum(dim=1)
    assert torch.allclose(out["install_repr"], manual, atol=1e-5)


def test_towers_independent(small_mmoe):
    tower_click = small_mmoe.towers[0]
    tower_install = small_mmoe.towers[1]
    assert tower_click[0].weight.data_ptr() != tower_install[0].weight.data_ptr()
    # Stage 5 tower spec: Linear -> GELU -> Dropout -> Linear.
    assert isinstance(tower_click[0], torch.nn.Linear)
    assert isinstance(tower_click[1], torch.nn.GELU)
    assert isinstance(tower_click[2], torch.nn.Dropout)
    assert isinstance(tower_click[3], torch.nn.Linear)


# ----------------------------------------------------------------------
# 15. Loss tests
# ----------------------------------------------------------------------
def test_losses_finite_scalar_backward(small_mmoe, synthetic_batch):
    out = small_mmoe(synthetic_batch)
    losses = mmoe_loss(out, synthetic_batch)
    for key in ("click", "install", "total"):
        assert losses[key].dim() == 0, f"{key} loss must be scalar"
        assert torch.isfinite(losses[key]), f"{key} loss must be finite"
    losses["total"].backward()


def test_loss_weighting(small_mmoe, synthetic_batch):
    small_mmoe.eval()
    with torch.no_grad():
        out = small_mmoe(synthetic_batch)
        l1 = mmoe_loss(out, synthetic_batch, lambda_click=1.0, lambda_install=1.0)
        l2 = mmoe_loss(out, synthetic_batch, lambda_click=0.5, lambda_install=2.0)
    expected = 0.5 * l1["click"] + 2.0 * l1["install"]
    assert torch.allclose(l2["total"], expected, atol=1e-6)


def test_loss_matches_direct_bce(small_mmoe, synthetic_batch):
    small_mmoe.eval()
    with torch.no_grad():
        out = small_mmoe(synthetic_batch)
        losses = mmoe_loss(out, synthetic_batch)
        ref_click = torch.nn.functional.binary_cross_entropy_with_logits(
            out["click_logit"], synthetic_batch["click"]
        )
        ref_install = torch.nn.functional.binary_cross_entropy_with_logits(
            out["install_logit"], synthetic_batch["install"]
        )
    assert torch.allclose(losses["click"], ref_click)
    assert torch.allclose(losses["install"], ref_install)


# ----------------------------------------------------------------------
# 16. Gradient tests
# ----------------------------------------------------------------------
def test_gradients_across_all_components(small_mmoe, synthetic_batch):
    losses = mmoe_loss(small_mmoe(synthetic_batch), synthetic_batch)
    losses["total"].backward()
    for name, p in small_mmoe.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_experts_receive_multitask_gradient(small_mmoe, synthetic_batch):
    """Shared experts must get gradient from the TOTAL multi-task loss."""
    model = small_mmoe
    losses = mmoe_loss(model(synthetic_batch), synthetic_batch)
    losses["total"].backward()
    for i, expert in enumerate(model.experts):
        assert expert[0].weight.grad is not None
        assert expert[0].weight.grad.abs().sum() > 0, f"expert {i} received no gradient"

    # Decomposition check: click-only and install-only gradients should both
    # be nonzero and generally differ (tasks use different gates).
    model.zero_grad(set_to_none=True)
    mmoe_loss(model(synthetic_batch), synthetic_batch, lambda_install=0.0)["total"].backward()
    click_only = model.experts[0][0].weight.grad.clone()
    model.zero_grad(set_to_none=True)
    mmoe_loss(model(synthetic_batch), synthetic_batch, lambda_click=0.0)["total"].backward()
    install_only = model.experts[0][0].weight.grad.clone()
    assert click_only.abs().sum() > 0 and install_only.abs().sum() > 0
    assert not torch.allclose(click_only, install_only), (
        "task-specific gates should route differently (gradients differ)"
    )


def test_gates_and_towers_receive_gradient(small_mmoe, synthetic_batch):
    losses = mmoe_loss(small_mmoe(synthetic_batch), synthetic_batch)
    losses["total"].backward()
    for task, gate in zip(("click", "install"), small_mmoe.gates):
        assert gate.weight.grad is not None and gate.weight.grad.abs().sum() > 0
    for task, tower in zip(("click", "install"), small_mmoe.towers):
        assert tower[0].weight.grad is not None and tower[0].weight.grad.abs().sum() > 0
        assert tower[3].weight.grad is not None and tower[3].weight.grad.abs().sum() > 0


def test_backbone_receives_gradient(small_mmoe, synthetic_batch):
    losses = mmoe_loss(small_mmoe(synthetic_batch), synthetic_batch)
    losses["total"].backward()
    # Encoder layers and CLS token must participate.
    assert small_mmoe.backbone.cls_token.grad is not None
    assert small_mmoe.backbone.cls_token.grad.abs().sum() > 0
    layer0 = small_mmoe.backbone.encoder.layers[0]
    assert layer0.self_attn.in_proj_weight.grad is not None
    assert layer0.self_attn.in_proj_weight.grad.abs().sum() > 0


# ----------------------------------------------------------------------
# 17. Tiny optimization sanity
# ----------------------------------------------------------------------
def test_tiny_optimization_sanity(small_mmoe, synthetic_batch):
    model = small_mmoe
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses_seen = []
    ref_expert_w = model.experts[0][0].weight.detach().clone()
    ref_tower_w = model.towers[1][3].weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        out = model(synthetic_batch)
        losses = mmoe_loss(out, synthetic_batch)
        assert torch.isfinite(losses["total"])
        assert torch.isfinite(out["click_logit"]).all()
        assert torch.isfinite(out["install_logit"]).all()
        losses_seen.append(losses["total"].item())
        losses["total"].backward()
        optimizer.step()
    assert not torch.equal(ref_expert_w, model.experts[0][0].weight.detach()), "experts did not update"
    assert not torch.equal(ref_tower_w, model.towers[1][3].weight.detach()), "install tower did not update"
    assert all(torch.isfinite(torch.tensor(losses_seen)))


# ----------------------------------------------------------------------
# 19. CPU + 18. CUDA
# ----------------------------------------------------------------------
def test_cpu_forward_backward(small_mmoe, synthetic_batch):
    out = small_mmoe(synthetic_batch)
    losses = mmoe_loss(out, synthetic_batch)
    losses["total"].backward()
    assert out["click_logit"].device.type == "cpu"
    assert out["install_logit"].device.type == "cpu"


def test_cuda_forward_backward(real_feature_set):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    device = torch.device("cuda")
    model = TransformerMMoE(
        real_feature_set,
        backbone_config=TransformerConfig(d_model=64, nhead=4, num_layers=2, dim_feedforward=64),
    ).to(device)
    b = 32
    batch = {
        # IDs {0, 1} are valid for every real vocabulary (smallest = 3).
        "categorical": torch.randint(0, 2, (b, 30), device=device),
        "numerical": torch.randn(b, 38, device=device),
        "binary": torch.randint(0, 2, (b, 11), device=device).float(),
        "missing": torch.randint(0, 2, (b, 13), device=device).float(),
        "click": torch.randint(0, 2, (b,), device=device).float(),
        "install": torch.randint(0, 2, (b,), device=device).float(),
    }
    out = model(batch)
    assert out["click_logit"].shape == (b,) and out["install_logit"].shape == (b,)
    assert torch.isfinite(out["click_logit"]).all() and torch.isfinite(out["install_logit"]).all()
    losses = mmoe_loss(out, batch)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# ----------------------------------------------------------------------
# 22. Stage 4 compatibility through the MMoE path
# ----------------------------------------------------------------------
def test_stage4_baseline_semantics_unchanged(feature_set):
    """Stage 4 model with identical seed must be unaffected by Stage 5 code."""
    from recsys23_fedrec.models import TransformerBaseline

    torch.manual_seed(42)
    m1 = TransformerBaseline(
        feature_set, d_model=32, nhead=4, num_layers=2, dim_feedforward=32
    )
    torch.manual_seed(42)
    m2 = TransformerBaseline(
        feature_set, d_model=32, nhead=4, num_layers=2, dim_feedforward=32
    )
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)
    # Default construction still exposes install head.
    assert m1.head is not None and "install_logit" in m1({"categorical": torch.zeros(2, feature_set.num_categorical, dtype=torch.int64), "numerical": torch.zeros(2, feature_set.num_numerical), "binary": torch.zeros(2, feature_set.num_binary), "missing": torch.zeros(2, feature_set.num_missing)})
