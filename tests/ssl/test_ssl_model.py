"""Stage 7 SSL model tests: projection head, joint loss, gradients (spec 7.8)."""

from __future__ import annotations

import pytest
import torch

from recsys23_fedrec.models import SSLConfig, SSLTransformerMMoE
from recsys23_fedrec.models.mmoe import mmoe_loss
from recsys23_fedrec.ssl import (
    DEFAULT_ALPHA,
    JointLossError,
    FeatureCorruptionAugmentation,
    nt_xent_loss,
    ssl_joint_loss,
)

D_MODEL = 32  # small_test backbone width


# ---------------------------------------------------------------------------
# Shapes / output contract
# ---------------------------------------------------------------------------
def test_projection_output_shape(small_ssl_model, synthetic_batch):
    out = small_ssl_model(synthetic_batch)
    assert out["z1"].shape == (8, 64)
    assert out["z2"].shape == (8, 64)


def test_projections_are_l2_normalized(small_ssl_model, synthetic_batch):
    out = small_ssl_model(synthetic_batch)
    for key in ("z1", "z2"):
        norms = out[key].norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_cls_remains_128d_contract_via_single_view(small_ssl_model, synthetic_batch):
    """The MMoE heads keep consuming the original CLS [B, d_model] (32 here)."""
    out = small_ssl_model.supervised_forward(synthetic_batch)
    assert out["cls"].shape == (8, D_MODEL)


def test_click_install_logit_shapes(small_ssl_model, synthetic_batch):
    out = small_ssl_model(synthetic_batch)
    for view_key in ("view1", "view2"):
        assert out[view_key]["click_logit"].shape == (8,)
        assert out[view_key]["install_logit"].shape == (8,)


def test_dual_view_output_keys(small_ssl_model, synthetic_batch):
    out = small_ssl_model(synthetic_batch, return_diagnostics=True)
    for key in ("z1", "z2", "view1", "view2", "cls_view1", "cls_view2"):
        assert key in out, key
    assert out["cls_view1"].shape == (8, D_MODEL)


def test_forward_without_augmentations_raises(feature_set, synthetic_batch):
    model = SSLTransformerMMoE(
        feature_set,
        backbone_config=_small_backbone(),
    )
    with pytest.raises(Exception):
        model(synthetic_batch)


def _small_backbone():
    from recsys23_fedrec.models import TransformerConfig

    return TransformerConfig(d_model=32, nhead=4, num_layers=2, dim_feedforward=32)


# ---------------------------------------------------------------------------
# Stage 6 behavior preserved through the composition
# ---------------------------------------------------------------------------
def test_supervised_forward_matches_pure_mmoe_model(feature_set, synthetic_batch):
    """SSL wrapper in eval mode must be EXACTLY the Stage 5/6 model."""
    from recsys23_fedrec.models import MMoEConfig, TransformerMMoE, TransformerConfig

    torch.manual_seed(42)
    ssl_model = SSLTransformerMMoE(
        feature_set,
        backbone_config=TransformerConfig(d_model=32, nhead=4, num_layers=2, dim_feedforward=32),
        mmoe_config=MMoEConfig(num_experts=4),
    )
    torch.manual_seed(42)
    pure = TransformerMMoE(
        feature_set,
        backbone_config=TransformerConfig(d_model=32, nhead=4, num_layers=2, dim_feedforward=32),
        mmoe_config=MMoEConfig(num_experts=4),
    )
    ssl_model.eval(), pure.eval()
    with torch.no_grad():
        a = ssl_model.supervised_forward(synthetic_batch, return_diagnostics=False)
        b = pure(synthetic_batch, return_diagnostics=False)
    assert torch.equal(a["click_logit"], b["click_logit"])
    assert torch.equal(a["install_logit"], b["install_logit"])


def test_no_sigmoid_before_loss(small_ssl_model, synthetic_batch):
    """Logits must be raw: their range is unbounded, not [0, 1]."""
    small_ssl_model.eval()
    with torch.no_grad():
        out = small_ssl_model.supervised_forward(synthetic_batch, return_diagnostics=False)
    for key in ("click_logit", "install_logit"):
        assert out[key].abs().max() > 0  # finite floats, not squashed
        assert not ((out[key] >= 0.0) & (out[key] <= 1.0)).all(), (
            f"{key} looks sigmoid-squashed into [0, 1]"
        )


# ---------------------------------------------------------------------------
# Joint loss
# ---------------------------------------------------------------------------
def test_joint_loss_finite(small_ssl_model, synthetic_batch):
    out = small_ssl_model(synthetic_batch)
    total, comp = ssl_joint_loss(out, synthetic_batch, return_diagnostics=True)
    assert torch.isfinite(total)
    for key, value in comp.items():
        assert torch.isfinite(value), key


def test_joint_loss_components_match_definition(small_ssl_model, synthetic_batch):
    """total == alpha * supervised + (1 - alpha) * contrastive, exactly."""
    out = small_ssl_model(synthetic_batch)
    total, comp = ssl_joint_loss(out, synthetic_batch, return_diagnostics=True)
    expected = DEFAULT_ALPHA * comp["supervised"] + (1 - DEFAULT_ALPHA) * comp["contrastive"]
    assert torch.allclose(total, expected, atol=1e-6)


def test_alpha_semantics_extremes(small_ssl_model, synthetic_batch):
    """alpha=1 -> pure supervised; alpha=0 -> pure contrastive."""
    out = small_ssl_model(synthetic_batch)
    total_sup, comp = ssl_joint_loss(out, synthetic_batch, alpha=1.0, return_diagnostics=True)
    assert torch.allclose(total_sup, comp["supervised"])
    total_con, _ = ssl_joint_loss(out, synthetic_batch, alpha=0.0, return_diagnostics=True)
    assert torch.allclose(total_con, comp["contrastive"])


def test_invalid_alpha_rejected(small_ssl_model, synthetic_batch):
    out = small_ssl_model(synthetic_batch)
    with pytest.raises(JointLossError):
        ssl_joint_loss(out, synthetic_batch, alpha=1.5)
    with pytest.raises(JointLossError):
        ssl_joint_loss(out, synthetic_batch, alpha=-0.1)


def test_contrastive_component_is_label_free(small_ssl_model, synthetic_batch):
    """Flipping labels must not change the contrastive component."""
    out = small_ssl_model(synthetic_batch)
    flipped = dict(synthetic_batch)
    flipped["click"] = 1.0 - synthetic_batch["click"]
    flipped["install"] = 1.0 - synthetic_batch["install"]
    _, comp_a = ssl_joint_loss(out, synthetic_batch, return_diagnostics=True)
    _, comp_b = ssl_joint_loss(out, flipped, return_diagnostics=True)
    assert torch.allclose(comp_a["contrastive"], comp_b["contrastive"])
    assert not torch.allclose(comp_a["supervised"], comp_b["supervised"])


# ---------------------------------------------------------------------------
# Gradients / optimizer updates
# ---------------------------------------------------------------------------
def test_backward_succeeds_and_all_paths_receive_gradients(
    small_ssl_model, synthetic_batch
):
    out = small_ssl_model(synthetic_batch)
    total = ssl_joint_loss(out, synthetic_batch)
    total.backward()

    # projection head receives gradients
    for name, param in small_ssl_model.projection.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, name
    # Transformer backbone receives gradients
    backbone_params = list(small_ssl_model.mmoe.backbone.parameters())
    grads = [p for p in backbone_params if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(grads) > 0, "no Transformer backbone parameter received gradients"
    # MMoE experts/gates/towers receive gradients
    for component in ("experts", "gates", "towers"):
        module = getattr(small_ssl_model.mmoe, component)
        got = [
            p for p in module.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert got, f"MMoE {component} received no gradients"


def test_optimizer_updates_parameters(small_ssl_model, synthetic_batch):
    opt = torch.optim.AdamW(small_ssl_model.parameters(), lr=1e-3, weight_decay=1e-2)
    before = {n: p.detach().clone() for n, p in small_ssl_model.named_parameters()}
    for _ in range(3):
        opt.zero_grad()
        out = small_ssl_model(synthetic_batch)
        loss = ssl_joint_loss(out, synthetic_batch)
        loss.backward()
        opt.step()
    changed = [
        n for n, p in small_ssl_model.named_parameters()
        if not torch.equal(p.detach(), before[n])
    ]
    assert len(changed) >= 1, "no trainable parameter changed"
    # projection head must be among the updated parameters
    assert any(n.startswith("projection") for n in changed)


def test_two_views_differ_through_model(small_ssl_model, synthetic_batch):
    """Independently corrupted views must yield different CLS representations."""
    small_ssl_model.train()  # dropout active as in real training
    out = small_ssl_model(synthetic_batch, return_diagnostics=True)
    assert not torch.equal(out["cls_view1"], out["cls_view2"])


def test_ssl_model_state_dict_roundtrip(small_ssl_model, synthetic_batch, tmp_path):
    path = tmp_path / "ssl_ckpt.pt"
    torch.save(small_ssl_model.state_dict(), path)
    clone = SSLTransformerMMoE(
        small_ssl_model.feature_set,
        ssl_config=small_ssl_model.ssl_config,
        backbone_config=small_ssl_model.mmoe.backbone.config,
        mmoe_config=small_ssl_model.mmoe.mmoe_config,
        augmentations=small_ssl_model.augmentations,
    )
    clone.load_state_dict(torch.load(path))
    clone.eval(), small_ssl_model.eval()
    with torch.no_grad():
        a = clone.supervised_forward(synthetic_batch, return_diagnostics=False)
        b = small_ssl_model.supervised_forward(synthetic_batch, return_diagnostics=False)
    assert torch.equal(a["click_logit"], b["click_logit"])


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward_backward(small_ssl_model, synthetic_batch):
    model = small_ssl_model.cuda()
    cuda_batch = {k: v.cuda() for k, v in synthetic_batch.items()}
    out = model(cuda_batch)
    total = ssl_joint_loss(out, cuda_batch)
    total.backward()
    assert torch.isfinite(total)
    grad = next(model.projection.parameters()).grad
    assert grad is not None and torch.isfinite(grad).all()
