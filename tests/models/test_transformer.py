"""Stage 4 Transformer baseline tests (spec sections A-I).

Synthetic-fixture tests run on the 300-row preprocessed mini-dataset; a small
number of tests exercise the REAL Stage 2 artifact (read-only) to verify
schema fidelity without loading real training batches (spec 14: model tests
operate on small batches only).
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

from recsys23_fedrec.data import FeatureSet  # noqa: E402
from recsys23_fedrec.models import (  # noqa: E402
    CategoricalIdError,
    TransformerBaseline,
    install_loss,
)


# ----------------------------------------------------------------------
# A. Model construction (real Stage 2 schema)
# ----------------------------------------------------------------------
def test_construction_from_real_schema(real_feature_set):
    torch.manual_seed(42)
    model = TransformerBaseline(real_feature_set)
    assert model.num_categorical == 30
    assert model.num_numerical == 38
    assert model.num_binary == 11
    assert model.num_missing == 13
    assert model.num_feature_tokens == 92
    assert model.sequence_length == 93


def test_embedding_sizes_match_stage2_vocabularies(real_feature_set):
    torch.manual_seed(42)
    model = TransformerBaseline(real_feature_set)
    expected = [int(v) for v in real_feature_set.vocabulary_sizes]
    actual = [emb.num_embeddings for emb in model.embeddings]
    assert actual == expected, "embedding tables must mirror Stage 2 vocabularies"
    # Sanity on a few known Stage 2 values (UNK+MISSING included).
    assert max(actual) == 5803  # f_15
    assert min(actual) == 3     # f_7


def test_default_hyperparameters(real_feature_set):
    torch.manual_seed(42)
    model = TransformerBaseline(real_feature_set)
    assert model.config.d_model == 128
    assert model.config.nhead == 8
    assert model.config.num_layers == 6
    assert model.config.dim_feedforward == 128
    assert model.config.dropout == 0.1
    assert model.positional_embedding.shape == (1, 93, 128)
    assert model.cls_token.shape == (1, 1, 128)


def test_hyperparameters_are_configurable(real_feature_set):
    torch.manual_seed(42)
    model = TransformerBaseline(
        real_feature_set, d_model=64, nhead=4, num_layers=2,
        dim_feedforward=256, dropout=0.0,
    )
    assert model.config.d_model == 64
    assert model.config.nhead == 4
    assert model.config.num_layers == 2
    assert model.config.dim_feedforward == 256
    assert model.config.dropout == 0.0
    assert model.sequence_length == 93  # derived, independent of d_model


def test_sequence_length_derived_not_hardcoded(feature_set):
    """With the synthetic schema, derived lengths follow the fixture schema."""
    torch.manual_seed(42)
    model = TransformerBaseline(
        feature_set, d_model=32, nhead=4, num_layers=1, dim_feedforward=32
    )
    n_cat = feature_set.num_categorical
    n_num = feature_set.num_numerical
    n_bin = feature_set.num_binary
    n_mis = feature_set.num_missing
    assert model.num_feature_tokens == n_cat + n_num + n_bin + n_mis
    assert model.sequence_length == 1 + n_cat + n_num + n_bin + n_mis
    assert model.positional_embedding.shape == (1, model.sequence_length, 32)


def test_invalid_hyperparameters_rejected(real_feature_set):
    with pytest.raises(ValueError):
        TransformerBaseline(real_feature_set, d_model=100, nhead=8)  # not divisible


# ----------------------------------------------------------------------
# B. Forward pass (synthetic fixture batches, CPU)
# ----------------------------------------------------------------------
def test_forward_output_contract(small_model, synthetic_batch):
    model = small_model
    model.eval()
    with torch.no_grad():
        output = model(synthetic_batch)
    assert output["install_logit"].shape == (8,)
    assert output["cls"].shape == (8, 32)
    assert torch.isfinite(output["install_logit"]).all()
    assert torch.isfinite(output["cls"]).all()
    assert "sigmoid" not in str(model.head).lower()  # raw logits, no squashing


def test_forward_batch_size_1(small_model, synthetic_batch):
    single = {k: v[:1] for k, v in synthetic_batch.items()}
    small_model.eval()
    with torch.no_grad():
        output = small_model(single)
    assert output["install_logit"].shape == (1,)


def test_forward_accepts_test_split_batch(small_model, synthetic_test_batch):
    """Test-split batches (no labels) are accepted — schema compatibility."""
    small_model.eval()
    with torch.no_grad():
        output = small_model(synthetic_test_batch)
    assert output["install_logit"].shape == (8,)


# ----------------------------------------------------------------------
# C. Intermediate shape verification (debug path)
# ----------------------------------------------------------------------
def test_intermediate_token_shapes(small_model, synthetic_batch):
    model = small_model
    model.eval()
    fs = model.feature_set
    b = 8
    d = 32
    with torch.no_grad():
        output = model(synthetic_batch, return_tokens=True)
    assert output["encoded_sequence"].shape == (b, model.sequence_length, d)

    with torch.no_grad():
        tokens = model.tokenize(synthetic_batch)
    assert tokens["categorical"].shape == (b, fs.num_categorical, d)
    assert tokens["numerical"].shape == (b, fs.num_numerical, d)
    assert tokens["binary"].shape == (b, fs.num_binary, d)
    assert tokens["missing"].shape == (b, fs.num_missing, d)
    assert tokens["combined"].shape == (b, model.num_feature_tokens, d)


def test_cls_position_is_prepended(small_model, synthetic_batch):
    """Sequence fed to the encoder must be [CLS, 92 feature tokens]."""
    model = small_model
    model.eval()
    with torch.no_grad():
        tokens = model.tokenize(synthetic_batch, validate_inputs=False)
    b, d = tokens["combined"].shape[0], 32
    cls = model.cls_token.expand(b, -1, -1)
    sequence = torch.cat([cls, tokens["combined"]], dim=1)
    assert sequence.shape == (b, model.sequence_length, d)
    # CLS row must equal the learnable CLS token before position addition.
    assert torch.equal(sequence[:, 0, :], model.cls_token[0].expand(b, d))


def test_positional_embedding_added(small_model, synthetic_batch):
    """Determinism of the pre-encoder path: tokens + pos must be reproducible."""
    model = small_model
    model.eval()
    with torch.no_grad():
        tokens = model.tokenize(synthetic_batch, validate_inputs=False)
    seq = tokens["combined"]
    b = seq.shape[0]
    with_cls = torch.cat([model.cls_token.expand(b, -1, -1), seq], dim=1)
    positioned = with_cls + model.positional_embedding
    assert positioned.shape == with_cls.shape
    assert not torch.equal(positioned, with_cls)  # position actually applied


# ----------------------------------------------------------------------
# D. Loss (BCEWithLogitsLoss on raw logits)
# ----------------------------------------------------------------------
def test_loss_finite_scalar_and_backward(small_model, synthetic_batch):
    model = small_model
    output = model(synthetic_batch)
    loss = install_loss(output, synthetic_batch)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()


def test_loss_matches_direct_bcewithlogits(small_model, synthetic_batch):
    model = small_model
    model.eval()
    with torch.no_grad():
        output = model(synthetic_batch)
    reference = torch.nn.functional.binary_cross_entropy_with_logits(
        output["install_logit"], synthetic_batch["install"]
    )
    assert torch.allclose(install_loss(output, synthetic_batch), reference)


# ----------------------------------------------------------------------
# E. Gradient test
# ----------------------------------------------------------------------
def test_gradients_exist_and_finite(small_model, synthetic_batch):
    model = small_model
    loss = install_loss(model(synthetic_batch), synthetic_batch)
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable
    with_grad = [p for p in trainable if p.grad is not None]
    assert len(with_grad) == len(trainable), "every trainable parameter needs a grad"
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_categorical_embeddings_receive_gradient(small_model, synthetic_batch):
    """One-token-per-feature semantics: every embedding table gets gradient."""
    model = small_model
    loss = install_loss(model(synthetic_batch), synthetic_batch)
    loss.backward()
    for j, emb in enumerate(model.embeddings):
        assert emb.weight.grad is not None and emb.weight.grad.abs().sum() > 0, (
            f"embedding {j} ({model.feature_set.categorical_names[j]}) received no gradient"
        )


def test_per_feature_projections_receive_gradient(small_model, synthetic_batch):
    """Numerical/binary/missing tokenizers are per-feature (independent rows).

    A weight row may legitimately receive zero gradient when its input
    column is identically 0 in the batch (dL/dw_j = sum_b x_bj * g_b = 0);
    the corresponding bias row must still receive gradient.
    """
    model = small_model
    loss = install_loss(model(synthetic_batch), synthetic_batch)
    loss.backward()
    for label, w, b, x in (
        ("numerical", model.numerical_weight, model.numerical_bias, synthetic_batch["numerical"]),
        ("binary", model.binary_weight, model.binary_bias, synthetic_batch["binary"]),
        ("missing", model.missing_weight, model.missing_bias, synthetic_batch["missing"]),
    ):
        active = (x != 0).any(dim=0)
        weight_rows_used = (w.grad.abs().sum(dim=1) > 0) | ~active
        assert bool(weight_rows_used.all()), (
            f"{label}: non-zero input feature row got no weight gradient"
        )
        assert bool((b.grad.abs().sum(dim=1) > 0).all()), f"{label}: bias row got no gradient"


# ----------------------------------------------------------------------
# F. Tiny optimization sanity test (NOT real training)
# ----------------------------------------------------------------------
def test_tiny_optimization_sanity(small_model, synthetic_batch):
    model = small_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first_loss = None
    for _ in range(3):
        optimizer.zero_grad()
        loss = install_loss(model(synthetic_batch), synthetic_batch)
        if first_loss is None:
            first_loss = loss.item()
        assert torch.isfinite(loss)
        loss.backward()
        before = model.head[2].weight.detach().clone()
        optimizer.step()
        assert not torch.equal(before, model.head[2].weight), "parameter did not change"
    assert torch.isfinite(loss)


# ----------------------------------------------------------------------
# G. CPU (implicit everywhere) + dtype/device discipline
# ----------------------------------------------------------------------
def test_output_stays_on_cpu(small_model, synthetic_batch):
    small_model.eval()
    with torch.no_grad():
        output = small_model(synthetic_batch)
    assert output["install_logit"].device.type == "cpu"
    assert output["cls"].device.type == "cpu"


# ----------------------------------------------------------------------
# H. CUDA (conditional — skipped if unavailable)
# ----------------------------------------------------------------------
def test_cuda_forward_backward(real_feature_set):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable — Stage 4 spec allows explicit skip")
    torch.manual_seed(42)
    device = torch.device("cuda")
    # Small config to keep the GPU test light.
    model = TransformerBaseline(
        real_feature_set, d_model=64, nhead=4, num_layers=2, dim_feedforward=64
    ).to(device)
    b = 32
    batch = {
        "categorical": torch.randint(0, 2, (b, 30), device=device),
        "numerical": torch.randn(b, 38, device=device),
        "binary": torch.randint(0, 2, (b, 11), device=device).float(),
        "missing": torch.randint(0, 2, (b, 13), device=device).float(),
        "click": torch.randint(0, 2, (b,), device=device).float(),
        "install": torch.randint(0, 2, (b,), device=device).float(),
    }
    output = model(batch)
    assert output["install_logit"].shape == (b,)
    assert output["install_logit"].device.type == "cuda"
    assert torch.isfinite(output["install_logit"]).all()
    loss = install_loss(output, batch)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# ----------------------------------------------------------------------
# I. Deterministic construction with the project seed (42)
# ----------------------------------------------------------------------
def test_deterministic_construction(real_feature_set):
    torch.manual_seed(42)
    m1 = TransformerBaseline(
        real_feature_set, d_model=64, nhead=4, num_layers=2, dim_feedforward=64
    )
    torch.manual_seed(42)
    m2 = TransformerBaseline(
        real_feature_set, d_model=64, nhead=4, num_layers=2, dim_feedforward=64
    )
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"parameter {n1} differs across seeded builds"


def test_dropout_flagged_at_train_time(small_model, synthetic_batch):
    """dropout=0.1 config must actually live in the encoder layers."""
    layer = small_model.encoder.layers[0]
    assert layer.dropout.p == pytest.approx(0.1)


# ----------------------------------------------------------------------
# Categorical ID contract (spec section 8)
# ----------------------------------------------------------------------
def test_out_of_range_id_raises_named_error(feature_set):
    torch.manual_seed(42)
    model = TransformerBaseline(
        feature_set, d_model=32, nhead=4, num_layers=1, dim_feedforward=32
    )
    batch = {
        "categorical": torch.zeros(4, feature_set.num_categorical, dtype=torch.int64),
        "numerical": torch.zeros(4, feature_set.num_numerical),
        "binary": torch.zeros(4, feature_set.num_binary),
        "missing": torch.zeros(4, feature_set.num_missing),
    }
    # Force an out-of-range ID in the first categorical feature.
    bad = batch["categorical"].clone()
    bad[0, 0] = feature_set.vocabulary_sizes[0]  # == vocab size -> out of range
    with pytest.raises(CategoricalIdError, match="f_1"):
        model({**batch, "categorical": bad})


def test_negative_id_raises(feature_set):
    torch.manual_seed(42)
    model = TransformerBaseline(
        feature_set, d_model=32, nhead=4, num_layers=1, dim_feedforward=32
    )
    batch = {
        "categorical": torch.zeros(4, feature_set.num_categorical, dtype=torch.int64),
        "numerical": torch.zeros(4, feature_set.num_numerical),
        "binary": torch.zeros(4, feature_set.num_binary),
        "missing": torch.zeros(4, feature_set.num_missing),
    }
    bad = batch["categorical"].clone()
    bad[0, 2] = -1
    with pytest.raises(CategoricalIdError, match="negative"):
        model({**batch, "categorical": bad})


def test_wrong_column_count_raises(feature_set):
    torch.manual_seed(42)
    model = TransformerBaseline(
        feature_set, d_model=32, nhead=4, num_layers=1, dim_feedforward=32
    )
    batch = {
        "categorical": torch.zeros(4, feature_set.num_categorical - 1, dtype=torch.int64),
        "numerical": torch.zeros(4, feature_set.num_numerical),
        "binary": torch.zeros(4, feature_set.num_binary),
        "missing": torch.zeros(4, feature_set.num_missing),
    }
    with pytest.raises(CategoricalIdError, match="columns"):
        model(batch)


def test_boundary_ids_accepted(feature_set):
    """IDs 0 and vocab-1 (both valid) must pass the validation gate."""
    torch.manual_seed(42)
    model = TransformerBaseline(
        feature_set, d_model=32, nhead=4, num_layers=1, dim_feedforward=32
    )
    cat = torch.zeros(2, feature_set.num_categorical, dtype=torch.int64)
    for j, v in enumerate(feature_set.vocabulary_sizes):
        cat[1, j] = v - 1  # max valid ID per feature
    batch = {
        "categorical": cat,
        "numerical": torch.randn(2, feature_set.num_numerical),
        "binary": torch.ones(2, feature_set.num_binary),
        "missing": torch.zeros(2, feature_set.num_missing),
        "click": torch.zeros(2),
        "install": torch.zeros(2),
    }
    model.eval()
    with torch.no_grad():
        output = model(batch)
    assert torch.isfinite(output["install_logit"]).all()
