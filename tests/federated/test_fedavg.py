"""Stage 10A tests: sample-weighted FedAvg utility + frozen FederatedConfig.

AllFedAvg tests use tiny artificial state dicts; one integration test
builds the real TransformerMMoE from the synthetic fixture artifact and
aggregates perturbed copies of its state dict. No training, no optimizer,
no backward pass anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
for path in (str(SRC), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recsys23_fedrec.federated.config import (  # noqa: E402
    FEDAVG_METHOD,
    LOCAL_EPOCHS,
    PARTICIPATION_ALL,
    ROUNDS,
    FederatedConfig,
    get_federated_config,
)
from recsys23_fedrec.federated.fedavg import (  # noqa: E402
    FedAvgError,
    fedavg_state_dicts,
    fedavg_weights_from_counts,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def tiny_state(scale: float = 1.0) -> "dict[str, torch.Tensor]":
    return {
        "w": torch.tensor([[scale, 2.0 * scale], [3.0 * scale, 4.0 * scale]]),
        "b": torch.tensor([scale]),
    }


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------
def test_weights_exact_for_unequal_counts():
    w = fedavg_weights_from_counts([621_801, 59_999])
    assert w[0] == pytest.approx(621_801 / 681_800)
    assert w[1] == pytest.approx(59_999 / 681_800)
    assert sum(w) == pytest.approx(1.0)


def test_weights_reject_invalid_counts():
    for bad in ([0, 10], [-5, 15], []):
        with pytest.raises(FedAvgError):
            fedavg_weights_from_counts(bad)


# ---------------------------------------------------------------------------
# exact weighted result (hand-checkable)
# ---------------------------------------------------------------------------
def test_fedavg_exact_weighted_result_two_clients():
    s0 = tiny_state(1.0)  # w = [[1,2],[3,4]], b = [1]
    s1 = tiny_state(3.0)  # w = [[3,6],[9,12]], b = [3]
    # counts 3 : 1  -> weights 0.75 / 0.25
    out = fedavg_state_dicts([s0, s1], [3, 1])
    expected_w = 0.75 * s0["w"] + 0.25 * s1["w"]
    assert torch.allclose(out["w"], expected_w)
    assert out["w"][0, 0].item() == pytest.approx(0.75 * 1.0 + 0.25 * 3.0)
    assert out["b"][0].item() == pytest.approx(0.75 * 1.0 + 0.25 * 3.0)


def test_fedavg_weighted_result_unequal_realistic_counts():
    """10-client unequal counts; weighted mean must dominate the largest client."""
    states = [tiny_state(float(i + 1)) for i in range(10)]
    counts = [621_801, 544_574, 59_999, 488_090, 69_790, 89_988, 193_852, 201_902, 511_603, 355_667]
    out = fedavg_state_dicts(states, counts)
    weights = fedavg_weights_from_counts(counts)
    expected = sum(w * s["w"].to(torch.float64) for w, s in zip(weights, states)).to(torch.float32)
    assert torch.allclose(out["w"], expected)


# ---------------------------------------------------------------------------
# compatibility validation
# ---------------------------------------------------------------------------
def test_reject_missing_key():
    s0, s1 = tiny_state(), tiny_state()
    del s1["b"]
    with pytest.raises(FedAvgError, match="keys differ"):
        fedavg_state_dicts([s0, s1], [1, 1])


def test_reject_extra_key():
    s0, s1 = tiny_state(), tiny_state()
    s1 = dict(s1)
    s1["extra"] = torch.zeros(2)
    with pytest.raises(FedAvgError, match="keys differ"):
        fedavg_state_dicts([s0, s1], [1, 1])


def test_reject_shape_mismatch():
    s0, s1 = tiny_state(), tiny_state()
    s1 = dict(s1)
    s1["w"] = torch.zeros(3, 2)
    with pytest.raises(FedAvgError, match="shape"):
        fedavg_state_dicts([s0, s1], [1, 1])


def test_reject_dtype_mismatch():
    s0, s1 = tiny_state(), tiny_state()
    s1 = dict(s1)
    s1["w"] = s1["w"].to(torch.float64)
    with pytest.raises(FedAvgError, match="dtype"):
        fedavg_state_dicts([s0, s1], [1, 1])


def test_reject_empty_and_length_mismatch():
    with pytest.raises(FedAvgError):
        fedavg_state_dicts([], [])
    s = tiny_state()
    with pytest.raises(FedAvgError, match="sample counts"):
        fedavg_state_dicts([s, s], [1])
    with pytest.raises(FedAvgError, match="non-positive"):
        fedavg_state_dicts([s, s], [0, 1])
    with pytest.raises(FedAvgError, match="non-positive"):
        fedavg_state_dicts([s, s], [-1, 1])


# ---------------------------------------------------------------------------
# non-floating policy
# ---------------------------------------------------------------------------
def test_non_floating_rejected_by_default():
    s0, s1 = tiny_state(), tiny_state()
    s0 = dict(s0)
    s1 = dict(s1)
    s0["counter"] = torch.tensor(5, dtype=torch.int64)
    s1["counter"] = torch.tensor(9, dtype=torch.int64)
    with pytest.raises(FedAvgError, match="non-floating"):
        fedavg_state_dicts([s0, s1], [1, 1])


def test_non_floating_first_policy_copies_client_0():
    s0 = dict(tiny_state())
    s1 = dict(tiny_state())
    s0["counter"] = torch.tensor(5, dtype=torch.int64)
    s1["counter"] = torch.tensor(9, dtype=torch.int64)
    out = fedavg_state_dicts([s0, s1], [1, 999], on_non_floating="first")
    assert out["counter"].item() == 5  # first client's value, not averaged
    assert out["counter"].dtype == torch.int64


def test_unknown_non_floating_policy_rejected():
    with pytest.raises(FedAvgError, match="on_non_floating"):
        fedavg_state_dicts([tiny_state()], [1], on_non_floating="average")


# ---------------------------------------------------------------------------
# immutability + determinism
# ---------------------------------------------------------------------------
def test_inputs_are_not_mutated():
    s0, s1 = tiny_state(1.0), tiny_state(3.0)
    s0_before = {k: v.clone() for k, v in s0.items()}
    s1_before = {k: v.clone() for k, v in s1.items()}
    out = fedavg_state_dicts([s0, s1], [3, 1])
    for k in s0:
        assert torch.equal(s0[k], s0_before[k])
        assert torch.equal(s1[k], s1_before[k])
    # result tensors are fresh (no aliasing into the inputs)
    assert out["w"].data_ptr() != s0["w"].data_ptr()
    assert out["w"].data_ptr() != s1["w"].data_ptr()
    # and modifying the result must not touch the inputs
    out["w"] += 100.0
    assert torch.equal(s0["w"], s0_before["w"])


def test_deterministic_repeated_aggregation_and_client_order():
    states = [tiny_state(float(i + 1)) for i in range(5)]
    counts = [50, 40, 30, 20, 10]
    a = fedavg_state_dicts(states, counts)
    b = fedavg_state_dicts(states, counts)
    for k in a:
        assert torch.equal(a[k], b[k])  # bit-exact
    # reordering clients WITH their counts changes the float accumulation
    # order (and hence can change low-order bits) — the documented contract
    # is client-order determinism, not permutation invariance.
    perm = [4, 3, 2, 1, 0]
    c = fedavg_state_dicts([states[i] for i in perm], [counts[i] for i in perm])
    assert torch.allclose(a["w"], c["w"], atol=1e-6)  # same weighted mean
    # float64 accumulation keeps the error tiny regardless of order
    assert (a["w"] - c["w"]).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# real architecture integration (no training)
# ---------------------------------------------------------------------------
def test_real_transformer_mmoe_state_dict_aggregates(feature_set):
    """The frozen architecture (138 float32 entries, no BatchNorm) averages
    exactly; state-dict entry count and dtypes are preserved."""
    from recsys23_fedrec.models.mmoe import TransformerMMoE

    m0 = TransformerMMoE(feature_set)
    m1 = TransformerMMoE(feature_set)
    with torch.no_grad():  # perturb client 1's parameters (no training)
        for p in m1.parameters():
            p.add_(0.01)
    sd0 = {k: v.detach().clone() for k, v in m0.state_dict().items()}
    sd1 = {k: v.detach().clone() for k, v in m1.state_dict().items()}
    # The frozen architecture has no BatchNorm/running stats: every state
    # entry is float32 (138 entries on the real artifact, fewer on the
    # reduced synthetic fixture — the invariant is the dtype, not the count).
    assert all(v.dtype == torch.float32 for v in sd0.values())
    assert not any("running" in k or "num_batches" in k for k in sd0)

    out = fedavg_state_dicts([sd0, sd1], [621_801, 59_999])
    assert len(out) == len(sd0)
    assert all(v.dtype == torch.float32 for v in out.values())
    w0 = 621_801 / 681_800
    w1 = 59_999 / 681_800
    for k in sd0:
        expected = w0 * sd0[k].to(torch.float64) + w1 * sd1[k].to(torch.float64)
        assert torch.allclose(out[k].to(torch.float64), expected, atol=1e-6), k
    # aggregated state loads back into a fresh model with strict=True
    m2 = TransformerMMoE(feature_set)
    m2.load_state_dict(out, strict=True)


# ---------------------------------------------------------------------------
# FederatedConfig freeze
# ---------------------------------------------------------------------------
def test_federated_config_baseline_frozen():
    cfg = get_federated_config()
    assert cfg.num_clients == 10
    assert cfg.alpha == 0.5
    assert cfg.partition_seed == 42
    assert cfg.seed == 42
    assert cfg.participation == PARTICIPATION_ALL == "all_clients_every_round"
    assert cfg.local_epochs == LOCAL_EPOCHS == 1
    assert cfg.rounds == ROUNDS == 10
    assert cfg.local_batch_size == 1024
    assert cfg.optimizer == "adamw"
    assert cfg.learning_rate == 3e-4
    assert cfg.weight_decay == 1e-2
    assert cfg.lambda_click == 1.0 and cfg.lambda_install == 1.0
    assert cfg.aggregation == FEDAVG_METHOD == "sample_weighted_fedavg"
    assert cfg.selection_metric == "global_validation_install_logloss"
    assert cfg.use_ssl is False
    assert cfg.local_optimizer_state_persistence is False
    assert cfg.local_sampler_epoch_mode == "round_minus_one"


def test_federated_config_rejects_protocol_changes():
    base = dict(FederatedConfig().to_dict())
    for field, bad in [
        ("num_clients", 5),
        ("alpha", -1.0),
        ("participation", "random_5_per_round"),
        ("local_epochs", 2),
        ("rounds", 0),
        ("optimizer", "sgd"),
        ("aggregation", "uniform_fedavg"),
        ("use_ssl", True),
        ("selection_metric", "test_install_logloss"),
        ("local_batch_size", 0),
        ("learning_rate", 0.0),
    ]:
        payload = dict(base)
        payload[field] = bad
        with pytest.raises(ValueError):
            FederatedConfig(**payload)


def test_federated_config_roundtrip():
    cfg = get_federated_config()
    assert FederatedConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
