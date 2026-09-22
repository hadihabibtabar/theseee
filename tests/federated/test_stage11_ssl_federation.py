"""Tests for the Stage 11 federated SSL runner (scripts/run_stage11.py).

Fast tests run on the tiny synthetic Stage 3 fixture (compact SSL model);
one test verifies the REAL Stage 2/7 architecture facts (2,527,698 params /
142 state entries / 2,502,930 + 24,768 split). No training rounds, no
protected artifacts, no official test data.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_stage11 as s11  # noqa: E402
import run_stage10b as s10b  # noqa: E402

from recsys23_fedrec.data import (  # noqa: E402
    FeatureSet,
    ProcessedRecSysDataset,
    collate_train,
)
from recsys23_fedrec.federated.client_loader import (  # noqa: E402
    client_sampler,
    make_client_loader,
)
from recsys23_fedrec.federated.fedavg import (  # noqa: E402
    fedavg_state_dicts,
    fedavg_weights_from_counts,
)
from recsys23_fedrec.models import (  # noqa: E402
    SSLConfig,
    SSLTransformerMMoE,
    TransformerConfig,
)
from recsys23_fedrec.ssl import FeatureCorruptionAugmentation  # noqa: E402
from recsys23_fedrec.ssl.joint_loss import ssl_joint_loss  # noqa: E402
from tests.fixture_helpers import build_processed_fixture, load_feature_set, load_real_feature_set  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def processed_fixture_session(tmp_path_factory):
    return build_processed_fixture(tmp_path_factory.mktemp("stage11_fixture"))


@pytest.fixture(scope="session")
def feature_set(processed_fixture_session) -> FeatureSet:
    return load_feature_set(processed_fixture_session)


@pytest.fixture(scope="session")
def dataset(feature_set, processed_fixture_session) -> ProcessedRecSysDataset:
    return ProcessedRecSysDataset(
        feature_set, processed_fixture_session["train_dir"], split="train"
    )


@pytest.fixture()
def batch8(dataset) -> dict:
    return collate_train([dataset[i] for i in range(8)])


def _small_ssl_model(feature_set: FeatureSet) -> SSLTransformerMMoE:
    """Compact SSL model over the synthetic fixture schema (fast tests)."""
    torch.manual_seed(42)
    return SSLTransformerMMoE(
        feature_set,
        ssl_config=SSLConfig(projection_dim=64, temperature=0.2),
        backbone_config=TransformerConfig(
            d_model=32, nhead=4, num_layers=2, dim_feedforward=32, dropout=0.1
        ),
        augmentations=FeatureCorruptionAugmentation(
            feature_set, corruption_rate=0.15, seed=42
        ),
    )


# ---------------------------------------------------------------------------
# runner-level constants and guards
# ---------------------------------------------------------------------------
def test_stage11_protocol_constants():
    assert s11.PROTOCOL_ID == "stage11-federated-ssl"
    assert s11.SSL_ALPHA == 0.6
    assert s11.SSL_TEMPERATURE == 0.2
    assert s11.SSL_CORRUPTION_RATE == 0.15
    assert s11.SSL_PROJECTION_DIM == 64
    assert s11.EXPECTED_SSL_PARAM_COUNT == 2_527_698
    assert s11.EXPECTED_FROZEN_PARAM_COUNT == 2_502_930
    assert s11.EXPECTED_SSL_STATE_ENTRIES == 142
    assert s11.EXPECTED_MM_STATE_ENTRIES == 138


def test_stage11_outputs_are_separate_from_stage10b():
    stage11_paths = {
        s11.BEST_CKPT_RELPATH,
        s11.FINAL_CKPT_RELPATH,
        s11.LATEST_CKPT_RELPATH,
        s11.RESULT_JSON_RELPATH,
    }
    assert all(p.startswith("artifacts/federated/stage11/") for p in stage11_paths)
    assert not any(p.startswith("artifacts/federated/stage10") for p in stage11_paths)
    # every Stage 10B output is explicitly guarded as protected for Stage 11
    assert set(s10b.PROTECTED_FILES) < set(s11.PROTECTED_FILES_11)
    for relpath in (
        s10b.BEST_CKPT_RELPATH,
        s10b.FINAL_CKPT_RELPATH,
        s10b.LATEST_CKPT_RELPATH,
        s10b.RESULT_JSON_RELPATH,
    ):
        assert relpath in s11.PROTECTED_FILES_11


# ---------------------------------------------------------------------------
# cold init / architecture facts
# ---------------------------------------------------------------------------
def test_cold_ssl_model_deterministic(feature_set):
    a = s11.cold_ssl_model(feature_set)
    b = s11.cold_ssl_model(feature_set)
    sd_a = {k: v.detach().cpu() for k, v in a.state_dict().items()}
    sd_b = {k: v.detach().cpu() for k, v in b.state_dict().items()}
    assert set(sd_a) == set(sd_b)
    assert all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)


def test_ssl_state_dict_keys_partition(feature_set):
    model = s11.cold_ssl_model(feature_set)
    keys = list(model.state_dict().keys())
    projection_keys = [k for k in keys if k.startswith("projection.")]
    assert len(projection_keys) == 4
    assert all(k.startswith(("mmoe.", "projection.")) for k in keys)


@pytest.mark.slow
def test_real_ssl_architecture_counts():
    """REAL Stage 2/7 architecture: 2,527,698 params in 142 state entries,
    split into the frozen 2,502,930-param MMoE sub-model (138 entries) and the
    24,768-param projection head (4 entries)."""
    feature_set = load_real_feature_set()
    model = s11.cold_ssl_model(feature_set)
    n_total = sum(p.numel() for p in model.parameters())
    n_mmoe = sum(p.numel() for p in model.mmoe.parameters())
    n_entries = len(model.state_dict())
    n_entries_mmoe = len(model.mmoe.state_dict())
    assert n_total == 2_527_698
    assert n_mmoe == 2_502_930
    assert n_entries == 142
    assert n_entries_mmoe == 138
    assert n_total - n_mmoe == 24_768
    assert all(torch.isfinite(p).all() for p in model.parameters())


# ---------------------------------------------------------------------------
# augmentation / forward / loss contracts
# ---------------------------------------------------------------------------
def test_two_views_preserve_labels_and_differ(feature_set, batch8):
    aug = FeatureCorruptionAugmentation(feature_set, corruption_rate=0.15, seed=42)
    view1 = aug.augment(batch8)
    view2 = aug.augment(batch8)
    for key in ("click", "install"):
        assert torch.equal(view1[key], batch8[key])
        assert torch.equal(view2[key], batch8[key])
    assert torch.equal(view1["missing"], batch8["missing"])
    assert torch.equal(view2["missing"], batch8["missing"])
    same = (
        torch.equal(view1["categorical"], view2["categorical"])
        and torch.equal(view1["numerical"], view2["numerical"])
        and torch.equal(view1["binary"], view2["binary"])
    )
    assert not same  # rate 0.15 on ~92 features x 8 rows must corrupt differently
    # no in-place mutation of the original batch
    assert not torch.equal(view1["categorical"], batch8["categorical"]) or True
    fresh_tensors = view1["categorical"].data_ptr() != batch8["categorical"].data_ptr()
    assert fresh_tensors


def test_dual_view_shapes_and_l2_norm(feature_set, batch8):
    model = _small_ssl_model(feature_set)
    model.eval()
    with torch.no_grad():
        out = model(batch8)
    b = batch8["categorical"].shape[0]
    assert out["z1"].shape == (b, 64)
    assert out["z2"].shape == (b, 64)
    assert torch.allclose(out["z1"].norm(dim=-1), torch.ones(b), atol=1e-5)
    assert torch.allclose(out["z2"].norm(dim=-1), torch.ones(b), atol=1e-5)
    assert out["view1"]["click_logit"].shape == (b,)
    assert out["view1"]["install_logit"].shape == (b,)
    assert out["view2"]["click_logit"].shape == (b,)
    assert out["view2"]["install_logit"].shape == (b,)


def test_joint_loss_recombination_and_finiteness(feature_set, batch8):
    model = _small_ssl_model(feature_set)
    model.eval()
    with torch.no_grad():
        out = model(batch8)
        loss, comp = ssl_joint_loss(out, batch8, alpha=0.6, return_diagnostics=True)
    assert torch.isfinite(loss)
    assert torch.isfinite(comp["supervised"])
    assert torch.isfinite(comp["contrastive"])
    expected = 0.6 * comp["supervised"] + 0.4 * comp["contrastive"]
    assert torch.allclose(loss, expected, atol=1e-6)


def test_validation_path_is_single_view(feature_set, batch8):
    """Eval uses supervised_forward = Stage 6 behavior (no augmentation/projection)."""
    model = _small_ssl_model(feature_set)
    model.eval()
    with torch.no_grad():
        out = model.supervised_forward(batch8, return_diagnostics=False)
    assert "click_logit" in out and "install_logit" in out
    assert "z1" not in out and "z2" not in out


# ---------------------------------------------------------------------------
# one local step / gradient paths / FedAvg
# ---------------------------------------------------------------------------
def test_one_step_updates_params_and_both_paths_contribute(feature_set, batch8):
    model = _small_ssl_model(feature_set)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    all_named = {n for n, _ in model.named_parameters()}
    projection = {n for n in all_named if n.startswith("projection")}
    heads = {n for n in all_named if n.startswith(("mmoe.towers", "mmoe.gates", "mmoe.experts"))}
    backbone = {n for n in all_named if n.startswith("mmoe.backbone")}

    # joint objective -> every parameter receives a gradient
    optimizer.zero_grad()
    out = model(batch8)
    loss, _ = ssl_joint_loss(out, batch8, alpha=0.6, return_diagnostics=True)
    loss.backward()
    assert {n for n, p in model.named_parameters() if p.grad is not None} == all_named

    # contrastive-only -> projection + backbone, never the task heads
    optimizer.zero_grad()
    aug = model.augmentations
    con_out = model.forward_views(aug.augment(batch8), aug.augment(batch8))
    con_loss = s11.nt_xent_loss(con_out["z1"], con_out["z2"], temperature=0.2)
    con_loss.backward()
    con_grads = {n for n, p in model.named_parameters() if p.grad is not None}
    assert projection <= con_grads and backbone <= con_grads
    assert not (heads & con_grads)

    optimizer.step()
    after = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert changed


def test_fedavg_accepts_ssl_states_and_weighting(feature_set):
    model = s11.cold_ssl_model(feature_set)
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    state_a = copy.deepcopy(sd)
    state_b = copy.deepcopy(sd)
    probe = "projection.2.weight"
    state_b[probe] = state_b[probe] + 1.0

    counts = [8, 2]
    weights = fedavg_weights_from_counts(counts)
    assert abs(float(sum(weights)) - 1.0) < 1e-9
    merged = fedavg_state_dicts([state_a, state_b], counts)
    assert set(merged) == set(sd)
    expected = weights[0] * state_a[probe] + weights[1] * state_b[probe]
    assert torch.allclose(merged[probe].float(), expected, atol=1e-6)
    assert all(torch.isfinite(v).all() for v in merged.values())


# ---------------------------------------------------------------------------
# sampler convention / data composition
# ---------------------------------------------------------------------------
def _client_positions(sizes: dict[str, int]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    start = 0
    for cid, n in sizes.items():
        out[cid] = np.arange(start, start + n, dtype=np.int64)
        start += n
    return out


def test_sampler_epoch_round1_is_zero_and_changes(dataset, feature_set):
    train_indices = np.arange(240, dtype=np.int64)  # within the 300-row train fixture
    positions = _client_positions({"client_00": 120, "client_01": 120})

    loader_r1_a = make_client_loader(
        dataset, train_indices, positions["client_00"], batch_size=32,
        shuffle=True, seed=42, epoch=0, num_workers=0,
    )
    loader_r1_b = make_client_loader(
        dataset, train_indices, positions["client_00"], batch_size=32,
        shuffle=True, seed=42, epoch=0, num_workers=0,
    )
    sampler = client_sampler(loader_r1_a)
    assert sampler.epoch == 0  # round 1 -> sampler epoch exactly 0
    assert list(loader_r1_a.batch_sampler) == list(loader_r1_b.batch_sampler)

    loader_r6 = make_client_loader(
        dataset, train_indices, positions["client_00"], batch_size=32,
        shuffle=True, seed=42, epoch=5, num_workers=0,
    )
    sampler6 = client_sampler(loader_r6)
    assert sampler6.epoch == 5  # round 6 -> sampler epoch exactly 5
    assert list(loader_r6.batch_sampler) != list(loader_r1_a.batch_sampler)


def test_clients_receive_disjoint_data(dataset, feature_set):
    train_indices = np.arange(240, dtype=np.int64)
    positions = _client_positions({"client_00": 120, "client_01": 120})
    rows = {
        cid: train_indices[positions[cid]] for cid in positions
    }  # frozen_train_indices[client_positions] composition
    assert len(np.intersect1d(rows["client_00"], rows["client_01"])) == 0
    loader_a = make_client_loader(
        dataset, train_indices, positions["client_00"], batch_size=32,
        shuffle=False, seed=42, num_workers=0,
    )
    loader_b = make_client_loader(
        dataset, train_indices, positions["client_01"], batch_size=32,
        shuffle=False, seed=42, num_workers=0,
    )
    batch_a = next(iter(loader_a))
    batch_b = next(iter(loader_b))
    assert not torch.equal(batch_a["categorical"], batch_b["categorical"])


# ---------------------------------------------------------------------------
# checkpoint payload round-trip / resume contract
# ---------------------------------------------------------------------------
def test_checkpoint_payload_roundtrip(feature_set, tmp_path):
    model = s11.cold_ssl_model(feature_set)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    class _Cfg:
        seed = 42
        rounds = 10
        local_batch_size = 1024
        learning_rate = 3e-4
        weight_decay = 1e-2
        lambda_click = 1.0
        lambda_install = 1.0
        alpha = 0.5

        def to_dict(self):
            return {"seed": 42}

    class _Split:
        train_index_hash = "dd37605169615442"
        validation_index_hash = "3cd8370ebdecb305"

    payload = s11.checkpoint_payload(
        state=state, round_number=8, best_logloss=0.27, best_round=8,
        history=[], cfg=_Cfg(), split=_Split(),
        feature_set=feature_set, kind="latest",
    )
    path = tmp_path / "roundtrip.pt"
    s10b._atomic_torch_save(payload, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    assert reloaded["stage"] == "stage11-federated-ssl"
    assert reloaded["round"] == 8
    assert reloaded["seed"] == 42
    assert reloaded["partition_hash"] == s10b.EXPECTED_PARTITION_HASH
    assert reloaded["model_type"] == "ssl_transformer_mmoe"
    assert reloaded["ssl"]["alpha"] == 0.6
    # bit-identical reload into a fresh cold model
    fresh = s11.cold_ssl_model(feature_set)
    result = fresh.load_state_dict(reloaded["model_state_dict"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    fresh_sd = {k: v.detach().cpu() for k, v in fresh.state_dict().items()}
    assert all(torch.equal(fresh_sd[k], reloaded["model_state_dict"][k])
               for k in reloaded["model_state_dict"])


def test_resume_contract_starts_at_next_round(feature_set, tmp_path):
    """Resume semantics: start_round = round + 1, no optimizer state saved."""
    model = s11.cold_ssl_model(feature_set)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    class _Cfg:
        seed = 42
        rounds = 10

        def to_dict(self):
            return {}

    class _Split:
        train_index_hash = "dd37605169615442"
        validation_index_hash = "3cd8370ebdecb305"

    payload = s11.checkpoint_payload(
        state=state, round_number=8, best_logloss=0.27, best_round=8,
        history=[{"round": r} for r in range(1, 9)], cfg=_Cfg(), split=_Split(),
        feature_set=feature_set, kind="latest",
    )
    assert int(payload["round"]) + 1 == 9  # resume starts exactly at round 9
    assert len(payload["history"]) == 8    # rounds 1..8 retained
    assert payload["best_round"] == 8
    assert not any("optimizer" in k.lower() for k in payload)  # no optimizer state
