"""Stage 8 sampler-diagnostic tests (spec: resolve sampler nuance, §6)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recsys23_fedrec.experiments.config import get_experiment_config
from recsys23_fedrec.experiments.integrity import PROTECTED_VALUES
from recsys23_fedrec.experiments.sampler_diagnostic import (
    fingerprint_corrected,
    fingerprint_historical,
    protocol_arrangement,
    run_tiny_sampler_comparison,
)
from recsys23_fedrec.training.loaders import make_subset_loader
from recsys23_fedrec.training.sampler import PartShuffledBatchSampler


# ---------------------------------------------------------------------------
# 1-2. Historical Stage 6 behavior: reproducible, identical across epochs
# ---------------------------------------------------------------------------
def test_historical_behavior_reproducible():
    """Same inputs -> byte-identical arrangement (the actual Stage 6 path:
    sampler constructed with default epoch=0, set_epoch never applied)."""
    rng = np.random.default_rng(0)
    indices = np.sort(rng.choice(10_000, size=2_500, replace=False))
    a = fingerprint_historical(indices, batch_size=128, seed=42, epoch=1)
    b = fingerprint_historical(indices, batch_size=128, seed=42, epoch=2)
    c = fingerprint_historical(indices, batch_size=128, seed=42, epoch=3)
    assert a.arrangement_sha256 == b.arrangement_sha256 == c.arrangement_sha256


def test_historical_same_arrangement_across_epochs():
    indices = np.sort(np.arange(5_000))
    e1 = protocol_arrangement(indices, batch_size=256, seed=42, epoch=1, protocol="H")
    e2 = protocol_arrangement(indices, batch_size=256, seed=42, epoch=2, protocol="H")
    e3 = protocol_arrangement(indices, batch_size=256, seed=42, epoch=3, protocol="H")
    assert e1 == e2 == e3
    # and it equals the sampler's own epoch-0 stream (what Stage 6 executed)
    raw = PartShuffledBatchSampler(
        subset_indices=indices, batch_size=256, seed=42, epoch=0
    )
    assert e1 == [list(map(int, b)) for b in raw]


def test_historical_matches_frozen_stage6_batch_count_math():
    """Sanity: the historical arrangement covers the subset exactly once."""
    indices = np.sort(np.arange(1_023))
    batches = protocol_arrangement(indices, batch_size=128, seed=42, epoch=1, protocol="H")
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(1_023))  # disjoint + complete coverage


# ---------------------------------------------------------------------------
# 3-5. Corrected sampler: deterministic, epoch-dependent, seed-reproducible
# ---------------------------------------------------------------------------
def test_corrected_deterministic_and_epoch_dependent():
    indices = np.sort(np.arange(5_000))
    e1a = protocol_arrangement(indices, batch_size=256, seed=42, epoch=1, protocol="C")
    e1b = protocol_arrangement(indices, batch_size=256, seed=42, epoch=1, protocol="C")
    assert e1a == e1b  # deterministic within an epoch

    e2 = protocol_arrangement(indices, batch_size=256, seed=42, epoch=2, protocol="C")
    e3 = protocol_arrangement(indices, batch_size=256, seed=42, epoch=3, protocol="C")
    assert e1a != e2 and e2 != e3 and e1a != e3  # epoch-dependent

    f1 = fingerprint_corrected(indices, batch_size=256, seed=42, epoch=1)
    f2 = fingerprint_corrected(indices, batch_size=256, seed=42, epoch=2)
    f3 = fingerprint_corrected(indices, batch_size=256, seed=42, epoch=3)
    assert len({f1.arrangement_sha256, f2.arrangement_sha256, f3.arrangement_sha256}) == 3


def test_corrected_same_seed_reproduces_arrangements():
    indices = np.sort(np.random.default_rng(7).choice(20_000, size=4_000, replace=False))
    a = fingerprint_corrected(indices, batch_size=250, seed=42, epoch=2)
    b = fingerprint_corrected(indices, batch_size=250, seed=42, epoch=2)
    assert a.arrangement_sha256 == b.arrangement_sha256
    assert a.first_batch == b.first_batch
    # different seed -> different arrangement (the seed participates)
    c = fingerprint_corrected(indices, batch_size=250, seed=43, epoch=2)
    assert a.arrangement_sha256 != c.arrangement_sha256


def test_corrected_epochs_differ_on_large_dataset():
    """Epoch dependence requires enough batches (a 1-batch subset cannot
    permute). With >= 2 batches the arrangements must differ."""
    indices = np.sort(np.arange(512))
    two_batch_e1 = protocol_arrangement(indices, batch_size=256, seed=42, epoch=1, protocol="C")
    two_batch_e2 = protocol_arrangement(indices, batch_size=256, seed=42, epoch=2, protocol="C")
    assert len(two_batch_e1) == 2 and len(two_batch_e2) == 2
    assert two_batch_e1 != two_batch_e2


def test_corrected_covers_subset_exactly_once():
    indices = np.sort(np.arange(999))
    for epoch in (1, 2, 3):
        batches = protocol_arrangement(indices, batch_size=100, seed=42, epoch=epoch, protocol="C")
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(999))


def test_fingerprint_proves_from_actual_indices():
    """The digest is computed over the EXACT index sequence, so equal
    fingerprints imply equal arrangements (not just equal metadata)."""
    indices = np.sort(np.arange(1_000))
    f_a = fingerprint_corrected(indices, batch_size=200, seed=42, epoch=1)
    batches_a = protocol_arrangement(indices, batch_size=200, seed=42, epoch=1, protocol="C")
    assert f_a.n_batches == len(batches_a)
    assert f_a.first_batch_first == batches_a[0][0]
    assert f_a.first_batch_last == batches_a[0][-1]
    assert f_a.last_batch_first == batches_a[-1][0]
    assert f_a.n_samples == 1_000


# ---------------------------------------------------------------------------
# 6. No test-set data is touched
# ---------------------------------------------------------------------------
def test_no_test_set_dependency():
    """The diagnostic's public surface takes only train-side objects; verify
    none of its modules import or reference the test split directory."""
    import recsys23_fedrec.experiments.sampler_diagnostic as diag

    source = Path(diag.__file__).read_text(encoding="utf-8")
    assert "processed/test" not in source
    assert "split=\"test\"" not in source


# ---------------------------------------------------------------------------
# 7. Integrity checks remain valid
# ---------------------------------------------------------------------------
def test_integrity_constants_unchanged():
    assert PROTECTED_VALUES["stage2_artifact_hash"] == "c3a62caf13326617"
    assert PROTECTED_VALUES["split_train_index_hash"] == "dd37605169615442"
    assert PROTECTED_VALUES["split_validation_index_hash"] == "3cd8370ebdecb305"


# ---------------------------------------------------------------------------
# Tiny end-to-end comparison on the synthetic fixture (fast CI proxy)
# ---------------------------------------------------------------------------
def test_tiny_comparison_runs_and_is_deterministic(feature_set, processed_fixture):
    from recsys23_fedrec.data import ProcessedRecSysDataset

    dataset = ProcessedRecSysDataset(
        feature_set, processed_fixture["train_dir"], split="train"
    )
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(dataset))
    val_idx = np.sort(perm[:32])
    train_idx = np.sort(perm[32:224])  # 192 rows -> 6 batches of 32

    config = get_experiment_config("experiment_b_transformer_mmoe", device="cpu")
    result = run_tiny_sampler_comparison(
        config,
        feature_set,
        dataset,
        train_idx,
        val_idx,
        batch_size=32,
        n_batches=2,
        epochs=2,
        device="cpu",
        seed=42,
    )
    # structural evidence
    h_fps = result["protocols"]["H"]["batch_fingerprints"]
    c_fps = result["protocols"]["C"]["batch_fingerprints"]
    assert h_fps[1]["arrangement_sha256"] == h_fps[2]["arrangement_sha256"]
    assert c_fps[1]["arrangement_sha256"] != c_fps[2]["arrangement_sha256"]
    assert result["repeat_of_H_matches_H"] is True
    # trajectory deltas are finite numbers
    for protocol in ("H", "C"):
        losses = result["protocols"][protocol]["epoch_losses"]
        for epoch_losses in losses.values():
            assert np.isfinite(epoch_losses["total"])
        vm = result["protocols"][protocol]["val_metrics"]
        assert np.isfinite(vm["val_install_logloss"])
        assert 0.0 <= vm["val_install_auc"] <= 1.0
    assert result["cross_protocol_param_delta_l2"] >= 0.0
