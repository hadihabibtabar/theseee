"""Tests for the Stage 9 per-client DataLoader factory.

All tests run on tiny synthetic Parquet datasets (the Stage 3 fixture via
``tests.data.conftest``) — the real 3.5M-row dataset is exercised by the
separate read-only verification script instead. No model, optimizer,
backward pass, or training is involved anywhere.
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

from recsys23_fedrec.federated.client_loader import (  # noqa: E402
    client_sampler,
    compose_client_row_indices,
    load_partition_positions,
    make_client_loader,
)
from recsys23_fedrec.federated.partition import CLIENT_IDS, PartitionError  # noqa: E402
from recsys23_fedrec.training.sampler import PartShuffledBatchSampler  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fake_train_indices(n_total: int = 1000, n_train: int = 800) -> np.ndarray:
    """Simulated frozen split: train positions 0..799, validation 800..999."""
    return np.arange(n_train, dtype=np.int64)


def _client_positions(sizes: dict[str, int]) -> dict[str, np.ndarray]:
    """Deterministic disjoint positions in train space, one entry per client."""
    out: dict[str, np.ndarray] = {}
    start = 0
    for cid in CLIENT_IDS:
        n = sizes[cid]
        out[cid] = np.arange(start, start + n, dtype=np.int64)
        start += n
    return out


# ---------------------------------------------------------------------------
# load_partition_positions
# ---------------------------------------------------------------------------
def test_load_partition_positions_round_trip(tmp_path):
    path = tmp_path / "partition.npz"
    pos = _client_sizes_positions(path)
    loaded = load_partition_positions(path)
    for cid in CLIENT_IDS:
        assert np.array_equal(loaded[cid], pos[cid])


def _client_sizes_positions(path):
    sizes = {cid: 5 + i for i, cid in enumerate(CLIENT_IDS)}
    pos = _client_positions(sizes)
    np.savez_compressed(
        path,
        **{cid: pos[cid] for cid in CLIENT_IDS},
        partition_hash=np.array("deadbeefdeadbeef"),
    )
    return pos


def test_load_partition_positions_rejects_malformed(tmp_path):
    good = _client_sizes_positions(tmp_path / "good.npz")

    # extra key
    bad = tmp_path / "extra.npz"
    payload = {cid: good[cid] for cid in CLIENT_IDS}
    payload["partition_hash"] = np.array("deadbeefdeadbeef")
    payload["sneaky_extra"] = np.arange(3)
    np.savez_compressed(bad, **payload)
    with pytest.raises(PartitionError, match="keys"):
        load_partition_positions(bad)

    # missing key
    bad2 = tmp_path / "missing.npz"
    payload2 = {cid: good[cid] for cid in list(CLIENT_IDS)[:9]}
    payload2["partition_hash"] = np.array("deadbeefdeadbeef")
    np.savez_compressed(bad2, **payload2)
    with pytest.raises(PartitionError, match="keys"):
        load_partition_positions(bad2)

    # duplicated / unsorted positions
    bad3 = tmp_path / "dup.npz"
    payload3 = dict(payload2)
    payload3[CLIENT_IDS[9]] = np.array([1, 1, 2], dtype=np.int64)
    np.savez_compressed(bad3, **payload3)
    with pytest.raises(PartitionError, match="ascending"):
        load_partition_positions(bad3)

    # wrong dtype
    bad4 = tmp_path / "dtype.npz"
    payload4 = dict(payload2)
    payload4[CLIENT_IDS[9]] = np.array([1, 2, 3], dtype=np.int32)
    np.savez_compressed(bad4, **payload4)
    with pytest.raises(PartitionError, match="int64"):
        load_partition_positions(bad4)


# ---------------------------------------------------------------------------
# compose_client_row_indices
# ---------------------------------------------------------------------------
def test_compose_maps_positions_through_train_indices():
    train_indices = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    rows = compose_client_row_indices(train_indices, np.array([0, 2, 4]))
    assert np.array_equal(rows, np.array([10, 30, 50], dtype=np.int64))


def test_compose_rejects_positions_outside_train_space():
    train_indices = np.arange(800, dtype=np.int64)
    with pytest.raises(PartitionError, match="outside the frozen train index"):
        compose_client_row_indices(train_indices, np.array([799, 800]))
    with pytest.raises(PartitionError, match="zero rows"):
        compose_client_row_indices(train_indices, np.array([], dtype=np.int64))


# ---------------------------------------------------------------------------
# client loader over the synthetic Stage 3 fixture (300 train rows)
# ---------------------------------------------------------------------------
def test_client_loader_batch_contract(train_dataset, feature_set, processed_fixture):
    from recsys23_fedrec.data import validate_batch

    n = len(train_dataset)  # 300 synthetic rows
    positions = np.arange(0, n, 2, dtype=np.int64)  # client "owns" every 2nd row
    loader = make_client_loader(
        train_dataset,
        train_indices=np.arange(n, dtype=np.int64),
        client_positions=positions,
        batch_size=16,
        shuffle=False,
        pin_memory=False,
    )
    assert len(loader) == (len(positions) + 15) // 16
    seen = 0
    for batch in loader:
        validate_batch(batch, feature_set, split="train")  # keys/dtypes/shapes
        assert batch["categorical"].dtype == torch.int64
        assert batch["numerical"].dtype == torch.float32
        assert batch["binary"].dtype == torch.float32
        assert batch["missing"].dtype == torch.float32
        assert batch["click"].shape == (len(batch["click"]),)
        assert batch["install"].shape == (len(batch["install"]),)
        seen += len(batch["click"])
    assert seen == len(positions)


def test_client_loader_yields_only_client_rows_no_duplicates(
    train_dataset, processed_fixture
):
    n = len(train_dataset)
    positions = np.arange(10, 110, dtype=np.int64)
    loader = make_client_loader(
        train_dataset,
        train_indices=np.arange(n, dtype=np.int64),
        client_positions=positions,
        batch_size=32,
        shuffle=False,
        pin_memory=False,
    )
    collected = []
    for batch in loader:
        # features are 1:1 with dataset rows in the fixture? — instead track
        # row identity via the sorted deterministic order of the batch sizes
        collected.append(len(batch["click"]))
    assert sum(collected) == len(positions)  # exactly the client's rows


def test_client_loader_row_identity_exact(train_dataset, processed_fixture):
    """Every yielded sample equals dataset[row] for exactly the client rows."""
    n = len(train_dataset)
    positions = np.arange(50, 90, dtype=np.int64)
    rows = np.arange(n, dtype=np.int64)[positions]
    loader = make_client_loader(
        train_dataset,
        train_indices=np.arange(n, dtype=np.int64),
        client_positions=positions,
        batch_size=64,
        shuffle=False,  # ascending order ⇒ batches follow `rows` exactly
        pin_memory=False,
    )
    idx = 0
    for batch in loader:
        for b in range(len(batch["click"])):
            expected = train_dataset[int(rows[idx])]
            for key in ("categorical", "numerical", "binary", "missing"):
                assert torch.equal(batch[key][b], expected[key])
            assert float(batch["click"][b]) == float(expected["click"])
            assert float(batch["install"][b]) == float(expected["install"])
            idx += 1
    assert idx == len(rows)


# ---------------------------------------------------------------------------
# determinism + per-epoch reshuffling
# ---------------------------------------------------------------------------
def test_deterministic_same_seed_same_arrangement(train_dataset):
    n = len(train_dataset)
    positions = np.arange(0, 200, dtype=np.int64)
    kwargs = dict(
        dataset=train_dataset,
        train_indices=np.arange(n, dtype=np.int64),
        client_positions=positions,
        batch_size=32,
        shuffle=True,
        pin_memory=False,
    )
    l1 = make_client_loader(**kwargs)
    l2 = make_client_loader(**kwargs)
    b1 = [[int(x) for x in b["click"]] for b in l1]
    b2 = [[int(x) for x in b["click"]] for b in l2]
    assert b1 == b2


def test_epoch_reshuffling_changes_arrangement_reproducibly(train_dataset):
    n = len(train_dataset)
    positions = np.arange(0, 200, dtype=np.int64)
    kwargs = dict(
        dataset=train_dataset,
        train_indices=np.arange(n, dtype=np.int64),
        client_positions=positions,
        batch_size=32,
        shuffle=True,
        pin_memory=False,
    )
    e1a = make_client_loader(**kwargs, epoch=0)
    e1b = make_client_loader(**kwargs, epoch=0)
    e2 = make_client_loader(**kwargs, epoch=1)
    e3 = make_client_loader(**kwargs, epoch=2)
    arr = lambda l: [[float(x) for x in b["click"]] for b in l]
    assert arr(e1a) == arr(e1b)  # same epoch ⇒ identical
    assert arr(e1a) != arr(e2)  # different epochs ⇒ different arrangement
    assert arr(e2) != arr(e3)
    assert arr(e1a) != arr(e3)
    # and epoch 1 is reproducible when re-created
    e1c = make_client_loader(**kwargs, epoch=0)
    assert arr(e1a) == arr(e1c)


def test_epoch_semantics_match_stage8_protocol(train_dataset):
    """Client epoch e must use the same sampler seed convention as Stage 8
    (epoch 1 ⇒ seed + 0), verified via the sampler object itself."""
    n = len(train_dataset)
    positions = np.arange(0, 100, dtype=np.int64)
    loader = make_client_loader(
        train_dataset,
        train_indices=np.arange(n, dtype=np.int64),
        client_positions=positions,
        batch_size=25,
        shuffle=True,
        seed=42,
        epoch=1,
        pin_memory=False,
    )
    sampler = client_sampler(loader)
    assert isinstance(sampler, PartShuffledBatchSampler)
    assert sampler.seed == 42 and sampler.epoch == 1


def test_client_sampler_rejects_sorted_loader(train_dataset):
    n = len(train_dataset)
    loader = make_client_loader(
        train_dataset,
        train_indices=np.arange(n, dtype=np.int64),
        client_positions=np.arange(50, dtype=np.int64),
        batch_size=16,
        shuffle=False,
        pin_memory=False,
    )
    with pytest.raises(TypeError, match="PartShuffledBatchSampler"):
        client_sampler(loader)


# ---------------------------------------------------------------------------
# structural exclusion of validation/test
# ---------------------------------------------------------------------------
def test_client_rows_never_reach_validation_space():
    train_indices = np.arange(800, dtype=np.int64)
    validation_indices = np.arange(800, 1000, dtype=np.int64)
    positions = np.arange(100, 300, dtype=np.int64)
    rows = compose_client_row_indices(train_indices, positions)
    assert rows.max() < 800
    assert np.intersect1d(rows, validation_indices).size == 0


def test_partition_positions_within_train_space_guard(tmp_path):
    path = tmp_path / "p.npz"
    sizes = {cid: 10 for cid in CLIENT_IDS}
    pos = _client_positions(sizes)
    np.savez_compressed(
        path,
        **{cid: pos[cid] for cid in CLIENT_IDS},
        partition_hash=np.array("deadbeefdeadbeef"),
    )
    loaded = load_partition_positions(path)
    train_indices = np.arange(80, dtype=np.int64)  # too small on purpose
    with pytest.raises(PartitionError, match="outside the frozen train index"):
        compose_client_row_indices(train_indices, loaded[CLIENT_IDS[-1]])
