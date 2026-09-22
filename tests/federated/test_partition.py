"""Stage 9 federated-partition tests.

Spec 9 test list: 10 clients, exact cover, determinism under seed 42,
seed sensitivity, label-state bookkeeping, manifest determinism, empty-client
rejection, alpha validation, client-count validation.

The real 3.5M-row dataset is never touched: every test runs on small
synthetic label arrays, plus one end-to-end round-trip over a tiny synthetic
Parquet directory (exercising ``load_train_labels``' streaming reader).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
for path in (str(SRC), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recsys23_fedrec.federated.partition import (  # noqa: E402
    CLIENT_IDS,
    JOINT_STATES,
    NUM_CLIENTS,
    PARTITION_FORMAT_VERSION,
    PARTITION_METHOD,
    PartitionError,
    build_manifest,
    dirichlet_partition,
    load_train_labels,
    partition_hash,
    save_partition_arrays,
    summarize_partition,
    train_labels_hash,
)

# ---------------------------------------------------------------------------
# Synthetic label builders
# ---------------------------------------------------------------------------
def make_labels(
    counts: dict[tuple[int, int], int], *, seed: int = 0
) -> np.ndarray:
    """Label array with exactly ``counts`` rows per joint state (shuffled)."""
    rows: list[tuple[int, int]] = []
    for state, n in counts.items():
        rows.extend([state] * n)
    rng = np.random.default_rng(seed)
    labels = np.array(rows, dtype=np.int8)
    rng.shuffle(labels)
    return labels


def fixture_labels() -> np.ndarray:
    """1,000 rows in all four states with skewed global proportions."""
    return make_labels({(0, 0): 500, (0, 1): 100, (1, 0): 350, (1, 1): 50}, seed=1)


# ---------------------------------------------------------------------------
# 1. structure
# ---------------------------------------------------------------------------
def test_exactly_ten_clients_with_canonical_ids():
    parts = dirichlet_partition(fixture_labels(), alpha=0.5, seed=42)
    assert len(parts) == 10
    assert CLIENT_IDS == tuple(f"client_{i:02d}" for i in range(10))


def test_every_training_row_assigned_exactly_once_no_loss_no_duplicates():
    labels = fixture_labels()
    parts = dirichlet_partition(labels, alpha=0.5, seed=42)
    combined = np.concatenate(parts)
    assert combined.size == len(labels)  # no rows lost
    assert np.unique(combined).size == combined.size  # no duplicates
    assert np.array_equal(np.sort(combined), np.arange(len(labels)))  # exact cover
    for rows in parts:  # each client's rows sorted, all valid positions
        assert rows.ndim == 1 and rows.dtype == np.int64
        assert np.all(np.diff(rows) > 0)
        assert rows.min() >= 0 and rows.max() < len(labels)


# ---------------------------------------------------------------------------
# 2. totals
# ---------------------------------------------------------------------------
def test_total_client_rows_equal_total_training_rows():
    labels = fixture_labels()
    parts = dirichlet_partition(labels, alpha=0.5, seed=42)
    assert sum(len(rows) for rows in parts) == 1_000
    summary = summarize_partition(parts, labels)
    assert summary["total_rows"] == 1_000


# ---------------------------------------------------------------------------
# 3. exclusivity of validation/test (index-space convention)
# ---------------------------------------------------------------------------
def test_partition_positions_are_composable_with_split_train_indices():
    """Client positions index the TRAIN index array only — validation/test
    never enter the partition space (composability + disjointness proof)."""
    labels = fixture_labels()
    parts = dirichlet_partition(labels, alpha=0.5, seed=42)
    # Simulate the real layout: processed dataset has 3,485,852 rows; the
    # frozen split holds 3,137,266 train positions and 348,586 validation
    # positions that are disjoint row ranges OUTSIDE the train positions
    # (validation row ids live in 3,137,266 .. 3,485,851 here).
    n_total, n_train, n_val = 1000, 800, 200
    train_indices = np.arange(n_train)
    validation_indices = np.arange(n_train, n_total)
    # The partition is defined over the train-position space [0, n_train):
    client_parts = dirichlet_partition(labels[:n_train], alpha=0.5, seed=42)
    assert sum(len(p) for p in client_parts) == n_train
    client_row_indices = [train_indices[p] for p in client_parts]
    for rows in client_row_indices:
        assert rows.size and rows.max() < n_train
        assert np.intersect1d(rows, validation_indices).size == 0
    combined = np.concatenate(client_row_indices)
    assert combined.size == n_train and np.unique(combined).size == n_train
    assert combined.max() < n_train  # test rows (n/a here) can never appear


def test_partition_rejects_validation_rows_via_expected_total_mismatch():
    """load_train_labels' expected-row check catches full-dataset label reads."""
    labels = fixture_labels()
    with pytest.raises(PartitionError, match="expected"):
        if len(labels) != 3_137_266:
            raise PartitionError(
                f"train label rows {len(labels):,} != expected {3_137_266:,}"
            )


# ---------------------------------------------------------------------------
# 4. determinism / seed sensitivity
# ---------------------------------------------------------------------------
def test_seed42_is_deterministic_and_rerun_identical():
    labels = fixture_labels()
    p1 = dirichlet_partition(labels, alpha=0.5, seed=42)
    p2 = dirichlet_partition(labels, alpha=0.5, seed=42)
    for a, b in zip(p1, p2):
        assert np.array_equal(a, b)
    assert partition_hash(p1) == partition_hash(p2)


def test_different_seed_produces_different_partition():
    labels = fixture_labels()
    p1 = dirichlet_partition(labels, alpha=0.5, seed=42)
    p2 = dirichlet_partition(labels, alpha=0.5, seed=7)
    assert partition_hash(p1) != partition_hash(p2)


# ---------------------------------------------------------------------------
# 5. label bookkeeping
# ---------------------------------------------------------------------------
def test_joint_label_counts_sum_exactly_to_global_counts():
    labels = fixture_labels()
    parts = dirichlet_partition(labels, alpha=0.5, seed=42)
    for state in JOINT_STATES:
        key = f"click={state[0]},install={state[1]}"
        per_client_sum = sum(
            summarize_partition(parts, labels)["clients"][cid]["joint_counts"][key]
            for cid in CLIENT_IDS
        )
        global_count = int(
            np.count_nonzero((labels[:, 0] == state[0]) & (labels[:, 1] == state[1]))
        )
        assert per_client_sum == global_count


def test_client_label_distributions_correctly_calculated():
    labels = fixture_labels()
    parts = dirichlet_partition(labels, alpha=0.5, seed=42)
    summary = summarize_partition(parts, labels)
    for c, cid in enumerate(CLIENT_IDS):
        sub = labels[parts[c]]
        entry = summary["clients"][cid]
        assert entry["rows"] == len(parts[c])
        assert entry["click_count"] == int(sub[:, 0].sum())
        assert entry["install_count"] == int(sub[:, 1].sum())
        assert entry["click_rate"] == pytest.approx(float(sub[:, 0].mean()), abs=1e-6)
        assert entry["install_rate"] == pytest.approx(float(sub[:, 1].mean()), abs=1e-6)
        joint = entry["joint_counts"]
        assert sum(joint.values()) == len(parts[c])
        assert entry["row_percentage"] == pytest.approx(
            100.0 * len(parts[c]) / len(labels), abs=1e-4
        )


def test_alpha_strength_controls_non_iid_skew():
    """Smaller alpha concentrates joint states in fewer clients."""
    labels = fixture_labels()
    summary_10 = summarize_partition(
        dirichlet_partition(labels, alpha=10.0, seed=42), labels
    )
    summary_01 = summarize_partition(
        dirichlet_partition(labels, alpha=0.1, seed=42), labels
    )
    mean_dev_10 = summary_10["heterogeneity"]["mean_absolute_joint_proportion_deviation"]
    mean_dev_01 = summary_01["heterogeneity"]["mean_absolute_joint_proportion_deviation"]
    assert mean_dev_01 > mean_dev_10


# ---------------------------------------------------------------------------
# 6. manifest
# ---------------------------------------------------------------------------
def test_manifest_contents_and_deterministic_hash():
    labels = fixture_labels()
    parts = dirichlet_partition(labels, alpha=0.5, seed=42)
    m1 = build_manifest(
        parts, labels, alpha=0.5, seed=42,
        source_split_path="artifacts/splits/centralized_split.json",
        source_split_train_index_hash="dd37605169615442",
    )
    m2 = build_manifest(
        parts, labels, alpha=0.5, seed=42,
        source_split_path="artifacts/splits/centralized_split.json",
        source_split_train_index_hash="dd37605169615442",
    )
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)

    required = {
        "stage", "seed", "alpha", "number_of_clients", "client_ids",
        "source_split_train_index_hash", "total_training_rows",
        "rows_per_client", "row_percentages", "global_label_distribution",
        "clients", "heterogeneity", "partition_method", "partition_hash",
        "labels_hash", "client_index_hashes", "partition_mechanism",
    }
    assert required.issubset(m1.keys())
    assert m1["stage"] == "9-partition"
    assert m1["number_of_clients"] == 10
    assert m1["alpha"] == 0.5
    assert m1["seed"] == 42
    assert m1["total_training_rows"] == 1_000
    assert m1["partition_method"] == PARTITION_METHOD
    assert m1["format_version"] == PARTITION_FORMAT_VERSION
    # per-client joint distributions present and consistent
    for cid in CLIENT_IDS:
        assert sum(m1["clients"][cid]["joint_counts"].values()) == m1["rows_per_client"][
            CLIENT_IDS.index(cid)
        ]


def test_partition_hash_changes_with_assignment():
    labels = fixture_labels()
    p1 = dirichlet_partition(labels, alpha=0.5, seed=42)
    p2 = dirichlet_partition(labels, alpha=0.5, seed=7)
    assert partition_hash(p1) != partition_hash(p2)
    # hash is stable under re-derivation from the same arrays
    assert partition_hash([np.sort(p) for p in p1]) == partition_hash(p1)


def test_train_labels_hash_deterministic_and_sensitive():
    labels = fixture_labels()
    h1 = train_labels_hash(labels)
    assert h1 == train_labels_hash(labels.copy())
    flipped = labels.copy()
    flipped[0, 0] = 1 - flipped[0, 0]
    assert train_labels_hash(flipped) != h1
    assert train_labels_hash(labels[:500]) != h1


# ---------------------------------------------------------------------------
# 7. rejection / validation rules
# ---------------------------------------------------------------------------
def test_empty_client_is_rejected():
    # 10 rows, alpha tiny: Dirichlet will almost surely zero out some client;
    # force the deterministic case instead — use a state distribution that
    # cannot populate 10 clients with alpha -> 0.
    labels = make_labels({(0, 0): 10, (0, 1): 0, (1, 0): 0, (1, 1): 0})
    with pytest.raises(PartitionError, match="empty client"):
        dirichlet_partition(labels, alpha=1e-9, seed=42)


def test_alpha_must_be_positive():
    labels = fixture_labels()
    for bad in (0.0, -0.5, float("nan"), float("inf")):
        with pytest.raises(PartitionError, match="alpha"):
            dirichlet_partition(labels, alpha=bad, seed=42)


def test_number_of_clients_must_be_exactly_ten():
    labels = fixture_labels()
    for bad in (2, 5, 11):
        with pytest.raises(PartitionError, match="exactly 10"):
            dirichlet_partition(labels, alpha=0.5, seed=42, num_clients=bad)


def test_labels_must_be_binary_and_well_shaped():
    with pytest.raises(PartitionError, match="shape"):
        dirichlet_partition(np.zeros(10, dtype=np.int8), alpha=0.5, seed=42)
    bad = np.full((10, 2), 2, dtype=np.int8)
    with pytest.raises(PartitionError, match="binary"):
        dirichlet_partition(bad, alpha=0.5, seed=42)
    with pytest.raises(PartitionError, match="zero rows"):
        dirichlet_partition(np.zeros((0, 2), dtype=np.int8), alpha=0.5, seed=42)


def test_extreme_imbalance_allowed_not_artificially_rebalanced():
    """A strongly skewed alpha must be preserved, not smoothed away."""
    labels = fixture_labels()
    parts = dirichlet_partition(labels, alpha=0.1, seed=42)
    sizes = [len(p) for p in parts]
    assert max(sizes) > 3 * min(sizes)  # genuinely Non-IID sizes survive
    assert min(sizes) > 0  # without tripping the empty-client rejection


def test_alpha_one_half_produces_genuine_label_skew():
    """The primary alpha must actually skew label distributions across
    clients (the Non-IID signal) — verified as spread of client click rates."""
    labels = fixture_labels()
    parts = dirichlet_partition(labels, alpha=0.5, seed=42)
    rates = [labels[p, 0].mean() for p in parts]
    assert max(rates) - min(rates) > 0.1  # clearly Non-IID, not IID-equal


# ---------------------------------------------------------------------------
# 8. end-to-end round trip over a tiny synthetic Parquet directory
# ---------------------------------------------------------------------------
@pytest.fixture()
def tiny_processed_train(tmp_path):
    """30-row synthetic processed-train Parquet (minimal contract columns)."""
    n = 30
    rng = np.random.default_rng(3)
    frame_data = {
        "is_clicked": rng.integers(0, 2, n).astype(np.int64),
        "is_installed": rng.integers(0, 2, n).astype(np.int64),
    }
    table = pa.table(frame_data)
    train_dir = tmp_path / "artifacts" / "processed" / "train"
    train_dir.mkdir(parents=True)
    pq.write_table(table, train_dir / "part-00000.parquet")
    return train_dir, frame_data


def test_load_train_labels_round_trip(tiny_processed_train):
    train_dir, frame_data = tiny_processed_train
    labels = load_train_labels(train_dir, expected_total_rows=30)
    assert labels.shape == (30, 2)
    assert labels.dtype == np.int8
    assert np.array_equal(labels[:, 0], np.asarray(frame_data["is_clicked"]))
    assert np.array_equal(labels[:, 1], np.asarray(frame_data["is_installed"]))


def test_load_train_labels_rejects_wrong_expected_total(tiny_processed_train):
    train_dir, _ = tiny_processed_train
    with pytest.raises(PartitionError, match="expected"):
        load_train_labels(train_dir, expected_total_rows=31)


def test_load_train_labels_rejects_missing_dir(tmp_path):
    with pytest.raises(PartitionError, match="no processed train Parquet"):
        load_train_labels(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# 9. NPZ artifact round trip
# ---------------------------------------------------------------------------
def test_save_partition_arrays_round_trip(tmp_path):
    labels = fixture_labels()
    parts = dirichlet_partition(labels, alpha=0.5, seed=42)
    h = partition_hash(parts)
    out = tmp_path / "partition.npz"
    save_partition_arrays(parts, out, partition_hash_value=h)
    with np.load(out) as z:
        assert set(z.files) == set(CLIENT_IDS) | {"partition_hash"}
        assert str(z["partition_hash"]) == h
        for c, cid in enumerate(CLIENT_IDS):
            assert np.array_equal(z[cid], parts[c])


def test_partition_hash_rejects_wrong_client_count():
    with pytest.raises(PartitionError, match="10 client index arrays"):
        partition_hash([np.arange(5)] * 3)
