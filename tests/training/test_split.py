"""Stage 6 split tests (spec 21a)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
for path in (str(SRC), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recsys23_fedrec.training.split import (  # noqa: E402
    SplitMetadataError,
    make_split,
    load_split,
)


def test_split_is_deterministic():
    s1 = make_split(300, validation_ratio=0.1, seed=42)
    s2 = make_split(300, validation_ratio=0.1, seed=42)
    assert np.array_equal(s1.train_indices, s2.train_indices)
    assert np.array_equal(s1.validation_indices, s2.validation_indices)


def test_split_sizes_and_ratio():
    split = make_split(3_485_852, validation_ratio=0.10, seed=42)
    expected_val = int(np.ceil(3_485_852 * 0.10))
    assert split.validation_rows == expected_val
    assert split.train_rows == 3_485_852 - expected_val
    assert split.total_rows == 3_485_852


def test_disjoint_and_complete():
    split = make_split(1000, validation_ratio=0.1, seed=42)
    split.validate()  # raises on violation
    combined = np.union1d(split.train_indices, split.validation_indices)
    assert combined[0] == 0 and combined[-1] == 999
    assert combined.size == 1000


def test_hash_stable_across_regeneration():
    s1 = make_split(500, validation_ratio=0.1, seed=42)
    s2 = make_split(500, validation_ratio=0.1, seed=42)
    assert s1.train_index_hash == s2.train_index_hash
    assert s1.validation_index_hash == s2.validation_index_hash
    s3 = make_split(500, validation_ratio=0.1, seed=43)
    assert s1.train_index_hash != s3.train_index_hash


def test_seed_changes_split():
    s1 = make_split(400, validation_ratio=0.1, seed=42)
    s2 = make_split(400, validation_ratio=0.1, seed=7)
    assert not np.array_equal(np.sort(s1.train_indices), np.sort(s2.train_indices)) or not np.array_equal(
        np.sort(s1.validation_indices), np.sort(s2.validation_indices)
    )


def test_save_load_round_trip(tmp_path):
    split = make_split(300, validation_ratio=0.1, seed=42)
    path = split.save(tmp_path / "centralized_split")
    loaded = load_split(path, expected_total_rows=300)
    assert np.array_equal(loaded.train_indices, split.train_indices)
    assert np.array_equal(loaded.validation_indices, split.validation_indices)
    assert loaded.train_index_hash == split.train_index_hash
    assert loaded.validation_index_hash == split.validation_index_hash
    meta = loaded.to_metadata()
    for key in (
        "seed", "validation_ratio", "total_rows", "train_rows", "validation_rows",
        "train_index_hash", "validation_index_hash",
    ):
        assert key in meta


def test_tampered_indices_rejected(tmp_path):
    split = make_split(300, validation_ratio=0.1, seed=42)
    path = split.save(tmp_path / "split")
    # Tamper with the npz after saving.
    data = np.load(path.with_suffix(".npz"))
    tampered = data["train_indices"].copy()
    tampered[0] = (tampered[0] + 1) % 300
    np.savez_compressed(
        path.with_suffix(".npz"), train_indices=tampered, validation_indices=data["validation_indices"]
    )
    with pytest.raises(SplitMetadataError, match="hash"):
        load_split(path)


def test_sorted_indices():
    split = make_split(1000, validation_ratio=0.1, seed=42)
    assert np.all(np.diff(split.train_indices) > 0)
    assert np.all(np.diff(split.validation_indices) > 0)


def test_invalid_ratio_rejected():
    with pytest.raises(ValueError):
        make_split(100, validation_ratio=0.0)
    with pytest.raises(ValueError):
        make_split(100, validation_ratio=1.0)
