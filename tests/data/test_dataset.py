"""Stage 3 data-layer tests (synthetic fixture; no real 3.5M-row data)."""

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

from recsys23_fedrec.data import (  # noqa: E402
    BatchValidationError,
    FeatureOrderError,
    FeatureSet,
    ProcessedRecSysDataset,
    collate_test,
    collate_train,
    create_dataloader,
    validate_batch,
)
from recsys23_fedrec.preprocessing import (  # noqa: E402
    PreprocessingConfig,
    Preprocessor,
)


# ----------------------------------------------------------------------
# Feature ordering comes from the artifact, not from Parquet/filesystem
# ----------------------------------------------------------------------
def test_feature_order_matches_artifact(train_dataset, feature_set):
    assert train_dataset.columns == list(feature_set.expected_train_columns())
    # Categorical block first, in artifact order (Stage 2 guarantees f_1<...).
    assert train_dataset.columns[: feature_set.num_categorical] == list(
        feature_set.categorical_names
    )


def test_parquet_column_mismatch_raises(processed_fixture, tmp_path):
    """A tampered Parquet schema must fail loudly (never silently tolerated)."""
    import pyarrow.parquet as pq
    import pyarrow as pa

    part = processed_fixture["train_dir"] / "part-00000.parquet"
    table = pq.read_table(part)
    # Drop one column -> schema no longer matches artifact order.
    dropped = table.drop_columns(["f_1"])
    pq.write_table(dropped, part)
    feature_set = FeatureSet.load(processed_fixture["artifact_dir"])
    with pytest.raises(FeatureOrderError):
        ProcessedRecSysDataset(feature_set, processed_fixture["train_dir"], split="train")


def test_artifact_hash_mismatch_raises(processed_fixture, tmp_path):
    """Data written by a different artifact must be rejected."""
    # Reload preprocessing with a different chunksize -> different artifact
    # hash (config participates in the hash), same underlying data columns.
    config = PreprocessingConfig(
        data_dir=str(processed_fixture["root"]), chunksize=32
    )
    config.stage1_report_path = str(
        processed_fixture["root"] / "reports" / "dataset_report.json"
    )
    other = Preprocessor(config).fit()
    other.save(str(tmp_path / "other_artifacts"))
    other_feature_set = FeatureSet.load(str(tmp_path / "other_artifacts" / "preprocessing"))
    assert other_feature_set.artifact_hash != FeatureSet.load(
        processed_fixture["artifact_dir"]
    ).artifact_hash
    # Same columns but different artifact hash -> linkage error.
    with pytest.raises(FeatureOrderError):
        ProcessedRecSysDataset(
            other_feature_set, processed_fixture["train_dir"], split="train"
        )


# ----------------------------------------------------------------------
# Sample structure, dtypes, values
# ----------------------------------------------------------------------
def test_sample_structure_and_dtypes(train_dataset):
    sample = train_dataset[0]
    fs = train_dataset.feature_set
    assert sample["categorical"].dtype == torch.int64
    assert sample["categorical"].shape == (fs.num_categorical,)
    assert sample["numerical"].dtype == torch.float32
    assert sample["numerical"].shape == (fs.num_numerical,)
    assert sample["binary"].dtype == torch.float32
    assert sample["binary"].shape == (fs.num_binary,)
    assert sample["missing"].dtype == torch.float32
    assert sample["missing"].shape == (fs.num_missing,)
    assert sample["click"].dtype == torch.float32
    assert sample["click"].dim() == 0
    assert sample["install"].dim() == 0
    assert sample["click"].item() in (0.0, 1.0)
    assert sample["install"].item() in (0.0, 1.0)


def test_negative_indexing(train_dataset):
    first = train_dataset[0]
    last = train_dataset[-1]
    assert first["categorical"].shape == last["categorical"].shape


def test_slice_access(train_dataset):
    samples = train_dataset[3:7]
    assert len(samples) == 4
    assert all("click" in s for s in samples)


def test_out_of_range_index_raises(train_dataset):
    with pytest.raises(IndexError):
        train_dataset[len(train_dataset)]
    with pytest.raises(IndexError):
        train_dataset[-(len(train_dataset) + 1)]


# ----------------------------------------------------------------------
# Batch shapes from the collate functions (widths derived from artifact)
# ----------------------------------------------------------------------
def test_collate_train_shapes(train_dataset, feature_set):
    samples = [train_dataset[i] for i in range(16)]
    batch = collate_train(samples)
    assert batch["categorical"].shape == (16, feature_set.num_categorical)
    assert batch["numerical"].shape == (16, feature_set.num_numerical)
    assert batch["binary"].shape == (16, feature_set.num_binary)
    assert batch["missing"].shape == (16, feature_set.num_missing)
    assert batch["click"].shape == (16,)
    assert batch["install"].shape == (16,)
    assert batch["categorical"].dtype == torch.int64
    for key in ("numerical", "binary", "missing", "click", "install"):
        assert batch[key].dtype == torch.float32


def test_collate_test_shapes(test_dataset, feature_set):
    samples = [test_dataset[i] for i in range(8)]
    batch = collate_test(samples)
    assert batch["categorical"].shape == (8, feature_set.num_categorical)
    assert "click" not in batch and "install" not in batch


def test_collate_rejects_bad_samples():
    import torch as _torch

    with pytest.raises(Exception):
        collate_train([{"categorical": _torch.zeros(3)}])


# ----------------------------------------------------------------------
# validate_batch utility
# ----------------------------------------------------------------------
def test_validate_batch_accepts_good_batch(train_dataset, feature_set):
    batch = collate_train([train_dataset[i] for i in range(16)])
    validate_batch(batch, feature_set, split="train")


def test_validate_batch_rejects_missing_key(train_dataset, feature_set):
    batch = collate_train([train_dataset[i] for i in range(4)])
    del batch["click"]
    with pytest.raises(BatchValidationError, match="click"):
        validate_batch(batch, feature_set, split="train")


def test_validate_batch_rejects_wrong_dtype(train_dataset, feature_set):
    batch = collate_train([train_dataset[i] for i in range(4)])
    batch["numerical"] = batch["numerical"].double()
    with pytest.raises(BatchValidationError, match="numerical"):
        validate_batch(batch, feature_set, split="train")


def test_validate_batch_rejects_out_of_range_ids(feature_set):
    batch = {
        "categorical": torch.full((4, feature_set.num_categorical), 10**6, dtype=torch.int64),
        "numerical": torch.zeros(4, feature_set.num_numerical),
        "binary": torch.zeros(4, feature_set.num_binary),
        "missing": torch.zeros(4, feature_set.num_missing),
        "click": torch.zeros(4),
        "install": torch.zeros(4),
    }
    with pytest.raises(BatchValidationError, match="range"):
        validate_batch(batch, feature_set, split="train")


def test_validate_batch_rejects_nan(feature_set):
    batch = {
        "categorical": torch.zeros(4, feature_set.num_categorical, dtype=torch.int64),
        "numerical": torch.full((4, feature_set.num_numerical), float("nan")),
        "binary": torch.zeros(4, feature_set.num_binary),
        "missing": torch.zeros(4, feature_set.num_missing),
        "click": torch.zeros(4),
        "install": torch.zeros(4),
    }
    with pytest.raises(BatchValidationError, match="NaN"):
        validate_batch(batch, feature_set, split="train")


def test_validate_batch_rejects_non_indicator_values(feature_set):
    batch = {
        "categorical": torch.zeros(4, feature_set.num_categorical, dtype=torch.int64),
        "numerical": torch.zeros(4, feature_set.num_numerical),
        "binary": torch.full((4, feature_set.num_binary), 0.5),
        "missing": torch.zeros(4, feature_set.num_missing),
        "click": torch.zeros(4),
        "install": torch.zeros(4),
    }
    with pytest.raises(BatchValidationError, match="binary"):
        validate_batch(batch, feature_set, split="train")


def test_validate_batch_rejects_labels_in_test_split(feature_set):
    batch = {
        "categorical": torch.zeros(4, feature_set.num_categorical, dtype=torch.int64),
        "numerical": torch.zeros(4, feature_set.num_numerical),
        "binary": torch.zeros(4, feature_set.num_binary),
        "missing": torch.zeros(4, feature_set.num_missing),
        "click": torch.zeros(4),
    }
    with pytest.raises(BatchValidationError, match="click"):
        validate_batch(batch, feature_set, split="test")


# ----------------------------------------------------------------------
# DataLoader end-to-end
# ----------------------------------------------------------------------
def test_dataloader_shapes_and_order(train_dataset, feature_set):
    loader = create_dataloader(train_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    assert batch["categorical"].shape == (32, feature_set.num_categorical)
    assert batch["numerical"].shape == (32, feature_set.num_numerical)
    assert batch["binary"].shape == (32, feature_set.num_binary)
    assert batch["missing"].shape == (32, feature_set.num_missing)
    assert batch["click"].shape == (32,)
    assert batch["install"].shape == (32,)
    validate_batch(batch, feature_set, split="train")


def test_dataloader_sequential_order_matches_dataset(train_dataset):
    """shuffle=False: batch contents must equal dataset order exactly."""
    loader = create_dataloader(train_dataset, batch_size=10, shuffle=False, num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    for row, sample in enumerate(train_dataset[0:10]):
        assert torch.equal(batch["categorical"][row], sample["categorical"])
        assert torch.equal(batch["numerical"][row], sample["numerical"])
        assert batch["click"][row].item() == sample["click"].item()


def test_dataloader_drop_last_affects_batches_not_length(train_dataset):
    loader = create_dataloader(train_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=0, pin_memory=False)
    assert len(train_dataset) == 300  # dataset length untouched
    batch_sizes = [len(b["click"]) for b in loader]
    assert all(size == 32 for size in batch_sizes)
    assert 300 % 32 != 0  # remainder dropped from iteration, rows still exist


def test_dataloader_determinism_with_fixed_generator(train_dataset, feature_set):
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    b1 = next(iter(create_dataloader(train_dataset, batch_size=16, shuffle=True, generator=gen1, num_workers=0, pin_memory=False)))
    b2 = next(iter(create_dataloader(train_dataset, batch_size=16, shuffle=True, generator=gen2, num_workers=0, pin_memory=False)))
    assert torch.equal(b1["categorical"], b2["categorical"])
    assert torch.equal(b1["click"], b2["click"])


# ----------------------------------------------------------------------
# Parquet part-boundary integrity
# ----------------------------------------------------------------------
def test_no_duplication_or_skipping_around_boundaries(train_dataset):
    """Row identity must survive part boundaries (no dup/skip)."""
    counts = train_dataset.part_row_counts()
    if train_dataset.num_parts < 2:
        pytest.skip("fixture produced a single part")
    # Stage 2 writes one part per source chunk; fixture chunksize=64 over
    # 300 rows -> parts of 64..48 rows. Verify each boundary pair.
    for part in range(1, train_dataset.num_parts):
        boundary = train_dataset.part_boundaries()[part]
        last_of_prev = train_dataset[boundary - 1]
        first_of_curr = train_dataset[boundary]
        # Rows must differ (no duplication) but be structurally identical.
        for key in ("categorical", "numerical", "binary", "missing"):
            assert last_of_prev[key].shape == first_of_curr[key].shape
        distinct = any(
            not torch.equal(last_of_prev[k], first_of_curr[k])
            for k in ("categorical", "numerical")
        )
        # With 300 synthetic rows, adjacent rows are different records.
        assert distinct, f"rows {boundary - 1} and {boundary} appear duplicated"


def test_row_group_cache_is_bounded(train_dataset):
    """Cache must hold one row group (or less), never unbounded history."""
    for i in range(0, min(len(train_dataset), 200), 7):
        _ = train_dataset[i]
    assert train_dataset.cached_row_group_rows <= 64  # fixture chunksize
    assert train_dataset._cached_key is not None


# ----------------------------------------------------------------------
# Workers (num_workers=2)
# ----------------------------------------------------------------------
def test_dataloader_num_workers_2(train_dataset, feature_set):
    loader = create_dataloader(train_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=False)
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) == 3:
            break
    assert len(batches) == 3
    validate_batch(batches[0], feature_set, split="train")


def test_worker_equivalence_with_main_process(train_dataset, feature_set):
    """Same indices must yield identical samples with and without workers."""
    loader0 = create_dataloader(train_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
    main_batch = next(iter(loader0))
    loader2 = create_dataloader(train_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=False)
    worker_batch = next(iter(loader2))
    for key in ("categorical", "numerical", "binary", "missing", "click", "install"):
        assert torch.equal(main_batch[key], worker_batch[key]), f"worker mismatch in {key}"


# ----------------------------------------------------------------------
# Row-count integrity
# ----------------------------------------------------------------------
def test_dataset_lengths(train_dataset, test_dataset, processed_fixture):
    assert len(train_dataset) == processed_fixture["n_train"]
    assert len(test_dataset) == processed_fixture["n_test"]
    assert train_dataset.include_targets is True
    assert test_dataset.include_targets is False


def test_sum_of_parts_equals_length(train_dataset):
    assert sum(train_dataset.part_row_counts()) == len(train_dataset)
