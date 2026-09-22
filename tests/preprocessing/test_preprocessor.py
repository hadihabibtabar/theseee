"""Unit tests for the Stage 2 preprocessing system.

Uses a synthetic mini-dataset (tests/preprocessing/fixture_generator.py) so
tests are fast and independent of the real 3.5M-row dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
for path in (str(SRC), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recsys23_fedrec.preprocessing import (  # noqa: E402
    DataContractError,
    PreprocessingConfig,
    PreprocessingError,
    Preprocessor,
    UNK_ID,
    MISSING_ID,
)
from recsys23_fedrec.preprocessing.feature_schema import (  # noqa: E402
    FeatureSchemaError,
    load_stage1_report_columns,
    schema_from_stage1_report,
    sort_columns_like_raw,
)
from recsys23_fedrec.preprocessing.storage import PreprocessingArtifactStore  # noqa: E402
from tests.preprocessing.fixture_generator import (  # noqa: E402
    BINARY_FIXTURE_COLUMNS,
    CATEGORICAL_FIXTURE_COLUMNS,
    NUMERICAL_FIXTURE_COLUMNS,
    TARGET_FIXTURE_COLUMNS,
    build_test_frame,
    build_train_frame,
    write_fixture_files,
)


@pytest.fixture()
def fixture_env(tmp_path):
    """Synthetic dataset directory with a Stage 1-style report."""
    meta = write_fixture_files(tmp_path, n_train_rows=500, n_test_rows=120)
    config = PreprocessingConfig(data_dir=str(tmp_path), chunksize=128)
    config.stage1_report_path = str(meta["report_path"])
    preprocessor = Preprocessor(config).fit()
    return meta, config, preprocessor


# ----------------------------------------------------------------------
# 1. f_0 exclusion from model features
# ----------------------------------------------------------------------
def test_f0_excluded_from_model_features(fixture_env):
    _, _, preprocessor = fixture_env
    schema = preprocessor.schema
    assert "f_0" in schema.id_columns
    all_model_features = schema.model_feature_columns
    assert "f_0" not in all_model_features
    processed_cols = preprocessor.processed_feature_columns()
    assert "f_0" not in processed_cols


# ----------------------------------------------------------------------
# 2. Labels remain available
# ----------------------------------------------------------------------
def test_targets_remain_labels(fixture_env):
    _, _, preprocessor = fixture_env
    schema = preprocessor.schema
    assert schema.target_columns == ["is_clicked", "is_installed"]  # raw-file order
    out = preprocessor.transform(build_train_frame(50), split="train")
    assert "is_clicked" in out.columns
    assert "is_installed" in out.columns


# ----------------------------------------------------------------------
# 3. Feature ordering is deterministic
# ----------------------------------------------------------------------
def test_feature_ordering_deterministic(fixture_env):
    _, config, first = fixture_env
    meta = write_fixture_files(Path(config.data_dir), n_train_rows=500, n_test_rows=120)
    second = Preprocessor(config).fit()
    assert first.processed_feature_columns() == second.processed_feature_columns()
    assert first.schema.schema_hash == second.schema.schema_hash
    # Categoricals precede binaries, which precede numericals; indicators last.
    cols = first.processed_feature_columns()
    cat_idx = [cols.index(c) for c in CATEGORICAL_FIXTURE_COLUMNS]
    bin_idx = [cols.index(c) for c in BINARY_FIXTURE_COLUMNS]
    num_idx = [cols.index(c) for c in NUMERICAL_FIXTURE_COLUMNS if not c.startswith(("f_30", "f_31"))]
    assert max(cat_idx) < min(bin_idx)
    assert max(bin_idx) < min(num_idx)
    indicator_cols = [c for c in cols if c.endswith("_missing")]
    if indicator_cols:
        assert all(c.endswith("_missing") for c in cols[cols.index(indicator_cols[0]):])


# ----------------------------------------------------------------------
# 4/5/12. Numerical preprocessing: finite, zero-variance safe, no NaN/Inf
# ----------------------------------------------------------------------
def test_numerical_outputs_finite_and_fitted_stats(fixture_env):
    _, _, preprocessor = fixture_env
    frame = build_train_frame(200, seed=7)
    out = preprocessor.transform(frame, split="train")
    feature_cols = preprocessor.processed_feature_columns()
    values = out[feature_cols].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all()
    # Verify the exact standardization for a fully observed column.
    raw = frame["f_42"].to_numpy(dtype=float)
    expected = (raw - preprocessor.numerical_stats["f_42"].mean) / preprocessor.numerical_stats["f_42"].std
    np.testing.assert_allclose(out["f_42"].to_numpy(dtype=float), expected, rtol=1e-5)


def test_zero_variance_numerical_maps_to_zero(fixture_env):
    _, _, preprocessor = fixture_env
    assert preprocessor.numerical_stats["f_71"].std == 0.0
    frame = build_train_frame(100, seed=11)
    out = preprocessor.transform(frame, split="train")
    assert (out["f_71"] == 0.0).all()
    assert np.isfinite(out["f_71"].to_numpy(dtype=float)).all()


# ----------------------------------------------------------------------
# 6/7. Missing numerical values: imputation + indicators
# ----------------------------------------------------------------------
def test_missing_numerical_imputed_with_train_mean(fixture_env):
    _, _, preprocessor = fixture_env
    stats = preprocessor.numerical_stats["f_43"]
    assert stats.missing_count > 0
    frame = build_train_frame(300, seed=3)
    out = preprocessor.transform(frame, split="train")
    missing_rows = frame["f_43"].isna().to_numpy()
    expected = (stats.imputation_value - stats.mean) / stats.std
    np.testing.assert_allclose(
        out.loc[missing_rows, "f_43"].to_numpy(dtype=float), expected, rtol=1e-6
    )


def test_missing_indicators_match_missingness(fixture_env):
    _, _, preprocessor = fixture_env
    frame = build_train_frame(200, seed=5)
    out = preprocessor.transform(frame, split="train")
    for column in ["f_30", "f_31", "f_43", "f_51", "f_64"]:
        name = f"{column}_missing"
        assert name in out.columns
        np.testing.assert_array_equal(
            out[name].to_numpy(), frame[column].isna().to_numpy(dtype=float)
        )


# ----------------------------------------------------------------------
# 8. Binary features: no vocabulary, {0,1} validation, deterministic fill
# ----------------------------------------------------------------------
def test_binary_features_have_no_vocabulary_and_stay_binary(fixture_env):
    _, _, preprocessor = fixture_env
    for column in BINARY_FIXTURE_COLUMNS:
        assert column not in preprocessor.vocabularies
    frame = build_train_frame(150, seed=13)
    out = preprocessor.transform(frame, split="train")
    for column in BINARY_FIXTURE_COLUMNS:
        present = out[column].dropna()
        assert present.isin([0.0, 1.0]).all()


def test_binary_non_binary_value_raises(fixture_env):
    _, _, preprocessor = fixture_env
    frame = build_train_frame(50, seed=17)
    frame.loc[0, "f_33"] = 5
    with pytest.raises(PreprocessingError):
        preprocessor.transform(frame, split="train")


# ----------------------------------------------------------------------
# 9/10. Categorical IDs: known, UNK, MISSING, test values never refit
# ----------------------------------------------------------------------
def test_categorical_values_map_to_integer_ids(fixture_env):
    _, _, preprocessor = fixture_env
    frame = build_train_frame(100, seed=19)
    out = preprocessor.transform(frame, split="train")
    for column in CATEGORICAL_FIXTURE_COLUMNS:
        vocab = preprocessor.vocabularies[column]
        assert vocab.vocab_size >= 2
        ids = out[column].to_numpy()
        assert ids.min() >= 0
        assert ids.max() < vocab.vocab_size
        # Known training values must not map to UNK.
        known = frame[column].dropna().unique()[:3]
        for value in known:
            row_mask = frame[column] == value
            assert (out.loc[row_mask, column] != UNK_ID).all()


def test_unseen_categorical_maps_to_unk(fixture_env):
    _, _, preprocessor = fixture_env
    frame = build_train_frame(50, seed=23)
    frame["f_1"] = 99999  # unseen everywhere
    out = preprocessor.transform(frame, split="train")
    assert (out["f_1"] == UNK_ID).all()


def test_missing_categorical_maps_to_missing_id(fixture_env):
    _, _, preprocessor = fixture_env
    frame = build_train_frame(50, seed=29)
    frame["f_2"] = np.nan
    out = preprocessor.transform(frame, split="train")
    assert (out["f_2"] == MISSING_ID).all()


def test_test_only_values_do_not_extend_vocabulary(fixture_env):
    meta, config, preprocessor = fixture_env
    test_frame = meta["test_frame"]
    assert 999 in test_frame["f_1"].to_numpy()
    before = dict(preprocessor.vocabularies["f_1"].value_to_id)
    out = preprocessor.transform(test_frame, split="test")
    after = preprocessor.vocabularies["f_1"].value_to_id
    assert before == after, "transform must never mutate the fitted vocabulary"
    unseen_mask = test_frame["f_1"] == 999
    assert (out.loc[unseen_mask, "f_1"] == UNK_ID).all()


# ----------------------------------------------------------------------
# 11. Labels only 0/1; invalid values raise
# ----------------------------------------------------------------------
def test_labels_only_zero_or_one_and_invalid_raises(fixture_env):
    _, _, preprocessor = fixture_env
    frame = build_train_frame(80, seed=31)
    out = preprocessor.transform(frame, split="train")
    for column in TARGET_FIXTURE_COLUMNS:
        assert set(np.unique(out[column])) <= {0, 1}
    broken = build_train_frame(50, seed=37)
    broken.loc[0, "is_installed"] = 7
    with pytest.raises(PreprocessingError):
        preprocessor.transform(broken, split="train")


def test_missing_labels_raise_during_training_transform(fixture_env):
    _, _, preprocessor = fixture_env
    frame = build_train_frame(50, seed=41)
    frame.loc[0, "is_clicked"] = np.nan
    with pytest.raises(PreprocessingError):
        preprocessor.transform(frame, split="train")


def test_test_split_must_not_contain_targets(fixture_env):
    _, _, preprocessor = fixture_env
    frame = build_train_frame(50, seed=43)
    with pytest.raises(PreprocessingError):
        preprocessor.transform(frame, split="test")


# ----------------------------------------------------------------------
# 13. Determinism (same data + config -> identical artifact)
# ----------------------------------------------------------------------
def test_preprocessing_is_deterministic(tmp_path):
    write_fixture_files(tmp_path, n_train_rows=400, n_test_rows=100)
    config = PreprocessingConfig(data_dir=str(tmp_path), chunksize=64)
    config.stage1_report_path = str(tmp_path / "reports" / "dataset_report.json")
    first = Preprocessor(config).fit().summary()
    second = Preprocessor(config).fit().summary()
    assert first["artifact_hash"] == second["artifact_hash"]
    assert first["vocabulary_sizes"] == second["vocabulary_sizes"]
    assert first["categorical_columns"] == second["categorical_columns"]


# ----------------------------------------------------------------------
# 14. Raw files unchanged by preprocessing
# ----------------------------------------------------------------------
def test_raw_files_unchanged(fixture_env):
    meta, config, preprocessor = fixture_env
    train_bytes = meta["train_path"].read_bytes()
    test_bytes = meta["test_path"].read_bytes()
    report_bytes = meta["report_path"].read_bytes()
    preprocessor.transform(build_train_frame(50), split="train")
    assert meta["train_path"].read_bytes() == train_bytes
    assert meta["test_path"].read_bytes() == test_bytes
    assert meta["report_path"].read_bytes() == report_bytes


# ----------------------------------------------------------------------
# 15. Train/test processed schemas compatible apart from labels
# ----------------------------------------------------------------------
def test_train_test_processed_schema_compatible(fixture_env):
    _, _, preprocessor = fixture_env
    train_cols = preprocessor.processed_feature_columns() + TARGET_FIXTURE_COLUMNS
    test_cols = preprocessor.processed_feature_columns()
    assert set(train_cols) - set(test_cols) == set(TARGET_FIXTURE_COLUMNS)
    train_frame = preprocessor.transform(build_train_frame(40), split="train")
    test_frame = build_test_frame(40)
    test_out = preprocessor.transform(test_frame, split="test")
    assert list(train_frame.drop(columns=TARGET_FIXTURE_COLUMNS).columns) == list(test_out.columns)


# ----------------------------------------------------------------------
# Artifact round-trip
# ----------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path):
    write_fixture_files(tmp_path, n_train_rows=300, n_test_rows=80)
    config = PreprocessingConfig(data_dir=str(tmp_path), chunksize=100)
    config.stage1_report_path = str(tmp_path / "reports" / "dataset_report.json")
    original = Preprocessor(config).fit()
    original.save(str(tmp_path / "artifacts"))
    restored = Preprocessor.load(str(tmp_path / "artifacts"))

    frame = build_train_frame(120, seed=99)
    a = original.transform(frame, split="train")
    b = restored.transform(frame, split="train")
    pd.testing.assert_frame_equal(a, b)
    assert original.artifact_hash == restored.artifact_hash


def test_artifact_load_detects_tampering(tmp_path):
    import json as _json

    write_fixture_files(tmp_path, n_train_rows=200, n_test_rows=60)
    config = PreprocessingConfig(data_dir=str(tmp_path), chunksize=100)
    config.stage1_report_path = str(tmp_path / "reports" / "dataset_report.json")
    preprocessor = Preprocessor(config).fit()
    preprocessor.save(str(tmp_path / "artifacts"))
    preprocessing_dir = tmp_path / "artifacts" / "preprocessing"

    # (a) Tampering with a derived manifest field must be caught by the
    # cross-validation checks.
    manifest_path = preprocessing_dir / "artifact.json"
    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["vocabulary_sizes"]["f_1"] = 12345
    manifest_path.write_text(_json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PreprocessingError):
        Preprocessor.load(str(tmp_path / "artifacts"))

    # (b) Tampering with hashed content (numerical stats) must be caught by
    # the content-hash verification.
    write_fixture_files(tmp_path, n_train_rows=200, n_test_rows=60)
    preprocessor = Preprocessor(config).fit()
    preprocessor.save(str(tmp_path / "artifacts"))
    stats_path = preprocessing_dir / "numerical_stats.json"
    stats = _json.loads(stats_path.read_text(encoding="utf-8"))
    stats["f_42"]["mean"] = stats["f_42"]["mean"] + 1.0
    stats_path.write_text(_json.dumps(stats), encoding="utf-8")
    with pytest.raises(PreprocessingError):
        Preprocessor.load(str(tmp_path / "artifacts"))


# ----------------------------------------------------------------------
# Parquet storage round-trip
# ----------------------------------------------------------------------
def test_parquet_streaming_roundtrip(fixture_env, tmp_path):
    meta, config, preprocessor = fixture_env
    out_dir = tmp_path / "processed_out"
    store = PreprocessingArtifactStore(out_dir)
    train_rows = preprocessor.transform_split_to_parquet(store, "train")
    test_rows = preprocessor.transform_split_to_parquet(store, "test")
    assert train_rows == len(meta["train_frame"])
    assert test_rows == len(meta["test_frame"])
    parts = store.read_split_metadata("train")
    assert all(p["schema_hash"] == preprocessor.artifact_hash for p in parts)
    # iter_split yields one frame per Parquet part (bounded-memory reads);
    # concatenating them must reproduce the full processed split.
    frames = list(store.iter_split("train"))
    assert sum(len(frame) for frame in frames) == train_rows
    reopened = frames[0]
    assert "is_installed" in reopened.columns
    assert "f_0" not in reopened.columns


# ----------------------------------------------------------------------
# Schema-source classification
# ----------------------------------------------------------------------
def test_schema_classification_from_report():
    """Stage 1 report drives roles: 30 categorical / 38 numerical / 11 binary."""
    report = load_stage1_report_columns("reports/dataset_report.json")
    config = PreprocessingConfig()
    schema = schema_from_stage1_report(report, config)
    assert schema.id_columns == ["f_0"]
    assert schema.target_columns == ["is_clicked", "is_installed"]  # raw-file order
    assert len(schema.categorical_columns) == 30
    assert len(schema.numerical_columns) == 38
    assert len(schema.binary_columns) == 11
    assert "f_0" not in schema.model_feature_columns
    # f_26..f_29 are two-value but NOT {0,1}: categorical, not binary.
    for column in ["f_26", "f_27", "f_28", "f_29"]:
        assert column in schema.categorical_columns
        assert column not in schema.binary_columns
    # True binary columns.
    for column in ["f_30", "f_31", "f_33", "f_34", "f_35", "f_36", "f_37", "f_38", "f_39", "f_40", "f_41"]:
        assert column in schema.binary_columns
    # Sparse numerical columns.
    for column in ["f_43", "f_51", "f_58", "f_59", "f_64", "f_65", "f_66", "f_67", "f_68", "f_69", "f_70"]:
        assert column in schema.numerical_columns


def test_schema_rejects_unassigned_column():
    from recsys23_fedrec.preprocessing.feature_schema import build_feature_schema

    with pytest.raises(FeatureSchemaError):
        build_feature_schema(
            id_columns=["f_0"],
            target_columns=["is_clicked", "is_installed"],
            observed_columns=["f_0", "f_1", "is_clicked", "is_installed"],
            categorical_columns=[],
            numerical_columns=[],
            binary_columns=[],
        )


def test_not_fitted_transform_raises():
    preprocessor = Preprocessor()
    with pytest.raises(PreprocessingError):
        preprocessor.transform(build_train_frame(10), split="train")
