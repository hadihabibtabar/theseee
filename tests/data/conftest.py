"""Fixtures for the data-layer tests.

Reuses the Stage 2 synthetic fixture (tests/preprocessing/fixture_generator)
and runs the real preprocessing pipeline on it, producing artifacts + Parquet
parts in a tmp directory. This keeps Stage 3 tests independent of the real
3.5M-row dataset while exercising the exact same code paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
for path in (str(SRC), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recsys23_fedrec.data import FeatureSet, ProcessedRecSysDataset  # noqa: E402
from recsys23_fedrec.preprocessing import (  # noqa: E402
    PreprocessingArtifactStore,
    PreprocessingConfig,
    Preprocessor,
)
from tests.preprocessing.fixture_generator import write_fixture_files  # noqa: E402


@pytest.fixture()
def processed_fixture(tmp_path):
    """Synthetic dataset preprocessed into artifacts + Parquet parts."""
    write_fixture_files(tmp_path, n_train_rows=300, n_test_rows=80)
    config = PreprocessingConfig(data_dir=str(tmp_path), chunksize=64)
    config.stage1_report_path = str(tmp_path / "reports" / "dataset_report.json")
    preprocessor = Preprocessor(config).fit()
    store = PreprocessingArtifactStore(tmp_path / "artifacts")
    preprocessor.save(str(tmp_path / "artifacts"))
    preprocessor.transform_split_to_parquet(store, "train")
    preprocessor.transform_split_to_parquet(store, "test")
    return {
        "root": tmp_path,
        "artifact_dir": str(tmp_path / "artifacts" / "preprocessing"),
        "train_dir": tmp_path / "artifacts" / "processed" / "train",
        "test_dir": tmp_path / "artifacts" / "processed" / "test",
        "preprocessor": preprocessor,
        "n_train": 300,
        "n_test": 80,
    }


@pytest.fixture()
def feature_set(processed_fixture):
    return FeatureSet.load(processed_fixture["artifact_dir"])


@pytest.fixture()
def train_dataset(feature_set, processed_fixture):
    return ProcessedRecSysDataset(
        feature_set, processed_fixture["train_dir"], split="train"
    )


@pytest.fixture()
def test_dataset(feature_set, processed_fixture):
    return ProcessedRecSysDataset(
        feature_set, processed_fixture["test_dir"], split="test"
    )
