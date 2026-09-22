"""Fixtures for the Stage 4 model tests.

Reuses the Stage 3 synthetic fixture (preprocessed 300-row mini-dataset) so
model tests run fast and independently of the real 3.5M-row dataset. Tests
that verify token counts / vocabulary sizes against the true Stage 2 schema
load the real ``artifacts/`` checkout read-only.
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

from recsys23_fedrec.data import FeatureSet, ProcessedRecSysDataset  # noqa: E402
from recsys23_fedrec.models import TransformerBaseline  # noqa: E402
from recsys23_fedrec.preprocessing import (  # noqa: E402
    PreprocessingArtifactStore,
    PreprocessingConfig,
    Preprocessor,
)
from tests.preprocessing.fixture_generator import write_fixture_files  # noqa: E402

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@pytest.fixture()
def processed_fixture(tmp_path):
    """Synthetic dataset preprocessed into artifacts + Parquet (same as Stage 3)."""
    write_fixture_files(tmp_path, n_train_rows=300, n_test_rows=80)
    config = PreprocessingConfig(data_dir=str(tmp_path), chunksize=64)
    config.stage1_report_path = str(tmp_path / "reports" / "dataset_report.json")
    preprocessor = Preprocessor(config).fit()
    preprocessor.save(str(tmp_path / "artifacts"))
    store = PreprocessingArtifactStore(tmp_path / "artifacts")
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
def feature_set(processed_fixture) -> FeatureSet:
    """FeatureSet over the synthetic fixture schema (fast tests)."""
    return FeatureSet.load(processed_fixture["artifact_dir"])


@pytest.fixture()
def real_feature_set() -> FeatureSet:
    """FeatureSet from the real Stage 2 artifact (read-only)."""
    return FeatureSet.load(ARTIFACTS_DIR)


@pytest.fixture()
def synthetic_batch(feature_set, processed_fixture) -> dict[str, torch.Tensor]:
    """A small collated train batch from the synthetic Stage 3 fixture."""
    from recsys23_fedrec.data import collate_train

    dataset = ProcessedRecSysDataset(
        feature_set, processed_fixture["train_dir"], split="train"
    )
    return collate_train([dataset[i] for i in range(8)])


@pytest.fixture()
def synthetic_test_batch(feature_set, processed_fixture) -> dict[str, torch.Tensor]:
    from recsys23_fedrec.data import collate_test

    dataset = ProcessedRecSysDataset(
        feature_set, processed_fixture["test_dir"], split="test"
    )
    return collate_test([dataset[i] for i in range(8)])


@pytest.fixture()
def small_model(feature_set: FeatureSet) -> TransformerBaseline:
    """Compact model over the synthetic fixture schema (fast tests)."""
    torch.manual_seed(42)
    return TransformerBaseline(
        feature_set, d_model=32, nhead=4, num_layers=2, dim_feedforward=32, dropout=0.1
    )
