"""Shared fixture-building helpers used by multiple test-package conftests.

Plain functions (not pytest fixtures) so each package's conftest can wrap
them independently without cross-package fixture visibility issues.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for path in (str(SRC), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recsys23_fedrec.data import FeatureSet  # noqa: E402
from recsys23_fedrec.preprocessing import (  # noqa: E402
    PreprocessingArtifactStore,
    PreprocessingConfig,
    Preprocessor,
)
from tests.preprocessing.fixture_generator import write_fixture_files  # noqa: E402

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def build_processed_fixture(tmp_path, n_train_rows: int = 300, n_test_rows: int = 80) -> dict:
    """Synthetic dataset preprocessed into artifacts + Parquet parts."""
    write_fixture_files(tmp_path, n_train_rows=n_train_rows, n_test_rows=n_test_rows)
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
        "n_train": n_train_rows,
        "n_test": n_test_rows,
    }


def load_feature_set(processed_fixture: dict) -> FeatureSet:
    return FeatureSet.load(processed_fixture["artifact_dir"])


def load_real_feature_set() -> FeatureSet:
    """FeatureSet from the real Stage 2 artifact (read-only)."""
    return FeatureSet.load(ARTIFACTS_DIR)
