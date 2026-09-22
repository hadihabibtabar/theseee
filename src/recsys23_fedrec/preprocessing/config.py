"""Configuration for the preprocessing stage.

All values are plain serializable dataclasses so the whole configuration can
be stored inside the fitted preprocessing artifact and later dumped as
``preprocessing_config.json``. Defaults match the Stage 1 dataset inspection
report (root ``.``, 30 train shards, seed 42, chunksize 50000).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

SEED = 42

TRAIN_SUBDIR = "train"
TEST_SUBDIR = "test"
SHARD_GLOB = "*.csv"
REPORT_PATH = "reports/dataset_report.json"

DEFAULT_CHUNKSIZE = 50_000
DEFAULT_DELIMITER = "\t"
DEFAULT_ENCODING = "utf-8"

# Two-pass memory guardrails (tunable by callers; not dataset-specific).
MAX_CATEGORICAL_CARDINALITY = 1_000_000


@dataclass
class PreprocessingConfig:
    """Complete, serializable preprocessing configuration.

    Attributes are grouped by concern. Everything here is deterministic:
    running the pipeline twice with equal configs must yield equal artifacts.
    """

    # Raw data location
    data_dir: str = "."
    train_subdir: str = TRAIN_SUBDIR
    test_subdir: str = TEST_SUBDIR
    shard_glob: str = SHARD_GLOB

    # CSV parsing (Stage 1 finding: files are TAB-separated despite .csv)
    delimiter: str = DEFAULT_DELIMITER
    encoding: str = DEFAULT_ENCODING

    # Chunked processing; None disables chunking (only sensible on small data)
    chunksize: int | None = DEFAULT_CHUNKSIZE

    # Column roles. Defaults follow the Stage 1 report.
    id_columns: tuple[str, ...] = ("f_0",)
    target_columns: tuple[str, ...] = ("is_clicked", "is_installed")

    # Vocabulary policy
    max_categorical_cardinality: int = MAX_CATEGORICAL_CARDINALITY

    # Reproducibility. Kept for artifact traceability; preprocessing itself
    # performs no random sampling.
    seed: int = SEED

    # Artifact provenance
    stage1_report_path: str = REPORT_PATH

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreprocessingConfig":
        field_names = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in dict(payload).items():
            if key not in field_names:
                continue
            if key in ("id_columns", "target_columns") and value is not None:
                value = tuple(value)
            kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "PreprocessingConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def resolved_shard_paths(self, split: str) -> list[Path]:
        """Return the deterministic, sorted shard paths for a split."""
        base = Path(self.data_dir)
        subdir = self.train_subdir if split == "train" else self.test_subdir
        shard_dir = base if subdir == "." else base / subdir
        paths = sorted(
            p for p in shard_dir.glob(self.shard_glob) if p.is_file()
        )
        if not paths:
            raise FileNotFoundError(
                f"No shard files matching {self.shard_glob!r} found in "
                f"{shard_dir} for split {split!r}"
            )
        return paths
