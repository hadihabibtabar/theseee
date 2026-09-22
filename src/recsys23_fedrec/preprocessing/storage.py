"""Parquet persistence for processed chunks and preprocessing artifacts.

Layout (all under ``output_dir``):

* ``preprocessing/feature_schema.json``           - deterministic feature roles
* ``preprocessing/numerical_stats.json``          - mean/std/imputation per column
* ``preprocessing/categorical_vocabularies.json`` - full vocabulary mappings
* ``preprocessing/preprocessing_config.json``     - run configuration + conventions
* ``preprocessing/artifact.json``                 - linkage/provenance manifest
* ``processed/train/part-00000.parquet`` ...      - processed training chunks
* ``processed/test/part-00000.parquet``           - processed test chunks

Every Parquet part stores the ``artifact_hash`` of the fitted preprocessing
artifact in its key-value metadata, so processed data can always be linked
back to the exact artifact that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .exceptions import ArtifactError

PREPROCESSING_DIRNAME = "preprocessing"
PROCESSED_DIRNAME = "processed"

SCHEMA_HASH_KEY = "recsys23.schema_hash"
SPLIT_KEY = "recsys23.split"

ARTIFACT_FILENAME = "artifact.json"
FEATURE_SCHEMA_FILENAME = "feature_schema.json"
NUMERICAL_STATS_FILENAME = "numerical_stats.json"
VOCABULARIES_FILENAME = "categorical_vocabularies.json"
CONFIG_FILENAME = "preprocessing_config.json"


class PreprocessingArtifactStore:
    """Filesystem layout for preprocessing artifacts and processed data."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.preprocessing_dir = self.output_dir / PREPROCESSING_DIRNAME
        self.processed_dir = self.output_dir / PROCESSED_DIRNAME

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def artifact_path(self) -> Path:
        return self.preprocessing_dir / ARTIFACT_FILENAME

    def feature_schema_path(self) -> Path:
        return self.preprocessing_dir / FEATURE_SCHEMA_FILENAME

    def numerical_stats_path(self) -> Path:
        return self.preprocessing_dir / NUMERICAL_STATS_FILENAME

    def vocabularies_path(self) -> Path:
        return self.preprocessing_dir / VOCABULARIES_FILENAME

    def config_path(self) -> Path:
        return self.preprocessing_dir / CONFIG_FILENAME

    def split_dir(self, split: str) -> Path:
        return self.processed_dir / split

    # ------------------------------------------------------------------
    # JSON payload I/O
    # ------------------------------------------------------------------
    def write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def read_json(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise ArtifactError(f"Artifact file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    # ------------------------------------------------------------------
    # Chunked Parquet writing
    # ------------------------------------------------------------------
    def open_writer(
        self,
        split: str,
        schema: pa.Schema,
        schema_hash: str,
    ) -> "ChunkedParquetWriter":
        split_dir = self.split_dir(split)
        split_dir.mkdir(parents=True, exist_ok=True)
        return ChunkedParquetWriter(split_dir, schema, schema_hash, split)

    def iter_split(self, split: str) -> Iterator[pd.DataFrame]:
        """Yield processed chunks for a split, in deterministic part order."""
        part_files = self.part_files(split)
        for part in part_files:
            yield pq.read_table(part).to_pandas()

    def part_files(self, split: str) -> list[Path]:
        split_dir = self.split_dir(split)
        if not split_dir.is_dir():
            raise ArtifactError(
                f"No processed data directory for split {split!r}: {split_dir}"
            )
        part_files = sorted(split_dir.glob("part-*.parquet"))
        if not part_files:
            raise ArtifactError(f"No processed Parquet parts in {split_dir}")
        return part_files

    def read_split_metadata(self, split: str) -> list[dict[str, Any]]:
        """Read per-part metadata (rows + artifact linkage)."""
        records = []
        for part in self.part_files(split):
            meta = pq.read_metadata(part)
            kv = meta.metadata or {}
            schema_hash = None
            split_name = None
            if kv is not None:
                for key, value in kv.items():
                    key_str = key.decode("utf-8", errors="replace")
                    if key_str == SCHEMA_HASH_KEY:
                        schema_hash = value.decode("utf-8")
                    elif key_str == SPLIT_KEY:
                        split_name = value.decode("utf-8")
            records.append(
                {
                    "part": part.name,
                    "num_rows": meta.num_rows,
                    "num_row_groups": meta.num_row_groups,
                    "schema_hash": schema_hash,
                    "split": split_name,
                }
            )
        return records

    def count_split_rows(self, split: str) -> int:
        return sum(rec["num_rows"] for rec in self.read_split_metadata(split))

    def read_split_schema(self, split: str) -> pa.Schema:
        return pq.read_schema(self.part_files(split)[0])


class ChunkedParquetWriter:
    """Writes each processed chunk as ``part-NNNNN.parquet`` with metadata."""

    def __init__(
        self,
        split_dir: Path,
        schema: pa.Schema,
        schema_hash: str,
        split: str,
    ) -> None:
        self.split_dir = split_dir
        self.schema = schema
        self.schema_hash = schema_hash
        self.split = split
        self._part_index = 0
        self._rows_written = 0

    def write_chunk(self, df: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        linkage = {
            SCHEMA_HASH_KEY: self.schema_hash,
            SPLIT_KEY: self.split,
        }
        existing = {
            k.decode("utf-8"): v.decode("utf-8")
            for k, v in (table.schema.metadata or {}).items()
        }
        merged = {**existing, **linkage}
        table = table.replace_schema_metadata(
            {k.encode("utf-8"): v.encode("utf-8") for k, v in merged.items()}
        )
        path = self.split_dir / f"part-{self._part_index:05d}.parquet"
        pq.write_table(table, path, compression="snappy")
        self._part_index += 1
        self._rows_written += len(df)

    def close(self) -> int:
        return self._rows_written
