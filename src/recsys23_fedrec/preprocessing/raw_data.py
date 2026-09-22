"""Chunked raw-data access (deterministic, bounded-memory).

Responsibilities kept deliberately narrow:

* discover shard paths deterministically (sorted by filename);
* yield bounded chunks (``config.chunksize`` rows) in file/row order;
* validate that raw headers match the fitted feature schema.

Counting/vocabulary building does NOT happen here — the fitting logic lives
in :class:`~recsys23_fedrec.preprocessing.preprocessor.Preprocessor`, which
merges per-chunk statistics incrementally and never retains per-row data.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import pandas as pd

from .config import PreprocessingConfig
from .exceptions import DataContractError


class RawShardReader:
    """Reads raw shard files in deterministic order, in bounded-memory chunks."""

    def __init__(
        self,
        config: PreprocessingConfig,
        *,
        id_columns: Sequence[str],
        categorical_columns: Sequence[str],
        numerical_columns: Sequence[str],
        binary_columns: Sequence[str],
        target_columns: Sequence[str],
    ) -> None:
        self.config = config
        self.id_columns = list(id_columns)
        self.categorical_columns = list(categorical_columns)
        self.numerical_columns = list(numerical_columns)
        self.binary_columns = list(binary_columns)
        self.target_columns = list(target_columns)

    # ------------------------------------------------------------------
    # Chunk iteration (deterministic file and row order)
    # ------------------------------------------------------------------
    def iter_train_chunks(self) -> Iterator[pd.DataFrame]:
        """Yield training chunks in deterministic file/row order."""
        paths = self.config.resolved_shard_paths("train")
        for path in paths:
            reader = pd.read_csv(
                path,
                sep=self.config.delimiter,
                encoding=self.config.encoding,
                chunksize=self.config.chunksize,
            )
            for chunk in reader:
                yield chunk

    def iter_test_chunks(self) -> Iterator[pd.DataFrame]:
        """Yield test chunks in deterministic file/row order."""
        paths = self.config.resolved_shard_paths("test")
        for path in paths:
            reader = pd.read_csv(
                path,
                sep=self.config.delimiter,
                encoding=self.config.encoding,
                chunksize=self.config.chunksize,
            )
            for chunk in reader:
                yield chunk

    def iter_first_train_chunk(self, row_limit: int) -> pd.DataFrame:
        """Read at most ``row_limit`` rows of the first training shard.

        Used by smoke tests; never by the full pipeline.
        """
        path = self.config.resolved_shard_paths("train")[0]
        return pd.read_csv(
            path,
            sep=self.config.delimiter,
            encoding=self.config.encoding,
            nrows=row_limit,
        )

    def iter_first_test_chunk(self, row_limit: int) -> pd.DataFrame:
        """Read at most ``row_limit`` rows of the first test shard."""
        path = self.config.resolved_shard_paths("test")[0]
        return pd.read_csv(
            path,
            sep=self.config.delimiter,
            encoding=self.config.encoding,
            nrows=row_limit,
        )

    # ------------------------------------------------------------------
    # Contract checks
    # ------------------------------------------------------------------
    def _check_header(self, split: str, include_targets: bool) -> None:
        paths = self.config.resolved_shard_paths(split)
        header = pd.read_csv(paths[0], sep=self.config.delimiter, nrows=0)
        expected = set(
            self.id_columns
            + self.categorical_columns
            + self.numerical_columns
            + self.binary_columns
        )
        if include_targets:
            expected |= set(self.target_columns)
        missing = expected - set(header.columns)
        extra = set(header.columns) - expected
        label = "Training" if split == "train" else "Test"
        if missing:
            raise DataContractError(f"{label} data missing columns: {sorted(missing)}")
        if extra:
            raise DataContractError(f"{label} data has unexpected columns: {sorted(extra)}")

    def validate_train_columns(self) -> None:
        """Verify the training header contains exactly the schema columns."""
        self._check_header("train", include_targets=True)

    def validate_test_columns(self) -> None:
        """Verify the test header matches train minus targets."""
        self._check_header("test", include_targets=False)


def count_raw_rows(config: PreprocessingConfig, split: str) -> int:
    """Count raw rows of a split without loading data (fast line counting)."""
    paths = config.resolved_shard_paths(split)
    total = 0
    for path in paths:
        with open(path, "rb") as handle:
            total += sum(1 for _ in handle)
    return max(total - len(paths), 0)  # one header line per shard


def _hash_key(value) -> object:
    """Stable dict key for a category value (keeps ints/floats/strs apart)."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return ("b", bool(value))
    if isinstance(value, (int, np.integer)):
        return ("i", int(value))
    if isinstance(value, float) and not np.isnan(value):
        return ("f", value)
    if isinstance(value, str):
        return ("s", value)
    return ("o", value)
