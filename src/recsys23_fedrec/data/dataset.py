"""Parquet-backed, memory-bounded PyTorch datasets for the processed data.

``ProcessedRecSysDataset`` reads the Stage 2 processed Parquet parts via
**per-part Parquet row-group streaming** (read one row group at a time,
convert to a pre-stacked numpy-array cache, discard on part rollover). RAM
holds at most one row group (~50k rows), never the full 3.5M-row dataset.

Design notes:

* Feature order comes exclusively from the Stage 2 artifact (frozen in
  :class:`~recsys23_fedrec.data.feature_schema.FeatureSet`). Parquet column
  order is *verified* against it at construction time and mismatches raise
  :class:`FeatureOrderError` instead of being silently tolerated.
* ``__getitem__`` supports int indices (positive and negative) and slices.
* Per-worker safety: each worker process opens its own Parquet readers
  lazily (handles are closed on ``__del__``); no handle is ever shared
  across processes. State is reconstructible from plain fields, so the
  dataset works with Windows spawn workers.
* Device handling: tensors stay on CPU; the training loop owns transfers.
"""

from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .exceptions import DataError, FeatureOrderError
from .feature_schema import FeatureSet

SCHEMA_HASH_KEY = "recsys23.schema_hash"


class ProcessedRecSysDataset(Dataset):
    """Memory-bounded dataset over the Stage 2 processed Parquet shards.

    Args:
        feature_set: frozen Stage 2 artifact (feature order + vocab bounds).
        split_dir: directory holding ``part-*.parquet`` files.
        split: ``"train"`` (labels included) or ``"test"`` (features only).

    Sample structure (see ``samples.py``): dict with int64 ``categorical``
    [30], float32 ``numerical`` [38], float32 ``binary`` [11], float32
    ``missing`` [13] and — for train — float32 scalars ``click``/``install``.
    """

    def __init__(self, feature_set: FeatureSet, split_dir: str | Path, split: str = "train") -> None:
        self.feature_set = feature_set
        self.split = split
        split_dir = Path(split_dir)
        if split == "train":
            self.columns = list(feature_set.expected_train_columns())
            self.include_targets = True
        elif split == "test":
            self.columns = list(feature_set.expected_test_columns())
            self.include_targets = False
        else:
            raise ValueError(f"Unknown split {split!r}; expected 'train' or 'test'")

        part_files = sorted(split_dir.glob("part-*.parquet"))
        if not part_files:
            raise FileNotFoundError(f"No processed Parquet parts in {split_dir}")
        self.part_paths = part_files

        # ---- schema verification against the artifact -------------------
        parquet_schema = pq.read_schema(part_files[0])
        parquet_columns = list(parquet_schema.names)
        if parquet_columns != self.columns:
            missing = set(self.columns) - set(parquet_columns)
            extra = set(parquet_columns) - set(self.columns)
            raise FeatureOrderError(
                "Processed Parquet columns do not match the Stage 2 artifact "
                f"ordering (missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}). "
                "The artifact feature order is authoritative."
            )
        # Linkage: the parts must have been produced by the same artifact.
        file_metadata = pq.read_metadata(part_files[0]).metadata or {}
        kv = {
            k.decode("utf-8"): v.decode("utf-8") for k, v in file_metadata.items()
        }
        linked_hash = kv.get(SCHEMA_HASH_KEY)
        if linked_hash is not None and linked_hash != feature_set.artifact_hash:
            raise FeatureOrderError(
                f"Parquet artifact-hash {linked_hash} does not match the loaded "
                f"preprocessing artifact {feature_set.artifact_hash}; the data "
                "and artifact are out of sync."
            )

        # ---- row-group index (metadata only, no data read) --------------
        self._row_counts: list[int] = []
        self._row_group_sizes: list[list[int]] = []
        for path in part_files:
            meta = pq.read_metadata(path)
            self._row_counts.append(meta.num_rows)
            self._row_group_sizes.append(
                [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
            )
        self._cumulative_rows = np.cumsum([0] + self._row_counts).tolist()
        self.length = int(self._cumulative_rows[-1])

        # ---- per-process lazy state --------------------------------------
        self._rg_cache: dict[str, tuple[int, int]] = {}   # name -> (part, row_group)
        self._rg_cache_order: list[str] = []
        self._cached_frame: pd.DataFrame | None = None
        self._cached_key: tuple[int, int] | None = None
        self._readers: list = []
        self._worker_init = False
        self._read_count = 0

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.length)
            indices = range(start, stop, step)
            return [self._get_single(i) for i in indices]
        if isinstance(index, int):
            i = index
            if i < 0:
                i += self.length
            if not 0 <= i < self.length:
                raise IndexError(f"index {index} out of range for dataset of length {self.length}")
            return self._get_single(i)
        raise TypeError(f"Invalid index type {type(index).__name__}; expected int or slice")

    # ------------------------------------------------------------------
    # Row-group streaming
    # ------------------------------------------------------------------
    def _locate_row_group(self, row: int) -> tuple[int, int]:
        part = bisect.bisect_right(self._cumulative_rows, row) - 1
        within = row - self._cumulative_rows[part]
        rg_sizes = self._row_group_sizes[part]
        acc = 0
        for rg, size in enumerate(rg_sizes):
            if within < acc + size:
                return part, rg
            acc += size
        raise DataError(f"Row {row} not locatable in part {part} (index corruption)")

    def _get_reader(self, part: int):
        # Pad lazily so random-access order (e.g. negative indexing hitting a
        # late part first) never indexes past the list.
        while len(self._readers) <= part:
            self._readers.append(None)
        if self._readers[part] is None:
            self._readers[part] = pq.ParquetFile(self.part_paths[part])
        return self._readers[part]

    def _read_row_group(self, part: int, rg: int) -> dict[str, np.ndarray]:
        """Read one row group into the bounded per-process cache.

        The cache holds pre-stacked 2D numpy arrays per tensor block (not a
        pandas DataFrame) so ``__getitem__`` is O(1) array indexing instead
        of per-row pandas Series extraction. Memory is unchanged: exactly
        one row group (~50k rows) per process.
        """
        key = (part, rg)
        if self._cached_key == key and self._cached_frame is not None:
            return self._cached_frame
        reader = self._get_reader(part)
        table = reader.read_row_group(rg, columns=self.columns)
        frame = table.to_pandas()
        fs = self.feature_set
        # Column order in each stacked array is the artifact order.
        cached: dict[str, np.ndarray | None] = {
            "categorical": frame[list(fs.categorical_names)].to_numpy(dtype=np.int64),
            "numerical": frame[list(fs.numerical_names)].to_numpy(dtype=np.float32),
            "binary": frame[list(fs.binary_names)].to_numpy(dtype=np.float32),
            "missing": frame[list(fs.missing_indicator_names)].to_numpy(dtype=np.float32),
            "is_clicked": None,
            "is_installed": None,
        }
        if self.include_targets:
            cached["is_clicked"] = frame["is_clicked"].to_numpy(dtype=np.float32)
            cached["is_installed"] = frame["is_installed"].to_numpy(dtype=np.float32)
        del frame, table
        self._cached_frame = cached
        self._cached_key = key
        self._read_count += 1
        return cached

    def _get_single(self, i: int):
        part, rg = self._locate_row_group(i)
        cached = self._read_row_group(part, rg)
        within = i - self._cumulative_rows[part]
        local = within - sum(self._row_group_sizes[part][:rg])

        sample = {
            "categorical": torch.from_numpy(cached["categorical"][local]),
            "numerical": torch.from_numpy(cached["numerical"][local]),
            "binary": torch.from_numpy(cached["binary"][local]),
            "missing": torch.from_numpy(cached["missing"][local]),
        }
        if self.include_targets:
            sample["click"] = torch.tensor(cached["is_clicked"][local], dtype=torch.float32)
            sample["install"] = torch.tensor(cached["is_installed"][local], dtype=torch.float32)
        return sample

    # ------------------------------------------------------------------
    # Pickling / worker / lifecycle management
    # ------------------------------------------------------------------
    def __getstate__(self) -> dict:
        """Strip runtime Parquet handles so the dataset pickles cleanly.

        ``torch`` DataLoader workers receive the dataset through pickle;
        on Windows (spawn) every field must survive round-tripping. Open
        ``pq.ParquetFile`` readers and cached frames are process-local
        runtime state — they are dropped here and rebuilt lazily by
        ``_get_reader``/``_read_row_group`` in each worker.
        """
        state = self.__dict__.copy()
        state["_readers"] = []
        state["_cached_frame"] = None
        state["_cached_key"] = None
        state["_read_count"] = 0
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._reset_for_worker()

    def _reset_for_worker(self) -> None:
        """Close shared handles; called in worker processes after fork/spawn."""
        self._readers = []
        self._cached_frame = None
        self._cached_key = None
        self._worker_init = True

    def worker_init_fn(self, worker_id: int = 0) -> None:
        """Pass to DataLoader(worker_init_fn=dataset.worker_init_fn) so each
        worker process starts with clean, unshared Parquet state."""
        self._reset_for_worker()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            for reader in getattr(self, "_readers", []):
                if reader is not None:
                    try:
                        reader.close()
                    except Exception:
                        pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Introspection helpers (used by tests / smoke scripts)
    # ------------------------------------------------------------------
    @property
    def num_parts(self) -> int:
        return len(self.part_paths)

    def part_row_counts(self) -> list[int]:
        return list(self._row_counts)

    def part_boundaries(self) -> list[int]:
        """Global row index of the first row of each part (len == num_parts)."""
        return self._cumulative_rows[:-1]

    def parquet_read_count(self) -> int:
        return self._read_count

    @property
    def cached_row_group_rows(self) -> int:
        """Rows held in the single-row-group cache (memory-boundness evidence)."""
        if self._cached_frame is None:
            return 0
        return len(self._cached_frame["categorical"])
