"""Deterministic train/validation index split (Stage 6).

The split is defined over dataset *indices* — no data is materialized. A
seeded ``numpy.random.Generator`` (PCG64, project seed) produces one global
permutation; the first ``ceil(n * validation_ratio)`` indices become the
validation set, the rest the training set. Both are stored **sorted** so
index lookup against the Stage 3 Parquet dataset is cache-friendly
(monotone access reuses row groups) while membership stays O(log n).

Split identity is proven by 64-bit truncated SHA-256 hashes over the exact
sorted index arrays; metadata is saved to JSON so every later experiment can
verify it is using the identical split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recsys23_fedrec.preprocessing.config import SEED

__all__ = ["TrainValidationSplit", "make_split", "load_split", "SplitMetadataError"]

SPLIT_FORMAT_VERSION = 1


class SplitMetadataError(ValueError):
    """Split metadata is missing, malformed, or fails hash verification."""


def _index_hash(indices: np.ndarray) -> str:
    """Deterministic hash of an index array (content + dtype)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(indices, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


@dataclass
class TrainValidationSplit:
    """Disjoint train/validation index sets covering [0, total_rows)."""

    train_indices: np.ndarray  # sorted int64
    validation_indices: np.ndarray  # sorted int64
    seed: int
    validation_ratio: float
    total_rows: int

    # ------------------------------------------------------------------
    def validate(self) -> None:
        n_train = len(self.train_indices)
        n_val = len(self.validation_indices)
        if n_train + n_val != self.total_rows:
            raise SplitMetadataError(
                f"split sizes {n_train} + {n_val} != total_rows {self.total_rows}"
            )
        if np.intersect1d(self.train_indices, self.validation_indices).size:
            raise SplitMetadataError("train and validation indices overlap")
        combined = np.union1d(self.train_indices, self.validation_indices)
        if combined.size != self.total_rows or combined[0] != 0 or combined[-1] != self.total_rows - 1:
            raise SplitMetadataError("split does not cover every row exactly once")

    @property
    def train_rows(self) -> int:
        return len(self.train_indices)

    @property
    def validation_rows(self) -> int:
        return len(self.validation_indices)

    @property
    def train_index_hash(self) -> str:
        return _index_hash(self.train_indices)

    @property
    def validation_index_hash(self) -> str:
        return _index_hash(self.validation_indices)

    # ------------------------------------------------------------------
    def to_metadata(self) -> dict:
        return {
            "format_version": SPLIT_FORMAT_VERSION,
            "seed": int(self.seed),
            "validation_ratio": float(self.validation_ratio),
            "total_rows": int(self.total_rows),
            "train_rows": int(self.train_rows),
            "validation_rows": int(self.validation_rows),
            "train_index_hash": self.train_index_hash,
            "validation_index_hash": self.validation_index_hash,
        }

    def save(self, path: str | Path) -> Path:
        """Persist indices + metadata to ``<path>.npz`` and ``<path>.json``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            train_indices=self.train_indices.astype(np.int64),
            validation_indices=self.validation_indices.astype(np.int64),
        )
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as handle:
            json.dump(self.to_metadata(), handle, indent=2, sort_keys=True)
        return path

    # ------------------------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
        *,
        seed: int,
        validation_ratio: float,
        verify: bool = True,
    ) -> "TrainValidationSplit":
        split = cls(
            train_indices=np.asarray(train_indices, dtype=np.int64),
            validation_indices=np.asarray(validation_indices, dtype=np.int64),
            seed=int(seed),
            validation_ratio=float(validation_ratio),
            total_rows=int(len(train_indices) + len(validation_indices)),
        )
        if verify:
            split.validate()
        return split

    @classmethod
    def load(cls, path: str | Path, *, expected_total_rows: int | None = None) -> "TrainValidationSplit":
        """Load saved split metadata + indices and verify the hashes."""
        path = Path(path)
        json_path = path.with_suffix(".json")
        npz_path = path.with_suffix(".npz")
        if not json_path.is_file() or not npz_path.is_file():
            raise FileNotFoundError(f"split files not found under {path} (*.json + *.npz)")
        with open(json_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        data = np.load(npz_path)
        train_indices = np.asarray(data["train_indices"], dtype=np.int64)
        validation_indices = np.asarray(data["validation_indices"], dtype=np.int64)

        if meta.get("format_version") != SPLIT_FORMAT_VERSION:
            raise SplitMetadataError(f"unsupported split format {meta.get('format_version')}")
        split = cls(
            train_indices=train_indices,
            validation_indices=validation_indices,
            seed=int(meta["seed"]),
            validation_ratio=float(meta["validation_ratio"]),
            total_rows=int(meta["total_rows"]),
        )
        if meta["train_rows"] != split.train_rows or meta["validation_rows"] != split.validation_rows:
            raise SplitMetadataError("saved row counts do not match index arrays")
        if meta["train_index_hash"] != split.train_index_hash:
            raise SplitMetadataError("train index hash mismatch (split files corrupted or altered)")
        if meta["validation_index_hash"] != split.validation_index_hash:
            raise SplitMetadataError("validation index hash mismatch (split files corrupted or altered)")
        if expected_total_rows is not None and split.total_rows != expected_total_rows:
            raise SplitMetadataError(
                f"split total_rows {split.total_rows} != expected dataset rows {expected_total_rows}"
            )
        split.validate()
        return split


def load_split(
    path: str | Path,
    *,
    expected_total_rows: int | None = None,
) -> TrainValidationSplit:
    """Module-level convenience wrapper around :meth:`TrainValidationSplit.load`."""
    return TrainValidationSplit.load(path, expected_total_rows=expected_total_rows)


def make_split(
    total_rows: int,
    *,
    validation_ratio: float = 0.10,
    seed: int = SEED,
) -> TrainValidationSplit:
    """Deterministic global-permutation split of ``range(total_rows)``.

    Validation takes the first ``ceil(n * validation_ratio)`` indices of the
    seeded permutation; training takes the remainder. Re-running with the
    same seed/total/ratio reproduces byte-identical index arrays.
    """
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be in (0, 1), got {validation_ratio}")
    if total_rows <= 0:
        raise ValueError(f"total_rows must be positive, got {total_rows}")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total_rows)
    n_validation = int(np.ceil(total_rows * validation_ratio))
    validation_indices = np.sort(permutation[:n_validation])
    train_indices = np.sort(permutation[n_validation:])
    return TrainValidationSplit.from_arrays(
        train_indices, validation_indices, seed=seed, validation_ratio=validation_ratio
    )
