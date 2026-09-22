"""Deterministic part-shuffled batch sampler over a dataset index subset.

Full per-row shuffling would require random access into the Stage 3
row-group-streaming Parquet dataset, which re-reads a ~50k-row row group per
hop (impractical on this storage). Instead, each epoch's global index order
is produced deterministically (seeded ``numpy.random.Generator``), then
translated into the subset's *local* positions and grouped into batches that
respect ascending global order **within each batch**:

* global-order blocks are loaded from Parquet sequentially (cache-friendly,
  bounded memory), while
* the *sequence of blocks/batches* is shuffled across the epoch, which
  decorrelates training batches enough for a clean supervised baseline.

Determinism: identical seed + epoch count + subset => identical batch
sequence. The epoch count participates in the order, so epoch 2 sees a
different (but reproducible) arrangement than epoch 1.

``SortedBatchSampler`` is the validation counterpart: batches in plain
ascending local order (monotone dataset access).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Sampler

__all__ = ["PartShuffledBatchSampler", "SortedBatchSampler"]


def _validate_batch_indices(local_positions: np.ndarray, subset_size: int) -> None:
    if local_positions.min() < 0 or local_positions.max() >= subset_size:
        raise IndexError(
            f"local position out of range for subset of size {subset_size}"
        )


@dataclass
class PartShuffledBatchSampler(Sampler):
    """Yield lists of *local* subset positions, batch by batch.

    Args:
        subset_indices: global dataset indices of the subset (sorted).
        batch_size: samples per batch (last batch may be smaller).
        seed: project seed; drives the deterministic permutation.
        epoch: starting epoch counter; call ``set_epoch`` before each pass
            so successive epochs permute differently but reproducibly.
    """

    subset_indices: np.ndarray
    batch_size: int
    seed: int = 42
    epoch: int = 0

    def __post_init__(self) -> None:
        self.subset_indices = np.asarray(self.subset_indices, dtype=np.int64)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        self._global_sorted = np.sort(self.subset_indices)
        # local position of each global index when the subset is sorted:
        # pos_of_global[g - base] for global indices in the subset.
        base = int(self._global_sorted[0]) if len(self._global_sorted) else 0
        self._base = base
        self._pos_of_global = np.full(int(self._global_sorted[-1]) - base + 1, -1, dtype=np.int64)
        self._pos_of_global[self._global_sorted - base] = np.arange(len(self._global_sorted))

    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _epoch_batches(self, epoch: int) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + int(epoch))
        order = rng.permutation(len(self._global_sorted))  # local positions
        n_batches = math.ceil(len(order) / self.batch_size)
        batches: list[list[int]] = []
        for b in range(n_batches):
            chunk = np.sort(order[b * self.batch_size : (b + 1) * self.batch_size])
            _validate_batch_indices(chunk, len(self._global_sorted))
            batches.append(chunk.tolist())
        rng.shuffle(batches)  # shuffle the *batch sequence* only
        return batches

    def __iter__(self):
        return iter(self._epoch_batches(self.epoch))

    def __len__(self) -> int:
        return math.ceil(len(self._global_sorted) / self.batch_size)


@dataclass
class SortedBatchSampler(Sampler):
    """Batches in ascending local-position order (validation pass)."""

    subset_size: int
    batch_size: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

    def __iter__(self):
        for start in range(0, self.subset_size, self.batch_size):
            yield list(range(start, min(start + self.batch_size, self.subset_size)))

    def __len__(self) -> int:
        return math.ceil(self.subset_size / self.batch_size)
