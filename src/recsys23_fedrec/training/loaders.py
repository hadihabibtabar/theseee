"""DataLoader factories for index subsets of the Stage 3 Parquet dataset.

``make_subset_loader`` wraps the memory-bounded ``ProcessedRecSysDataset``
with a batch sampler over *local* positions of an index subset. The dataset
itself is never copied or materialized; batches are still read row-group by
row-group.

Training uses :class:`PartShuffledBatchSampler` (deterministic per-epoch
part-shuffled batches); validation uses :class:`SortedBatchSampler` (ascending
local order, monotone Parquet access).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from recsys23_fedrec.data import ProcessedRecSysDataset, collate_train
from recsys23_fedrec.training.sampler import PartShuffledBatchSampler, SortedBatchSampler

__all__ = ["SubsetDataset", "make_subset_loader"]


class SubsetDataset(torch.utils.data.Dataset):
    """View of the underlying dataset through local subset positions."""

    def __init__(self, dataset: ProcessedRecSysDataset, subset_indices: np.ndarray) -> None:
        self.dataset = dataset
        self._indices = np.asarray(subset_indices, dtype=np.int64)
        if self._indices.size and (self._indices.min() < 0 or self._indices.max() >= len(dataset)):
            raise IndexError("subset indices outside the dataset range")

    def __len__(self) -> int:
        return int(self._indices.size)

    def __getitem__(self, local_index):
        if isinstance(local_index, int):
            return self.dataset[int(self._indices[local_index])]
        # Slice support: translate to the underlying dataset slice protocol.
        idx = self._indices[local_index]
        return [self.dataset[int(i)] for i in idx]

    @property
    def indices(self) -> np.ndarray:
        return self._indices


def make_subset_loader(
    dataset: ProcessedRecSysDataset,
    subset_indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader over ``dataset[subset_indices]`` (bounded memory).

    ``shuffle=True`` selects the deterministic part-shuffled batch sampler
    (training); ``shuffle=False`` yields ascending batches (validation).
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    subset = SubsetDataset(dataset, subset_indices)
    if shuffle:
        sampler = PartShuffledBatchSampler(
            subset_indices=np.asarray(subset_indices, dtype=np.int64),
            batch_size=batch_size,
            seed=seed,
        )
    else:
        sampler = SortedBatchSampler(subset_size=len(subset), batch_size=batch_size)
    return torch.utils.data.DataLoader(
        subset,
        batch_sampler=sampler,
        collate_fn=collate_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
