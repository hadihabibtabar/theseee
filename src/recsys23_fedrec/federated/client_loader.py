"""Stage 9: per-client DataLoaders over the frozen Stage 3 infrastructure.

Composition (nothing new is implemented — everything is reused):

    frozen train_indices            (artifacts/splits/centralized_split.npz)
        ↓
    client_positions                (Stage 9 partition NPZ, 0 .. 3,137,265)
        ↓
    train_indices[client_positions] (dataset row indices of ONE client)
        ↓
    make_subset_loader              (existing Stage 3/8 loader factory)
        ↓
    client DataLoader

The dataset, preprocessing, collate contract, and the deterministic
part-shuffled batch sampler are exactly the Stage 3/Stage 8 objects; this
module only composes index arrays and forwards arguments. Validation rows
(they live outside ``train_indices``) and official test data can never
enter a client loader by construction.

Memory stays bounded: the loader streams the Stage 3 Parquet row groups
(~50k rows per process) exactly like centralized training; no client
subset is materialized.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from recsys23_fedrec.federated.partition import CLIENT_IDS, PartitionError
from recsys23_fedrec.training.loaders import make_subset_loader
from recsys23_fedrec.training.sampler import PartShuffledBatchSampler

__all__ = [
    "load_partition_positions",
    "compose_client_row_indices",
    "make_client_loader",
    "client_sampler",
]


def load_partition_positions(path: str | Path) -> dict[str, np.ndarray]:
    """Load the Stage 9 partition NPZ into validated per-client positions.

    Returns ``{client_id: sorted int64 positions into train_indices}``.
    Raises :class:`PartitionError` on a malformed artifact (missing keys,
    extra keys, wrong dtype/dimensionality, unsorted or duplicated
    positions).
    """
    with np.load(path) as npz:
        keys = set(npz.files)
        expected = set(CLIENT_IDS) | {"partition_hash"}
        if keys != expected:
            raise PartitionError(
                f"partition NPZ keys {sorted(keys)} != expected {sorted(expected)}"
            )
        positions = {cid: np.asarray(npz[cid]) for cid in CLIENT_IDS}
        stored_hash = str(npz["partition_hash"])
    if not stored_hash:
        raise PartitionError("partition NPZ is missing a partition_hash value")
    for cid, pos in positions.items():
        if pos.ndim != 1 or pos.size == 0:
            raise PartitionError(f"{cid}: positions must be a non-empty 1-D array")
        if pos.dtype != np.int64:
            raise PartitionError(f"{cid}: positions must be int64, got {pos.dtype}")
        if np.any(np.diff(pos) <= 0):
            raise PartitionError(f"{cid}: positions must be strictly ascending (unique)")
        if pos[0] < 0:
            raise PartitionError(f"{cid}: negative position found")
    return positions


def compose_client_row_indices(
    train_indices: np.ndarray, client_positions: np.ndarray
) -> np.ndarray:
    """Compose one client's dataset row indices (frozen split ⊕ partition).

    Validates that the client positions lie inside the train-index space so
    validation/test rows are structurally unreachable.
    """
    train_indices = np.asarray(train_indices, dtype=np.int64)
    client_positions = np.asarray(client_positions, dtype=np.int64)
    if client_positions.size == 0:
        raise PartitionError("client has zero rows")
    if client_positions[-1] >= len(train_indices) or client_positions[0] < 0:
        raise PartitionError(
            "client positions reach outside the frozen train index space; "
            "validation/test rows must never enter a client partition"
        )
    return train_indices[client_positions]


def make_client_loader(
    dataset,
    train_indices: np.ndarray,
    client_positions: np.ndarray,
    *,
    batch_size: int = 1024,
    shuffle: bool = True,
    seed: int = 42,
    epoch: int = 0,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> torch.utils.data.DataLoader:
    """DataLoader over ONE client's rows (see module docstring).

    Thin wrapper around :func:`make_subset_loader` with the Stage 8
    corrected per-epoch reshuffling exposed: ``epoch`` (0-based) is applied
    via :meth:`PartShuffledBatchSampler.set_epoch`, so epoch ``e`` uses the
    deterministic arrangement ``seed + e`` (epoch 1 ⇒ ``seed + 0``,
    matching the frozen Stage 8 protocol). ``shuffle=False`` yields
    ascending-order batches (deterministic streaming/verification).
    """
    rows = compose_client_row_indices(train_indices, client_positions)
    loader = make_subset_loader(
        dataset,
        rows,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if shuffle and epoch:
        loader.batch_sampler.set_epoch(epoch)
    return loader


def client_sampler(loader: torch.utils.data.DataLoader) -> PartShuffledBatchSampler:
    """Return the client loader's batch sampler (for ``set_epoch`` use)."""
    sampler = loader.batch_sampler
    if not isinstance(sampler, PartShuffledBatchSampler):
        raise TypeError(
            f"expected a PartShuffledBatchSampler batch sampler, got {type(sampler).__name__}"
        )
    return sampler
