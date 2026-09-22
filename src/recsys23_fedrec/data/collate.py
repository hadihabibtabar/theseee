"""Collate functions and DataLoader factory.

The collate functions stack per-sample dicts into batch tensors with shapes
derived from the Stage 2 artifact (never hard-coded):

    categorical: int64   [B, num_categorical]        (30)
    numerical:   float32 [B, num_numerical]          (38)
    binary:      float32 [B, num_binary]             (11)
    missing:     float32 [B, num_missing_indicators] (13)
    click:       float32 [B]        (train only)
    install:     float32 [B]        (train only)

Feature ordering is preserved because every per-sample tensor is built in
the artifact's deterministic column order; stacking concatenates along the
batch dimension only. No device transfer happens here (CPU only).
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from .exceptions import DataError
from .samples import SAMPLE_KEYS


def collate_train(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack training samples into the batch contract (see module docstring)."""
    if not batch:
        raise DataError("collate_train received an empty batch")
    first = batch[0]
    unexpected = set(first) - set(SAMPLE_KEYS)
    if unexpected:
        raise DataError(f"unexpected sample keys {sorted(unexpected)}")
    missing_keys = {"click", "install"} - set(first)
    if missing_keys:
        raise DataError(f"training samples missing keys {sorted(missing_keys)}")
    return {
        "categorical": torch.stack([s["categorical"] for s in batch]).contiguous(),
        "numerical": torch.stack([s["numerical"] for s in batch]).contiguous(),
        "binary": torch.stack([s["binary"] for s in batch]).contiguous(),
        "missing": torch.stack([s["missing"] for s in batch]).contiguous(),
        "click": torch.stack([s["click"] for s in batch]).contiguous(),
        "install": torch.stack([s["install"] for s in batch]).contiguous(),
    }


def collate_test(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack label-free samples (test/validation)."""
    if not batch:
        raise DataError("collate_test received an empty batch")
    first = batch[0]
    unexpected = set(first) - set(SAMPLE_KEYS)
    if unexpected:
        raise DataError(f"unexpected sample keys {sorted(unexpected)}")
    return {
        "categorical": torch.stack([s["categorical"] for s in batch]).contiguous(),
        "numerical": torch.stack([s["numerical"] for s in batch]).contiguous(),
        "binary": torch.stack([s["binary"] for s in batch]).contiguous(),
        "missing": torch.stack([s["missing"] for s in batch]).contiguous(),
    }


def create_dataloader(
    dataset,
    *,
    batch_size: int = 1024,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    drop_last: bool = False,
    collate_fn=None,
    generator: torch.Generator | None = None,
    **kwargs: Any,
) -> DataLoader:
    """Build a DataLoader with project-consistent defaults.

    ``pin_memory`` defaults to ``torch.cuda.is_available()`` when not given
    (only meaningful with CUDA). ``drop_last`` affects iteration count only —
    the Dataset length is untouched. ``num_workers>0`` uses the dataset's
    ``worker_init_fn`` so each worker process builds its own Parquet readers.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if collate_fn is None:
        collate_fn = collate_train if getattr(dataset, "include_targets", False) else collate_test
    worker_init = getattr(dataset, "worker_init_fn", None) if num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        worker_init_fn=worker_init,
        generator=generator,
        **kwargs,
    )
