"""Stage 6 centralized supervised training subsystem.

Split (deterministic, hashed), metrics (log-loss + ROC-AUC), deterministic
part-shuffled samplers, and the centralized trainer. SSL and federated
components belong to later stages.
"""

from .config import TrainingConfig
from .loaders import SubsetDataset, make_subset_loader
from .metrics import binary_logloss, roc_auc
from .sampler import PartShuffledBatchSampler, SortedBatchSampler
from .split import (
    SplitMetadataError,
    TrainValidationSplit,
    load_split,
    make_split,
)
from .trainer import EpochLog, Trainer, resolve_device

__all__ = [
    "TrainingConfig",
    "SubsetDataset",
    "make_subset_loader",
    "binary_logloss",
    "roc_auc",
    "PartShuffledBatchSampler",
    "SortedBatchSampler",
    "TrainValidationSplit",
    "SplitMetadataError",
    "make_split",
    "load_split",
    "Trainer",
    "EpochLog",
    "resolve_device",
]
