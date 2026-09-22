"""Stage 3 data layer: PyTorch datasets over the Stage 2 processed Parquet.

Public API::

    FeatureSet.load("artifacts/preprocessing")   # artifact-frozen feature order
    ProcessedRecSysDataset(feature_set, "artifacts/processed/train", split="train")
    create_dataloader(dataset, batch_size=1024, ...)
    validate_batch(batch, feature_set)           # contract checks
"""

from .collate import collate_test, collate_train, create_dataloader
from .dataset import ProcessedRecSysDataset
from .exceptions import BatchValidationError, DataError, FeatureOrderError
from .feature_schema import FeatureSet, GPU
from .samples import (
    BatchValidator,
    TestSample,
    TrainingSample,
    validate_batch,
)

__all__ = [
    "ProcessedRecSysDataset",
    "FeatureSet",
    "GPU",
    "create_dataloader",
    "collate_train",
    "collate_test",
    "BatchValidator",
    "validate_batch",
    "TrainingSample",
    "TestSample",
    "DataError",
    "BatchValidationError",
    "FeatureOrderError",
]
