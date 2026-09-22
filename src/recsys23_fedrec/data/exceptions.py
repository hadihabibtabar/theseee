"""Data-layer exceptions for the Dataset/DataLoader subsystem."""

from __future__ import annotations


class DataError(Exception):
    """Base class for data-layer errors."""


class BatchValidationError(DataError):
    """Raised when a batch violates the schema contract (shape/dtype/range)."""


class FeatureOrderError(DataError):
    """Raised when the processed data does not match the artifact feature order."""
