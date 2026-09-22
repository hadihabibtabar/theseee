"""Shared exception types for the preprocessing subsystem.

All preprocessing errors derive from :class:`PreprocessingError` so callers
can catch the whole family with a single except clause.
"""

from __future__ import annotations


class PreprocessingError(Exception):
    """Base class for all preprocessing subsystem errors."""


class FeatureSchemaError(PreprocessingError):
    """Raised for invalid or inconsistent feature schema definitions."""


class DataContractError(PreprocessingError):
    """Raised when raw input data violates the preprocessing data contract.

    Examples: missing columns, unexpected label values, missing labels in a
    training split, non-binary values in binary columns.
    """


class NotFittedError(PreprocessingError):
    """Raised when ``transform`` is called before ``fit`` completed."""


class ArtifactError(PreprocessingError):
    """Raised when a preprocessing artifact is missing, unreadable, or not
    linked to the processed data being loaded."""
