"""Preprocessing subsystem for the RecSys23 federated-installation project.

Stage 2 of the research program: convert raw TAB-separated CSV rows into
model-ready numerical representations with a leakage-safe, deterministic,
streaming pipeline. Neural models (Transformer, MMoE, SSL) and federated
simulation arrive in later stages.
"""

from .config import PreprocessingConfig
from .encoders import (
    CategoricalVocabulary,
    NumericalStats,
    MISSING_ID,
    UNK_ID,
)
from .exceptions import (
    ArtifactError,
    DataContractError,
    FeatureSchemaError,
    NotFittedError,
    PreprocessingError,
)
from .feature_schema import FeatureRole, FeatureSchema, sort_columns_like_raw
from .preprocessor import Preprocessor
from .raw_data import RawShardReader, count_raw_rows
from .storage import PreprocessingArtifactStore

__all__ = [
    "PreprocessingConfig",
    "Preprocessor",
    "PreprocessingArtifactStore",
    "FeatureSchema",
    "FeatureRole",
    "sort_columns_like_raw",
    "CategoricalVocabulary",
    "NumericalStats",
    "RawShardReader",
    "count_raw_rows",
    "UNK_ID",
    "MISSING_ID",
    "PreprocessingError",
    "FeatureSchemaError",
    "DataContractError",
    "NotFittedError",
    "ArtifactError",
]
