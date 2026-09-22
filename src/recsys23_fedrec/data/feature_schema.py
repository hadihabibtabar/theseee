"""GPU/feature-schema definition shared by the data layer and later stages.

``FeatureSet`` is the frozen bridge to the Stage 2 preprocessing artifact: it
holds the artifact-defined feature ordering, vocabulary bounds, and tensor
column indices. Nothing in the data layer may reconstruct feature order from
Parquet column order, alphabetic sorting, or dictionary iteration — the
artifact is the single source of truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from recsys23_fedrec.preprocessing import Preprocessor, PreprocessingArtifactStore
from recsys23_fedrec.preprocessing.encoders import missing_indicator_name

from .exceptions import FeatureOrderError

ARTIFACT_FILENAME = "artifact.json"


@dataclass(frozen=True)
class FeatureSet:
    """Artifact-frozen feature ordering and per-column metadata.

    ``*_names`` lists are in Stage 2's deterministic processed order
    (categorical, binary, numerical, then missing indicators).
    """

    categorical_names: tuple[str, ...]
    numerical_names: tuple[str, ...]
    binary_names: tuple[str, ...]
    missing_indicator_names: tuple[str, ...]
    vocabulary_sizes: tuple[int, ...]
    artifact_hash: str
    schema_hash: str
    artifact_dir: str

    # ------------------------------------------------------------------
    @classmethod
    def from_preprocessor(cls, preprocessor: Preprocessor, artifact_dir: str) -> "FeatureSet":
        """Freeze a FeatureSet from a fitted/loaded Preprocessor."""
        schema = preprocessor.schema
        return cls(
            categorical_names=tuple(schema.categorical_columns),
            numerical_names=tuple(schema.numerical_columns),
            binary_names=tuple(schema.binary_columns),
            # Artifact stores base feature names; Parquet columns carry the
            # '<feature>_missing' suffix applied by Stage 2 at write time.
            missing_indicator_names=tuple(
                missing_indicator_name(c) for c in preprocessor.missing_indicator_columns
            ),
            vocabulary_sizes=tuple(preprocessor.vocabulary_sizes_by_order()),
            artifact_hash=preprocessor.artifact_hash,
            schema_hash=schema.schema_hash,
            artifact_dir=artifact_dir,
        )

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "FeatureSet":
        """Load and hash-verify the Stage 2 artifact, then freeze it.

        Accepts either the output root (``artifacts``) or the inner
        artifact directory (``artifacts/preprocessing``); both conventions
        resolve to the same artifact files.
        """
        path = Path(artifact_dir)
        if (path / "preprocessing" / ARTIFACT_FILENAME).is_file():
            root = path
            store_dir = path / "preprocessing"
        elif (path / ARTIFACT_FILENAME).is_file():
            root = path.parent
            store_dir = path
        else:
            raise FileNotFoundError(
                f"No preprocessing artifact found under {path} "
                f"(looked for {path / 'preprocessing' / ARTIFACT_FILENAME} "
                f"and {path / ARTIFACT_FILENAME})"
            )
        preprocessor = Preprocessor.load(str(root))
        return cls.from_preprocessor(preprocessor, str(store_dir))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def num_categorical(self) -> int:
        return len(self.categorical_names)

    @property
    def num_numerical(self) -> int:
        return len(self.numerical_names)

    @property
    def num_binary(self) -> int:
        return len(self.binary_names)

    @property
    def num_missing(self) -> int:
        return len(self.missing_indicator_names)

    @property
    def target_names(self) -> tuple[str, str]:
        """Label column names in (click, install) order, per Stage 2 artifact."""
        return ("is_clicked", "is_installed")

    def expected_train_columns(self) -> tuple[str, ...]:
        return (
            *self.categorical_names,
            *self.binary_names,
            *self.numerical_names,
            *self.missing_indicator_names,
            *self.target_names,
        )

    def expected_test_columns(self) -> tuple[str, ...]:
        return (
            *self.categorical_names,
            *self.binary_names,
            *self.numerical_names,
            *self.missing_indicator_names,
        )


@dataclass(frozen=True)
class GPU:
    """Feature schema + device context bundle for later training stages.

    Bundle so training code can create one object and pass it around. The
    data layer itself stays on CPU; ``device`` merely records the intended
    training device for the later loop (never used to move tensors here).
    """

    feature_set: FeatureSet
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    pin_memory: bool = False

    def __post_init__(self) -> None:
        if self.pin_memory and not torch.cuda.is_available():
            raise ValueError("pin_memory=True requires CUDA to be available")

    def to_dict(self) -> dict:
        return {
            "device": str(self.device),
            "pin_memory": self.pin_memory,
            "artifact_hash": self.feature_set.artifact_hash,
        }
