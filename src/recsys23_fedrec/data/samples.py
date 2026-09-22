"""Sample structure and batch validation for the data layer.

``TrainingSample`` documents the exact per-sample structure; ``validate_batch``
is the reusable contract checker used by tests, the smoke script, and (later)
training loops.

Field contracts (shapes shown per batch of size B; ``BatchValidator`` derives
all widths from the Stage 2 artifact):

==================  ==========  =============================================
field               dtype       shape
==================  ==========  =============================================
``categorical``     int64       [B, num_categorical]        (30)
``numerical``       float32     [B, num_numerical]          (38)
``binary``          float32     [B, num_binary]             (11)
``missing``         float32     [B, num_missing_indicators] (13)
``click``           float32     [B]
``install``         float32     [B]
==================  ==========  =============================================

Labels are exactly {0, 1} (no sigmoid applied — downstream uses
BCEWithLogitsLoss). The dataset stays on CPU; device transfer belongs to the
training loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .exceptions import BatchValidationError

SAMPLE_KEYS = ("categorical", "numerical", "binary", "missing", "click", "install")


@dataclass
class TrainingSample:
    """One processed training example (see module docstring for contracts)."""

    categorical: torch.Tensor  # int64, [num_categorical]
    numerical: torch.Tensor    # float32, [num_numerical]
    binary: torch.Tensor       # float32, [num_binary]
    missing: torch.Tensor      # float32, [num_missing_indicators]
    click: torch.Tensor        # float32, scalar
    install: torch.Tensor      # float32, scalar


@dataclass
class TestSample:
    """One processed test example: features only, no labels."""

    categorical: torch.Tensor  # int64, [num_categorical]
    numerical: torch.Tensor    # float32, [num_numerical]
    binary: torch.Tensor       # float32, [num_binary]
    missing: torch.Tensor      # float32, [num_missing_indicators]


class BatchValidator:
    """Schema-driven batch contract checker.

    All expected widths come from the Stage 2 artifact's feature schema, so
    nothing about the dataset layout is hard-coded here.
    """

    def __init__(self, feature_schema) -> None:
        # Duck-type both public schemas: the data layer's FeatureSet
        # (``*_names`` + ``vocabulary_sizes``) and the preprocessing
        # Preprocessor (``*_columns`` + ``vocabulary_sizes_by_order()``).
        def names(attr: str) -> list[str]:
            for candidate in (f"{attr}_names", f"{attr}_columns"):
                if hasattr(feature_schema, candidate):
                    return list(getattr(feature_schema, candidate))
            raise AttributeError(
                f"feature schema exposes neither {attr}_names nor {attr}_columns"
            )

        self.num_categorical = len(names("categorical"))
        self.num_numerical = len(names("numerical"))
        self.num_binary = len(names("binary"))
        self.num_missing = len(names("missing_indicator"))
        if hasattr(feature_schema, "vocabulary_sizes") and not callable(
            feature_schema.vocabulary_sizes
        ):
            self.vocabulary_sizes = list(feature_schema.vocabulary_sizes)
        else:
            self.vocabulary_sizes = list(feature_schema.vocabulary_sizes_by_order())

    # ------------------------------------------------------------------
    def _check(
        self,
        batch: dict,
        key: str,
        *,
        dtype: torch.dtype,
        shape: tuple,
        batch_size: int,
        failures: list[str],
    ) -> None:
        if key not in batch:
            failures.append(f"missing key {key!r}")
            return
        tensor = batch[key]
        if not isinstance(tensor, torch.Tensor):
            failures.append(f"{key}: expected torch.Tensor, got {type(tensor).__name__}")
            return
        if tensor.dtype != dtype:
            failures.append(f"{key}: dtype {tensor.dtype} != expected {dtype}")
        expected_shape = (batch_size,) + shape
        if tuple(tensor.shape) != expected_shape:
            failures.append(f"{key}: shape {tuple(tensor.shape)} != expected {expected_shape}")

    def _check_values(
        self,
        batch: dict,
        key: str,
        checker,
        failures: list[str],
        *,
        skip_if_missing: bool = True,
    ) -> None:
        tensor = batch.get(key)
        if not isinstance(tensor, torch.Tensor):
            return
        result = checker(tensor)
        if result is not True:
            failures.append(f"{key}: value check failed ({result})")

    # ------------------------------------------------------------------
    def validate_train_batch(self, batch: dict) -> None:
        """Validate a training batch; raise BatchValidationError on failure."""
        failures: list[str] = []
        if "categorical" not in batch:
            failures.append("missing key 'categorical'")
            raise BatchValidationError("; ".join(failures))
        batch_size = int(batch["categorical"].shape[0]) if isinstance(batch["categorical"], torch.Tensor) else -1

        self._check(batch, "categorical", dtype=torch.int64, shape=(self.num_categorical,), batch_size=batch_size, failures=failures)
        self._check(batch, "numerical", dtype=torch.float32, shape=(self.num_numerical,), batch_size=batch_size, failures=failures)
        self._check(batch, "binary", dtype=torch.float32, shape=(self.num_binary,), batch_size=batch_size, failures=failures)
        self._check(batch, "missing", dtype=torch.float32, shape=(self.num_missing,), batch_size=batch_size, failures=failures)
        self._check(batch, "click", dtype=torch.float32, shape=(), batch_size=batch_size, failures=failures)
        self._check(batch, "install", dtype=torch.float32, shape=(), batch_size=batch_size, failures=failures)

        self._check_values(batch, "categorical", self._categorical_in_range, failures)
        self._check_values(batch, "numerical", self._finite, failures)
        self._check_values(batch, "binary", self._indicator_like, failures)
        self._check_values(batch, "missing", self._indicator_like, failures)
        self._check_values(batch, "click", self._indicator_like, failures)
        self._check_values(batch, "install", self._indicator_like, failures)

        if failures:
            raise BatchValidationError("; ".join(failures))

    def validate_test_batch(self, batch: dict) -> None:
        """Validate a label-free (test/validation) batch."""
        failures: list[str] = []
        if "categorical" not in batch:
            failures.append("missing key 'categorical'")
            raise BatchValidationError("; ".join(failures))
        batch_size = int(batch["categorical"].shape[0]) if isinstance(batch["categorical"], torch.Tensor) else -1

        for key, dtype, width in (
            ("categorical", torch.int64, self.num_categorical),
            ("numerical", torch.float32, self.num_numerical),
            ("binary", torch.float32, self.num_binary),
            ("missing", torch.float32, self.num_missing),
        ):
            self._check(batch, key, dtype=dtype, shape=(width,), batch_size=batch_size, failures=failures)
        self._check_values(batch, "categorical", self._categorical_in_range, failures)
        self._check_values(batch, "numerical", self._finite, failures)
        self._check_values(batch, "binary", self._indicator_like, failures)
        self._check_values(batch, "missing", self._indicator_like, failures)
        for key in ("click", "install"):
            if key in batch:
                failures.append(f"test batch must not contain label key {key!r}")
        if failures:
            raise BatchValidationError("; ".join(failures))

    # ------------------------------------------------------------------
    # Value checkers return True or a human-readable reason.
    # ------------------------------------------------------------------
    def _categorical_in_range(self, tensor: torch.Tensor):
        # Per-column bounds: each feature j must satisfy 0 <= id < vocab[j].
        max_ids = tensor.amax(dim=0)
        min_ids = tensor.amin(dim=0)
        bounds = torch.tensor(self.vocabulary_sizes, dtype=torch.int64)
        if tensor.shape[1] != len(self.vocabulary_sizes):
            return f"expected {len(self.vocabulary_sizes)} columns"
        if bool((min_ids < 0).any()):
            return "negative categorical IDs present"
        bad = (max_ids >= bounds).nonzero().flatten().tolist()
        if bad:
            columns = [f"col{j}" for j in bad]
            return f"out-of-range IDs in {columns} (vocabulary overflow)"
        return True

    @staticmethod
    def _finite(tensor: torch.Tensor):
        if not bool(torch.isfinite(tensor).all()):
            return "NaN/Inf values present"
        return True

    @staticmethod
    def _indicator_like(tensor: torch.Tensor):
        unique = torch.unique(tensor)
        if not bool(((unique == 0.0) | (unique == 1.0)).all()):
            return f"values outside {{0,1}}: {unique.tolist()[:6]}"
        return True


def validate_batch(batch: dict, feature_schema, *, split: str = "train") -> None:
    """Module-level convenience wrapper around :class:`BatchValidator`.

    ``feature_schema`` must expose the same interface as
    ``recsys23_fedrec.preprocessing.Preprocessor`` (categorical/numerical/
    binary/missing lists and ``vocabulary_sizes_by_order()``).
    """
    validator = BatchValidator(feature_schema)
    if split == "train":
        validator.validate_train_batch(batch)
    else:
        validator.validate_test_batch(batch)
