"""Deterministic encoders for categorical, numerical, and binary features.

Encoder conventions (documented in the artifact):

* Categorical vocabulary IDs: ``0 = UNK/PAD`` (reserved for unseen values),
  ``1 = MISSING`` (reserved for missing values), ``2..N = known categories``
  ordered by (descending train frequency, then ascending value) so that
  construction is deterministic and stable across runs.
* Numerical: standardized ``(x - mean) / std`` using training statistics
  only; missing values imputed with the training mean (computed before
  normalization); zero-variance columns map every value to ``0.0``.
* Binary: validated against {0, 1} (NaN allowed and preserved); passed
  through as float32. No vocabulary is created for binary features.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .exceptions import DataContractError

UNK_ID = 0
MISSING_ID = 1
FIRST_KNOWN_ID = 2


def _category_sort_key(value: Any) -> tuple[int, float, str]:
    """Total, dtype-safe ordering key: numerics by value, strings after."""
    if isinstance(value, (bool, np.bool_)):
        return (0, float(value), "")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (0, float(value), "")
    return (1, 0.0, str(value))


def _stable_category_order(value_counts: pd.Series) -> list[Any]:
    """Order categories by (descending frequency, ascending value)."""
    items = [(int(count), value) for value, count in value_counts.items()]
    items.sort(key=lambda item: (-item[0], _category_sort_key(item[1])))
    return [value for _, value in items]


@dataclass
class CategoricalVocabulary:
    """Train-fitted integer-ID vocabulary for one categorical feature.

    Reserved IDs: ``0 = UNK`` (unseen at transform time), ``1 = MISSING``.
    Known category IDs start at 2. Ordering is deterministic: categories are
    sorted by descending train frequency, ties broken by ascending value.
    """

    feature: str
    id_to_value: dict[int, Any] = field(default_factory=dict)
    unknown_id: int = UNK_ID
    missing_id: int = MISSING_ID

    def __post_init__(self) -> None:
        self.id_to_value = {int(k): v for k, v in self.id_to_value.items()}

    @property
    def value_to_id(self) -> dict[Any, int]:
        return {v: k for k, v in self.id_to_value.items()}

    @property
    def vocab_size(self) -> int:
        """Number of embedding rows needed: max known ID + 1, at least 2."""
        max_known = max(self.id_to_value) if self.id_to_value else 1
        return max(max_known + 1, self.unknown_id + 1, self.missing_id + 1)

    @property
    def num_known_categories(self) -> int:
        return len(self.id_to_value)

    def transform(self, series: pd.Series) -> np.ndarray:
        mapping = self.value_to_id
        # Dict-based map: unseen values and NaN both become NaN here and are
        # disambiguated below (NaN input -> MISSING, unseen -> UNK).
        mapped = series.map(mapping)
        is_missing = series.isna()
        ids = mapped.fillna(self.unknown_id).astype("int64")
        ids[is_missing] = self.missing_id
        return ids.to_numpy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            # Keys serialized as strings; values as their string form so the
            # mapping survives JSON round-trips without dtype surprises.
            "id_to_value": {str(k): _json_value(v) for k, v in self.id_to_value.items()},
            "unknown_id": self.unknown_id,
            "missing_id": self.missing_id,
            "vocab_size": self.vocab_size,
            "num_known_categories": self.num_known_categories,
            "category_order": "descending_train_frequency_then_ascending_value",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CategoricalVocabulary":
        id_to_value = {
            int(k): _from_json_value(v) for k, v in payload["id_to_value"].items()
        }
        return cls(
            feature=str(payload["feature"]),
            id_to_value=id_to_value,
            unknown_id=int(payload.get("unknown_id", UNK_ID)),
            missing_id=int(payload.get("missing_id", MISSING_ID)),
        )

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _json_value(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)


def _from_json_value(value: str) -> Any:
    """Inverse of :func:`_json_value` for int/float/str categories."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def fit_categorical_vocabulary(
    feature: str,
    value_counts: pd.Series,
    *,
    max_cardinality: int | None = None,
) -> CategoricalVocabulary:
    """Fit a vocabulary from a value->count table (training data only).

    ``value_counts`` must be a Series whose index holds the distinct category
    values and whose values hold their training frequencies (as produced by
    ``Series.value_counts()`` or the streaming counting pass).

    Raises :class:`DataContractError` when the observed cardinality exceeds
    ``max_cardinality`` so hashing is never applied silently.
    """
    if max_cardinality is not None and len(value_counts) > max_cardinality:
        raise DataContractError(
            f"Categorical feature {feature!r} has {len(value_counts)} distinct "
            f"training values > max_cardinality={max_cardinality}. Refusing to "
            "fit an explicit vocabulary; hashing would have to be decided "
            "explicitly."
        )
    ordered = _stable_category_order(value_counts)
    return CategoricalVocabulary(
        feature=feature,
        id_to_value={FIRST_KNOWN_ID + i: value for i, value in enumerate(ordered)},
    )


# A fitted std below this (relative to |mean|, floored at 1.0) is treated as
# zero variance. Floating-point summation can leave a residual std of ~1e-17
# on perfectly constant columns; such columns must map to 0.0, not to
# enormous normalized values.
ZERO_VARIANCE_EPS = 1e-12


@dataclass
class NumericalStats:
    """Train-fitted statistics for one numerical feature.

    ``mean`` is computed over non-missing training values and doubles as the
    missing-value imputation constant. ``std`` is the population standard
    deviation over non-missing training values. For zero-variance columns
    ``std == 0`` and the transform maps every value to ``0.0``.
    """

    feature: str
    mean: float
    std: float
    missing_count: int
    total_count: int
    imputation_value: float | None = None  # None => no missing values seen

    def __post_init__(self) -> None:
        self.mean = float(self.mean)
        self.std = float(self.std)
        self.missing_count = int(self.missing_count)
        self.total_count = int(self.total_count)
        # Zero-variance snap: constant training columns must map to 0.0
        # (see ZERO_VARIANCE_EPS). Deterministic in the fitted statistics.
        if self.std < ZERO_VARIANCE_EPS * max(1.0, abs(self.mean)):
            self.std = 0.0
        # Imputation constant defaults to the training mean even for columns
        # with no training missingness: a later split may still contain NaNs,
        # and transform must remain total (never crash, never emit NaN).
        if self.imputation_value is None:
            self.imputation_value = self.mean

    def transform(self, series: pd.Series) -> np.ndarray:
        values = series.to_numpy(dtype=np.float64, copy=True)
        present = ~np.isnan(values)
        if self.std == 0.0:
            # Zero-variance column: every value (and any imputation) maps to 0.
            normalized = np.zeros(values.shape, dtype=np.float64)
        else:
            normalized = np.where(
                present,
                (values - self.mean) / self.std,
                (self.imputation_value - self.mean) / self.std,
            )
        return normalized.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "mean": self.mean,
            "std": self.std,
            "missing_count": self.missing_count,
            "total_count": self.total_count,
            "imputation_value": self.imputation_value,
            "missing_indicator": self.missing_count > 0,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NumericalStats":
        return cls(
            feature=str(payload["feature"]),
            mean=float(payload["mean"]),
            std=float(payload["std"]),
            missing_count=int(payload["missing_count"]),
            total_count=int(payload["total_count"]),
            imputation_value=(
                None
                if payload.get("imputation_value") is None
                else float(payload["imputation_value"])
            ),
        )

    @staticmethod
    def fit(feature: str, series: pd.Series) -> "NumericalStats":
        """Fit statistics from training data only."""
        values = series.to_numpy(dtype=np.float64)
        present = values[~np.isnan(values)]
        missing_count = int(values.size - present.size)
        if present.size == 0:
            raise DataContractError(
                f"Numerical feature {feature!r} has no observed training values; "
                "cannot fit statistics."
            )
        mean = float(present.mean())
        std = float(present.std())
        return NumericalStats(
            feature=feature,
            mean=mean,
            std=std,
            missing_count=missing_count,
            total_count=int(values.size),
        )


MISSING_INDICATOR_SUFFIX = "_missing"


def missing_indicator_name(feature: str) -> str:
    return f"{feature}{MISSING_INDICATOR_SUFFIX}"


def build_missing_indicators(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return a float32 frame of 0/1 missing indicators for ``columns``."""
    indicators = {}
    for column in columns:
        indicators[missing_indicator_name(column)] = (
            df[column].isna().to_numpy(dtype=np.float32)
        )
    return pd.DataFrame(indicators, index=df.index)


def validate_binary_column(feature: str, series: pd.Series) -> None:
    """Validate that a binary column contains only 0, 1, or missing values.

    Missing values are an allowed, documented representation (the feature
    stays float; downstream models treat NaN as 'unknown').
    """
    present = series.dropna()
    bad = present[~present.isin([0, 1])]
    if len(bad) > 0:
        examples = sorted({repr(v) for v in bad.unique()[:5]})
        raise DataContractError(
            f"Binary feature {feature!r} contains non-binary values {examples} "
            f"({len(bad)} offending rows)."
        )
