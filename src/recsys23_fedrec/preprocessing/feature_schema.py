"""Feature schema: deterministic feature categorization from Stage 1 findings.

The schema assigns every raw column exactly one role:

* ``id``           - row identifier, never a model feature (``f_0``)
* ``target``       - binary label (``is_clicked``, ``is_installed``)
* ``categorical``  - discrete token, mapped to embedding IDs via vocabulary
* ``numerical``    - continuous value, standardized
* ``binary``       - numerical feature with values in {0, 1} (kept as-is)

Ordering is deterministic everywhere: all column lists are sorted by raw
column index derived from the numeric suffix (``f_2`` < ``f_10``), and
targets keep their raw-file order (``is_clicked`` before ``is_installed``).

Classification rules (implemented in :func:`build_feature_schema` and driven
by the Stage 1 report / light raw-data probing):

* ``id_columns`` and ``target_columns`` come from Stage 1 verbatim.
* A feature is *categorical* when its training dtype is integer, it contains
  no missing values, it has at most ``max_categorical_cardinality`` distinct
  values, and it is NOT binary under the {0, 1}-set rule below.
* A feature is *binary* when the set of observed training values is a subset
  of {0, 1} (missing values allowed). This rule deliberately overrides the
  Stage 1 cardinality-based "binary" flag: Stage 1 flagged ``f_26``..``f_29``
  as binary because each has exactly two distinct values, but those values
  are raw IDs such as {0, 14897}, not {0, 1}. Such features are categorical.
  The resolved classification is stored in the fitted artifact, so nothing
  is decided implicitly at transform time.
* All remaining features (training dtype float64, non-integral or with
  missing values) are *numerical*.

The resulting categorization is an explicit, data-derived decision recorded
in the artifact; no column is silently dropped and no feature is silently
re-classified at transform time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .config import PreprocessingConfig
from .exceptions import FeatureSchemaError


class FeatureRole(str, Enum):
    """Role of a raw column in the preprocessing pipeline."""

    ID = "id"
    TARGET = "target"
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    BINARY = "binary"


def _natural_index(name: str) -> tuple[int, str]:
    """Sort key reproducing raw column order: f_2 < f_10 < f_79 < targets."""
    if name.startswith("f_"):
        try:
            return (0, int(name.split("_", 1)[1]), name)
        except ValueError:
            pass
    if name.startswith("is_"):
        # Targets sit after all f_ columns, in raw-file order.
        return (1, 0, name)
    return (2, 0, name)


def sort_columns_like_raw(columns: Iterable[str]) -> list[str]:
    """Return ``columns`` sorted in raw dataset column order (deterministic)."""
    return sorted(set(columns), key=_natural_index)


@dataclass
class FeatureSchema:
    """Deterministic feature categorization for the preprocessing pipeline.

    Stored inside the fitted preprocessing artifact; the processed Parquet
    data references it by ``schema_hash``.
    """

    id_columns: list[str]
    target_columns: list[str]
    categorical_columns: list[str]
    numerical_columns: list[str]
    binary_columns: list[str]

    # Raw column order of all columns seen in the training file, and the
    # hash of the model-feature lists (used to link processed data to schema).
    raw_column_order: list[str] = field(default_factory=list)
    schema_hash: str = ""
    source_report: str = ""

    @property
    def model_feature_columns(self) -> list[str]:
        """All non-ID, non-target features in deterministic order."""
        return sort_columns_like_raw(
            [*self.categorical_columns, *self.numerical_columns, *self.binary_columns]
        )

    def role_of(self, column: str) -> FeatureRole:
        """Return the role of ``column`` or raise FeatureSchemaError."""
        for role, names in (
            (FeatureRole.ID, self.id_columns),
            (FeatureRole.TARGET, self.target_columns),
            (FeatureRole.CATEGORICAL, self.categorical_columns),
            (FeatureRole.NUMERICAL, self.numerical_columns),
            (FeatureRole.BINARY, self.binary_columns),
        ):
            if column in names:
                return role
        raise FeatureSchemaError(f"Unknown column {column!r} in feature schema")

    def validate(self) -> None:
        """Fail loudly on overlapping, missing, or non-deterministic roles."""
        buckets = {
            "id": self.id_columns,
            "target": self.target_columns,
            "categorical": self.categorical_columns,
            "numerical": self.numerical_columns,
            "binary": self.binary_columns,
        }
        seen: dict[str, str] = {}
        for role, names in buckets.items():
            for name in names:
                if name in seen:
                    raise FeatureSchemaError(
                        f"Column {name!r} assigned to both {seen[name]!r} and {role!r}"
                    )
                seen[name] = role

        feature_names = (
            self.categorical_columns + self.numerical_columns + self.binary_columns
        )
        if len(feature_names) != len(set(feature_names)):
            raise FeatureSchemaError("Duplicate feature names in schema")
        if self.raw_column_order:
            expected = sort_columns_like_raw(seen)
            if set(expected) != set(self.raw_column_order):
                raise FeatureSchemaError(
                    "raw_column_order does not match the union of schema roles"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id_columns": list(self.id_columns),
            "target_columns": list(self.target_columns),
            "categorical_columns": list(self.categorical_columns),
            "numerical_columns": list(self.numerical_columns),
            "binary_columns": list(self.binary_columns),
            "raw_column_order": list(self.raw_column_order),
            "schema_hash": self.schema_hash,
            "source_report": self.source_report,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureSchema":
        return cls(
            id_columns=list(payload["id_columns"]),
            target_columns=list(payload["target_columns"]),
            categorical_columns=list(payload["categorical_columns"]),
            numerical_columns=list(payload["numerical_columns"]),
            binary_columns=list(payload["binary_columns"]),
            raw_column_order=list(payload.get("raw_column_order", [])),
            schema_hash=str(payload.get("schema_hash", "")),
            source_report=str(payload.get("source_report", "")),
        )

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json())

    @classmethod
    def load_json(cls, path: str | Path) -> "FeatureSchema":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def build_feature_schema(
    *,
    id_columns: Sequence[str],
    target_columns: Sequence[str],
    observed_columns: Sequence[str],
    categorical_columns: Iterable[str],
    numerical_columns: Iterable[str],
    binary_columns: Iterable[str],
    source_report: str = "",
) -> FeatureSchema:
    """Assemble and validate a :class:`FeatureSchema` from observed columns.

    ``observed_columns`` must list every column present in the training data.
    Every observed column must be assigned exactly one role; anything
    unassigned raises :class:`FeatureSchemaError` instead of being dropped.
    """
    observed = sort_columns_like_raw(observed_columns)
    categorical = sort_columns_like_raw(categorical_columns)
    numerical = sort_columns_like_raw(numerical_columns)
    binary = sort_columns_like_raw(binary_columns)

    unknown = [c for c in categorical + numerical + binary if c not in observed]
    if unknown:
        raise FeatureSchemaError(f"Features not present in raw data: {unknown}")

    schema = FeatureSchema(
        id_columns=sort_columns_like_raw(id_columns),
        target_columns=list(target_columns),
        categorical_columns=categorical,
        numerical_columns=numerical,
        binary_columns=binary,
        raw_column_order=observed,
        source_report=source_report,
    )
    schema.validate()
    schema.schema_hash = _schema_hash(schema)
    return schema


def _schema_hash(schema: FeatureSchema) -> str:
    import hashlib

    payload = json.dumps(schema.to_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def infer_roles_from_dataframe(
    df: pd.DataFrame,
    *,
    id_columns: Sequence[str],
    target_columns: Sequence[str],
    max_categorical_cardinality: int,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """Classify columns of a (training) dataframe into schema roles.

    Returns ``(id, target, categorical, numerical, binary)`` lists in raw
    column order. Rules:

    * id/target columns are taken verbatim;
    * binary: observed values (excluding NaN) are a subset of {0, 1};
    * categorical: integer dtype, no NaN, distinct count <= cap, and not
      already binary;
    * numerical: everything else (float dtype, or integral with NaN).

    The maximum observed distinct count is returned to the caller via the
    returned ``categorical`` list; cardinality per column is available from
    the data itself, so this function never silently hashes anything.
    """
    role_map: dict[str, str] = {}
    for c in id_columns:
        role_map[c] = "id"
    for c in target_columns:
        role_map[c] = "target"

    categorical: list[str] = []
    numerical: list[str] = []
    binary: list[str] = []

    for column in df.columns:
        role = role_map.get(column)
        if role in ("id", "target"):
            continue
        series = df[column]
        present = series.dropna()
        distinct = int(present.nunique())
        is_binary = bool(distinct > 0) and set(present.unique().tolist()).issubset({0, 1})
        if is_binary:
            binary.append(column)
            continue
        if pd.api.types.is_integer_dtype(series) and series.isna().sum() == 0:
            if distinct <= max_categorical_cardinality:
                categorical.append(column)
                continue
            # Too large for an explicit vocabulary: leave unassigned so the
            # caller must decide explicitly (never silently hashed).
            raise FeatureSchemaError(
                f"Column {column!r} is integer-like with {distinct} distinct "
                f"values > max_categorical_cardinality={max_categorical_cardinality}. "
                "Assign its role explicitly (categorical/hashing/numerical) "
                "instead of guessing."
            )
        numerical.append(column)

    return (
        sort_columns_like_raw(id_columns),
        list(target_columns),
        sort_columns_like_raw(categorical),
        sort_columns_like_raw(numerical),
        sort_columns_like_raw(binary),
    )


def load_stage1_report_columns(report_path: str | Path) -> dict[str, Any]:
    """Load the Stage 1 machine-readable report (reports/dataset_report.json)."""
    with open(report_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def schema_from_stage1_report(
    report: Mapping[str, Any],
    config: PreprocessingConfig,
) -> FeatureSchema:
    """Build the feature schema directly from the Stage 1 report.

    This is the primary, report-driven path used by the CLI. Column roles
    follow the rules documented at module level; cardinalities observed in
    the report are respected (f_0 excluded as ID; nothing hashed).
    """
    train_columns = list(report["schema"]["train_columns"])
    report_targets = list(report["targets"]["columns"])  # main target first
    # The Stage 1 report lists targets in priority order (main, auxiliary);
    # the config (and this schema) keeps raw-file order. Compare as sets so
    # the two documented conventions cannot silently diverge in content.
    if set(report_targets) != set(config.target_columns):
        raise FeatureSchemaError(
            f"Stage 1 targets {report_targets} do not match configured targets "
            f"{list(config.target_columns)}"
        )
    targets = list(config.target_columns)  # raw-file order: is_clicked, is_installed

    per_column = report["columns"]["train"]
    id_columns = list(report["ids"]["id_like_columns"])
    if id_columns != list(config.id_columns):
        raise FeatureSchemaError(
            f"Stage 1 IDs {id_columns} do not match configured IDs "
            f"{list(config.id_columns)}"
        )

    categorical: list[str] = []
    numerical: list[str] = []
    binary: list[str] = []

    for column in train_columns:
        if column in id_columns or column in targets:
            continue
        stats = per_column[column]
        distinct = int(stats.get("distinct_global", 0))
        dtype = str(stats.get("dtype_mode", ""))
        # Stage 1 'binary' flag is cardinality-based. Resolve to the {0,1}-set
        # rule: binary iff the observed label set is a subset of {0, 1}.
        # Truncated label counts (Stage 1 stores counts only when distinct
        # <= max_label_values=64) can never describe a binary column, so a
        # truncated column safely falls through to dtype-based classification.
        labels = stats.get("label_counts_union") or {}
        if labels and set(labels).issubset({"0", "1"}):
            binary.append(column)
            continue
        if dtype == "int64":
            if distinct > config.max_categorical_cardinality:
                raise FeatureSchemaError(
                    f"Column {column!r} has {distinct} distinct values which "
                    "exceeds the vocabulary cap; decide explicitly how to "
                    "handle it (hashing / numerical) before proceeding."
                )
            categorical.append(column)
            continue
        numerical.append(column)

    schema = build_feature_schema(
        id_columns=id_columns,
        target_columns=targets,
        observed_columns=train_columns,
        categorical_columns=categorical,
        numerical_columns=numerical,
        binary_columns=binary,
        source_report=str(config.stage1_report_path),
    )
    return schema
