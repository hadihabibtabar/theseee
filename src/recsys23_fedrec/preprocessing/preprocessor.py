"""The central, reusable preprocessing object.

Lifecycle::

    preprocessor = Preprocessor(config)
    preprocessor.fit()                        # training data only, streaming
    preprocessor.transform(chunk, split=...)  # any split, pure function of state
    preprocessor.transform_split_to_parquet(store, split)  # streaming writes
    preprocessor.save(output_dir)
    restored = Preprocessor.load(output_dir)

Leakage-safety invariants:

* ``fit`` observes training shards only; test shards are never opened.
* Vocabularies, statistics, imputation constants, and the missing-indicator
  list are computed exclusively from training chunks.
* ``transform`` is a pure function of fitted state: it never updates any
  statistic or vocabulary, so train / validation / test / federated-client
  processing behave identically.
* Fitting never holds per-row data in memory: only per-column accumulators
  (distinct-value counts, streaming mean/variance, missing counts).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa

from .config import PreprocessingConfig
from .encoders import (
    CategoricalVocabulary,
    NumericalStats,
    build_missing_indicators,
    fit_categorical_vocabulary,
    missing_indicator_name,
    validate_binary_column,
)
from .exceptions import ArtifactError, DataContractError, NotFittedError
from .feature_schema import FeatureSchema, schema_from_stage1_report
from .raw_data import RawShardReader
from .storage import PreprocessingArtifactStore


class _StreamingStats:
    """Streaming mean/variance accumulator for one numerical column.

    Uses the parallel (Chan et al.) merge of per-chunk statistics, so the
    result equals the single-pass population mean/std over all non-missing
    training values while memory stays bounded to one chunk.
    """

    __slots__ = ("n", "mean", "m2", "missing", "total")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.missing = 0
        self.total = 0

    def update(self, values: np.ndarray) -> None:
        present = values[~np.isnan(values)]
        n_b = int(present.size)
        self.total += int(values.size)
        self.missing += int(values.size - n_b)
        if n_b == 0:
            return
        mean_b = float(present.mean())
        m2_b = float(((present - mean_b) ** 2).sum())
        if self.n == 0:
            self.n = n_b
            self.mean = mean_b
            self.m2 = m2_b
            return
        delta = mean_b - self.mean
        total = self.n + n_b
        self.mean += delta * n_b / total
        self.m2 += m2_b + delta * delta * self.n * n_b / total
        self.n = total

    @property
    def std(self) -> float:
        if self.n == 0:
            return 0.0
        return float(np.sqrt(self.m2 / self.n))


def _hash_key(value) -> object:
    """Stable dict key for a category value (keeps int/float/str apart)."""
    if isinstance(value, (bool, np.bool_)):
        return ("b", bool(value))
    if isinstance(value, (int, np.integer)):
        return ("i", int(value))
    if isinstance(value, float) and not np.isnan(value):
        return ("f", value)
    if isinstance(value, str):
        return ("s", value)
    return ("o", value)


def _feature_sort_key(name: str) -> tuple[int, int, str]:
    """Sort helper reproducing raw column order (f_2 < f_10 < f_79)."""
    if name.startswith("f_"):
        try:
            return (0, int(name.split("_", 1)[1]), name)
        except ValueError:
            return (2, 0, name)
    return (2, 0, name)


class Preprocessor:
    """Fit-once, transform-anywhere preprocessing for the RecSys23 dataset."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()
        self.schema: FeatureSchema | None = None
        self.vocabularies: dict[str, CategoricalVocabulary] = {}
        self.numerical_stats: dict[str, NumericalStats] = {}
        self.missing_indicator_columns: list[str] = []
        self.label_counts: dict[str, dict[int, int]] = {}
        self.fitted = False
        self.artifact_hash: str = ""
        self.source_row_counts: dict[str, int] = {}
        self._binary_imputation: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, report: Mapping[str, Any] | None = None) -> "Preprocessor":
        """Fit all preprocessing state from training data only (streaming).

        ``report`` is the Stage 1 machine-readable report; when omitted it
        is loaded from ``config.stage1_report_path``. The report drives the
        feature classification (source of truth).
        """
        if report is None:
            from .feature_schema import load_stage1_report_columns

            report = load_stage1_report_columns(self.config.stage1_report_path)
        schema = schema_from_stage1_report(report, self.config)
        reader = self._make_reader(schema)
        reader.validate_train_columns()
        self.fit_from_chunks(reader.iter_train_chunks(), schema)
        return self

    def fit_from_chunks(
        self,
        chunks: Iterable[pd.DataFrame],
        schema: FeatureSchema,
    ) -> "Preprocessor":
        """Fit from an iterable of training chunks (streaming, bounded RAM).

        Public so synthetic fixtures and federated-client code can fit from
        any chunk source; state still originates exclusively from the given
        (training) chunks.
        """
        counts: dict[str, dict[object, int]] = {
            c: {} for c in schema.categorical_columns
        }
        num_acc = {c: _StreamingStats() for c in schema.numerical_columns}
        binary_missing = {c: 0 for c in schema.binary_columns}
        binary_ones = {c: 0 for c in schema.binary_columns}
        binary_bad: dict[str, set] = {c: set() for c in schema.binary_columns}
        label_counts: dict[str, dict[int, int]] = {
            c: {0: 0, 1: 0} for c in schema.target_columns
        }
        total_rows = 0

        for chunk in chunks:
            total_rows += len(chunk)
            self._validate_chunk_columns(chunk, schema, require_targets=True)
            for column in schema.categorical_columns:
                self._accumulate_counts(
                    column, chunk[column], counts[column]
                )
            for column in schema.numerical_columns:
                num_acc[column].update(chunk[column].to_numpy(dtype=np.float64))
            for column in schema.binary_columns:
                self._accumulate_binary(
                    column, chunk[column], binary_missing, binary_ones, binary_bad
                )
            for column in schema.target_columns:
                self._accumulate_labels(column, chunk[column], label_counts[column])

        for column, bad in binary_bad.items():
            if bad:
                raise DataContractError(
                    f"Binary feature {column!r} observed non-binary training "
                    f"values: {sorted(bad)[:5]}"
                )

        self.schema = schema
        self.vocabularies = {}
        for column in schema.categorical_columns:
            value_counts = _untagged_counts_to_series(counts[column])
            self.vocabularies[column] = fit_categorical_vocabulary(
                column,
                value_counts,
                max_cardinality=self.config.max_categorical_cardinality,
            )
        self.numerical_stats = {
            column: NumericalStats(
                feature=column,
                mean=acc.mean,
                std=acc.std,
                missing_count=acc.missing,
                total_count=acc.total,
            )
            for column, acc in num_acc.items()
        }
        for column, stats in self.numerical_stats.items():
            if stats.total_count != total_rows:
                raise DataContractError(
                    f"Numerical column {column!r} accumulated {stats.total_count} "
                    f"values but the fitting pass saw {total_rows} rows"
                )
        self.missing_indicator_columns = sorted(
            [c for c, s in self.numerical_stats.items() if s.missing_count > 0]
            + [c for c, n in binary_missing.items() if n > 0],
            key=_feature_sort_key,
        )
        self._binary_imputation = {}
        for column in schema.binary_columns:
            present_count = total_rows - binary_missing[column]
            if binary_missing[column] == 0 or present_count == 0:
                self._binary_imputation[column] = 0.0
            else:
                self._binary_imputation[column] = float(
                    round(binary_ones[column] / present_count)
                )
        self.label_counts = label_counts
        self.source_row_counts = {"train_rows": total_rows}
        self.fitted = True
        self.artifact_hash = self._compute_artifact_hash()
        return self

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------
    def transform(self, chunk: pd.DataFrame, *, split: str = "test") -> pd.DataFrame:
        """Transform one chunk into model-ready form (pure, state-preserving).

        ``split="train"`` requires and validates both label columns;
        ``split="test"`` or ``"validation"`` must not contain them. The
        output schema is identical across splits except for the label
        columns, which only training splits carry.
        """
        self._ensure_fitted()
        schema = self._require_schema()
        require_targets = split == "train"
        if not require_targets:
            unexpected = set(schema.target_columns) & set(chunk.columns)
            if unexpected:
                raise DataContractError(
                    f"Split {split!r} must not contain target columns; found "
                    f"{sorted(unexpected)}"
                )
        self._validate_chunk_columns(chunk, schema, require_targets=require_targets)

        out = pd.DataFrame(index=range(len(chunk)))

        for column in schema.categorical_columns:
            out[column] = self.vocabularies[column].transform(chunk[column])

        for column in schema.binary_columns:
            values = chunk[column].to_numpy(dtype=np.float64, copy=True)
            present = ~np.isnan(values)
            bad = values[present][~np.isin(values[present], [0.0, 1.0])]
            if bad.size > 0:
                raise DataContractError(
                    f"Binary feature {column!r} contains non-binary values at "
                    f"transform time, e.g. {np.unique(bad)[:5].tolist()}"
                )
            values = np.where(present, values, self._binary_imputation[column])
            out[column] = values.astype(np.float32)

        for column in schema.numerical_columns:
            out[column] = self.numerical_stats[column].transform(chunk[column])

        for column in self.missing_indicator_columns:
            indicator = chunk[column].isna().to_numpy(dtype=np.float32)
            out[missing_indicator_name(column)] = indicator

        if require_targets:
            for column in schema.target_columns:
                out[column] = self._validated_labels(chunk[column], column)

        return out

    def iter_transformed_train_chunks(self) -> Iterator[pd.DataFrame]:
        """Stream training shards through ``transform`` in bounded memory."""
        self._ensure_fitted()
        reader = self._make_reader(self._require_schema())
        for chunk in reader.iter_train_chunks():
            yield self.transform(chunk, split="train")

    def iter_transformed_test_chunks(self) -> Iterator[pd.DataFrame]:
        """Stream test shards through ``transform`` in bounded memory."""
        self._ensure_fitted()
        reader = self._make_reader(self._require_schema())
        for chunk in reader.iter_test_chunks():
            yield self.transform(chunk, split="test")

    def transform_split_to_parquet(
        self,
        store: PreprocessingArtifactStore,
        split: str,
    ) -> int:
        """Stream-transform a split and write chunked Parquet parts.

        Returns the number of processed rows. Each part file carries the
        artifact hash in its metadata for provenance linkage.
        """
        self._ensure_fitted()
        include_targets = split == "train"
        iterator = (
            self.iter_transformed_train_chunks()
            if include_targets
            else self.iter_transformed_test_chunks()
        )
        writer = store.open_writer(
            split,
            self.arrow_schema(include_targets=include_targets),
            self.artifact_hash,
        )
        rows = 0
        for transformed in iterator:
            writer.write_chunk(transformed)
            rows += len(transformed)
        return rows

    def fit_transform(
        self,
        store: PreprocessingArtifactStore | None = None,
        report: Mapping[str, Any] | None = None,
    ) -> "Preprocessor":
        """Fit on training data, then (optionally) write the training split."""
        self.fit(report=report)
        if store is not None:
            self.transform_split_to_parquet(store, "train")
        return self

    # ------------------------------------------------------------------
    # Schema / serialization
    # ------------------------------------------------------------------
    def arrow_schema(self, *, include_targets: bool) -> pa.Schema:
        """PyArrow schema of the processed output (deterministic order).

        Test/validation splits exclude the label columns; only training
        splits carry ``is_clicked``/``is_installed``.
        """
        schema = self._require_schema()
        fields: list[pa.Field] = []
        for column in schema.categorical_columns:
            fields.append(pa.field(column, pa.int32()))
        for column in schema.binary_columns:
            fields.append(pa.field(column, pa.float32()))
        for column in schema.numerical_columns:
            fields.append(pa.field(column, pa.float32()))
        for column in self.missing_indicator_columns:
            fields.append(pa.field(missing_indicator_name(column), pa.float32()))
        if include_targets:
            for column in schema.target_columns:
                fields.append(pa.field(column, pa.int64()))
        return pa.schema(fields)

    def processed_feature_columns(self) -> list[str]:
        """Deterministic processed column order (labels excluded)."""
        schema = self._require_schema()
        return (
            list(schema.categorical_columns)
            + list(schema.binary_columns)
            + list(schema.numerical_columns)
            + [missing_indicator_name(c) for c in self.missing_indicator_columns]
        )

    def vocabulary_sizes_by_order(self) -> list[int]:
        """Vocabulary sizes in processed categorical column order.

        Index j corresponds to the j-th categorical tensor column, so the
        data layer can validate ID ranges without name lookups.
        """
        schema = self._require_schema()
        return [self.vocabularies[c].vocab_size for c in schema.categorical_columns]

    def save(self, output_dir: str) -> None:
        """Persist the fitted artifact (schema, stats, vocabularies, config)."""
        self._ensure_fitted()
        store = PreprocessingArtifactStore(output_dir)
        store.write_json(store.feature_schema_path(), self.schema.to_dict())
        store.write_json(
            store.numerical_stats_path(),
            {c: s.to_dict() for c, s in self.numerical_stats.items()},
        )
        store.write_json(
            store.vocabularies_path(),
            {c: v.to_dict() for c, v in self.vocabularies.items()},
        )
        store.write_json(
            store.config_path(),
            {
                "config": self.config.to_dict(),
                "missing_indicator_columns": self.missing_indicator_columns,
                "source_row_counts": self.source_row_counts,
                "artifact_hash": self.artifact_hash,
                "conventions": self.conventions_documentation(),
            },
        )
        store.write_json(store.artifact_path(), self.artifact_manifest())

    @classmethod
    def load(cls, output_dir: str) -> "Preprocessor":
        """Restore a fitted artifact; transform-only usage stays consistent."""
        store = PreprocessingArtifactStore(output_dir)
        manifest = store.read_json(store.artifact_path())
        config = PreprocessingConfig.from_dict(manifest["config"])
        preprocessor = cls(config)
        preprocessor.schema = FeatureSchema.from_dict(
            store.read_json(store.feature_schema_path())
        )
        preprocessor.vocabularies = {
            c: CategoricalVocabulary.from_dict(v)
            for c, v in store.read_json(store.vocabularies_path()).items()
        }
        preprocessor.numerical_stats = {
            c: NumericalStats.from_dict(s)
            for c, s in store.read_json(store.numerical_stats_path()).items()
        }
        config_payload = store.read_json(store.config_path())
        preprocessor.missing_indicator_columns = list(
            config_payload["missing_indicator_columns"]
        )
        preprocessor.source_row_counts = dict(
            config_payload.get("source_row_counts", {})
        )
        preprocessor._binary_imputation = {
            k: float(v) for k, v in manifest["binary_imputation"].items()
        }
        preprocessor.artifact_hash = str(manifest["artifact_hash"])
        preprocessor.fitted = True

        # Cross-validate derived manifest fields against the restored state.
        expected_sizes = {c: v.vocab_size for c, v in preprocessor.vocabularies.items()}
        if manifest.get("vocabulary_sizes") != expected_sizes:
            raise ArtifactError(
                "Preprocessing artifact inconsistent: manifest vocabulary_sizes "
                f"{manifest.get('vocabulary_sizes')} does not match restored "
                f"vocabularies {expected_sizes}"
            )
        if list(manifest.get("missing_indicator_columns", [])) != list(
            preprocessor.missing_indicator_columns
        ):
            raise ArtifactError(
                "Preprocessing artifact inconsistent: missing-indicator list "
                "differs between artifact.json and preprocessing_config.json"
            )
        schema = preprocessor.schema
        if manifest.get("schema_hash") != schema.schema_hash:
            raise ArtifactError(
                "Preprocessing artifact inconsistent: schema_hash differs "
                "between artifact.json and feature_schema.json"
            )
        for key, names in (
            ("categorical_features", schema.categorical_columns),
            ("numerical_features", schema.numerical_columns),
            ("binary_features", schema.binary_columns),
            ("target_columns", schema.target_columns),
            ("id_columns", schema.id_columns),
        ):
            if list(manifest.get(key, [])) != list(names):
                raise ArtifactError(
                    f"Preprocessing artifact inconsistent: manifest field "
                    f"{key!r} does not match feature_schema.json"
                )

        # Content hash over everything transform depends on.
        recomputed = preprocessor._compute_artifact_hash()
        if recomputed != preprocessor.artifact_hash:
            raise ArtifactError(
                "Preprocessing artifact hash mismatch: "
                f"stored={preprocessor.artifact_hash} recomputed={recomputed}"
            )
        return preprocessor

    def artifact_manifest(self) -> dict[str, Any]:
        """Linkage manifest stored next to the processed data."""
        schema = self._require_schema()
        return {
            "artifact_hash": self.artifact_hash,
            "schema_hash": schema.schema_hash,
            "config": self.config.to_dict(),
            "missing_indicator_columns": self.missing_indicator_columns,
            "binary_imputation": dict(self._binary_imputation),
            "vocabulary_sizes": {c: v.vocab_size for c, v in self.vocabularies.items()},
            "categorical_features": list(schema.categorical_columns),
            "numerical_features": list(schema.numerical_columns),
            "binary_features": list(schema.binary_columns),
            "target_columns": list(schema.target_columns),
            "id_columns": list(schema.id_columns),
            "label_counts": self.label_counts,
            "source_row_counts": self.source_row_counts,
            "processed_feature_columns": self.processed_feature_columns(),
        }

    def conventions_documentation(self) -> dict[str, str]:
        """Human-readable documentation of the encoding conventions."""
        return {
            "categorical_ids": (
                "0=UNK (unseen), 1=MISSING, 2+=known categories ordered by "
                "descending train frequency then ascending value"
            ),
            "numerical": (
                "(x - train_mean) / train_std; missing imputed with train_mean "
                "(normalized to 0.0); zero-variance columns map to 0.0"
            ),
            "binary": (
                "values restricted to {0,1}; missing imputed with the rounded "
                "training mean; no vocabulary created"
            ),
            "missing_indicators": (
                "one 0/1 float column <feature>_missing per feature with "
                f"training missingness: {self.missing_indicator_columns}"
            ),
            "leakage": (
                "all fitted state originates from training shards only; "
                "transform never updates state"
            ),
        }

    def summary(self) -> dict[str, Any]:
        """Concise summary for CLI reporting."""
        schema = self._require_schema()
        return {
            "id_columns": schema.id_columns,
            "target_columns": schema.target_columns,
            "categorical_columns": schema.categorical_columns,
            "numerical_columns": schema.numerical_columns,
            "binary_columns": schema.binary_columns,
            "missing_indicator_columns": self.missing_indicator_columns,
            "vocabulary_sizes": {c: v.vocab_size for c, v in self.vocabularies.items()},
            "label_counts": self.label_counts,
            "source_row_counts": self.source_row_counts,
            "artifact_hash": self.artifact_hash,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_reader(self, schema: FeatureSchema) -> RawShardReader:
        return RawShardReader(
            self.config,
            id_columns=schema.id_columns,
            categorical_columns=schema.categorical_columns,
            numerical_columns=schema.numerical_columns,
            binary_columns=schema.binary_columns,
            target_columns=schema.target_columns,
        )

    def _validate_chunk_columns(
        self,
        chunk: pd.DataFrame,
        schema: FeatureSchema,
        *,
        require_targets: bool,
    ) -> None:
        required = set(schema.model_feature_columns)
        if require_targets:
            required |= set(schema.target_columns)
        missing = required - set(chunk.columns)
        if missing:
            raise DataContractError(
                f"Chunk is missing required columns: {sorted(missing)}"
            )

    @staticmethod
    def _accumulate_counts(
        column: str,
        series: pd.Series,
        counts: dict[object, int],
    ) -> None:
        for value, count in series.value_counts(dropna=True).items():
            key = _hash_key(value)
            counts[key] = counts.get(key, 0) + int(count)

    @staticmethod
    def _accumulate_binary(
        column: str,
        series: pd.Series,
        missing: dict[str, int],
        ones: dict[str, int],
        bad: dict[str, set],
    ) -> None:
        values = series.to_numpy(dtype=np.float64)
        present = ~np.isnan(values)
        # `present` is a boolean mask over `values`: count the False entries.
        missing[column] += int((~present).sum())
        ones[column] += int((values[present] == 1.0).sum())
        offending = values[present][~np.isin(values[present], [0.0, 1.0])]
        if offending.size > 0:
            bad[column].update(np.unique(offending).tolist())
            validate_binary_column(column, series)

    @staticmethod
    def _accumulate_labels(
        column: str,
        series: pd.Series,
        counts: dict[int, int],
    ) -> None:
        values = series.to_numpy(dtype=np.float64)
        if np.isnan(values).any():
            raise DataContractError(
                f"Target column {column!r} contains missing labels in the "
                "training data; training preprocessing requires complete labels."
            )
        if not np.isin(values, [0.0, 1.0]).all():
            offending = np.unique(values[~np.isin(values, [0.0, 1.0])])[:5]
            raise DataContractError(
                f"Target column {column!r} contains unexpected label values "
                f"{offending.tolist()}; labels must be exactly 0 or 1."
            )
        counts[0] += int((values == 0.0).sum())
        counts[1] += int((values == 1.0).sum())

    @staticmethod
    def _validated_labels(series: pd.Series, column: str) -> np.ndarray:
        values = series.to_numpy(dtype=np.float64)
        if np.isnan(values).any():
            raise DataContractError(
                f"Target column {column!r} contains missing labels; training "
                "preprocessing requires complete labels."
            )
        if not np.isin(values, [0.0, 1.0]).all():
            offending = np.unique(values[~np.isin(values, [0.0, 1.0])])[:5]
            raise DataContractError(
                f"Target column {column!r} contains unexpected label values "
                f"{offending.tolist()}; labels must be exactly 0 or 1."
            )
        return values.astype(np.int64)

    def _ensure_fitted(self) -> None:
        if not self.fitted:
            raise NotFittedError(
                "Preprocessor is not fitted; call fit() (or load an artifact) "
                "before transforming data."
            )

    def _require_schema(self) -> FeatureSchema:
        if self.schema is None:
            raise NotFittedError("Preprocessor has no fitted feature schema.")
        return self.schema

    def _compute_artifact_hash(self) -> str:
        payload = json.dumps(
            {
                "schema": self.schema.to_dict(),
                "vocabularies": {c: v.to_dict() for c, v in self.vocabularies.items()},
                "numerical_stats": {
                    c: s.to_dict() for c, s in self.numerical_stats.items()
                },
                "missing_indicator_columns": self.missing_indicator_columns,
                "binary_imputation": dict(self._binary_imputation),
                "config": self.config.to_dict(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _untagged_counts_to_series(counts: dict[object, int]) -> pd.Series:
    """Convert hash-keyed counts back to a value->count Series."""
    items = sorted(counts.items(), key=lambda kv: repr(kv[0]))
    values = [_untag(key) for key, _ in items]
    frequency = [count for _, count in items]
    return pd.Series(frequency, index=values, dtype="int64")


def _untag(key: object):
    kind, value = key  # type: ignore[misc]
    return value
