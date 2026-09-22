"""Synthetic mini-dataset fixture for preprocessing unit tests.

Builds a TAB-separated train/test pair whose columns mirror the Stage 1
schema (f_0 ID, f_1..f_79 features with representative dtypes, plus both
targets), so all preprocessing logic is exercised on small, controlled data
without touching the real 3.5M-row dataset.

Deterministic under a fixed seed; the raw CSVs are written into a tmp_path
directory that the caller manages (raw files are never modified afterwards).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42

CATEGORICAL_FIXTURE_COLUMNS = ["f_1", "f_2", "f_26", "f_29", "f_32"]
NUMERICAL_FIXTURE_COLUMNS = [
    "f_30", "f_31", "f_42", "f_43", "f_51",
    "f_64", "f_71", "f_79",
]
BINARY_FIXTURE_COLUMNS = ["f_33", "f_34", "f_35", "f_36", "f_37", "f_38"]
ID_FIXTURE_COLUMNS = ["f_0"]
TARGET_FIXTURE_COLUMNS = ["is_clicked", "is_installed"]

FEATURE_COLUMNS = (
    CATEGORICAL_FIXTURE_COLUMNS
    + NUMERICAL_FIXTURE_COLUMNS
    + BINARY_FIXTURE_COLUMNS
)


def build_train_frame(n_rows: int = 500, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}

    data["f_0"] = np.arange(1, n_rows + 1)

    # Categoricals: small vocabularies, including a zero-variance column.
    data["f_1"] = rng.integers(45, 67, n_rows)          # 22 categories
    data["f_2"] = rng.integers(0, 10, n_rows)           # 10 categories
    data["f_26"] = np.where(rng.random(n_rows) < 0.5, 0, 14897)  # two raw IDs
    data["f_29"] = np.where(rng.random(n_rows) < 0.3, 0, 6697)
    data["f_32"] = np.full(n_rows, 2, dtype=np.int64)   # zero variance

    # Numericals: mixed scales, missingness, zero-variance, non-integral.
    data["f_30"] = np.where(rng.random(n_rows) < 0.48, np.nan, rng.integers(0, 2, n_rows).astype(float))
    data["f_31"] = np.where(rng.random(n_rows) < 0.48, np.nan, rng.integers(0, 2, n_rows).astype(float))
    data["f_42"] = rng.gamma(2.0, 9.0, n_rows)
    data["f_43"] = np.where(rng.random(n_rows) < 0.05, np.nan, rng.gamma(1.5, 1.0, n_rows))
    data["f_51"] = np.where(rng.random(n_rows) < 0.05, np.nan, rng.exponential(9.0, n_rows))
    data["f_64"] = np.where(rng.random(n_rows) < 0.05, np.nan, rng.integers(0, 1_000_000, n_rows).astype(float))
    data["f_71"] = np.full(n_rows, 0.0427, dtype=float)  # zero variance float
    data["f_79"] = rng.gamma(1.0, 0.35, n_rows)

    # Binaries {0,1} stored as int64 (matches the real dataset's f_33..f_41).
    for name, p in [
        ("f_33", 0.075),
        ("f_34", 0.525),
        ("f_35", 0.254),
        ("f_36", 0.611),
        ("f_37", 0.932),
        ("f_38", 0.878),
    ]:
        data[name] = (rng.random(n_rows) < p).astype(np.int64)

    frame = pd.DataFrame(data)
    # Both labels derived from features but keep both classes present.
    logits = (
        0.5 * frame["f_42"] / (frame["f_42"].mean() + 1e-9)
        + frame["f_34"]
        - 0.5 * frame["f_36"]
    )
    prob_click = 1 / (1 + np.exp(-logits / 2))
    frame["is_clicked"] = (rng.random(n_rows) < prob_click).astype(np.int64)
    frame["is_installed"] = (rng.random(n_rows) < prob_click * 0.8).astype(np.int64)
    return frame


def build_test_frame(n_rows: int = 120, seed: int = SEED + 1) -> pd.DataFrame:
    """Test frame: train-minus-targets schema plus UNK/exotic values."""
    rng = np.random.default_rng(seed)
    train = build_train_frame(n_rows + 40, seed=seed + 100)
    frame = train.drop(columns=TARGET_FIXTURE_COLUMNS).tail(n_rows).reset_index(drop=True)
    # Inject unseen category values (must map to UNK, not extend vocabulary).
    frame.loc[0, "f_1"] = 999
    frame.loc[1, "f_2"] = 777
    # Inject NaN into a numerical column for imputation checks.
    frame.loc[2, "f_42"] = np.nan
    return frame


def write_fixture_files(
    directory,
    n_train_rows: int = 500,
    n_test_rows: int = 120,
    seed: int = SEED,
    delimiter: str = "\t",
) -> dict:
    """Write train/test TSV files + a synthetic Stage 1 report.

    Returns metadata dict (paths, expected values) for assertions.
    """
    directory = Path(directory)
    train_dir = directory / "train"
    test_dir = directory / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = directory / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df = build_train_frame(n_train_rows, seed)
    test_df = build_test_frame(n_test_rows, seed + 1)

    train_path = train_dir / "000000000000.csv"
    test_path = test_dir / "000000000000.csv"
    train_df.to_csv(train_path, sep=delimiter, index=False)
    test_df.to_csv(test_path, sep=delimiter, index=False)

    # Build a synthetic Stage 1 report describing the fixture.
    per_column = {}
    all_columns = list(train_df.columns)

    def label_key(value) -> str:
        """Mirror Stage 1 label normalization: integral values lose '.0'."""
        try:
            as_float = float(value)
            if as_float.is_integer():
                return str(int(as_float))
        except (TypeError, ValueError):
            pass
        return str(value)

    for column in all_columns:
        series = train_df[column]
        present = series.dropna()
        labels = {label_key(v): int(c) for v, c in present.value_counts().items()}
        distinct = int(present.nunique())
        per_column[column] = {
            "dtype_mode": "int64" if pd.api.types.is_integer_dtype(series) else "float64",
            "distinct_global": distinct,
            "missing": int(series.isna().sum()),
            "missing_ratio": float(series.isna().mean()),
            "label_counts_union": labels if distinct <= 64 else {},
            "labels_truncated": distinct > 64,
            "binary": distinct == 2,
        }

    report = {
        "targets": {
            "main": "is_installed",
            "auxiliary": "is_clicked",
            "columns": ["is_installed", "is_clicked"],
        },
        "ids": {"id_like_columns": ["f_0"]},
        "schema": {"train_columns": all_columns},
        "columns": {"train": per_column},
    }
    report_path = reports_dir / "dataset_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    return {
        "root": directory,
        "train_path": train_path,
        "test_path": test_path,
        "report_path": report_path,
        "train_frame": train_df,
        "test_frame": test_df,
    }


# Imported at the bottom so the module docstring stays on top.
import json  # noqa: E402
from pathlib import Path  # noqa: E402
