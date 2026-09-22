"""Integrity verification for a preprocessing output directory (Stage 2).

Usage::

    python scripts/verify_preprocessing.py --output_dir artifacts
    python scripts/verify_preprocessing.py --output_dir artifacts_smoke --smoke

Checks (all read-only; raw files are never touched):

1. Preprocessing artifact files exist and the artifact hash verifies.
2. Processed Parquet parts reopen successfully.
3. Processed train rows == raw train rows; test likewise (full mode).
4. No rows silently disappear / appear (per-part counts sum exactly).
5. ``f_0`` is absent from processed data; all expected feature columns
   (and no extras) are present.
6. Train/test processed schemas are identical apart from the two labels.
7. Categorical IDs are within ``[0, vocab_size)`` per fitted vocabulary.
8. Numerical/binary/indicator values are finite (no NaN/Inf).
9. Binary values are in {0, 1}.
10. Labels are exactly 0/1 and their totals match the artifact manifest.
11. Every part's key-value metadata links back to the artifact hash.
12. Prints one raw row next to its processed representation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from recsys23_fedrec.preprocessing import (  # noqa: E402
    Preprocessor,
    PreprocessingArtifactStore,
    count_raw_rows,
    UNK_ID,
    MISSING_ID,
)

SCHEMA_HASH_KEY = "recsys23.schema_hash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="artifacts")
    parser.add_argument("--data_dir", default=".")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Skip raw-row-count equality (smoke artifacts cover data prefixes)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    store = PreprocessingArtifactStore(out_dir)
    failures: list[str] = []

    # 1. Artifact presence + hash verification (raises on tampering).
    for name in (
        "feature_schema.json",
        "numerical_stats.json",
        "categorical_vocabularies.json",
        "preprocessing_config.json",
        "artifact.json",
    ):
        if not (out_dir / "preprocessing" / name).is_file():
            failures.append(f"missing artifact file: preprocessing/{name}")
    if failures:
        return _report(failures)
    preprocessor = Preprocessor.load(str(out_dir))  # hash-verified
    manifest = store.read_json(store.artifact_path())
    print(f"[ok] artifact files present; hash {preprocessor.artifact_hash} verifies")

    schema = preprocessor.schema
    vocab_sizes = {c: v.vocab_size for c, v in preprocessor.vocabularies.items()}
    expected_features = preprocessor.processed_feature_columns()
    targets = schema.target_columns

    # 2-5. Per-split part iteration with bounded memory.
    split_rows: dict[str, int] = {}
    split_train_frame_columns: list[str] | None = None
    for split in ("train", "test"):
        parts = store.part_files(split)
        total = 0
        include_targets = split == "train"
        for part in parts:
            meta = pq.read_metadata(part)
            total += meta.num_rows
            # 11. Linkage metadata on every part.
            kv = {
                k.decode("utf-8"): v.decode("utf-8")
                for k, v in (meta.metadata or {}).items()
            }
            if kv.get(SCHEMA_HASH_KEY) != preprocessor.artifact_hash:
                failures.append(f"{split}/{part.name}: artifact-hash metadata mismatch")
            table = pq.read_table(part)  # 2. reopen check
            frame = table.to_pandas()
            # 5. Column presence.
            expected = expected_features + (targets if include_targets else [])
            if list(frame.columns) != expected:
                missing = set(expected) - set(frame.columns)
                extra = set(frame.columns) - set(expected)
                failures.append(
                    f"{split}/{part.name}: column mismatch "
                    f"(missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]})"
                )
            if "f_0" in frame.columns:
                failures.append(f"{split}/{part.name}: f_0 present in processed data")
            # 7. Categorical ID ranges.
            for column in schema.categorical_columns:
                ids = frame[column].to_numpy()
                if ids.min() < UNK_ID or ids.max() >= vocab_sizes[column]:
                    failures.append(
                        f"{split}/{part.name}: {column} IDs outside "
                        f"[0, {vocab_sizes[column]})"
                    )
            # 8. Finiteness of all model-ready numeric columns.
            numeric = frame[expected_features].to_numpy(dtype=np.float64)
            if not np.isfinite(numeric).all():
                failures.append(f"{split}/{part.name}: non-finite values in features")
            # 9. Binary values.
            for column in schema.binary_columns:
                values = frame[column].to_numpy()
                if not np.isin(values, [0.0, 1.0]).all():
                    failures.append(f"{split}/{part.name}: {column} non-binary values")
            # 10. Labels.
            if include_targets:
                for column in targets:
                    labels = frame[column].to_numpy()
                    if not np.isin(labels, [0, 1]).all():
                        failures.append(f"{split}/{part.name}: {column} non-01 labels")
                if split_train_frame_columns is None:
                    split_train_frame_columns = list(frame.columns)
            else:
                # 6. Test schema == train schema minus labels.
                if split_train_frame_columns is not None:
                    pass  # train processed first; final check below
            del table, frame
        split_rows[split] = total
        print(f"[ok] split {split!r}: {total:,} rows across {len(parts)} part file(s)")

    # 3. Raw vs processed row parity (full mode).
    if not args.smoke:
        from recsys23_fedrec.preprocessing.config import PreprocessingConfig

        config = PreprocessingConfig(data_dir=args.data_dir)
        raw_train = count_raw_rows(config, "train")
        raw_test = count_raw_rows(config, "test")
        if raw_train != split_rows.get("train"):
            failures.append(
                f"train row mismatch: raw {raw_train:,} vs processed {split_rows.get('train'):,}"
            )
        if raw_test != split_rows.get("test"):
            failures.append(
                f"test row mismatch: raw {raw_test:,} vs processed {split_rows.get('test'):,}"
            )
        if not failures:
            print(f"[ok] row parity: train {raw_train:,}, test {raw_test:,} (raw == processed)")

    # 6. Schema compatibility apart from labels.
    if split_train_frame_columns is not None:
        test_parts = store.part_files("test")
        test_columns = list(pq.read_schema(test_parts[0]).names)
        if set(split_train_frame_columns) - set(test_columns) != set(targets):
            failures.append("train/test processed schema differ beyond the label columns")

    # 10. Label totals vs manifest.
    processed_label_counts = manifest.get("label_counts", {})
    for column, counts in processed_label_counts.items():
        stated = {int(k): int(v) for k, v in counts.items()}
        total = stated.get(0, 0) + stated.get(1, 0)
        if total != split_rows.get("train"):
            failures.append(
                f"manifest label_counts[{column}] totals {total:,} != processed train rows"
            )

    # 12. Representative sample: raw row vs processed row.
    _print_sample(args.data_dir, store, preprocessor)

    return _report(failures)


def _print_sample(data_dir: str, store: PreprocessingArtifactStore, preprocessor: Preprocessor) -> None:
    from recsys23_fedrec.preprocessing.config import PreprocessingConfig

    config = PreprocessingConfig(data_dir=data_dir)
    train_path = config.resolved_shard_paths("train")[0]
    with open(train_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=config.delimiter)
        header = next(reader)
        raw_values = next(reader)
    raw_row = dict(zip(header, raw_values))

    first_part = store.part_files("train")[0]
    processed = pq.read_table(first_part).to_pandas().head(1)

    show = ["f_1", "f_2", "f_26", "f_30", "f_30_missing", "f_42", "f_43",
            "f_43_missing", "is_clicked", "is_installed"]
    print("\nRepresentative sample (raw row 0 of first train shard vs processed row 0):")
    for name in show:
        if name in raw_row and name in processed.columns:
            print(f"  {name:>14}: raw={raw_row[name]!r:>24}  processed={processed.iloc[0][name]!r}")
        elif name in raw_row:
            print(f"  {name:>14}: raw={raw_row[name]!r}  (not a processed feature)")
    click = processed.iloc[0]["is_clicked"]
    install = processed.iloc[0]["is_installed"]
    print(f"  Click label: {click} | Install label: {install}")


def _report(failures: list[str]) -> int:
    if failures:
        print("\nVERIFICATION FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nVERIFICATION PASSED: all integrity checks succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
