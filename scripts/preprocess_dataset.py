"""Stage 2 CLI: fit preprocessing on training data only, transform both
splits, persist artifacts and processed Parquet data.

Usage::

    python scripts/preprocess_dataset.py --data_dir . --output_dir artifacts
    python scripts/preprocess_dataset.py --smoke_rows 200000 --output_dir artifacts_smoke

The raw CSV files are never modified.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402  (import after sys.path setup)

from recsys23_fedrec.preprocessing import (  # noqa: E402
    PreprocessingArtifactStore,
    PreprocessingConfig,
    Preprocessor,
    count_raw_rows,
)
from recsys23_fedrec.preprocessing.feature_schema import (  # noqa: E402
    load_stage1_report_columns,
    schema_from_stage1_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and apply leakage-safe preprocessing (Stage 2)."
    )
    parser.add_argument("--data_dir", default=".", help="Directory holding train/ and test/")
    parser.add_argument(
        "--output_dir", default="artifacts", help="Output root for artifacts and processed data"
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=PreprocessingConfig().chunksize,
        help="Rows per processing chunk (memory bound)",
    )
    parser.add_argument(
        "--smoke_rows",
        type=int,
        default=None,
        help=(
            "Smoke mode: fit and transform only the first N rows of the first "
            "train/test shards, writing into --output_dir"
        ),
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> PreprocessingConfig:
    return PreprocessingConfig(
        data_dir=args.data_dir,
        chunksize=args.chunksize,
    )


def smoke_chunk_iterator(config: PreprocessingConfig, limit: int):
    """Yield at most ``limit`` rows total from the first train shard."""
    path = config.resolved_shard_paths("train")[0]
    remaining = limit
    reader = pd.read_csv(
        path,
        sep=config.delimiter,
        encoding=config.encoding,
        chunksize=min(config.chunksize, limit),
    )
    for chunk in reader:
        if remaining <= 0:
            break
        yield chunk.iloc[:remaining]
        remaining -= len(chunk)


def _transform_row_prefix(
    preprocessor: Preprocessor,
    config: PreprocessingConfig,
    split: str,
    limit: int,
) -> pd.DataFrame:
    """Transform only the first ``limit`` rows of the first shard of a split."""
    path = config.resolved_shard_paths(split)[0]
    frame = pd.read_csv(
        path,
        sep=config.delimiter,
        encoding=config.encoding,
        nrows=limit,
    )
    return preprocessor.transform(frame, split=split)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    config = build_config(args)
    report = load_stage1_report_columns(config.stage1_report_path)
    schema = schema_from_stage1_report(report, config)

    preprocessor = Preprocessor(config)
    smoke = args.smoke_rows is not None

    if smoke:
        print(
            f"[smoke] fitting on the first {args.smoke_rows:,} training rows "
            f"(first shard: {config.resolved_shard_paths('train')[0].name})"
        )
        preprocessor.fit_from_chunks(
            smoke_chunk_iterator(config, args.smoke_rows), schema
        )
    else:
        preprocessor.fit(report=report)

    store = PreprocessingArtifactStore(args.output_dir)

    # Persist artifacts BEFORE transforming, so processed data can always be
    # linked to the exact artifact version that produced it.
    preprocessor.save(args.output_dir)

    # Stream both splits to Parquet.
    if smoke:
        # Smoke mode: transform only row prefixes of the first shard of each
        # split, so the smoke run stays fast and writes consistent data.
        train_df = _transform_row_prefix(preprocessor, config, "train", args.smoke_rows)
        test_df = _transform_row_prefix(preprocessor, config, "test", args.smoke_rows)
        writer = store.open_writer(
            "train",
            preprocessor.arrow_schema(include_targets=True),
            preprocessor.artifact_hash,
        )
        writer.write_chunk(train_df)
        writer2 = store.open_writer(
            "test",
            preprocessor.arrow_schema(include_targets=False),
            preprocessor.artifact_hash,
        )
        writer2.write_chunk(test_df)
        train_rows, test_rows = len(train_df), len(test_df)
        raw_train_rows, raw_test_rows = train_rows, test_rows
    else:
        train_rows = preprocessor.transform_split_to_parquet(store, "train")
        test_rows = preprocessor.transform_split_to_parquet(store, "test")
        raw_train_rows = count_raw_rows(config, "train")
        raw_test_rows = count_raw_rows(config, "test")
        if train_rows != raw_train_rows or test_rows != raw_test_rows:
            raise SystemExit(
                f"Row-count mismatch: processed train {train_rows:,} vs raw "
                f"{raw_train_rows:,}; processed test {test_rows:,} vs raw "
                f"{raw_test_rows:,}"
            )

    print_summary(preprocessor.summary(), raw_train_rows, raw_test_rows,
                  train_rows, test_rows, config, args.output_dir, started)
    return 0


def print_summary(
    summary: dict,
    raw_train_rows: int,
    raw_test_rows: int,
    train_rows: int,
    test_rows: int,
    config: PreprocessingConfig,
    output_dir: str,
    started: float,
) -> None:
    def fmt_list(names: list[str], per_line: int = 10) -> str:
        return "\n".join(
            ", ".join(names[i : i + per_line]) for i in range(0, len(names), per_line)
        )

    def fmt_vocab(name: str) -> int:
        try:
            return int(name.split("_")[1])
        except (IndexError, ValueError):
            return 0

    print("=" * 72)
    print("Stage 2 preprocessing summary")
    print("=" * 72)
    print(f"Raw train rows: {raw_train_rows:,}")
    print(f"Raw test rows: {raw_test_rows:,}")
    print(f"Processed train rows: {train_rows:,}")
    print(f"Processed test rows: {test_rows:,}")
    print(f"ID features removed: {', '.join(summary['id_columns'])}")
    print(f"Targets kept as labels: {', '.join(summary['target_columns'])}")
    label_counts = summary.get("label_counts", {})
    for target, counts in label_counts.items():
        print(
            f"  {target}: 0={counts.get(0, 0):,}  1={counts.get(1, 0):,}"
        )
    print(f"\nCategorical features ({len(summary['categorical_columns'])}):")
    print(fmt_list(summary["categorical_columns"]))
    print(f"\nNumerical features ({len(summary['numerical_columns'])}):")
    print(fmt_list(summary["numerical_columns"]))
    print(f"\nBinary features ({len(summary['binary_columns'])}):")
    print(fmt_list(summary["binary_columns"]))
    print(f"\nMissing indicators ({len(summary['missing_indicator_columns'])}):")
    print(fmt_list(summary["missing_indicator_columns"]))
    print("\nCategorical vocabulary sizes (incl. UNK + MISSING):")
    for name, size in sorted(
        summary["vocabulary_sizes"].items(), key=lambda kv: fmt_vocab(kv[0])
    ):
        print(f"  {name}: {size}")
    print(f"\nChunksize: {config.chunksize:,} rows")
    print(f"Output directory: {output_dir}")
    print(f"Artifact hash: {summary['artifact_hash']}")
    print(f"Duration: {time.time() - started:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(run())
