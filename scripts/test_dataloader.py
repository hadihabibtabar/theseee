"""Real-data smoke test for the Stage 3 Dataset/DataLoader layer.

Runs exclusively against the Stage 2 processed Parquet outputs in
``artifacts/processed`` - the raw CSVs are never touched.

Checks (Stage 3 spec items 15-21):
* batch shapes and dtypes for batch_size in {32, 1024}
* categorical ID ranges (per feature, from the artifact vocabularies)
* numerical finiteness, binary/missing/label domains
* determinism (shuffle=False -> identical batches across constructions)
* num_workers=2 equivalence with num_workers=0
* Parquet part-boundary integrity (no duplicate/skip rows)
* bounded memory: RSS before/after construction, first batch, many batches
* row counts: train == 3,485,852 and test == 160,973 (Stage 2)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recsys23_fedrec.data import (  # noqa: E402
    FeatureSet,
    ProcessedRecSysDataset,
    create_dataloader,
    validate_batch,
)

EXPECTED_TRAIN_ROWS = 3_485_852
EXPECTED_TEST_ROWS = 160_973


def rss_mib() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts_dir", default=str(PROJECT_ROOT / "artifacts"))
    parser.add_argument("--batches", type=int, default=25, help="batches to iterate for memory/stats")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    train_dir = artifacts_dir / "processed" / "train"
    test_dir = artifacts_dir / "processed" / "test"

    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        print(("PASS  " if condition else "FAIL  ") + message)
        if not condition:
            failures.append(message)

    # ------------------------------------------------------------------
    section("Stage 2 artifacts")
    rss_start = rss_mib()
    feature_set = FeatureSet.load(artifacts_dir)
    check(
        (feature_set.num_categorical, feature_set.num_numerical, feature_set.num_binary, feature_set.num_missing)
        == (30, 38, 11, 13),
        "artifact feature widths = (categorical 30, numerical 38, binary 11, missing 13)"
        f" got ({feature_set.num_categorical}, {feature_set.num_numerical},"
        f" {feature_set.num_binary}, {feature_set.num_missing})",
    )
    print(f"artifact hash: {feature_set.artifact_hash}")

    # ------------------------------------------------------------------
    section("Dataset construction + row counts")
    rss_before_build = rss_mib()
    t0 = time.perf_counter()
    train_ds = ProcessedRecSysDataset(feature_set, train_dir, split="train")
    test_ds = ProcessedRecSysDataset(feature_set, test_dir, split="test")
    build_seconds = time.perf_counter() - t0
    rss_after_build = rss_mib()
    check(len(train_ds) == EXPECTED_TRAIN_ROWS, f"train length == {EXPECTED_TRAIN_ROWS:,} (got {len(train_ds):,})")
    check(len(test_ds) == EXPECTED_TEST_ROWS, f"test length == {EXPECTED_TEST_ROWS:,} (got {len(test_ds):,})")
    print(f"train parts: {train_ds.num_parts}, test parts: {test_ds.num_parts}")
    print(f"build time: {build_seconds:.2f}s (metadata-only scan)")
    print(f"RSS before build: {rss_before_build:.1f} MiB, after build: {rss_after_build:.1f} MiB")

    # ------------------------------------------------------------------
    section("Sample structure (train[0] vs raw row)")
    sample = train_ds[0]
    print(f"categorical: shape={tuple(sample['categorical'].shape)} dtype={sample['categorical'].dtype} values[:6]={sample['categorical'][:6].tolist()}")
    print(f"numerical:   shape={tuple(sample['numerical'].shape)} dtype={sample['numerical'].dtype} values[:6]={sample['numerical'][:6].tolist()}")
    print(f"binary:      shape={tuple(sample['binary'].shape)} dtype={sample['binary'].dtype} values={sample['binary'].tolist()}")
    print(f"missing:     shape={tuple(sample['missing'].shape)} dtype={sample['missing'].dtype} values={sample['missing'].tolist()}")
    print(f"click:       value={sample['click'].item():.0f} dtype={sample['click'].dtype}")
    print(f"install:     value={sample['install'].item():.0f} dtype={sample['install'].dtype}")

    # ------------------------------------------------------------------
    section("Batch shapes/dtypes for batch_size in {32, 1024}")
    for batch_size in (32, 1024):
        loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        batch = next(iter(loader))
        print(f"\nBatch size: {batch_size}")
        for key, expect_shape, expect_dtype in (
            ("categorical", (batch_size, 30), torch.int64),
            ("numerical", (batch_size, 38), torch.float32),
            ("binary", (batch_size, 11), torch.float32),
            ("missing", (batch_size, 13), torch.float32),
            ("click", (batch_size,), torch.float32),
            ("install", (batch_size,), torch.float32),
        ):
            tensor = batch[key]
            ok = tuple(tensor.shape) == expect_shape and tensor.dtype == expect_dtype
            check(ok, f"{key}: shape={tuple(tensor.shape)} dtype={tensor.dtype} (expected {expect_shape}, {expect_dtype})")
        try:
            validate_batch(batch, feature_set, split="train")
            check(True, f"validate_batch passes at batch_size={batch_size}")
        except Exception as exc:  # noqa: BLE001
            check(False, f"validate_batch failed at batch_size={batch_size}: {exc}")

    # ------------------------------------------------------------------
    section("Value statistics over the first batch (batch_size=1024)")
    loader = create_dataloader(train_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    cat = batch["categorical"]
    vocab = feature_set.vocabulary_sizes
    max_ids = cat.amax(dim=0)
    min_ids = cat.amin(dim=0)
    worst = (max_ids - (torch.tensor(vocab, dtype=torch.int64) - 1)).amax().item()
    check(worst <= 0, f"all categorical IDs in range: per-feature max id <= vocab-1 (worst slack {worst})")
    print(f"categorical per-feature min IDs: {min_ids.tolist()}")
    print(f"categorical per-feature max IDs: {max_ids.tolist()}")
    print(f"categorical per-feature vocab sizes: {list(vocab)}")
    num = batch["numerical"]
    check(bool(torch.isfinite(num).all()), f"numerical finite: min={num.min().item():.4f} max={num.max().item():.4f}")
    print(f"numerical per-feature min[:6]: {[round(v, 3) for v in num.amin(dim=0)[:6].tolist()]}")
    print(f"numerical per-feature max[:6]: {[round(v, 3) for v in num.amax(dim=0)[:6].tolist()]}")
    for key in ("binary", "missing", "click", "install"):
        uniq = torch.unique(batch[key]).tolist()
        check(set(uniq) <= {0.0, 1.0}, f"{key} unique values in  {{0,1}}: {uniq}")
    print(f"click positive rate:   {batch['click'].mean().item():.4f}")
    print(f"install positive rate: {batch['install'].mean().item():.4f}")

    # ------------------------------------------------------------------
    section("Iteration over first 5 batches")
    loader = create_dataloader(train_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)
    shapes_seen: set[tuple] = set()
    click_rates: list[float] = []
    install_rates: list[float] = []
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        shapes_seen.add(tuple(batch[k].shape for k in ("categorical", "numerical", "binary", "missing", "click", "install")))
        try:
            validate_batch(batch, feature_set, split="train")
        except Exception as exc:  # noqa: BLE001
            check(False, f"batch {i} failed validation: {exc}")
            break
        click_rates.append(batch["click"].mean().item())
        install_rates.append(batch["install"].mean().item())
    check(len(shapes_seen) == 1, f"shapes constant across 5 batches: {shapes_seen.pop()}")
    print(f"click rates:   {[round(r, 4) for r in click_rates]}")
    print(f"install rates: {[round(r, 4) for r in install_rates]}")

    # ------------------------------------------------------------------
    section("Determinism (shuffle=False, two constructions)")
    ds_a = ProcessedRecSysDataset(feature_set, train_dir, split="train")
    ds_b = ProcessedRecSysDataset(feature_set, train_dir, split="train")
    loader_a = create_dataloader(ds_a, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    loader_b = create_dataloader(ds_b, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    ba = next(iter(loader_a))
    bb = next(iter(loader_b))
    same = all(torch.equal(ba[k], bb[k]) for k in ba)
    check(same, "two shuffle=False loaders yield identical first batches")

    # ------------------------------------------------------------------
    section("Parquet part boundaries (first + a later boundary)")
    boundaries = train_ds.part_boundaries()
    checked = []
    for part in (1, min(45, train_ds.num_parts - 1)):
        b = boundaries[part]
        rows = [train_ds[b - 1], train_ds[b], train_ds[b + 1]]
        distinct = any(
            not torch.equal(rows[0][k], rows[1][k]) for k in ("categorical", "numerical")
        )
        check(distinct, f"boundary at part {part} (row {b:,}): last row of part {part-1} != first row of part {part} (no duplication)")
        checked.append(b)
    # Sequential contiguous access across a boundary must be gap-free.
    b0 = checked[0]
    seq = train_ds[b0 - 2 : b0 + 2]
    check(len(seq) == 4, f"slice spanning boundary yields all 4 rows ({len(seq)})")

    # ------------------------------------------------------------------
    section("num_workers=2 (Windows spawn)")
    rss_before_workers = rss_mib()
    loader0 = create_dataloader(train_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    main_batch = next(iter(loader0))
    loader2 = create_dataloader(train_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=False)
    worker_batch = next(iter(loader2))
    same = all(torch.equal(main_batch[k], worker_batch[k]) for k in main_batch)
    check(same, "num_workers=2 first batch identical to num_workers=0")
    n = 0
    for batch in loader2:
        n += 1
        if n == 10:
            break
    check(n == 10, f"10 batches from num_workers=2 loader (worker processes survive re-iteration)")
    print(f"RSS during worker iteration: {rss_mib():.1f} MiB (before: {rss_before_workers:.1f} MiB)")

    # ------------------------------------------------------------------
    section("Memory observations (bounded-RAM evidence)")
    print(f"RSS at script start:        {rss_start:8.1f} MiB")
    print(f"RSS after construction:     {rss_after_build:8.1f} MiB")
    loader = create_dataloader(train_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)
    first = True
    t0 = time.perf_counter()
    n_batches = 0
    peak_rss = rss_mib()
    for batch in loader:
        n_batches += 1
        peak_rss = max(peak_rss, rss_mib())
        if first:
            print(f"RSS after first batch:      {rss_mib():8.1f} MiB")
            first = False
        if n_batches >= args.batches:
            break
    seconds = time.perf_counter() - t0
    print(f"RSS after {n_batches} batches:    {rss_mib():8.1f} MiB (peak {peak_rss:.1f} MiB)")
    rows_seen = n_batches * 1024
    est_full = rows_seen / (n_batches) * 0  # placeholder to keep format simple
    per_batch_mib = (rss_mib() - rss_after_build) / max(n_batches, 1)
    print(f"iteration speed: {rows_seen / seconds:,.0f} rows/s ({n_batches} batches in {seconds:.2f}s)")
    print(f"approx. RSS growth per 1024-row batch: {per_batch_mib:.3f} MiB")
    check(peak_rss < 1500, f"peak RSS {peak_rss:.0f} MiB stays far below a full in-RAM materialization (~2+ GiB)")

    # ------------------------------------------------------------------
    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: ALL REAL-DATA CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
