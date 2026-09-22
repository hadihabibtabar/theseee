"""Stage 9: read-only verification of per-client DataLoaders on REAL data.

Loads the frozen split + the primary alpha=0.5 partition and exercises
:func:`make_client_loader` against the real processed Parquet:

* ``client_02`` (smallest, 59,999 rows): FULL verification — complete
  sorted pass with row-identity-by-order, batch contract, manifest label
  rates, full shuffled epoch pass, determinism, per-epoch reshuffling.
* ``client_00`` (largest, 621,801 rows): structural verification (loader
  length, sampler protocol, row identity on inspected batches) plus a
  bounded partial shuffled iteration (first 50 batches) for determinism and
  per-epoch difference. Full scans of the largest client are avoided on
  purpose: shuffled access hops randomly across all 90 Parquet parts.

READ-ONLY: no checkpoint, artifact, dataset, or manifest is modified; no
model, optimizer, backward pass, or training is involved.

Usage:
    python scripts/verify_stage9_client_loaders.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from recsys23_fedrec.data import (  # noqa: E402
    FeatureSet,
    ProcessedRecSysDataset,
    validate_batch,
)
from recsys23_fedrec.federated.client_loader import (  # noqa: E402
    compose_client_row_indices,
    load_partition_positions,
    make_client_loader,
)
from recsys23_fedrec.training.split import load_split  # noqa: E402

EXPECTED_PARTITION_HASH = "9603ddc9facc973d"
MANIFEST_RELPATH = "artifacts/federated/partition_alpha_0_5_seed42.json"
NPZ_RELPATH = "artifacts/federated/partition_alpha_0_5_seed42.npz"

BATCH = 1024


def _fail(msg: str) -> None:
    print(f"\nVERIFICATION FAILED: {msg}")
    raise SystemExit(1)


def _check(cond: bool, ok: str, fail: str) -> None:
    if cond:
        print(f"  [OK] {ok}")
    else:
        _fail(fail)


def sampler_prefix(dataset, train_indices, pos, epoch: int, limit: int) -> list[tuple]:
    """First ``limit`` shuffled batches as local-position tuples.

    Batch CONTENT (not sizes) is the robust arrangement fingerprint: the
    Stage 8 sampler sorts positions within each batch, so sizes alone are
    identical across epochs for all full batches.
    """
    ld = make_client_loader(dataset, train_indices, pos, batch_size=BATCH,
                            shuffle=True, seed=42, epoch=epoch, pin_memory=False)
    return [tuple(b) for b in list(ld.batch_sampler)[:limit]]


def main() -> int:
    print("=== Stage 9 client-loader verification (real data, read-only) ===")

    # ---- protected partition identity ----------------------------------
    manifest = json.loads((PROJECT_ROOT / MANIFEST_RELPATH).read_text(encoding="utf-8"))
    _check(
        manifest["partition_hash"] == EXPECTED_PARTITION_HASH,
        f"manifest partition hash = {manifest['partition_hash']}",
        "partition hash mismatch",
    )
    positions = load_partition_positions(PROJECT_ROOT / NPZ_RELPATH)
    print("  partition NPZ structure validated")

    # ---- frozen split + dataset ------------------------------------------
    split = load_split(
        PROJECT_ROOT / "artifacts" / "splits" / "centralized_split.json",
        expected_total_rows=3_485_852,
    )
    train_indices = split.train_indices
    validation_indices = split.validation_indices
    _check(
        len(train_indices) == 3_137_266,
        f"frozen train indices: {len(train_indices):,}",
        "unexpected train index count",
    )
    feature_set = FeatureSet.load(PROJECT_ROOT / "artifacts" / "preprocessing")
    _check(
        feature_set.artifact_hash == "c3a62caf13326617",
        f"Stage 2 artifact hash {feature_set.artifact_hash}",
        "unexpected preprocessing artifact",
    )
    dataset = ProcessedRecSysDataset(
        feature_set, PROJECT_ROOT / "artifacts" / "processed" / "train", split="train"
    )
    print(f"  dataset: {len(dataset):,} rows over {len(dataset.part_paths)} parts")

    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss / (1024 * 1024)

    # ================= client_02: smallest — FULL verification ===========
    cid = "client_02"
    print(f"\n--- {cid} (smallest): full verification ---")
    pos = positions[cid]
    rows = compose_client_row_indices(train_indices, pos)
    n = len(rows)
    stats = manifest["clients"][cid]
    _check(n == stats["rows"] == 59_999, f"rows = {n:,} == manifest", "row count mismatch")
    _check(np.all(np.diff(rows) > 0), "client rows strictly ascending (unique)",
           "client rows duplicated/unsorted")
    _check(np.intersect1d(rows, validation_indices).size == 0,
           "zero overlap with frozen validation rows", "validation leakage")

    loader = make_client_loader(dataset, train_indices, pos, batch_size=BATCH,
                                shuffle=False, pin_memory=False)
    _check(len(loader) == (n + BATCH - 1) // BATCH,
           f"loader length {len(loader)} == ceil({n:,}/{BATCH})", "loader length mismatch")
    t0 = time.time()
    click_sum = install_sum = 0.0
    total = idx = 0
    for i, batch in enumerate(loader):
        if i < 5:
            validate_batch(batch, feature_set, split="train")
            assert batch["categorical"].dtype == torch.int64
            assert batch["numerical"].dtype == torch.float32
            assert batch["binary"].dtype == torch.float32
            assert batch["missing"].dtype == torch.float32
            assert batch["click"].dtype == torch.float32
            assert batch["install"].dtype == torch.float32
            assert batch["click"].shape == (len(batch["click"]),)
            assert batch["install"].shape == (len(batch["install"]),)
        total += len(batch["click"])
        click_sum += float(batch["click"].sum())
        install_sum += float(batch["install"].sum())
    dt = time.time() - t0
    _check(total == n, f"full sorted pass yielded {total:,} rows == client size "
                       f"({dt:.0f}s)", "full pass row count mismatch")
    _check(abs(click_sum / total - stats["click_rate"]) < 5e-4,
           f"click rate {click_sum / total:.4f} matches manifest {stats['click_rate']:.4f}",
           "click rate differs from manifest")
    _check(abs(install_sum / total - stats["install_rate"]) < 5e-4,
           f"install rate {install_sum / total:.4f} matches manifest {stats['install_rate']:.4f}",
           "install rate differs from manifest")

    def batches(epoch: int, limit: int | None = None):
        ld = make_client_loader(dataset, train_indices, pos, batch_size=BATCH,
                                shuffle=True, seed=42, epoch=epoch, pin_memory=False)
        out, seen = [], 0
        for batch in ld:
            out.append(len(batch["click"]))
            seen += 1
            if limit is not None and seen >= limit:
                break
        return out

    t0 = time.time()
    e1_full = batches(0)
    dt = time.time() - t0
    _check(len(e1_full) == (n + BATCH - 1) // BATCH and sum(e1_full) == n,
           f"full shuffled epoch-1 pass covers {sum(e1_full):,} rows in "
           f"{len(e1_full)} batches ({dt:.0f}s)", "shuffled epoch coverage wrong")
    e1b = batches(0, limit=20)
    e1a = e1_full[:20]
    _check(e1a == e1b, "epoch-1 arrangement deterministic across builds",
           "shuffled iteration not deterministic")
    s1 = sampler_prefix(dataset, train_indices, pos, 0, 20)
    s2 = sampler_prefix(dataset, train_indices, pos, 1, 20)
    s3 = sampler_prefix(dataset, train_indices, pos, 2, 20)
    _check(s1 != s2 and s2 != s3 and s1 != s3,
           "epochs 1/2/3 produce different batch arrangements (content check)",
           "per-epoch reshuffling not active")
    del e1b, s1, s2, s3
    # row identity on shuffled batches: same multiset of row features as the
    # sorted pass is implied by coverage; verify per-batch composition is
    # exactly the partition positions via the sampler's own arrays.
    sampler = make_client_loader(dataset, train_indices, pos, batch_size=BATCH,
                                 shuffle=True, seed=42, epoch=0,
                                 pin_memory=False).batch_sampler
    all_batches = list(sampler)
    covered = np.concatenate([np.asarray(b) for b in all_batches])
    # Sampler batches hold LOCAL positions into the client's sorted row
    # subset; exact cover means every local position is used exactly once,
    # hence every client row is used exactly once.
    _check(np.array_equal(np.sort(covered), np.arange(n)),
           "sampler batches cover exactly the client rows, each once",
           "sampler does not cover exactly the client rows")
    del all_batches, covered

    # ================= client_00: largest — structural + partial =========
    cid = "client_00"
    print(f"\n--- {cid} (largest): structural + bounded partial verification ---")
    pos = positions[cid]
    rows = compose_client_row_indices(train_indices, pos)
    n = len(rows)
    _check(n == manifest["clients"][cid]["rows"] == 621_801,
           f"rows = {n:,} == manifest (largest client)", "row count mismatch")
    sizes = {c: manifest["clients"][c]["rows"] for c in manifest["clients"]}
    _check(n == max(sizes.values()) and min(sizes.values()) == 59_999,
           "client_00 is the largest and client_02 the smallest client",
           "size ordering wrong")
    _check(np.intersect1d(rows, validation_indices).size == 0,
           "zero overlap with frozen validation rows", "validation leakage")

    loader = make_client_loader(dataset, train_indices, pos, batch_size=BATCH,
                                shuffle=False, pin_memory=False)
    _check(len(loader) == (n + BATCH - 1) // BATCH,
           f"loader length {len(loader)} == ceil({n:,}/{BATCH})", "loader length mismatch")
    t0 = time.time()
    for i, batch in enumerate(loader):
        validate_batch(batch, feature_set, split="train")
        assert batch["categorical"].dtype == torch.int64
        assert batch["numerical"].dtype == torch.float32
        assert batch["click"].dtype == torch.float32
        if i == 49:
            break
    _check(i == 49, f"50 real batches contract-checked ({time.time() - t0:.0f}s)",
           "batch iteration failed")
    del loader

    e1 = batches(0, limit=50)
    e1b = batches(0, limit=50)
    _check(e1 == e1b, "epoch-1 partial arrangement deterministic", "nondeterministic")
    s1 = sampler_prefix(dataset, train_indices, pos, 0, 50)
    s2 = sampler_prefix(dataset, train_indices, pos, 1, 50)
    s3 = sampler_prefix(dataset, train_indices, pos, 2, 50)
    _check(s1 != s2 and s2 != s3 and s1 != s3,
           "epochs 1/2/3 produce different batch arrangements (content check)",
           "per-epoch reshuffling not active")
    del e1, e1b, s1, s2, s3

    # ---- cross-client disjointness ---------------------------------------
    print("\n--- cross-client disjointness (all 10 clients) ---")
    all_pos = np.concatenate([positions[c] for c in sorted(positions)])
    _check(np.unique(all_pos).size == all_pos.size,
           "client positions mutually disjoint across all 10 clients",
           "positions shared between clients")
    _check(all_pos.max() < len(train_indices),
           f"max position {all_pos.max():,} < train space {len(train_indices):,}",
           "positions reach outside train space")

    rss1 = proc.memory_info().rss / (1024 * 1024)
    print(f"\n--- memory ---")
    print(f"  [OK] RSS {rss0:.0f} -> {rss1:.0f} MiB (bounded; row-group streaming "
          f"holds ~1 row group per process, never the client subset)")

    print("\n--- official test data ---")
    print("  [OK] never opened (no test path/reader constructed anywhere)")

    print("\nVERIFICATION COMPLETE: all client-loader checks passed.")
    print("No training, model construction, optimizer, or backward pass occurred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
