# Stage 3 Report — PyTorch Dataset / DataLoader Layer

Built on the frozen Stage 2 preprocessing outputs (artifact hash `c3a62caf13326617`). No raw data touched; no model code created.

## Files created/modified

Created:
- `src/recsys23_fedrec/data/__init__.py` — public API re-exports
- `src/recsys23_fedrec/data/dataset.py` — `ProcessedRecSysDataset` (memory-bounded Parquet row-group streaming)
- `src/recsys23_fedrec/data/collate.py` — `collate_train`, `collate_test`, `create_dataloader`
- `src/recsys23_fedrec/data/samples.py` — sample-structure docs, `BatchValidator`, `validate_batch`
- `src/recsys23_fedrec/data/feature_schema.py` — `FeatureSet` (artifact-frozen feature order), `GPU` device bundle
- `src/recsys23_fedrec/data/exceptions.py` — `DataError`, `FeatureOrderError`, `BatchValidationError`
- `tests/data/conftest.py`, `tests/data/test_dataset.py` — 27 synthetic-fixture tests
- `scripts/test_dataloader.py` — real-data smoke test (all Stage 3 spec items 15–21)

Modified:
- `src/recsys23_fedrec/preprocessing/preprocessor.py` — added `vocabulary_sizes_by_order()` accessor (used by `FeatureSet`)

## Dataset architecture

`ProcessedRecSysDataset(feature_set, split_dir, split)`:

- **Construction reads metadata only** (row counts + row-group sizes per part): 0.05 s for 90 train parts + 4 test parts, no data I/O.
- **`__getitem__`** locates the part via cumulative row offsets (bisect), then the row group within the part, and reads **one row group at a time** (`pq.ParquetFile.read_row_group`). The row group is converted to pre-stacked numpy arrays (one 2D array per tensor block, artifact column order) held in a **single-slot cache**; rollover discards the previous group.
- **RAM holds at most one row group (~50k rows ≈ 30 MiB)**, never the full 3.5M rows. No full-dataset DataFrame, no per-row Python lists.
- **Feature order** comes exclusively from the Stage 2 artifact via `FeatureSet` (categorical 30 → binary 11 → numerical 38 → missing 13 → labels). Parquet column order is *verified* against it at construction; mismatch raises `FeatureOrderError`. Parts carry the artifact hash in Parquet key-value metadata; a mismatch is rejected (data/artifact out of sync).
- **Worker safety (Windows spawn):** `__getstate__`/`__setstate__` strip open Parquet handles and the cache; every worker rebuilds its own readers lazily. No handle is shared across processes. `worker_init_fn` is auto-wired by `create_dataloader` for `num_workers > 0`.
- **Device:** everything stays on CPU; no `.cuda()`/`.to(device)` anywhere in the data layer.

## Batch contract (widths derived from the artifact, not hard-coded)

| field | dtype | shape (batch size B) |
|---|---|---|
| categorical | int64 (torch.long) | [B, 30] |
| numerical | float32 | [B, 38] |
| binary | float32 | [B, 11] |
| missing | float32 | [B, 13] |
| click | float32 | [B] |
| install | float32 | [B] |

Labels are raw {0,1} — no sigmoid (for later `BCEWithLogitsLoss`). Test split returns the same dict without `click`/`install`. `validate_batch(batch, feature_schema)` checks keys, shapes, dtypes, per-feature ID ranges (0 ≤ id < vocab[j]), numerical finiteness, and {0,1} domains, naming the offending field in every error.

## Row counts (no rows lost)

- train length = **3,485,852** (90 parts, one per Stage 2 chunk)
- test length = **160,973** (4 parts)
- sum of part row counts == dataset length; boundary rows verified distinct (no dup/skip at part 1 and part 45 boundaries; slices spanning boundaries are gap-free)
- `drop_last=True` affects iteration count only; dataset length untouched (tested)

## Test results

- Synthetic-fixture suite: **52/52 passed** (`pytest tests/`: 25 Stage 2 + 27 Stage 3), covering: artifact-driven ordering, Parquet-column-mismatch and artifact-hash-mismatch rejection, dtypes/shapes, negative/slice/out-of-range indexing, collate contracts, all `validate_batch` rejection paths, sequential-order == dataset order, fixed-generator shuffle determinism, bounded cache, worker equivalence (num_workers=2 == 0), row-count integrity, part boundaries.
- Real-data script `scripts/test_dataloader.py`: **all checks passed** —
  - batch sizes 32 and 1024: exact shapes/dtypes per contract, `validate_batch` passes
  - categorical per-feature max ID ≤ vocab−1 for all 30 features (vocab sizes: f_15=5803, f_6=5169, f_18=903, f_4=635, …, min f_7=3; first-batch min IDs all ≥ 2, i.e. no UNK hits in the first batch)
  - numerical finite (min −8.23, max 29.48 in first 1024-row batch); binary/missing/click/install unique values ⊆ {0,1}
  - first-batch positive rates: click 0.2178, install 0.1631 (consistent with Stage 1 class balance)
  - 5-batch iteration: shapes constant, all batches validate
  - determinism: two `shuffle=False` constructions yield identical first batches
  - workers: `num_workers=2` (Windows spawn) first batch bit-identical to `num_workers=0`; 10 batches OK

## Memory observations (bounded-RAM evidence)

| moment | RSS |
|---|---|
| script start | 373.5 MiB |
| after Dataset construction (metadata only) | 378.7 MiB |
| after first 1024-row batch (one row group cached) | 647.2 MiB |
| after 25 batches | 647.6–783.9 MiB, plateau |

RSS rises once (first row-group read into the single-slot cache) then plateaus — it does **not** grow with batches. Full materialization would be ~2+ GiB; observed peak < 0.8 GiB (interpreter + torch baseline included).

## Performance note

Row-group caching with pre-stacked numpy arrays (rather than per-row pandas `iloc`) yields **~33,000 rows/s** sequential read throughput on CPU (25 batches in 0.78 s), a ~30× improvement over the initial pandas-per-row implementation, with identical memory bounds. No further optimization done (premature).

## Limitations / notes

- Windows spawn requires picklable datasets; the handle-stripping pickle protocol is documented in `dataset.py`. Worker support is verified but performance tuning with workers is deferred to the training stage.
- Random-access `__getitem__` re-reads a row group when jumping across groups (single-slot cache); sequential/shuffled-with-worker iteration amortizes this. Acceptable for Stage 3; revisiting only if training profiles show it matters.
- The test split was opened and length-verified only; no training ever touches it.
- Raw CSV files verified untouched (byte totals match Stage 1: train 1,919,412,753 B; test 88,684,100 B). Stage 2 `verify_preprocessing.py` still passes end-to-end against the same artifacts.

Stopped after Stage 3 — no Transformer, MMoE, SSL, training loops, validation split, Dirichlet partitioning, or federated code was created.
