# Preprocessing Report - recsys23-federated-install

Generated: 2026-09-15 | Stage: `preprocessing` | Framework: pytorch | Artifact hash: `c3a62caf13326617`

## 1. Inputs & provenance

- Source of truth: Stage 1 report (`reports/dataset_report.json`, seed 42)
- Raw data: `train/` (30 shards, 3,485,852 rows), `test/` (1 shard, 160,973 rows); TAB-separated
- Raw files unmodified (byte totals match Stage 1: train 1,919,412,753 B; test 88,684,100 B)

## 2. Feature categorization (deterministic, saved in `feature_schema.json`)

- ID (excluded from features): **f_0**
- Targets (kept as labels): **is_clicked**, **is_installed** (raw-file order)
- Categorical: **30** — f_1..f_25, f_26, f_27, f_28, f_29, f_32
- Binary: **11** — f_30, f_31, f_33..f_41 (train values ⊆ {0,1})
- Numerical: **38** — f_42..f_79
- Missing indicators: **13** — f_30_missing, f_31_missing, f_43_missing, f_51_missing,
  f_58_missing..f_70_missing (one per feature with training missingness)
- Ambiguity resolved: Stage 1's cardinality-based `binary` flag marks f_26..f_29 as binary,
  but their two values are raw IDs (e.g. {0, 14897}), not {0,1}; they are categorical here.
  f_32 ({0,1,2,3}) is also categorical (small embedding), not numerical.

## 3. Encoding conventions (documented in `preprocessing_config.json`)

- Categorical IDs: 0 = UNK (unseen), 1 = MISSING, 2+ = known categories ordered by
  (descending train frequency, then ascending value). No embeddings created at this stage.
- Numerical: `(x - train_mean) / train_std`; missing imputed with train mean → normalized 0.0;
  zero-variance columns map to 0.0 (f_7 constant col handled via categorical vocab; f_71-style
  zero-variance floats snap to std=0 within 1e-12 relative tolerance).
- Binary: values restricted to {0,1}; missing imputed with rounded training mean; no vocabulary.
- Labels: validated exactly {0,1}; missing labels raise `DataContractError`.
- Leakage: all fitted state (vocabularies, mean/std, imputation, indicator list) originates
  from training shards only; `transform` never mutates state.

## 4. Categorical vocabulary sizes (incl. UNK + MISSING)

f_1: 24, f_2: 138, f_3: 7, f_4: 635, f_5: 8, f_6: 5169, f_7: 3, f_8: 8, f_9: 9, f_10: 5,
f_11: 26, f_12: 28, f_13: 331, f_14: 21, f_15: 5803, f_16: 12, f_17: 51, f_18: 903,
f_19: 21, f_20: 57, f_21: 36, f_22: 26, f_23: 6, f_24: 6, f_25: 5, f_26: 4, f_27: 4,
f_28: 4, f_29: 4, f_32: 6 — total embedding rows ≈ 13,833 across 30 fields.
Max non-ID cardinality ≈ 5,801 (f_15): explicit vocabularies everywhere; **no hashing used**.

## 5. Processing results

- Processed train rows: **3,485,852** (90 Parquet parts, snappy, int32/float32/int64)
- Processed test rows: **160,973** (4 Parquet parts; schema == train minus labels)
- Row parity raw vs processed: exact; no rows created or lost
- Label totals (match Stage 1 exactly): is_clicked 0=2,719,538 / 1=766,314;
  is_installed 0=2,879,250 / 1=606,602
- Chunksize: 50,000 rows (configurable); peak memory bounded to ~1 chunk per column
- Duration: 54.6 s (fit + transform train + transform test)

## 6. Verification

- Unit tests: **25 passed** (`python scripts/test_preprocessor.py`, synthetic fixture only)
- Integrity script (`python scripts/verify_preprocessing.py --output_dir artifacts`): PASSED
  (artifact hash verifies; per-part artifact-hash linkage; f_0 absent; column sets exact;
  categorical IDs within vocabulary bounds; features finite; binaries {0,1}; labels {0,1};
  train/test schema compatible apart from labels; Parquet reopens)
- Determinism: double-fit produces identical artifact hash (content hash over schema +
  vocabularies + stats + config); `Preprocessor.load` rejects tampered artifacts
