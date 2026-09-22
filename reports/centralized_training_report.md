# Stage 6 Report — Centralized Supervised Training (Transformer + MMoE)

**Status: COMPLETE — all 3 epochs finished; verified 2026-09-16.** The 3-epoch run started
2026-09-15 17:06 (PID 13696, `python -u scripts/run_centralized_training.py`) and the process
exited normally after the final epoch; no crash occurred and no retraining is required.
Log and best checkpoint were last written 2026-09-16 16:32:34, followed by the run's own
checkpoint-reload verification (`RELOAD PASSED`), the official-test schema/inference
compatibility check, and `RESULT: STAGE 6 CENTRALIZED TRAINING COMPLETE`.

### Final Stage 6 baseline (frozen)

| Item | Value |
| --- | --- |
| Best epoch | 3 of 3 |
| **Best validation Install LogLoss** (primary selection metric) | **0.2562108886731335** |
| Best validation Install AUC | 0.9132 |
| Click LogLoss (same epoch) | 0.2640 |
| Click AUC (same epoch) | 0.9114 |
| Checkpoint | `artifacts/checkpoints/centralized_transformer_mmoe_best.pt` |

These validation metrics (fixed seed-42 split) are the Stage 6 supervised reference for later
comparison with SSL (Stages 7/8) and Federated Learning (Stages 10/11). The official test set
was used for a schema/inference compatibility check only — never for training, validation,
model selection, early stopping, or tuning.

> This is centralized supervised training.
> No federated clients, FedAvg, SSL, or contrastive learning were used.
> The official test set was not used for training or model selection.

---

## 1. Dataset size

| Item | Value |
| --- | --- |
| Raw train shards | 30 tab-separated CSVs, 1.79 GB (`train/`) |
| Raw test shard | 1 CSV, 0.08 GB (`test/`) |
| Processed train (Stage 2 Parquet) | 3,485,852 rows, 90 parts, 0.15 GB |
| Processed test (Stage 2 Parquet) | 160,973 rows, 4 parts, 0.01 GB |
| Features | 92 tokens: 30 categorical + 38 numerical + 11 binary + 13 missing-indicators |
| Labels (train) | click 22.0% positive, install 17.4% positive (Stage 1 report) |
| Data loader | Stage 3 `ProcessedRecSysDataset` — row-group streaming, ≤ 1 row group (~50k rows) in RAM |

## 2. Train / validation split

* Deterministic index split over the 3,485,852 processed train rows only.
* `seed = 42` (project `SEED` from `preprocessing.config`), `validation_ratio = 0.10`.
* One seeded PCG64 permutation; first `ceil(n * 0.10)` indices → validation, rest → training.
  Indices stored sorted (cache-friendly monotone Parquet access).
* Split files: `artifacts/splits/centralized_split.json` + `.npz` (hash-verified on load).

| Subset | Rows |
| --- | --- |
| Training | 3,137,266 (90.0%) |
| Validation | 348,586 (10.0%) |
| Total | 3,485,852 (disjoint, complete coverage — enforced by `TrainValidationSplit.validate()`) |

## 3. Split hashes

| Field | Value |
| --- | --- |
| `train_index_hash` | `dd37605169615442` |
| `validation_index_hash` | `3cd8370ebdecb305` |
| Hash function | SHA-256 (truncated to 64 bits) over the exact sorted int64 index array |

Every later experiment must load the split via `load_split(...)`; the hashes are re-verified
against the JSON on every load, and the train hash is embedded in every checkpoint.

## 4. Model architecture reference

Frozen Stage 4/5 architecture, unmodified in Stage 6:
92 feature tokens + CLS → 6-layer Transformer (d_model 128, nhead 8, dim_feedforward 128,
dropout 0.1) → CLS [128] → 8 shared MMoE experts (expert_dim 128) → click gate + install gate
→ click tower + install tower → click logit + install logit.
Parameter count: **2,502,930**.

## 5. Training configuration

`TrainingConfig` defaults (no hyperparameter search performed):

```
seed = 42                      batch_size = 1024          epochs = 3
learning_rate = 3e-4           weight_decay = 1e-2        optimizer = AdamW
lambda_click = 1.0             lambda_install = 1.0
early_stopping_patience = 2    device = CUDA if available else CPU
```

## 6. Optimizer

`torch.optim.AdamW(lr=3e-4, weight_decay=1e-2)` over all model parameters.
Constant LR (no scheduler — out of scope for the baseline).

## 7. Loss formulation

Stage 5 `mmoe_loss`, unchanged:

```
L_click   = BCEWithLogitsLoss(click_logit, click_label)
L_install = BCEWithLogitsLoss(install_logit, install_label)
L_total   = 1.0 * L_click + 1.0 * L_install
```

Raw logits only — sigmoid is never applied before the loss (it is fused inside
`BCEWithLogitsLoss`). No class weighting, no focal loss, no oversampling.

## 8. Metric definitions

* **LogLoss** (per task): mean binary cross-entropy from raw logits via the numerically
  stable fused form `softplus(-z) + (1 - y)·z`; aggregated **sample-weighted** across
  validation batches (sum of per-sample losses / total samples). Equivalent to
  `sklearn.metrics.log_loss(y, sigmoid(z))`.
* **ROC-AUC** (per task): exact rank-based AUC with tie handling
  (`AUC = P(score_pos > score_neg) + 0.5·P(tie)`), computed over all validation logits in
  one pass. Verified against `sklearn.metrics.roc_auc_score` in the test suite
  (`test_auc_matches_sklearn_on_realistic_sample`). No sklearn dependency at runtime.
* **Primary model-selection metric: validation Install LogLoss** (lower is better).
  Secondary: validation Install AUC, validation Click LogLoss, validation Click AUC.

Note: all three epochs of this run were evaluated with the previous batch-mean logloss
aggregation (the sample-weighted fix landed only in the source after the training process had
already loaded the old code); the difference on 348,586 samples in 341 batches is < 1e-5 and
does not affect selection or comparability with future runs.

## 9. Epoch-by-epoch results (final)

| Epoch | Train total | Train click | Train install | Val click LogLoss | Val install LogLoss | Val click AUC | Val install AUC | Duration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5951 | 0.3065 | 0.2886 | 0.2736 | 0.2643 | 0.9041 | 0.9059 | 22,711.6 s |
| 2 | 0.5337 | 0.2711 | 0.2626 | 0.2664 | 0.2586 | 0.9098 | 0.9112 | 58,431.1 s |
| 3 | 0.5243 | 0.2663 | 0.2580 | **0.2640** | **0.2562** | **0.9114** | **0.9132** | 3,209.9 s |

Every epoch set a new best validation Install LogLoss, so early stopping never triggered
(patience 2 unused). All metrics are finite, LogLoss positive, AUC within [0, 1]. Sanity
expectations (spec 16) are met; no threshold judgements are made. Epoch durations are
wall-clock and include the overnight machine-suspension gap (epoch 2 spans it), so the
epoch-to-epoch timing comparison is not meaningful — the metric trends are.

## 10. Best checkpoint epoch

**Epoch 3** (the final epoch; each of the three epochs improved on the previous best, so the
last write is the best). The checkpoint is re-written whenever validation Install LogLoss
improves; Click metrics, training loss, and the official test set are never used for
selection.

## 11. Best validation Install LogLoss

**0.2562108886731335** (epoch 3).

## 12. Best validation Install AUC

**0.9132** (epoch 3).

## 13. Click metrics (at best-install epoch)

Click LogLoss **0.2640**, Click AUC **0.9114** (epoch 3).

## 14. Training duration

Logged epoch durations: 22,711.6 s + 58,431.1 s + 3,209.9 s = **84,352.8 s total training
time** (≈ 23.4 h) on the RTX 4060 Ti. These are wall-clock figures that include the overnight
machine-suspension gap (epoch 2 spans it), so pure GPU compute was lower. Wall-clock span of
the run: 2026-09-15 17:06 → 2026-09-16 16:32:34 (≈ 23.4 h).

## 15. Peak / approximate memory

* Training-process RSS during the real run: start 644 MiB, peak **845 MiB** (final run
  summary). The 3.5M-row dataset is never materialized — the Stage 3 row-group cache holds at
  most one ~50k-row group per process.
* Tiny smoke run (CPU, batch 128): RSS start 476 MiB, peak 2,095 MiB (includes model init +
  one-time CUDA/context-free import overhead in the reporting harness).
* Final run summary: `RSS start/peak: 644 / 845 MiB`; `peak GPU allocated: 3,594 MiB`
  (`torch.cuda.max_memory_allocated()`), on the 8 GiB card.

## 16. CUDA information

| Item | Value |
| --- | --- |
| Device | NVIDIA GeForce RTX 4060 Ti (8 GiB) |
| CUDA available | True |
| torch / CUDA build | 2.7.0+cu118 |
| Python | 3.13.3 |
| Deterministic CUDA algorithms | **Disabled** (opt-in; recorded in checkpoints as `deterministic_cuda_algorithms: false`) |
| cuDNN benchmark | False |
| Seeding | `random`, `numpy`, `torch`, `torch.cuda.manual_seed_all` all seeded with 42 |
| Batch-order determinism | Per-epoch seeded permutation in `PartShuffledBatchSampler` (batch sequence shuffled, rows sorted within each batch for cache-friendly Parquet access) |

Because deterministic CUDA algorithms are off, bit-exact run-to-run equality on GPU is not
guaranteed; reproducibility is provided by seed + fixed split + fixed config instead.

## 17. Checkpoint reload verification

**PASSED** (performed automatically by the run script after training):

```
=== checkpoint reload verification ===
  click_logit:   finite=True, matches_saved=True
  install_logit: finite=True, matches_saved=True
checkpoint payload: epoch=3, seed=42, split_hash=dd37605169615442,
                    best_val_install_logloss=0.2562
RELOAD PASSED
```

Fresh model ← `model_state_dict`; two inference passes on one deterministic validation batch;
finiteness + `allclose(atol=1e-5)` vs. the in-memory best model.

Checkpoint payload verified post-run
(`artifacts/checkpoints/centralized_transformer_mmoe_best.pt`, 30,217,243 bytes):

```
keys: best_validation_install_logloss, epoch, model_state_dict (138 tensors),
      optimizer_state_dict (138 state entries, 1 param group), seed, split_hash, stage,
      training_config
epoch = 3, seed = 42, split_hash = dd37605169615442, stage = 6-centralized
best_validation_install_logloss = 0.2562108886731335
model parameters = 2,502,930
training_config = {seed 42, batch_size 1024, epochs 3, lr 3e-4, weight_decay 1e-2, adamw,
                   lambda_click 1.0, lambda_install 1.0, patience 2, device auto}
```

## 18. Official test compatibility

**PASSED — schema/inference compatibility only.** The run script's final check:

```
=== official test set: schema/inference compatibility only ===
test dataset length: 160,973 (labels absent, features only)
one inference-only forward on 64 test rows: click (64,), install (64,), finite=True
NOT used for training, validation, model selection, or early stopping.
```

The official test DataLoader was constructed over all 160,973 rows (dataset length verified,
labels absent as expected) and a one-batch inference forward produced finite, correctly
shaped logits. The test set was not used for training, validation, model selection, early
stopping, or tuning, and no test metrics were computed. Its processed Parquet was verified
intact (160,973 rows, artifact-hash metadata matches).

## 19. Full regression test count

**139 passed** (`python -m pytest`, re-run after the final Stage 6 code changes):

| Suite | Tests |
| --- | --- |
| tests/data (Stage 3) | 27 |
| tests/models (Stages 4–5) | 51 (26 transformer + 25 mmoe) |
| tests/preprocessing (Stage 2) | 25 |
| tests/training (Stage 6, new) | 36 (9 split + 15 metrics + 12 trainer) |
| **Total** | **139** (103 previous + 36 new; no regressions) |

Stage 6 tests cover: split determinism/disjointness/coverage/ratio/hash-stability/tamper
detection; logloss known values, perfect/worst predictions, stability at ±1e4 logits,
BCEWithLogits equivalence; AUC perfect/zero/0.75/0.25/tie cases, manual pairwise and sklearn
cross-checks; sampler coverage/determinism/sorted-within-batch; subset loaders over the real
Stage 3 dataset; trainer end-to-end fit, best-epoch selection by Install LogLoss, early
stopping, checkpoint round-trip. No test depends on the 3.5M-row run.

## 20. Stage 2 artifact integrity

`python scripts/verify_preprocessing.py --output_dir artifacts --data_dir .` —
**VERIFICATION PASSED** (all 12 checks):

```
[ok] artifact files present; hash c3a62caf13326617 verifies
[ok] split 'train': 3,485,852 rows across 90 part file(s)
[ok] split 'test': 160,973 rows across 4 part file(s)
[ok] row parity: train 3,485,852, test 160,973 (raw == processed)
```

The Stage 2 artifact (vocabularies, numerical stats, feature schema, config, manifest) was
not refit or modified by Stage 6; every processed part carries the artifact-hash linkage
metadata, which the Stage 3 dataset re-verifies at load time.

## 21. Raw data integrity

Raw files are read-only to the whole pipeline: `train/` (30 shards, 1.79 GB) and `test/`
(1 shard, 0.08 GB) are opened for reading only; no preprocessing, split, or training step
writes to them. Row parity raw == processed confirms no row was lost or duplicated.

---

## Stage 6 code inventory

Implemented under `src/recsys23_fedrec/training/`:

* `split.py` — deterministic, hashed, save/load-verified train/validation index split.
* `sampler.py` — `PartShuffledBatchSampler` (per-epoch seeded, sorted-within-batch) and
  `SortedBatchSampler` (validation order).
* `loaders.py` — `SubsetDataset` + `make_subset_loader` over the memory-bounded Stage 3
  dataset (no materialization; `num_workers = 0` and `= 2` both verified, including the
  Windows-spawn worker path; determinism of batch shapes verified).
* `metrics.py` — stable binary logloss + exact tie-aware ROC-AUC (no sklearn at runtime).
* `config.py` — serializable `TrainingConfig` (stored in checkpoints).
* `trainer.py` — training/validation loops, checkpointing on best validation Install
  LogLoss, early stopping (patience 2), epoch logging, `resolve_device`.

Runner: `scripts/run_centralized_training.py` (`--smoke` for the tiny run;
`--device`/`--epochs`/`--batch_size` overrides). Tests: `tests/training/`.

### Session fixes (2026-09-16)

1. **Smoke-run hazard fixed** (spec 15 protection): smoke mode previously shared the real
   checkpoint path, so a smoke trainer (best = ∞) would have overwritten the real best
   checkpoint. Smoke runs now write `artifacts/checkpoints/smoke_centralized_best.pt` (or
   under `--artifacts_dir`).
2. **`--device` override added** so verification work (e.g. the CPU smoke run) can execute
   while the GPU run owns the device.
3. **Validation logloss aggregation** changed from mean-of-batch-means to sample-weighted
   (removes sub-1e-5 bias from the final partial batch; the completed Stage 6 run used the
   prior aggregation — see §8).
4. Removed an unused `numpy` import from `trainer.py` after (3).

Full suite re-run after these changes: 139/139 passed.

### Tiny smoke run evidence (spec 14)

`--smoke` on the smoke artifacts (116,474 rows): 2 train batches + 1 validation batch,
batch 128, 1 epoch, CPU:

```
epoch 01 | 2.8s | train total 1.3844 (click 0.6925, install 0.6919)
          | val logloss click 0.6907 install 0.6893 | val AUC click 0.5132 install 0.4932
SMOKE RUN PASSED
```

Forward, backward, optimizer update, validation, checkpoint save, and reload all exercised;
all metrics finite (≈ log 2 / AUC ≈ 0.5 as expected for an untrained 2-batch run); no NaN/Inf.
