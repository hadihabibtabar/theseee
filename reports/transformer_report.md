# Stage 4 Report — Transformer Baseline

Single-task (is_installed) Transformer over per-feature tokens, consuming Stage 3 batches. No MMoE, no SSL, no click head, no training pipeline. Stage 2 artifact and raw data untouched.

## Files created/modified

Created:
- `src/recsys23_fedrec/models/__init__.py` — public API
- `src/recsys23_fedrec/models/transformer.py` — `TransformerBaseline`, `TransformerConfig`, `CategoricalIdError`, `install_loss`
- `tests/models/__init__.py`, `tests/models/conftest.py`, `tests/models/test_transformer.py` — 26 tests
- `scripts/test_transformer.py` — real-data test (spec sections A–I, 13–15)
- `reports/transformer_report.md` — this file

## 1. Architecture summary

```
Stage 3 batch {categorical[30], numerical[38], binary[11], missing[13]}
  |
  |-- 30 x per-feature nn.Embedding (vocab from Stage 2 artifact, own table each)
  |-- 38 x per-feature affine token: x_j * w_j + b_j,  w_j,b_j in R^128
  |-- 11 x per-feature affine token (binary)
  |-- 13 x per-feature affine token (missing indicator)
  v
92 feature tokens [B, 92, 128]
  + learnable CLS token (prepend)          -> [B, 93, 128]
  + learnable positional embedding [1,93,128]
  v
6-layer nn.TransformerEncoder (8 heads, ffn=128, GELU, dropout 0.1, batch_first)
  v
[B, 93, 128]  ->  CLS = [:, 0, :]  ->  [B, 128]
  v
head: Linear(128->128) -> GELU -> Linear(128->1) -> squeeze(-1)
  v
{"install_logit": [B], "cls": [B, 128]}     (raw logit, NO sigmoid)
```

- `install_loss(output, batch)` = `BCEWithLogitsLoss()(install_logit, install)` — the only loss helper; click/SSL losses intentionally absent.
- `cls` is returned for MMoE/SSL reuse in later stages.
- No preprocessing/normalization/encoding inside the model; no `.to(device)` in the data path.

## 2. Feature-token breakdown (all derived from the Stage 2 schema, nothing hard-coded)

| group | count | token mechanism | shapes |
|---|---|---|---|
| categorical | 30 | one `nn.Embedding(vocab_j, 128)` per feature | [B, 30, 128] |
| numerical | 38 | per-feature affine `x_j*w_j + b_j` | [B, 38, 128] |
| binary | 11 | per-feature affine | [B, 11, 128] |
| missing | 13 | per-feature affine | [B, 13, 128] |
| **features total** | **92** | | [B, 92, 128] |
| CLS | 1 | learnable parameter | [B, 1, 128] |
| **sequence** | **93** | + learnable positional embedding | [B, 93, 128] |

## 3. Tensor shape flow (verified in tests + real data)

```
categorical tokens  [B, 30, 128]
numerical tokens    [B, 38, 128]
binary tokens       [B, 11, 128]
missing tokens      [B, 13, 128]
combined            [B, 92, 128]
with CLS            [B, 93, 128]
encoder output      [B, 93, 128]
CLS output          [B, 128]
install_logit       [B]
```

## 4. Hyperparameters

`d_model=128, nhead=8, num_layers=6, dim_feedforward=128, dropout=0.1` (head dim 16). All configurable via `TransformerBaseline(...)` / frozen `TransformerConfig`; validation rejects `d_model % nhead != 0`, non-positive values, dropout outside [0,1). Encoder activation GELU, `batch_first=True`, standard PyTorch components.

## 5. Parameter count

| component | parameters |
|---|---|
| categorical embeddings (30 tables) | 1,710,080 |
| numerical projections (38 x 2x128) | 9,728 |
| binary projections (11 x 2x128) | 2,816 |
| missing projections (13 x 2x128) | 3,328 |
| CLS + positional (129 x 128 + 128) | 12,032 |
| Transformer encoder (6 layers) | 597,504 |
| install head (128x128 + 128 + 128 + 1) | 16,641 |
| **total** | **2,352,129** |
| **trainable** | **2,352,129** |

## 6. Real-data forward-pass results (Stage 3 DataLoader, batch sizes 32 and 1024)

All shapes exactly as in section 3 at both batch sizes; `install_logit` and `cls` finite; intermediate token shapes verified via the `tokenize()`/`return_tokens` debug path. Dataset opened metadata-only (90 parts, 3,485,852 rows; RSS 393.7 -> 395.9 MiB).

## 7. BCEWithLogitsLoss result

Finite scalar losses: 0.6930 (B=32), 0.6960 (B=1024) — as expected near `ln(2)` for an untrained model with no class weighting.

## 8. Backward / gradient result

Real batch B=1024: gradients exist for **all 114 trainable parameter tensors**, all finite (no NaN/Inf), zero parameters with exactly-zero gradient, backward time 1.06 s (CPU).

## 9. Tiny optimization sanity result (NOT training)

3 steps of AdamW(lr=1e-3) on real batches: losses [0.6960, 0.6420, 0.6068] — finite and decreasing; head parameters changed after `optimizer.step()`. Explicitly not evidence of model quality.

## 10. CPU result

All unit tests and the full real-data script run on CPU. Outputs stay on CPU unless the caller explicitly moves the model (`test_output_stays_on_cpu`).

## 11. CUDA result

**Executed** (not skipped): NVIDIA GeForce RTX 4060 Ti. Forward on CUDA produces finite `install_logit`; backward produces finite gradients; model moved back to CPU afterwards. Data transfer to GPU is performed by the test script only — the model/data layers remain device-agnostic.

## 12. Test counts

- Stage 4 unit tests (`tests/models/test_transformer.py`): **26 passed** — construction from real schema (token counts 30/38/11/13, 92, 93), embedding sizes == Stage 2 vocabularies (max 5803, min 3), default + configurable hyperparameters, schema-derived sequence length, forward contracts (incl. batch size 1 and label-free test-split batch), intermediate shapes, CLS prepending, positional embedding application, loss (incl. equivalence with direct `BCEWithLogitsLoss`), gradient presence/finiteness, per-feature gradient semantics, tiny optimization sanity, CPU discipline, conditional CUDA, deterministic construction under seed 42, dropout wiring, categorical ID contract (out-of-range names the feature, negative IDs, wrong column count, boundary IDs valid).
- Full suite: **78 passed** (25 Stage 2 + 27 Stage 3 + 26 Stage 4) in ~12 s.

## 13. Stage 1–3 regression-test results

- `pytest tests/` → 78/78 passed; no regression.
- Stage 2 artifact hash re-verified after Stage 4 work: **`c3a62caf13326617`** (unchanged).
- Stage 3 data-layer contract unchanged (the model consumes Stage 3 batches as-is; no compatibility fix was needed).

## 14. Raw data / artifact confirmation

- Raw CSV byte totals identical to Stage 1: train 1,919,412,753 B; test 88,684,100 B.
- Official test split opened **only** to confirm schema compatibility (length 160,973 + one label-free forward); never fitted, never tuned on.
- No preprocessing behavior or artifact format modified.

## Issues discovered and fixed during this stage

1. **Memory probe artifact (script bug, not model bug):** the first memory-sanity loop kept autograd graphs alive across batches (~18 GiB RSS growth). Re-run under `torch.no_grad()` + `model.eval()`: RSS delta over 25 forward batches is +410 MiB on a torch-heavy baseline process, peak 1.94 GiB, plateauing — bounded as required.
2. **Determinism check compared against a mutated model:** section F's optimizer steps had changed `model` before section I compared parameters. Fixed to compare two fresh seed-42 constructions (also covered by a dedicated unit test).
3. **Device discipline in the script:** the test-split check initially fed a CPU batch to the CUDA-resident model; fixed by moving the model back to CPU first. The model itself correctly raised the PyTorch device-mismatch error.
4. **Zero-gradient weight rows are legitimate:** a per-feature weight row whose input column is identically 0 in a batch (zero-variance numerical feature standardized to 0) receives zero weight gradient while its bias still learns; the per-feature gradient test accounts for this instead of asserting incorrectly.

Stopped after Stage 4 — no MMoE, click head, SSL, training pipeline, validation strategy, federated clients, FedAvg, Flower, or experiment runner was implemented.
