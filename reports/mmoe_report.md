# Stage 5 Report — Transformer + MMoE (Click + Install)

Two-task MMoE model on the frozen Stage 4 Transformer backbone. No SSL, no federated learning, no training pipeline. Scientific note: this reproduces the known multi-task MMoE architecture concept (Ma et al., KDD 2018) as a baseline for later controlled experiments — no novelty claim, and no model-quality comparison is performed at this stage.

## Files created/modified

Created:
- `src/recsys23_fedrec/models/mmoe.py` — `TransformerMMoE`, `MMoEConfig`, `mmoe_loss`
- `tests/models/test_mmoe.py` — 25 tests
- `scripts/test_mmoe.py` — real-data test (spec 12–22)
- `reports/mmoe_report.md` — this file

Modified (additive only, Stage 4 semantics unchanged):
- `src/recsys23_fedrec/models/transformer.py` — optional `include_install_head` (default True) and a public `encode()` reuse path so MMoE reuses tokenization + CLS + positional embedding + encoder verbatim instead of duplicating it
- `src/recsys23_fedrec/models/__init__.py` — re-exports

## 1–2. Architecture

```
Stage 3 batch {categorical[30], numerical[38], binary[11], missing[13]}
        |            (Stage 4 backbone, reused verbatim)
  92 feature tokens + CLS + positional embedding
        v
  [B, 93, 128]  ->  6-layer TransformerEncoder (8 heads, ffn 128, dropout 0.1)
        v
  CLS [B, 128]
        v
  +---------------------------+
  | 8 shared experts          |   each: Linear(128->128) -> GELU -> Dropout(0.1)
  |   -> [B, 8, 128]          |
  +---------------------------+
        |                   |
  Click gate            Install gate      each: Linear(128->8) -> softmax(dim=-1)
    [B, 8]                 [B, 8]
        v                   v
  einsum weighted aggregation (per sample):  task_repr = sum_i G[i] * E_i
   click_repr [B,128]   install_repr [B,128]
        v                   v
  Click tower          Install tower    each: Linear(128->128) -> GELU
        |                   |                 -> Dropout(0.1) -> Linear(128->1)
   click_logit [B]     install_logit [B]     (raw logits, NO sigmoid)
```

## 3. Number of experts
8 shared experts (one set, not per-task) — verified structurally and behaviorally (perturbing one expert changes both task representations).

## 4. Expert architecture
`Linear(128, 128) -> GELU -> Dropout(0.1)`, configurable via `MMoEConfig(num_experts, num_tasks, dropout)`.

## 5. Gate architecture
One `Linear(128 -> 8)` per task on the shared CLS; softmax over the expert dimension. Gates are sample-specific functions of the input (verified: distinct CLS inputs produce distinct gate distributions; on real data the mean per-sample click-gate max-min spread is 0.081–0.085 and the mean |click_gate − install_gate| is 0.030).

## 6. Gate normalization
`softmax(..., dim=-1)`; every row sums to 1 within 1e-6 and weights lie in [0, 1] — tested per sample, not batch-averaged.

## 7. Task tower architecture
Independent towers (no parameter sharing): `Linear(128->128) -> GELU -> Dropout(0.1) -> Linear(128->1)`, raw logit outputs.

## 8. Tensor shape flow (verified on real data at B=32 and B=1024)

```
cls             [B, 128]
expert_outputs  [B, 8, 128]
click_gate      [B, 8]        install_gate      [B, 8]
click_repr      [B, 128]      install_repr      [B, 128]
click_logit     [B]           install_logit     [B]
encoder sequence [B, 93, 128] (before CLS extraction)
```

## 9. Loss equations

```
L_click   = BCEWithLogitsLoss(click_logit, batch["click"])
L_install = BCEWithLogitsLoss(install_logit, batch["install"])
L_total   = lambda_click * L_click + lambda_install * L_install      (defaults 1.0 / 1.0)
```

No sigmoid before the loss, no class weighting, no focal loss, no special install treatment. `mmoe_loss` returns `{"click", "install", "total"}` for observability.

## 10. Parameter count and breakdown

| component | parameters |
|---|---|
| Transformer/tokenization (backbone, headless) | 2,335,488 |
| MMoE experts (8 × (128·128 + 128)) | 132,096 |
| gates (2 × (128·8 + 8)) | 2,064 |
| click tower | 16,641 |
| install tower | 16,641 |
| **total** | **2,502,930** |
| **trainable** | **2,502,930** |

Breakdown sums exactly to the total. For later federated communication-cost analysis: the added multi-task head is 167,442 parameters on top of the headless backbone (Stage 4 baseline with head: 2,352,129).

## 11. CPU test
Full forward/backward on CPU: finite logits/losses, gradients on all 138 parameter tensors (0 zero-gradient), backward 1.53 s at B=1024.

## 12. CUDA test
Executed on NVIDIA GeForce RTX 4060 Ti: model + real batch moved to CUDA, finite click/install logits, finite loss (1.3403), backward succeeds with finite gradients. Device transfer performed by the test script only.

## 13. Real-data forward/backward results
Stage 3 DataLoader over the processed Parquet (3,485,852 rows, 90 parts; dataset construction metadata-only, RSS 394.7 → 396.9 MiB). At B=32 and B=1024: all shapes exact, all outputs finite, gate rows sum to 1 per sample, losses finite (click ≈ 0.6932, install ≈ 0.6928, total ≈ 1.3860 — expected ~2·ln 2 for an untrained model).

## 14. Gate sanity results
Gates are functions of the input: distinct synthetic CLS inputs yield distinct gate distributions for both tasks; per-sample spread on real data is non-degenerate (see section 5). Gate weights always within [0, 1], rows sum to 1.

## 15. Gradient results
`L_total.backward()` on a real B=1024 batch: gradients exist and are finite for all 138 parameter tensors; zero exactly-zero-gradient parameters; spot-checks confirm non-zero gradients for backbone CLS, shared experts, both gates, both towers. A decomposition test shows experts receive differing gradients from click-only vs install-only losses (consistent with task-specific routing), while both tasks demonstrably route through the same shared experts.

## 16. Tiny optimization sanity (3 steps, NOT training)
AdamW(1e-3) on real batches: (click, install, total) losses [(0.6933, 0.6927, 1.3860) → (0.6901, 0.6882, 1.3783) → (0.6833, 0.6781, 1.3614)] — finite, decreasing; experts and gates change. Explicitly not evidence of model quality.

## 17. Full regression test count
**103 passed** (25 Stage 2 + 27 Stage 3 + 26 Stage 4 + 25 Stage 5) in ~21 s. Previous total was 78; no regressions.

## 18. Stage 2 artifact integrity
Hash re-verified after all Stage 5 changes: **`c3a62caf13326617`** (unchanged). Vocabulary sizes in the model identical to the artifact.

## 19. Raw-data integrity
Raw CSV byte totals identical to Stage 1: train **1,919,412,753** B, test **88,684,100** B. The official test split was opened only to confirm schema compatibility (length 160,973 + one label-free forward); never trained on or tuned on.

## 20. Explicit statement
SSL and Federated Learning are **not** implemented in this stage. No centralized training loop, validation split, Dirichlet partitioning, FedAvg, Flower, experiment runner, or hyperparameter search exists yet.

## Issues found and fixed during this stage
1. **Additive Stage 4 refactor:** the `encode()` reuse path initially left `_init_parameters` indexing `self.head` when `include_install_head=False`; guarded. Stage 4 default construction/behavior verified unchanged (dedicated test: seeded Stage 4 builds remain identical, install head present).
2. **Encoder null-space test artifact:** perturbing an expert's weight additively (+1.0 all-ones) changed click_repr by only ~1.3e-7 because the encoder's final LayerNorm makes `sum(cls) ≈ 0`, putting the perturbation in the linear layer's null space. Fixed with a multiplicative perturbation (×3) — both task representations then change, correctly proving shared routing.
3. **CUDA test fixture IDs:** `randint(0, 4)` violated the real f_7 vocabulary (size 3) — the model's `CategoricalIdError` correctly caught it; fixture now uses IDs {0, 1}, valid for every real vocabulary.
4. **Sequence-length test:** the `[B, 93, 128]` assertion was hard-coded to the real schema; on the smaller synthetic fixture it must be the derived `1 + num_feature_tokens` (25). The real 93 is verified in `test_construction_from_real_schema` and the real-data script.

Stopped after Stage 5 — no SSL, centralized training, validation splitting, federated clients, FedAvg, Flower, experiment runner, or hyperparameter search.
