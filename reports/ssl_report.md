# Stage 7 Report — Self-Supervised Contrastive Learning Mechanism

**Status: COMPLETE — SSL mechanism implemented, tested, and smoke-verified. Stage 6 remains
frozen and untouched.** No claim is made that SSL improves performance: this stage validates
the mechanism only. The controlled SSL experiments (pretraining vs fine-tuning) are Stage 8.

> This stage adds a self-supervised contrastive path. No federated clients, Dirichlet
> partitioning, FedAvg/FedProx, or Flower components were implemented. The Stage 6 baseline
> checkpoint, split, preprocessing artifact, and raw data are untouched.

---

## 1. Stage 7 objective

Add a clean, modular SSL contrastive-learning path based on the Transformer CLS
representation:

* feature-corruption augmentation producing two independent views per impression,
* NT-Xent / InfoNCE contrastive loss over L2-normalized projections,
* a projection head on the CLS path (add-on; MMoE keeps the original 128-d CLS),
* a joint objective `L = α·L_supervised + (1-α)·L_contrastive` with α = 0.6,

implemented by composition over the frozen Stage 4/5/6 architecture (no architecture
rewrite), with full test coverage and a tiny real-data smoke verification.

## 2–4. Augmentation definition (`FeatureCorruptionAugmentation`)

Module: `src/recsys23_fedrec/ssl/augmentations.py`. Operates on the **already-preprocessed**
Stage 2/3 representation; raw CSVs, preprocessing artifacts, and vocabularies are never
touched or refit.

**Corruption rate: 0.15** (per-entry Bernoulli, independently sampled per view).

| Feature group | Corruption | Neutral value | Notes |
| --- | --- | --- | --- |
| categorical [30] | replace token with UNK | **index 0** | existing UNK representation; always in-vocabulary |
| numerical [38] | replace value | **0.0** | zero in the already-standardized space; no new normalization |
| binary [11] | replace value | **0.0** | neutral consistent with model input representation |
| missing [13] | **preserved** (default) | — | semantic meaning ("was missing") kept; opt-in corruption to 0.0 via `corrupt_missing=True` |

Documented behaviors:

* **Labels never altered**: `click`/`install` tensors pass through unchanged; both views
  share the original impression's labels (verified by tests).
* **No in-place mutation**: outputs are fresh tensors (`torch.where`), so the original
  batch stays intact for the supervised loss (verified by tests).
* **Independent views**: view 1 and view 2 draw independent Bernoulli masks; for the same
  input, `augment(x)` normally yields two different views (verified by tests).
* **Single seed convention**: `preprocessing.config.SEED = 42`. Views are drawn from
  per-call CPU `torch.Generator` streams (`SEED + 1_000_003·step`), never ambient global
  RNG state; masks are sampled on CPU and moved to the batch device, so CPU/CUDA runs with
  the same seed produce identical views (verified by tests).

## 5. Projection head

`Linear(128 → 128) → GELU → Linear(128 → 64)`, then **L2 normalization** at the loss
boundary. Output dimension **64**. It exists only for contrastive learning; the downstream
MMoE heads continue to receive the original 128-dimensional CLS representation (verified:
SSL wrapper in eval mode produces bit-identical logits to the pure Stage 5 model).

## 6–8. Contrastive loss (`nt_xent_loss`)

Module: `src/recsys23_fedrec/ssl/contrastive_loss.py`. Symmetric NT-Xent / InfoNCE
(SimCLR form) over the 2B views of a batch of B impressions:

```
sim(z_i, z_j) = z_i · z_j            (projections L2-normalized → dot = cosine)
l(i, p(i))    = -log( exp(sim(z_i, z_{p(i)}) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) )
L_NT-Xent     = (1/2B) · Σ_i l(i, p(i)),   p(i) = (i + B) mod 2B
```

* **Temperature: τ = 0.2** (Stage 7 spec default).
* **Positive**: the other augmented view of the same impression (`(i + B) mod 2B`).
* **Negatives**: all other views in the 2B batch; the view itself is **excluded** (diagonal
  masked to `-inf` before a stable `log_softmax`; no manual exp/log, no overflow).
* **Symmetric**: averaged over all 2B rows; loss is invariant to view order (tested).
* **Label-free**: the loss never receives Click/Install labels (tested by flipping labels).
* Numerically stable: no NaN/Inf for random, scaled (×100, ×0.001), and degenerate inputs;
  verified against independent loop-based reference implementations (rel. tol ≤ 1e-5).

## 9–10. Joint loss (`ssl_joint_loss`)

Module: `src/recsys23_fedrec/ssl/joint_loss.py`.

```
L_total = α · L_supervised + (1 − α) · L_contrastive,        α = 0.6
L_supervised = λ_click · BCEWithLogits(click_logit, click)
             + λ_install · BCEWithLogits(install_logit, install)     (λ = 1.0)
```

* Supervised part is exactly the Stage 5 `mmoe_loss` (raw logits, no sigmoid, no class
  weighting). In dual-view training the supervised term is the mean over both views.
* Contrastive part is label-free; α ∈ [0,1] validated (α=1 → pure supervised, α=0 → pure
  contrastive; tested).

## 11. Model integration (`SSLTransformerMMoE`)

Module: `src/recsys23_fedrec/models/ssl_transformer_mmoe.py`. Composition over the frozen
Stage 5 model (backbone, experts, gates, towers reused verbatim; `head_from_cls` extracted
in `mmoe.py` so the head can run from a precomputed CLS without duplicating Stage 5 logic —
`TransformerMMoE.forward` delegates to it, behavior unchanged).

```
original batch x ──► augment ──► view1 ─┐
                 └─► augment ──► view2 ─┤   (independent corruption masks)
                                        ▼
        tokenized together as [2B, 92, d] → +CLS → Transformer encoder
                                        ▼
        split → CLS h(1) [B,128], h(2) [B,128]     (independent rows)
          ├─────────────────► MMoE → click_logit [B], install_logit [B]   (per view)
          └─► projection MLP → L2 → z1 [B,64], z2 [B,64]
```

Model outputs: `z1`, `z2` (normalized projections), `view1`, `view2` (dicts with raw task
logits), optionally `cls_view1`/`cls_view2`. `supervised_forward(batch)` is the exact
Stage 6 path (no augmentation, no projection). Parameters: **2,527,698**
(Stage 6 model 2,502,930 + projection head 24,768). The single encoder pass over both views
avoids duplicate Transformer calls while keeping each view an independent row.

## 12. Test results

New suite `tests/ssl/` — **62 tests, all passing**:

| File | Tests | Coverage |
| --- | --- | --- |
| `test_augmentations.py` | 21 | shape, non-mutation, rate ≈ 0.15 on 4096-sample batch, UNK validity, neutral values, missing preserved, labels unchanged, two views differ, determinism under seed 42, explicit generator, CPU, CUDA, CPU/CUDA parity, invalid rate |
| `test_contrastive_loss.py` | 23 | B∈{2,4,8,32}, dims∈{1,3,64,128}, hand-computed values (manual reference, log 3 exact case, degenerate-vector exact case), positive indexing (aligned < shuffled), self-exclusion, symmetry/order invariance, finiteness ×20, no overflow at τ=0.05, non-zero finite gradients, gradient symmetry, temperature behavior, unnormalized input, CUDA parity, invalid inputs |
| `test_ssl_model.py` | 18 | z shape [B,64] + unit norm, CLS remains 32-d (fixture) / 128-d (full), logit shapes, joint loss finite + exact decomposition, α extremes, label-free contrastive, gradients through Transformer/MMoE/projection, optimizer updates params, views differ through model, no sigmoid squashing, Stage 6 equivalence, state-dict round-trip, CUDA forward/backward |

**Full regression suite: 201 passed** (139 previous Stages 1–6 + 62 new; zero regressions,
no existing test weakened or deleted).

## 13. Smoke-run results (real data, CUDA)

`python scripts/run_ssl_smoke.py` — real Stage 2 Parquet through the memory-bounded Stage 3
loader (3,485,852 rows, 90 parts), frozen architecture, batch 128, 2 steps, seed 42:

```
device: cuda (NVIDIA GeForce RTX 4060 Ti)
SSL model parameters: 2,527,698
step 0: joint 2.6215 (supervised 1.3860 [click 0.6932 / install 0.6928] | contrastive 4.4748) | finite=True
step 1: joint 2.4521 (supervised 1.3838 [click 0.6924 / install 0.6914] | contrastive 4.0544) | finite=True
parameters updated by optimizer: 142
  projection: updated | mmoe.backbone: updated | mmoe.experts: updated | mmoe.towers: updated
projection norms: mean 1.0000
peak GPU allocated: 950 MiB
STAGE 7 SSL SMOKE RUN PASSED in 6.0s
```

All 10 spec-7.9 items verified: real batch load → view1/view2 → both through encoder →
z1/z2 → contrastive/supervised/joint losses finite → backward → optimizer step → parameters
changed (142 of 142 trainable) → no NaN/Inf. (Initial untrained supervised losses ≈ log 2 =
0.693 per task, as expected; contrastive already decreasing across the two steps.)

## 14–15. Gradient & parameter-update verification

* **Gradients** reach the projection head (all 4 tensors), the Transformer backbone, and
  the MMoE experts/gates/towers (test asserts non-zero finite grads per component).
* **Optimizer updates** 142/142 trainable parameters in the smoke run, including the
  projection head and the frozen-architecture backbone components.

## 16. CPU / CUDA verification

* CPU: all 62 SSL tests run on CPU; augmentation, loss, and model paths verified.
* CUDA (RTX 4060 Ti, torch 2.7.0+cu118): dedicated tests for augmentation on device,
  contrastive loss parity (rel. tol 1e-5 vs CPU), model forward/backward on device; smoke
  run executed on CUDA. Mask sampling is CPU-side, so seeded views are identical across
  devices (tested).

## 17. Memory observations

* Data path unchanged: Stage 3 row-group streaming (≤ one ~50k-row group in RAM);
  no materialization of the 3.5M-row dataset anywhere in Stage 7.
* Smoke run (batch 128, full-size model): peak GPU allocated **950 MiB**; dual-view forward
  doubles only the batch dimension of the encoder pass (2B rows), which is bounded.
* Stage 6 training memory (RSS 845 MiB peak) applies unchanged to any Stage 8 reuse of this
  loader.

## 18. Integrity confirmations

| Item | Before Stage 7 | After Stage 7 |
| --- | --- | --- |
| Stage 6 checkpoint sha256 | `25f3267eba572bf99e72c19eee2351236fe65327f71af7c7a8e39fae8e1d8617` | **identical** |
| Stage 6 checkpoint mtime | 1789563754.0476708 (2026-09-16 16:32:34) | **identical** |
| Stage 2 artifact hash | `c3a62caf13326617` | **identical** |
| Raw train bytes (30 shards) | 1,919,412,753 | **identical** |
| Raw test bytes (1 shard) | 88,684,100 | **identical** |

No checkpoint was written by any Stage 7 code path (the smoke script saves nothing).

## 19. Limitations / design decisions

* **Missing indicators preserved** during corruption: corrupting them would create
  inconsistent views (a categorical replaced by UNK implies "missing", which the indicator
  would deny). Opt-in flag exists (`corrupt_missing`) but defaults off; documented for
  Stage 8 ablation.
* **Views share labels** by construction (corruption cannot touch labels), so both views
  contribute supervised signal for the same impression.
* **Dual-view supervised loss** is the mean over the two views; the alternative (supervised
  on the uncorrupted batch only) is intentionally deferred to Stage 8 experiment design.
* **Projection head is discarded at evaluation/fine-tuning time**: `supervised_forward` and
  the MMoE heads never consume projections, matching the standard SSL literature setup.
* **One encoder pass over 2B rows** (views stacked) is exactly equivalent to two separate
  passes for this architecture (per-row LayerNorm/attention; no cross-sample mixing) and
  halves kernel-launch overhead; correctness over micro-optimization was prioritized
  elsewhere.
* **Single seed convention retained**: no new SSL seed system; augmentation streams derive
  from `SEED = 42`.
* Not done here (by design): full SSL pretraining runs, Stage 8 experiment matrix, α/τ
  sweeps, federated anything.

---

### Files created/modified (Stage 7)

| File | Change |
| --- | --- |
| `src/recsys23_fedrec/ssl/__init__.py` | **new** — public SSL API |
| `src/recsys23_fedrec/ssl/augmentations.py` | **new** — feature-corruption augmentation |
| `src/recsys23_fedrec/ssl/contrastive_loss.py` | **new** — NT-Xent / InfoNCE |
| `src/recsys23_fedrec/ssl/joint_loss.py` | **new** — joint objective (α = 0.6) |
| `src/recsys23_fedrec/models/ssl_transformer_mmoe.py` | **new** — SSL model (projection head + dual-view forward) |
| `src/recsys23_fedrec/models/mmoe.py` | modified — `head_from_cls` extracted; `forward` now delegates to it (behavior verified identical by tests) |
| `src/recsys23_fedrec/models/__init__.py` | modified — exports `SSLTransformerMMoE`, `SSLConfig` |
| `tests/ssl/{__init__,conftest,test_augmentations,test_contrastive_loss,test_ssl_model}.py` | **new** — 62 tests |
| `scripts/run_ssl_smoke.py` | **new** — smoke verification script |
| `reports/ssl_report.md` | **new** — this report |
