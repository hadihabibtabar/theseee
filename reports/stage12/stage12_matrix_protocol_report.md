# Stage 12 — Final Federated Evaluation Matrix: Protocol and Preparation Report

**Stage type:** protocol/evaluation preparation ONLY. No training was performed, no
protected artifact was modified, and the official test set was never loaded.

**Date:** 2026-09-21 · **Seed:** 42 (every stage) · **Status:** preparation complete,
four new federated cells ready to run.

---

## 1. Purpose

Stage 12 defines the final scientific evaluation matrix of the thesis so that three
factors can be studied on a shared, frozen protocol:

1. **Federated vs. centralized training** — federated cells are compared against the
   frozen Stage 8 centralized controls (A–D).
2. **Effect of SSL inside federated training** — SSL and no-SSL federated variants
   share everything except the local objective.
3. **Effect of Non-IID severity** — the same two variants are run at three Dirichlet
   concentration levels (alpha = 1.0, 0.5, 0.1).
4. **Stability of the alpha=0.5 SSL behavior** — whether the SSL-vs-no-SSL pattern
   observed at alpha=0.5 persists at milder (1.0) and harsher (0.1) heterogeneity.

All generated reports use **descriptive comparisons only**; no model is labelled a
"winner", "best", or "superior".

## 2. The complete 2 × 3 federated matrix

| Variant \ Non-IID severity | alpha = 1.0 (mild) | alpha = 0.5 (primary) | alpha = 0.1 (harsh) |
|---|---|---|---|
| **Transformer + MMoE (no SSL)** | NEW · `stage12_fed_nossl_alpha_1_0_seed42` | **EXISTING · Stage 10B (completed)** | NEW · `stage12_fed_nossl_alpha_0_1_seed42` |
| **Transformer + MMoE + SSL** | NEW · `stage12_fed_ssl_alpha_1_0_seed42` | **EXISTING · Stage 11 (completed)** | NEW · `stage12_fed_ssl_alpha_0_1_seed42` |

* The two existing cells (Stage 10B, Stage 11) are **NOT retrained**; their artifacts
  are guarded byte-identical by the Stage 12 runner fingerprint mechanism.
* The four new cells run under the **exact frozen protocol** of Stage 10A/10B/11,
  each over its own Stage 12 partition, in a **dedicated output directory**.

### Centralized controls (frozen in Stage 8; referenced, never retrained)

| ID | Description | Checkpoint |
|---|---|---|
| A | Transformer + supervised Install (centralized) | `artifacts/checkpoints/stage8/experiment_a_transformer_supervised_best.pt` |
| B | Transformer + MMoE supervised Click+Install (centralized) | `artifacts/checkpoints/centralized_transformer_mmoe_best.pt` |
| C | Transformer + SSL Install (centralized) | `artifacts/checkpoints/stage8/experiment_c_transformer_ssl_best.pt` |
| D | Transformer + MMoE + SSL Click+Install (centralized) | `artifacts/checkpoints/stage8/experiment_d_transformer_mmoe_ssl_best.pt` |

## 3. Exact frozen protocol (identical for every federated cell)

Validated at import time against the frozen Stage 10A `FederatedConfig`
(`src/recsys23_fedrec/federated/config.py`):

| Setting | Value |
|---|---|
| Simulated clients | 10 (`client_00` … `client_09`) |
| Participation | all clients, every round |
| Communication rounds | 10 |
| Local epochs per client per round | 1 |
| Local batch size | 1024 |
| Optimizer | AdamW, **fresh per client per round** (no state persistence) |
| Learning rate / weight decay | 3e-4 / 1e-2 |
| Supervised objective | BCE(click) + BCE(install), lambdas 1.0 / 1.0 |
| SSL objective (SSL cells only) | `0.6 * (BCE click + BCE install) + 0.4 * NT-Xent` |
| NT-Xent temperature | 0.2 (symmetric) |
| Augmentation | two independently corrupted views, feature corruption rate 0.15 |
| Projection head dim | 64, L2-normalized z vectors (training-time only) |
| Aggregation | sample-weighted FedAvg (`fedavg_state_dicts`, client order 00→09) |
| Initialization | cold seed-42; no Stage 6/8/10B/11 checkpoint warm start |
| Sampler | epoch = round − 1 on `loader.batch_sampler` only (`round_minus_one`) |
| Client rows | `frozen_train_indices[client_positions]` (never raw positions) |
| Validation | frozen 348,586-row global split after every round (single-view supervised path) |
| Checkpoint selection | best global-validation **Install LogLoss** |
| Official test | **NEVER loaded or evaluated** by any cell |
| Partition | Dirichlet over joint (click, install) label states, seed 42 |

## 4. Partition method and alpha definitions

Partitions are produced by the **unchanged Stage 9 implementation**
(`recsys23_fedrec.federated.partition`, method
`dirichlet_joint_label_proportional_v1`): for each joint label state
`(0,0), (0,1), (1,0), (1,1)` in fixed canonical order, rows are split across the 10
clients proportionally to one `Dirichlet(10, alpha)` draw per state
(largest-remainder integer correction; counts sum exactly to the state count);
chunks are cut contiguously in ascending dataset order with clients in id order.
One `numpy.random.default_rng(42)` (PCG64) is consumed in fixed order, so identical
seed + alpha + labels reproduce **byte-identical** partitions (verified).

* **alpha = 1.0** — milder Non-IID: client label distributions deviate moderately
  from the global distribution.
* **alpha = 0.5** — the frozen Stage 9 primary partition (owned by Stage 9/10B/11;
  NOT regenerated here).
* **alpha = 0.1** — harsher Non-IID: each joint state concentrates in few clients;
  extreme size and label skew is intentionally preserved (no rebalancing).

All partitions use the **same frozen centralized training indices**
(split train hash `dd37605169615442`, 3,137,266 rows) and exclude the frozen
validation split (hash `3cd8370ebdecb305`, 348,586 rows) and the official test set
by construction.

## 5. Stage 12 partition artifacts (generated, verified)

| Partition | File | Partition hash | Client sizes (rows) |
|---|---|---|---|
| alpha = 1.0, seed 42 | `artifacts/federated/partitions/stage12_partition_alpha_1_0_seed42.{npz,json}` | `c382d47f5824f64a` | 452,032 · 460,777 · 585,448 · 151,573 · 137,132 · 272,725 · 253,784 · 488,550 · 104,506 · 230,739 |
| alpha = 0.1, seed 42 | `artifacts/federated/partitions/stage12_partition_alpha_0_1_seed42.{npz,json}` | `23146aebf184345b` | 342,065 · 1,200,830 · 251,984 · 266,440 · 30,811 · 11,892 · 366,104 · 44,021 · 11,698 · 611,421 |

Both sums are exactly 3,137,266; clients are mutually disjoint; validation overlap is
zero; joint-label state counts sum exactly to the global counts; both partitions
re-derive byte-identically from the frozen Stage 9 code (verified twice, including a
full delete-and-regenerate round trip).

Client label skews (descriptive):

| Client | alpha 1.0 click / install | alpha 0.1 click / install |
|---|---|---|
| client_00 | 0.2184 / 0.0425 | 0.0490 / 0.0887 |
| client_01 | 0.1429 / 0.2301 | 0.0357 / 0.2348 |
| client_02 | 0.2578 / 0.1984 | 0.9278 / 0.1110 |
| client_03 | 0.6023 / 0.1924 | 0.0107 / 0.0107 |
| client_04 | 0.4539 / 0.5899 | 1.0000 / 0.0000 |
| client_05 | 0.1966 / 0.1233 | 0.0000 / 0.9834 |
| client_06 | 0.1735 / 0.0728 | 0.8650 / 0.5155 |
| client_07 | 0.0371 / 0.0592 | 0.9704 / 0.0026 |
| client_08 | 0.4533 / 0.7752 | 0.0502 / 0.0004 |
| client_09 | 0.2496 / 0.1424 | 0.0043 / 0.0043 |

Heterogeneity summary (mean absolute joint-proportion deviation from global):
alpha 1.0 → 0.1256, alpha 0.5 → (see Stage 9 report), alpha 0.1 → 0.2629 — the
severity ladder behaves as intended.

## 6. Train / validation / test separation

| Set | Rows | Used by Stage 12 |
|---|---|---|
| Centralized training rows | 3,137,266 (hash `dd37605169615442`) | partitioned into clients; the ONLY rows any client ever sees |
| Global validation rows | 348,586 (hash `3cd8370ebdecb305`) | per-round model selection only; identical for every cell; never trained on |
| Official test rows | 160,973 (raw `test/`, 88,684,100 bytes) | **never loaded, never constructed, never evaluated** |

## 7. Deterministic experiment IDs, outputs, and expected paths

ID convention: `stage12_fed_{ssl|nossl}_alpha_{tag}_seed42` with tag `1.0 → 1_0`,
`0.1 → 0_1`. Output convention: `artifacts/federated/stage12/<experiment_id>/`
containing `<experiment_id>_best.pt`, `<experiment_id>_final_round10.pt`,
`<experiment_id>_latest.pt`, `<experiment_id>_result.json`.

| Experiment ID | Status | Partition (hash) | Output directory |
|---|---|---|---|
| `stage12_fed_nossl_alpha_0_5_seed42` | existing (Stage 10B) | Stage 9 `9603ddc9facc973d` | `artifacts/federated/stage10` |
| `stage12_fed_ssl_alpha_0_5_seed42` | existing (Stage 11) | Stage 9 `9603ddc9facc973d` | `artifacts/federated/stage11` |
| `stage12_fed_nossl_alpha_1_0_seed42` | new | `c382d47f5824f64a` | `artifacts/federated/stage12/stage12_fed_nossl_alpha_1_0_seed42` |
| `stage12_fed_ssl_alpha_1_0_seed42` | new | `c382d47f5824f64a` | `artifacts/federated/stage12/stage12_fed_ssl_alpha_1_0_seed42` |
| `stage12_fed_nossl_alpha_0_1_seed42` | new | `23146aebf184345b` | `artifacts/federated/stage12/stage12_fed_nossl_alpha_0_1_seed42` |
| `stage12_fed_ssl_alpha_0_1_seed42` | new | `23146aebf184345b` | `artifacts/federated/stage12/stage12_fed_ssl_alpha_0_1_seed42` |

No Stage 12 path collides with any Stage 9/10B/11 artifact (verified by tests).

## 8. Implementation, validation, and execution boundary

**Implemented and validated in Stage 12 preparation:**

* `src/recsys23_fedrec/federated/stage12_matrix.py` — matrix registry, IDs, paths,
  protocol validation (import-time-checked), matrix snapshot writer.
* `scripts/run_stage12_partitions.py` — partition generation with integrity gates,
  re-run safety, and byte-reproducibility.
* `scripts/run_stage12_federated.py` — ONE generic runner for the four new cells;
  the local training functions, FedAvg, checkpointing, validation and resume
  machinery are imported **verbatim** from the frozen `run_stage10b.py` /
  `run_stage11.py` (no protocol reimplementation), parameterized only by the cell's
  model kind, partition, and output paths.
* `scripts/audit_stage12_matrix.py` — read-only audit (matrix, protocol, SSL
  equality with Stage 11, partition re-derivation, no-test-construction scan,
  protected-artifact values). Result: **ALL CHECKS PASSED**.
* `tests/federated/test_stage12_matrix.py` — 22 tests (see §9). Result: **22/22 passed**.
* `reports/stage12/stage12_matrix.json` — machine-readable matrix snapshot.

**NOT yet executed (explicitly):** the four new federated trainings. The runner is
preflight-verified for two representative cells (no-SSL alpha 1.0, SSL alpha 0.1),
but no Stage 12 training round has run and no Stage 12 checkpoint exists.

## 9. Verification summary

| Check | Result |
|---|---|
| Stage 12 tests (`tests/federated/test_stage12_matrix.py`) | 22 / 22 passed |
| Full pytest suite (all stages) | passed (see validation log; run 2026-09-21) |
| Partition audit (`scripts/audit_stage12_matrix.py`) | ALL CHECKS PASSED |
| alpha=1.0 / alpha=0.1 partition byte-reproducibility | verified (sha256 identical after full regeneration) |
| Protected artifacts (Stage 2/6/8/9/10B/11, alpha=0.5 partition, splits) | byte-identical before/after all Stage 12 preparation |
| Official test construction scan | clean (no Stage 12 file references the test split) |

## 10. Launch commands for the four new experiments (to run later)

```powershell
python scripts/run_stage12_federated.py --experiment stage12_fed_nossl_alpha_1_0_seed42
python scripts/run_stage12_federated.py --experiment stage12_fed_ssl_alpha_1_0_seed42
python scripts/run_stage12_federated.py --experiment stage12_fed_nossl_alpha_0_1_seed42
python scripts/run_stage12_federated.py --experiment stage12_fed_ssl_alpha_0_1_seed42
```

Each supports `--preflight_only` (integrity + data checks only), `--resume`
(from the cell's `<experiment_id>_latest.pt`), and `--allow_restart` (discard that
cell's own Stage 12 outputs). Stage 10B/11 outputs can never be touched by this
runner: existing cells are refused outright and every file they guard is
fingerprint-verified before and after each run.
