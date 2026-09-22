# Stage 10A — Federated Learning Protocol Design & Audit Report

Status: **PROTOCOL DESIGN AND AGGREGATION UTILITY ONLY — no federated
training, no FedAvg round, no real training loop, no optimizer, and no
backward pass had been executed at audit time.** Stage 10B (the actual
federated run) is a separate, later decision. Sections 1–14 describe the
frozen Stage 10A state; §15 records the subsequent ONE-ROUND integration
smoke test, which is not Stage 10B.

## 1. Scope of this stage

Stage 10A formalizes and freezes the Stage 10 baseline federated protocol,
implements the aggregation mathematics as a standalone tested utility, and
audits every protocol assumption against the actual repository artifacts.
Nothing was connected to a training loop.

## 2. Stage 10 baseline protocol (frozen)

| Element | Value |
|---|---|
| Clients | exactly 10 simulated federated clients (`client_00` … `client_09`) |
| Data partition | Stage 9 frozen partition, Dirichlet α = 0.5, seed 42, hash `9603ddc9facc973d` |
| Participation | **all 10 clients in every round** (`all_clients_every_round`) |
| Model | existing frozen Transformer + MMoE (Stage 5/6 architecture, 2,502,930 params) |
| Local objective | `BCE(click) + BCE(install)`, λ_click = λ_install = 1.0, raw logits + BCEWithLogitsLoss |
| SSL | **none** in the Stage 10 baseline |
| Local batch size | 1024 (identical to centralized protocol) |
| Local epochs | **1** per client per round |
| FedAvg rounds | **10** |
| Local optimizer | AdamW, lr 3e-4, weight decay 1e-2 — **fresh per client per round** |
| Aggregation | sample-weighted FedAvg |
| Validation | existing frozen global validation split (348,586 rows) only |
| Selection metric | global validation Install LogLoss |
| Class weighting / focal / oversampling / schedulers / HPO | none |
| Official test set | untouched, never evaluated |

Configuration object: `FederatedConfig`
(`src/recsys23_fedrec/federated/config.py`) with strict validation that
rejects any deviation from the frozen baseline (client count, local epochs,
participation policy, aggregation method, SSL flag, selection metric are
all locked). Serialization follows the existing
`to_dict`/`from_dict` convention so the full protocol can be embedded in
future checkpoints — no parallel configuration system was introduced.

## 3. Client participation policy

All 10 clients participate in every round (baseline). Consequently the
FedAvg weights are **constant across rounds** and equal the client data
fractions of the frozen partition (audited values in §7). Client sampling
(partial participation) is explicitly out of scope for this baseline.

## 4. Global/local model lifecycle (intended, not yet executed)

For round `r = 1..10`:

1. one global model state `θ_global^(r)` exists at the start of the round;
2. **every participating client starts from that exact global state**;
3. each client performs **exactly one local epoch** over its own rows using
   the Stage 9 client loader (`make_client_loader`, batch 1024);
4. the local optimizer is **created fresh per client per round**
   (`local_optimizer_state_persistence = False`). Rationale: AdamW
   moment estimates are client-local transient state; persisting them
   across rounds would entangle clients' adaptive statistics with the
   aggregation history and complicate attribution. This is a documented,
   reversible baseline decision — a persistent-optimizer variant would be a
   new protocol version;
5. client states are aggregated by sample-weighted FedAvg into
   `θ_global^(r+1)`;
6. after each round, the global model is evaluated on the frozen global
   validation split; **the primary selection metric is global validation
   Install LogLoss**; the best-round global state is the Stage 10 artifact.

Initialization decision (frozen for the baseline): `θ_global^(0)` is a
**fresh, seed-42-initialized TransformerMMoE** — NOT the Stage 6 checkpoint.
The federated baseline must measure what FedAvg itself achieves under the
federated data distribution; initializing from the centralized Stage 6
result would confound the federated measurement with centralized
pretraining. Loading Stage 6 weights is recorded as an alternative
protocol variant if a "warm-start federated" comparison is ever desired.

## 5. FedAvg definition (implemented + tested)

    θ_global_next = Σ_k (n_k / Σ_j n_j) · θ_client_k

with `n_k` = training samples assigned to client k by the frozen Stage 9
partition. Implementation: `fedavg_state_dicts`
(`src/recsys23_fedrec/federated/fedavg.py`), a standalone pure-tensor
utility with:

* compatibility validation: identical key sets, shapes, dtypes across all
  clients (missing/extra keys, shape or dtype mismatch → `FedAvgError`
  before any arithmetic);
* floating tensors: weighted average accumulated in **float64** and cast
  back to the source dtype (protects 10-client accumulation);
* non-floating tensors: **rejected by default** (`on_non_floating="error"`)
  — the frozen architecture has none, so such an entry indicates a foreign
  state dict; `"first"` is available for future architectures with
  legitimate counter buffers (e.g. BatchNorm `num_batches_tracked`), where
  the conventional policy is client 0's value;
* zero/negative/missing sample counts → `FedAvgError`;
* **inputs are never mutated** (freshly allocated output; verified by
  tests including data-pointer checks);
* determinism: accumulation in **client order (client_00 first)** makes
  repeated aggregation bit-exact; identical inputs to all clients
  reproduce the input exactly (identity property verified on the real
  client weights).

## 6. Model state aggregation policy (verified from the implementation)

Inspection of the actual `TransformerMMoE` (not assumed):

* real-artifact state dict: **138 entries, all `torch.float32`**;
* module inventory contains **no BatchNorm** (only LayerNorm, Dropout,
  Embedding, Linear, MultiheadAttention, GELU) → **no running statistics,
  no `num_batches_tracked`**, so weighted averaging of every entry is safe
  and well-defined;
* therefore the Stage 10 aggregation policy is: **average all 138 entries**
  with sample weights; no entry is excluded or special-cased;
* checkpoint format compatibility: the Stage 6 canonical checkpoint stores
  `model_state_dict` (+ optimizer/epoch/metadata) — the federated global
  checkpoints will follow the same flat payload convention with
  federated-specific metadata (round, participation, weights, partition
  hash, protocol config).

## 7. Audit results (reproducible: `python scripts/audit_stage10_protocol.py`)

* Partition hash `9603ddc9facc973d` confirmed; client sizes sum to
  3,137,266. Implied constant FedAvg weights:

| Client | n_k | weight n_k / Σn |
|---|---|---|
| client_00 | 621,801 | 0.198198 |
| client_01 | 544,574 | 0.173582 |
| client_02 | 59,999 | 0.019125 |
| client_03 | 488,090 | 0.155578 |
| client_04 | 69,790 | 0.022245 |
| client_05 | 89,988 | 0.028684 |
| client_06 | 193,852 | 0.061790 |
| client_07 | 201,902 | 0.064356 |
| client_08 | 511,603 | 0.163073 |
| client_09 | 355,667 | 0.113368 |

* Model invariants: 138 float32 entries, no BatchNorm buffers, 2,502,930
  parameters (matches Stage 6 architecture); FedAvg of identical states
  reproduces a state exactly.
* Stage 6 checkpoint payload: `{model_state_dict, optimizer_state_dict,
  epoch, best_validation_install_logloss, training_config, seed,
  split_hash, stage}`; metadata verified (stage=6-centralized, seed=42,
  epoch=3, LogLoss 0.2562108886731335).

## 8. Batch-sampler / epoch convention (verified)

The Stage 8/9 convention is preserved exactly:

* `make_client_loader` (Stage 9, verified earlier) calls `set_epoch` **only
  on `loader.batch_sampler`** — a `PartShuffledBatchSampler`. The
  `DataLoader.sampler` attribute is a plain `SequentialSampler` (batch
  samplers bypass it); the Stage 6-era trap of calling `set_epoch` on it is
  structurally avoided, and `client_sampler()` raises `TypeError` if the
  batch sampler is anything else;
* the epoch parameterization is **per client-local epoch**, not per global
  round: with `local_epochs = 1`, round `r`'s single local epoch uses
  `set_epoch(r - 1)` (`local_sampler_epoch_mode = "round_minus_one"`), so
  round 1 matches the centralized epoch-1 arrangement convention
  (`seed + e - 1`) and every round sees a different, reproducible
  arrangement. If `local_epochs > 1` were ever adopted, client-local epoch
  `e` within round `r` would use `set_epoch((r-1) * local_epochs + e - 1)`
  — recorded here so the frozen convention extends without ambiguity.

## 9. Validation / test policy

* Validation: the existing frozen global validation split (348,586 rows,
  hash `3cd8370ebdecb305`) is evaluated centrally after each round with the
  global model — identical evaluation code path to Stage 8.
* Primary selection metric: global validation Install LogLoss.
* The official test set (160,973 rows) is never loaded, evaluated, or
  referenced for selection/tuning.
* Client-local validation during training: **none** in the baseline (all
  evaluation is global and centralized).

## 10. Reproducibility / seed policy

* One project seed (42) drives everything: the frozen partition (Stage 9),
  global initialization, and the per-round client sampler arrangements
  (`seed + r - 1`).
* Aggregation is deterministic (client-order float64 accumulation).
* No RNG is consumed by FedAvg itself; there is no client sampling, no
  dropout-at-aggregation, no noise injection.
* Every future Stage 10 checkpoint must embed `FederatedConfig.to_dict()`,
  the partition hash, and the round number for exact reproduction.

## 11. Files created/modified

* `src/recsys23_fedrec/federated/config.py` (new) — `FederatedConfig`
* `src/recsys23_fedrec/federated/fedavg.py` (new) — `fedavg_state_dicts`,
  `fedavg_weights_from_counts`, `FedAvgError`
* `src/recsys23_fedrec/federated/__init__.py` (modified — exports only)
* `tests/federated/test_fedavg.py` (new) — 18 tests
* `scripts/audit_stage10_protocol.py` (new) — reproducible read-only audit
* `reports/stage10_federated_protocol_report.md` (new — this report)

No Stage 1–9 implementation, artifact, checkpoint, split, or raw data file
was modified.

## 12. Tests

* 18 new tests in `tests/federated/test_fedavg.py`: exact hand-checkable
  weighted results (2 clients; 10 clients with the real unequal client
  sizes); weight normalization; rejection of missing/extra keys, shape and
  dtype mismatch, empty input, state/count length mismatch, zero/negative
  counts; non-floating rejection by default + `"first"` policy + unknown
  policy rejection; input immutability (incl. data-pointer and
  post-mutation checks); bit-exact repeated aggregation; client-order
  contract with float64-protected permutation stability; real
  `TransformerMMoE` state-dict aggregation (all-float32, no running-stats
  keys, weighted-mean correctness on every entry, strict `load_state_dict`
  into a fresh model); `FederatedConfig` freeze + protocol-change
  rejection + serialization round-trip.
* Full suite after implementation: **299 passed** (281 prior + 18 new),
  zero regressions.

## 13. Unresolved decisions (must be resolved before Stage 10B training)

None blocking. For the record, two documented baseline choices that could
be revisited as explicit protocol variants (never silently):

1. **Fresh local optimizer per client per round** (vs persisted AdamW
   moments) — chosen for attribution cleanliness; a persisted variant is a
   new protocol version.
2. **Cold global initialization** (seed-42 fresh model, not Stage 6
   weights) — chosen to measure FedAvg itself; a warm-start variant would
   confound unless separately controlled.

## 14. Confirmation (at Stage 10A close)

No federated training, no real FedAvg round, no client-local training, no
optimizer creation, and no backward pass had been executed at the time the
Stage 10A protocol was closed/frozen. The Stage 9 partition hash remains
`9603ddc9facc973d`; all protected checkpoints and artifacts are untouched
(verified in §7 and re-verified after the test suite run).

---

## 15. One-Round Smoke Test (integration validation — NOT Stage 10B)

**Scope:** after the Stage 10A freeze, exactly ONE real federated round was
executed to validate the complete integration path (cold init → client
loaders → local training → tested FedAvg → global update → frozen-split
validation → temporary checkpoint → reload). This is **not** the 10-round
Stage 10B experiment; no Stage 10B artifact, best-model selection, or
multi-round history exists. All frozen protocol values (§2) were honored
unchanged: 10 clients, α = 0.5, seed 42, all clients participating,
TransformerMMoE, cold seed-42 initialization (Stage 6 weights NOT loaded),
BCE(click) + BCE(install) with λ = 1.0/1.0, batch 1024, one local epoch,
fresh AdamW 3e-4/1e-2 per client, sample-weighted FedAvg, no SSL / class
weighting / focal / oversampling / scheduler / HPO, official test untouched.

Execution: `scripts/run_stage10_smoke.py` (one round, real data, CUDA
RTX 4060 Ti); completion/verification: `scripts/complete_stage10_smoke.py`.

### 15.1 Round record

* Global model: fresh seed-42 `TransformerMMoE`, 2,502,930 parameters,
  138 float32 state entries; two independent seed-42 creations are
  bit-identical (deterministic); **138/138 entries differ from the Stage 6
  checkpoint** (cold-init proof).
* Sampler: `make_client_loader(..., epoch=0)` per client; sampler epoch
  exactly 0 for round 1 via `loader.batch_sampler` only
  (`PartShuffledBatchSampler`); `DataLoader.sampler` (a plain
  `SequentialSampler`) was never touched. Client rows composed as
  `train_indices[client_positions]`; overlap with the frozen validation
  rows = 0 for all 10 clients; official test data never constructed.
* Aggregation: the tested `fedavg_state_dicts` utility, client order
  `client_00..client_09`; weights sum exactly 1.0; aggregated state
  differs from the initial state in 138/138 entries (max abs diff 1.24e-01)
  and loads bit-exactly into the global model.

### 15.2 Per-client local epoch (all 10 participated; 138/138 params changed each)

| Client | Samples | Batches | Local total loss | Click | Install | Seconds |
|---|---|---|---|---|---|---|
| client_00 | 621,801 | 608 | 0.7460 | 0.3705 | 0.3755 | 473.5 |
| client_01 | 544,574 | 532 | 0.3066 | 0.1305 | 0.1761 | 392.3 |
| client_02 | 59,999 | 59 | 0.8409 | 0.4280 | 0.4129 | 39.8 |
| client_03 | 488,090 | 477 | 0.8779 | 0.4593 | 0.4186 | 482.7 |
| client_04 | 69,790 | 69 | 0.9150 | 0.5164 | 0.3986 | 45.8 |
| client_05 | 89,988 | 88 | 1.1339 | 0.6231 | 0.5108 | 60.4 |
| client_06 | 193,852 | 190 | 0.6607 | 0.3369 | 0.3238 | 129.3 |
| client_07 | 201,902 | 198 | 0.9874 | 0.5833 | 0.4041 | 157.7 |
| client_08 | 511,603 | 500 | 0.6628 | 0.3510 | 0.3118 | 400.1 |
| client_09 | 355,667 | 348 | 0.5430 | 0.3262 | 0.2168 | 186.3 |
| **total** | **3,137,266** | 3,068 | — | — | — | 2,367.9 |

Client-local loss spread reflects the intended Non-IID partition (e.g.
client_02 is click-dominated, client_01 low-engagement) — the integration
path, not result quality, is the object of this test.

### 15.3 Global validation (frozen 348,586-row split only)

| Metric | Value |
|---|---|
| Install LogLoss (**primary**) | 0.542303 |
| Install AUC | 0.8327 |
| Click LogLoss | 0.636163 |
| Click AUC | 0.7661 |

All logits/parameters finite; all four metrics independently recomputed
from the reloaded checkpoint on the full split and matched exactly
(|Δ| = 0). These round-1 numbers are a cold-start integration reading, not
a Stage 10 result.

### 15.4 Runtime / memory

One round wall time **2,400.9 s** (~40 min); peak RSS 2,510 MiB; peak GPU
allocated/reserved 3,604 / 3,928 MiB.

### 15.5 Temporary artifacts (separate path; nothing protected written)

* `artifacts/federated/smoke/stage10_smoke_round1.pt` — temporary smoke
  checkpoint (sha256 `6e250706dc19c05504b3f00e6014358fb34fa02cf541be4caad64a776fdea36f`),
  marked `smoke_test: true, rounds_executed: 1` with an explicit
  NOT-Stage-10B note in the payload.
* `artifacts/federated/smoke/stage10_smoke_round1_result.json` — metrics,
  per-client records, config, hashes (no tensors).
* Reload verified: strict load into a fresh seed-42 model, bit-identical
  state, identical logits on a validation batch (atol 1e-5).

### 15.6 Integrity after the smoke round

Stage 2 hash `c3a62caf13326617`, split hashes `dd37605169615442` /
`3cd8370ebdecb305`, Stage 6 checkpoint sha256, Stage 8 A/C/D checkpoints,
raw train/test byte sizes, and the Stage 9 partition artifacts (hash
`9603ddc9facc973d`) were all re-verified byte-identical (sha256 + size +
mtime) after the round.

### 15.7 Files created for the smoke test

* `scripts/run_stage10_smoke.py` (new) — one-round smoke runner
* `scripts/complete_stage10_smoke.py` (new) — post-checkpoint reload/
  integrity completion + result JSON (the first run crashed only in its
  reload step after the round itself had completed; fixed in
  `run_stage10_smoke.py`, and completion was performed without re-training)
* `artifacts/federated/smoke/stage10_smoke_round1.pt` + `..._result.json`
  (new, temporary)
* this report (§15 appended)

**Stage 10B remains NOT started: no 10-round run, no per-round history,
no best-round selection has been executed.**
