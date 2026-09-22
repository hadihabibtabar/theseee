# Stage 9 — Federated Client Partition Report

## 1. Stage 9 objective

Create exactly **10 simulated federated clients** with controlled Non-IID
data distributions, via a reproducible Dirichlet partition of the
centralized **training rows only**, and scientifically verify the partition
before any federated training exists.

Research context: this dataset contains no real organizations. The clients
are **simulated federated clients** (horizontal FL: same feature schema,
same task, different local samples). They are not real companies,
publishers, or organizations.

No model was trained and no neural-network code was executed in Stage 9.

## 2. Data scope and leakage guarantees

| Portion | Rows | Role |
|---|---|---|
| Centralized train rows | **3,137,266** | partitioned into 10 clients |
| Centralized validation | 348,586 | **globally held out — never partitioned** |
| Official test | 160,973 | untouched, never loaded |

* Verified non-overlap of the client rows with the frozen validation indices
  (independent post-run check: `overlap with validation rows = 0`).
* Labels for the partition come from the frozen Stage 2 Parquet
  (`is_clicked`, `is_installed` columns); **no validation or test labels**
  are used anywhere.
* Preprocessing artifact remains the frozen Stage 2 artifact
  (`c3a62caf13326617`); vocabularies and normalization are untouched; no
  per-client preprocessing exists.

**Important composition note:** the processed train Parquet directory holds
all 3,485,852 processed rows (train+validation space). The partition
therefore operates on *positions into the frozen centralized train index
array* (`artifacts/splits/centralized_split.npz: train_indices`,
hash `dd37605169615442`). A leakage guard in the runner subsets the streamed
labels with `split.train_indices` and asserts exactly 3,137,266 rows before
partitioning. An initial implementation defect that would have included
validation rows was caught by this guard and fixed before any artifact was
finalized; the shipped partition provably contains only training rows.

## 3. Partition mechanism (exact definition)

For each joint task-relevant label state `(is_clicked, is_installed)` in the
fixed canonical order `(0,0), (0,1), (1,0), (1,1)`:

1. take all training rows exhibiting that state (ascending dataset order);
2. draw `Dirichlet(alpha, ..., alpha)` with one component per client
   (`numpy.random.default_rng(seed)`, PCG64; one draw per state, states in
   fixed order);
3. convert proportions to integer client counts with **largest-remainder**
   rounding so the counts sum exactly to the state's row count;
4. cut contiguous chunks in ascending dataset order, assigning clients in id
   order.

No post-hoc normalization, rebalancing, or smoothing is applied — client
size and label skew are the intended, preserved Non-IID signal.

Determinism: identical `seed + alpha + labels` reproduce byte-identical
client assignments (verified: two independent runs produced
byte-identical JSON and NPZ artifacts).

Planned reproducible variants (not generated now): `alpha = 1.0`,
`alpha = 0.5` (primary), `alpha = 0.1` — same code path, different
concentration. Empty clients are rejected; `alpha` must be positive; the
client count is fixed at exactly 10 by the Stage 9 protocol.

## 4. Primary partition: alpha = 0.5, seed = 42

Artifacts:

* Manifest: `artifacts/federated/partition_alpha_0_5_seed42.json`
* Client index mapping: `artifacts/federated/partition_alpha_0_5_seed42.npz`
  (per-client sorted int64 positions into `train_indices`; raw rows are not
  duplicated on disk)
* Partition hash: **`9603ddc9facc973d`**
* Train-labels hash: `800e2aa9d0bf7b79`
* Source split: train `dd37605169615442`, validation `3cd8370ebdecb305`
* Partition method: `dirichlet_joint_label_proportional_v1` (format version 1)

### Client sizes

| Client | Rows | % of 3,137,266 |
|---|---|---|
| client_00 | 621,801 | 19.82% |
| client_01 | 544,574 | 17.36% |
| client_02 | 59,999 | 1.91% |
| client_03 | 488,090 | 15.56% |
| client_04 | 69,790 | 2.22% |
| client_05 | 89,988 | 2.87% |
| client_06 | 193,852 | 6.18% |
| client_07 | 201,902 | 6.44% |
| client_08 | 511,603 | 16.31% |
| client_09 | 355,667 | 11.34% |

min 59,999 / max 621,801 (max/min = 10.36×) — size imbalance is intended.

### Label distributions (global click rate 0.2199, install rate 0.1742)

Joint-state proportions per client in the order
`(c=0,i=0) (c=0,i=1) (c=1,i=0) (c=1,i=1)`; global:
`0.6772 / 0.1030 / 0.1487 / 0.0712`.

| Client | (0,0) | (0,1) | (1,0) | (1,1) | Click rate | Install rate |
|---|---|---|---|---|---|---|
| client_00 | 0.4932 | 0.2995 | 0.1747 | 0.0327 | 0.2073 | 0.3322 |
| client_01 | 0.8961 | 0.0764 | 0.0274 | 0.0001 | 0.0275 | 0.0765 |
| client_02 | 0.0564 | 0.0201 | 0.8744 | 0.0492 | 0.9236 | 0.0693 |
| client_03 | 0.5902 | 0.0725 | 0.1745 | 0.1628 | 0.3373 | 0.2352 |
| client_04 | 0.0896 | 0.0770 | 0.8303 | 0.0031 | 0.8334 | 0.0801 |
| client_05 | 0.5824 | 0.1240 | 0.2323 | 0.0614 | 0.2937 | 0.1853 |
| client_06 | 0.8817 | 0.0044 | 0.0100 | 0.1039 | 0.1139 | 0.1083 |
| client_07 | 0.5874 | 0.0129 | 0.2446 | 0.1552 | 0.3997 | 0.1681 |
| client_08 | 0.7942 | 0.0189 | 0.0630 | 0.1239 | 0.1869 | 0.1428 |
| client_09 | 0.7982 | 0.0813 | 0.1204 | 0.0001 | 0.1205 | 0.0814 |

The partition is strongly Non-IID, as designed: per-state client click
rates span 0.027–0.924 and install rates span 0.069–0.332; e.g. client_02
is a click-dominated tiny client while client_01 is a low-engagement
majority client.

### Heterogeneity measure (transparent, no ranking)

Per-client absolute deviation of each joint-state proportion (and of the
row share) from the global value:

* mean absolute joint-proportion deviation across all client×state cells:
  **0.1389**
* max single deviation: **0.7257**
* per-client deviation sums are recorded in the manifest
  (`heterogeneity.per_client_joint_abs_deviation_sum`).

## 5. Verification performed

* 24 new unit/integration tests in `tests/federated/test_partition.py`
  (10 clients; exact cover; no loss; no duplicates; totals; train-space
  composition + validation disjointness; determinism under seed 42;
  re-run identity; seed sensitivity; joint-count conservation; client
  distribution correctness; alpha-skew monotonicity; manifest contents +
  determinism; hash sensitivity; empty-client rejection; alpha
  validation; client-count validation; binary/shape validation;
  extreme-imbalance preservation; Parquet label round-trip; NPZ round-trip).
* Full regression suite re-run: **268 passed** (244 prior + 24 new), zero
  regressions, no existing test weakened.
* Independent post-run audit: sizes sum to 3,137,266; positions unique and
  < 3,137,266; mapped dataset rows unique; zero overlap with the frozen
  validation rows; NPZ `partition_hash` matches the manifest.
* Re-run determinism: identical manifest + NPZ bytes across two runs.
* Protected artifacts re-verified intact before and after: Stage 2
  `c3a62caf13326617`; splits `dd37605169615442` / `3cd8370ebdecb305`;
  Stage 6 checkpoint `25f3267eba...e1d8617` (mtime unchanged); raw train
  1,919,412,753 bytes; raw test 88,684,100 bytes; Stage 8 A/C/D checkpoints
  untouched.

## 6. Files created/modified

* `src/recsys23_fedrec/federated/__init__.py` (new)
* `src/recsys23_fedrec/federated/partition.py` (new)
* `tests/federated/__init__.py` (new)
* `tests/federated/test_partition.py` (new)
* `scripts/run_stage9_partition.py` (new)
* `reports/stage9_partition_report.md` (new — this report)
* `artifacts/federated/partition_alpha_0_5_seed42.json` (new — generated)
* `artifacts/federated/partition_alpha_0_5_seed42.npz` (new — generated)

No Stage 1–8 code, model, preprocessing, split, checkpoint, or raw data file
was modified.

## 7. Limitations / design decisions

* Client index arrays are positions into the frozen `train_indices` (not raw
  dataset rows) so the partition is provably disjoint from validation/test
  and composes directly with the existing `make_subset_loader` /
  `PartShuffledBatchSampler` infrastructure for the later federated
  DataLoader.
* The Dirichlet allocation is per joint state (4 independent draws) rather
  than one 10×4 matrix draw — this keeps integer conservation per state
  exact and the RNG consumption order simple and auditable.
* Contiguous chunk cutting in ascending dataset order keeps client reads
  cache-friendly for the row-group-streaming dataset; the within-state
  segment order is deterministic given the sorted labels.
* Clients are simulated; any resemblance of client_02-style extremes to
  real-world publishers is an artifact of the label distribution, not a
  modeling claim.
* Only the primary `alpha=0.5` partition is generated; `alpha=1.0/0.1` use
  the identical code path and can be produced later without code changes.

## 8. Recommendation for the next Stage 9 sub-step

Based on the observed partition:

1. **Client DataLoader verification (suggested next):** build a thin
   per-client loader factory over the existing `make_subset_loader`
   (composing `train_indices[client_positions]`) and verify batch contracts,
   determinism, and memory bounds for the largest (client_00, 621,801 rows)
   and smallest (client_02, 59,999 rows) clients — no training.
2. Confirm client_02's 59,999 rows are workable for local training batches
   (58 full batches of 1024 + remainder) — they are, so no minimum-size
   adjustment is needed.
3. Only after (1)–(2): design the Stage 10 federated training protocol
   (FedAvg client/round configuration) as a separate, controlled decision.

No federated training, no client models, and no communication rounds were
started in Stage 9.

## 9. Per-client DataLoader verification (follow-up task)

A thin per-client DataLoader factory was added on top of the existing
Stage 3/8 infrastructure - no second dataset implementation, no new
preprocessing, no client-specific fitting:

* `src/recsys23_fedrec/federated/client_loader.py`:
  `load_partition_positions` (NPZ validation),
  `compose_client_row_indices` (frozen split + client positions, with a
  hard guard that positions stay inside the train space),
  `make_client_loader` (thin wrapper over the existing
  `make_subset_loader`; epoch e uses the Stage 8 convention seed + e - 1
  via `PartShuffledBatchSampler.set_epoch`), and `client_sampler`.
* Tests: 13 new synthetic tests in `tests/federated/test_client_loader.py`
  (fixtures reused from `tests/data/conftest.py`): batch contract
  (keys/dtypes/shapes via the Stage 3 `validate_batch`), exact row identity
  per yielded sample, no duplicates, determinism under the same seed,
  per-epoch reshuffling reproducibility, Stage 8 sampler-seed convention,
  and structural validation/test exclusion.

Real-data verification (`scripts/verify_stage9_client_loaders.py`,
read-only): partition hash `9603ddc9facc973d` confirmed unchanged; frozen
split + Stage 2 artifact verified.

**client_02 (smallest, 59,999 rows) - full verification:** sorted pass
yields exactly 59,999 rows in 59 batches (1 s); full shuffled epoch pass
covers all 59,999 rows in 59 batches (14 s); measured click rate 0.9236 /
install rate 0.0693 match the manifest exactly; epoch arrangements for
epochs 1/2/3 differ (content check); sampler batches cover the client
positions exactly once.

**client_00 (largest, 621,801 rows) - structural + bounded partial
verification:** loader length 608 = ceil(621,801/1024); 50 real batches
contract-checked; epoch-1 arrangement deterministic; epochs 1/2/3 differ.
(Full shuffled scans of the largest client were deliberately skipped: they
hop randomly across all 90 Parquet parts.)

**Exclusion + memory:** zero overlap between every client rows and the
frozen validation indices; all 10 clients positions mutually disjoint;
max position 3,137,265 < 3,137,266; official test data never opened; RSS
431 -> 995 MiB during full client iteration (bounded row-group streaming -
the client subset is never materialized). No model, optimizer, backward
pass, or training was executed.

Full-suite state after this addition: **281 tests (244 prior + 24 partition
+ 13 client-loader), all passing.**
