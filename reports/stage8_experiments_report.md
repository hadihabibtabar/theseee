# Stage 8 Report — Controlled Centralized Experiments (A/B/C/D)

**Status: Experiment A COMPLETE (2026-09-17, §13.1); canonical B reused frozen; C and D
NOT yet launched (awaiting inspection).**

> No claim is made that any component improves performance: comparisons are stated only as
> direct numerical differences between completed runs. The official test set remains
> completely held out (never used for training, validation, model selection, or
> hyperparameter choices).

---

## 1. Scientific questions

* **Q1** — What changes when MMoE is added to the Transformer? (A vs B)
* **Q2** — What changes when SSL is added without MMoE? (A vs C)
* **Q3** — What changes when SSL is added to Transformer + MMoE? (B vs D)
* **Q4** — Does the SSL effect remain consistent after introducing MMoE? ((A→C) vs (B→D))

Answers will be stated only as direct numerical comparisons after the runs exist — no
ranking, no "winner", no causal claims beyond these controlled comparisons.

## 2. Four experiment definitions

| Experiment | Architecture | Tasks | SSL | Loss |
| --- | --- | --- | --- | --- |
| **A** `experiment_a_transformer_supervised` | `TransformerBaseline` (Stage 4, install head) | Install | — | `BCEWithLogitsLoss(install_logit, install)` |
| **B** `experiment_b_transformer_mmoe` | `TransformerMMoE` (Stage 5) | Click + Install | — | `1.0·BCE(click) + 1.0·BCE(install)` |
| **C** `experiment_c_transformer_ssl` | `SSLTransformerBaseline` (Stage 4 + projection head) | Install | Feature corruption 0.15, τ=0.2 | `0.6·BCE(install) + 0.4·NT-Xent` |
| **D** `experiment_d_transformer_mmoe_ssl` | `SSLTransformerMMoE` (Stage 5 + projection head) | Click + Install | Feature corruption 0.15, τ=0.2 | `0.6·[BCE(click)+BCE(install)] + 0.4·NT-Xent` |

All four share the identical frozen Stage 4 Transformer backbone
(d_model 128, nhead 8, 6 layers, FFN 128, dropout 0.1; verified by test) and, where present,
the frozen Stage 5 MMoE head (8 experts, expert_dim 128) and Stage 7 projection head
(128→128→GELU→64, L2-normalized output). Raw logits everywhere; sigmoid never applied before
BCEWithLogitsLoss; no class weighting, focal loss, or oversampling anywhere.

## 3. Controlled training protocol (identical for all new runs)

```
seed = 42 (project SEED; python/numpy/torch/torch.cuda all seeded)
batch_size = 1024          epochs = 3
optimizer = AdamW          learning_rate = 3e-4      weight_decay = 1e-2
validation every epoch     model selection = min validation Install LogLoss
early stopping patience = 2
train loader: deterministic per-epoch part-shuffled batches (Stage 3 sampler)
validation loader: ascending order, shuffle = False, drop_last = False everywhere
device: CUDA when available (RTX 4060 Ti)
```

No learning-rate schedulers, no additional augmentation types, no hyperparameter search, no
architecture changes.

**Protocol nuance (documented for fairness):** the Stage 6 `Trainer.fit` attempted a
per-epoch sampler reset via `loader.sampler`, but `make_subset_loader` attaches the
sampler as `batch_sampler`, so the reset never fired — all three Stage 6 epochs used the
epoch-1 batch arrangement. The Stage 8 runner calls `set_epoch` on the `batch_sampler`
directly (the originally intended behavior). Consequence: Experiment A/C/D training batches
are reshuffled per epoch while canonical Experiment B's were not. This affects only batch
ORDER within an epoch (same samples, same batches-per-epoch); if strict identity of this
detail with B were required, B would need a fresh ~19 h retrain for a protocol nuance with
no a-priori directional effect — flagged here for an explicit decision rather than silently
chosen. A controlled diagnostic of this nuance is documented in §4.

**Decision (2026-09-17): Experiment B is NOT retrained.** The Stage 6 result remains the
canonical baseline, described as **"Canonical Stage 6 B (historical sampler)"**; Experiments
A/C/D run under the corrected sampler with this difference documented (§4). Experiment A's
per-epoch batch arrangement therefore differs from canonical B's from epoch 2 onward.

## 4. Sampler Protocol Investigation (evidence, no verdict)

**Question.** Does the sampler correction materially change the training trajectory enough
that using the frozen Stage 6 checkpoint as Experiment B while training A/C/D with
corrected epoch reshuffling creates an important protocol mismatch?

**What Stage 6 actually did (proven from batch indices).** `Trainer.fit` called
`set_epoch` through `loader.sampler`, but `make_subset_loader` attaches the
`PartShuffledBatchSampler` as `batch_sampler`; `loader.sampler` is a plain
`SequentialSampler` without `set_epoch`, so the call was a silent no-op. Every Stage 6
epoch therefore consumed the sampler's `seed + 0` stream.

**Fingerprints over the REAL training subset** (3,137,266 rows → 3,064 batches of 1024,
seed 42; SHA-256 over the exact batch-index sequences — `run_sampler_diagnostic.py`):

| Epoch | Protocol H (historical) | Protocol C (corrected) |
| --- | --- | --- |
| 1 | `bf973a7e6cf076ed…` | `bf973a7e6cf076ed…` |
| 2 | `bf973a7e6cf076ed…` | `99778a27cd2a9b99…` |
| 3 | `bf973a7e6cf076ed…` | `8f3940716d700c7b…` |

* H is **identical across epochs 1–3** (one digest) — direct proof the Stage 6 reshuffle
  never fired; regeneration reproduces it exactly (deterministic).
* C is **distinct per epoch** and deterministic under regeneration.
* Structural fact: **C(epoch 1) ≡ H(epoch 1) by construction** — `set_epoch(0)` yields the
  same `seed + 0` stream the historical run used. The protocols diverge only from epoch 2
  onward; Experiment B's epochs 2–3 are the divergent part.

**Tiny controlled training comparison** (Experiment B model, real split rows, batch 1024,
4 train batches/epoch × 2 epochs, 2,048 validation rows, identical initialization/seed/
optimizer/loss; only the epoch-2 batch arrangement differs):

| Quantity | Protocol H | Protocol C |
| --- | --- | --- |
| Epoch-1 train total loss | 1.380741 | 1.380741 (identical — same batches) |
| Epoch-2 train total loss | 1.361015 | 1.361163 (Δ ≈ 1.5e-4) |
| Val install LogLoss (2,048 rows) | 0.666801 | 0.666808 (Δ ≈ 7e-6) |
| Val install AUC (2,048 rows) | 0.535317 | 0.533544 (Δ ≈ 1.8e-3) |
| Parameter movement from init (L2) | 1.500375 | 1.501053 |

* **Cross-protocol parameter delta: 0.0789 (L2)** — ≈ **5.3%** of the movement from
  initialization, after only 4 divergent gradient steps.
* **Reproducibility: confirmed.** Repeating Protocol H bit-exactly reproduces the model
  (`deterministic_repeat = true`); both fingerprint sets regenerate identically.

**Limitations of this comparison.** 4 batches/epoch, 2 epochs, one seed, one model, a
4,096-row slice of the split. Validation deltas at this scale are noise and MUST NOT be
read as quality differences; the diagnostic cannot bound the divergence at full scale
(3,064 batches × 2 divergent epochs).

**What the evidence establishes — and what it does not.** The correction produces a
**real, reproducible trajectory divergence beginning in epoch 2** (≈5% of init movement
at 4 divergent steps). The diagnostic does **not** establish that this divergence is
material at 3-epoch scale, and equally does **not** certify the protocols as equivalent.
It is therefore **not by itself sufficient to justify a ~19 h B retrain**; the decision
remains open between (i) accepting the now-precisely-documented asymmetry (B historical,
A/C/D corrected — identical data, seed, objective, and batches-per-epoch) and (ii)
retraining B under the corrected protocol for strict identity. No performance claim is
made or implied by either option.

## 5. Data / split provenance

* Raw dataset: `train/` 30 tab-separated shards (1,919,412,753 bytes), `test/` 1 shard
  (88,684,100 bytes) — read-only throughout.
* Stage 2 preprocessing artifact: `artifacts/preprocessing`, artifact hash
  `c3a62caf13326617` — not refit, not modified; the single source of truth for feature
  order, vocabularies, numerical normalization, and missing-value handling.
* Train/validation split: `artifacts/splits/centralized_split.json` — reused, NOT
  regenerated; train hash `dd37605169615442`, validation hash `3cd8370ebdecb305`,
  3,137,266 / 348,586 rows of 3,485,852. The hash is re-verified on every load
  (`load_split`) and embedded in every Stage 8 checkpoint.
* Data path: Stage 3 memory-bounded row-group streaming via `make_subset_loader` (Stage 6
  layer, unchanged). The official test set is not opened by any Stage 8 training path.

## 6. Metric definitions

Identical to Stage 6 (no redefinition):

* **Install LogLoss** (primary selection metric): sample-weighted mean binary cross-entropy
  from raw logits via the stable fused form `softplus(-z) + (1-y)·z`.
* **Install AUC**: exact tie-aware rank AUC (`P(pos>neg) + 0.5·P(tie)`), computed once over
  all validation logits.
* **Click LogLoss / Click AUC**: same formulas, click task; reported for B/D, `N/A` for A/C.
* Per-epoch records: train total / supervised / contrastive (if SSL) / click (if MMoE) /
  install losses, validation metrics, epoch duration, learning rate.

## 7. Canonical Experiment B baseline (reused, source = Stage 6 canonical baseline)

Experiment B is NOT retrained: the verified Stage 6 result is the canonical B baseline —
same seed, split, protocol, metrics, and checkpoint semantics.

| Field | Value |
| --- | --- |
| Source | Stage 6 canonical baseline |
| Checkpoint | `artifacts/checkpoints/centralized_transformer_mmoe_best.pt` (sha256 `25f3267e…e1d8617`, UNTOUCHED) |
| Best epoch | 3 |
| Validation Install LogLoss | **0.2562108886731335** |
| Validation Install AUC | 0.9132 |
| Validation Click LogLoss | 0.2640 |
| Validation Click AUC | 0.9114 |
| Split hash | `dd37605169615442` |

## 8. SSL protocol (single-stage joint objective)

The Stage 7 infrastructure supports both interpretations cleanly: (1) **joint SSL +
supervised training from initialization** — `SSLTransformerMMoE.forward` (dual-view) with
`compute_joint_loss`, α = 0.6; and (2) a supervised-only path (`supervised_forward`) usable
for fine-tuning after a contrastive phase.

**Stage 8 default decision: the exact Stage 7 joint objective (α = 0.6) is used for C and D,
starting from the same random initialization protocol as A/B.** No two-stage
pretraining/fine-tuning experiment is introduced in Stage 8; if pursued later it must be
implemented and documented as a separate explicitly named experiment.

## 9. Pilot results (spec 8.13 — PASSED)

`python scripts/run_stage8_pilot.py` — real Stage 2 Parquet, hash-verified split, CUDA,
batch 128, 2 train batches × 2 epochs + 2 validation batches per experiment; checkpoints
written to a temporary directory only.

| Experiment | Params | Train steps (total loss) | Val (pilot batches) | Checkpoint reload |
| --- | --- | --- | --- | --- |
| A | 2,352,129 | 0.6954 → 0.6704 → 0.6575 → 0.6299 | install LogLoss 0.6237, AUC 0.5745 | PASSED |
| C | 2,376,897 | 2.2944 → 2.0900 → 1.9485 → 1.8950 (contrastive 4.69→3.73) | install LogLoss 0.6690, AUC 0.5227 | PASSED |
| D | 2,527,698 | 2.6365 → 2.4908 → 2.3619 → 2.2748 (contrastive 4.51→3.62) | install LogLoss 0.6884 AUC 0.5952, click LogLoss 0.6900 AUC 0.5084 | PASSED |
| B | 2,502,930 | — (constructs from frozen defaults; canonical result reused) | — | — |

All losses finite on every step; pilot AUCs ≈ 0.5–0.6 and LogLosses ≈ log 2 as expected for
a handful of untrained batches; parameter counts confirm the intended architectures
(A = backbone+head; C = A + 24,768 projection params; D = B + 24,768).
Pilot verified per experiment: forward → loss → backward → optimizer step → checkpoint
write (temp path) → reload → metric calculation. **Integrity after pilot: unchanged.**

## 10. Integrity verification (start/end of Phase 1; re-verified after Experiment A)

| Protected artifact | Frozen value | Verified |
| --- | --- | --- |
| Stage 2 artifact hash | `c3a62caf13326617` | ✅ unchanged |
| Split train hash | `dd37605169615442` | ✅ unchanged |
| Split validation hash | `3cd8370ebdecb305` | ✅ unchanged |
| Stage 6 checkpoint sha256 | `25f3267eba572bf99e72c19eee2351236fe65327f71af7c7a8e39fae8e1d8617` | ✅ unchanged |
| Stage 6 checkpoint mtime | 1789563754.0476708 (2026-09-16 16:32:34) | ✅ unchanged |
| Raw train bytes | 1,919,412,753 | ✅ unchanged |
| Raw test bytes | 88,684,100 | ✅ unchanged |

Enforced by `src/recsys23_fedrec/experiments/integrity.py` (snapshot before runs,
`verify_protected_artifacts` after; the pilot asserts zero violations). Stage 8 checkpoints
go to `artifacts/checkpoints/stage8/` only (atomic tmp-file + `os.replace` writes); no code
path can overwrite the Stage 6 file. Re-verified after the Experiment A full run (2026-09-17):
zero violations — Stage 6 checkpoint sha256 `25f3267eba…e1d8617` and mtime 2026-09-16
16:32:34 unchanged, all other protected values identical.

## 11. Test results

`tests/experiments/` — **43 new tests, all passing** (32 experiment-infrastructure + 11
sampler-diagnostic); full suite **244 passed** (201 previous + 43 new; zero regressions).
Full suite re-run after the Experiment A full run (2026-09-17): 244/244 green.
Coverage: ExperimentConfig validation/dispatch/serialization;
model + loss dispatch for A/B/C/D (SSL only for C/D, MMoE only for B/D, shared frozen
backbone); runner fit (parameter updates, best-epoch selection, per-epoch records, click
columns only when MMoE, contrastive only when SSL); raw-logit stability at ±50; checkpoint
metadata completeness (experiment name, model type, seed, alpha, corruption rate,
temperature, split/validation hashes, preprocessing hash, sampler protocol, timestamp,
parameter count, stage) and atomicity (no tmp leftovers, foreign files untouched);
result JSON serialization; integrity-guard tamper detection; frozen protected-value
constants; sampler diagnostic (historical reproducibility + cross-epoch identity, corrected
determinism + epoch dependence + exact subset coverage, digest-over-actual-indices proof,
no-test-set dependency check, tiny end-to-end comparison determinism on the fixture).

## 12. Limitations

* Single seed (42), single protocol: differences between experiments will carry seed noise;
  no variance estimate is claimed.
* 3 epochs only (controlled comparability with the Stage 6 baseline, not convergence).
* Per-epoch reshuffle nuance (§3): Experiment A (and the future C/D) use corrected
  epoch-dependent reshuffling while canonical B uses the frozen historical Stage 6 batch
  arrangement; quantified by the controlled diagnostic in §4, decision (B not retrained)
  documented there rather than hidden.
* Click metrics for single-task experiments are `N/A` by design; Q1/Q3 click comparisons
  are only possible among B/D.
* SSL views are label-preserving by construction (Stage 7); the supervised term in C/D is
  averaged over both corrupted views (uncorrupted-batch supervision is a separate
  untested variant, deferred).

---

## 13. Results

### 13.1 Experiment A — Transformer supervised, install-only (full run complete 2026-09-17)

Fresh Stage 8 run (`scripts/run_stage8_experiment.py --experiment A`) under the controlled
protocol (§3) with the **corrected per-epoch reshuffling sampler**; canonical B (§7) used
the historical Stage 6 batch arrangement (difference documented in §3/§4; no ranking
implied). Data: 3,137,266 train rows (3,064 batches/epoch), 348,586 validation rows
(341 batches); official test set never loaded.

| Epoch | Duration (s) | Train install BCE | Val Install LogLoss | Val Install AUC | Batches (train/val) | LR |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4,244.4 | 0.2890 | 0.26798583503064066 | 0.9029 | 3,064 / 341 | 3e-4 |
| 2 | 3,181.0 | 0.2651 | 0.26038144870648744 | 0.9097 | 3,064 / 341 | 3e-4 |
| 3 | 3,803.3 | 0.2589 | **0.2589328218594451** | **0.9120** | 3,064 / 341 | 3e-4 |

Every epoch improved validation Install LogLoss; early stopping never triggered; best epoch
= 3 by minimum validation Install LogLoss.

**Best checkpoint:**
`artifacts/checkpoints/stage8/experiment_a_transformer_supervised_best.pt` — sha256
`dcc42aee28…edec981`, 28,363,055 bytes, atomic write, timestamp 2026-09-17T11:06:32Z.

Checkpoint metadata verified on load: stage = `8-controlled-experiments` (stage_number 8),
experiment = A, model_type = `transformer_baseline`, seed = 42, epoch = 3,
parameter_count = 2,352,129, architecture (d_model 128, nhead 8, num_layers 6, FFN 128,
dropout 0.1), optimizer (adamw, lr 3e-4, wd 1e-2), batch_size 1024, split train
`dd37605169615442` / validation `3cd8370ebdecb305`, preprocessing hash `c3a62caf13326617`,
sampler protocol `stage8-corrected-per-epoch-reshuffle`, best validation Install LogLoss
0.2589328218594451, validation Install AUC 0.9120110869407654.

**Checkpoint reload verification: PASSED.** (i) In-process by the runner: fresh model, one
validation batch, outputs match atol 1e-5. (ii) Independent post-run check: strict
`load_state_dict` into a fresh `TransformerBaseline` — 114 tensors, 2,352,129 parameters,
no missing/unexpected keys.

**Runtime/resources:** total 11,229 s (3.12 h); peak CPU RSS 3,167 MiB; peak GPU
allocated/reserved 3,639 / 4,160 MiB. NaN/Inf guards never tripped; no OOM; per-epoch batch
counts exactly 3,064 / 341 as expected.

**Official test set: NOT evaluated (never loaded).** Machine-readable result:
`artifacts/checkpoints/stage8/experiment_a_transformer_supervised_result.json`.

### 13.2 Experiment C — Transformer + SSL, install-only (full run complete 2026-09-18)

Fresh Stage 8 run (`scripts/run_stage8_experiment.py --experiment C`) under the identical
controlled protocol (§3) with the **corrected per-epoch reshuffling sampler**: same seed,
split, preprocessing, optimizer, batch size, epochs, and checkpoint-selection rule as
Experiment A. The intentional difference is `C = A + contrastive SSL` (Stage 7 frozen
mechanism: feature corruption 0.15, projection 128→128→GELU→64, symmetric NT-Xent,
τ = 0.2, joint loss α = 0.6; supervised term averaged over the two corrupted views — the
frozen Stage 7 implementation, documented in §12). No ranking is implied vs A.

| Epoch | Duration (s) | Train joint | Train install | Train contrastive | Val Install LogLoss | Val Install AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 18,779.3 | 1.5349 | 0.3091 | 3.3737 | 0.2777 | 0.8930 |
| 2 | 16,818.2 | 1.4115 | 0.2815 | 3.1064 | 0.2724 | 0.8985 |
| 3 | 53,098.1* | 1.3903 | 0.2769 | 3.0604 | **0.2697706961950431** | **0.9017** |

Every epoch improved validation Install LogLoss; early stopping never triggered; best epoch
= 3 by minimum validation Install LogLoss.

\* Epoch 3's wall time includes ~10 h during which the host was in OS standby (no compute
occurred; see §14 operational caveat). Losses and batches were unaffected.

**Best checkpoint:** `artifacts/checkpoints/stage8/experiment_c_transformer_ssl_best.pt`
— sha256 `dacb4c3407823ffe…a054e499`, 28,665,999 bytes, atomic write,
timestamp 2026-09-18T12:03:31Z. Metadata verified: stage = `8-controlled-experiments`,
experiment = C, model_type = `ssl_transformer_baseline`, seed = 42, epoch = 3,
parameter_count = 2,376,897, both split hashes, preprocessing hash `c3a62caf13326617`,
sampler protocol `stage8-corrected-per-epoch-reshuffle`, alpha 0.6, corruption 0.15,
temperature 0.2, supervised task = install.

**Checkpoint reload verification: PASSED** (runner: fresh model, one validation batch,
atol 1e-5). Total runtime 88,700.9 s (24.64 h incl. the standby gap); peak RSS 8,645 MiB;
peak GPU allocated/reserved 7,175 / 7,698 MiB. Official test set: NOT evaluated (never
loaded). Machine-readable result:
`artifacts/checkpoints/stage8/experiment_c_transformer_ssl_result.json`.

### 13.3 Experiment D — Transformer + MMoE + SSL, Click+Install (RUNNING / pending final results)

Launched `scripts/run_stage8_experiment.py --experiment D` under the identical controlled
protocol (§3) with the corrected per-epoch reshuffling sampler. Architecture:
`ssl_transformer_mmoe` — frozen Stage 5 Transformer+MMoE backbone/head (8 shared experts,
separate Click/Install gates and towers) plus the Stage 7 SSL projection head; supervised
tasks Click (auxiliary) + Install (primary); joint loss α = 0.6; corruption 0.15; τ = 0.2.
Split/preprocessing/seed/optimizer identical to A/C. Checkpoint (when written):
`artifacts/checkpoints/stage8/experiment_d_transformer_mmoe_ssl_best.pt`; log:
`stage8_experiment_d_log.txt`. Final metrics will be appended here by the training
process's result JSON — nothing is claimed before the run completes. No ranking vs
A/B/C is implied.

### 13.4 Pending

Experiment D completion, then the controlled-comparison table (A/B/C/D × metrics × best
epoch) and direct numerical difference statements for Q1–Q4. Nothing is ranked until all
runs exist.

Checkpoints present:

```
artifacts/checkpoints/stage8/experiment_a_transformer_supervised_best.pt       (2026-09-17)
artifacts/checkpoints/stage8/experiment_a_transformer_supervised_result.json   (2026-09-17)
artifacts/checkpoints/stage8/experiment_c_transformer_ssl_best.pt              (2026-09-18)
artifacts/checkpoints/stage8/experiment_c_transformer_ssl_result.json          (2026-09-18)
artifacts/checkpoints/stage8/experiment_d_transformer_mmoe_ssl_best.pt         (pending)
```
