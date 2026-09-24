# Final Federated Experimental Matrix

**Date:** 2026-09-24
**Scope:** final scientific audit of the complete six-cell federated evaluation matrix (Stage 10B, Stage 11, four Stage 12 cells) on the RecSys-2023 challenge dataset.
**Nature of this document:** read-only analysis and reporting. No training was performed, no checkpoint was created or modified, and no protected artifact was touched while producing this report. All numbers are extracted verbatim from the completed result artifacts; nothing is inferred, imputed, or estimated.

Machine-readable companions:
- `reports/stage12/final_federated_matrix.json` — full protocol, partitions, per-cell records, per-round trajectories, integrity status.
- `reports/stage12/final_federated_matrix.csv` — one row per cell.
- `reports/stage12/final_federated_matrix.md` — compact thesis-ready table + notes.

---

## 1. Experimental protocol

Every cell in the matrix uses one frozen federated protocol, unchanged across all six experiments:

| Protocol element | Value |
| --- | --- |
| Simulated clients | 10 |
| Partitioning | Dirichlet (alpha in {1.0, 0.5, 0.1}), partition seed 42 |
| Participating clients | all 10 clients participate in every round |
| Communication rounds | 10 |
| Local epochs per round | 1 |
| Local batch size | 1,024 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.01 |
| Aggregation | sample-weighted FedAvg (client sample counts as weights) |
| Local optimizer state | not persisted across rounds (fresh optimizer each client/round) |
| Initialization | cold seed-42 initialization; no checkpoint load |
| Checkpoint selection | best global checkpoint by validation Install LogLoss |
| Seed | 42 (single seed drives partition, init, and per-round sampler arrangements) |
| Official test set | never loaded, never evaluated, by any federated stage |

Additional fixed facts verified by the frozen audit machinery:

- The federated training pool is exactly **3,137,266 rows**, partitioned across exactly **10 clients**, with **zero overlap** with the frozen validation split (348,586 rows).
- Partition identity is hash-pinned: alpha=1.0 -> `c382d47f5824f64a`; alpha=0.5 -> `9603ddc9facc973d`; alpha=0.1 -> `23146aebf184345b`. Each result JSON's recorded partition hash equals the manifest value, and the frozen audit re-derives each hash from the npz contents.
- The only experimental variable between cells at fixed alpha is the model family (SSL vs no-SSL); the only variable at fixed model family is alpha.

The two model families:

- **No-SSL family** — Transformer + MMoE trained jointly on click and install heads (2,502,930 parameters; 138 state entries; supervised `supervised_forward` evaluation path, Stage 6 behavior).
- **SSL family** — `SSLTransformerMMoE` (2,527,698 parameters; 142 state entries), trained with the joint objective 0.6 x supervised + 0.4 x NT-Xent contrastive loss over two-view `FeatureCorruptionAugmentation` (corruption rate 0.15, temperature 0.2, projection dimension 64). Evaluation uses the same supervised path as the no-SSL family.

Model identity was verified directly against checkpoint payloads (state-entry counts and parameter sums) during this audit, and the recorded `ssl` config dicts in the SSL result JSONs are identical across the three SSL cells (`{alpha: 0.6, temperature: 0.2, projection_dim: 64, corruption_rate: 0.15}`).

## 2. Complete results matrix

All metrics are global validation metrics on the frozen 348,586-row validation split, at each cell's selected round. Exact values as recorded in the artifacts.

| alpha | SSL | Experiment | Best round | Install LL | Install AUC | Click LL | Click AUC | Runtime (s) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | no | `stage12_fed_nossl_alpha_1_0_seed42` | 10 | 0.262026822 | 0.908208787 | 0.269312284 | 0.907604516 | 20,397.49 | complete |
| 1.0 | yes | `stage12_fed_ssl_alpha_1_0_seed42` | 10 | 0.274393036 | 0.897050083 | 0.285715126 | 0.893344581 | 41,460.69 | complete |
| 0.5 | no | `stage12_fed_nossl_alpha_0_5_seed42` (Stage 10B) | 10 | 0.265536048 | 0.908881783 | 0.271194215 | 0.907656491 | 5,226.76* | complete |
| 0.5 | yes | `stage12_fed_ssl_alpha_0_5_seed42` (Stage 11) | 10 | 0.273903786 | 0.899003267 | 0.286338097 | 0.894202173 | 63,523.83 | complete |
| 0.1 | no | `stage12_fed_nossl_alpha_0_1_seed42` | 7 | 0.287310072 | 0.898739874 | 0.305709367 | 0.900956631 | 33,560.50 | complete |
| 0.1 | yes | `stage12_fed_ssl_alpha_0_1_seed42` | 9 | 0.303964521 | 0.889158964 | 0.347246284 | 0.890221179 | 64,510.01 | complete |

\* Stage 10B's recorded `total_elapsed_seconds` is a documented metadata anomaly (Section 7); the sum of its per-round wall times is ~25,494 s.

**Extraction path (verbatim, no guessing):** every cell's best-round metrics were computed by indexing the artifact's own `history` at `best_round` and reading `validation_metrics` — the same values the runner wrote at training time and the same values `reload_verified=true` / `reloaded_metrics_match=true` re-verified by reloading the best checkpoint and re-evaluating on validation. All five externally-known cells were additionally re-verified programmatically in this audit to within 5e-9 of the values above. Stage 10B's metrics were extracted from `artifacts/federated/stage10/stage10b_result.json` in this audit, not carried over from any earlier report.

Stage 10B (protected artifact) extracted facts:

- `status=complete`, 10/10 rounds, `best_round=10`
- `best_val_install_logloss=0.26553604847797624`
- best-round `validation_metrics`: rows 348,586; Install LL 0.265536048; Install AUC 0.908881783; Click LL 0.271194215; Click AUC 0.907656491
- 10-entry per-round history; `reload_verified=true`, `reloaded_metrics_match=true`; device `cuda`
- Runtime metadata anomaly documented in Section 7 (artifact metadata only; per-round metrics internally consistent)

## 3. Effect of Non-IID intensity (alpha) within each model family

All comparisons below are descriptive observations under this specific protocol (10 rounds, 1 local epoch, batch 1,024, AdamW lr 3e-4); no causal claim is made and no configuration is characterized as best or worst.

**No-SSL family (validation Install LogLoss):** 0.262026822 (alpha=1.0) -> 0.265536048 (alpha=0.5) -> 0.287310072 (alpha=0.1). The validation Install LogLoss increased as alpha decreased under this protocol. The alpha=1.0-to-0.5 increment is +0.003509; the alpha=0.5-to-0.1 increment is +0.021774 — most of the change occurs between alpha=0.5 and alpha=0.1. Validation Install AUC moved 0.908208787 -> 0.908881783 -> 0.898739874: +0.000673 between alpha=1.0 and 0.5, then -0.010142 at alpha=0.1.

**SSL family (validation Install LogLoss):** 0.274393036 (alpha=1.0) -> 0.273903786 (alpha=0.5) -> 0.303964521 (alpha=0.1): -0.000489 between alpha=1.0 and 0.5, then +0.030061 at alpha=0.1. Validation Install AUC moved 0.897050083 -> 0.899003267 -> 0.889158964: +0.001953 between alpha=1.0 and 0.5, then -0.009844 at alpha=0.1.

**Shared descriptive pattern:** in both families, the largest validation change associated with the harshest tested Non-IID setting occurs in the alpha=0.5-to-0.1 step, not the alpha=1.0-to-0.5 step; between alpha=1.0 and 0.5 the validation metrics are nearly unchanged (differences of order 5e-4 to 3.5e-3), while at alpha=0.1 Install LogLoss is 0.022-0.030 higher than at alpha=0.5 in both families. Neither family's metric ordering across alpha is claimed to generalize beyond this protocol and this single seed.

**Auxiliary Click metrics under the same protocol:** in the no-SSL family, Click LogLoss increased as alpha decreased (0.269312284 -> 0.271194215 -> 0.305709367) and Click AUC was nearly identical between alpha=1.0 and 0.5 (+0.000052) and lower at alpha=0.1 (-0.006700). In the SSL family, Click LogLoss increased from alpha=1.0 to 0.5 (+0.000623) and increased further at alpha=0.1 (+0.060908 from alpha=0.5); Click AUC increased slightly from alpha=1.0 to 0.5 (+0.000858) and decreased at alpha=0.1 (-0.003981 from alpha=0.5).

## 4. SSL vs No-SSL at matched alpha

All differences are absolute validation-metric differences computed from the recorded artifacts (SSL minus No-SSL), under the fixed SSL hyperparameters: supervised weight 0.6, contrastive weight 0.4, corruption rate 0.15, temperature 0.2, projection dim 64. No global ranking of the model families is claimed.

| alpha | Install LL No-SSL | Install LL SSL | Delta Install LL | Install AUC No-SSL | Install AUC SSL | Delta Install AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.262026822 | 0.274393036 | +0.012366 | 0.908208787 | 0.897050083 | -0.011159 |
| 0.5 | 0.265536048 | 0.273903786 | +0.008368 | 0.908881783 | 0.899003267 | -0.009879 |
| 0.1 | 0.287310072 | 0.303964521 | +0.016654 | 0.898739874 | 0.889158964 | -0.009581 |

| alpha | Click LL No-SSL | Click LL SSL | Delta Click LL | Click AUC No-SSL | Click AUC SSL | Delta Click AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.269312284 | 0.285715126 | +0.016403 | 0.907604516 | 0.893344581 | -0.014260 |
| 0.5 | 0.271194215 | 0.286338097 | +0.015144 | 0.907656491 | 0.894202173 | -0.013454 |
| 0.1 | 0.305709367 | 0.347246284 | +0.041537 | 0.900956631 | 0.890221179 | -0.010735 |

Under the tested hyperparameters and this protocol, at all three matched alpha values the SSL configuration's validation metrics differ from the no-SSL configuration's in the same direction: Install LogLoss is higher for the SSL configuration by 0.0084-0.0167, Install AUC is lower by 0.0096-0.0112, Click LogLoss is higher by 0.0151-0.0415, and Click AUC is lower by 0.0107-0.0143. These are observations of one fixed SSL configuration under this protocol at one seed; they do not establish that the SSL approach is intrinsically beneficial or harmful, and no claim is made beyond the tested configuration.

## 5. Convergence and checkpoint selection

Checkpoint selection was fixed in advance: the global checkpoint with the lowest validation Install LogLoss across the 10 rounds. Round-level Install LogLoss trajectories (verbatim from the artifacts, 6 decimal places):

| Round | No-SSL alpha=1.0 | SSL alpha=1.0 | No-SSL alpha=0.5 (Stage 10B) | SSL alpha=0.5 (Stage 11) | No-SSL alpha=0.1 | SSL alpha=0.1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.450684 | 0.340365 | 0.542303 | 0.339669 | 0.454852 | 0.389630 |
| 2 | 0.289534 | 0.302554 | 0.289734 | 0.299331 | 0.307176 | 0.306231 |
| 3 | 0.281197 | 0.289397 | 0.280502 | 0.285456 | 0.306714 | 0.317884 |
| 4 | 0.275955 | 0.283944 | 0.279772 | 0.282631 | 0.308393 | 0.320017 |
| 5 | 0.273321 | 0.281977 | 0.274396 | 0.279321 | 0.295678 | 0.312523 |
| 6 | 0.270071 | 0.278527 | 0.275587 | 0.279006 | 0.296795 | 0.313369 |
| 7 | 0.268329 | 0.277702 | 0.271264 | 0.276875 | **0.287310** | 0.308537 |
| 8 | 0.265890 | 0.276860 | 0.267468 | 0.275165 | 0.291686 | 0.305642 |
| 9 | 0.263039 | 0.275104 | 0.269273 | 0.276123 | 0.290523 | **0.303965** |
| 10 | **0.262027** | **0.274393** | **0.265536** | **0.273904** | 0.290748 | 0.307488 |

Bold marks each trajectory's selected round (lowest validation Install LogLoss). Descriptive notes:

- **No-SSL alpha=1.0** — Install LogLoss decreased every round from 1 through 10; the selected round is 10, so the metric was still improving at the end of the observed horizon. Click LogLoss and both AUCs show the same monotone pattern.
- **SSL alpha=1.0** — decreased every round through round 10; selected round 10, still improving at the horizon end.
- **No-SSL alpha=0.5 (Stage 10B)** — decreased through round 8 (0.267468) apart from a small rise at round 6 (0.274396 -> 0.275587), rose slightly at round 9 (0.269273), then reached its lowest observed value at round 10 (0.265536); selected round 10. The trajectory fluctuated slightly after round 5 but ended at its best observed value.
- **SSL alpha=0.5 (Stage 11)** — decreased through round 8 (0.275165), rose slightly at round 9 (0.276123), then reached its lowest observed value at round 10 (0.273904); selected round 10. The post-round-8 fluctuation is bounded by about 0.001.
- **No-SSL alpha=0.1** — best observed Install LogLoss at round 7 (0.287310); rounds 8-10 remained above the round-7 value (0.291686 / 0.290523 / 0.290748). The metric fluctuated within a narrow band (about 0.003) after the best round without improving on it. This is the only no-SSL cell whose selected checkpoint is not the final round.
- **SSL alpha=0.1** — best observed value at round 9 (0.303965); round 10 (0.307488) was above round 9. The trajectory fluctuated after the best round within the observed horizon.

The observed horizon is 10 rounds; whether any of these trajectories would continue improving beyond round 10 is not established by these experiments and is not inferred.

## 6. Auxiliary Click task

Click is the auxiliary task in this project's design; Install is the primary target. Across all six cells, the auxiliary Click validation metrics co-moved with the Install metrics: cells with lower Install LogLoss also have lower Click LogLoss and higher Click AUC, both across alpha at fixed family and across the SSL/no-SSL comparison at fixed alpha. Within every trajectory, Click metrics improved round by round alongside Install metrics; no cell shows the auxiliary task improving while the primary task degrades, or the reverse.

This co-movement is consistent with the multi-task setup training a shared representation for both heads. However, the experimental design includes no ablation that removes the Click head or re-weights the tasks (lambda_click = lambda_install = 1.0 in all cells), so the design does not directly support a claim that the Click task causes Install improvement. The co-movement is reported as an observation of this protocol only.

## 7. Integrity and reproducibility

The repository's frozen audit machinery (`scripts/audit_stage12_matrix.py`, explicitly read-only — its only write is the documentation snapshot `reports/stage12/stage12_matrix.json`) was executed during this audit and reported **ALL CHECKS PASSED**, covering:

- **Completion (checks A-D):** all six cells have complete result artifacts (status `complete`, 10/10 rounds). Stage 11 alpha=0.5 SSL intact; Stage 10B alpha=0.5 no-SSL intact; all four Stage 12 cells complete. Each result JSON records `reload_verified=true` and `reloaded_metrics_match=true` (best checkpoint reloaded and re-evaluated, metrics matched) — available for all six cells.
- **Partition identity (check E):** alpha=1.0 `c382d47f5824f64a`; alpha=0.5 `9603ddc9facc973d`; alpha=0.1 `23146aebf184345b` — stored = manifest = re-derived from npz contents (the audit re-derives each hash from the frozen Stage 9 code). The audit also re-verified: 10 non-empty, mutually disjoint clients; row counts sum to exactly 3,137,266; composed client rows are unique and disjoint from validation (0 overlap rows); per-state client label sums conserve the global joint-label counts.
- **Frozen data/protocol (check F):** preprocessing hash `c3a62caf13326617`; frozen split hashes `dd37605169615442` (train) / `3cd8370ebdecb305` (validation); split sizes 3,137,266 / 348,586 of 3,485,852; train-labels hash `800e2aa9d0bf7b79`. FederatedConfig values asserted frozen: 10 clients, all-clients-every-round, 10 rounds, 1 local epoch, batch 1,024, AdamW, lr 0.0003, weight decay 0.01, lambda_click = lambda_install = 1.0, sample-weighted FedAvg, selection by global validation Install LogLoss, no local optimizer persistence, sampler epoch mode `round_minus_one`, seed 42, partition seed 42. Cold initialization: no checkpoint is loaded by any cell (Stage 6/8/10B/11 checkpoints are never load paths in the Stage 12 runner); the Stage 10 protocol report documents the cold-init proof (all 138 state entries differ from the Stage 6 checkpoint at initialization).
- **Official test protection:** the audit scans the Stage 12 preparation sources and confirms the official test set is never loaded or constructed by any federated training path; no official-test metrics exist in any artifact by design.
- **SSL identity (check G):** the audit asserts Stage 11-equality of the SSL constants (joint weight 0.6, temperature 0.2, corruption 0.15, projection 64; 2,527,698 params / 142 state entries); this audit additionally verified the parameter sums and state-entry counts directly against the `best.pt` payloads of the SSL and no-SSL cells, and confirmed the three SSL result JSONs record identical `ssl` config dicts.
- **No protected artifact was modified:** the audit and this report's extraction steps are read-only with respect to `artifacts/`; the only files written are documentation files under `reports/stage12/`.

Documented metadata quirks (reported, not corrected — read-only audit):

1. **Recorded-protocol fields:** every result JSON's `protocol.use_ssl` / `protocol.alpha` fields hold the runner's default values (False / 0.5) rather than the effective cell settings — including Stage 11. The effective identity is recorded elsewhere in each artifact: `stage12.use_ssl` / `stage12.alpha` (Stage 12 cells), the `ssl` config dict, the partition hash, and the experiment identifier, all of which agree with each other. The training-relevant protocol fields (clients, rounds, epochs, batch, optimizer, lr, weight decay, aggregation, participation, selection metric, seed) are recorded correctly and are identical across all six cells.
2. **Stage 10B runtime anomaly:** Stage 10B's `total_elapsed_seconds` (5,226.76 s) is inconsistent with the sum of its per-round wall times (~25,494 s); the recorded total equals the sum of rounds 1-2 only (5,225.6 s), suggesting the elapsed-time accumulator stopped updating after round 2. Its recorded `started_at`/`completed_at` span (34 s) is likewise inconsistent with the training log. The per-round metrics are internally consistent and reload-verified; the anomaly affects runtime metadata only.

## 8. Limitations

- **One random seed.** All cells use seed 42. No variance estimate exists; differences between cells — including the SSL-vs-no-SSL differences — cannot be attributed to the manipulated variable rather than seed-level noise.
- **Simulated federation.** Ten simulated clients partitioning one dataset by Dirichlet statistics, not real organizations with independently collected data, privacy constraints, or system heterogeneity.
- **Ten communication rounds, one local epoch.** Five of six cells were still improving (or had only just peaked) at round 10, so the protocol may understate achievable performance and may interact with the alpha effect (the harshest Non-IID cells moved more slowly within the horizon).
- **Only three alpha values.** 1.0, 0.5, and 0.1 tested; nothing between 0.5 and 0.1 or beyond 1.0.
- **Fixed SSL hyperparameters.** Supervised weight 0.6, contrastive weight 0.4, corruption 0.15, temperature 0.2 were fixed for all cells and were not tuned per alpha; the SSL observations are specific to this one configuration.
- **Validation-based selection.** Model selection used validation metrics; no official-test evaluation occurred at any point by design, so no test-set generalization numbers exist in this matrix.
- **Stage 6 historical sampler asymmetry.** Stage 6's historical Protocol-B batch-arrangement asymmetry (documented in `reports/stage8_experiments_report.md`: a real, reproducible trajectory divergence beginning in epoch 2, approximately 5% of init movement at 4 divergent steps, with B historical and A/C/D corrected under identical data, seed, objective, and batches-per-epoch) must not be silently ignored when comparing federated results to centralized results trained under Stage 6 conventions. The federated rounds use the corrected convention (`round_minus_one`), which matches the centralized epoch-1 arrangement at round 1; any centralized-vs-federated comparison must state which sampler convention each side used.
- **Descriptive comparisons only.** No statistical significance testing was performed or is appropriate at one seed; all cross-cell differences are reported as raw absolute differences.
- **Runtime metadata anomalies** (Stage 10B elapsed-time accumulator; the recorded `protocol.use_ssl`/`protocol.alpha` default-value quirk) are documented in Section 7 and do not affect the metrics.

## 9. Thesis-ready factual conclusions

1. Under the frozen protocol (10 simulated clients, seed 42, 10 rounds, 1 local epoch, batch 1,024, AdamW lr 0.0003, weight decay 0.01, sample-weighted FedAvg, all-clients participation, cold seed-42 initialization, validation-Install-LogLoss checkpoint selection), all six cells of the 2x3 matrix (SSL x alpha in {1.0, 0.5, 0.1}) completed 10/10 rounds and produced reload-verified best checkpoints; all metrics reported here are on the frozen 348,586-row validation split, and the official test set was never loaded by any stage.

2. In the no-SSL family, validation Install LogLoss increased as alpha decreased under this protocol (0.262027 at alpha=1.0, 0.265536 at alpha=0.5, 0.287310 at alpha=0.1); the change between alpha=1.0 and alpha=0.5 was small (+0.003509) compared with the change between alpha=0.5 and alpha=0.1 (+0.021774). Validation Install AUC was nearly unchanged between alpha=1.0 and 0.5 and lower at alpha=0.1.

3. In the SSL family, validation Install LogLoss was nearly unchanged between alpha=1.0 and alpha=0.5 (0.274393 -> 0.273904) and higher at alpha=0.1 (0.303965); validation Install AUC rose slightly from alpha=1.0 to 0.5 and was lower at alpha=0.1. As in the no-SSL family, the largest change occurred in the alpha=0.5-to-0.1 step.

4. At matched alpha, and under the fixed SSL configuration (supervised weight 0.6, contrastive weight 0.4, corruption 0.15, temperature 0.2, projection dim 64), the SSL configuration's validation metrics differed from the no-SSL configuration's in the same direction at all three alpha values: Install LogLoss higher by 0.0084-0.0167, Install AUC lower by 0.0096-0.0112, Click LogLoss higher by 0.0151-0.0415, Click AUC lower by 0.0107-0.0143. This is an observation of one fixed SSL configuration under this protocol at one seed; it does not establish that the SSL approach is intrinsically beneficial or harmful.

5. Checkpoint selection by validation Install LogLoss selected round 10 in four of six cells (No-SSL alpha=1.0, SSL alpha=1.0, Stage 10B, Stage 11), round 7 for No-SSL alpha=0.1, and round 9 for SSL alpha=0.1. The two alpha=0.1 cells are the only cells whose best observed value did not occur at the final round; within the observed 10-round horizon they fluctuated in a narrow band after their best rounds. No behavior beyond round 10 is inferred.

6. Across all six cells, the auxiliary Click validation metrics co-moved with the primary Install metrics at every observed round; however, the design includes no task-removal or task-reweighting ablation, so the experiments do not directly support a causal claim that the auxiliary Click task improves the Install task.

7. Integrity: all three partition hashes (alpha=1.0 `c382d47f5824f64a`, alpha=0.5 `9603ddc9facc973d`, alpha=0.1 `23146aebf184345b`) verify as stored = manifest = re-derived from contents; the 3,137,266 federated rows are disjoint from validation (0 overlap) and conserve the global joint-label counts; all six cells record `reload_verified=true` / `reloaded_metrics_match=true`; two metadata anomalies (Stage 10B elapsed-time accumulator; the result-JSON `protocol.use_ssl`/`protocol.alpha` default-value quirk) are documented and affect metadata only, not metrics.

8. The metric ordering across cells is conditional on this protocol, this seed, and this horizon: with one seed, ten rounds, and three alpha values, the observed differences are descriptive measurements of this experimental matrix, not established orderings of the methods in general.

---

*Audit trail: audit script `scripts/audit_stage12_matrix.py` run 2026-09-24 (ALL CHECKS PASSED, read-only); all five externally-known result cells re-verified programmatically against their artifacts to <5e-9; Stage 10B extracted verbatim from its protected artifact; JSON/CSV generated by `reports/stage12/generate_final_matrix_reports.py` (read-only extraction, no artifact writes).*
