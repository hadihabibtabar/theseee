# Final Federated Matrix — Compact Reference

**Date:** 2026-09-24. Read-only audit output; no training performed, no protected artifact modified.
Full detail: `final_federated_results_report.md` (narrative), `final_federated_matrix.json` (machine-readable), `final_federated_matrix.csv` (table).

## Protocol (frozen, identical in all six cells)

10 simulated clients · Dirichlet alpha in {1.0, 0.5, 0.1} · partition seed 42 · all 10 clients participate every round · 10 rounds · 1 local epoch · batch 1,024 · AdamW lr 0.0003, weight decay 0.01 · sample-weighted FedAvg · fresh local optimizer each client/round · cold seed-42 init (no checkpoint load) · best checkpoint by validation Install LogLoss · official test never loaded.

Data: 3,137,266 federated train rows / 348,586 validation rows; zero client–validation overlap.

## Models

| Family | Model | Params | State entries | Objective |
| --- | --- | --- | --- | --- |
| No-SSL | Transformer + MMoE (click+install heads) | 2,502,930 | 138 | supervised |
| SSL | `SSLTransformerMMoE` | 2,527,698 | 142 | 0.6 supervised + 0.4 NT-Xent (two-view corruption 0.15, temperature 0.2, projection 64) |

## Results (global validation at selected round)

| alpha | SSL | Experiment | Best round | Install LL | Install AUC | Click LL | Click AUC | Runtime (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | no | `stage12_fed_nossl_alpha_1_0_seed42` | 10 | 0.262026822 | 0.908208787 | 0.269312284 | 0.907604516 | 20,397.49 |
| 1.0 | yes | `stage12_fed_ssl_alpha_1_0_seed42` | 10 | 0.274393036 | 0.897050083 | 0.285715126 | 0.893344581 | 41,460.69 |
| 0.5 | no | Stage 10B (`stage12_fed_nossl_alpha_0_5_seed42`) | 10 | 0.265536048 | 0.908881783 | 0.271194215 | 0.907656491 | 5,226.76* |
| 0.5 | yes | Stage 11 (`stage12_fed_ssl_alpha_0_5_seed42`) | 10 | 0.273903786 | 0.899003267 | 0.286338097 | 0.894202173 | 63,523.83 |
| 0.1 | no | `stage12_fed_nossl_alpha_0_1_seed42` | 7 | 0.287310072 | 0.898739874 | 0.305709367 | 0.900956631 | 33,560.50 |
| 0.1 | yes | `stage12_fed_ssl_alpha_0_1_seed42` | 9 | 0.303964521 | 0.889158964 | 0.347246284 | 0.890221179 | 64,510.01 |

\* Stage 10B recorded `total_elapsed_seconds` is a metadata anomaly (accumulator stopped after round 2); sum of per-round walls ~25,494 s.

All six cells: status `complete`, 10/10 rounds, `reload_verified=true`, `reloaded_metrics_match=true`.

## Partitions (hash-pinned, stored = manifest = re-derived)

| alpha | Hash | Rows | Clients | Validation overlap |
| --- | --- | --- | --- | --- |
| 1.0 | `c382d47f5824f64a` | 3,137,266 | 10 | 0 |
| 0.5 | `9603ddc9facc973d` | 3,137,266 | 10 | 0 |
| 0.1 | `23146aebf184345b` | 3,137,266 | 10 | 0 |

## SSL-vs-No-SSL absolute differences (SSL − No-SSL, validation)

| alpha | Delta Install LL | Delta Install AUC | Delta Click LL | Delta Click AUC |
| --- | --- | --- | --- | --- |
| 1.0 | +0.012366 | −0.011159 | +0.016403 | −0.014260 |
| 0.5 | +0.008368 | −0.009879 | +0.015144 | −0.013454 |
| 0.1 | +0.016654 | −0.009581 | +0.041537 | −0.010735 |

Observation of one fixed SSL configuration (sup 0.6 / NT-Xent 0.4 / corruption 0.15 / temperature 0.2 / projection 64) under this protocol at seed 42; no global ranking claimed.

## Notes

- Checkpoint selection picked round 10 in four cells; round 7 (No-SSL alpha=0.1) and round 9 (SSL alpha=0.1) — the two alpha=0.1 cells fluctuated in a narrow band after their best rounds within the observed 10-round horizon.
- Every result JSON's `protocol.use_ssl`/`protocol.alpha` fields hold runner defaults (False/0.5), not the effective settings (metadata quirk, documented; effective identity verified via `stage12.*` fields, `ssl` dict, partition hashes, identifiers).
- No official-test metrics exist anywhere by design; all numbers above are validation metrics.
- Limitations (single seed, simulated clients, 10 rounds, 1 local epoch, three alpha values, fixed SSL hyperparameters, Stage 6 sampler-asymmetry caveat for centralized-vs-federated comparisons): see `final_federated_results_report.md` Section 8.
