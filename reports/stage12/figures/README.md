# Stage 12 Final Figures — Plotting Metadata

Read-only plotting stage for the finalized federated experimental matrix.
Generated 2026-09-24. No training, no evaluation, and no artifact modification
was performed in this stage; only the files in this `figures/` directory were
created (including the generator script `generate_figures.py`).

## 1. Source

- Authoritative data source:
  `reports/stage12/final_federated_matrix.json` (read verbatim by
  `generate_figures.py`; no metric value is hard-coded in the script).
- Secondary cross-check during validation: the six underlying result JSONs in
  `artifacts/federated/stage10/`, `artifacts/federated/stage11/`, and
  `artifacts/federated/stage12/` (read-only).

## 2. The six experiments plotted

| alpha | SSL   | Experiment                        | Stage   | Best round |
|------:|-------|-----------------------------------|---------|-----------:|
| 1.0   | No    | `stage12_fed_nossl_alpha_1_0_seed42` | Stage 12 | 10 |
| 1.0   | Yes   | `stage12_fed_ssl_alpha_1_0_seed42`   | Stage 12 | 10 |
| 0.5   | No    | `stage12_fed_nossl_alpha_0_5_seed42` (Stage 10B) | Stage 10B | 10 |
| 0.5   | Yes   | `stage12_fed_ssl_alpha_0_5_seed42` (Stage 11)    | Stage 11  | 10 |
| 0.1   | No    | `stage12_fed_nossl_alpha_0_1_seed42` | Stage 12 | 7  |
| 0.1   | Yes   | `stage12_fed_ssl_alpha_0_1_seed42`   | Stage 12 | 9  |

All six cells are complete (10/10 rounds, reload-verified in the JSON).

## 3. Figure files

| File (`.png` + `.pdf`) | Content |
|---|---|
| `fig_4_1_install_logloss_vs_alpha` | Final (best-round) validation Install LogLoss vs Dirichlet alpha, No-SSL and SSL series |
| `fig_4_2_install_auc_vs_alpha` | Final (best-round) validation Install AUC vs Dirichlet alpha, No-SSL and SSL series |
| `fig_4_3_nossl_convergence` | Validation Install LogLoss per round (1–10), No-SSL at alpha = 1.0 / 0.5 / 0.1 |
| `fig_4_4_ssl_convergence` | Validation Install LogLoss per round (1–10), SSL at alpha = 1.0 / 0.5 / 0.1 |
| `fig_4_5_ssl_vs_nossl_install_logloss` | Grouped bars: matched No-SSL vs SSL pairs at each alpha (Install LogLoss) |
| `fig_4_6_click_logloss_vs_alpha` | Final (best-round) validation Click LogLoss vs Dirichlet alpha (auxiliary task) |

## 4. Metrics used

- Best-round metrics per cell: `best_val_install_logloss`,
  `best_val_install_auc`, `best_val_click_logloss` (from the result JSONs,
  mirrored in `final_federated_matrix.json`).
- Convergence figures: exact per-round `validation_metrics.val_install_logloss`
  for rounds 1–10, unsmoothed, no interpolation.
- Checkpoint-selection criterion: lowest validation Install LogLoss across the
  10 rounds. The selected round is circled in Figures 4-3 and 4-4 (alpha=0.1:
  No-SSL round 7, SSL round 9; all other cells selected round 10).

## 5. Validation metrics only

Every plotted value is a **validation** metric. The official test set was not
used for model selection, not evaluated, and does not appear in any figure.

## 6. Single seed

All experiments use seed 42 only. Because there is a single run per cell, no
error bars or confidence intervals are shown, and the figures do not support
claims of statistical significance.

## 7. No training in this stage

This stage only read `final_federated_matrix.json` and wrote the figure files
in this directory. No training or evaluation process was launched, and no file
under `artifacts/` (checkpoints, partitions, result JSONs) was modified.

## 8. Scope of interpretation

The plots show the measured validation values under the frozen protocol
(10 clients, seed 42, 10 rounds, 1 local epoch, batch 1024, AdamW lr 3e-4 /
wd 0.01, sample-weighted FedAvg, cold init). They do not establish universal
superiority of either method, causality, or generalization to real federated
deployments.
