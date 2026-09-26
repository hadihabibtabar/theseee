# Final Experiment Audit — Complete Repository (Chapter 4 Source)

**Audit date:** 2026-09-25  
**Nature:** READ-ONLY audit. No training launched, no artifact modified. Every number below is read from a named artifact; `NOT_VERIFIED` marks values that could not be verified.

## 1. EXECUTIVE SUMMARY

- The final six-cell matrix is **complete and verified**: Cells A/B (alpha_D=1.0), C/D (alpha_D=0.5), E/F (alpha_D=0.1), all `status=complete`, 10/10 rounds, `reload_verified=true`, `reloaded_metrics_match=true`.
- Cell lettering confirmed from artifacts: **C = Stage 10B** (federated No-SSL alpha_D=0.5) and **D = Stage 11** (federated SSL alpha_D=0.5); no separate Stage 12 alpha_0_5 runs exist (AN-7).
- Partition hashes verified in manifests AND cell result JSONs: alpha_D=1.0 `c382d47f5824f64a`, 0.5 `9603ddc9facc973d`, 0.1 `23146aebf184345b`; each partition totals 3,137,266 rows.
- SSL identity identical across B/D/F: `SSLTransformerMMoE`, 2,527,698 params, 142 state entries, supervised 0.6 / NT-Xent 0.4, T=0.2, corruption 0.15, projection 64.
- Canonical Stage 6 metric re-verified by torch reload in this audit: best val Install LogLoss = 0.2562108886731335 (epoch 3, None params) from `artifacts/checkpoints/centralized_transformer_mmoe_best.pt`.
- Key anomaly (AN-1): Stage 10B `total_elapsed_seconds`=5,226.8 conflicts with Σ round walls=25,494.0; reported, not corrected.
- Official test set: never loaded/evaluated anywhere (no test hash in any result JSON; report statements; runner design).

## 2. COMPLETE EXPERIMENT TIMELINE

| When (UTC, artifact-sourced) | Event | Category | Source |
|---|---|---|---|
| 2026-09 (early) | Stages 1-5: preprocessing, splits, loaders, Transformer, MMoE | B | reports/preprocessing_report.md, transformer_report.md, mmoe_report.md |
| 2026-09-15..16 | Stage 6 centralized run (canonical B), 3 epochs, 22.7k+58.4k+3.2k s | D (reused) | reports/centralized_training_report.md |
| (Stage 7) | SSL mechanism + real-data smoke | B/C | reports/ssl_report.md |
| 2026-09-17 | Stage 8 Experiment A complete; sampler investigation; B-not-retrained decision | B/C | reports/stage8_experiments_report.md |
| 2026-09-18 | Stage 8 Experiment C complete | B | same |
| (by 2026-09-19) | Stage 8 Experiment D complete (result JSON present) | B | artifacts/checkpoints/stage8/experiment_d_transformer_mmoe_ssl_result.json |
| 2026-09-19 13:32 | stage10_smoke_round1 | C | artifacts/federated/smoke/stage10_smoke_round1_result.json |
| 2026-09-19 20:45 | Stage 10B starts (partition alpha_D=0.5) | A | artifacts/federated/stage10/stage10b_result.json |
| 2026-09-20..21 | Stage 10B r9/c08 diagnostic; Stage 11 CUDA diagnostic + launcher retries | C | diagnostic_report.json files |
| 2026-09-21 07:28 | Stage 11 completes (10/10, reload-verified) | A | artifacts/federated/stage11/stage11_result.json |
| 2026-09-21 15:49 | Stage 12 Cell A starts | A | artifacts/federated/stage12/stage12_fed_nossl_alpha_1_0_seed42/stage12_fed_nossl_alpha_1_0_seed42_result.json |
| 2026-09-22 06:59 | Stage 12 Cell B starts | A | artifacts/federated/stage12/stage12_fed_ssl_alpha_1_0_seed42/stage12_fed_ssl_alpha_1_0_seed42_result.json |
| 2026-09-22 22:28 | Stage 12 Cell E starts | A | artifacts/federated/stage12/stage12_fed_nossl_alpha_0_1_seed42/stage12_fed_nossl_alpha_0_1_seed42_result.json |
| 2026-09-23 20:55 | Stage 12 Cell F starts | A | artifacts/federated/stage12/stage12_fed_ssl_alpha_0_1_seed42/stage12_fed_ssl_alpha_0_1_seed42_result.json |
| 2026-09-24/25 | Cells E/F (alpha_D=0.1) complete; matrix frozen; audit | A | result JSONs completed_at |

## 3. STAGE 8 — A/B/C/D (centralized, pre-final)

All four share the frozen Stage 4 backbone (d_model 128, nhead 8, 6 layers, FFN 128, dropout 0.1), batch 1024, AdamW 3e-4/wd 1e-2, seed 42, 3 epochs, selection by val Install LogLoss. A/C/D use the corrected per-epoch reshuffle; canonical B (Stage 6) predates the sampler fix (AN-4). All are **pre-final**: none is a six-cell matrix cell.

| Exp | Architecture | SSL | Params | Best epoch | Val Install LL | Val Install AUC | Val Click LL | Val Click AUC |
|---|---|---|---|---|---|---|---|---|
| A | TransformerBaseline (install-only) | no | 2,352,129 | 3 | 0.2589328218594451 | 0.9120110869407654 | — | — |
| B (canonical Stage 6) | Transformer+MMoE | no | 2,502,930 | 3 | 0.2562108886731335 (torch-reloaded) | 0.9132 (report) | 0.2640 (report) | 0.9114 (report) |
| C | SSLTransformerBaseline (install-only) | yes | 2,376,897 | 3 | 0.2697706961950431 | 0.9016668796539307 | — | — |
| D | SSLTransformerMMoE | yes | 2,527,698 | 3 | 0.2632836117438265 | 0.9072108864784241 | 0.2718169281140416 | 0.9048312306404114 |

Epoch-level metrics for A/C/D are stored verbatim in the JSON under `experiments.stage8_*`. Checkpoints: `artifacts/checkpoints/stage8/experiment_{a,c,d}_*_best.pt` + canonical `artifacts/checkpoints/centralized_transformer_mmoe_best.pt`. Sampler diagnostic (H vs C batch digests) in JSON under `diagnostics`.

## 4. STAGE 9 — PARTITIONING

- Method `dirichlet_joint_label_proportional_v1` (per joint-state Dirichlet draws, largest-remainder rounding, contiguous cuts), seed 42, exactly 10 clients, train rows only (3,137,266), validation excluded (overlap 0).
- Primary partition alpha_D=0.5 created in Stage 9; alpha_D=1.0 and 0.1 created later via the identical frozen code path for Stage 12 (`scripts/run_stage12_partitions.py`).
- Independent post-run audit: sizes sum, uniqueness, zero validation overlap, determinism (byte-identical re-runs); 24 partition tests + 13 client-loader tests added (suite 281 passing at Stage 9 close).
- Per-client rows (all three partitions, from manifests):

| Client | alpha_D=1.0 | alpha_D=0.5 | alpha_D=0.1 |
|---|---|---|---|
| client_00 | 452,032 | 621,801 | 342,065 |
| client_01 | 460,777 | 544,574 | 1,200,830 |
| client_02 | 585,448 | 59,999 | 251,984 |
| client_03 | 151,573 | 488,090 | 266,440 |
| client_04 | 137,132 | 69,790 | 30,811 |
| client_05 | 272,725 | 89,988 | 11,892 |
| client_06 | 253,784 | 193,852 | 366,104 |
| client_07 | 488,550 | 201,902 | 44,021 |
| client_08 | 104,506 | 511,603 | 11,698 |
| client_09 | 230,739 | 355,667 | 611,421 |

- alpha_1.0_seed42: hash `c382d47f5824f64a`, rows 3,137,266, manifest `artifacts/federated/partitions/stage12_partition_alpha_1_0_seed42.json`.
- alpha_0.5_seed42: hash `9603ddc9facc973d`, rows 3,137,266, manifest `artifacts/federated/partition_alpha_0_5_seed42.json`.
- alpha_0.1_seed42: hash `23146aebf184345b`, rows 3,137,266, manifest `artifacts/federated/partitions/stage12_partition_alpha_0_1_seed42.json`.

## 5. STAGE 10B — FEDERATED BASELINE (FINAL CELL C)

- Purpose: 10-round federated No-SSL run on the alpha_D=0.5 partition — **this is final matrix Cell C**, not merely a baseline.
- Identity: Transformer+MMoE 2,502,930 params / 138 state entries; protocol dict identical to frozen Stage 12 protocol (verified field-by-field in result JSON `protocol`).
- Best round 10: Install LL 0.26553604847797624, Install AUC 0.908881783, Click LL 0.271194215, Click AUC 0.907656491.
- **Runtime inconsistency (AN-1):** `total_elapsed_seconds`=5,226.7641265 (JSON top level) vs Σ `round_wall_seconds`=25,493.95 (same JSON history) vs log line "total runtime: 5260.9 s (1.46 h)". Two of three agree; the per-round sum is the plausible compute total. Reported verbatim; artifact untouched.
- Round-1 metrics equal the integration smoke's round-1 metrics exactly (AN-8). Reload verification: true/true. Peak RSS 2,035.7 MiB.

## 6. STAGE 11 — SSL INTEGRATION (FINAL CELL D)

- Dual role: (1) SSL integration validation for the federated path — completed successfully with reload verification; (2) **final matrix Cell D** (SSL, alpha_D=0.5) because protocol and SSL constants are exactly the frozen Stage 12 configuration.
- Model identity (result JSON `ssl` dict + prior torch payload checks): `SSLTransformerMMoE`, 2,527,698 params, 142 state entries; supervised weight 0.6 (stored under key `ssl.alpha` — see AN-3), temperature 0.2, projection 64, corruption 0.15; NT-Xent; two views; supervised eval path unchanged.
- Best round 10: Install LL 0.27390378597608317, Install AUC 0.899003267, Click LL 0.286338097, Click AUC 0.894202173.
- Runtime 63523.8 s self-consistent with Σ round walls 63,522.6 s. Round-1 CUDA failure diagnosed (diagnostic_r1_c00); completed via 10 launcher attempts; final artifacts reload-verified.

## 7. SMOKE TESTS

| Test | Kind | What it validated | Result | Source |
|---|---|---|---|---|
| stage10_smoke_round1 | smoke | partition loaders + local train + FedAvg + global eval + checkpoint reload | PASS | artifacts/federated/smoke/stage10_smoke_round1_result.json |
| run_ssl_smoke (Stage 7) | smoke | SSL augmentations + NT-Xent CPU/CUDA parity + 142 params trained | PASS | reports/ssl_report.md |
| Stage 12 preflights (x4) | smoke/preflight | partition hash/rows/overlap, model identity, no-test guarantee | PASS | root stage12 logs |
| sampler diagnostic (Stage 8) | diagnostic | batch-order fingerprints H vs C protocols | complete; decision recorded | reports/stage8_experiments_report.md #4 |

The Stage 10 round-1 smoke is explicitly labeled in its own artifact: "TEMPORARY SMOKE TEST ARTIFACT … NOT the 10-round Stage 10B experiment and NOT a best-model checkpoint." It validated integration only and must not be quoted as a performance result.

## 8. DIAGNOSTIC AND REGRESSION TESTS

- **Stage 8 sampler protocol investigation (run_sampler_diagnostic.py)** — Protocol H (historical): identical batch-index digest bf973a7e6cf076ed... across epochs 1-3; Protocol C (corrected): distinct per epoch; C(epoch1)==H(epoch1) by construction
- **Stage 10B round-9 client_08 diagnostic (diag_stage10b_r9_c08.py)** — re-run completed cleanly; protected artifacts unchanged (verified in earlier audit run of scripts/audit_stage12_matrix.py) (artifact: `artifacts/federated/stage10/diagnostic_r9_c08/diagnostic_report.json`)
- **Stage 11 round-1 client_00 CUDA diagnostic (diagnose_stage11_cuda.py)** — CUDA error diagnosed; run completed via 10 launcher attempts (stage11_attempt1-10 logs); final result reload-verified (artifact: `artifacts/federated/stage11/diagnostic_r1_c00/diagnostic_report.json`)
- **Stage 10B runtime anomaly re-check (this audit)** — total_elapsed_seconds=5,226.8 vs sum(round_wall_seconds)=25,494.0; log prints 'total runtime: 5260.9 s (1.46 h)'

The historical sampler issue: Stage 6's `Trainer.fit` reset attempt was a silent no-op (`loader.sampler` vs `batch_sampler`); fingerprints proved Protocol H identical across epochs while Protocol C differs per epoch, with C(epoch 1) ≡ H(epoch 1). Consequence handled by documented decision, not silently.

## 9. STAGE 12 — FINAL SIX-CELL MATRIX

Frozen protocol (identical in all six result JSONs): 10 clients, 10 rounds, 1 local epoch, batch 1024, AdamW lr 3e-4 wd 1e-2, sample-weighted FedAvg, all clients every round, seed 42, cold init, fresh local optimizer per round, selection = global validation Install LogLoss, lambda_click=lambda_install=1.0. SSL cells add: 0.6·sup + 0.4·NT-Xent, corruption 0.15, T=0.2, projection 64.

| Cell | Stage | Experiment | Partition hash | Status | Reload | Peak RSS (MiB) | Total elapsed (s) | Σ round walls (s) |
|---|---|---|---|---|---|---|---|---|
| A | Stage 12 | stage12_fed_nossl_alpha_1_0_seed42 | `c382d47f5824f64a` | complete | true/true | 1536.5 | 20397.5 | 20393.9 |
| B | Stage 12 | stage12_fed_ssl_alpha_1_0_seed42 | `c382d47f5824f64a` | complete | true/true | 2210.7 | 41460.7 | 41459.3 |
| C | Stage 10B | stage10b (canonical Stage 10B federated baseline) | `9603ddc9facc973d` | complete | true/true | 2035.7 | 5226.8 | 25494.0 |
| D | Stage 11 | stage11 (canonical Stage 11 federated SSL) | `9603ddc9facc973d` | complete | true/true | 2779.6 | 63523.8 | 63522.6 |
| E | Stage 12 | stage12_fed_nossl_alpha_0_1_seed42 | `23146aebf184345b` | complete | true/true | 2501.3 | 33560.5 | 33559.3 |
| F | Stage 12 | stage12_fed_ssl_alpha_0_1_seed42 | `23146aebf184345b` | complete | true/true | 3085.1 | 64510.0 | 64508.3 |

AN-1 applies to Cell C only (elapsed vs Σ walls). GPU peaks: No-SSL cells ~3,604 MiB allocated; SSL cells ~7,185 MiB (per-round values in JSON `round_metrics`).

## 10. FULL ROUND-BY-ROUND RESULTS (60 rows)

| Cell | Round | Click LogLoss | Click AUC | Install LogLoss | Install AUC |
|---|---|---|---|---|---|
| A | 1 | 0.617073096962766 | 0.7294928431510925 | 0.45068423570840166 | 0.8372268676757812 |
| A | 2 | 0.29833631421889445 | 0.8835058212280273 | 0.289534287270409 | 0.8808194398880005 |
| A | 3 | 0.29230560329312344 | 0.889859676361084 | 0.28119652406116713 | 0.8909234404563904 |
| A | 4 | 0.28736237740531434 | 0.8927131295204163 | 0.27595525366415985 | 0.8951399922370911 |
| A | 5 | 0.2844695663419729 | 0.8956278562545776 | 0.2733214121641764 | 0.8976982235908508 |
| A | 6 | 0.2820081970153925 | 0.8986018300056458 | 0.27007074991029145 | 0.9013163447380066 |
| A | 7 | 0.27799652766410027 | 0.9020892381668091 | 0.2683289269575722 | 0.903318464756012 |
| A | 8 | 0.2751427329637892 | 0.9041666984558105 | 0.26589024882751505 | 0.9049825072288513 |
| A | 9 | 0.27149451099686717 | 0.9063180088996887 | 0.2630389975212407 | 0.9070092439651489 |
| A | 10 | 0.2693122841266592 | 0.9076045155525208 | 0.2620268215836922 | 0.9082087874412537 |
| B | 1 | 0.42711182677059056 | 0.805413544178009 | 0.3403653021025459 | 0.8339340686798096 |
| B | 2 | 0.3275967734567558 | 0.8676827549934387 | 0.3025545016040819 | 0.8680973052978516 |
| B | 3 | 0.3019303730599032 | 0.8819810748100281 | 0.28939681043709303 | 0.8810822367668152 |
| B | 4 | 0.29528579936840865 | 0.8862892389297485 | 0.2839442185124519 | 0.8866345286369324 |
| B | 5 | 0.29233524083470835 | 0.8885485529899597 | 0.28197691886137194 | 0.8893831968307495 |
| B | 6 | 0.2902452326348324 | 0.8898402452468872 | 0.27852705694249397 | 0.8928098678588867 |
| B | 7 | 0.28858805788253233 | 0.8910948634147644 | 0.2777015643188194 | 0.8938984870910645 |
| B | 8 | 0.28799919087311343 | 0.8918055891990662 | 0.27686037537793656 | 0.8951202630996704 |
| B | 9 | 0.2862598656909368 | 0.8925409317016602 | 0.27510407642909496 | 0.8960363864898682 |
| B | 10 | 0.28571512554052625 | 0.8933445811271667 | 0.27439303566673534 | 0.8970500826835632 |
| C | 1 | 0.6361632836370261 | 0.7660733461380005 | 0.5423033578075764 | 0.8326510190963745 |
| C | 2 | 0.2989584456116318 | 0.886354923248291 | 0.2897341979343487 | 0.8859612345695496 |
| C | 3 | 0.29128565674075224 | 0.8915613293647766 | 0.2805022224792874 | 0.8936946392059326 |
| C | 4 | 0.29216950866769414 | 0.893086850643158 | 0.27977198869950737 | 0.8971990346908569 |
| C | 5 | 0.2860918493136353 | 0.8965034484863281 | 0.2743961314323253 | 0.8999456763267517 |
| C | 6 | 0.28511504739339805 | 0.8985102772712708 | 0.2755873987294815 | 0.9020417928695679 |
| C | 7 | 0.27805910835374564 | 0.9025135040283203 | 0.2712636781213293 | 0.9051476120948792 |
| C | 8 | 0.2756187487019471 | 0.9052262306213379 | 0.26746814379841566 | 0.906978964805603 |
| C | 9 | 0.27430350284544874 | 0.9067431092262268 | 0.26927270377776547 | 0.9078153371810913 |
| C | 10 | 0.27119421469311195 | 0.9076564908027649 | 0.26553604847797624 | 0.9088817834854126 |
| D | 1 | 0.4257842053940918 | 0.8113903999328613 | 0.33966941586600524 | 0.8349900245666504 |
| D | 2 | 0.32567708672471507 | 0.8709589838981628 | 0.29933057347174485 | 0.8719696402549744 |
| D | 3 | 0.30036518049752536 | 0.8838387727737427 | 0.2854564251089323 | 0.8856303691864014 |
| D | 4 | 0.2962414398559852 | 0.8873009085655212 | 0.2826309308931979 | 0.889992892742157 |
| D | 5 | 0.29212252447603804 | 0.8895174264907837 | 0.27932154455687125 | 0.892598032951355 |
| D | 6 | 0.2917248719487204 | 0.89065021276474 | 0.2790066674547903 | 0.8944588899612427 |
| D | 7 | 0.28880137098084296 | 0.8920960426330566 | 0.27687496029386727 | 0.896441638469696 |
| D | 8 | 0.28769141937785964 | 0.8930537104606628 | 0.27516456925260196 | 0.8973539471626282 |
| D | 9 | 0.28829462275639234 | 0.8935515880584717 | 0.27612325984380165 | 0.8981664180755615 |
| D | 10 | 0.28633809680909117 | 0.8942021727561951 | 0.27390378597608317 | 0.899003267288208 |
| E | 1 | 0.6561603311937871 | 0.7247951030731201 | 0.45485246883469493 | 0.8171190619468689 |
| E | 2 | 0.33730220444182873 | 0.8650222420692444 | 0.3071755204158312 | 0.8751615881919861 |
| E | 3 | 0.3378940157836684 | 0.8854241967201233 | 0.3067141254506746 | 0.881205141544342 |
| E | 4 | 0.33651937848459734 | 0.8908189535140991 | 0.3083927163589453 | 0.8853007555007935 |
| E | 5 | 0.32903079888324954 | 0.89542156457901 | 0.29567772770595796 | 0.8920027613639832 |
| E | 6 | 0.3353491233246469 | 0.8982303738594055 | 0.296794966237268 | 0.8954371809959412 |
| E | 7 | 0.30570936704173984 | 0.9009566307067871 | 0.287310071567868 | 0.8987398743629456 |
| E | 8 | 0.3133799272700791 | 0.9032361507415771 | 0.2916861691869381 | 0.8994027376174927 |
| E | 9 | 0.30570487644083877 | 0.9038118720054626 | 0.29052332297584693 | 0.8998451828956604 |
| E | 10 | 0.3141113476404744 | 0.9040754437446594 | 0.2907483709109982 | 0.9002980589866638 |
| F | 1 | 0.5399008383688292 | 0.7287682294845581 | 0.3896299370618227 | 0.813164234161377 |
| F | 2 | 0.3832664180155411 | 0.8556698560714722 | 0.30623138395333865 | 0.8670082092285156 |
| F | 3 | 0.35754469452886023 | 0.8761104345321655 | 0.31788352253571195 | 0.8757956624031067 |
| F | 4 | 0.36082681837528974 | 0.8816577792167664 | 0.3200172457251774 | 0.8783866167068481 |
| F | 5 | 0.3564606406143055 | 0.8848495483398438 | 0.3125234839610746 | 0.8828960061073303 |
| F | 6 | 0.35862814590807596 | 0.8863003253936768 | 0.31336881536047717 | 0.8833458423614502 |
| F | 7 | 0.35438875372944145 | 0.8873353004455566 | 0.30853686614090264 | 0.8858286738395691 |
| F | 8 | 0.3509968395710024 | 0.8893418908119202 | 0.30564232096534694 | 0.8874363899230957 |
| F | 9 | 0.34724628407156044 | 0.8902211785316467 | 0.30396452078086456 | 0.8891589641571045 |
| F | 10 | 0.35131737688292036 | 0.8902915716171265 | 0.3074881901105931 | 0.8892875909805298 |

## 11. INSTALL RESULTS (primary task)

| Cell | alpha_D | Best Round | Install LogLoss | Install AUC |
|---|---|---|---|---|
| A | 1.0 | 10 | 0.2620268215836922 | 0.9082087874412537 |
| B | 1.0 | 10 | 0.27439303566673534 | 0.8970500826835632 |
| C | 0.5 | 10 | 0.26553604847797624 | 0.9088817834854126 |
| D | 0.5 | 10 | 0.27390378597608317 | 0.899003267288208 |
| E | 0.1 | 7 | 0.287310071567868 | 0.8987398743629456 |
| F | 0.1 | 9 | 0.30396452078086456 | 0.8891589641571045 |

Trajectory classification (computed from the 60 rows): A, B, C(Stage10B), D(Stage11) improve Install LogLoss in every round (strictly monotonic down); **E best at round 7** with increases afterward (r8–r10 all above r7); **F best at round 9** (r10 worse); A/B/C/D best at round 10. AUC and LogLoss move in opposite directions between r9 and r10 in E and F (both metrics rose, LL worsened while AUC improved). Install LL at round 1 is high for every cell (0.34–0.55) and drops sharply by round 2.

## 12. CLICK RESULTS (auxiliary task)

| Cell | alpha_D | Click LogLoss | Click AUC |
|---|---|---|---|
| A | 1.0 | 0.2693122841266592 | 0.9076045155525208 |
| B | 1.0 | 0.28571512554052625 | 0.8933445811271667 |
| C | 0.5 | 0.27119421469311195 | 0.9076564908027649 |
| D | 0.5 | 0.28633809680909117 | 0.8942021727561951 |
| E | 0.1 | 0.30570936704173984 | 0.9009566307067871 |
| F | 0.1 | 0.34724628407156044 | 0.8902211785316467 |

Click is trained with lambda=1.0 in the supervised loss but plays **no role in checkpoint selection** (Install LogLoss only). Descriptive observations: Click LL is higher than Install LL in every cell/round; Click AUC at round 10 is 0.89–0.91 for A–D but lower for E/F SSL-era alphas; within E and F, Click metrics fluctuate more than Install across rounds. No causal Click→Install claim is supported by the design.

## 13. SSL VS NO-SSL COMPARISON (matched alpha_D)

| alpha_D | No-SSL Install LL | SSL Install LL | d(LL) SSL-NoSSL | d(LL) % | No-SSL Install AUC | SSL Install AUC | d(AUC) | No-SSL Click LL | SSL Click LL | No-SSL Click AUC | SSL Click AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.2620268215836922 | 0.27439303566673534 | 0.012366214 | 4.719 | 0.9082087874412537 | 0.8970500826835632 | -0.011158705 | 0.2693122841266592 | 0.28571512554052625 | 0.9076045155525208 | 0.8933445811271667 |
| 0.5 | 0.26553604847797624 | 0.27390378597608317 | 0.008367737 | 3.151 | 0.9088817834854126 | 0.899003267288208 | -0.009878516 | 0.27119421469311195 | 0.28633809680909117 | 0.9076564908027649 | 0.8942021727561951 |
| 0.1 | 0.287310071567868 | 0.30396452078086456 | 0.016654449 | 5.797 | 0.8987398743629456 | 0.8891589641571045 | -0.00958091 | 0.30570936704173984 | 0.34724628407156044 | 0.9009566307067871 | 0.8902211785316467 |

Reading (neutral): at alpha_D=1.0 and 0.5, adding SSL under the frozen hyperparameters coincides with **higher** Install LogLoss (+0.0124 / +0.0084) and **lower** Install AUC; at alpha_D=0.1 the gap widens (+0.0167 LL, −0.0096 AUC). Click shows the same direction at every alpha_D. These are observations under the tested configuration (0.6/0.4, 0.15, 0.2, 64) — no global ranking is implied.

## 14. ALPHA_D COMPARISON

| Family | alpha_D | Best round | Install LL | Install AUC | Click LL | Click AUC |
|---|---|---|---|---|---|---|
| No-SSL | 1.0 | 10 | 0.2620268215836922 | 0.9082087874412537 | 0.2693122841266592 | 0.9076045155525208 |
| No-SSL | 0.5 | 10 | 0.26553604847797624 | 0.9088817834854126 | 0.27119421469311195 | 0.9076564908027649 |
| No-SSL | 0.1 | 7 | 0.287310071567868 | 0.8987398743629456 | 0.30570936704173984 | 0.9009566307067871 |
| SSL | 1.0 | 10 | 0.27439303566673534 | 0.8970500826835632 | 0.28571512554052625 | 0.8933445811271667 |
| SSL | 0.5 | 10 | 0.27390378597608317 | 0.899003267288208 | 0.28633809680909117 | 0.8942021727561951 |
| SSL | 0.1 | 9 | 0.30396452078086456 | 0.8891589641571045 | 0.34724628407156044 | 0.8902211785316467 |

Computed deltas (Install LL, family rows ordered 1.0→0.5→0.1): No-SSL +0.00351 (1.0→0.5) then +0.02177 (0.5→0.1); SSL −0.00049 (1.0→0.5) then +0.03006 (0.5→0.1). Install AUC deltas: No-SSL +0.000673 then −0.010142; SSL +0.001953 then −0.009844. Under this protocol the validation metric generally increased (No-SSL) / stayed flat then increased (SSL) as alpha_D decreased; the largest degradation occurs between alpha_D=0.5 and 0.1 in both families. Best rounds: 10, 10, 7 (No-SSL) and 10, 10, 9 (SSL).

## 15. CHECKPOINT AUDIT

| Cell | best ckpt | final ckpt | bytes(best) | reload | metrics match | selection |
|---|---|---|---|---|---|---|
| A | stage12_fed_nossl_alpha_1_0_seed42_best.pt | stage12_fed_nossl_alpha_1_0_seed42_final_round10.pt | 10070443 | True | True | min val Install LogLoss |
| B | stage12_fed_ssl_alpha_1_0_seed42_best.pt | stage12_fed_ssl_alpha_1_0_seed42_final_round10.pt | 10173283 | True | True | min val Install LogLoss |
| C | stage10b_best.pt | stage10b_final_round10.pt | 10070891 | True | True | min val Install LogLoss |
| D | stage11_best.pt | stage11_final_round10.pt | 10173027 | True | True | min val Install LogLoss |
| E | stage12_fed_nossl_alpha_0_1_seed42_best.pt | stage12_fed_nossl_alpha_0_1_seed42_final_round10.pt | 10066219 | True | True | min val Install LogLoss |
| F | stage12_fed_ssl_alpha_0_1_seed42_best.pt | stage12_fed_ssl_alpha_0_1_seed42_final_round10.pt | 10171683 | True | True | min val Install LogLoss |
| Stage 6 (canonical B) | centralized_transformer_mmoe_best.pt | - | 138 state entries | torch-reloaded | metric match | min val Install LogLoss |

Every final cell's best checkpoint corresponds to min validation Install LogLoss (protocol selection rule); `latest` equals the final-round state; `best` equals `final` in A/B/C/D (best round 10) and differs in E (round 7) and F (round 9). Reload verification flags come from the result JSONs. Stage 8 and Stage 6 checkpoints listed in JSON `checkpoints.stage8` / `stage6_canonical_B`.

## 16. INTEGRITY AND REPRODUCIBILITY

| six final cells complete | — | 6/6 status=complete, rounds_completed=10/10 | PASS |
|---|---|---|---|
| partition hash alpha_D=1.0 | c382d47f5824f64a | c382d47f5824f64a | PASS |
| partition hash alpha_D=0.5 | 9603ddc9facc973d | 9603ddc9facc973d | PASS |
| partition hash alpha_D=0.1 | 23146aebf184345b | 23146aebf184345b | PASS |
| partition rows total = 3,137,266 (all three partitions) | — | [3137266, 3137266, 3137266] | PASS |
| same partition for SSL/No-SSL at matched alpha | — | matched hashes per alpha in cell records | PASS |
| seed 42 everywhere | — | seed=42 in all 6 cell JSONs, smoke, Stage 11, 10B | PASS |
| validation overlap = 0 | — | 0 (Stage 9 report independent post-run audit; frozen audit script ALL CHECKS PASSED) | PASS |
| official test never loaded/evaluated | — | no test hash in any result JSON; reports state never loaded; runner has no test path | PASS |
| reload verification | — | reload_verified=true, reloaded_metrics_match=true for all 6 cells + Stage 8 A/C/D + smoke | PASS |
| SSL identity equal across B/D/F (torch payload check in prior audit) | — | 2,527,698 params / 142 state entries; sup 0.6, NT-Xent 0.4, T=0.2, corruption 0.15, proj 64 | PASS |
| canonical Stage 6 metric (torch reload, this audit) | — | 0.2562108886731335 | PASS |
| bit-for-bit reproducibility | — | NOT claimable: PyTorch does not guarantee identical results across releases/platforms/CPU-vs-GPU; runs on RTX 4060 Ti; deterministic CUDA algorithms DISABLED (recorded as deterministic_cuda_algorithms=false in checkpoints, per reports/centralized_training_report.md section on determinism) | LIMITATION |

Reproducibility statement: seed 42 is used consistently and partitions are byte-reproducible, but **strict bit-for-bit reproducibility is not claimed** (PyTorch does not guarantee identical results across releases/platforms/CPU-vs-GPU; runs executed on an RTX 4060 Ti with no documented deterministic-algorithms flag).

## 17. ANOMALIES / INCONSISTENCIES / CAVEATS

- **AN-1 (medium) Stage 10B runtime metadata inconsistency** — total_elapsed_seconds=5,226.764 (result JSON) vs sum of round_wall_seconds=25,494.0 (same JSON) vs log line 'total runtime: 5260.9 s (1.46 h)' (stage10b_resume_log.txt family). Two sources agree (~5.2e3 s) but the per-round walls sum to 25,494 s. All other five cells are self-consistent (deltas < 4 s). Resolution: report all three values; treat 25,494 s (per-round sum) as the plausible compute total; artifact NOT modified.
- **AN-2 (low) protocol.use_ssl / protocol.alpha hold runner defaults** — In ALL runner-written result JSONs (incl. Stage 11) protocol.use_ssl=False and protocol.alpha=0.5 regardless of the actual cell. Resolution: raw values reported verbatim; effective identity from the listed fields; artifact NOT modified.
- **AN-3 (low) SSL weight key named 'alpha' (0.6) collides with alpha_D symbol** — ssl.alpha=0.6 in Stage 11/12 JSONs is the supervised loss weight (joint-loss coefficient), NOT the Dirichlet alpha_D. Resolution: keep the two concepts separate in the thesis (alpha_D vs joint-loss weight).
- **AN-4 (medium) canonical B sampler nuance** — Stage 6 (canonical Experiment B) trained with a silent set_epoch no-op (no per-epoch reshuffle); Stage 8 A/C/D and all Stage 10-12 runs use the corrected per-epoch reshuffle. Diagnostic proved C(epoch1)==H(epoch1); divergence starts epoch 2. Resolution: documented decision (2026-09-17): B not retrained; flagged wherever B is compared.
- **AN-5 (low) centralized run batch-mean logloss note** — all 3 centralized epochs evaluated with previous batch-mean logloss aggregation; difference <1e-5 on 348,586 samples (report section 8). Resolution: reported only.
- **AN-6 (low) Stage 8 C epoch-3 wall time includes ~10 h OS standby (no compute).** —  Resolution: reported only.
- **AN-7 (info) alpha_D=0.5 cells are Stage 10B/11 artifacts reused as Cells C/D** — No separate Stage 12 alpha_0_5 runs exist; the six-cell matrix lettering retrofits C=Stage 10B, D=Stage 11. Protocol identical (same frozen config). Resolution: reported only.
- **AN-8 (info) Stage 10B round-1 metrics identical to stage10_smoke_round1 metrics (same integration path, seed, data order).** —  Resolution: reported only.
- **AN-9 (info) runtime scale heterogeneity across cells** — mean round wall: A ~2,039 s; C ~2,549 s; E ~3,356 s; B ~4,146 s; D ~6,352 s; F ~6,451 s. SSL runs ~2x No-SSL; alpha_D=0.1 slower than 1.0 (data-order locality). Operational, not algorithmic. Resolution: reported only.
- **AN-10 (low) Stage 11 required 10 launcher attempts after a round-1 CUDA diagnostic; final result complete and reload-verified.** —  Resolution: reported only.

## 18. THESIS CHAPTER 4 DATA TABLES

**TABLE A — final six-cell results**

| Cell | SSL | alpha_D | Best Round | Install LogLoss | Install AUC | Click LogLoss | Click AUC |
|---|---|---|---|---|---|---|---|
| A | No-SSL | 1.0 | 10 | 0.2620268215836922 | 0.9082087874412537 | 0.2693122841266592 | 0.9076045155525208 |
| B | SSL | 1.0 | 10 | 0.27439303566673534 | 0.8970500826835632 | 0.28571512554052625 | 0.8933445811271667 |
| C | No-SSL | 0.5 | 10 | 0.26553604847797624 | 0.9088817834854126 | 0.27119421469311195 | 0.9076564908027649 |
| D | SSL | 0.5 | 10 | 0.27390378597608317 | 0.899003267288208 | 0.28633809680909117 | 0.8942021727561951 |
| E | No-SSL | 0.1 | 7 | 0.287310071567868 | 0.8987398743629456 | 0.30570936704173984 | 0.9009566307067871 |
| F | SSL | 0.1 | 9 | 0.30396452078086456 | 0.8891589641571045 | 0.34724628407156044 | 0.8902211785316467 |

**TABLE B — 60-row round-by-round** (see section 10; identical content.)

**TABLE C — SSL vs No-SSL per alpha_D**

| alpha_D | No-SSL Install LL | SSL Install LL | d(LL) SSL-NoSSL | No-SSL Install AUC | SSL Install AUC | d(AUC) |
|---|---|---|---|---|---|---|
| 1.0 | 0.2620268215836922 | 0.27439303566673534 | 0.012366214 | 0.9082087874412537 | 0.8970500826835632 | -0.011158705 |
| 0.5 | 0.26553604847797624 | 0.27390378597608317 | 0.008367737 | 0.9088817834854126 | 0.899003267288208 | -0.009878516 |
| 0.1 | 0.287310071567868 | 0.30396452078086456 | 0.016654449 | 0.8987398743629456 | 0.8891589641571045 | -0.00958091 |

**TABLE D — Install** (section 11)  **TABLE E — Click** (section 12)  **TABLE F — stages and roles**

| Stage/Run | Category | In final matrix? | Why |
|---|---|---|---|
| Stages 1-3 (preprocessing/splits/loaders) | B (pre-final infrastructure) | No (frozen inputs) | frozen data foundation; hashes verified |
| Stage 4/5 (Transformer / +MMoE) | B (architecture) | No (components) | architecture of every final cell |
| Stage 6 centralized (canonical B) | D (historical, reused) | No (centralized reference) | canonical supervised baseline; sampler nuance documented |
| Stage 7 SSL mechanism + smoke | B (component validation) | No (component) | SSL implementation frozen and reused |
| Stage 8 A | B (pre-final controlled) | No | centralized Transformer-only comparison |
| Stage 8 B (canonical) | D (historical) | No | same run as Stage 6 row |
| Stage 8 C | B (pre-final controlled) | No | centralized Transformer+SSL comparison |
| Stage 8 D | B (pre-final controlled) | No | centralized Transformer+MMoE+SSL comparison |
| Stage 9 partitioning (alpha 0.5) + loader verification | B (data preparation) | No (inputs) | created Cell C/D partition; verified |
| Stage 10A protocol design | B (design) | No | frozen FL protocol + aggregation utility |
| Stage 10 smoke round-1 | C (smoke) | No | integration validation only |
| Stage 10B | A (FINAL) | YES - Cell C | federated No-SSL alpha_D=0.5 final result |
| Stage 10B/11 diagnostics | C (diagnostic) | No | r9 c08 / r1 c00 CUDA and stability checks |
| Stage 11 | A (FINAL) + integration validation | YES - Cell D | federated SSL alpha_D=0.5 final result |
| Stage 12 A/B/E/F | A (FINAL) | YES - Cells A/B/E/F | four matrix cells run under frozen Stage 12 runner |
| Stage 12 in-log preflights | C (smoke/verification) | No | per-cell pre-training verification |

**TABLE G — smoke/diagnostic/regression tests** (section 7–8)  **TABLE H — checkpoints/integrity** (section 15)

## 19. SOURCE ARTIFACT INDEX

| Artifact | Role |
|---|---|
| `artifacts/federated/stage12/stage12_fed_nossl_alpha_1_0_seed42/stage12_fed_nossl_alpha_1_0_seed42_result.json` | Cell A result (rounds, protocol, checkpoints, runtime) |
| `artifacts/federated/stage12/stage12_fed_ssl_alpha_1_0_seed42/stage12_fed_ssl_alpha_1_0_seed42_result.json` | Cell B result |
| `artifacts/federated/stage10/stage10b_result.json` | Cell C result (Stage 10B) |
| `artifacts/federated/stage11/stage11_result.json` | Cell D result (Stage 11) |
| `artifacts/federated/stage12/stage12_fed_nossl_alpha_0_1_seed42/stage12_fed_nossl_alpha_0_1_seed42_result.json` | Cell E result |
| `artifacts/federated/stage12/stage12_fed_ssl_alpha_0_1_seed42/stage12_fed_ssl_alpha_0_1_seed42_result.json` | Cell F result |
| `artifacts/federated/smoke/stage10_smoke_round1_result.json` | Stage 10 round-1 integration smoke |
| `artifacts/checkpoints/stage8/experiment_a_transformer_supervised_result.json` | Stage 8 A epochs |
| `artifacts/checkpoints/stage8/experiment_c_transformer_ssl_result.json` | Stage 8 C epochs |
| `artifacts/checkpoints/stage8/experiment_d_transformer_mmoe_ssl_result.json` | Stage 8 D epochs |
| `artifacts/checkpoints/centralized_transformer_mmoe_best.pt` | canonical B checkpoint (torch-verified in this audit) |
| `artifacts/federated/partition_alpha_0_5_seed42.json` | alpha_D=0.5 manifest |
| `artifacts/federated/partitions/stage12_partition_alpha_1_0_seed42.json` | alpha_D=1.0 manifest |
| `artifacts/federated/partitions/stage12_partition_alpha_0_1_seed42.json` | alpha_D=0.1 manifest |
| `reports/stage12/final_federated_matrix.json` | authoritative frozen matrix (cross-checked) |
| `reports/stage8_experiments_report.md` | Stage 8 narrative + sampler diagnostic |
| `reports/stage9_partition_report.md` | partition verification narrative |
| `reports/stage10_federated_protocol_report.md` | Stage 10A protocol + smoke section |
| `reports/centralized_training_report.md` | Stage 6 canonical B epochs |
| `reports/ssl_report.md` | Stage 7 SSL mechanism + smoke |
| `reports/stage12/stage12_matrix_protocol_report.md` | Stage 12 protocol + cell lettering |
| `scripts/audit_stage12_matrix.py` | frozen read-only matrix audit (ALL CHECKS PASSED) |

Values reported as NOT_VERIFIED: exact dates of Stages 1-5/7 reports; Stage 8 D peak memory; final unit-test count; Stage 8 C/D per-epoch GPU peaks (report text has run-level only for A/C). No metric was inferred.


AUDIT_COMPLETE