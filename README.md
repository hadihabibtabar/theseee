# Robust User Engagement Modeling With Transformers and Self-Supervised Learning

Implementation and experimental artifacts for a master's thesis on **user engagement prediction in advertising** using:

**Transformer + MMoE + Federated Learning + Contrastive Self-Supervised Learning**

The primary prediction task is **Install**, while **Click** is used as an auxiliary task.

---

## 0. Where to Start

If you only want to inspect the completed research, start with `reports/stage12/final_federated_matrix.md`; no GPU or raw dataset is required.
For a full repository verification, install the pinned dependencies with `pip install -r requirements.txt`, then run the preprocessing verification and final matrix audit.
For a fresh reproduction, follow the data, partition, and federated-training sections in order; final training requires an NVIDIA GPU.
The six final experiment cells are already completed and frozen, so reproduction is optional and should not be confused with inspection.
Checkpoints are under `artifacts/federated/`, and final metrics are under `reports/stage12/`.

## 1. Overview

This project investigates whether contrastive self-supervised learning changes the performance of a multi-task Transformer model when training is performed in a **federated and non-IID setting**.

The experiments use the ShareChat Ads dataset from the **RecSys Challenge 2023**.

The complete pipeline is:

```text
ShareChat Ads Dataset
        ↓
Leakage-safe preprocessing
        ↓
Frozen train / validation split
        ↓
Dirichlet non-IID partitioning
        ↓
10 simulated federated clients
        ↓
Transformer Encoder
        ↓
MMoE multi-task prediction
        ↓
Optional contrastive SSL
        ↓
FedAvg aggregation
        ↓
Global validation
        ↓
Install / Click metrics
```

The final experiment consists of **six federated configurations**:

* 3 levels of data heterogeneity: α_D = 1.0, 0.5, 0.1
* 2 learning conditions: No-SSL and SSL

All six experiments use the same federated protocol and seed.

---

## 2. Project Status

**Final experiments: complete and frozen.**

The six-cell federated experiment matrix has been executed for 10 rounds per cell. The resulting checkpoints, metrics, partitions, reports, and figures are preserved as the research record.

The repository can therefore be used in two different ways:

| Purpose                            | GPU required? |
| ---------------------------------- | ------------: |
| Read reports and results           |            No |
| Inspect figures                    |            No |
| Run integrity audits               |            No |
| Run fast tests                     |            No |
| Reproduce preprocessing            |            No |
| Reproduce partitions               |            No |
| Reproduce final federated training |       **Yes** |

> **Important:** Do not rerun the final experiments simply to inspect the repository.
> The final federated runs require several hours per cell.

---

# 3. Research Setup

## 3.1 Dataset

The experiments use the ShareChat Ads RecSys Challenge 2023 dataset.

### Dataset access

The dataset is obtained through the challenge's documented access route. The official challenge page is the primary reference for dataset access and challenge documentation:

https://www.recsyschallenge.com/2023/

ShareChat's challenge page (https://sharechat.com/recsys2023) documents the registration and data-access procedure. Data access requires registration through the official challenge route; no direct downloadable archive link is reproduced here because no such link could be independently verified as official and persistently available.

For a fresh reproduction, obtain the dataset through that documented access route and extract it into the project root as `train/` and `test/`, with the original tab-separated `.csv` shards (30 training shards and 1 test shard, ≈ 1.87 GB total).

### Dataset is not included in this repository

The raw ShareChat dataset is **not included** in the published repository version. From the current version onward, `train/` and `test/` are not tracked by Git; researchers must obtain the data themselves through the official access procedure above.

Two filesystem names are deliberately distinct:

* `train/` and `test/` — **raw data directories** that the pipeline expects locally after you obtain the dataset (see the structure table and Section 16). They are ignored by Git.
* `tests/` — the **software test suite** (`python -m pytest`), unrelated to the raw data.

No direct downloadable archive link is reproduced here beyond the official access references above, because no such link could be independently verified as official and persistently available. Redistribution of the raw dataset is not covered here either; follow the access requirements stated on the official challenge pages.

The final experiments use the frozen processed artifacts already stored under `artifacts/`; downloading the raw dataset is therefore only necessary for a fresh reproduction. Where frozen processed artifacts are available locally they can be used directly for inspection.

| Property               |     Value |
| ---------------------- | --------: |
| Training shards        |        30 |
| Training rows          | 3,485,852 |
| Training columns       |        82 |
| Official test rows     |   160,973 |
| Features               |        80 |
| Train/validation split | 90% / 10% |
| Federated clients      |        10 |
| Random seed            |        42 |

The training data contains two binary targets:

* **Click** — auxiliary task
* **Install** — primary task

The final experiments use only the training data and the frozen validation split.

### Official test set

The official test set was **not evaluated** in the final Stage 12 experiments.

Therefore, this repository does **not** report an official-test score.

All final metrics are calculated on the frozen validation set containing:

**348,586 samples**

---

# 4. Model Architecture

The model consists of three main components:

```text
Input Features
      │
      ▼
Feature Tokenization
      │
      ▼
Transformer Encoder
      │
      ▼
CLS Representation
      │
      ▼
      MMoE
     /   \
    /     \
 Click    Install
```

### Input representation

The preprocessing pipeline produces:

* 30 categorical features
* 38 numerical features
* 11 binary features
* 13 missing-value indicators

Total:

**92 feature tokens + 1 CLS token = 93 Transformer tokens**

### Transformer

| Parameter              | Value |
| ---------------------- | ----: |
| Encoder layers         |     6 |
| Embedding dimension    |   128 |
| Attention heads        |     8 |
| Feed-forward dimension |   128 |
| Activation             |  GELU |
| Dropout                |   0.1 |


### MMoE

The Transformer CLS representation is passed to an MMoE layer containing:

* 8 shared experts
* one task-specific gate for Click
* one task-specific gate for Install
* one tower per task

```text
                 CLS
                  │
          ┌───────┴───────┐
          │   8 Experts   │
          └───────┬───────┘
                  │
          ┌───────┴───────┐
          │               │
      Click Gate      Install Gate
          │               │
     Click Tower      Install Tower
          │               │
       Click           Install
```

The model outputs two raw logits:

```text
click_logit
install_logit
```

`BCEWithLogitsLoss` is used during supervised training.

---

# 5. Self-Supervised Learning

The SSL component uses **contrastive learning**.

For each training sample, two independently corrupted views are generated:

```text
                 Original Sample
                  /           \
                 /             \
                ▼               ▼
          Corrupted View 1  Corrupted View 2
                │               │
                ▼               ▼
           Transformer      Transformer
                │               │
                ▼               ▼
          Projection Head   Projection Head
                │               │
                └───────┬───────┘
                        ▼
                 NT-Xent Loss
```

The two views of the same impression form a **positive pair**.

Views belonging to other impressions form **negative pairs**.

### SSL configuration

| Parameter               | Value |
| ----------------------- | ----: |
| Corruption rate         |  0.15 |
| Projection dimension    |    64 |
| Temperature τ           |   0.2 |
| Supervised weight α_SSL |   0.6 |
| Contrastive weight      |   0.4 |

The joint objective is:

```text
L_total = 0.6 L_supervised + 0.4 L_contrastive
```

The contrastive loss does **not** use Click or Install labels.

### α_D and α_SSL are different

This distinction is important:

| Symbol | Meaning                                                  |
| ------ | -------------------------------------------------------- |
| α_D    | Dirichlet concentration controlling client heterogeneity |
| α_SSL  | Supervised-loss weight in the SSL objective              |

The final experiments use:

```text
α_D   ∈ {1.0, 0.5, 0.1}
α_SSL = 0.6
```

---

# 6. Federated Learning

The dataset does not contain real organizational clients.

Instead, 10 simulated clients are created by applying **Dirichlet partitioning** to the training data.

For each joint `(Click, Install)` label state, the corresponding samples are distributed across the clients using a Dirichlet distribution.

```text
                 Training Data
                      │
                      ▼
             Dirichlet Partition
                      │
        ┌─────┬─────┬─────┬─────┐
        ▼     ▼     ▼     ▼     ▼
      Client Client Client ... Client
       01      02      03        10
        │       │       │         │
        └───────┴───────┴─────────┘
                      │
                      ▼
                    FedAvg
                      │
                      ▼
               Global Model
```

Smaller α_D produces stronger label-distribution heterogeneity in this experimental setup:

```text
α_D = 1.0   → weaker heterogeneity
α_D = 0.5   → intermediate heterogeneity
α_D = 0.1   → stronger heterogeneity
```

---

# 7. Federated Training Protocol

All six final experiments use the same training protocol.

| Setting              |                   Value |
| -------------------- | ----------------------: |
| Clients              |                      10 |
| Client participation |                    100% |
| Federated rounds     |                      10 |
| Local epochs         |                       1 |
| Batch size           |                    1024 |
| Optimizer            |                   AdamW |
| Learning rate        |                3 × 10⁻⁴ |
| Weight decay         |                    10⁻² |
| Aggregation          |  Sample-weighted FedAvg |
| Seed                 |                      42 |
| Validation           |   Global validation set |
| Model selection      | Minimum Install LogLoss |
| Scheduler            |                    None |
| Class weighting      |                    None |
| Focal loss           |                    None |

Each round follows:

```text
Global Model
     ↓
Broadcast to 10 Clients
     ↓
Local Training
     ↓
10 Updated Client Models
     ↓
Sample-weighted FedAvg
     ↓
New Global Model
     ↓
Global Validation
     ↓
Checkpoint Selection
```

The best checkpoint is selected using **global validation Install LogLoss**.

---

# 8. Final Experiment Matrix

The final experiment contains six cells.

| Cell | SSL | α_D | Best Round | Install LogLoss | Install AUC |
| ---- | --- | --: | ---------: | --------------: | ----------: |
| A    | No  | 1.0 |         10 |        0.262027 |    0.908209 |
| B    | Yes | 1.0 |         10 |        0.274393 |    0.897050 |
| C    | No  | 0.5 |         10 |        0.265536 |    0.908882 |
| D    | Yes | 0.5 |         10 |        0.273904 |    0.899003 |
| E    | No  | 0.1 |          7 |        0.287310 |    0.898740 |
| F    | Yes | 0.1 |          9 |        0.303965 |    0.889159 |

Additional Click metrics are recorded in the corresponding result files.

### Interpretation

Under this **single frozen protocol and seed**, the SSL configurations recorded higher validation Install LogLoss and lower validation Install AUC than their corresponding No-SSL configurations.

These results describe the observed behavior of this particular experimental setup.

They should not be interpreted as a universal conclusion about contrastive SSL, federated learning, or Transformer models.

The experiment uses:

* one random seed
* one SSL configuration
* three α_D values
* ten federated rounds
* simulated clients

---

# 9. Results and Reports

The main result files are located in:

```text
reports/stage12/
```

### Main reports

| File                                | Purpose                      |
| ----------------------------------- | ---------------------------- |
| `final_federated_matrix.md`         | Compact final results        |
| `final_federated_results_report.md` | Detailed scientific report   |
| `final_federated_matrix.json`       | Machine-readable results     |
| `final_federated_matrix.csv`        | Spreadsheet-friendly results |
| `stage12_matrix_protocol_report.md` | Final protocol documentation |

The complete research audit is available at:

```text
reports/final_experiment_audit.md
reports/final_experiment_audit.json
```

---

# 10. Figures

Figures generated from the verified final result matrix are stored in:

```text
reports/stage12/figures/
```

The currently verified six-figure set is:

| Figure | Description |
| ------ | ----------- |
| 4-1 | Install LogLoss vs α_D |
| 4-2 | Install AUC vs α_D |
| 4-3 | No-SSL convergence |
| 4-4 | SSL convergence |
| 4-5 | SSL vs No-SSL Install LogLoss |
| 4-6 | Click LogLoss vs α_D |

Each verified figure is available in both PNG and PDF form. Additional generators for later Chapter 4 material may exist in the repository, but a generator script is not treated as evidence that its output file already exists.

---

# 11. Where to Find the Checkpoints

Federated checkpoints are stored under:

```text
artifacts/federated/
```

Final cells A, B, E and F:

```text
artifacts/federated/stage12/<experiment_id>/
```

Cell C (the α_D = 0.5 No-SSL cell):

```text
artifacts/federated/stage10/
```

Cell D (the α_D = 0.5 SSL cell):

```text
artifacts/federated/stage11/
```

Checkpoint tracking in Git and storage location:

* The checkpoint files listed above **are committed to this Git repository** as regular (non-LFS) Git blobs.
* **Git LFS is not used** for checkpoints; the repository has no `.gitattributes` LFS configuration.
* Tracked federated `.pt` files are approximately 9.6–9.7 MB each; the complete set of tracked `.pt` files (federated cells plus the centralized Stage 6 / Stage 8 baseline checkpoints, excluding smoke-test outputs) is 23 files measured at 309,998,417 bytes (≈ 296 MB).
* Because checkpoints are tracked in ordinary Git, cloning this repository retrieves them; no external checkpoint download is required.

The `.pt` files are PyTorch checkpoints and are kept as binary research artifacts; they are not intended to be edited manually. Typical files are:

```text
*_best.pt
*_final_round10.pt
*_latest.pt
*_result.json
```

### Checkpoint meaning

`*_best.pt`

> Model state selected using the lowest global validation Install LogLoss.

`*_final_round10.pt`

> Model state after round 10.

`*_latest.pt`

> Latest/resume state maintained during training.

To inspect checkpoint files from PowerShell without changing them:

```powershell
Get-ChildItem .\artifacts\federated -Recurse -Filter *.pt |
    Select-Object FullName, Length, LastWriteTime
```

A checkpoint can be opened with PyTorch for inspection using `torch.load(..., map_location="cpu")`; loading it for inference or continued training must use the corresponding model architecture and checkpoint format implemented by the repository. The accompanying `*_result.json` files should be used for the recorded metrics and experiment metadata.

All six best checkpoints passed reload verification.

---

# 12. Quick Start

## 12.1 Install the environment

Use the repository requirements file so the software environment is defined explicitly rather than installing packages one by one:


```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The final experiments were produced with Python 3.13.3 and PyTorch 2.7.0+cu118. `requirements.txt` uses version-pinned requirements (plus an `--extra-index-url` line so that the CUDA wheel `torch==2.7.0+cu118` resolves); note that a pinned requirements file is **not** a complete environment lockfile — transitive dependency versions are not frozen, so environments resolved at different times may differ slightly in transitive dependencies. The pins match the environment used to produce the final experiments; no bit-for-bit or cross-platform reproducibility is promised.

The `+cu118` wheel is hosted only on PyTorch's own wheel index (not PyPI); `requirements.txt` therefore contains

```text
--extra-index-url https://download.pytorch.org/whl/cu118
```

so that `pip install -r requirements.txt` resolves `torch==2.7.0+cu118` from the PyTorch CUDA-11.8 wheel index.

### Clean-install verification

A fresh installation of `requirements.txt` was verified in a temporary Python 3.13.3 virtual environment outside the repository (pip upgraded first, pinned versions unchanged). The command above completed successfully with exit code 0 and the import/repository smoke checks below also passed in that environment. This verifies the pinned install route itself; it is not a promise that every future downstream pip resolution will behave identically, since transitive dependency versions are not frozen.

### CPU-only installation (no NVIDIA GPU)

For machines without a supported NVIDIA GPU, install the CPU build of the same pinned PyTorch version **instead of** the CUDA wheel. Use the separate pinned file `requirements-cpu.txt`:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-cpu.txt
```

`requirements-cpu.txt` pins `torch==2.7.0` (the CPU build) from PyTorch's official CPU wheel index (`https://download.pytorch.org/whl/cpu`) and contains the same pinned non-PyTorch dependencies as `requirements.txt`. Important: install it into a fresh virtual environment — installing `requirements-cpu.txt` **after** `requirements.txt` in the same environment would not replace the CUDA-specific `torch==2.7.0+cu118` wheel, because the version numbers are equal (`2.7.0 == 2.7.0+cu118` and the local version ordering keeps CUDA-local versions newer, so pip will keep the already-installed CUDA wheel).

The GPU requirements in `requirements.txt` remain the authoritative pinned environment for reproducing the final experiments; `requirements-cpu.txt` exists only for inspection-and-development use on CPU-only machines and the CPU route was not used to produce any experiment in this repository.

## Inspect the completed research

No GPU or raw dataset is required.

```bash
git clone https://github.com/hadihabibtabar/theseee.git
cd theseee
```

Read:

```text
reports/stage12/final_federated_matrix.md
```

or:

```text
reports/stage12/final_federated_results_report.md
```

You can also inspect:

```text
reports/stage12/figures/
artifacts/federated/
reports/final_experiment_audit.md
```

---

## Verify the repository

### Run the test suite

```bash
python -m pytest -m "not slow"
```

### Verify preprocessing

```bash
python scripts/verify_preprocessing.py \
    --output_dir artifacts \
    --data_dir .
```

### Run the final matrix audit

```bash
python scripts/audit_stage12_matrix.py
```

The audit checks the frozen protocol, partitions, SSL configuration, artifact integrity, and protection of the official test set.

---

# 13. Reproducing the Experiments

> **Warning: expensive.**

### Cell versus Stage

The terminology is intentionally separated:

* A **Cell** is one configuration in the final six-cell experimental matrix (Section 8).
* A **Stage** is a historical implementation/execution stage (Section 18).
* Cells C and D belong to the **same final matrix** as the other cells; they are not separate studies.
* Cell C was executed by `scripts/run_stage10b.py`, and Cell D by `scripts/run_stage11.py`, because those two configurations were completed and frozen earlier than the Stage 12 runner.
* The distinct script names do **not** indicate different evaluation criteria or additional final experimental conditions; all six cells share the identical frozen protocol of Section 7.

(The script paths above were verified to exist in the repository.)

### Cell-to-runner mapping (verified from runner code and frozen result metadata)

The mapping below was verified from the runner implementations and the `runner`/`stage12.experiment_id`/`protocol.*` fields recorded inside each frozen result JSON, not inferred from filenames:

| Cell | Configuration | Runner script | Frozen result artifact |
| ---- | ------------- | ------------- | ---------------------- |
| A | No-SSL, α_D = 1.0 | `scripts/run_stage12_federated.py` (`stage12.use_ssl=False`, `stage12.alpha=1.0`, `stage12.runner` recorded in the result) | `artifacts/federated/stage12/stage12_fed_nossl_alpha_1_0_seed42/stage12_fed_nossl_alpha_1_0_seed42_result.json` |
| B | SSL, α_D = 1.0 | `scripts/run_stage12_federated.py` (`stage12.use_ssl=True`, `stage12.alpha=1.0`) | `artifacts/federated/stage12/stage12_fed_ssl_alpha_1_0_seed42/stage12_fed_ssl_alpha_1_0_seed42_result.json` |
| C | No-SSL, α_D = 0.5 | `scripts/run_stage10b.py` (result `stage = stage10b-federated-baseline`, `protocol.use_ssl=False`, `protocol.alpha=0.5`) | `artifacts/federated/stage10/stage10b_result.json` |
| D | SSL, α_D = 0.5 | `scripts/run_stage11.py` (result `stage = stage11-federated-ssl`; SSLTransformerMMoE model, SSL joint loss; partition α_D = 0.5) | `artifacts/federated/stage11/stage11_result.json` |
| E | No-SSL, α_D = 0.1 | `scripts/run_stage12_federated.py` (`stage12.use_ssl=False`, `stage12.alpha=0.1`) | `artifacts/federated/stage12/stage12_fed_nossl_alpha_0_1_seed42/stage12_fed_nossl_alpha_0_1_seed42_result.json` |
| F | SSL, α_D = 0.1 | `scripts/run_stage12_federated.py` (`stage12.use_ssl=True`, `stage12.alpha=0.1`) | `artifacts/federated/stage12/stage12_fed_ssl_alpha_0_1_seed42/stage12_fed_ssl_alpha_0_1_seed42_result.json` |

Documented metadata quirk: in the Stage 10B/11 result JSONs, `protocol.alpha` holds the effective Dirichlet parameter (0.5); the Stage 12 result JSONs record the effective settings in `stage12.alpha`/`stage12.use_ssl` (the `stage12.ssl.alpha` field shown there is the SSL supervised-loss weight α_SSL = 0.6, not α_D). All six cells share the same federated protocol and evaluation criteria.

The final experiments require an NVIDIA GPU with approximately 8 GiB or more of VRAM.

Before training, verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The final six cells are:

```bash
# Cell A — No SSL, α_D = 1.0
python scripts/run_stage12_federated.py \
    --experiment stage12_fed_nossl_alpha_1_0_seed42

# Cell B — SSL, α_D = 1.0
python scripts/run_stage12_federated.py \
    --experiment stage12_fed_ssl_alpha_1_0_seed42

# Cell C — No SSL, α_D = 0.5
python scripts/run_stage10b.py

# Cell D — SSL, α_D = 0.5
python scripts/run_stage11.py

# Cell E — No SSL, α_D = 0.1
python scripts/run_stage12_federated.py \
    --experiment stage12_fed_nossl_alpha_0_1_seed42

# Cell F — SSL, α_D = 0.1
python scripts/run_stage12_federated.py \
    --experiment stage12_fed_ssl_alpha_0_1_seed42
```

### Preflight before a final run

For Stage 12 cells:

```bash
python scripts/run_stage12_federated.py \
    --experiment stage12_fed_nossl_alpha_1_0_seed42 \
    --preflight_only
```

Preflight performs verification and stops before training.

---

# 14. Reproducing the Data Pipeline

The repository already contains the frozen preprocessing and partition artifacts.

If they are present and verified, **do not regenerate them just to inspect the project**.

For a fresh reproduction:

### Preprocessing

```bash
python scripts/preprocess_dataset.py \
    --data_dir . \
    --output_dir artifacts
```

### α_D = 0.5

```bash
python scripts/run_stage9_partition.py \
    --alpha 0.5 \
    --seed 42
```

### α_D = 1.0 and 0.1

```bash
python scripts/run_stage12_partitions.py \
    --alphas 1.0 0.1 \
    --seed 42
```

---

# 15. Environment

The final experiments were produced in the following environment:

| Component    | Version                         |
| ------------ | ------------------------------- |
| OS           | Windows                         |
| Python       | 3.13.3                          |
| PyTorch      | 2.7.0+cu118                     |
| GPU          | NVIDIA GeForce RTX 4060 Ti 8 GB |
| NumPy        | 2.2.6                           |
| Pandas       | 2.2.3                           |
| PyArrow      | 20.0.0                          |
| scikit-learn | 1.6.1                           |
| Matplotlib   | 3.10.0                          |
| pytest       | 9.1.1                           |

The repository does not contain a lockfile.

Bit-exact GPU reproduction across different hardware, drivers, or software environments is therefore not guaranteed.

---

# 16. Repository Structure

```text
theseee/
│
├── train/                         # RAW ShareChat training data — NOT tracked by Git; expected locally after obtaining the dataset
├── test/                          # RAW official test data — NOT tracked by Git; expected locally via the official access route
│
├── src/
│   └── recsys23_fedrec/
│       ├── preprocessing/         # Data preprocessing
│       ├── data/                  # Dataset and batching
│       ├── models/                # Transformer + MMoE + SSL
│       ├── ssl/                   # Contrastive learning
│       ├── training/              # Training and metrics
│       ├── federated/             # Partitioning + FedAvg
│       └── experiments/           # Experiment integrity
│
├── scripts/                       # Training, audit and verification scripts
│   └── ops/                       # PowerShell maintenance launchers/monitor (launch_*.ps1, check_*.ps1)
│
├── artifacts/
│   ├── preprocessing/
│   ├── processed/
│   ├── splits/
│   ├── checkpoints/
│   └── federated/
│
├── reports/
│   ├── stage12/
│   │   ├── figures/
│   │   └── final experiment reports
│   ├── logs/                      # Appendix-style operational logs and diagnostics (see below)
│   └── final_experiment_audit.*
│
├── results/                       # Preserved scientific run logs of Stages 6–12 (see below)
│
├── tests/                         # Software test suite (pytest) — NOT the raw data
│
├── requirements.txt               # Pinned GPU (CUDA 11.8) environment
├── requirements-cpu.txt           # Pinned CPU-only environment
└── README.md
```

### `train/` / `test/` versus `tests/`

`train/` and `test/` are the **raw dataset directories** the pipeline expects locally (obtained through the official access route; ignored by Git, see Dataset Access). `tests/` is the **software test suite** run with `python -m pytest`. They are unrelated.

### `results/` (run logs) versus `reports/logs/`

These two directories serve different purposes, verified from their contents:

* `results/` — the preserved **scientific run logs** of Stages 6 through 12: the actual training-run output of the centralized baseline, the Stage 8 experiments, Stage 10B/11 federality cells, and all Stage 12 matrix cells (e.g. `stage10b_log.txt`, `stage11_log.txt`, `stage12_fed_ssl_alpha_0_1_seed42_log.txt`). These are thesis evidence and are tracked in Git.
* `reports/logs/` — operation-level **appendix logs**: a dataset-inspection pipeline log (`inspect_20260913_154923.log`) and saved stderr transcripts of failed smoke/baseline attempts (`stage10_smoke_err.txt`, `stage10b_err.txt`). They document diagnostics and failure forensics rather than training progress, and are kept separate from the run logs.

---

# 17. Important Source Files

| File                             | Role                                         |
| -------------------------------- | -------------------------------------------- |
| `models/transformer.py`          | Transformer encoder and feature tokenization |
| `models/mmoe.py`                 | MMoE architecture                            |
| `models/ssl_transformer_mmoe.py` | Transformer + MMoE + SSL                     |
| `ssl/augmentations.py`           | Two-view feature corruption                  |
| `ssl/contrastive_loss.py`        | NT-Xent contrastive loss                     |
| `ssl/joint_loss.py`              | Supervised + contrastive objective           |
| `federated/partition.py`         | Dirichlet non-IID partitioning               |
| `federated/fedavg.py`            | Sample-weighted FedAvg                       |
| `federated/config.py`            | Frozen federated protocol                    |
| `training/metrics.py`            | LogLoss and AUC                              |
| `training/trainer.py`            | Centralized training                         |
| `experiments/integrity.py`       | Frozen-artifact protection                   |

---

# 18. Research Timeline

The implementation was developed incrementally:

| Stage | Description                         |
| ----- | ----------------------------------- |
| 1     | Dataset inspection                  |
| 2     | Leakage-safe preprocessing          |
| 3     | Memory-bounded data pipeline        |
| 4     | Transformer baseline                |
| 5     | Transformer + MMoE                  |
| 6     | Centralized supervised baseline     |
| 7     | SSL implementation and verification |
| 8     | Controlled centralized experiments  |
| 9     | Dirichlet partitioning              |
| 10A   | Federated protocol design and audit |
| 10B   | Federated No-SSL, α_D = 0.5         |
| 11    | Federated SSL, α_D = 0.5            |
| 12    | Final six-cell experiment matrix    |

Stages 10B and 11 provide the α_D = 0.5 cells in the final matrix.

---

# 19. Reproducibility and Research Integrity

The repository preserves the experimental record using:

* fixed random seed: **42**
* frozen preprocessing artifacts
* frozen train/validation split
* hash-verified federated partitions
* fixed federated configuration
* sample-weighted FedAvg
* checkpoint reload verification
* machine-readable result files
* automated integrity audits

The final result reports and figures are generated from the stored result JSON files rather than manually entering metric values.

The complete integrity and audit information is available in:

```text
reports/final_experiment_audit.md
```

---

# 20. Known Limitations

The final experiments should be interpreted within the following limitations:

1. **Single seed** — only seed 42 was evaluated.
2. **Simulated clients** — the 10 clients are created from one dataset rather than representing real organizations.
3. **Fixed SSL configuration** — SSL hyperparameters were not optimized for each α_D.
4. **Fixed federated protocol** — only 10 rounds and one local epoch were evaluated.
5. **Validation-only evaluation** — no official-test score is reported.
6. **Environment dependence** — exact GPU-level numerical reproduction is not guaranteed.
7. **Historical Stage 6 sampler difference** — the centralized reference uses a historical sampling convention that differs from the corrected convention used in later experiments.

For the complete details of these limitations and all historical diagnostics, see:

```text
reports/final_experiment_audit.md
```

---

# 21. Thesis Mapping

The main thesis components correspond to the following repository modules:

| Thesis Component         | Repository                                   |
| ------------------------ | -------------------------------------------- |
| Dataset                  | `train/`, `reports/dataset_report.md`        |
| Preprocessing            | `src/recsys23_fedrec/preprocessing/`         |
| Transformer              | `src/recsys23_fedrec/models/transformer.py`  |
| MMoE                     | `src/recsys23_fedrec/models/mmoe.py`         |
| Self-Supervised Learning | `src/recsys23_fedrec/ssl/`                   |
| Non-IID partitioning     | `src/recsys23_fedrec/federated/partition.py` |
| Federated Learning       | `src/recsys23_fedrec/federated/`             |
| Evaluation               | `src/recsys23_fedrec/training/metrics.py`    |
| Final experiments        | `reports/stage12/`                           |
| Figures                  | `reports/stage12/figures/`                   |
| Integrity verification   | `scripts/audit_stage12_matrix.py`            |

---

# 22. References

The implementation is based on established methods and the ShareChat RecSys Challenge 2023 literature:

* Rahul Agrawal, Sarang Brahme, Sourav Maitra, Abhishek Srivastava, Athirai A. Irissappane, Yong Liu, and Saikishore Kalloori. "RecSys Challenge 2023 Dataset: Ads Recommendations in Online Advertising." *Proceedings of the Recommender Systems Challenge 2023 (RecSysChallenge '23)*, 2023, pp. 1–3. https://doi.org/10.1145/3626221.3627283
* Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, and Lichan Hong. "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts." *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '18)*, 2018, pp. 1930–1939. https://doi.org/10.1145/3219819.3220007
* Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. "A Simple Framework for Contrastive Learning of Visual Representations." *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)*, PMLR 119:1597–1607. https://proceedings.mlr.press/v119/chen20j.html (arXiv:2002.05709)
* Aaron van den Oord, Yazhe Li, and Oriol Vinyals. "Representation Learning with Contrastive Predictive Coding." *arXiv preprint* arXiv:1807.03748, 2018. https://arxiv.org/abs/1807.03748
* Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas. "Communication-Efficient Learning of Deep Networks from Decentralized Data." *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017)*, PMLR 54:1273–1282. https://proceedings.mlr.press/v54/mcmahan17a.html (arXiv:1602.05629)

---

## Final Note

This repository is primarily a **research artifact**.

For understanding the work, start with:

```text
1. Final Results
2. Model Architecture
3. Self-Supervised Learning
4. Federated Learning
5. Experimental Matrix
6. Reports and Figures
```

For verifying the research record:

```text
reports/final_experiment_audit.md
```

For reproducing the experiments, follow the **Reproducing the Experiments** section above.

For thesis reporting, use the verified result reports and figures under `reports/stage12/`.
