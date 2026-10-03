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

The official challenge page is the primary reference for dataset access and challenge documentation:

https://www.recsyschallenge.com/2023/

A commonly mirrored direct download URL for the original challenge archive is:

https://cdn.sharechat.com/2a161f8e_1679936280892_sc.zip

The repository expects the extracted dataset in the project root as `train/` and `test/`, with the original tab-separated `.csv` shards. The final experiments use the frozen processed artifacts already stored under `artifacts/`; downloading the raw dataset is therefore only necessary for a fresh reproduction.

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

The implemented Transformer encoder has **6 layers/blocks**. Any thesis text, figure, or table that describes this final architecture as having 1 Transformer block is inconsistent with the implemented model and must be corrected to 6 blocks.

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

The final experiments were produced with Python 3.13.3 and PyTorch 2.7.0+cu118. The repository should keep `requirements.txt` at the root so the command above is reproducible.

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

The terminology is intentionally separated here: **Cell** refers to one configuration in the final six-cell experiment matrix; **Stage** refers to the historical implementation/execution stage in which that cell was completed. Therefore Cells C and D are part of the same final matrix as A, B, E and F, even though C was executed by `run_stage10b.py` and D by `run_stage11.py`. These dedicated scripts are preserved because those two cells were completed earlier and are frozen research artifacts; they are not different experimental conditions.

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
├── train/                         # Raw ShareChat training data
├── test/                          # Official test data
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
│   └── final_experiment_audit.*
│
├── tests/
│
└── README.md
```

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

* ShareChat Ads — RecSys Challenge 2023
* Ma et al. — Multi-gate Mixture-of-Experts (MMoE), KDD 2018
* Chen et al. — SimCLR, 2020
* Oord et al. — InfoNCE, 2018
* McMahan et al. — Federated Averaging (FedAvg), 2017

---

# 23. Repository Hygiene Before Submission

The README itself does not claim that the repository root is clean unless the working tree has been checked directly. Before submitting the repository, verify that generated logs, PID/error files, and Python `__pycache__` directories are not unintentionally tracked or left at the repository root.

From PowerShell, a read-only check is:

```powershell
git status --short
Get-ChildItem -Force -Recurse -Directory -Filter __pycache__ | Select-Object FullName
Get-ChildItem -Force -File -Include *.log,*.txt,*.pid,*.err -Recurse | Select-Object FullName
```

Only after this check should the root-cleanliness status be reported. The commands above do not delete anything.

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