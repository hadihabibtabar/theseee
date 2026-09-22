"""Stage 12: the final 2 x 3 federated evaluation matrix (protocol preparation).

Stage 12 is **protocol/evaluation preparation only**: it defines, validates and
documents the six-cell federated experiment matrix that the thesis compares
against the frozen Stage 8 centralized controls. NO training happens in Stage
12 preparation; no frozen artifact is modified.

The matrix (rows = federated variant, columns = Non-IID Dirichlet alpha):

    variant                                  | alpha=1.0 | alpha=0.5 | alpha=0.1
    -----------------------------------------+-----------+-----------+----------
    Transformer + MMoE (no SSL)              |   NEW     | Stage 10B |   NEW
    Transformer + MMoE + SSL (joint loss)    |   NEW     | Stage 11  |   NEW

Existing cells (Stage 10B / Stage 11) are NOT retrained: they already ran over
the frozen alpha=0.5, seed=42 Stage 9 partition. The four new cells reuse the
identical protocol and differ ONLY in (model objective, alpha):

    FED-NOSSL-A10 / FED-SSL-A10  (alpha 1.0, milder Non-IID)
    FED-NOSSL-A01 / FED-SSL-A01  (alpha 0.1, harsher Non-IID)

Every cell inherits the frozen Stage 10A protocol (``FederatedConfig``):
10 clients, all clients every round, 10 rounds, 1 local epoch, batch 1024,
AdamW lr 3e-4 / wd 1e-2, sample-weighted FedAvg, fresh local optimizer per
client per round, sampler epoch = round - 1, cold seed-42 init, global
validation on the frozen 348,586-row split after every round, best checkpoint
selected by validation Install LogLoss, official test NEVER loaded.

Centralized controls A-D (Stage 8) stay frozen and are referenced here only
as read-only comparison anchors.

Deterministic experiment IDs
----------------------------
``stage12_fed_<ssl|nossl>_alpha_<a>_seed42`` with ``a`` rendered via
:func:`alpha_tag` (``1.0 -> "1_0"``, ``0.1 -> "0_1"``), matching the Stage 9
partition naming convention so IDs, partition files and output directories
align unambiguously.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from recsys23_fedrec.federated.config import FederatedConfig
from recsys23_fedrec.federated.partition import PLANNED_ALPHAS

__all__ = [
    "STAGE_NUMBER",
    "PROTOCOL_ID",
    "MATRIX_ALPHAS",
    "PARTITION_SEED",
    "EXPECTED_TOTAL_TRAIN_ROWS",
    "EXPECTED_VAL_ROWS",
    "STAGE12_PARTITIONS_DIR",
    "STAGE12_REPORT_DIR",
    "STAGE12_PARTITION_PREFIX",
    "PARTITION_NPZ_RELPATH",
    "PARTITION_MANIFEST_RELPATH",
    "Stage12ExperimentSpec",
    "STAGE12_EXPERIMENTS",
    "EXISTING_EXPERIMENTS",
    "NEW_EXPERIMENTS",
    "CENTRALIZED_CONTROLS",
    "alpha_tag",
    "experiment_id",
    "stage12_output_dir",
    "get_experiment",
    "get_partition_paths",
    "partition_relpaths",
    "validate_stage12_matrix",
    "matrix_summary",
]

#: Stage 12 is preparation for the final federated evaluation matrix.
STAGE_NUMBER = 12
PROTOCOL_ID = "stage12-evaluation-matrix"

#: The three Non-IID severity levels of the matrix (Stage 9 PLANNED_ALPHAS).
MATRIX_ALPHAS: tuple[float, ...] = PLANNED_ALPHAS  # (1.0, 0.5, 0.1)
#: All partitions (existing and new) use the frozen Stage 9 seed.
PARTITION_SEED = 42

#: Frozen Stage 9/10 data facts (verified again by the Stage 12 audit).
EXPECTED_TOTAL_TRAIN_ROWS = 3_137_266
EXPECTED_VAL_ROWS = 348_586

#: Dedicated Stage 12 artifact areas (nothing outside them is written).
STAGE12_PARTITIONS_DIR = "artifacts/federated/partitions"
STAGE12_REPORT_DIR = "reports/stage12"
STAGE12_PARTITION_PREFIX = "stage12_partition"

#: Partition naming convention: <prefix>_alpha_<tag>_seed42.{npz,json}.
PARTITION_NPZ_RELPATH = "artifacts/federated/partitions/{name}.npz"
PARTITION_MANIFEST_RELPATH = "artifacts/federated/partitions/{name}.json"


def alpha_tag(alpha: float) -> str:
    """``1.0 -> '1_0'``, ``0.5 -> '0_5'``, ``0.1 -> '0_1'`` (Stage 9 style)."""
    return str(float(alpha)).replace(".", "_")


def experiment_id(*, use_ssl: bool, alpha: float) -> str:
    """Deterministic experiment identifier (naming convention, no RNG)."""
    return f"stage12_fed_{'ssl' if use_ssl else 'nossl'}_alpha_{alpha_tag(alpha)}_seed{PARTITION_SEED}"


def partition_name(alpha: float) -> str:
    """Deterministic partition artifact stem for one alpha."""
    return f"{STAGE12_PARTITION_PREFIX}_alpha_{alpha_tag(alpha)}_seed{PARTITION_SEED}"


def stage12_output_dir(exp_id: str) -> str:
    """Output directory of one Stage 12 experiment (dedicated; no collisions)."""
    return f"artifacts/federated/stage12/{exp_id}"


@dataclass(frozen=True)
class Stage12ExperimentSpec:
    """One cell of the 2 x 3 federated evaluation matrix.

    Attributes:
        experiment_id: deterministic identifier (see :func:`experiment_id`).
        use_ssl: False = Stage 10B-style supervised baseline; True = Stage 11
            SSL variant (same joint objective and architecture).
        alpha: Dirichlet concentration of the client partition.
        model: descriptive model label (no "winner" language anywhere).
        runner: the script that executes (or executed) this cell.
        output_dir: dedicated output directory (Stage 10B/11 paths untouched).
        partition_npz / partition_manifest: the cell's partition artifacts.
        status: ``existing`` (already trained: Stage 10B / Stage 11) or
            ``new`` (to be trained later with the exact frozen protocol).
        provenance: which stage produced / will produce this cell.
    """

    experiment_id: str
    use_ssl: bool
    alpha: float
    model: str
    runner: str
    output_dir: str
    partition_npz: str
    partition_manifest: str
    status: str
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _spec(*, use_ssl: bool, alpha: float, status: str, provenance: str,
          runner: str, output_dir: str) -> Stage12ExperimentSpec:
    name = partition_name(alpha)
    return Stage12ExperimentSpec(
        experiment_id=experiment_id(use_ssl=use_ssl, alpha=alpha),
        use_ssl=use_ssl,
        alpha=float(alpha),
        model=(
            "Transformer + MMoE + SSL (joint 0.6*sup + 0.4*NT-Xent)"
            if use_ssl
            else "Transformer + MMoE (supervised click+install, no SSL)"
        ),
        runner=runner,
        output_dir=output_dir,
        partition_npz=PARTITION_NPZ_RELPATH.format(name=name),
        partition_manifest=PARTITION_MANIFEST_RELPATH.format(name=name),
        status=status,
        provenance=provenance,
    )


#: The complete 2 x 3 matrix in fixed canonical order.
STAGE12_EXPERIMENTS: tuple[Stage12ExperimentSpec, ...] = (
    # -- existing cells: already trained under the frozen protocol ---------
    _spec(
        use_ssl=False, alpha=0.5, status="existing", provenance="Stage 10B (completed)",
        runner="scripts/run_stage10b.py",
        output_dir="artifacts/federated/stage10",
    ),
    _spec(
        use_ssl=True, alpha=0.5, status="existing", provenance="Stage 11 (completed)",
        runner="scripts/run_stage11.py",
        output_dir="artifacts/federated/stage11",
    ),
    # -- new cells: prepared here, trained later with the same protocol ----
    _spec(
        use_ssl=False, alpha=1.0, status="new", provenance="Stage 12 (prepared)",
        runner="scripts/run_stage12_federated.py",
        output_dir=stage12_output_dir(experiment_id(use_ssl=False, alpha=1.0)),
    ),
    _spec(
        use_ssl=True, alpha=1.0, status="new", provenance="Stage 12 (prepared)",
        runner="scripts/run_stage12_federated.py",
        output_dir=stage12_output_dir(experiment_id(use_ssl=True, alpha=1.0)),
    ),
    _spec(
        use_ssl=False, alpha=0.1, status="new", provenance="Stage 12 (prepared)",
        runner="scripts/run_stage12_federated.py",
        output_dir=stage12_output_dir(experiment_id(use_ssl=False, alpha=0.1)),
    ),
    _spec(
        use_ssl=True, alpha=0.1, status="new", provenance="Stage 12 (prepared)",
        runner="scripts/run_stage12_federated.py",
        output_dir=stage12_output_dir(experiment_id(use_ssl=True, alpha=0.1)),
    ),
)

EXISTING_EXPERIMENTS: tuple[Stage12ExperimentSpec, ...] = tuple(
    s for s in STAGE12_EXPERIMENTS if s.status == "existing"
)
NEW_EXPERIMENTS: tuple[Stage12ExperimentSpec, ...] = tuple(
    s for s in STAGE12_EXPERIMENTS if s.status == "new"
)

#: Frozen Stage 8 centralized controls (read-only comparison anchors).
CENTRALIZED_CONTROLS: tuple[dict[str, str], ...] = (
    {
        "id": "A",
        "description": "Transformer + supervised Install (centralized)",
        "checkpoint": "artifacts/checkpoints/stage8/experiment_a_transformer_supervised_best.pt",
        "result": "artifacts/checkpoints/stage8/experiment_a_transformer_supervised_result.json",
    },
    {
        "id": "B",
        "description": "Transformer + MMoE supervised Click+Install (centralized)",
        "checkpoint": "artifacts/checkpoints/centralized_transformer_mmoe_best.pt",
        "result": "reports/centralized_training_report.md",
    },
    {
        "id": "C",
        "description": "Transformer + SSL Install (centralized)",
        "checkpoint": "artifacts/checkpoints/stage8/experiment_c_transformer_ssl_best.pt",
        "result": "artifacts/checkpoints/stage8/experiment_c_transformer_ssl_result.json",
    },
    {
        "id": "D",
        "description": "Transformer + MMoE + SSL Click+Install (centralized)",
        "checkpoint": "artifacts/checkpoints/stage8/experiment_d_transformer_mmoe_ssl_best.pt",
        "result": "artifacts/checkpoints/stage8/experiment_d_transformer_mmoe_ssl_result.json",
    },
)


def get_experiment(exp_id: str) -> Stage12ExperimentSpec:
    """Look up one matrix cell by its deterministic experiment ID."""
    for spec in STAGE12_EXPERIMENTS:
        if spec.experiment_id == exp_id:
            return spec
    known = ", ".join(s.experiment_id for s in STAGE12_EXPERIMENTS)
    raise KeyError(f"unknown Stage 12 experiment {exp_id!r}; known IDs: {known}")


def get_partition_paths(alpha: float) -> tuple[str, str]:
    """(npz, manifest) relative paths of the Stage 12 partition for ``alpha``."""
    name = partition_name(alpha)
    return (
        PARTITION_NPZ_RELPATH.format(name=name),
        PARTITION_MANIFEST_RELPATH.format(name=name),
    )


def partition_relpaths() -> tuple[str, ...]:
    """All Stage 12 partition artifacts (npz + manifest, both new alphas)."""
    out: list[str] = []
    for alpha in MATRIX_ALPHAS:
        if alpha == 0.5:
            continue  # Stage 9 partition owns alpha=0.5 (frozen, untouched)
        npz, manifest = get_partition_paths(alpha)
        out.extend((npz, manifest))
    return tuple(out)


def validate_stage12_matrix() -> list[str]:
    """Structural validation of the frozen matrix definition.

    Returns a list of violations (empty == the matrix is internally
    consistent and collides with no frozen artifact).
    """
    violations: list[str] = []

    # exactly 2 x 3 = 6 unique cells, covering every (use_ssl, alpha) pair
    ids = [s.experiment_id for s in STAGE12_EXPERIMENTS]
    if len(ids) != 6 or len(set(ids)) != 6:
        violations.append(f"matrix must hold 6 unique experiment IDs, got {ids}")
    pairs = {(s.use_ssl, s.alpha) for s in STAGE12_EXPERIMENTS}
    if pairs != {(b, a) for b in (False, True) for a in MATRIX_ALPHAS}:
        violations.append(f"matrix coverage incomplete: {sorted(pairs)}")

    # alphas must be the Stage 9 planned set, exactly (each twice: 2 variants)
    if sorted(s.alpha for s in STAGE12_EXPERIMENTS) != sorted(list(MATRIX_ALPHAS) * 2):
        violations.append("alphas do not match the frozen matrix set")

    # every protocol-critical field must equal the frozen Stage 10A values
    frozen = FederatedConfig()
    frozen_dict = frozen.to_dict()
    protocol_keys = (
        "num_clients", "participation", "local_epochs", "rounds",
        "local_batch_size", "optimizer", "learning_rate", "weight_decay",
        "aggregation", "selection_metric", "local_optimizer_state_persistence",
        "local_sampler_epoch_mode",
    )
    for key in protocol_keys:
        if frozen_dict[key] != getattr(frozen, key):  # pragma: no cover - trivial
            violations.append(f"FederatedConfig {key} drifted from the frozen value")

    # IDs / paths must be internally consistent and collision-free
    for spec in STAGE12_EXPERIMENTS:
        expected_id = experiment_id(use_ssl=spec.use_ssl, alpha=spec.alpha)
        if spec.experiment_id != expected_id:
            violations.append(f"{spec.experiment_id} does not follow the ID convention")
        expected_npz, expected_manifest = get_partition_paths(spec.alpha)
        if (spec.partition_npz, spec.partition_manifest) != (expected_npz, expected_manifest):
            violations.append(f"{spec.experiment_id}: partition paths inconsistent")
        # no accidental collision with the frozen Stage 10B / Stage 11 outputs
        if spec.status == "new" and spec.output_dir in (
            "artifacts/federated/stage10", "artifacts/federated/stage11",
        ):
            violations.append(f"{spec.experiment_id}: output dir collides with a frozen stage")
        if spec.status == "existing" and spec.provenance not in (
            "Stage 10B (completed)", "Stage 11 (completed)",
        ):
            violations.append(f"{spec.experiment_id}: unexpected existing-cell provenance")

    # output dirs must be unique across the whole matrix
    dirs = [s.output_dir for s in STAGE12_EXPERIMENTS]
    if len(set(dirs)) != len(dirs):
        violations.append("output directories are not unique across the matrix")

    # the two new alphas must not reuse the frozen Stage 9 alpha=0.5 artifacts
    for relpath in partition_relpaths():
        if "alpha_0_5" in relpath:
            violations.append(f"Stage 12 partition path collides with Stage 9: {relpath}")

    return violations


def matrix_summary() -> dict[str, Any]:
    """Serializable snapshot of the matrix (embedded in the protocol report)."""
    return {
        "stage": STAGE_NUMBER,
        "protocol_id": PROTOCOL_ID,
        "purpose": (
            "final federated evaluation matrix: 2 federated variants x 3 "
            "Non-IID severity levels, compared against the frozen Stage 8 "
            "centralized controls (descriptive comparisons only)"
        ),
        "alphas": list(MATRIX_ALPHAS),
        "partition_seed": PARTITION_SEED,
        "protocol": FederatedConfig().to_dict(),
        "experiments": [s.to_dict() for s in STAGE12_EXPERIMENTS],
        "existing": [s.experiment_id for s in EXISTING_EXPERIMENTS],
        "new": [s.experiment_id for s in NEW_EXPERIMENTS],
        "centralized_controls": [dict(c) for c in CENTRALIZED_CONTROLS],
        "rules": {
            "checkpoint_selection": "global validation Install LogLoss (frozen 348,586-row split)",
            "official_test": "never loaded, never evaluated, by any Stage 12 cell",
            "report_language": (
                "descriptive comparisons only; no 'winner'/'best'/'superior' claims"
            ),
        },
    }


def write_matrix_json(path: str | Path) -> Path:
    """Persist the matrix snapshot (used by the audit script)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matrix_summary(), indent=2, sort_keys=True), encoding="utf-8")
    return path


# eager protocol validation at import time: a broken matrix must fail loudly
_matrix_violations = validate_stage12_matrix()
if _matrix_violations:  # pragma: no cover - guards against future edits
    raise RuntimeError("Stage 12 matrix definition is invalid: " + "; ".join(_matrix_violations))
