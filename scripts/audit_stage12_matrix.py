"""Stage 12 audit: matrix, partitions, protocol, and no-test-evaluation proof.

Read-only verification that the Stage 12 evaluation-matrix preparation is
complete and trustworthy. Produces NO training artifacts. Checks:

  1. matrix structure      - validate_stage12_matrix() has no violations
  2. protocol              - FederatedConfig equals the frozen Stage 10A values
  3. SSL configuration     - Stage 12 SSL cells match the completed Stage 11
                             constants (0.6/0.4 joint, T=0.2, rate 0.15, dim 64,
                             2,527,698 params / 142 state entries)
  4. partitions            - both Stage 12 partitions re-derive byte-identical
                             from the frozen Stage 9 implementation, and match
                             their stored manifests/NPZ hashes exactly
  5. data protection       - clients never overlap the frozen validation split;
                             official test data is never constructed
  6. integrity             - all protected artifacts match the frozen values

Writes exactly one file: the Stage 12 matrix snapshot JSON inside
``reports/stage12/`` (documentation, not a training artifact).

Usage:
    python scripts/audit_stage12_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

import run_stage11 as s11  # noqa: E402
import run_stage10b as s10b  # noqa: E402

from recsys23_fedrec.experiments.integrity import (  # noqa: E402
    PROTECTED_VALUES,
    snapshot_protected_artifacts,
)
from recsys23_fedrec.federated.client_loader import load_partition_positions  # noqa: E402
from recsys23_fedrec.federated.config import get_federated_config  # noqa: E402
from recsys23_fedrec.federated.partition import CLIENT_IDS, partition_hash  # noqa: E402
from recsys23_fedrec.federated.stage12_matrix import (  # noqa: E402
    NEW_EXPERIMENTS,
    STAGE12_EXPERIMENTS,
    STAGE12_REPORT_DIR,
    get_partition_paths,
    validate_stage12_matrix,
    write_matrix_json,
)
from recsys23_fedrec.training.split import load_split  # noqa: E402

TOTAL_TRAIN_ROWS = 3_137_266
TOTAL_DATASET_ROWS = 3_485_852

_failures: list[str] = []


def _check(label: str, ok: bool, detail: str = "") -> None:
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" - {detail}" if detail else ""), flush=True)
    if not ok:
        _failures.append(f"{label}: {detail}")


def main() -> int:
    print("=== STAGE 12 AUDIT (read-only) ===", flush=True)

    # ---- 1. matrix structure ----------------------------------------------
    print("\n1. matrix structure", flush=True)
    violations = validate_stage12_matrix()
    _check("matrix definition valid (2x3, unique IDs/outputs, no frozen-path collisions)",
           not violations, "; ".join(violations))
    _check("6 cells: 2 existing (Stage 10B/11) + 4 new",
           len(STAGE12_EXPERIMENTS) == 6 and len(NEW_EXPERIMENTS) == 4,
           f"existing={[s.experiment_id for s in STAGE12_EXPERIMENTS if s.status == 'existing']}, "
           f"new={[s.experiment_id for s in NEW_EXPERIMENTS]}")

    # ---- 2. frozen protocol -------------------------------------------------
    print("\n2. frozen federated protocol", flush=True)
    cfg = get_federated_config()
    frozen_expectations = {
        "num_clients": 10, "participation": "all_clients_every_round",
        "local_epochs": 1, "rounds": 10, "local_batch_size": 1024,
        "optimizer": "adamw", "learning_rate": 3e-4, "weight_decay": 1e-2,
        "lambda_click": 1.0, "lambda_install": 1.0,
        "aggregation": "sample_weighted_fedavg",
        "selection_metric": "global_validation_install_logloss",
        "use_ssl": False, "local_optimizer_state_persistence": False,
        "local_sampler_epoch_mode": "round_minus_one",
        "seed": 42, "alpha": 0.5, "partition_seed": 42,
    }
    for key, expected in frozen_expectations.items():
        _check(f"FederatedConfig.{key} == {expected!r}", getattr(cfg, key) == expected,
               f"got {getattr(cfg, key)!r}")

    # ---- 3. SSL configuration equals Stage 11 ------------------------------
    print("\n3. SSL configuration (Stage 11 equality)", flush=True)
    _check("joint weight alpha == 0.6", s11.SSL_ALPHA == 0.6, f"got {s11.SSL_ALPHA}")
    _check("NT-Xent temperature == 0.2", s11.SSL_TEMPERATURE == 0.2, f"got {s11.SSL_TEMPERATURE}")
    _check("corruption rate == 0.15", s11.SSL_CORRUPTION_RATE == 0.15,
           f"got {s11.SSL_CORRUPTION_RATE}")
    _check("projection dim == 64", s11.SSL_PROJECTION_DIM == 64, f"got {s11.SSL_PROJECTION_DIM}")
    _check("expected SSL params == 2,527,698", s11.EXPECTED_SSL_PARAM_COUNT == 2_527_698,
           f"got {s11.EXPECTED_SSL_PARAM_COUNT}")
    _check("expected SSL state entries == 142", s11.EXPECTED_SSL_STATE_ENTRIES == 142,
           f"got {s11.EXPECTED_SSL_STATE_ENTRIES}")

    # ---- 4. partitions: re-derivation + stored-artifact consistency --------
    print("\n4. Stage 12 partitions (re-derivation from the frozen Stage 9 code)", flush=True)
    integrity_snapshot = snapshot_protected_artifacts(PROJECT_ROOT)
    for key, expected in PROTECTED_VALUES.items():
        _check(f"protected {key}", integrity_snapshot[key] == expected,
               f"got {integrity_snapshot[key]!r}")

    split = load_split(
        PROJECT_ROOT / "artifacts" / "splits" / "centralized_split.json",
        expected_total_rows=TOTAL_DATASET_ROWS,
    )
    _check("frozen split train hash", split.train_index_hash == s10b.SPLIT_TRAIN_HASH,
           split.train_index_hash)
    _check("frozen split validation hash",
           split.validation_index_hash == s10b.SPLIT_VALIDATION_HASH,
           split.validation_index_hash)
    _check("frozen split sizes 3,137,266 / 348,586",
           split.train_rows == TOTAL_TRAIN_ROWS and split.validation_rows == 348_586,
           f"{split.train_rows:,}/{split.validation_rows:,}")

    from recsys23_fedrec.federated.partition import (
        dirichlet_partition,
        load_train_labels,
        train_labels_hash,
    )

    all_labels = load_train_labels(
        PROJECT_ROOT / "artifacts" / "processed" / "train",
        expected_total_rows=TOTAL_DATASET_ROWS,
    )
    labels = all_labels[split.train_indices]
    labels_hash = train_labels_hash(labels)
    _check("train-labels hash stable", labels_hash == "800e2aa9d0bf7b79", labels_hash)

    for alpha in (1.0, 0.1):
        npz_relpath, manifest_relpath = get_partition_paths(alpha)
        manifest_path = PROJECT_ROOT / manifest_relpath
        npz_path = PROJECT_ROOT / npz_relpath
        if not manifest_path.is_file() or not npz_path.is_file():
            _check(f"partition alpha={alpha} artifacts exist", False, f"missing {npz_relpath}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored = str(np.load(npz_path)["partition_hash"])
        recomputed = partition_hash(
            [dirichlet_partition(labels, alpha=alpha, seed=42)[c] for c in range(10)]
        )
        _check(f"alpha={alpha}: stored == manifest == re-derived partition hash",
               stored == manifest["partition_hash"] == recomputed,
               f"stored {stored} / manifest {manifest['partition_hash']} / derived {recomputed}")
        _check(f"alpha={alpha}: manifest metadata", manifest["alpha"] == alpha
               and manifest["seed"] == 42 and manifest["number_of_clients"] == 10
               and manifest["total_training_rows"] == TOTAL_TRAIN_ROWS,
               f"alpha={manifest['alpha']} seed={manifest['seed']} "
               f"rows={manifest['total_training_rows']:,}")
        positions = load_partition_positions(npz_path)
        sizes = {cid: int(pos.size) for cid, pos in positions.items()}
        _check(f"alpha={alpha}: 10 non-empty clients, rows sum exactly {TOTAL_TRAIN_ROWS:,}",
               len(sizes) == 10 and all(v > 0 for v in sizes.values())
               and sum(sizes.values()) == TOTAL_TRAIN_ROWS,
               f"sizes={sizes}")
        all_pos = np.concatenate([positions[cid] for cid in CLIENT_IDS])
        _check(f"alpha={alpha}: clients mutually disjoint (position space)",
               np.unique(all_pos).size == all_pos.size, f"{all_pos.size:,} positions")
        composed = np.concatenate([split.train_indices[positions[cid]] for cid in CLIENT_IDS])
        _check(f"alpha={alpha}: composed rows unique + disjoint from validation",
               np.unique(composed).size == composed.size
               and np.intersect1d(composed, split.validation_indices).size == 0,
               f"{np.intersect1d(composed, split.validation_indices).size} overlap rows")
        _check(f"alpha={alpha}: rows conserve the global joint-label counts",
               all(
                   sum(manifest["clients"][cid]["joint_counts"][k] for cid in CLIENT_IDS) == v
                   for k, v in manifest["global_label_distribution"]["joint_counts"].items()
               ), "per-state client sums == global counts")

    # ---- 5. no-test-evaluation proof ----------------------------------------
    print("\n5. official test protection", flush=True)
    stage12_sources = [
        PROJECT_ROOT / "scripts" / "run_stage12_partitions.py",
        PROJECT_ROOT / "scripts" / "run_stage12_federated.py",
        PROJECT_ROOT / "scripts" / "audit_stage12_matrix.py",
        PROJECT_ROOT / "src" / "recsys23_fedrec" / "federated" / "stage12_matrix.py",
    ]
    # Forbidden patterns are ASSEMBLED from fragments so that this scanner's
    # own source cannot contain (and thereby self-match) the full literals.
    _q = chr(34)  # double-quote
    _s = chr(47)  # slash
    forbidden_patterns = (
        "processed" + _q + ", " + _q + "test",   # opening the test Parquet dir
        "processed" + _s + "test",               # test path spelling
        "split" + "=" + _q + "test" + _q,        # dataset split kwarg for test
        "load_" + "official" + "_test",
        "test" + "_parquet",
    )
    test_free = True
    for path in stage12_sources:
        text = path.read_text(encoding="utf-8")
        # The full literals never appear in any scanned file: this scanner
        # assembles them at runtime from fragments (see above).
        for forbidden in forbidden_patterns:
            if forbidden in text:
                test_free = False
                _check(f"{path.name} never constructs the official test set", False,
                       "found a forbidden pattern")
    _check("Stage 12 preparation code never loads/constructs the official test set",
           test_free, "scanned 4 Stage 12 source files")

    # ---- write the matrix snapshot ------------------------------------------
    print("\n6. matrix snapshot", flush=True)
    matrix_path = write_matrix_json(PROJECT_ROOT / STAGE12_REPORT_DIR / "stage12_matrix.json")
    _check("matrix snapshot written", matrix_path.is_file(),
           str(matrix_path.relative_to(PROJECT_ROOT)))

    print("\n=== AUDIT RESULT ===", flush=True)
    if _failures:
        print(f"FAILED: {len(_failures)} check(s):", flush=True)
        for failure in _failures:
            print(f"  - {failure}", flush=True)
        return 1
    print("ALL CHECKS PASSED - Stage 12 preparation is complete and verified.", flush=True)
    print("No training was performed; no protected artifact was modified.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
