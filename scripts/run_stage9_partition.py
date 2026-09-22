"""Stage 9: generate the reproducible federated client partition.

Streams the (is_clicked, is_installed) labels of the processed train Parquet,
verifies the frozen Stage 2 artifact linkage, partitions the 3,137,266
centralized training rows across 10 simulated federated clients via Dirichlet
allocation over the joint (click, install) label states, and writes:

* the partition manifest (audit JSON) to
  ``artifacts/federated/partition_alpha_<alpha>_seed<seed>.json``
* the compact client index mapping (NPZ) next to it
* a human-readable summary to stdout

No features, validation rows, test rows, or models are involved.

Example:
    python scripts/run_stage9_partition.py --alpha 0.5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recsys23_fedrec.data import FeatureSet  # noqa: E402
from recsys23_fedrec.experiments.integrity import (  # noqa: E402
    PROTECTED_VALUES,
    snapshot_protected_artifacts,
)
from recsys23_fedrec.federated.partition import (  # noqa: E402
    CLIENT_IDS,
    PRIMARY_ALPHA,
    build_manifest,
    dirichlet_partition,
    load_train_labels,
    save_partition_arrays,
    train_labels_hash,
)
from recsys23_fedrec.training.split import load_split  # noqa: E402

TOTAL_TRAIN_ROWS = 3_137_266


def _alpha_tag(alpha: float) -> str:
    return str(alpha).replace(".", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=PRIMARY_ALPHA)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Stage 9: federated client partition ===")
    print(f"alpha={args.alpha} seed={args.seed} clients=10")

    # ---- integrity gate: frozen artifacts must be intact ----------------
    print("\n--- protected artifact integrity ---")
    snapshot = snapshot_protected_artifacts(PROJECT_ROOT)
    for key, expected in PROTECTED_VALUES.items():
        actual = snapshot[key]
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  {key}: {actual} (expected {expected}) [{status}]")
        if actual != expected:
            print("FATAL: protected artifact integrity violated; aborting.")
            return 1
    print("  all protected artifacts intact")

    # ---- frozen split verification --------------------------------------
    print("\n--- frozen split verification ---")
    split = load_split(
        PROJECT_ROOT / "artifacts" / "splits" / "centralized_split.json",
        expected_total_rows=3_485_852,
    )
    print(f"  train rows: {split.train_rows:,} (hash {split.train_index_hash})")
    print(
        f"  validation rows: {len(split.validation_indices):,} "
        f"(hash {split.validation_index_hash}) — stays globally held out"
    )
    if split.train_rows != TOTAL_TRAIN_ROWS:
        print(f"FATAL: expected {TOTAL_TRAIN_ROWS:,} train rows.")
        return 1

    # ---- labels ----------------------------------------------------------
    print("\n--- streaming train labels (is_clicked, is_installed) ---")
    train_dir = PROJECT_ROOT / "artifacts" / "processed" / "train"
    all_labels = load_train_labels(train_dir, expected_total_rows=3_485_852)
    print(f"  processed rows read: {len(all_labels):,} (train+validation space)")
    feature_set = FeatureSet.load(PROJECT_ROOT / "artifacts" / "preprocessing")
    print(f"  Stage 2 artifact hash: {feature_set.artifact_hash}")
    # CRITICAL leakage guard: the processed train Parquet holds the full
    # 3,485,852 rows; clients are partitioned over the FROZEN centralized
    # train indices only. Client index arrays are positions into
    # split.train_indices (0 .. 3,137,265); validation/test never enter.
    labels = all_labels[split.train_indices]
    if len(labels) != TOTAL_TRAIN_ROWS:
        print(f"FATAL: train label rows {len(labels):,} != {TOTAL_TRAIN_ROWS:,}")
        return 1
    print(f"  train-only labels after split subsetting: {len(labels):,}")
    print("  validation rows (348,586) excluded from the partition space")
    labels_hash = train_labels_hash(labels)
    print(f"  train-labels hash: {labels_hash}")

    # ---- partition ---------------------------------------------------------
    print(f"\n--- Dirichlet partition (alpha={args.alpha}) ---")
    client_indices = dirichlet_partition(labels, alpha=args.alpha, seed=args.seed)

    manifest = build_manifest(
        client_indices,
        labels,
        alpha=args.alpha,
        seed=args.seed,
        source_split_path="artifacts/splits/centralized_split.json",
        source_split_train_index_hash=split.train_index_hash,
        labels_hash=labels_hash,
    )

    out_dir = PROJECT_ROOT / "artifacts" / "federated"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"partition_alpha_{_alpha_tag(args.alpha)}_seed{args.seed}.json"
    npz_path = out_dir / f"partition_alpha_{_alpha_tag(args.alpha)}_seed{args.seed}.npz"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    save_partition_arrays(client_indices, npz_path, partition_hash_value=manifest["partition_hash"])

    # ---- report ------------------------------------------------------------
    g = manifest["global_label_distribution"]
    print(f"\n  partition hash: {manifest['partition_hash']}")
    print(f"  manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"  index arrays: {npz_path.relative_to(PROJECT_ROOT)}")
    print(
        f"\n  global: click {g['click_rate']:.4f} | install {g['install_rate']:.4f}"
    )
    print("\n  client |    rows |   %    | click | install")
    for c, cid in enumerate(CLIENT_IDS):
        s = manifest["clients"][cid]
        print(
            f"  {cid} | {s['rows']:>7,} | {s['row_percentage']:>5.2f}% | "
            f"{s['click_rate']:.4f} | {s['install_rate']:.4f}"
        )
    sizes = manifest["rows_per_client"]
    print(
        f"\n  sizes: min {min(sizes):,} / max {max(sizes):,} "
        f"(max/min = {max(sizes) / min(sizes):.2f})"
    )
    het = manifest["heterogeneity"]
    print(
        f"  heterogeneity: mean |delta joint proportion| = "
        f"{het['mean_absolute_joint_proportion_deviation']:.4f}, max = "
        f"{het['max_absolute_joint_proportion_deviation']:.4f}"
    )
    print("\n  no validation/test rows used; no training performed in Stage 9.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
