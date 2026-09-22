"""Stage 8 sampler-protocol diagnostic runner (read-only, tiny).

Produces the evidence requested for the sampler-nuance decision:

1. batch-index fingerprints for Protocol H (historical Stage 6) and
   Protocol C (corrected Stage 8) across epochs 1-3 on the REAL training
   split (index arithmetic only — no model, no GPU, no labels);
2. a tiny controlled training comparison (Experiment B model, batch 1024,
   4 batches/epoch, 2 epochs) under both protocols + a determinism repeat;
3. integrity verification of every protected artifact.

No official test data is loaded; nothing is written outside stdout.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch

from recsys23_fedrec.data import FeatureSet, ProcessedRecSysDataset
from recsys23_fedrec.experiments import get_experiment_config
from recsys23_fedrec.experiments.integrity import (
    snapshot_protected_artifacts,
    verify_protected_artifacts,
)
from recsys23_fedrec.experiments.sampler_diagnostic import (
    fingerprint_corrected,
    fingerprint_historical,
    run_tiny_sampler_comparison,
)
from recsys23_fedrec.preprocessing.config import SEED
from recsys23_fedrec.training.split import load_split


def main() -> int:
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    integrity = snapshot_protected_artifacts(PROJECT_ROOT)
    print(
        f"integrity snapshot: stage2={integrity['stage2_artifact_hash']}, "
        f"split={integrity['split_train_index_hash']}, "
        f"stage6_ckpt={integrity['stage6_checkpoint_sha256'][:10]}…"
    )

    feature_set = FeatureSet.load(PROJECT_ROOT / "artifacts" / "preprocessing")
    dataset = ProcessedRecSysDataset(
        feature_set, PROJECT_ROOT / "artifacts" / "processed" / "train", split="train"
    )
    split = load_split(
        PROJECT_ROOT / "artifacts" / "splits" / "centralized_split",
        expected_total_rows=len(dataset),
    )
    print(
        f"split loaded (hash verified): train={split.train_rows:,} "
        f"val={split.validation_rows:,}, hash={split.train_index_hash}"
    )

    # ------------------------------------------------------------------
    # Part 1: batch fingerprints across epochs 1-3 (full training subset)
    # ------------------------------------------------------------------
    print("\n=== Part 1: batch-arrangement fingerprints (indices only) ===")
    fingerprints: dict[str, dict] = {"H": {}, "C": {}}
    for epoch in (1, 2, 3):
        h = fingerprint_historical(
            split.train_indices, batch_size=1024, seed=SEED, epoch=epoch
        )
        c = fingerprint_corrected(
            split.train_indices, batch_size=1024, seed=SEED, epoch=epoch
        )
        fingerprints["H"][epoch] = h.to_dict()
        fingerprints["C"][epoch] = c.to_dict()
        print(
            f"epoch {epoch} | H sha256={h.arrangement_sha256[:16]} "
            f"first={h.first_batch_first} last_batch_first={h.last_batch_first} | "
            f"C sha256={c.arrangement_sha256[:16]} "
            f"first={c.first_batch_first} last_batch_first={c.last_batch_first}"
        )

    h1, h2, h3 = (fingerprints["H"][e]["arrangement_sha256"] for e in (1, 2, 3))
    c1, c2, c3 = (fingerprints["C"][e]["arrangement_sha256"] for e in (1, 2, 3))
    h_identical_across_epochs = h1 == h2 == h3
    c_distinct_across_epochs = len({c1, c2, c3}) == 3
    h_differs_from_c_epoch1 = h1 != c1
    print(f"\nH identical across epochs 1-3:        {h_identical_across_epochs}")
    print(f"C distinct across epochs 1-3:         {c_distinct_across_epochs}")
    print(f"H(epoch e) differs from C(epoch e):   {h_differs_from_c_epoch1}")

    # determinism of both protocols under regeneration
    h_again = fingerprint_historical(
        split.train_indices, batch_size=1024, seed=SEED, epoch=2
    )
    c_again = fingerprint_corrected(
        split.train_indices, batch_size=1024, seed=SEED, epoch=3
    )
    h_deterministic = h_again.arrangement_sha256 == h2
    c_deterministic = c_again.arrangement_sha256 == c3
    print(f"H regeneration reproduces epoch-2:    {h_deterministic}")
    print(f"C regeneration reproduces epoch-3:    {c_deterministic}")

    # ------------------------------------------------------------------
    # Part 2: tiny controlled training comparison (Experiment B model)
    # ------------------------------------------------------------------
    print("\n=== Part 2: tiny controlled training comparison (B model) ===")
    config = get_experiment_config("experiment_b_transformer_mmoe", device=str(device))
    n_train = 4 * config.batch_size      # 4 batches per epoch
    n_val = 2 * config.batch_size        # 2 validation batches (deterministic slice)
    train_idx = np.asarray(split.train_indices[:n_train], dtype=np.int64)
    val_idx = np.asarray(split.validation_indices[:n_val], dtype=np.int64)
    print(f"tiny subset: {n_train} train rows (4 batches x 1024), {n_val} val rows, 2 epochs")

    comparison = run_tiny_sampler_comparison(
        config,
        feature_set,
        dataset,
        train_idx,
        val_idx,
        batch_size=config.batch_size,
        n_batches=4,
        epochs=2,
        device=device,
        seed=SEED,
    )
    for protocol in ("H", "C"):
        rep = comparison["protocols"][protocol]
        print(f"\nprotocol {protocol}:")
        for epoch, fp in rep["batch_fingerprints"].items():
            print(
                f"  epoch {epoch}: sha256={fp['arrangement_sha256'][:16]} "
                f"first_batch_first={fp['first_batch_first']}"
            )
        for epoch, losses in rep["epoch_losses"].items():
            extra = ""
            if losses.get("click"):
                extra += f", click {losses['click']:.6f}"
            print(
                f"  epoch {epoch}: total {losses['total']:.6f} "
                f"(install {losses['install']:.6f}{extra})"
            )
        vm = rep["val_metrics"]
        print(
            f"  val: install logloss {vm['val_install_logloss']:.6f}, "
            f"AUC {vm['val_install_auc']:.6f} (n={vm['n_val_samples']})"
        )
        print(f"  param delta vs own init (L2): {rep['param_delta_l2']:.6f}")
        print(f"  deterministic repeat: {rep['deterministic_repeat']}")

    print(
        f"\ncross-protocol param delta (H vs C, L2): "
        f"{comparison['cross_protocol_param_delta_l2']:.6f}"
    )
    print(f"  relative to init H: {comparison['relative_to_init_h']:.6f}")
    print(f"  relative to init C: {comparison['relative_to_init_c']:.6f}")
    ratio_h = comparison["cross_protocol_param_delta_l2"] / max(comparison["relative_to_init_h"], 1e-12)
    ratio_c = comparison["cross_protocol_param_delta_l2"] / max(comparison["relative_to_init_c"], 1e-12)
    print(f"  H-vs-C delta / H movement since init: {ratio_h:.3f}")
    print(f"  H-vs-C delta / C movement since init: {ratio_c:.3f}")

    # ------------------------------------------------------------------
    # Part 3: integrity re-verification
    # ------------------------------------------------------------------
    violations = verify_protected_artifacts(integrity, PROJECT_ROOT)
    if violations:
        for v in violations:
            print(f"INTEGRITY VIOLATION: {v}")
        return 1
    print("\nProtected artifacts unchanged (hashes, sizes, mtimes identical).")

    elapsed = time.perf_counter() - started
    print(f"\nSAMPLER DIAGNOSTIC COMPLETE in {elapsed:.1f}s")
    print(
        json.dumps(
            {
                "h_identical_across_epochs": h_identical_across_epochs,
                "c_distinct_across_epochs": c_distinct_across_epochs,
                "h_differs_from_c": h_differs_from_c_epoch1,
                "h_deterministic": h_deterministic,
                "c_deterministic": c_deterministic,
                "cross_protocol_param_delta_l2": comparison["cross_protocol_param_delta_l2"],
                "repeat_of_H_matches_H": comparison["repeat_of_H_matches_H"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
