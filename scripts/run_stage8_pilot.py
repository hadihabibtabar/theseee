"""Stage 8 pilot: verify the complete experiment path on a few real batches.

For each of Experiments A, C, D (B is the reused Stage 6 canonical baseline):

 * build the model from :class:`ExperimentConfig` (dispatch check)
 * load a few REAL train/validation batches through the Stage 3 loader and
   the hash-verified centralized split
 * forward pass, joint loss (supervised [+ contrastive]), backward,
   optimizer step
 * checkpoint write to a TEMPORARY path + reload verification
 * metric calculation (validation Install LogLoss/AUC on the pilot batches)

No full training, no official test set, no writes to any protected artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch

from recsys23_fedrec.data import FeatureSet, ProcessedRecSysDataset
from recsys23_fedrec.experiments import (
    EXPERIMENT_A,
    EXPERIMENT_C,
    EXPERIMENT_D,
    ExperimentRunner,
    get_experiment_config,
)
from recsys23_fedrec.experiments.integrity import (
    snapshot_protected_artifacts,
    verify_protected_artifacts,
)
from recsys23_fedrec.experiments.runner import compute_joint_loss
from recsys23_fedrec.models import TransformerMMoE
from recsys23_fedrec.preprocessing.config import SEED
from recsys23_fedrec.training.loaders import make_subset_loader
from recsys23_fedrec.training.split import load_split


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 8 pilot")
    parser.add_argument("--artifacts_dir", default=str(PROJECT_ROOT / "artifacts"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_train_batches", type=int, default=2)
    parser.add_argument("--n_val_batches", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    started = time.perf_counter()
    artifacts_dir = Path(args.artifacts_dir)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    integrity = snapshot_protected_artifacts(PROJECT_ROOT)
    print(f"protected snapshot OK: stage2={integrity['stage2_artifact_hash']}, "
          f"split={integrity['split_train_index_hash']}, "
          f"stage6_ckpt={integrity['stage6_checkpoint_sha256'][:16]}…")

    set_seed(SEED)
    feature_set = FeatureSet.load(artifacts_dir / "preprocessing")
    dataset = ProcessedRecSysDataset(feature_set, artifacts_dir / "processed" / "train", split="train")
    split = load_split(artifacts_dir / "splits" / "centralized_split", expected_total_rows=len(dataset))
    print(f"dataset rows: {dataset.length:,}; split train={split.train_rows:,} val={split.validation_rows:,} "
          f"(hash verified: {split.train_index_hash})")

    # Small deterministic pilot subsets: first batches of train + validation.
    train_idx = np.asarray(split.train_indices[: args.batch_size * args.n_train_batches], dtype=np.int64)
    val_idx = np.asarray(split.validation_indices[: args.batch_size * args.n_val_batches], dtype=np.int64)

    # The Experiment B model type must still construct from the reused config.
    config_b = get_experiment_config("experiment_b_transformer_mmoe")
    torch.manual_seed(SEED)
    model_b = TransformerMMoE(feature_set)
    print(f"experiment B constructs from frozen defaults: {sum(p.numel() for p in model_b.parameters()):,} params")

    overall_ok = True
    with tempfile.TemporaryDirectory(prefix="stage8_pilot_") as tmp:
        tmp_dir = Path(tmp)
        for name in (EXPERIMENT_A, EXPERIMENT_C, EXPERIMENT_D):
            print(f"\n--- pilot: {name} ---")
            config = get_experiment_config(name, device=str(device), batch_size=args.batch_size)
            runner = ExperimentRunner(
                config=config,
                feature_set=feature_set,
                train_dataset=dataset,
                train_indices=train_idx,
                val_indices=val_idx,
                checkpoint_path=tmp_dir / f"{name}.pt",
                split_hash=split.train_index_hash,
                device=device,
            )
            n_params = sum(p.numel() for p in runner.model.parameters())
            print(f"model: {config.model_type}, {n_params:,} params")

            # ---- training-path verification (a few real batches) --------
            runner.model.train()
            step_losses = []
            for epoch in (1, 2):
                loader = runner._train_loader(epoch)
                for i, batch in enumerate(loader):
                    if i >= args.n_train_batches:
                        break
                    batch = runner._to_device(batch, device)
                    runner.optimizer.zero_grad()
                    output = (
                        runner.model(batch)
                        if config.use_ssl
                        else runner._forward_supervised(runner.model, batch)
                    )
                    losses = compute_joint_loss(output, batch, config)
                    losses["total"].backward()
                    runner.optimizer.step()
                    step_losses.append({k: (float(v) if v is not None else None) for k, v in losses.items()})
                    finite = bool(torch.isfinite(losses["total"]))
                    print(f"  epoch {epoch} step {i}: total {float(losses['total']):.4f} "
                          f"(install {float(losses['install']):.4f}"
                          + (f", click {float(losses['click']):.4f}" if losses.get('click') is not None else "")
                          + (f", contrastive {float(losses['contrastive']):.4f}" if 'contrastive' in losses else "")
                          + f") finite={finite}")
                    overall_ok &= finite

            # ---- validation-path verification (metrics) -----------------
            metrics = runner.validate()
            print(f"  val metrics (pilot batches): install logloss "
                  f"{metrics['val_install_logloss']:.4f} AUC {metrics['val_install_auc']:.4f}"
                  + (f", click logloss {metrics['val_click_logloss']:.4f} AUC {metrics['val_click_auc']:.4f}"
                     if metrics['val_click_logloss'] is not None else ""))
            import math as _math

            overall_ok &= _math.isfinite(metrics["val_install_logloss"]) and 0.0 <= metrics["val_install_auc"] <= 1.0

            # ---- checkpoint write + reload ------------------------------
            runner.best_val_install_logloss = metrics["val_install_logloss"]
            runner.best_epoch = 1
            path = runner.save_checkpoint(1)
            payload = torch.load(path, weights_only=False, map_location="cpu")
            print(f"  checkpoint written: {path.name} ({path.stat().st_size:,} bytes), "
                  f"epoch={payload['epoch']}, exp={payload['experiment_name']}, "
                  f"alpha={payload['alpha']}, temperature={payload['temperature']}, "
                  f"split_hash={payload['split_hash']}")
            reload_ok = runner.verify_checkpoint_reload()
            print(f"  checkpoint reload: {'PASSED' if reload_ok else 'FAILED'}")
            overall_ok &= reload_ok

            del runner

    violations = verify_protected_artifacts(integrity, PROJECT_ROOT)
    if violations:
        for v in violations:
            print(f"  INTEGRITY VIOLATION: {v}")
        overall_ok = False
    else:
        print("\nProtected artifacts unchanged (all hashes/bytes/mtimes identical).")

    elapsed = time.perf_counter() - started
    print(f"\nSTAGE 8 PILOT {'PASSED' if overall_ok else 'FAILED'} in {elapsed:.1f}s")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
