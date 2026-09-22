"""Stage 6 centralized training runner.

Creates/loads the deterministic train/validation split, builds subset
loaders over the Stage 3 memory-bounded Parquet dataset, and trains the
Stage 5 Transformer + MMoE model with the Stage 6 configuration.

Modes:
    --smoke          tiny run (2 train batches, 1 val batch) for sanity
    (default)        the full 3-epoch centralized baseline

The official test set is only opened for an inference-only schema
compatibility check (clearly labeled; never used for training, validation,
model selection, or early stopping).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recsys23_fedrec.data import (  # noqa: E402
    FeatureSet,
    ProcessedRecSysDataset,
    collate_test,
    create_dataloader,
)
from recsys23_fedrec.models import TransformerMMoE  # noqa: E402
from recsys23_fedrec.preprocessing.config import SEED  # noqa: E402
from recsys23_fedrec.training import (  # noqa: E402
    Trainer,
    TrainingConfig,
    make_subset_loader,
    make_split,
    resolve_device,
)

SPLIT_PATH = PROJECT_ROOT / "artifacts" / "splits" / "centralized_split"
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "centralized_transformer_mmoe_best.pt"


def set_deterministic_seed(seed: int, deterministic_cuda: bool = False) -> dict:
    """Seed python/numpy/torch consistently with the project seed.

    Deterministic CUDA algorithms are opt-in (they can slow kernels and some
    ops have no deterministic implementation); by default we rely on seed
    reproducibility only. The choice is recorded in checkpoints.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cuda:
        torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "deterministic_cuda_algorithms": bool(deterministic_cuda),
        "cudnn_benchmark": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts_dir", default=str(PROJECT_ROOT / "artifacts"))
    parser.add_argument("--smoke", action="store_true", help="tiny sanity run")
    parser.add_argument("--epochs", type=int, default=None, help="override epoch count")
    parser.add_argument("--batch_size", type=int, default=None, help="override batch size")
    parser.add_argument(
        "--device", default=None, help="override device (e.g. cpu); default is config 'auto'"
    )
    args = parser.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    config = TrainingConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device
    if args.smoke:
        config.batch_size = 128
        config.epochs = 1

    seed_info = set_deterministic_seed(config.seed)
    device = resolve_device(config.device)
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"config: {json.dumps(config.to_dict())}")
    print(f"seed policy: {json.dumps(seed_info)}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    feature_set = FeatureSet.load(artifacts_dir)
    train_dataset = ProcessedRecSysDataset(feature_set, artifacts_dir / "processed" / "train", split="train")
    print(f"processed train rows: {len(train_dataset):,} ({train_dataset.num_parts} parts)")

    split_path = artifacts_dir / "splits" / "centralized_split"
    if (split_path.with_suffix(".json")).is_file():
        split = __import__("recsys23_fedrec.training", fromlist=["load_split"]).load_split(
            split_path, expected_total_rows=len(train_dataset)
        )
        print("split: loaded existing (hash-verified)")
    else:
        split = make_split(len(train_dataset), validation_ratio=0.10, seed=SEED)
        split.save(split_path)
        print("split: created and saved")
    meta = split.to_metadata()
    print(
        f"split: {split.train_rows:,} train / {split.validation_rows:,} validation "
        f"(seed {split.seed}, ratio {split.validation_ratio}); "
        f"hashes train={split.train_index_hash} val={split.validation_index_hash}"
    )

    if args.smoke:
        train_indices = split.train_indices[: 2 * config.batch_size]
        val_indices = split.validation_indices[: config.batch_size]
    else:
        train_indices = split.train_indices
        val_indices = split.validation_indices

    train_loader = make_subset_loader(
        train_dataset, train_indices, batch_size=config.batch_size,
        shuffle=True, seed=config.seed, pin_memory=(device.type == "cuda"), num_workers=0,
    )
    val_loader = make_subset_loader(
        train_dataset, val_indices, batch_size=config.batch_size,
        shuffle=False, pin_memory=(device.type == "cuda"), num_workers=0,
    )
    print(f"batches per epoch: train {len(train_loader)}, validation {len(val_loader)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    torch.manual_seed(config.seed)
    model = TransformerMMoE(feature_set)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    # Smoke runs must never touch the real best checkpoint: the trainer would
    # see best=inf and immediately overwrite it with an untrained model.
    if args.smoke:
        checkpoint_path = artifacts_dir / "checkpoints" / "smoke_centralized_best.pt"
    else:
        checkpoint_path = CHECKPOINT_PATH

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=checkpoint_path,
        split_hash=split.train_index_hash,
        device=device,
        patience=config.early_stopping_patience,
    )
    rss_start = psutil.Process().memory_info().rss / (1024 * 1024)
    peak = rss_start
    started = time.perf_counter()

    summary = trainer.fit(max_epochs=config.epochs)
    peak = max(peak, psutil.Process().memory_info().rss / (1024 * 1024))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()

    print("\n=== summary ===")
    print(f"epochs completed:        {summary['epochs_completed']}")
    print(f"best epoch:              {summary['best_epoch']}")
    print(f"best val install logloss: {summary['best_val_install_logloss']:.4f}")
    print(f"best val install AUC:     {summary['best_val_install_auc']:.4f}")
    print(f"  click logloss:          {summary['best_val_click_logloss']:.4f}")
    print(f"  click AUC:              {summary['best_val_click_auc']:.4f}")
    print(f"total training time:     {summary['total_seconds']:.1f}s")
    print(f"RSS start/peak:          {rss_start:.0f} / {peak:.0f} MiB")
    if torch.cuda.is_available():
        print(f"peak GPU allocated:      {gpu_mib:.0f} MiB")

    if args.smoke:
        print("SMOKE RUN PASSED")
        return 0

    # ------------------------------------------------------------------
    # Checkpoint reload verification (spec 18)
    # ------------------------------------------------------------------
    print("\n=== checkpoint reload verification ===")
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False, map_location="cpu")
    fresh_model = TransformerMMoE(feature_set)
    fresh_model.load_state_dict(checkpoint["model_state_dict"])
    fresh_model = fresh_model.to(device).eval()
    trainer.model.eval()
    reload_ok = True
    with torch.no_grad():
        for batch in val_loader:
            batch_d = {k: v.to(device) for k, v in batch.items()}
            out_fresh = fresh_model(batch_d, return_diagnostics=False)
            out_saved = trainer.model(batch_d, return_diagnostics=False)
            for key in ("click_logit", "install_logit"):
                finite = torch.isfinite(out_fresh[key]).all().item()
                close = torch.allclose(out_fresh[key], out_saved[key], atol=1e-5)
                print(f"  {key}: finite={finite}, matches_saved={close}")
                reload_ok = reload_ok and finite and close
            break  # one deterministic validation batch suffices
    print(f"checkpoint payload: epoch={checkpoint['epoch']}, seed={checkpoint['seed']}, "
          f"split_hash={checkpoint['split_hash']}, "
          f"best_val_install_logloss={checkpoint['best_validation_install_logloss']:.4f}")
    print(f"RELOAD {'PASSED' if reload_ok else 'FAILED'}")

    # ------------------------------------------------------------------
    # Official test set: schema/inference compatibility ONLY (spec 19)
    # ------------------------------------------------------------------
    print("\n=== official test set: schema/inference compatibility only ===")
    test_dataset = ProcessedRecSysDataset(feature_set, artifacts_dir / "processed" / "test", split="test")
    print(f"test dataset length: {len(test_dataset):,} (labels absent, features only)")
    tb = collate_test([test_dataset[i] for i in range(64)])
    fresh_model = fresh_model.cpu()
    with torch.no_grad():
        out = fresh_model(tb, return_diagnostics=False)
    print(f"one inference-only forward on 64 test rows: click {tuple(out['click_logit'].shape)}, "
          f"install {tuple(out['install_logit'].shape)}, finite="
          f"{bool(torch.isfinite(out['click_logit']).all() and torch.isfinite(out['install_logit']).all())}")
    print("NOT used for training, validation, model selection, or early stopping.")

    print("\nRESULT: STAGE 6 CENTRALIZED TRAINING COMPLETE" if reload_ok else "\nRESULT: RELOAD VERIFICATION FAILED")
    return 0 if reload_ok else 1


if __name__ == "__main__":
    sys.exit(main())
