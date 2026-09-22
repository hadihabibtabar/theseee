"""Stage 8 controlled-experiment runner (full A/C/D runs, one at a time).

Pipeline per the Stage 8 spec:

 1. pre-run integrity snapshot (Stage 2 artifact, split hashes, Stage 6
    checkpoint sha256+mtime, raw byte sizes) + frozen-value verification
 2. CUDA / dataset / split verification
 3. PREFLIGHT: a 1-batch train + 1-batch validation pass on real data that
    verifies forward/backward/optimizer step, finite metrics, and that the
    full-run model initializes identically to the preflight model (seed)
 4. the full controlled run via :class:`ExperimentRunner` (corrected
    per-epoch reshuffle; atomic checkpoints under artifacts/checkpoints/stage8/)
 5. checkpoint payload + reload verification
 6. post-run integrity verification (protected artifacts byte-identical)
 7. result JSON next to the checkpoint

The official test set is NEVER loaded by this script. Experiment B is NOT
run here — it is the frozen Stage 6 canonical baseline (historical sampler).

Example:
    python scripts/run_stage8_experiment.py --experiment A
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import psutil  # noqa: E402
import torch  # noqa: E402

from recsys23_fedrec.data import FeatureSet, ProcessedRecSysDataset  # noqa: E402
from recsys23_fedrec.experiments.config import (  # noqa: E402
    EXPERIMENT_A,
    EXPERIMENT_C,
    EXPERIMENT_D,
    get_experiment_config,
)
from recsys23_fedrec.experiments.integrity import (  # noqa: E402
    PROTECTED_VALUES,
    snapshot_protected_artifacts,
    verify_protected_artifacts,
)
from recsys23_fedrec.experiments.runner import ExperimentRunner  # noqa: E402
from recsys23_fedrec.training.loaders import make_subset_loader
from recsys23_fedrec.training.split import load_split  # noqa: E402

EXPERIMENT_SHORTNAMES = {"A": EXPERIMENT_A, "C": EXPERIMENT_C, "D": EXPERIMENT_D}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(message: str) -> None:
    print(f"\nPREFLIGHT FAILED: {message}")
    raise SystemExit(1)


class RSSMonitor(threading.Thread):
    """Sample this process's RSS until stopped; report the peak."""

    def __init__(self, interval: float = 10.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()
        self.process = psutil.Process()
        self.peak_mib = self.process.memory_info().rss / (1024 * 1024)

    def run(self) -> None:
        while not self._stop.is_set():
            rss = self.process.memory_info().rss / (1024 * 1024)
            self.peak_mib = max(self.peak_mib, rss)
            self._stop.wait(self.interval)

    def stop(self) -> float:
        self._stop.set()
        self.join(timeout=self.interval * 2)
        return self.peak_mib


def preflight_integrity() -> dict:
    """Verify every protected artifact BEFORE the run; return the snapshot."""
    print("=== pre-run integrity verification ===")
    snapshot = snapshot_protected_artifacts(PROJECT_ROOT)
    violations = verify_protected_artifacts(snapshot, PROJECT_ROOT)
    if violations:
        _fail("protected artifacts do not match frozen values: " + "; ".join(violations))
    print(f"  Stage 2 artifact hash:      {snapshot['stage2_artifact_hash']} (expected {PROTECTED_VALUES['stage2_artifact_hash']}) OK")
    print(f"  split train hash:           {snapshot['split_train_index_hash']} (expected {PROTECTED_VALUES['split_train_index_hash']}) OK")
    print(f"  split validation hash:      {snapshot['split_validation_index_hash']} (expected {PROTECTED_VALUES['split_validation_index_hash']}) OK")
    print(f"  Stage 6 checkpoint sha256:  {snapshot['stage6_checkpoint_sha256'][:10]}...{snapshot['stage6_checkpoint_sha256'][-7:]} OK")
    print(f"  Stage 6 checkpoint mtime:   {datetime.fromtimestamp(snapshot['stage6_checkpoint_mtime']).isoformat(timespec='seconds')}")
    print(f"  raw train/test bytes:       {snapshot['raw_train_bytes']:,} / {snapshot['raw_test_bytes']:,} OK")
    return snapshot


def preflight_environment(feature_set: FeatureSet, dataset: ProcessedRecSysDataset) -> None:
    print("=== environment verification ===")
    if not torch.cuda.is_available():
        _fail("CUDA is not available; the protocol requires CUDA")
    print(f"  CUDA: {torch.cuda.get_device_name(0)}")
    print(f"  torch: {torch.__version__}")
    if len(dataset) != 3_485_852:
        _fail(f"dataset row count {len(dataset):,} != expected 3,485,852")
    print(f"  dataset opens: {len(dataset):,} rows, {dataset.num_parts} parts")
    assert feature_set.artifact_hash == PROTECTED_VALUES["stage2_artifact_hash"]
    print("  official test data: NOT loaded (never constructed in this script)")
    # Prior Stage 8 results must exist untouched; this run writes only its own
    # experiment's files (each checkpoint path is experiment-specific).
    a_ckpt = PROJECT_ROOT / "artifacts/checkpoints/stage8/experiment_a_transformer_supervised_best.pt"
    if a_ckpt.exists():
        print(f"  Experiment A checkpoint present: sha256 {_sha256_file(a_ckpt)[:10]}... (guarded: never overwritten)")


def run_preflight(feature_set, dataset, split, config, device, stage8_dir: Path) -> None:
    """1 train batch + 1 val batch end-to-end; parameter-change check."""
    print("=== preflight (1 train batch + 1 validation batch, real data) ===")
    preflight_ckpt = stage8_dir / "preflight_tmp.pt"
    batch_size = config.batch_size
    tiny = ExperimentRunner(
        config=get_experiment_config(config.experiment_name),
        feature_set=feature_set,
        train_dataset=dataset,
        train_indices=split.train_indices[:batch_size],
        val_indices=split.validation_indices[:batch_size],
        checkpoint_path=preflight_ckpt,
        split_hash=split.train_index_hash,
        device=device,
        preprocessing_hash=feature_set.artifact_hash,
    )
    params_before = {n: p.detach().cpu().clone() for n, p in tiny.model.named_parameters()}
    metrics = tiny.train_one_epoch(epoch=1)
    print(f"  train pass: total {metrics['total']:.4f} install {metrics['install']:.4f} "
          f"({'click ' + format(metrics['click'], '.4f') + ' ' if metrics['click'] else ''}"
          f"{'contrastive ' + format(metrics['contrastive'], '.4f') if metrics['contrastive'] else ''})")
    if not all(v == v and abs(v) != float("inf") for v in metrics.values()):
        _fail(f"non-finite preflight training metrics: {metrics}")
    changed = [
        n for n, p in tiny.model.named_parameters() if not torch.equal(p.detach().cpu(), params_before[n])
    ]
    print(f"  parameters changed by optimizer step: {len(changed)}/{len(params_before)}")
    if not changed:
        _fail("optimizer step changed no parameters")
    val = tiny.validate()
    print(f"  validation pass: install logloss {val['val_install_logloss']:.4f} AUC {val['val_install_auc']:.4f}")
    if not all(v is None or (v == v and abs(v) != float("inf")) for v in val.values()):
        _fail(f"non-finite preflight validation metrics: {val}")

    # SSL-specific preflight (Experiments C/D): independent views, unchanged
    # labels, L2-normalized projections — spec-required checks on real data.
    if config.use_ssl:
        print("  ssl preflight: dual-view / label-preservation / projection-norm checks")
        augmentation = tiny.model.augmentations
        sample_loader = make_subset_loader(
            dataset,
            split.train_indices[: config.batch_size],
            batch_size=config.batch_size,
            shuffle=False,
        )
        sample_batch = {k: v.to(device) for k, v in next(iter(sample_loader)).items()}
        labels_before = {
            k: sample_batch[k].clone() for k in ("click", "install") if k in sample_batch
        }
        view1 = augmentation.augment(sample_batch)
        view2 = augmentation.augment(sample_batch)
        cat_diff = (view1["categorical"] != view2["categorical"]).float().mean().item()
        num_diff = (view1["numerical"] != view2["numerical"]).float().mean().item()
        if cat_diff == 0.0 and num_diff == 0.0:
            _fail("two independent augmentations produced identical views")
        for key, before in labels_before.items():
            if not torch.equal(view1[key], before) or not torch.equal(view2[key], before):
                _fail(f"augmentation altered the {key} label")
        with torch.no_grad():
            ssl_out = tiny.model.forward_views(view1, view2, return_diagnostics=True)
        for name in ("z1", "z2"):
            norms = ssl_out[name].norm(dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
                _fail(
                    f"projection {name} is not L2-normalized (norm range "
                    f"{norms.min().item():.6f}-{norms.max().item():.6f})"
                )
        print(f"    views differ (entry fraction): categorical {cat_diff:.4f}, numerical {num_diff:.4f}")
        print("    labels unchanged: True; projection norms 1.0000 for z1/z2 (atol 1e-5)")
        # Output-contract shapes (spec: logits [B], CLS [B, 128], z [B, 64]).
        batch_n = int(sample_batch["categorical"].shape[0])
        z_shape = tuple(ssl_out["z1"].shape)
        if z_shape != (batch_n, 64):
            _fail(f"projection z shape {z_shape} != ({batch_n}, 64)")
        cls = ssl_out.get("cls_view1")
        if cls is not None and tuple(cls.shape) != (batch_n, 128):
            _fail(f"CLS shape {tuple(cls.shape)} != ({batch_n}, 128)")
        head_out = ssl_out.get("view1") or {}
        for key in ("click_logit", "install_logit"):
            if key in head_out and tuple(head_out[key].shape) != (batch_n,):
                _fail(f"{key} shape {tuple(head_out[key].shape)} != ({batch_n},)")
        print(
            f"    shapes: CLS [{batch_n}, 128], z [{batch_n}, 64], "
            f"logits [{batch_n}] (click {'yes' if 'click_logit' in head_out else 'n/a'})"
        )

    # The full-run model must initialize identically to the preflight model.
    full_probe = ExperimentRunner(
        config=get_experiment_config(config.experiment_name),
        feature_set=feature_set,
        train_dataset=dataset,
        train_indices=split.train_indices[:batch_size],
        val_indices=split.validation_indices[:batch_size],
        checkpoint_path=preflight_ckpt.with_suffix(".probe.pt"),
        split_hash=split.train_index_hash,
        device="cpu",
        preprocessing_hash=feature_set.artifact_hash,
    )
    identical = all(
        torch.equal(p.detach().cpu(), params_before[n])
        for n, p in full_probe.model.named_parameters()
    )
    print(f"  full-run model initializes identically to preflight model: {identical}")
    preflight_ckpt.with_suffix(".probe.pt").unlink(missing_ok=True)
    preflight_ckpt.unlink(missing_ok=True)
    if not identical:
        _fail("full-run initialization differs from preflight initialization")
    print("  preflight PASSED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_SHORTNAMES), default="A")
    parser.add_argument("--device", default="auto", help="'auto' (CUDA if available) or 'cpu'")
    parser.add_argument("--log_every", type=int, default=500, help="progress log interval in batches")
    parser.add_argument("--allow_overwrite", action="store_true",
                        help="overwrite an existing checkpoint for this experiment")
    parser.add_argument("--preflight_only", action="store_true",
                        help="run integrity + environment + 1-batch preflight, then stop")
    args = parser.parse_args()

    experiment_name = EXPERIMENT_SHORTNAMES[args.experiment]
    config = get_experiment_config(experiment_name)

    # Windows only: hold an execution-state request so the OS cannot sleep
    # under this console process. The Experiment C run lost ~10 h of epoch 3
    # to Modern-Standby (documented in the Stage 8 report); this changes
    # nothing about training semantics and is a no-op off Windows.
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    stage8_dir = PROJECT_ROOT / "artifacts" / "checkpoints" / "stage8"
    checkpoint_path = stage8_dir / f"{config.experiment_name}_best.pt"
    results_path = stage8_dir / f"{config.experiment_name}_result.json"

    print(f"=== Stage 8 experiment {args.experiment} ({config.experiment_name}) ===")
    print(f"start: {datetime.now().isoformat(timespec='seconds')}")
    print(f"protocol: seed {config.seed}, batch {config.batch_size}, epochs {config.epochs}, "
          f"AdamW lr {config.learning_rate} wd {config.weight_decay}, patience {config.patience}, "
          f"alpha {config.alpha}, corruption {config.corruption_rate}, tau {config.temperature}")
    print("protocol notes: corrected per-epoch reshuffle (Stage 8 sampler); "
          "canonical B = frozen Stage 6 result (historical sampler); "
          "no class weighting/focal/oversampling/scheduler/search")
    if checkpoint_path.exists() and not args.allow_overwrite:
        print(f"\nSTOP: checkpoint already exists: {checkpoint_path}")
        print("The runner does not support resume (documented). Re-run with "
              "--allow_overwrite to discard it and start fresh, or remove it manually.")
        return 1

    if args.experiment != "A":
        print(f"\nNOTE: this invocation targets experiment {args.experiment}; "
              "make sure the intended experiment order is being followed.")

    # ------------------------------------------------------------------
    # 1+2. Integrity + environment verification
    # ------------------------------------------------------------------
    snapshot = preflight_integrity()
    feature_set = FeatureSet.load(PROJECT_ROOT / "artifacts")
    dataset = ProcessedRecSysDataset(
        feature_set, PROJECT_ROOT / "artifacts" / "processed" / "train", split="train"
    )
    preflight_environment(feature_set, dataset)

    split = load_split(
        PROJECT_ROOT / "artifacts" / "splits" / "centralized_split",
        expected_total_rows=len(dataset),
    )
    print(f"  split verified: {split.train_rows:,} train / {split.validation_rows:,} val; "
          f"train {split.train_index_hash} val {split.validation_index_hash}")

    # ------------------------------------------------------------------
    # 3. Preflight
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    stage8_dir.mkdir(parents=True, exist_ok=True)
    run_preflight(feature_set, dataset, split, config, device, stage8_dir)
    if args.preflight_only:
        print("\nRESULT: PREFLIGHT ONLY — stopping before the full run as requested")
        return 0

    # ------------------------------------------------------------------
    # 4. Full run
    # ------------------------------------------------------------------
    print("\n=== full experiment run ===")
    runner = ExperimentRunner(
        config=config,
        feature_set=feature_set,
        train_dataset=dataset,
        train_indices=split.train_indices,
        val_indices=split.validation_indices,
        checkpoint_path=checkpoint_path,
        split_hash=split.train_index_hash,
        device=device,
        results_path=results_path,
        preprocessing_hash=feature_set.artifact_hash,
        validation_index_hash=split.validation_index_hash,
        log_every=args.log_every,
    )
    print(f"batches per epoch: train {math.ceil(split.train_rows / config.batch_size)}, "
          f"validation {math.ceil(split.validation_rows / config.batch_size)}")
    monitor = RSSMonitor(interval=10.0)
    monitor.start()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()

    try:
        result = runner.fit()
    except Exception as error:  # noqa: BLE001 - report exactly, change nothing
        monitor.stop()
        print(f"\nRUNTIME FAILURE: {type(error).__name__}: {error}")
        print("Protocol unchanged; no checkpoint beyond the last improving epoch is valid.")
        raise

    wall = time.perf_counter() - wall_start
    peak_rss = monitor.stop()
    gpu_alloc = gpu_reserved = None
    if device.type == "cuda":
        torch.cuda.synchronize()
        gpu_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
        gpu_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)

    # ------------------------------------------------------------------
    # 5. Checkpoint payload + reload verification
    # ------------------------------------------------------------------
    print("\n=== checkpoint verification ===")
    payload = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  payload: stage={payload['stage']} experiment={payload['experiment']} "
          f"epoch={payload['epoch']} seed={payload['seed']} "
          f"params={payload['parameter_count']:,} split_train={payload['train_index_hash']} "
          f"split_val={payload['validation_index_hash']} prep={payload['preprocessing_hash']}")
    print(f"  sampler protocol: {payload['sampler_protocol']}")
    print(f"  written: {payload['checkpoint_timestamp']}")
    print(f"  reload verified by runner (fresh model, one val batch, atol 1e-5): {result.reload_verified}")
    if not result.reload_verified:
        _fail("checkpoint reload verification failed")
    print(f"  checkpoint sha256: {_sha256_file(checkpoint_path)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== experiment summary ===")
    print(f"{'epoch':>5} | {'seconds':>9} | {'train total':>11} | {'train install':>13} | "
          f"{'val LL':>8} | {'val AUC':>8}")
    for record in result.epochs:
        print(f"{record['epoch']:>5} | {record['duration_seconds']:>9.1f} | "
              f"{record['train_total_loss']:>11.4f} | {record['train_install_loss']:>13.4f} | "
              f"{record['val_install_logloss']:>8.4f} | {record['val_install_auc']:>8.4f}")
    print(f"rows: train {split.train_rows:,} / val {split.validation_rows:,} "
          f"(batches {result.epochs[0]['n_train_batches']} / {result.epochs[0]['n_val_batches']})")
    print(f"best epoch: {result.best_epoch}")
    print(f"best val install LogLoss: {result.best_val_install_logloss!r}")
    print(f"best val install AUC:     {result.best_val_install_auc:.4f}")
    print(f"total runtime: {wall:.1f}s ({wall / 3600:.2f} h)")
    print(f"peak RSS: {peak_rss:.0f} MiB")
    if gpu_alloc is not None:
        print(f"peak GPU allocated/reserved: {gpu_alloc:.0f} / {gpu_reserved:.0f} MiB")
    print(f"result JSON: {results_path}")
    print("official test set: NOT evaluated (never loaded)")

    # ------------------------------------------------------------------
    # 6. Post-run integrity verification
    # ------------------------------------------------------------------
    print("\n=== post-run integrity verification ===")
    violations = verify_protected_artifacts(snapshot, PROJECT_ROOT)
    if violations:
        for violation in violations:
            print(f"  VIOLATION: {violation}")
        return 1
    print("  protected artifacts unchanged: Stage 2 hash, split hashes, "
          "Stage 6 checkpoint sha256+mtime, raw train/test bytes all OK")

    print(f"\nRESULT: STAGE 8 EXPERIMENT {args.experiment} COMPLETE")
    print("No other Stage 8 experiment was launched by this process.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
