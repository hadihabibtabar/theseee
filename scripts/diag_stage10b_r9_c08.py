"""TEMPORARY diagnostic: reproduce the Stage 10B Round-9 / Client-08 CUDA crash.

Context: the detached Stage 10B run completed Rounds 1-8 and crashed during
Round 9 at client_08 (~batch 200) with ``RuntimeError: CUDA error: unknown
error`` inside ``losses["total"].backward()``. ``stage10b_latest.pt`` still
holds the completed Round-8 global state. This script is DIAGNOSTIC ONLY:

* it loads ``stage10b_latest.pt`` READ-ONLY and verifies it is Round 8,
* it runs ONLY Round 9 -> client_08 -> exactly one local epoch, through the
  exact same code path as ``scripts/run_stage10b.py`` (which is imported and
  reused, not duplicated): Stage 9 ``make_client_loader`` with
  ``epoch=8`` (sampler epoch = round - 1) set on ``loader.batch_sampler``
  only, fresh cold seed-42 model strict-loaded with the Round-8 global
  state, fresh AdamW (3e-4 / 1e-2), BCE(click)+BCE(install) via the existing
  ``mmoe_loss`` with lambdas 1.0/1.0, batch 1024, shuffle=True,
  num_workers=0, pin_memory on CUDA, same device selection (`auto` = CUDA if
  available),
* it NEVER aggregates (no FedAvg), NEVER validates, NEVER writes or modifies
  any checkpoint; the only output is a JSON report plus stdout under
  ``artifacts/federated/stage10/diagnostic_r9_c08/``,
* it refuses to run if any Stage 10B runner process is alive, and it
  fingerprints ``stage10b_best.pt`` / ``stage10b_latest.pt`` /
  ``stage10b_result.json`` before and after and fails loudly if any of them
  changed.

Modes (the diagnostic epoch is NEVER started automatically):
    python scripts/diag_stage10b_r9_c08.py --static_only   # imports + checkpoint structure + data layer
    python scripts/diag_stage10b_r9_c08.py --run           # the one diagnostic local epoch
    python scripts/diag_stage10b_r9_c08.py --run --cuda_launch_blocking  # same, with synchronous CUDA errors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
for _path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import psutil  # noqa: E402
import torch  # noqa: E402

import run_stage10b as s10b  # noqa: E402  (reuse: cold_model, verify_data_layer, monitors, constants)

from recsys23_fedrec.data import FeatureSet, ProcessedRecSysDataset  # noqa: E402
from recsys23_fedrec.federated.client_loader import (  # noqa: E402
    client_sampler,
    make_client_loader,
)
from recsys23_fedrec.federated.config import get_federated_config  # noqa: E402
from recsys23_fedrec.models.mmoe import mmoe_loss  # noqa: E402
from recsys23_fedrec.training.split import load_split  # noqa: E402

DIAG_CLIENT_ID = "client_08"
DIAG_ROUND = 9
DIAG_SAMPLER_EPOCH = DIAG_ROUND - 1  # frozen convention: sampler epoch = round - 1
DIAG_DIR_RELPATH = "artifacts/federated/stage10/diagnostic_r9_c08"
STAGE10B_GUARDED = (
    "artifacts/federated/stage10/stage10b_best.pt",
    "artifacts/federated/stage10/stage10b_latest.pt",
    "artifacts/federated/stage10/stage10b_result.json",
)
FROZEN_PROTOCOL_EXPECTATION = {
    "seed": 42,
    "rounds": 10,
    "local_epochs": 1,
    "local_batch_size": 1024,
    "optimizer": "adamw",
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "lambda_click": 1.0,
    "lambda_install": 1.0,
    "use_ssl": False,
    "aggregation": "sample_weighted_fedavg",
    "local_sampler_epoch_mode": "round_minus_one",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stage10b_runner_pids() -> list[int]:
    """PIDs of any live Stage 10B runner (never this diagnostic process)."""
    pids = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
        except Exception:
            continue
        if proc.info["pid"] == os.getpid():
            continue
        if "run_stage10b" in cmdline and "diag_stage10b" not in cmdline:
            pids.append(proc.info["pid"])
    return pids


def _fingerprint_stage10b() -> dict:
    out = {}
    for relpath in STAGE10B_GUARDED:
        path = PROJECT_ROOT / relpath
        out[relpath] = {
            "sha256": s10b._sha256_file(path),
            "size": os.path.getsize(path),
            "mtime": os.path.getmtime(path),
        }
    return out


def _check_fingerprint_unchanged(before: dict) -> list[str]:
    after = _fingerprint_stage10b()
    return [f"Stage 10B artifact changed: {rel}" for rel, exp in before.items()
            if after.get(rel) != exp]


def _fail(report: dict | None, message: str) -> int:
    print(f"\nDIAGNOSTIC FAILED: {message}", flush=True)
    if report is not None:
        report["outcome"] = "failed"
        report["failure"] = message
        _write_report(report)
    return 1


def _write_report(report: dict) -> None:
    out_dir = PROJECT_ROOT / DIAG_DIR_RELPATH
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "diagnostic_report.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    print(f"diagnostic report written: {path}", flush=True)


def load_and_verify_round8_state(feature_set: FeatureSet) -> dict:
    """Read-only load of stage10b_latest.pt + full Round-8 verification."""
    latest_path = PROJECT_ROOT / s10b.LATEST_CKPT_RELPATH
    print(f"loading (read-only): {latest_path}", flush=True)
    payload = torch.load(latest_path, map_location="cpu", weights_only=False)

    if payload.get("stage") != s10b.PROTOCOL_ID:
        raise ValueError(f"payload stage {payload.get('stage')!r} != {s10b.PROTOCOL_ID!r}")
    if payload.get("kind") != "latest":
        raise ValueError(f"payload kind {payload.get('kind')!r} != 'latest'")
    if int(payload["round"]) != 8:
        raise ValueError(f"payload round {payload['round']} != 8 (must be completed Round 8)")
    history_rounds = [h["round"] for h in payload["history"]]
    if history_rounds != [1, 2, 3, 4, 5, 6, 7, 8]:
        raise ValueError(f"payload history rounds {history_rounds} != [1..8]")
    if int(payload["seed"]) != 42 or payload.get("partition_hash") != s10b.EXPECTED_PARTITION_HASH:
        raise ValueError("payload seed/partition hash mismatch against the frozen protocol")
    if payload.get("split_train_index_hash") != s10b.SPLIT_TRAIN_HASH or \
            payload.get("split_validation_index_hash") != s10b.SPLIT_VALIDATION_HASH:
        raise ValueError("payload split hashes mismatch against the frozen splits")
    state = payload["model_state_dict"]
    if len(state) != 138:
        raise ValueError(f"state has {len(state)} entries, expected 138")
    if s10b._count_nonfinite(state):
        raise ValueError("Round-8 global state contains non-finite values")

    # structural check: strict-load into a fresh cold model (CPU)
    fresh = s10b.cold_model(feature_set, seed=42)
    load_result = fresh.load_state_dict(state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(f"strict reload reported problems: {load_result}")
    if sum(p.numel() for p in fresh.parameters()) != s10b.EXPECTED_PARAMETER_COUNT:
        raise ValueError("parameter count mismatch against the frozen architecture")
    del fresh

    print(f"  [OK] stage={payload['stage']} kind={payload['kind']} round={payload['round']}", flush=True)
    print(f"  [OK] history rounds {history_rounds}; best round {payload['best_round']} "
          f"(Install LogLoss {payload['best_validation_install_logloss']:.6f})", flush=True)
    print(f"  [OK] seed 42, partition {payload['partition_hash']}, splits "
          f"{payload['split_train_index_hash']}/{payload['split_validation_index_hash']}", flush=True)
    print(f"  [OK] {len(state)} state entries strictly reloadable into a fresh cold seed-42 model",
          flush=True)
    return state


def verify_environment_and_data(report: dict) -> tuple:
    """Frozen config + CUDA + dataset/split/partition + no-runner guard."""
    cfg = get_federated_config()
    drifted = {k: (getattr(cfg, k), v) for k, v in FROZEN_PROTOCOL_EXPECTATION.items()
               if getattr(cfg, k) != v}
    if drifted:
        raise ValueError(f"frozen protocol drifted: {drifted}")
    print("  [OK] FederatedConfig matches the frozen Stage 10A protocol", flush=True)

    runner_pids = _stage10b_runner_pids()
    if runner_pids:
        raise RuntimeError(f"refusing to run: Stage 10B runner process(es) alive: {runner_pids}")
    print("  [OK] no Stage 10B runner process is alive (single-copy guard)", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # same as Stage 10B 'auto'
    cuda_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    print(f"  device: {device}" + (f" ({cuda_name})" if cuda_name else ""), flush=True)

    feature_set = FeatureSet.load(PROJECT_ROOT / "artifacts" / "preprocessing")
    if feature_set.artifact_hash != s10b.STAGE2_ARTIFACT_HASH:
        raise ValueError("Stage 2 artifact hash mismatch")
    dataset = ProcessedRecSysDataset(
        feature_set, PROJECT_ROOT / "artifacts" / "processed" / "train", split="train"
    )
    split = load_split(
        PROJECT_ROOT / "artifacts" / "splits" / "centralized_split",
        expected_total_rows=len(dataset),
    )
    counts, positions = s10b.verify_data_layer(dataset, split, PROJECT_ROOT, cfg)
    print(f"  [OK] client_08 expected samples: {counts[DIAG_CLIENT_ID]:,}", flush=True)
    return cfg, device, cuda_name, dataset, split, positions, counts


def run_diagnostic_epoch(report: dict, cfg, device, cuda_name, dataset, split, positions,
                         global_state: dict, cuda_launch_blocking: bool, log_every: int) -> int:
    """EXACTLY ONE local epoch for Round 9 / client_08 with diagnostics. No FedAvg, no validation."""
    started = time.perf_counter()
    client_positions = positions[DIAG_CLIENT_ID]
    n_expected = int(client_positions.size)
    report.update({
        "client_id": DIAG_CLIENT_ID,
        "round": DIAG_ROUND,
        "sampler_epoch": DIAG_SAMPLER_EPOCH,
        "expected_samples": n_expected,
        "batch_size": cfg.local_batch_size,
        "device": str(device),
        "cuda_device_name": cuda_name,
        "cuda_launch_blocking": cuda_launch_blocking,
        "log_every_batches": log_every,
    })

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rss_monitor = s10b.RSSMonitor(interval=10.0)
    rss_monitor.start()

    loader = make_client_loader(
        dataset,
        split.train_indices,
        client_positions,
        batch_size=cfg.local_batch_size,
        shuffle=True,
        seed=cfg.seed,
        epoch=DIAG_SAMPLER_EPOCH,  # applied to loader.batch_sampler ONLY (inside factory)
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    sampler = client_sampler(loader)  # asserts PartShuffledBatchSampler; loader.sampler untouched
    if sampler.epoch != DIAG_SAMPLER_EPOCH:
        return _fail(report, f"sampler epoch {sampler.epoch} != {DIAG_SAMPLER_EPOCH}")
    report["sampler_kind"] = type(sampler).__name__
    report["loader_sampler_kind"] = type(getattr(loader, "sampler", None)).__name__
    print(f"  [OK] loader ready: sampler {report['sampler_kind']} epoch={sampler.epoch} "
          f"(loader.sampler type {report['loader_sampler_kind']} untouched)", flush=True)

    local_model = s10b.cold_model(dataset.feature_set, seed=cfg.seed)
    load_result = local_model.load_state_dict(global_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        return _fail(report, f"local strict state load reported {load_result}")
    local_model = local_model.to(device)
    optimizer = torch.optim.AdamW(
        local_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    totals = {"total": 0.0, "click": 0.0, "install": 0.0}
    seen = 0
    n_batches = 0
    progress_log: list[dict] = []
    outcome = "completed"
    exception_info: dict | None = None
    print(f"\n=== diagnostic epoch: round {DIAG_ROUND} {DIAG_CLIENT_ID} "
          f"(sampler epoch {DIAG_SAMPLER_EPOCH}, {n_expected:,} rows) ===", flush=True)

    try:
        for batch in loader:
            batch_number = n_batches + 1  # 1-based, consistent with the runner's log lines
            try:
                batch = s10b._to_device(batch, device)
                optimizer.zero_grad()
                output = local_model(batch, return_diagnostics=False)
                losses = mmoe_loss(
                    output, batch,
                    lambda_click=cfg.lambda_click, lambda_install=cfg.lambda_install,
                )
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError(
                        f"non-finite local loss at round {DIAG_ROUND} batch {batch_number}"
                    )
                losses["total"].backward()  # <- crash site of the original failure
                optimizer.step()
            except Exception:
                outcome = "exception"
                exc_type, exc_value, exc_tb = sys.exc_info()
                gpu_snap = {}
                try:
                    if device.type == "cuda":
                        gpu_snap = {
                            "gpu_allocated_mib": torch.cuda.memory_allocated() / (1024 * 1024),
                            "gpu_reserved_mib": torch.cuda.memory_reserved() / (1024 * 1024),
                        }
                except Exception:
                    pass
                exception_info = {
                    "type": exc_type.__name__ if exc_type else "Exception",
                    "message": str(exc_value),
                    "batch_number": batch_number,
                    "elapsed_seconds": time.perf_counter() - started,
                    "rows_seen": seen,
                    "gpu_at_failure": gpu_snap,
                    "traceback": traceback.format_exc(),
                }
                print(f"\nEXCEPTION at batch {batch_number} after {seen:,} rows "
                      f"({time.perf_counter() - started:.1f}s): "
                      f"{exception_info['type']}: {exception_info['message']}", flush=True)
                break

            batch_n = int(batch["install"].shape[0])
            totals["total"] += float(losses["total"].item()) * batch_n
            totals["click"] += float(losses["click"].item()) * batch_n
            totals["install"] += float(losses["install"].item()) * batch_n
            seen += batch_n
            n_batches = batch_number

            if log_every and batch_number % log_every == 0:
                elapsed = time.perf_counter() - started
                entry = {
                    "batch": batch_number,
                    "rows_seen": seen,
                    "running_total_loss": totals["total"] / seen,
                    "elapsed_seconds": elapsed,
                }
                if device.type == "cuda":
                    entry["gpu_allocated_mib"] = torch.cuda.memory_allocated() / (1024 * 1024)
                    entry["gpu_reserved_mib"] = torch.cuda.memory_reserved() / (1024 * 1024)
                progress_log.append(entry)
                extra = (f" | gpu {entry.get('gpu_allocated_mib', 0):.0f}/"
                         f"{entry.get('gpu_reserved_mib', 0):.0f} MiB") if device.type == "cuda" else ""
                print(f"    r{DIAG_ROUND} {DIAG_CLIENT_ID} batch {batch_number} | running loss "
                      f"{entry['running_total_loss']:.4f} | {elapsed:.1f}s{extra}", flush=True)
    finally:
        peak_rss = rss_monitor.stop()

    total_elapsed = time.perf_counter() - started
    report.update({
        "outcome": outcome,
        "batches_completed": n_batches,
        "rows_seen": seen,
        "elapsed_seconds": total_elapsed,
        "peak_rss_mib": peak_rss,
        "progress_log": progress_log,
        "exception": exception_info,
    })
    if device.type == "cuda":
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        report["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        report["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / (1024 * 1024)
        print(f"peak GPU allocated/reserved: {report['peak_gpu_allocated_mib']:.0f}/"
              f"{report['peak_gpu_reserved_mib']:.0f} MiB", flush=True)
    print(f"peak RSS: {peak_rss:.0f} MiB | elapsed: {total_elapsed:.1f}s | "
          f"batches completed: {n_batches} | rows seen: {seen:,}", flush=True)

    if outcome == "exception":
        report["traceback"] = exception_info["traceback"]
        print(exception_info["traceback"], flush=True)
        return 1

    if seen != n_expected:
        return _fail(report, f"epoch saw {seen:,} rows, partition says {n_expected:,}")
    local_state = s10b._cpu_state(local_model)
    if s10b._count_nonfinite(local_state):
        return _fail(report, "non-finite local parameters after training")
    changed = [k for k in local_state if not torch.equal(local_state[k], global_state[k])]
    if not changed:
        return _fail(report, "local training changed no parameters")
    report.update({
        "local_losses": {
            "total": totals["total"] / seen,
            "click": totals["click"] / seen,
            "install": totals["install"] / seen,
        },
        "params_changed": len(changed),
        "params_total": len(local_state),
    })
    print(f"epoch completed: loss {totals['total'] / seen:.4f} "
          f"(click {totals['click'] / seen:.4f}, install {totals['install'] / seen:.4f}) | "
          f"params changed {len(changed)}/{len(local_state)} | local state NOT saved anywhere",
          flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static_only", action="store_true",
                      help="validate imports/checkpoint structure/data layer, then stop")
    mode.add_argument("--run", action="store_true",
                      help="run the single diagnostic Round-9 client_08 local epoch")
    parser.add_argument("--device", default="auto", help="'auto' (CUDA if available) or 'cpu'")
    parser.add_argument("--log_every", type=int, default=50, help="progress interval (batches)")
    parser.add_argument("--cuda_launch_blocking", action="store_true",
                        help="set CUDA_LAUNCH_BLOCKING=1 for synchronous error reporting "
                             "(does not change the math, only error timing)")
    args = parser.parse_args()

    print("=== STAGE 10B DIAGNOSTIC: round 9 / client_08 single-epoch reproduction ===", flush=True)
    print(f"start: {_now_iso()} | mode: {'static_only' if args.static_only else 'run'}", flush=True)

    fingerprint = _fingerprint_stage10b()
    print(f"  [OK] {len(fingerprint)} Stage 10B artifacts fingerprinted (read-only)", flush=True)

    report: dict = {
        "diagnostic": "stage10b_r9_c08_cuda_unknown_error",
        "mode": "static_only" if args.static_only else "run",
        "timestamp_utc": _now_iso(),
        "protected_stage10b_before": fingerprint,
    }

    try:
        cfg, device, cuda_name, dataset, split, positions, counts = \
            verify_environment_and_data(report)
        if args.device != "auto":
            device = torch.device(args.device)
        if args.cuda_launch_blocking and device.type == "cuda":
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        global_state = load_and_verify_round8_state(dataset.feature_set)
        report["checkpoint_round_loaded"] = 8
    except Exception:
        print(traceback.format_exc(), flush=True)
        report["outcome"] = "static_validation_failed"
        report["traceback"] = traceback.format_exc()
        _write_report(report)
        return 1

    if args.static_only:
        print("\nRESULT: STATIC VALIDATION PASSED - diagnostic epoch NOT started (as requested)", flush=True)
        return 0

    code = run_diagnostic_epoch(
        report, cfg, device, cuda_name, dataset, split, positions, global_state,
        cuda_launch_blocking=args.cuda_launch_blocking, log_every=args.log_every,
    )
    violations = _check_fingerprint_unchanged(fingerprint)
    report["protected_stage10b_artifacts_unchanged"] = not violations
    if violations:
        print("\nCRITICAL: " + "; ".join(violations), flush=True)
        return 1
    print("  [OK] stage10b_best.pt / stage10b_latest.pt / stage10b_result.json unchanged "
          "(sha256+size+mtime)", flush=True)
    report["finished_at_utc"] = _now_iso()
    _write_report(report)
    return code


if __name__ == "__main__":
    sys.exit(main())
