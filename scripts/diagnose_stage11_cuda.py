"""TEMPORARY diagnostic: localize the Stage 11 Round-1 / client_00 CUDA crash.

Context: the detached Stage 11 run (2026-09-19 21:39:56 UTC) died during
Round 1, client_00 (last progress log: batch 400) with
``RuntimeError: CUDA error: unknown error`` surfacing at
``totals["install"] += float(components["install"].item()) * batch_n``
(``scripts/run_stage11.py`` line 262). Windows logged ``nvlddmkm`` Event
ID 153 driver errors at the exact failure wall-clock time. This script is
DIAGNOSTIC ONLY; it re-runs the EXACT failed code path (reused — not
duplicated) with per-phase synchronization to either (a) reproduce the error
at the first failing operation, or (b) demonstrate the healthy path and
quantify memory behavior (retention / growth checks).

What it runs (identical to ``run_stage11.train_one_client_ssl``, round 1):
* real Stage 3 Parquet data, real client_00 (621,801 rows), sampler epoch 0,
* cold seed-42 ``SSLTransformerMMoE`` strict-loaded with the cold seed-42
  global state (exactly ``main()``'s round-1 initialization),
* ``make_client_loader(batch 1024, shuffle=True, seed=42, epoch=0,
  num_workers=0, pin_memory=True)`` — ``set_epoch`` on ``batch_sampler`` only,
* fresh ``AdamW(lr 3e-4, wd 1e-2)``,
* per batch: two corrupted views -> ``ssl_joint_loss(alpha=0.6, T=0.2)``
  -> finite check -> ``backward()`` -> ``optimizer.step()`` -> scalar
  metric extraction (the exact statements of the original loop, timed
  and synchronized individually),

It NEVER: aggregates (no FedAvg), validates, writes/overwrites any
checkpoint, or modifies any Stage 10B / frozen artifact. The only file it
creates is ``artifacts/federated/stage11/diagnostic_r1_c00/diagnostic_report.json``
(a new directory; it collides with nothing — ``run_stage11.py`` only checks
for its best/final checkpoint files). All 12 Stage 11 fingerprint-guarded
files (Stage 1-9 guards + Stage 10B outputs) are fingerprinted
(sha256+size+mtime) before and after and the run FAILS LOUDLY on any change.

Instrumentation:
* ``CUDA_LAUNCH_BLOCKING=1`` is forced BEFORE torch is imported (override in
  the environment with ``CUDA_LAUNCH_BLOCKING=0`` to measure realistic
  asynchronous per-batch timing) so a CUDA error surfaces at the exact
  failing operation; every phase (H2D, zero_grad, forward, joint loss,
  finite check, backward, optimizer step, metrics) is separately
  ``torch.cuda.synchronize()``-d, timed, and independently named so the
  FIRST failing phase is identified,
* per batch: phase times, losses, ``memory_allocated/reserved`` and
  ``max_memory_*``,
* end of run: memory-trajectory verdict (first-5 vs last-5 means), gradient
  finiteness, parameter-change verification, and a ``gc``-based audit of
  live CUDA tensors / graphs (cross-iteration retention check).

Usage:
    python scripts/diagnose_stage11_cuda.py [--batches 50] [--client client_00]
        [--log_every 1]

Exit codes: 0 clean; 3 CUDA error reproduced (phase named in the report);
1 fingerprint/integrity violation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Synchronous CUDA for exact error localization — MUST happen before torch
# is imported. Explicit CUDA_LAUNCH_BLOCKING=0 in the environment wins.
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

import run_stage10b as s10b  # noqa: E402  (reused helpers/constants)
import run_stage11 as s11  # noqa: E402  (reused model/config/fingerprint helpers)

from recsys23_fedrec.data import FeatureSet, ProcessedRecSysDataset  # noqa: E402
from recsys23_fedrec.federated.client_loader import (  # noqa: E402
    client_sampler,
    make_client_loader,
)
from recsys23_fedrec.federated.config import get_federated_config  # noqa: E402
from recsys23_fedrec.ssl.joint_loss import ssl_joint_loss  # noqa: E402
from recsys23_fedrec.training.split import load_split  # noqa: E402

REPORT_DIR_RELPATH = "artifacts/federated/stage11/diagnostic_r1_c00"
MIB = 1024 * 1024


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _refuse_if_runner_alive() -> None:
    """Refuse to run next to a live Stage 10B/11 runner (GPU contention)."""
    offenders = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
        except Exception:  # noqa: BLE001 (psutil access errors)
            continue
        if "run_stage10b" in cmdline or "run_stage11" in cmdline:
            offenders.append(f"PID {proc.info.get('pid')}: {cmdline[:140]}")
    if offenders:
        print("REFUSED: a Stage 10B/11 runner process is already alive:", flush=True)
        for line in offenders:
            print("  " + line, flush=True)
        raise SystemExit(1)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=50,
                        help="number of batches to run (default 50)")
    parser.add_argument("--client", default="client_00",
                        help="client id to simulate (default client_00)")
    parser.add_argument("--log_every", type=int, default=1,
                        help="print every Nth batch (default 1)")
    args = parser.parse_args()

    print("=== STAGE 11 CUDA DIAGNOSTIC (round 1 local SSL step, read-only) ===", flush=True)
    print(f"start: {_now_utc()} | batches={args.batches} client={args.client}", flush=True)
    print(f"CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')} "
          f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')!r}", flush=True)

    _refuse_if_runner_alive()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("REFUSED: CUDA is not available; this diagnostic must run on CUDA.", flush=True)
        return 1

    env_report = {
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "timestamp_utc_start": _now_utc(),
    }
    for key, value in env_report.items():
        print(f"  env {key}: {value}", flush=True)

    # ---- protected-artifact fingerprint (before) ---------------------------
    fingerprint_before = s11._fingerprint11(s11.PROJECT_ROOT)
    print(f"  fingerprinted {len(fingerprint_before)} protected files "
          f"(Stage 1-9 guards + Stage 10B outputs)", flush=True)

    report: dict = {"diagnostic": "stage11_r1_client00_cuda_unknown_error",
                    "mode": "run", "env": env_report,
                    "batches_requested": int(args.batches),
                    "client_id": args.client}

    cfg = get_federated_config()
    feature_set = FeatureSet.load(s11.PROJECT_ROOT / "artifacts" / "preprocessing")
    if feature_set.artifact_hash != s10b.STAGE2_ARTIFACT_HASH:
        print(f"FAIL: Stage 2 artifact hash {feature_set.artifact_hash} != "
              f"{s10b.STAGE2_ARTIFACT_HASH}", flush=True)
        return 1
    dataset = ProcessedRecSysDataset(
        feature_set, s11.PROJECT_ROOT / "artifacts" / "processed" / "train", split="train"
    )
    split = load_split(
        s11.PROJECT_ROOT / "artifacts" / "splits" / "centralized_split",
        expected_total_rows=len(dataset),
    )
    positions = s10b.load_partition_positions(s11.PROJECT_ROOT / s10b.NPZ_RELPATH)
    n_expected = int(positions[args.client].size)
    print(f"  dataset rows {len(dataset):,} | {args.client} rows {n_expected:,} "
          f"({int(np.ceil(n_expected / cfg.local_batch_size))} batches of "
          f"{cfg.local_batch_size})", flush=True)

    # ---- round-1 global state: cold seed-42 SSL init (exactly main()) ------
    global_model = s11.cold_ssl_model(feature_set, seed=cfg.seed)
    global_state = s10b._cpu_state(global_model)
    del global_model

    # ---- loader: exactly train_one_client_ssl(round_number=1) --------------
    loader = make_client_loader(
        dataset,
        split.train_indices,
        positions[args.client],
        batch_size=cfg.local_batch_size,
        shuffle=True,
        seed=cfg.seed,
        epoch=0,  # round 1 -> sampler epoch exactly 0
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    sampler = client_sampler(loader)
    if sampler.epoch != 0:
        print(f"FAIL: sampler epoch {sampler.epoch} != 0", flush=True)
        return 1

    local_model = s11.cold_ssl_model(dataset.feature_set, seed=cfg.seed)
    load_result = local_model.load_state_dict(global_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"FAIL: strict state load reported {load_result}", flush=True)
        return 1
    local_model = local_model.to(device)
    optimizer = torch.optim.AdamW(
        local_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    local_model.train()
    torch.cuda.reset_peak_memory_stats()
    _sync(device)

    records: list[dict] = []
    failure: dict | None = None
    phase = "setup"
    started = time.perf_counter()

    try:
        for batch_idx, batch_cpu in enumerate(loader, start=1):
            if batch_idx > args.batches:
                break
            row: dict = {"batch": batch_idx}

            phase = "H2D"
            t0 = time.perf_counter()
            batch = s10b._to_device(batch_cpu, device)
            _sync(device)
            row["h2d_ms"] = (time.perf_counter() - t0) * 1e3

            phase = "zero_grad"
            t0 = time.perf_counter()
            optimizer.zero_grad()
            _sync(device)
            row["zero_grad_ms"] = (time.perf_counter() - t0) * 1e3

            phase = "forward"
            t0 = time.perf_counter()
            output = local_model(batch, return_diagnostics=False)  # two corrupted views
            _sync(device)
            row["forward_ms"] = (time.perf_counter() - t0) * 1e3

            phase = "joint_loss"
            t0 = time.perf_counter()
            loss, components = ssl_joint_loss(
                output,
                batch,
                alpha=s11.SSL_ALPHA,
                lambda_click=cfg.lambda_click,
                lambda_install=cfg.lambda_install,
                temperature=s11.SSL_TEMPERATURE,
                return_diagnostics=True,
            )
            _sync(device)
            row["loss_ms"] = (time.perf_counter() - t0) * 1e3

            phase = "finite_check"
            t0 = time.perf_counter()
            if not torch.isfinite(loss):  # implicit .item() sync, as in the original
                raise RuntimeError("non-finite joint loss")
            _sync(device)
            row["finite_ms"] = (time.perf_counter() - t0) * 1e3

            phase = "backward"
            t0 = time.perf_counter()
            loss.backward()
            _sync(device)
            row["backward_ms"] = (time.perf_counter() - t0) * 1e3

            phase = "optimizer_step"
            t0 = time.perf_counter()
            optimizer.step()
            _sync(device)
            row["step_ms"] = (time.perf_counter() - t0) * 1e3

            phase = "metrics"  # exact original statements (line 259-263 of run_stage11.py)
            t0 = time.perf_counter()
            batch_n = int(batch["install"].shape[0])
            joint_v = float(loss.item()) * batch_n
            click_v = float(components["click"].item()) * batch_n
            install_v = float(components["install"].item()) * batch_n
            con_v = float(components["contrastive"].item()) * batch_n
            _sync(device)
            row["metrics_ms"] = (time.perf_counter() - t0) * 1e3

            row["joint"] = joint_v / batch_n
            row["click"] = click_v / batch_n
            row["install"] = install_v / batch_n
            row["contrastive"] = con_v / batch_n
            row["batch_rows"] = batch_n
            row["mem_alloc_mib"] = torch.cuda.memory_allocated() / MIB
            row["mem_res_mib"] = torch.cuda.memory_reserved() / MIB
            row["max_alloc_mib"] = torch.cuda.max_memory_allocated() / MIB
            row["max_res_mib"] = torch.cuda.max_memory_reserved() / MIB
            row["total_ms"] = sum(row[k] for k in (
                "h2d_ms", "zero_grad_ms", "forward_ms", "loss_ms", "finite_ms",
                "backward_ms", "step_ms", "metrics_ms"))
            records.append(row)

            if args.log_every and (batch_idx % args.log_every == 0 or batch_idx == 1):
                print(
                    f"  b{batch_idx:04d} | H2D {row['h2d_ms']:6.1f}ms | fwd {row['forward_ms']:7.1f}ms"
                    f" | loss {row['loss_ms']:6.1f}ms | bwd {row['backward_ms']:7.1f}ms"
                    f" | step {row['step_ms']:6.1f}ms | met {row['metrics_ms']:5.1f}ms"
                    f" | total {row['total_ms'] / 1e3:6.2f}s"
                    f" | joint {row['joint']:.4f} (sup {row['click'] + row['install']:.4f},"
                    f" con {row['contrastive']:.4f})"
                    f" | alloc {row['mem_alloc_mib']:7.1f} res {row['mem_res_mib']:7.1f}"
                    f" | maxA {row['max_alloc_mib']:7.1f} maxR {row['max_res_mib']:7.1f} MiB",
                    flush=True,
                )
        phase = "post_loop"
    except RuntimeError as exc:  # CUDA errors surface here, at the named phase
        failure = {
            "phase": phase,
            "batch": len(records) + 1,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"\nDIAGNOSTIC FAILURE in phase '{phase}' at batch {failure['batch']}: {exc!r}",
              flush=True)
        print(traceback.format_exc(), flush=True)

    # ---- post-loop analytics (guarded: the CUDA context may be dead) --------
    analytics: dict = {}
    try:
        _sync(device)
        if records:
            def _mean(key: str, rows: list[dict]) -> float:
                return sum(r[key] for r in rows) / len(rows)

            head, tail = records[:5], records[-5:]
            analytics["alloc_first5_mib"] = _mean("mem_alloc_mib", head)
            analytics["alloc_last5_mib"] = _mean("mem_alloc_mib", tail)
            analytics["reserved_first5_mib"] = _mean("mem_res_mib", head)
            analytics["reserved_last5_mib"] = _mean("mem_res_mib", tail)
            alloc_growth = (analytics["alloc_last5_mib"] / analytics["alloc_first5_mib"] - 1.0) \
                if analytics["alloc_first5_mib"] > 0 else 0.0
            res_growth = (analytics["reserved_last5_mib"] / analytics["reserved_first5_mib"] - 1.0) \
                if analytics["reserved_first5_mib"] > 0 else 0.0
            analytics["alloc_growth_frac"] = round(alloc_growth, 6)
            analytics["reserved_growth_frac"] = round(res_growth, 6)
            analytics["peak_alloc_mib"] = max(r["max_alloc_mib"] for r in records)
            analytics["peak_reserved_mib"] = max(r["max_res_mib"] for r in records)
            analytics["memory_verdict"] = (
                "GROWING" if max(alloc_growth, res_growth) > 0.05 else "STABLE"
            )
            analytics["median_total_ms"] = sorted(r["total_ms"] for r in records)[len(records) // 2]

        if failure is None:
            # ---- gradient health ------------------------------------------------
            grads_missing = sum(1 for p in local_model.parameters() if p.grad is None)
            grads_nonfinite = sum(
                int((~torch.isfinite(p.grad)).sum().item())
                for p in local_model.parameters() if p.grad is not None
            )
            analytics["grads_missing"] = grads_missing
            analytics["grads_nonfinite"] = grads_nonfinite
            # ---- parameters actually changed ------------------------------------
            local_state = s10b._cpu_state(local_model)
            analytics["params_changed"] = sum(
                1 for k in local_state if not torch.equal(local_state[k], global_state[k])
            )
            analytics["params_total"] = len(local_state)
            analytics["local_params_nonfinite"] = s10b._count_nonfinite(local_state)
            del local_state
        # ---- retention audit: live CUDA tensors after the final iteration ----
        gc.collect()
        live = [t for t in gc.get_objects() if torch.is_tensor(t) and t.is_cuda]
        live_bytes = sum(t.numel() * t.element_size() for t in live)
        analytics["live_cuda_tensors"] = len(live)
        analytics["live_cuda_tensor_mib"] = live_bytes / MIB
        analytics["live_cuda_tensors_with_graph"] = sum(1 for t in live if t.grad_fn is not None)
        analytics["allocated_after_gc_mib"] = torch.cuda.memory_allocated() / MIB
        del live
    except RuntimeError as exc:
        analytics["analytics_error"] = repr(exc)

    report.update({
        "timestamp_utc_end": _now_utc(),
        "wall_seconds": time.perf_counter() - started,
        "batches_completed": len(records),
        "failure": failure,
        "analytics": analytics,
        "records": records,
    })

    # ---- report + protected-artifact verification ---------------------------
    report_path = s11.PROJECT_ROOT / REPORT_DIR_RELPATH / "diagnostic_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nreport written: {report_path}", flush=True)
    print(f"summary: batches={len(records)} failure={bool(failure)} "
          f"memory={analytics.get('memory_verdict', 'n/a')} "
          f"peak_alloc={analytics.get('peak_alloc_mib', float('nan')):.0f}MiB "
          f"peak_res={analytics.get('peak_reserved_mib', float('nan')):.0f}MiB",
          flush=True)

    violations = s11._verify_fingerprint11(fingerprint_before, s11.PROJECT_ROOT)
    if violations:
        for violation in violations:
            print(f"VIOLATION: {violation}", flush=True)
        return 1
    print(f"protected artifacts unchanged: all {len(fingerprint_before)} guarded files "
          f"byte-identical (sha256+size+mtime)", flush=True)
    return 3 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
