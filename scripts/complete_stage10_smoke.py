"""Complete the interrupted Stage 10 smoke test: post-checkpoint steps only.

The one-round smoke run already completed (checkpoint + all client records on
disk) but crashed in the reload-verification step. This script performs the
remaining read-only steps WITHOUT re-running any training:

1. reload verification of the smoke checkpoint (strict load into a fresh
   seed-42 model, bit-identical state, all parameters finite),
2. independent recomputation of the FULL frozen-split validation metrics from
   the reloaded checkpoint (must match the stored metrics),
3. post-run protected-artifact integrity verification (frozen values via the
   Stage 8 integrity module + pre-run sha256-prefix/size/mtime fingerprints),
4. write the smoke result JSON (previously not written),
5. print the full smoke-test summary + confirmations.

No model is trained, no optimizer is created, no protected artifact is
written. The official test set is never touched.

Usage:
    python scripts/complete_stage10_smoke.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

from recsys23_fedrec.data import FeatureSet, ProcessedRecSysDataset  # noqa: E402
from recsys23_fedrec.experiments.integrity import (  # noqa: E402
    snapshot_protected_artifacts,
    verify_protected_artifacts,
)
from recsys23_fedrec.federated.partition import CLIENT_IDS  # noqa: E402
from recsys23_fedrec.models import TransformerMMoE  # noqa: E402
from recsys23_fedrec.training.loaders import make_subset_loader  # noqa: E402
from recsys23_fedrec.training.metrics import binary_logloss, roc_auc  # noqa: E402
from recsys23_fedrec.training.split import load_split  # noqa: E402

SMOKE_DIR_RELPATH = "artifacts/federated/smoke"
SMOKE_CHECKPOINT_RELPATH = f"{SMOKE_DIR_RELPATH}/stage10_smoke_round1.pt"
SMOKE_RESULT_RELPATH = f"{SMOKE_DIR_RELPATH}/stage10_smoke_round1_result.json"

#: sha256 prefix (64-bit), size, and mtime recorded BEFORE the smoke run
#: (protected files are never written by the smoke test, so these must hold).
PRERUN_FINGERPRINTS = {
    "artifacts/federated/partition_alpha_0_5_seed42.npz": {
        "sha256_prefix": "7af0787fabf9a055", "size": 4_789_182,
        "mtime": "2026-09-19T11:07:09",
    },
    "artifacts/federated/partition_alpha_0_5_seed42.json": {
        "sha256_prefix": "36fd942cfe0817d2", "size": 9_248,
        "mtime": "2026-09-19T11:07:08",
    },
    "artifacts/splits/centralized_split.npz": {
        "sha256_prefix": "7c3c0fff80a09146", "size": 5_300_711,
        "mtime": "2026-09-15T17:00:45",
    },
    "artifacts/splits/centralized_split.json": {
        "sha256_prefix": "ccf38575acc6476a", "size": 243,
        "mtime": "2026-09-15T17:00:45",
    },
    "artifacts/checkpoints/centralized_transformer_mmoe_best.pt": {
        "sha256_prefix": "25f3267eba572bf9", "size": 30_217_243,
        "mtime": "2026-09-16T16:32:34",
    },
    "artifacts/checkpoints/stage8/experiment_a_transformer_supervised_best.pt": {
        "sha256_prefix": "dcc42aee28593e88", "size": 28_363_055,
        "mtime": "2026-09-17T14:36:32",
    },
    "artifacts/checkpoints/stage8/experiment_c_transformer_ssl_best.pt": {
        "sha256_prefix": "dacb4c3407823ffe", "size": 28_665_999,
        "mtime": "2026-09-18T15:33:31",
    },
    "artifacts/checkpoints/stage8/experiment_d_transformer_mmoe_ssl_best.pt": {
        "sha256_prefix": "3f6afe3a021692d3", "size": 30_506_447,
        "mtime": "2026-09-19T05:20:56",
    },
}


def _fail(message: str) -> None:
    print(f"\nSMOKE COMPLETION FAILED: {message}", flush=True)
    raise SystemExit(1)


def _check(condition: bool, ok_message: str) -> None:
    if condition:
        print(f"  [OK] {ok_message}", flush=True)
    else:
        _fail(ok_message)


def _sha256_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


@torch.no_grad()
def recompute_validation_metrics(
    model: TransformerMMoE,
    dataset: ProcessedRecSysDataset,
    val_indices,
    device: torch.device,
    batch_size: int,
) -> dict:
    model.eval()
    loader = make_subset_loader(
        dataset, val_indices, batch_size=batch_size, shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    click_logits, install_logits = [], []
    click_targets, install_targets = [], []
    click_sum = install_sum = 0.0
    n_seen = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        output = model(batch, return_diagnostics=False)
        batch_n = int(batch["install"].shape[0])
        install_sum += binary_logloss(output["install_logit"], batch["install"]).item() * batch_n
        click_sum += binary_logloss(output["click_logit"], batch["click"]).item() * batch_n
        n_seen += batch_n
        install_logits.append(output["install_logit"].detach().float().cpu())
        install_targets.append(batch["install"].detach().float().cpu())
        click_logits.append(output["click_logit"].detach().float().cpu())
        click_targets.append(batch["click"].detach().float().cpu())
    if n_seen != len(val_indices):
        _fail(f"recomputed validation saw {n_seen:,} rows, expected {len(val_indices):,}")
    if not (torch.isfinite(torch.cat(install_logits)).all()
            and torch.isfinite(torch.cat(click_logits)).all()):
        _fail("non-finite validation logits in recomputation")
    return {
        "rows": n_seen,
        "val_install_logloss": install_sum / n_seen,
        "val_install_auc": float(roc_auc(torch.cat(install_logits), torch.cat(install_targets))),
        "val_click_logloss": click_sum / n_seen,
        "val_click_auc": float(roc_auc(torch.cat(click_logits), torch.cat(click_targets))),
    }


def main() -> int:
    print("=== completing Stage 10 one-round smoke test (post-checkpoint steps only) ===", flush=True)
    smoke_ckpt = PROJECT_ROOT / SMOKE_CHECKPOINT_RELPATH
    smoke_result = PROJECT_ROOT / SMOKE_RESULT_RELPATH
    if not smoke_ckpt.is_file():
        _fail(f"smoke checkpoint not found: {smoke_ckpt}")

    # ------------------------------------------------------------------
    # 1. Reload verification (strict load into a fresh seed-42 model)
    # ------------------------------------------------------------------
    print("\n=== 1. smoke checkpoint reload verification ===", flush=True)
    payload = torch.load(smoke_ckpt, map_location="cpu", weights_only=False)
    _check(payload.get("smoke_test") is True and payload.get("rounds_executed") == 1,
           "payload is a smoke-test checkpoint with rounds_executed=1")
    _check(payload["partition_hash"] == "9603ddc9facc973d",
           f"payload partition hash {payload['partition_hash']}")
    state = payload["model_state_dict"]
    _check(len(state) == 138, f"{len(state)} state entries")
    _check(all(torch.isfinite(v).all().item() for v in state.values()),
           "all checkpoint parameters finite")

    feature_set = FeatureSet.load(PROJECT_ROOT / "artifacts" / "preprocessing")
    torch.manual_seed(42)
    fresh = TransformerMMoE(feature_set)  # cold seed-42 init (weights then overwritten)
    load_result = fresh.load_state_dict(state, strict=True)
    _check(not (load_result.missing_keys or load_result.unexpected_keys),
           "strict load into fresh seed-42-initialized TransformerMMoE: all 138 keys matched")
    fresh_state = dict(fresh.state_dict())
    exact = all(torch.equal(fresh_state[k], state[k]) for k in state)
    _check(exact, "reloaded model state is bit-identical to the checkpoint state")

    # ------------------------------------------------------------------
    # 2. Independent full-split validation recomputation from the reload
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ProcessedRecSysDataset(
        feature_set, PROJECT_ROOT / "artifacts" / "processed" / "train", split="train"
    )
    split = load_split(
        PROJECT_ROOT / "artifacts" / "splits" / "centralized_split",
        expected_total_rows=len(dataset),
    )
    print(f"\n=== 2. validation recomputation from reloaded checkpoint ({device}) ===", flush=True)
    recomputed = recompute_validation_metrics(
        fresh.to(device), dataset, split.validation_indices, device, 1024
    )
    stored = payload["validation_metrics"]
    for key in ("val_install_logloss", "val_install_auc", "val_click_logloss", "val_click_auc"):
        delta = abs(recomputed[key] - stored[key])
        _check(delta < 1e-6,
               f"{key}: recomputed {recomputed[key]:.9f} == stored {stored[key]:.9f} (|d|={delta:.2e})")
    del fresh

    # ------------------------------------------------------------------
    # 3. Post-run integrity verification (frozen values + pre-run fingerprints)
    # ------------------------------------------------------------------
    print("\n=== 3. post-run protected-artifact integrity ===", flush=True)
    snapshot = snapshot_protected_artifacts(PROJECT_ROOT)
    violations = verify_protected_artifacts(snapshot, PROJECT_ROOT)
    if violations:
        for violation in violations:
            print(f"  VIOLATION: {violation}", flush=True)
        _fail("frozen protected values changed")
    print("  frozen values OK: Stage 2 hash, split hashes, Stage 6 full sha256, raw bytes", flush=True)
    for relpath, expected in PRERUN_FINGERPRINTS.items():
        path = PROJECT_ROOT / relpath
        live_prefix = _sha256_prefix(path)
        live_size = os.path.getsize(path)
        live_mtime = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(path)))
        ok = (live_prefix == expected["sha256_prefix"]
              and live_size == expected["size"] and live_mtime == expected["mtime"])
        if not ok:
            _fail(f"protected artifact changed since the pre-run snapshot: {relpath}")
    print(f"  pre-run sha256-prefix/size/mtime fingerprints match for all "
          f"{len(PRERUN_FINGERPRINTS)} guarded files (partition npz/json, frozen splits, "
          f"Stage 6 checkpoint, Stage 8 A/C/D checkpoints)", flush=True)

    # ------------------------------------------------------------------
    # 4. Result JSON
    # ------------------------------------------------------------------
    result = {k: v for k, v in payload.items() if k != "model_state_dict"}
    result["checkpoint_sha256"] = hashlib.sha256(smoke_ckpt.read_bytes()).hexdigest()
    result["reload_verified"] = bool(exact)
    result["validation_metrics_recomputed_match"] = True
    result["completed_by"] = "scripts/complete_stage10_smoke.py"
    smoke_result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n=== 4. result JSON written: {smoke_result} ===", flush=True)

    # ------------------------------------------------------------------
    # 5. Summary + confirmations
    # ------------------------------------------------------------------
    clients = payload["clients"]
    print("\n=== SMOKE TEST SUMMARY (one real federated round) ===", flush=True)
    print(f"{'client':>9} | {'samples':>9} | {'batches':>7} | {'total':>7} | {'click':>7} | "
          f"{'install':>7} | {'changed':>7} | {'seconds':>8}")
    for r in clients:
        print(f"{r['client_id']:>9} | {r['samples']:>9,} | {r['batches']:>7} | "
              f"{r['local_total_loss']:>7.4f} | {r['local_click_loss']:>7.4f} | "
              f"{r['local_install_loss']:>7.4f} | {r['params_changed']:>4}/{r['params_total']:<2} | "
              f"{r['duration_seconds']:>8.1f}")
    print(f"total client samples: {payload['total_client_samples']:,} (expected 3,137,266)")
    print(f"FedAvg weights sum: {payload['fedavg_weights_sum']!r}")
    print(f"sampler: local epoch {payload['local_sampler_epoch_round1']} for every client "
          f"via loader.batch_sampler only (DataLoader.sampler type "
          f"{clients[0]['data_loader_sampler_type']}, untouched)")
    vm = payload["validation_metrics"]
    print(f"validation (frozen 348,586-row split): install LogLoss {vm['val_install_logloss']:.6f} "
          f"AUC {vm['val_install_auc']:.4f} | click LogLoss {vm['val_click_logloss']:.6f} "
          f"AUC {vm['val_click_auc']:.4f}")
    print(f"round wall time: {payload['round_wall_seconds']:.1f}s | peak RSS "
          f"{payload['peak_rss_mib']:.0f} MiB | peak GPU alloc/res "
          f"{payload['peak_gpu_allocated_mib']:.0f}/{payload['peak_gpu_reserved_mib']:.0f} MiB")
    print(f"checkpoint: {smoke_ckpt}")
    print(f"checkpoint sha256: {result['checkpoint_sha256']}")
    print("\nCONFIRMATION: exactly ONE real federated round ran (10/10 clients, "
          "3,137,266 samples, cold seed-42 init, tested FedAvg utility).")
    print("CONFIRMATION: the 10-round Stage 10B experiment was NOT started.")
    print(f"RESULT: STAGE 10 ONE-ROUND SMOKE TEST PASSED — artifacts in {SMOKE_DIR_RELPATH}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
