"""Stage 10 (SMOKE): exactly ONE real federated round over the frozen Stage 9 clients.

This script validates the complete Stage 10 integration path end-to-end on
real data. It is a SMOKE TEST, not the Stage 10B experiment:

* exactly ONE FedAvg round (the frozen protocol has 10; none of them run here),
* cold seed-42 initialization of the global TransformerMMoE (the Stage 6
  checkpoint is NOT loaded; it is only hashed for integrity),
* per client: fresh model <- exact global state, fresh AdamW, ONE local epoch
  through the existing Stage 9 client-loader factory (batch 1024, sampler
  epoch exactly 0 via the frozen ``seed + epoch`` convention; ``set_epoch`` is
  only ever applied to ``loader.batch_sampler``, never ``loader.sampler``),
* aggregation by the already-tested ``fedavg_state_dicts`` utility
  (sample-weighted, client order client_00..client_09),
* global validation on the frozen 348,586-row validation split only,
* outputs written ONLY under ``artifacts/federated/smoke/``.

The official test set is never loaded. Protected Stage 1-9 artifacts are
snapshotted before and re-verified (sha256 + mtime) after the run.

Usage:
    python scripts/run_stage10_smoke.py [--device auto|cpu] [--preflight_only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

from recsys23_fedrec.data import FeatureSet, ProcessedRecSysDataset  # noqa: E402
from recsys23_fedrec.experiments.integrity import (  # noqa: E402
    PROTECTED_VALUES,
    snapshot_protected_artifacts,
    verify_protected_artifacts,
)
from recsys23_fedrec.federated.client_loader import (  # noqa: E402
    client_sampler,
    load_partition_positions,
    make_client_loader,
)
from recsys23_fedrec.federated.config import get_federated_config  # noqa: E402
from recsys23_fedrec.federated.fedavg import (  # noqa: E402
    fedavg_state_dicts,
    fedavg_weights_from_counts,
)
from recsys23_fedrec.federated.partition import CLIENT_IDS, partition_hash  # noqa: E402
from recsys23_fedrec.models import TransformerMMoE  # noqa: E402
from recsys23_fedrec.models.mmoe import mmoe_loss  # noqa: E402
from recsys23_fedrec.training.loaders import make_subset_loader  # noqa: E402
from recsys23_fedrec.training.metrics import binary_logloss, roc_auc  # noqa: E402
from recsys23_fedrec.training.split import load_split  # noqa: E402

EXPECTED_PARTITION_HASH = "9603ddc9facc973d"
EXPECTED_TOTAL_TRAIN_ROWS = 3_137_266
EXPECTED_VAL_ROWS = 348_586
EXPECTED_DATASET_ROWS = 3_485_852
EXPECTED_PARAMETER_COUNT = 2_502_930
SPLIT_VALIDATION_HASH = "3cd8370ebdecb305"
SPLIT_TRAIN_HASH = "dd37605169615442"
STAGE2_ARTIFACT_HASH = "c3a62caf13326617"

NPZ_RELPATH = "artifacts/federated/partition_alpha_0_5_seed42.npz"
MANIFEST_RELPATH = "artifacts/federated/partition_alpha_0_5_seed42.json"
SMOKE_DIR_RELPATH = "artifacts/federated/smoke"
SMOKE_CHECKPOINT_RELPATH = f"{SMOKE_DIR_RELPATH}/stage10_smoke_round1.pt"
SMOKE_RESULT_RELPATH = f"{SMOKE_DIR_RELPATH}/stage10_smoke_round1_result.json"

#: Audited per-client sizes from the frozen Stage 9 partition (Stage 10A §7).
AUDITED_CLIENT_SIZES = {
    "client_00": 621_801,
    "client_01": 544_574,
    "client_02": 59_999,
    "client_03": 488_090,
    "client_04": 69_790,
    "client_05": 89_988,
    "client_06": 193_852,
    "client_07": 201_902,
    "client_08": 511_603,
    "client_09": 355_667,
}

EXTRA_GUARDED_FILES = (
    NPZ_RELPATH,
    MANIFEST_RELPATH,
    "artifacts/splits/centralized_split.npz",
    "artifacts/splits/centralized_split.json",
    "artifacts/checkpoints/centralized_transformer_mmoe_best.pt",
    "artifacts/checkpoints/stage8/experiment_a_transformer_supervised_best.pt",
    "artifacts/checkpoints/stage8/experiment_c_transformer_ssl_best.pt",
    "artifacts/checkpoints/stage8/experiment_d_transformer_mmoe_ssl_best.pt",
)


def _fail(message: str) -> None:
    print(f"\nSMOKE TEST FAILED: {message}", flush=True)
    raise SystemExit(1)


def _check(condition: bool, ok_message: str, fail_message: str | None = None) -> None:
    if condition:
        print(f"  [OK] {ok_message}", flush=True)
    else:
        _fail(fail_message or ok_message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(project_root: Path) -> dict:
    """sha256 + mtime of every protected artifact guarded by this smoke test."""
    out = {}
    for relpath in EXTRA_GUARDED_FILES:
        path = project_root / relpath
        out[relpath] = {
            "sha256": _sha256_file(path),
            "mtime": os.path.getmtime(path),
            "size": os.path.getsize(path),
        }
    return out


def _verify_fingerprint(before: dict, project_root: Path) -> list[str]:
    violations = []
    after = _fingerprint(project_root)
    for relpath, expected in before.items():
        live = after.get(relpath)
        if live != expected:
            violations.append(f"protected artifact changed: {relpath} {expected} -> {live}")
    return violations


def _atomic_torch_save(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


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


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _cpu_state(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _count_nonfinite(state: dict) -> int:
    return sum(int((~torch.isfinite(v)).sum().item()) for v in state.values())


def _state_diff(left: dict, right: dict) -> dict:
    changed = [k for k in left if not torch.equal(left[k], right[k])]
    max_abs = max(float((left[k].float() - right[k].float()).abs().max()) for k in left)
    return {"entries": len(left), "changed": len(changed), "max_abs_diff": max_abs}


def cold_global_model(feature_set: FeatureSet, seed: int = 42) -> TransformerMMoE:
    """Fresh, deterministic, COLD seed-42 TransformerMMoE (no checkpoint load)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return TransformerMMoE(feature_set)


def train_one_client(
    client_id: str,
    client_positions: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    dataset: ProcessedRecSysDataset,
    global_state: dict,
    cfg,
    device: torch.device,
    log_every: int,
) -> tuple[dict, dict]:
    """ONE client-local training round: fresh model/optimizer, ONE local epoch.

    Returns ``(record, local_state_dict_on_cpu)``.
    """
    started = time.perf_counter()
    n_expected = int(client_positions.size)

    # Client DataLoader through the existing Stage 9 factory. Round 1 uses
    # sampler epoch 0 (the frozen seed + epoch convention: epoch e uses
    # seed + e). make_client_loader(epoch=0) leaves the PartShuffledBatchSampler
    # at epoch 0 by construction; set_epoch is ONLY ever applied to
    # loader.batch_sampler — never to loader.sampler.
    loader = make_client_loader(
        dataset,
        train_indices,
        client_positions,
        batch_size=cfg.local_batch_size,
        shuffle=True,
        seed=cfg.seed,
        epoch=0,  # round 1 -> sampler epoch exactly 0
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    sampler = client_sampler(loader)  # asserts the batch-sampler type
    if sampler.epoch != 0:
        _fail(f"{client_id}: sampler epoch {sampler.epoch} != 0 for round 1")
    sampler_kind = type(getattr(loader, "sampler", None)).__name__

    # Fresh local model <- exact global state (strict load, verified).
    local_model = cold_global_model(dataset.feature_set, seed=cfg.seed)
    load_result = local_model.load_state_dict(global_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        _fail(f"{client_id}: local strict state load reported {load_result}")
    for key in list(global_state)[:5]:
        if not torch.equal(local_model.state_dict()[key], global_state[key]):
            _fail(f"{client_id}: local model did not receive the exact global state ({key})")
    local_model = local_model.to(device)

    # Fresh local AdamW optimizer (per client per round; nothing persisted).
    optimizer = torch.optim.AdamW(
        local_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    local_model.train()
    totals = {"total": 0.0, "click": 0.0, "install": 0.0}
    seen = 0
    n_batches = 0
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad()
        output = local_model(batch, return_diagnostics=False)
        losses = mmoe_loss(
            output, batch, lambda_click=cfg.lambda_click, lambda_install=cfg.lambda_install
        )
        if not torch.isfinite(losses["total"]):
            _fail(f"{client_id}: non-finite local loss at batch {n_batches + 1}")
        losses["total"].backward()
        optimizer.step()

        batch_n = int(batch["install"].shape[0])
        totals["total"] += float(losses["total"].item()) * batch_n
        totals["click"] += float(losses["click"].item()) * batch_n
        totals["install"] += float(losses["install"].item()) * batch_n
        seen += batch_n
        n_batches += 1
        if log_every and n_batches % log_every == 0:
            print(
                f"    {client_id} batch {n_batches} | running loss "
                f"{totals['total'] / seen:.4f}",
                flush=True,
            )

    if seen != n_expected:
        _fail(f"{client_id}: local epoch saw {seen:,} rows, partition says {n_expected:,}")
    local_state = _cpu_state(local_model)
    if _count_nonfinite(local_state):
        _fail(f"{client_id}: non-finite local parameters after training")
    changed = [k for k in local_state if not torch.equal(local_state[k], global_state[k])]
    if not changed:
        _fail(f"{client_id}: local training changed no parameters")

    record = {
        "client_id": client_id,
        "samples": seen,
        "batches": n_batches,
        "local_total_loss": totals["total"] / seen,
        "local_click_loss": totals["click"] / seen,
        "local_install_loss": totals["install"] / seen,
        "params_changed": len(changed),
        "params_total": len(local_state),
        "sampler_epoch": int(sampler.epoch),
        "data_loader_sampler_type": sampler_kind,
        "duration_seconds": time.perf_counter() - started,
    }
    del optimizer, local_model, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record, local_state


@torch.no_grad()
def global_validation(model: TransformerMMoE, dataset, val_indices, device, batch_size: int):
    """Central evaluation on the frozen global validation split only."""
    model.eval()
    loader = make_subset_loader(
        dataset,
        val_indices,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    click_logits, install_logits = [], []
    click_targets, install_targets = [], []
    click_sum = install_sum = 0.0
    n_seen = 0
    for batch in loader:
        batch = _to_device(batch, device)
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
        _fail(f"validation saw {n_seen:,} rows, expected {len(val_indices):,}")
    if not (torch.isfinite(torch.cat(install_logits)).all() and torch.isfinite(torch.cat(click_logits)).all()):
        _fail("non-finite validation logits")
    return {
        "rows": n_seen,
        "val_install_logloss": install_sum / n_seen,
        "val_install_auc": float(roc_auc(torch.cat(install_logits), torch.cat(install_targets))),
        "val_click_logloss": click_sum / n_seen,
        "val_click_auc": float(roc_auc(torch.cat(click_logits), torch.cat(click_targets))),
    }


def preflight_integrity(project_root: Path) -> tuple[dict, dict]:
    print("=== pre-run integrity verification ===", flush=True)
    snapshot = snapshot_protected_artifacts(project_root)
    violations = verify_protected_artifacts(snapshot, project_root)
    if violations:
        _fail("protected artifacts do not match frozen values: " + "; ".join(violations))
    print(f"  Stage 2 artifact hash:     {snapshot['stage2_artifact_hash']} OK", flush=True)
    print(f"  split train hash:          {snapshot['split_train_index_hash']} OK", flush=True)
    print(f"  split validation hash:     {snapshot['split_validation_index_hash']} OK", flush=True)
    print(f"  Stage 6 checkpoint sha256: {snapshot['stage6_checkpoint_sha256'][:16]}... OK", flush=True)
    extra = _fingerprint(project_root)
    print(f"  extra guarded files sha256+mtime recorded: {len(extra)}", flush=True)
    return snapshot, extra


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="'auto' (CUDA if available) or 'cpu'")
    parser.add_argument("--log_every", type=int, default=200, help="client progress log interval (batches)")
    parser.add_argument("--allow_overwrite", action="store_true",
                        help="overwrite an existing smoke checkpoint/result")
    parser.add_argument("--preflight_only", action="store_true",
                        help="run integrity + data verification only, then stop")
    args = parser.parse_args()

    cfg = get_federated_config()
    smoke_ckpt = PROJECT_ROOT / SMOKE_CHECKPOINT_RELPATH
    smoke_result = PROJECT_ROOT / SMOKE_RESULT_RELPATH

    print("=== Stage 10 ONE-ROUND FEDERATED SMOKE TEST (real data) ===", flush=True)
    print(f"start: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", flush=True)
    print("scope: exactly ONE FedAvg round; NOT the 10-round Stage 10B experiment", flush=True)
    print(f"protocol: 10 clients, alpha {cfg.alpha}, seed {cfg.seed}, batch {cfg.local_batch_size}, "
          f"1 local epoch, AdamW lr {cfg.learning_rate} wd {cfg.weight_decay}, "
          f"lambdas ({cfg.lambda_click}, {cfg.lambda_install}), {cfg.aggregation}", flush=True)

    if smoke_ckpt.exists() and not args.allow_overwrite:
        print(f"\nSTOP: smoke checkpoint already exists: {smoke_ckpt}")
        print("Re-run with --allow_overwrite to discard it and start fresh.")
        return 1

    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

    # ------------------------------------------------------------------
    # 0. Integrity + environment
    # ------------------------------------------------------------------
    integrity_snapshot, fingerprint_before = preflight_integrity(PROJECT_ROOT)

    feature_set = FeatureSet.load(PROJECT_ROOT / "artifacts" / "preprocessing")
    if feature_set.artifact_hash != STAGE2_ARTIFACT_HASH:
        _fail(f"Stage 2 artifact hash {feature_set.artifact_hash} != {STAGE2_ARTIFACT_HASH}")
    dataset = ProcessedRecSysDataset(
        feature_set, PROJECT_ROOT / "artifacts" / "processed" / "train", split="train"
    )
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"  device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)
    if len(dataset) != EXPECTED_DATASET_ROWS:
        _fail(f"dataset rows {len(dataset):,} != expected {EXPECTED_DATASET_ROWS:,}")
    print(f"  dataset opens: {len(dataset):,} rows (train split only)", flush=True)
    print("  official test data: NOT loaded (never constructed in this script)", flush=True)

    # Frozen split (hash-verified by load_split).
    split = load_split(
        PROJECT_ROOT / "artifacts" / "splits" / "centralized_split",
        expected_total_rows=len(dataset),
    )
    if split.train_index_hash != SPLIT_TRAIN_HASH or split.validation_index_hash != SPLIT_VALIDATION_HASH:
        _fail("frozen split hashes do not match the expected values")
    if split.validation_rows != EXPECTED_VAL_ROWS:
        _fail(f"validation rows {split.validation_rows:,} != expected {EXPECTED_VAL_ROWS:,}")
    print(f"  split verified: {split.train_rows:,} train / {split.validation_rows:,} val", flush=True)

    # Frozen Stage 9 partition: stored hash, manifest hash, recomputed hash.
    manifest = json.loads((PROJECT_ROOT / MANIFEST_RELPATH).read_text(encoding="utf-8"))
    with np.load(PROJECT_ROOT / NPZ_RELPATH) as npz:
        stored_hash = str(npz["partition_hash"])
    positions = load_partition_positions(PROJECT_ROOT / NPZ_RELPATH)
    recomputed = partition_hash([positions[cid] for cid in CLIENT_IDS])
    _check(
        stored_hash == manifest["partition_hash"] == recomputed == EXPECTED_PARTITION_HASH,
        f"partition hash {recomputed} (stored == manifest == recomputed == expected)",
    )
    counts = {cid: int(pos.size) for cid, pos in positions.items()}
    if counts != AUDITED_CLIENT_SIZES:
        _fail(f"client sizes {counts} != audited {AUDITED_CLIENT_SIZES}")
    total_train = sum(counts.values())
    if total_train != EXPECTED_TOTAL_TRAIN_ROWS:
        _fail(f"total client rows {total_train:,} != expected {EXPECTED_TOTAL_TRAIN_ROWS:,}")
    print(f"  client sizes match the audited Stage 9 partition; total {total_train:,}", flush=True)

    # No client/validation overlap (composed dataset rows, not raw positions).
    composed_rows = {}
    max_overlap = 0
    for cid in CLIENT_IDS:
        rows = split.train_indices[positions[cid]]
        composed_rows[cid] = rows
        max_overlap = max(max_overlap, int(np.intersect1d(rows, split.validation_indices).size))
    if max_overlap:
        _fail(f"client rows overlap the frozen validation split ({max_overlap} rows)")
    print("  client/composed-row vs validation overlap: 0 rows (all 10 clients)", flush=True)

    if args.preflight_only:
        print("\nRESULT: PREFLIGHT ONLY - stopping before the smoke round as requested")
        return 0

    # ------------------------------------------------------------------
    # 1. Cold seed-42 global model + initial state capture
    # ------------------------------------------------------------------
    print("\n=== 1. cold seed-42 global initialization ===", flush=True)
    global_model = cold_global_model(feature_set, seed=cfg.seed).to(device)
    if sum(p.numel() for p in global_model.parameters()) != EXPECTED_PARAMETER_COUNT:
        _fail("global parameter count does not match the frozen architecture")
    initial_state = _cpu_state(global_model)
    if _count_nonfinite(initial_state):
        _fail("non-finite initial global parameters")

    # Determinism proof: an independently created seed-42 model is identical.
    probe = cold_global_model(feature_set, seed=cfg.seed)
    if any(not torch.equal(probe.state_dict()[k], initial_state[k]) for k in initial_state):
        _fail("seed-42 initialization is not deterministic across two creations")
    del probe

    # Cold-init proof: the initial state must NOT be the Stage 6 checkpoint.
    stage6 = torch.load(
        PROJECT_ROOT / "artifacts/checkpoints/centralized_transformer_mmoe_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    stage6_state = stage6["model_state_dict"]
    differing = sum(1 for k in initial_state if not torch.equal(initial_state[k], stage6_state[k]))
    print(f"  parameter count: {EXPECTED_PARAMETER_COUNT:,}; state entries: {len(initial_state)}", flush=True)
    print(f"  determinism: two seed-42 creations identical: True", flush=True)
    print(f"  cold init (vs Stage 6 checkpoint): {differing}/{len(initial_state)} entries differ "
          f"(Stage 6 weights NOT loaded)", flush=True)
    if differing != len(initial_state):
        _fail("initial global state coincides with the Stage 6 checkpoint")
    del stage6, stage6_state

    # ------------------------------------------------------------------
    # 2-3. One local epoch per client (fresh model/optimizer each)
    # ------------------------------------------------------------------
    print("\n=== 2. client-local training: exactly ONE local epoch per client ===", flush=True)
    monitor = RSSMonitor(interval=10.0)
    monitor.start()
    gpu_alloc = gpu_reserved = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    round_started = time.perf_counter()

    client_records: list[dict] = []
    local_states: dict[str, dict] = {}
    for cid in CLIENT_IDS:  # client order is significant for FedAvg reproducibility
        print(f"  {cid}: {counts[cid]:,} rows, {math.ceil(counts[cid] / cfg.local_batch_size)} batches",
              flush=True)
        record, state = train_one_client(
            cid,
            positions[cid],
            split.train_indices,
            split.validation_indices,
            dataset,
            initial_state,
            cfg,
            device,
            args.log_every,
        )
        client_records.append(record)
        local_states[cid] = state
        print(
            f"  {cid} done: loss {record['local_total_loss']:.4f} "
            f"(click {record['local_click_loss']:.4f}, install {record['local_install_loss']:.4f}), "
            f"params changed {record['params_changed']}/{record['params_total']}, "
            f"{record['duration_seconds']:.1f}s",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 4-6. Sample-weighted FedAvg via the tested utility + global update
    # ------------------------------------------------------------------
    print("\n=== 3. sample-weighted FedAvg (tested utility, client order) ===", flush=True)
    sample_counts = [counts[cid] for cid in CLIENT_IDS]
    weights = fedavg_weights_from_counts(sample_counts)
    weights_sum = float(sum(weights))
    print(f"  weights sum: {weights_sum!r} (tolerance 1e-9)", flush=True)
    if abs(weights_sum - 1.0) > 1e-9:
        _fail(f"FedAvg weights sum {weights_sum} != 1 within 1e-9")
    aggregated = fedavg_state_dicts(
        [local_states[cid] for cid in CLIENT_IDS], sample_counts
    )
    if _count_nonfinite(aggregated):
        _fail("non-finite entries in the aggregated state")
    agg_vs_initial = _state_diff(initial_state, aggregated)
    print(f"  aggregated vs initial: {agg_vs_initial['changed']}/{agg_vs_initial['entries']} "
          f"entries changed, max abs diff {agg_vs_initial['max_abs_diff']:.6e}", flush=True)
    if agg_vs_initial["changed"] == 0:
        _fail("FedAvg produced a state identical to the initial state")

    global_model.load_state_dict(aggregated, strict=True)
    current_state = _cpu_state(global_model)
    load_diff = _state_diff(aggregated, current_state)
    if load_diff["changed"] != 0 or load_diff["max_abs_diff"] != 0.0:
        _fail("aggregated state did not load exactly into the global model")
    print(f"  global model state changed vs initial: True (max abs diff "
          f"{agg_vs_initial['max_abs_diff']:.6e})", flush=True)

    # ------------------------------------------------------------------
    # 7-8. Global validation on the frozen split only
    # ------------------------------------------------------------------
    print("\n=== 4. global validation (frozen 348,586-row split only) ===", flush=True)
    val_metrics = global_validation(
        global_model, dataset, split.validation_indices, device, cfg.local_batch_size
    )
    for name, value in val_metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            _fail(f"non-finite validation metric {name}")
    print(f"  rows evaluated: {val_metrics['rows']:,}", flush=True)
    print(f"  Install LogLoss {val_metrics['val_install_logloss']:.6f} | "
          f"Install AUC {val_metrics['val_install_auc']:.4f}", flush=True)
    print(f"  Click   LogLoss {val_metrics['val_click_logloss']:.6f} | "
          f"Click   AUC {val_metrics['val_click_auc']:.4f}", flush=True)

    round_seconds = time.perf_counter() - round_started
    peak_rss = monitor.stop()
    if device.type == "cuda":
        torch.cuda.synchronize()
        gpu_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
        gpu_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    total_client_samples = sum(r["samples"] for r in client_records)
    print(f"\n  one-round wall time: {round_seconds:.1f}s | peak RSS {peak_rss:.0f} MiB"
          + (f" | peak GPU alloc/res {gpu_alloc:.0f}/{gpu_reserved:.0f} MiB" if gpu_alloc is not None else ""),
          flush=True)

    # ------------------------------------------------------------------
    # 9-11. Smoke checkpoint (temporary, separate path) + reload verification
    # ------------------------------------------------------------------
    print("\n=== 5. smoke checkpoint (temporary path only) + reload verification ===", flush=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = {
        "stage": "10-federated-smoke",
        "stage_number": 10,
        "smoke_test": True,
        "rounds_executed": 1,
        "protocol_rounds_full_experiment": cfg.rounds,
        "note": (
            "TEMPORARY SMOKE TEST ARTIFACT: validates the one-round integration "
            "path only. NOT the 10-round Stage 10B experiment and NOT a best-model "
            "checkpoint."
        ),
        "model_type": "transformer_mmoe",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "model_state_dict": {k: v.cpu() for k, v in global_model.state_dict().items()},
        "federated_config": cfg.to_dict(),
        "seed": cfg.seed,
        "partition_hash": EXPECTED_PARTITION_HASH,
        "partition_npz": NPZ_RELPATH,
        "split_train_index_hash": split.train_index_hash,
        "split_validation_index_hash": split.validation_index_hash,
        "preprocessing_hash": feature_set.artifact_hash,
        "local_sampler_epoch_round1": 0,
        "clients": client_records,
        "total_client_samples": total_client_samples,
        "fedavg_weights": weights,
        "fedavg_weights_sum": weights_sum,
        "validation_metrics": val_metrics,
        "round_wall_seconds": round_seconds,
        "peak_rss_mib": peak_rss,
        "peak_gpu_allocated_mib": gpu_alloc,
        "peak_gpu_reserved_mib": gpu_reserved,
        "device": str(device),
        "checkpoint_timestamp": now_iso,
    }
    if total_client_samples != EXPECTED_TOTAL_TRAIN_ROWS:
        _fail(f"total client samples {total_client_samples:,} != {EXPECTED_TOTAL_TRAIN_ROWS:,}")
    if len(client_records) != 10:
        _fail(f"{len(client_records)} clients participated, expected exactly 10")
    _atomic_torch_save(payload, smoke_ckpt)
    smoke_sha = _sha256_file(smoke_ckpt)
    print(f"  checkpoint written: {smoke_ckpt}", flush=True)
    print(f"  sha256: {smoke_sha}", flush=True)

    fresh = cold_global_model(feature_set, seed=cfg.seed)
    reloaded = torch.load(smoke_ckpt, map_location="cpu", weights_only=False)
    fresh.load_state_dict(reloaded["model_state_dict"], strict=True)
    fresh_state = dict(fresh.state_dict())
    exact = all(torch.equal(fresh_state[k], current_state[k]) for k in current_state)
    print(f"  reload: fresh model received a bit-identical state: {exact}", flush=True)
    if not exact:
        _fail("checkpoint reload is not bit-identical to the aggregated state")

    # One deterministic validation batch through the reloaded model.
    fresh = fresh.to(device).eval()
    val_probe_loader = make_subset_loader(
        dataset,
        split.validation_indices[: cfg.local_batch_size],
        batch_size=cfg.local_batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    batch = _to_device(next(iter(val_probe_loader)), device)
    global_model.eval()
    with torch.no_grad():
        out_saved = global_model(batch, return_diagnostics=False)
        out_fresh = fresh(batch, return_diagnostics=False)
    close = all(
        torch.allclose(out_fresh[key], out_saved[key], atol=1e-5)
        and torch.isfinite(out_fresh[key]).all()
        for key in ("click_logit", "install_logit")
    )
    print(f"  reload: one validation batch logits match (atol 1e-5): {close}", flush=True)
    if not close:
        _fail("reloaded model outputs differ from the global model")
    del fresh, reloaded, val_probe_loader

    # Result JSON (no tensors).
    result = {k: v for k, v in payload.items() if k != "model_state_dict"}
    result["checkpoint_sha256"] = smoke_sha
    result["reload_verified"] = bool(exact and close)
    smoke_result.parent.mkdir(parents=True, exist_ok=True)
    smoke_result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  result JSON: {smoke_result}", flush=True)

    # ------------------------------------------------------------------
    # 12-13. Post-run integrity verification
    # ------------------------------------------------------------------
    print("\n=== 6. post-run integrity verification ===", flush=True)
    violations = verify_protected_artifacts(integrity_snapshot, PROJECT_ROOT)
    violations += _verify_fingerprint(fingerprint_before, PROJECT_ROOT)
    if violations:
        for violation in violations:
            print(f"  VIOLATION: {violation}", flush=True)
        return 1
    print(f"  all {len(fingerprint_before)} guarded artifacts byte-identical "
          f"(sha256+mtime): partition npz/json, frozen splits, Stage 6 checkpoint, "
          f"Stage 8 A/C/D checkpoints", flush=True)
    print(f"  Stage 10A partition hash still {EXPECTED_PARTITION_HASH}", flush=True)

    print("\n=== SMOKE TEST SUMMARY ===", flush=True)
    print(f"{'client':>9} | {'samples':>9} | {'batches':>7} | {'total':>7} | {'click':>7} | "
          f"{'install':>7} | {'changed':>7} | {'seconds':>8}")
    for r in client_records:
        print(f"{r['client_id']:>9} | {r['samples']:>9,} | {r['batches']:>7} | "
              f"{r['local_total_loss']:>7.4f} | {r['local_click_loss']:>7.4f} | "
              f"{r['local_install_loss']:>7.4f} | {r['params_changed']:>4}/{r['params_total']:<2} | "
              f"{r['duration_seconds']:>8.1f}")
    print(f"total client samples: {total_client_samples:,} (expected {EXPECTED_TOTAL_TRAIN_ROWS:,})")
    print(f"FedAvg weights sum: {weights_sum!r}")
    print(f"validation: install LogLoss {val_metrics['val_install_logloss']:.6f} "
          f"AUC {val_metrics['val_install_auc']:.4f} | click LogLoss "
          f"{val_metrics['val_click_logloss']:.6f} AUC {val_metrics['val_click_auc']:.4f}")
    print(f"round wall time: {round_seconds:.1f}s | peak RSS {peak_rss:.0f} MiB"
          + (f" | peak GPU alloc/res {gpu_alloc:.0f}/{gpu_reserved:.0f} MiB" if gpu_alloc is not None else ""))
    print(f"sampler: local epoch 0 for every client via loader.batch_sampler only "
          f"(DataLoader.sampler untouched, type {client_records[0]['data_loader_sampler_type']})")
    print("\nCONFIRMATION: exactly ONE real federated round ran (10/10 clients).")
    print("CONFIRMATION: the 10-round Stage 10B experiment was NOT started.")
    print(f"RESULT: STAGE 10 ONE-ROUND SMOKE TEST PASSED — artifacts in {SMOKE_DIR_RELPATH}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
