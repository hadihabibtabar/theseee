"""Stage 10B: the full 10-round federated baseline experiment (real data).

The actual Stage 10 research run over the frozen Stage 9 partition. Protocol
is the FROZEN Stage 10A baseline (see
``reports/stage10_federated_protocol_report.md`` and
``FederatedConfig``) — nothing here may alter it:

* 10 simulated clients, Dirichlet alpha 0.5, seed 42, all clients every round,
* model: existing TransformerMMoE (2,502,930 params), COLD seed-42 init;
  the Stage 6 checkpoint and the Stage 10 smoke checkpoint are NEVER loaded,
* per client per round: loader via the Stage 9 client-loader factory with
  sampler epoch EXACTLY ``r - 1`` (set on ``loader.batch_sampler`` only —
  ``loader.sampler`` is never touched), fresh local model receiving the exact
  global state, fresh AdamW (lr 3e-4, wd 1e-2, no state carried between
  rounds), exactly ONE local epoch, BCE(click)+BCE(install), lambdas 1.0/1.0,
  batch 1024, no SSL / class weighting / focal / oversampling / scheduler /
  HPO,
* aggregation: the already-tested ``fedavg_state_dicts`` (sample-weighted,
  client order client_00..client_09),
* validation: the frozen 348,586-row global validation split only; primary
  metric global validation Install LogLoss; best round checkpointed,
* official test set is never loaded or evaluated,
* client rows are always ``frozen_train_indices[client_positions]`` —
  client positions are NEVER treated as global dataset positions.

Outputs (dedicated directory, separate from the smoke artifacts):
    artifacts/federated/stage10/
        stage10b_best.pt            best global checkpoint (by Install LogLoss)
        stage10b_final_round10.pt   final round-10 checkpoint
        stage10b_latest.pt          per-round resume state (crash safety)
        stage10b_result.json        protocol snapshot + per-round metrics

Checkpoint payloads contain model_state_dict, round, best validation Install
LogLoss, the full training configuration, seed, partition hash, split hashes,
and the stage/protocol identifier — sufficient for exact reload.

Protected Stage 1-9 artifacts are verified before the run and re-verified
after completion. A crashed run can be continued deterministically with
``--resume`` (the post-round N global state + history fully determine rounds
N+1..10; sampler arrangements come from the frozen ``seed + r - 1``
convention, so no other RNG state matters).

Usage:
    python scripts/run_stage10b.py [--device auto|cpu] [--resume]
        [--preflight_only] [--allow_restart]
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

STAGE10B_DIR_RELPATH = "artifacts/federated/stage10"
BEST_CKPT_RELPATH = f"{STAGE10B_DIR_RELPATH}/stage10b_best.pt"
FINAL_CKPT_RELPATH = f"{STAGE10B_DIR_RELPATH}/stage10b_final_round10.pt"
LATEST_CKPT_RELPATH = f"{STAGE10B_DIR_RELPATH}/stage10b_latest.pt"
RESULT_JSON_RELPATH = f"{STAGE10B_DIR_RELPATH}/stage10b_result.json"
PROTOCOL_ID = "stage10b-federated-baseline"

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

PROTECTED_FILES = (
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
    print(f"\nSTAGE 10B FAILED: {message}", flush=True)
    raise SystemExit(1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(project_root: Path) -> dict:
    out = {}
    for relpath in PROTECTED_FILES:
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
        if after.get(relpath) != expected:
            violations.append(f"protected artifact changed: {relpath}")
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


def _atomic_json_write(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


class RSSMonitor(threading.Thread):
    """Sample this process's RSS until stopped; track the peak."""

    def __init__(self, interval: float = 10.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()
        self.process = psutil.Process()
        self.peak_mib = self.process.memory_info().rss / (1024 * 1024)

    def reset(self) -> None:
        """Restart the peak window from the current RSS (per-round peaks)."""
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


def cold_model(feature_set: FeatureSet, seed: int = 42) -> TransformerMMoE:
    """Fresh, deterministic seed-42 TransformerMMoE (cold init; no loads)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return TransformerMMoE(feature_set)


def train_one_client(
    client_id: str,
    client_positions: np.ndarray,
    train_indices: np.ndarray,
    dataset: ProcessedRecSysDataset,
    global_state: dict,
    cfg,
    device: torch.device,
    round_number: int,
    log_every: int,
) -> tuple[dict, dict]:
    """ONE client-local training round: fresh model/optimizer, ONE local epoch.

    Sampler epoch is EXACTLY ``round_number - 1`` (frozen ``seed + epoch``
    convention), applied to ``loader.batch_sampler`` only. Returns
    ``(record, local_state_dict_on_cpu)``.
    """
    started = time.perf_counter()
    n_expected = int(client_positions.size)

    loader = make_client_loader(
        dataset,
        train_indices,
        client_positions,
        batch_size=cfg.local_batch_size,
        shuffle=True,
        seed=cfg.seed,
        epoch=round_number - 1,  # round r -> sampler epoch exactly r-1
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    sampler = client_sampler(loader)  # asserts PartShuffledBatchSampler
    if sampler.epoch != round_number - 1:
        _fail(f"{client_id}: sampler epoch {sampler.epoch} != {round_number - 1} (round {round_number})")
    sampler_kind = type(getattr(loader, "sampler", None)).__name__

    local_model = cold_model(dataset.feature_set, seed=cfg.seed)
    load_result = local_model.load_state_dict(global_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        _fail(f"{client_id}: local strict state load reported {load_result}")
    local_model = local_model.to(device)

    # Fresh local AdamW per client per round; nothing persists across rounds.
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
            _fail(f"{client_id}: non-finite local loss at round {round_number} batch {n_batches + 1}")
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
                f"    r{round_number} {client_id} batch {n_batches} | running loss "
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
    if not (torch.isfinite(torch.cat(install_logits)).all()
            and torch.isfinite(torch.cat(click_logits)).all()):
        _fail("non-finite validation logits")
    return {
        "rows": n_seen,
        "val_install_logloss": install_sum / n_seen,
        "val_install_auc": float(roc_auc(torch.cat(install_logits), torch.cat(install_targets))),
        "val_click_logloss": click_sum / n_seen,
        "val_click_auc": float(roc_auc(torch.cat(click_logits), torch.cat(click_targets))),
    }


def checkpoint_payload(
    *,
    state: dict,
    round_number: int,
    best_logloss: float,
    best_round: int | None,
    history: list[dict],
    cfg,
    split,
    feature_set: FeatureSet,
    kind: str,
) -> dict:
    """Checkpoint payload with everything needed for exact reload/verification."""
    return {
        "stage": PROTOCOL_ID,
        "stage_number": 10,
        "protocol": cfg.to_dict(),
        "kind": kind,
        "round": int(round_number),
        "protocol_rounds": cfg.rounds,
        "best_validation_install_logloss": float(best_logloss),
        "best_round": best_round,
        "model_type": "transformer_mmoe",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "model_state_dict": {k: v.cpu() for k, v in state.items()},
        "history": history,
        "seed": cfg.seed,
        "partition_hash": EXPECTED_PARTITION_HASH,
        "partition_npz": NPZ_RELPATH,
        "split_train_index_hash": split.train_index_hash,
        "split_validation_index_hash": split.validation_index_hash,
        "preprocessing_hash": feature_set.artifact_hash,
        "sampler_protocol": (
            "stage9/10A frozen: make_client_loader applies set_epoch(r-1) to "
            "loader.batch_sampler (PartShuffledBatchSampler) only; "
            "loader.sampler is never touched; round r local sampler epoch = r - 1"
        ),
        "checkpoint_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def preflight(project_root: Path) -> tuple[dict, dict]:
    print("=== pre-run integrity verification ===", flush=True)
    snapshot = snapshot_protected_artifacts(project_root)
    violations = verify_protected_artifacts(snapshot, project_root)
    if violations:
        _fail("protected artifacts do not match frozen values: " + "; ".join(violations))
    print(f"  Stage 2 artifact hash:     {snapshot['stage2_artifact_hash']} OK", flush=True)
    print(f"  split train hash:          {snapshot['split_train_index_hash']} OK", flush=True)
    print(f"  split validation hash:     {snapshot['split_validation_index_hash']} OK", flush=True)
    print(f"  Stage 6 checkpoint sha256: {snapshot['stage6_checkpoint_sha256'][:16]}... OK", flush=True)
    fingerprint = _fingerprint(project_root)
    print(f"  protected files sha256+size+mtime recorded: {len(fingerprint)}", flush=True)
    return snapshot, fingerprint


def verify_data_layer(dataset, split, project_root: Path, cfg) -> tuple[dict, np.ndarray]:
    """Dataset/split/partition verification shared by preflight and the run."""
    if feature_set_hash(project_root) != STAGE2_ARTIFACT_HASH:
        _fail("Stage 2 artifact hash mismatch")
    if len(dataset) != EXPECTED_DATASET_ROWS:
        _fail(f"dataset rows {len(dataset):,} != expected {EXPECTED_DATASET_ROWS:,}")
    if split.train_index_hash != SPLIT_TRAIN_HASH or split.validation_index_hash != SPLIT_VALIDATION_HASH:
        _fail("frozen split hashes do not match the expected values")
    if split.validation_rows != EXPECTED_VAL_ROWS:
        _fail(f"validation rows {split.validation_rows:,} != expected {EXPECTED_VAL_ROWS:,}")

    manifest = json.loads((project_root / MANIFEST_RELPATH).read_text(encoding="utf-8"))
    with np.load(project_root / NPZ_RELPATH) as npz:
        stored_hash = str(npz["partition_hash"])
    positions = load_partition_positions(project_root / NPZ_RELPATH)
    recomputed = partition_hash([positions[cid] for cid in CLIENT_IDS])
    if not (stored_hash == manifest["partition_hash"] == recomputed == EXPECTED_PARTITION_HASH):
        _fail(f"partition hash mismatch: stored {stored_hash} manifest "
              f"{manifest['partition_hash']} recomputed {recomputed}")
    print(f"  [OK] partition hash {recomputed} (stored == manifest == recomputed)", flush=True)

    counts = {cid: int(pos.size) for cid, pos in positions.items()}
    if counts != AUDITED_CLIENT_SIZES:
        _fail(f"client sizes {counts} != audited {AUDITED_CLIENT_SIZES}")
    if sum(counts.values()) != EXPECTED_TOTAL_TRAIN_ROWS:
        _fail(f"total client rows {sum(counts.values()):,} != {EXPECTED_TOTAL_TRAIN_ROWS:,}")
    print(f"  [OK] 10 client sizes match the audited partition; total {sum(counts.values()):,}", flush=True)

    max_overlap = 0
    for cid in CLIENT_IDS:
        rows = split.train_indices[positions[cid]]
        max_overlap = max(
            max_overlap, int(np.intersect1d(rows, split.validation_indices).size)
        )
    if max_overlap:
        _fail(f"client rows overlap the frozen validation split ({max_overlap} rows)")
    print("  [OK] composed client rows vs frozen validation overlap: 0 (all 10 clients)", flush=True)
    print("  official test data: NOT loaded (never constructed in this script)", flush=True)
    return counts, positions


def feature_set_hash(project_root: Path) -> str:
    from recsys23_fedrec.data import FeatureSet as FS

    return FS.load(project_root / "artifacts" / "preprocessing").artifact_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="'auto' (CUDA if available) or 'cpu'")
    parser.add_argument("--log_every", type=int, default=200, help="client progress log interval (batches)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from stage10b_latest.pt after an interrupted run")
    parser.add_argument("--allow_restart", action="store_true",
                        help="discard any existing Stage 10B outputs and start fresh")
    parser.add_argument("--preflight_only", action="store_true",
                        help="run integrity + data verification only, then stop")
    args = parser.parse_args()

    cfg = get_federated_config()
    stage_dir = PROJECT_ROOT / STAGE10B_DIR_RELPATH
    best_ckpt = PROJECT_ROOT / BEST_CKPT_RELPATH
    final_ckpt = PROJECT_ROOT / FINAL_CKPT_RELPATH
    latest_ckpt = PROJECT_ROOT / LATEST_CKPT_RELPATH
    result_json = PROJECT_ROOT / RESULT_JSON_RELPATH

    print("=== STAGE 10B: 10-ROUND FEDERATED BASELINE (real data) ===", flush=True)
    print(f"start: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", flush=True)
    print(f"protocol: 10 clients, alpha {cfg.alpha}, seed {cfg.seed}, {cfg.rounds} rounds, "
          f"1 local epoch, batch {cfg.local_batch_size}, AdamW lr {cfg.learning_rate} "
          f"wd {cfg.weight_decay}, lambdas ({cfg.lambda_click}, {cfg.lambda_install}), "
          f"{cfg.aggregation}, SSL off, cold seed-42 init", flush=True)

    if final_ckpt.exists() or best_ckpt.exists():
        if args.resume and latest_ckpt.exists():
            pass  # resume path below
        elif args.allow_restart:
            for path in (best_ckpt, final_ckpt, latest_ckpt, result_json):
                path.unlink(missing_ok=True)
            print("  existing Stage 10B outputs discarded (--allow_restart)", flush=True)
        else:
            print(f"\nSTOP: Stage 10B outputs already exist under {stage_dir}")
            print("Use --resume to continue an interrupted run, or --allow_restart to discard them.")
            return 1

    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

    # ------------------------------------------------------------------
    # 0. Integrity + environment + data layer
    # ------------------------------------------------------------------
    integrity_snapshot, fingerprint_before = preflight(PROJECT_ROOT)

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
    split = load_split(
        PROJECT_ROOT / "artifacts" / "splits" / "centralized_split",
        expected_total_rows=len(dataset),
    )
    print(f"  dataset opens: {len(dataset):,} rows; split {split.train_rows:,} train / "
          f"{split.validation_rows:,} val", flush=True)
    counts, positions = verify_data_layer(dataset, split, PROJECT_ROOT, cfg)

    if args.preflight_only:
        print("\nRESULT: PREFLIGHT ONLY - stopping before the Stage 10B run as requested")
        return 0

    # ------------------------------------------------------------------
    # 1. Global model: cold seed-42 init (or deterministic resume)
    # ------------------------------------------------------------------
    started_all = time.perf_counter()
    monitor = RSSMonitor(interval=10.0)
    monitor.start()

    history: list[dict] = []
    best_round: int | None = None
    best_val_install_logloss = float("inf")
    start_round = 1

    if args.resume:
        if not latest_ckpt.exists():
            _fail("--resume requested but no resume state found: " + str(latest_ckpt))
        resume_payload = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        if resume_payload.get("stage") != PROTOCOL_ID:
            _fail("resume state is not a Stage 10B payload")
        start_round = int(resume_payload["round"]) + 1
        if start_round > cfg.rounds:
            _fail("resume state says all rounds are already complete")
        global_model = cold_model(feature_set, seed=cfg.seed).to(device)
        global_model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        history = list(resume_payload["history"])
        best_round = resume_payload["best_round"]
        best_val_install_logloss = float(resume_payload["best_validation_install_logloss"])
        print(f"\n=== RESUME: continuing from round {start_round} "
              f"(best so far: round {best_round}, Install LogLoss "
              f"{best_val_install_logloss:.6f}) ===", flush=True)
    else:
        print("\n=== 1. cold seed-42 global initialization (no checkpoint loads) ===", flush=True)
        global_model = cold_model(feature_set, seed=cfg.seed).to(device)
        if sum(p.numel() for p in global_model.parameters()) != EXPECTED_PARAMETER_COUNT:
            _fail("global parameter count does not match the frozen architecture")
        initial_probe = cold_model(feature_set, seed=cfg.seed)
        global_sd_cpu = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        if any(not torch.equal(initial_probe.state_dict()[k], global_sd_cpu[k])
               for k in global_sd_cpu):
            _fail("seed-42 initialization is not deterministic across two creations")
        del initial_probe
        stage6 = torch.load(
            PROJECT_ROOT / "artifacts/checkpoints/centralized_transformer_mmoe_best.pt",
            map_location="cpu",
            weights_only=False,
        )
        differing = sum(
            1 for k in global_sd_cpu
            if not torch.equal(global_sd_cpu[k], stage6["model_state_dict"][k])
        )
        if differing != len(global_model.state_dict()):
            _fail("initial global state coincides with the Stage 6 checkpoint")
        del stage6
        smoke_state = torch.load(
            PROJECT_ROOT / "artifacts/federated/smoke/stage10_smoke_round1.pt",
            map_location="cpu",
            weights_only=False,
        )
        differing_smoke = sum(
            1 for k in global_sd_cpu
            if not torch.equal(global_sd_cpu[k], smoke_state["model_state_dict"][k])
        )
        if differing_smoke != len(global_model.state_dict()):
            _fail("initial global state coincides with the Stage 10 smoke checkpoint")
        del smoke_state
        print(f"  parameter count {EXPECTED_PARAMETER_COUNT:,}; state entries "
              f"{len(global_model.state_dict())}", flush=True)
        print("  determinism: two seed-42 creations identical: True", flush=True)
        print(f"  cold init: all {differing} entries differ from BOTH the Stage 6 "
              f"checkpoint and the smoke checkpoint (neither was loaded)", flush=True)
        if _count_nonfinite({k: v.cpu() for k, v in global_model.state_dict().items()}):
            _fail("non-finite initial global parameters")

    # ------------------------------------------------------------------
    # 2. Round loop r = 1..10
    # ------------------------------------------------------------------
    for round_number in range(start_round, cfg.rounds + 1):
        round_started = time.perf_counter()
        monitor.reset()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        current_state = _cpu_state(global_model)
        if _count_nonfinite(current_state):
            _fail(f"non-finite global parameters at the start of round {round_number}")

        print(f"\n=== ROUND {round_number}/{cfg.rounds} | sampler epoch {round_number - 1} "
              f"per client ===", flush=True)
        client_records: list[dict] = []
        local_states: dict[str, dict] = {}
        for cid in CLIENT_IDS:  # client order is significant for FedAvg reproducibility
            record, state = train_one_client(
                cid,
                positions[cid],
                split.train_indices,
                dataset,
                current_state,
                cfg,
                device,
                round_number,
                args.log_every,
            )
            client_records.append(record)
            local_states[cid] = state
            print(
                f"  r{round_number} {cid} done: {record['samples']:,} rows | loss "
                f"{record['local_total_loss']:.4f} (click {record['local_click_loss']:.4f}, "
                f"install {record['local_install_loss']:.4f}) | {record['duration_seconds']:.1f}s",
                flush=True,
            )

        # ---- per-round participation + integrity checks ----------------
        if len(client_records) != 10:
            _fail(f"round {round_number}: {len(client_records)} clients participated, expected 10")
        total_samples = sum(r["samples"] for r in client_records)
        if total_samples != EXPECTED_TOTAL_TRAIN_ROWS:
            _fail(f"round {round_number}: total client samples {total_samples:,} "
                  f"!= {EXPECTED_TOTAL_TRAIN_ROWS:,}")
        for record in client_records:
            if record["sampler_epoch"] != round_number - 1:
                _fail(f"round {round_number}: {record['client_id']} used sampler epoch "
                      f"{record['sampler_epoch']} != {round_number - 1}")

        # ---- sample-weighted FedAvg via the tested utility -------------
        sample_counts = [counts[cid] for cid in CLIENT_IDS]
        weights = fedavg_weights_from_counts(sample_counts)
        weights_sum = float(sum(weights))
        if abs(weights_sum - 1.0) > 1e-9:
            _fail(f"round {round_number}: FedAvg weights sum {weights_sum} != 1")
        aggregated = fedavg_state_dicts(
            [local_states[cid] for cid in CLIENT_IDS], sample_counts
        )
        local_states.clear()
        if _count_nonfinite(aggregated):
            _fail(f"round {round_number}: non-finite aggregated state")

        global_model.load_state_dict(aggregated, strict=True)
        if _count_nonfinite({k: v.detach().cpu() for k, v in global_model.state_dict().items()}):
            _fail(f"round {round_number}: non-finite global parameters after aggregation")

        # ---- global validation (frozen split only) ----------------------
        val_metrics = global_validation(
            global_model, dataset, split.validation_indices, device, cfg.local_batch_size
        )
        for name, value in val_metrics.items():
            if isinstance(value, float) and not math.isfinite(value):
                _fail(f"round {round_number}: non-finite validation metric {name}")

        round_seconds = time.perf_counter() - round_started
        round_peak_rss = monitor.peak_mib
        gpu_alloc = gpu_reserved = None
        if device.type == "cuda":
            torch.cuda.synchronize()
            gpu_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
            gpu_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)

        round_record = {
            "round": round_number,
            "clients": client_records,
            "total_client_samples": total_samples,
            "fedavg_weights_sum": weights_sum,
            "validation_metrics": val_metrics,
            "round_wall_seconds": round_seconds,
            "peak_rss_mib": round_peak_rss,
            "peak_gpu_allocated_mib": gpu_alloc,
            "peak_gpu_reserved_mib": gpu_reserved,
        }
        history.append(round_record)

        improved = val_metrics["val_install_logloss"] < best_val_install_logloss
        if improved:
            best_val_install_logloss = val_metrics["val_install_logloss"]
            best_round = round_number
            _atomic_torch_save(
                checkpoint_payload(
                    state={k: v.detach().cpu() for k, v in global_model.state_dict().items()},
                    round_number=round_number,
                    best_logloss=best_val_install_logloss,
                    best_round=best_round,
                    history=history,
                    cfg=cfg,
                    split=split,
                    feature_set=feature_set,
                    kind="best",
                ),
                best_ckpt,
            )
        _atomic_torch_save(
            checkpoint_payload(
                state={k: v.detach().cpu() for k, v in global_model.state_dict().items()},
                round_number=round_number,
                best_logloss=best_val_install_logloss,
                best_round=best_round,
                history=history,
                cfg=cfg,
                split=split,
                feature_set=feature_set,
                kind="latest",
            ),
            latest_ckpt,
        )
        if round_number == cfg.rounds:
            _atomic_torch_save(
                checkpoint_payload(
                    state={k: v.detach().cpu() for k, v in global_model.state_dict().items()},
                    round_number=round_number,
                    best_logloss=best_val_install_logloss,
                    best_round=best_round,
                    history=history,
                    cfg=cfg,
                    split=split,
                    feature_set=feature_set,
                    kind="final_round10",
                ),
                final_ckpt,
            )

        elapsed_total = time.perf_counter() - started_all
        result = {
            "stage": PROTOCOL_ID,
            "status": "running" if round_number < cfg.rounds else "complete",
            "rounds_completed": round_number,
            "protocol_rounds": cfg.rounds,
            "protocol": cfg.to_dict(),
            "seed": cfg.seed,
            "partition_hash": EXPECTED_PARTITION_HASH,
            "split_train_index_hash": split.train_index_hash,
            "split_validation_index_hash": split.validation_index_hash,
            "preprocessing_hash": feature_set.artifact_hash,
            "best_round": best_round,
            "best_val_install_logloss": best_val_install_logloss
            if best_round is not None else None,
            "history": history,
            "total_elapsed_seconds": elapsed_total,
            "device": str(device),
            "checkpoint_paths": {
                "best": BEST_CKPT_RELPATH,
                "final": FINAL_CKPT_RELPATH,
                "latest": LATEST_CKPT_RELPATH,
            },
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json_write(result, result_json)

        vm = val_metrics
        print(
            f"  >> round {round_number} SUMMARY: {round_seconds:.1f}s | val Install LogLoss "
            f"{vm['val_install_logloss']:.6f} (AUC {vm['val_install_auc']:.4f}) | Click LogLoss "
            f"{vm['val_click_logloss']:.6f} (AUC {vm['val_click_auc']:.4f}) | "
            f"{'NEW BEST -> checkpoint saved' if improved else f'best remains round {best_round}'} | "
            f"peak RSS {round_peak_rss:.0f} MiB"
            + (f" | peak GPU alloc/res {gpu_alloc:.0f}/{gpu_reserved:.0f} MiB"
               if gpu_alloc is not None else ""),
            flush=True,
        )
        print(f"  >> rounds completed: {round_number}/{cfg.rounds} | elapsed "
              f"{elapsed_total / 3600:.2f} h", flush=True)

    # ------------------------------------------------------------------
    # 3. Post-run verification
    # ------------------------------------------------------------------
    print("\n=== post-run verification ===", flush=True)
    peak_rss_total = monitor.stop()

    best_payload = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    if best_payload.get("stage") != PROTOCOL_ID or best_payload.get("kind") != "best":
        _fail("best checkpoint payload is not a Stage 10B best checkpoint")
    fresh = cold_model(feature_set, seed=cfg.seed)
    load_result = fresh.load_state_dict(best_payload["model_state_dict"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        _fail("best checkpoint strict reload reported missing/unexpected keys")
    reloaded_state = dict(fresh.state_dict())
    exact = all(torch.equal(reloaded_state[k], best_payload["model_state_dict"][k])
                for k in best_payload["model_state_dict"])
    print(f"  best checkpoint (round {best_payload['round']}): reload bit-identical: {exact}", flush=True)
    if not exact:
        _fail("best checkpoint reload is not bit-identical")

    reloaded_metrics = global_validation(
        fresh.to(device), dataset, split.validation_indices, device, cfg.local_batch_size
    )
    for key in ("val_install_logloss", "val_install_auc", "val_click_logloss", "val_click_auc"):
        delta = abs(reloaded_metrics[key] - best_payload["history"][best_round - 1]
                    ["validation_metrics"][key])
        print(f"  reloaded-best {key}: {reloaded_metrics[key]:.9f} (|d|={delta:.2e})", flush=True)
        if delta > 1e-6:
            _fail(f"reloaded best checkpoint {key} mismatch")
    del fresh, best_payload

    violations = verify_protected_artifacts(integrity_snapshot, PROJECT_ROOT)
    violations += _verify_fingerprint(fingerprint_before, PROJECT_ROOT)
    if violations:
        for violation in violations:
            print(f"  VIOLATION: {violation}", flush=True)
        return 1
    print(f"  protected artifacts unchanged: all {len(fingerprint_before)} guarded files "
          f"byte-identical (sha256+size+mtime)", flush=True)

    final_result = json.loads(result_json.read_text(encoding="utf-8"))
    final_result["status"] = "complete"
    final_result["reload_verified"] = True
    final_result["reloaded_metrics_match"] = True
    final_result["peak_rss_mib_overall"] = peak_rss_total
    final_result["completed_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json_write(final_result, result_json)

    print("\n=== STAGE 10B COMPLETE ===", flush=True)
    print(f"{'round':>5} | {'seconds':>8} | {'install LL':>10} | {'install AUC':>11} | "
          f"{'click LL':>9} | {'click AUC':>9} | best")
    for record in history:
        vm = record["validation_metrics"]
        marker = "  <== best" if record["round"] == best_round else ""
        print(f"{record['round']:>5} | {record['round_wall_seconds']:>8.1f} | "
              f"{vm['val_install_logloss']:>10.6f} | {vm['val_install_auc']:>11.4f} | "
              f"{vm['val_click_logloss']:>9.6f} | {vm['val_click_auc']:>9.4f} |{marker}")
    print(f"best round: {best_round} | best validation Install LogLoss: "
          f"{best_val_install_logloss:.6f}")
    print(f"total runtime: {time.perf_counter() - started_all:.1f}s "
          f"({(time.perf_counter() - started_all) / 3600:.2f} h) | peak RSS {peak_rss_total:.0f} MiB")
    print(f"outputs: {stage_dir}")
    print("official test set: NOT loaded, NOT evaluated (entire run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
