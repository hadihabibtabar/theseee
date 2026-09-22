"""Stage 12: federated runner for the four NEW matrix cells (real data).

ONE generic runner parameterized by a Stage 12 experiment ID; the training
machinery is NOT reimplemented — it is imported verbatim from the two
completed federated stages:

* no-SSL cells  -> ``run_stage10b.train_one_client`` (supervised BCE click+install)
* SSL cells     -> ``run_stage11.train_one_client_ssl`` (joint 0.6*sup + 0.4*NT-Xent)

Protocol = the frozen Stage 10A/10B/11 protocol, over the cell's own Stage 12
partition: 10 clients, all clients every round, 10 rounds, 1 local epoch,
batch 1024, AdamW lr 3e-4 / wd 1e-2 (fresh per client per round),
sample-weighted FedAvg, sampler epoch = round - 1, cold seed-42 init,
global validation on the frozen 348,586-row split after every round, best
checkpoint by validation Install LogLoss. Official test: never loaded.

Outputs (dedicated per-cell directory; Stage 10B/11 artifacts untouched):
    artifacts/federated/stage12/<experiment_id>/
        <prefix>_best.pt            best global checkpoint (Install LogLoss)
        <prefix>_final_round10.pt   final round-10 checkpoint
        <prefix>_latest.pt          per-round resume state (crash safety)
        <prefix>_result.json        protocol + partition + per-round metrics

Usage:
    python scripts/run_stage12_federated.py --experiment stage12_fed_nossl_alpha_1_0_seed42
    python scripts/run_stage12_federated.py --experiment ... [--preflight_only]
        [--device auto|cpu] [--resume] [--allow_restart] [--log_every N]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_stage10b as s10b  # noqa: E402  (frozen supervised training helpers)
import run_stage11 as s11  # noqa: E402    (frozen SSL training helpers)

from recsys23_fedrec.data import FeatureSet, ProcessedRecSysDataset  # noqa: E402
from recsys23_fedrec.experiments.integrity import verify_protected_artifacts  # noqa: E402
from recsys23_fedrec.federated.client_loader import load_partition_positions  # noqa: E402
from recsys23_fedrec.federated.config import get_federated_config  # noqa: E402
from recsys23_fedrec.federated.fedavg import (  # noqa: E402
    fedavg_state_dicts,
    fedavg_weights_from_counts,
)
from recsys23_fedrec.federated.partition import CLIENT_IDS, partition_hash  # noqa: E402
from recsys23_fedrec.federated.stage12_matrix import (  # noqa: E402
    STAGE_NUMBER,
    Stage12ExperimentSpec,
    get_experiment,
)
from recsys23_fedrec.training.split import load_split  # noqa: E402

# Frozen architecture identities for the two cell kinds (from Stage 10B/11).
SSL_CELL_PARAM_COUNT = s11.EXPECTED_SSL_PARAM_COUNT          # 2,527,698 / 142 entries
NOSSL_CELL_PARAM_COUNT = s10b.EXPECTED_PARAMETER_COUNT        # 2,502,930 / 138 entries

# ---------------------------------------------------------------------------


def _fail(message: str) -> None:
    print(f"\nSTAGE 12 FAILED: {message}", flush=True)
    raise SystemExit(1)


def _paths(spec: Stage12ExperimentSpec) -> dict[str, Path]:
    out = PROJECT_ROOT / spec.output_dir
    prefix = spec.experiment_id
    return {
        "dir": out,
        "best": out / f"{prefix}_best.pt",
        "final": out / f"{prefix}_final_round10.pt",
        "latest": out / f"{prefix}_latest.pt",
        "result": out / f"{prefix}_result.json",
    }


def _fingerprint(project_root: Path, relpaths: tuple[str, ...]) -> dict:
    out = {}
    for relpath in relpaths:
        path = project_root / relpath
        out[relpath] = {
            "sha256": s10b._sha256_file(path),
            "mtime": os.path.getmtime(path),
            "size": os.path.getsize(path),
        }
    return out


def _verify_fingerprint(before: dict, project_root: Path) -> list[str]:
    after = _fingerprint(project_root, tuple(before))
    return [f"protected artifact changed: {rel}"
            for rel, expected in before.items() if after.get(rel) != expected]


def _guarded_files(spec: Stage12ExperimentSpec) -> tuple[str, ...]:
    """Everything Stage 12 must leave byte-identical (Stage 11's list + both
    new Stage 12 partitions + the cell's own manifest)."""
    _, manifest_rel = spec.partition_npz, spec.partition_manifest
    return tuple(s11.PROTECTED_FILES_11) + (
        spec.partition_npz,
        manifest_rel,
        "artifacts/federated/partition_alpha_0_5_seed42.npz",
        "artifacts/federated/partition_alpha_0_5_seed42.json",
    )


def _result_relpaths(spec: Stage12ExperimentSpec) -> dict:
    out_dir_rel = spec.output_dir
    prefix = spec.experiment_id
    return {
        "best": f"{out_dir_rel}/{prefix}_best.pt",
        "final": f"{out_dir_rel}/{prefix}_final_round10.pt",
        "latest": f"{out_dir_rel}/{prefix}_latest.pt",
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
    spec: Stage12ExperimentSpec,
    partition_hash_value: str,
    model_state_entries: int,
) -> dict:
    """Stage 12 checkpoint payload (Stage 11 layout + matrix provenance)."""
    payload = s11.checkpoint_payload(
        state=state,
        round_number=round_number,
        best_logloss=best_logloss,
        best_round=best_round,
        history=history,
        cfg=cfg,
        split=split,
        feature_set=feature_set,
        kind=kind,
    )
    payload["stage"] = spec.experiment_id
    payload["stage_number"] = STAGE_NUMBER
    payload["stage12"] = {
        "experiment_id": spec.experiment_id,
        "use_ssl": spec.use_ssl,
        "alpha": spec.alpha,
        "model": spec.model,
        "matrix_runner": spec.runner,
    }
    payload["partition_hash"] = partition_hash_value
    payload["partition_npz"] = spec.partition_npz
    payload["model_state_dict"] = {k: v.cpu() for k, v in payload["model_state_dict"].items()}
    if not spec.use_ssl:
        # Stage 11's payload labels the supervised path as the SSL variant;
        # correct the model identity for the no-SSL cells.
        payload["model_type"] = "transformer_mmoe"
        payload["parameter_count"] = s10b.EXPECTED_PARAMETER_COUNT
        payload.pop("state_entry_count", None)
        payload.pop("ssl", None)
    assert len(payload["model_state_dict"]) == model_state_entries
    return payload


def _cold_model(spec: Stage12ExperimentSpec, feature_set: FeatureSet, seed: int):
    """Cold seed-42 model for the cell (Stage 11 SSL or Stage 10B supervised)."""
    if spec.use_ssl:
        return s11.cold_ssl_model(feature_set, seed=seed)
    return s10b.cold_model(feature_set, seed=seed)


def _train_one_client(spec, cid, positions, train_indices, dataset, global_state,
                      cfg, device, round_number, log_every):
    if spec.use_ssl:
        return s11.train_one_client_ssl(
            cid, positions, train_indices, dataset, global_state, cfg, device,
            round_number, log_every,
        )
    return s10b.train_one_client(
        cid, positions, train_indices, dataset, global_state, cfg, device,
        round_number, log_every,
    )


@torch.no_grad()
def _global_validation(spec, model, dataset, val_indices, device, batch_size: int):
    """Frozen 348,586-row global validation; single-view supervised path."""
    model.eval()
    from recsys23_fedrec.training.loaders import make_subset_loader
    from recsys23_fedrec.training.metrics import binary_logloss, roc_auc

    loader = make_subset_loader(
        dataset, val_indices, batch_size=batch_size, shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    click_logits, install_logits = [], []
    click_targets, install_targets = [], []
    click_sum = install_sum = 0.0
    n_seen = 0
    for batch in loader:
        batch = s10b._to_device(batch, device)
        # Both Stage 10B and Stage 11 evaluate through the plain supervised
        # forward (Stage 6 behavior, no augmentation, no projection head).
        output = model.supervised_forward(batch, return_diagnostics=False) \
            if hasattr(model, "supervised_forward") else model(batch, return_diagnostics=False)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True,
                        help="Stage 12 experiment ID (see stage12_matrix.STAGE12_EXPERIMENTS)")
    parser.add_argument("--device", default="auto", help="'auto' (CUDA if available) or 'cpu'")
    parser.add_argument("--log_every", type=int, default=200,
                        help="client progress log interval (batches)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from <prefix>_latest.pt after an interrupted run")
    parser.add_argument("--allow_restart", action="store_true",
                        help="discard this cell's existing Stage 12 outputs and start fresh")
    parser.add_argument("--preflight_only", action="store_true",
                        help="run integrity + data verification only, then stop")
    args = parser.parse_args()

    spec = get_experiment(args.experiment)
    if spec.status != "new":
        _fail(f"{spec.experiment_id} is an existing cell ({spec.provenance}); "
              "its original runner owns it. Stage 10B/11 artifacts are protected.")
    paths = _paths(spec)
    cfg = get_federated_config()  # frozen Stage 10A protocol for every cell

    print(f"=== STAGE 12: {spec.experiment_id} ===", flush=True)
    print(f"start: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", flush=True)
    print(f"cell: {'SSL' if spec.use_ssl else 'no-SSL'} | alpha {spec.alpha} | "
          f"partition {spec.partition_npz} | output {spec.output_dir}", flush=True)
    print(f"protocol: 10 clients, seed {cfg.seed}, {cfg.rounds} rounds, 1 local epoch, "
          f"batch {cfg.local_batch_size}, AdamW lr {cfg.learning_rate} wd {cfg.weight_decay}, "
          f"{cfg.aggregation}, cold seed-42 init", flush=True)

    # ------------------------------------------------------------------
    # 0. Integrity + environment + data layer
    # ------------------------------------------------------------------
    print("\n=== pre-run integrity verification ===", flush=True)
    integrity_snapshot = s10b.snapshot_protected_artifacts(PROJECT_ROOT)
    violations = verify_protected_artifacts(integrity_snapshot, PROJECT_ROOT)
    if violations:
        _fail("protected artifacts do not match frozen values: " + "; ".join(violations))
    print("  Stage 2 / split / Stage 6 / raw bytes: all frozen values OK", flush=True)

    # Stage 11's own guarded set (Stage 1-9 + Stage 10B + Stage 11 outputs)
    guarded = _guarded_files(spec)
    for relpath in guarded:
        if not (PROJECT_ROOT / relpath).exists():
            _fail(f"guarded file missing before the run: {relpath}")
    fingerprint_before = _fingerprint(PROJECT_ROOT, guarded)
    print(f"  fingerprint recorded over {len(fingerprint_before)} guarded files "
          "(Stage 1-9, Stage 10B, Stage 11, Stage 9 alpha=0.5 partition, this cell's partition)",
          flush=True)

    feature_set = FeatureSet.load(PROJECT_ROOT / "artifacts" / "preprocessing")
    if feature_set.artifact_hash != s10b.STAGE2_ARTIFACT_HASH:
        _fail(f"Stage 2 artifact hash {feature_set.artifact_hash} != {s10b.STAGE2_ARTIFACT_HASH}")
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

    # ---- cell partition verification (this cell's OWN Stage 12 artifact) ----
    manifest = json.loads((PROJECT_ROOT / spec.partition_manifest).read_text(encoding="utf-8"))
    positions = load_partition_positions(PROJECT_ROOT / spec.partition_npz)
    recomputed = partition_hash([positions[cid] for cid in CLIENT_IDS])
    if not (recomputed == manifest["partition_hash"]):
        _fail(f"partition hash mismatch: manifest {manifest['partition_hash']} != recomputed {recomputed}")
    if manifest["alpha"] != spec.alpha or manifest["seed"] != cfg.seed:
        _fail(f"partition metadata alpha={manifest['alpha']} seed={manifest['seed']} "
              f"does not match the cell (alpha={spec.alpha}, seed={cfg.seed})")
    counts = {cid: int(pos.size) for cid, pos in positions.items()}
    if sum(counts.values()) != s10b.EXPECTED_TOTAL_TRAIN_ROWS:
        _fail(f"client sizes sum {sum(counts.values()):,} != {s10b.EXPECTED_TOTAL_TRAIN_ROWS:,}")
    max_overlap = 0
    for cid in CLIENT_IDS:
        rows = split.train_indices[positions[cid]]
        max_overlap = max(max_overlap, int(np.intersect1d(rows, split.validation_indices).size))
    if max_overlap:
        _fail(f"client rows overlap the frozen validation split ({max_overlap} rows)")
    print(f"  [OK] partition hash {recomputed} | alpha {spec.alpha} | 10 clients | "
          f"rows {sum(counts.values()):,} | validation overlap 0", flush=True)
    print("  official test data: NOT loaded (never constructed in this script)", flush=True)

    if args.preflight_only:
        print("\nRESULT: PREFLIGHT ONLY - stopping before the Stage 12 run as requested")
        return 0

    # ------------------------------------------------------------------
    # 1. Global model: cold seed-42 init (or deterministic resume)
    # ------------------------------------------------------------------
    started_all = time.perf_counter()
    monitor = s10b.RSSMonitor(interval=10.0)
    monitor.start()

    history: list[dict] = []
    best_round: int | None = None
    best_val_install_logloss = float("inf")
    start_round = 1

    expected_params = (
        s11.EXPECTED_SSL_PARAM_COUNT if spec.use_ssl else s10b.EXPECTED_PARAMETER_COUNT
    )
    if args.resume:
        if not paths["latest"].exists():
            _fail("--resume requested but no resume state found: " + str(paths["latest"]))
        resume_payload = torch.load(paths["latest"], map_location="cpu", weights_only=False)
        if resume_payload.get("stage") != spec.experiment_id:
            _fail("resume state is not a payload of this Stage 12 cell")
        start_round = int(resume_payload["round"]) + 1
        if start_round > cfg.rounds:
            _fail("resume state says all rounds are already complete")
        global_model = _cold_model(spec, feature_set, cfg.seed).to(device)
        global_model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        history = list(resume_payload["history"])
        best_round = resume_payload["best_round"]
        best_val_install_logloss = float(resume_payload["best_validation_install_logloss"])
        print(f"\n=== RESUME: continuing from round {start_round} "
              f"(best so far: round {best_round}, Install LogLoss "
              f"{best_val_install_logloss:.6f}) ===", flush=True)
    else:
        print(f"\n=== 1. cold seed-42 global initialization ({'SSL' if spec.use_ssl else 'supervised'}) ===",
              flush=True)
        global_model = _cold_model(spec, feature_set, cfg.seed).to(device)
        if sum(p.numel() for p in global_model.parameters()) != expected_params:
            _fail("global parameter count does not match the frozen architecture")
        initial_probe = _cold_model(spec, feature_set, cfg.seed)
        global_sd_cpu = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        if any(not torch.equal(initial_probe.state_dict()[k], global_sd_cpu[k])
               for k in global_sd_cpu):
            _fail("seed-42 initialization is not deterministic across two creations")
        del initial_probe
        if s10b._count_nonfinite(global_sd_cpu):
            _fail("non-finite initial global parameters")
        print(f"  parameter count {expected_params:,}; state entries {len(global_sd_cpu)}",
              flush=True)
        print("  cold init: no checkpoint loaded (Stage 6/8/10B/11 outputs untouched)",
              flush=True)

    # ------------------------------------------------------------------
    # 2. Round loop r = 1..10
    # ------------------------------------------------------------------
    for round_number in range(start_round, cfg.rounds + 1):
        round_started = time.perf_counter()
        monitor.reset()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        current_state = s10b._cpu_state(global_model)
        if s10b._count_nonfinite(current_state):
            _fail(f"non-finite global parameters at the start of round {round_number}")

        print(f"\n=== ROUND {round_number}/{cfg.rounds} | sampler epoch {round_number - 1} "
              f"per client ===", flush=True)
        client_records: list[dict] = []
        local_states: dict[str, dict] = {}
        for cid in CLIENT_IDS:  # client order is significant for FedAvg reproducibility
            record, state = _train_one_client(
                spec, cid, positions[cid], split.train_indices, dataset,
                current_state, cfg, device, round_number, args.log_every,
            )
            client_records.append(record)
            local_states[cid] = state
            extra = (
                f" (contrastive {record['local_contrastive_loss']:.4f})"
                if spec.use_ssl and "local_contrastive_loss" in record else ""
            )
            print(
                f"  r{round_number} {cid} done: {record['samples']:,} rows | loss "
                f"{record['local_total_loss']:.4f} | {record['duration_seconds']:.1f}s"
                + extra,
                flush=True,
            )

        # ---- per-round participation + integrity checks ----------------
        if len(client_records) != 10:
            _fail(f"round {round_number}: {len(client_records)} clients participated, expected 10")
        total_samples = sum(r["samples"] for r in client_records)
        if total_samples != s10b.EXPECTED_TOTAL_TRAIN_ROWS:
            _fail(f"round {round_number}: total client samples {total_samples:,} "
                  f"!= {s10b.EXPECTED_TOTAL_TRAIN_ROWS:,}")
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
        if s10b._count_nonfinite(aggregated):
            _fail(f"round {round_number}: non-finite aggregated state")
        if len(aggregated) != len(global_model.state_dict()):
            _fail(f"round {round_number}: aggregated state has {len(aggregated)} entries, "
                  f"expected {len(global_model.state_dict())}")

        global_model.load_state_dict(aggregated, strict=True)
        if s10b._count_nonfinite({k: v.detach().cpu() for k, v in global_model.state_dict().items()}):
            _fail(f"round {round_number}: non-finite global parameters after aggregation")

        # ---- global validation (frozen split only) ----------------------
        val_metrics = _global_validation(
            spec, global_model, dataset, split.validation_indices, device, cfg.local_batch_size
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
            s10b._atomic_torch_save(
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
                    spec=spec,
                    partition_hash_value=recomputed,
                    model_state_entries=len(global_model.state_dict()),
                ),
                paths["best"],
            )
        s10b._atomic_torch_save(
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
                spec=spec,
                partition_hash_value=recomputed,
                model_state_entries=len(global_model.state_dict()),
            ),
            paths["latest"],
        )
        if round_number == cfg.rounds:
            s10b._atomic_torch_save(
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
                    spec=spec,
                    partition_hash_value=recomputed,
                    model_state_entries=len(global_model.state_dict()),
                ),
                paths["final"],
            )

        elapsed_total = time.perf_counter() - started_all
        result = {
            "stage": spec.experiment_id,
            "stage12": spec.to_dict(),
            "status": "running" if round_number < cfg.rounds else "complete",
            "rounds_completed": round_number,
            "protocol_rounds": cfg.rounds,
            "protocol": cfg.to_dict(),
            "ssl": ({
                "alpha": s11.SSL_ALPHA,
                "temperature": s11.SSL_TEMPERATURE,
                "projection_dim": s11.SSL_PROJECTION_DIM,
                "corruption_rate": s11.SSL_CORRUPTION_RATE,
            } if spec.use_ssl else None),
            "seed": cfg.seed,
            "partition_hash": recomputed,
            "partition_npz": spec.partition_npz,
            "split_train_index_hash": split.train_index_hash,
            "split_validation_index_hash": split.validation_index_hash,
            "preprocessing_hash": feature_set.artifact_hash,
            "best_round": best_round,
            "best_val_install_logloss": best_val_install_logloss
            if best_round is not None else None,
            "history": history,
            "total_elapsed_seconds": elapsed_total,
            "device": str(device),
            "checkpoint_paths": _result_relpaths(spec),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        s10b._atomic_json_write(result, paths["result"])

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

    best_payload = torch.load(paths["best"], map_location="cpu", weights_only=False)
    if best_payload.get("stage") != spec.experiment_id or best_payload.get("kind") != "best":
        _fail("best checkpoint payload is not a Stage 12 best checkpoint of this cell")
    fresh = _cold_model(spec, feature_set, cfg.seed)
    load_result = fresh.load_state_dict(best_payload["model_state_dict"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        _fail("best checkpoint strict reload reported missing/unexpected keys")
    reloaded_state = s10b._cpu_state(fresh)
    exact = all(torch.equal(reloaded_state[k], best_payload["model_state_dict"][k])
                for k in best_payload["model_state_dict"])
    print(f"  best checkpoint (round {best_payload['round']}): reload bit-identical: {exact}",
          flush=True)
    if not exact:
        _fail("best checkpoint reload is not bit-identical")

    reloaded_metrics = _global_validation(
        spec, fresh.to(device), dataset, split.validation_indices, device, cfg.local_batch_size
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
          "byte-identical (sha256+size+mtime)", flush=True)

    final_result = json.loads(paths["result"].read_text(encoding="utf-8"))
    final_result["status"] = "complete"
    final_result["reload_verified"] = True
    final_result["reloaded_metrics_match"] = True
    final_result["peak_rss_mib_overall"] = peak_rss_total
    final_result["completed_at"] = datetime.now(timezone.utc).isoformat()
    s10b._atomic_json_write(final_result, paths["result"])

    print(f"\n=== STAGE 12 CELL COMPLETE: {spec.experiment_id} ===", flush=True)
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
    print(f"outputs: {paths['dir']}")
    print("official test set: NOT loaded, NOT evaluated (entire run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
