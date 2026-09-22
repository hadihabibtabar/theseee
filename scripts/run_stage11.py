"""Stage 11: federated Transformer + MMoE + Self-Supervised Contrastive Learning.

The federated version of the already-tested Stage 7/8 SSL architecture over
the FROZEN Stage 9 partition. The protocol is the FROZEN Stage 10A/10B
federated baseline — nothing here may alter it — with ONE conceptual
addition: the already-implemented contrastive SSL mechanism.

Reused verbatim (nothing duplicated):
* ``SSLTransformerMMoE``          (Stage 7 model: frozen backbone+MMoE + projection head)
* ``FeatureCorruptionAugmentation`` (Stage 7 two-view feature corruption, rate 0.15)
* ``ssl_joint_loss``              (Stage 7: 0.6 * L_sup + 0.4 * L_contrast, symmetric
                                  NT-Xent temperature 0.2 inside)
* ``make_client_loader``          (Stage 9 client loader; sampler epoch = r - 1 on
                                  ``loader.batch_sampler`` only)
* ``fedavg_state_dicts``          (tested sample-weighted FedAvg)
* ``FederatedConfig``             (frozen Stage 10A protocol values)
* validation/metrics/integrity    (Stage 10B evaluation + integrity utilities, and
                                  helpers imported directly from ``run_stage10b.py``)

Protocol (identical to Stage 10B except the local objective):
* 10 simulated clients, Dirichlet alpha 0.5, seed 42, all clients every round,
* model: existing SSLTransformerMMoE (2,502,930 frozen backbone+MMoE params
  + 24,768 projection params = 2,527,698 total; 142 state entries),
  COLD seed-42 init; Stage 6 / Stage 8 A/C/D / Stage 10B checkpoints are
  NEVER loaded (Stage 10B outputs are guarded as protected artifacts),
* per client per round: fresh local SSL model with the exact global state,
  fresh AdamW (lr 3e-4, wd 1e-2, no state carried between rounds), exactly
  ONE local epoch, joint objective
  ``L = 0.6 * (BCE(click) + BCE(install)) + 0.4 * NT-Xent(T=0.2)`` over two
  independently corrupted views (rate 0.15; cat->UNK 0, num/bin -> 0.0,
  missing preserved, labels untouched, CPU-side deterministic generators),
  batch 1024, sampler epoch EXACTLY ``r - 1``,
* aggregation: the already-tested ``fedavg_state_dicts`` (sample-weighted,
  client order client_00..client_09),
* validation: frozen 348,586-row global validation split only (single-view
  supervised path = Stage 6 behavior); primary metric Install LogLoss,
* official test set is never loaded or evaluated,
* client rows are always ``frozen_train_indices[client_positions]``.

Outputs (dedicated directory; Stage 10B artifacts stay byte-identical):
    artifacts/federated/stage11/
        stage11_best.pt            best global checkpoint (by Install LogLoss)
        stage11_final_round10.pt   final round-10 checkpoint
        stage11_latest.pt          per-round resume state (crash safety)
        stage11_result.json        protocol snapshot + per-round metrics

Usage:
    python scripts/run_stage11.py [--device auto|cpu] [--resume]
        [--preflight_only] [--allow_restart] [--log_every N]
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

import run_stage10b as s10b  # noqa: E402  (reused helpers/constants — not duplicated)

from recsys23_fedrec.data import FeatureSet, ProcessedRecSysDataset  # noqa: E402
from recsys23_fedrec.experiments.integrity import (  # noqa: E402
    verify_protected_artifacts,
)
from recsys23_fedrec.federated.client_loader import (  # noqa: E402
    client_sampler,
    make_client_loader,
)
from recsys23_fedrec.federated.config import get_federated_config  # noqa: E402
from recsys23_fedrec.federated.fedavg import (  # noqa: E402
    fedavg_state_dicts,
    fedavg_weights_from_counts,
)
from recsys23_fedrec.federated.partition import CLIENT_IDS  # noqa: E402
from recsys23_fedrec.models import SSLConfig, SSLTransformerMMoE  # noqa: E402
from recsys23_fedrec.models.mmoe import mmoe_loss  # noqa: E402
from recsys23_fedrec.ssl import FeatureCorruptionAugmentation  # noqa: E402
from recsys23_fedrec.ssl.contrastive_loss import nt_xent_loss  # noqa: E402
from recsys23_fedrec.ssl.joint_loss import ssl_joint_loss  # noqa: E402
from recsys23_fedrec.training.loaders import make_subset_loader  # noqa: E402
from recsys23_fedrec.training.metrics import binary_logloss, roc_auc  # noqa: E402
from recsys23_fedrec.training.split import load_split  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen Stage 11 protocol constants (SSL parts) — Stage 10A values otherwise.
# ---------------------------------------------------------------------------
PROTOCOL_ID = "stage11-federated-ssl"
STAGE_NUMBER = 11
MODEL_TYPE = "ssl_transformer_mmoe"

SSL_ALPHA = 0.6                    # L = 0.6 * L_sup + 0.4 * L_contrast
SSL_TEMPERATURE = 0.2              # symmetric NT-Xent temperature
SSL_CORRUPTION_RATE = 0.15         # Stage 7 feature-corruption rate
SSL_PROJECTION_DIM = 64            # projection head output dim
SSL_AUG_SEED = 42                  # CPU-side deterministic generator convention

# Architecture facts verified against the existing Stage 7 implementation:
#   backbone + MMoE (mmoe.*) = 2,502,930 params / 138 state entries (Stage 10B model)
#   projection head (projection.*) = 24,768 params / 4 state entries (SSL add-on)
EXPECTED_FROZEN_PARAM_COUNT = 2_502_930    # mmoe.* sub-module (Stage 10B architecture)
EXPECTED_SSL_PARAM_COUNT = 2_527_698       # full SSLTransformerMMoE
EXPECTED_SSL_STATE_ENTRIES = 142           # 138 mmoe.* + 4 projection.*
EXPECTED_MM_STATE_ENTRIES = 138

STAGE11_DIR_RELPATH = "artifacts/federated/stage11"
BEST_CKPT_RELPATH = f"{STAGE11_DIR_RELPATH}/stage11_best.pt"
FINAL_CKPT_RELPATH = f"{STAGE11_DIR_RELPATH}/stage11_final_round10.pt"
LATEST_CKPT_RELPATH = f"{STAGE11_DIR_RELPATH}/stage11_latest.pt"
RESULT_JSON_RELPATH = f"{STAGE11_DIR_RELPATH}/stage11_result.json"

# Everything Stage 11 must leave byte-identical: Stage 1-9 guarded files
# (via s10b.PROTECTED_FILES) PLUS the completed Stage 10B baseline artifacts.
STAGE10B_OUTPUTS = (
    s10b.BEST_CKPT_RELPATH,
    s10b.FINAL_CKPT_RELPATH,
    s10b.LATEST_CKPT_RELPATH,
    s10b.RESULT_JSON_RELPATH,
)
PROTECTED_FILES_11 = tuple(s10b.PROTECTED_FILES) + STAGE10B_OUTPUTS


def _fail(message: str) -> None:
    print(f"\nSTAGE 11 FAILED: {message}", flush=True)
    raise SystemExit(1)


def _fingerprint11(project_root: Path) -> dict:
    """sha256+size+mtime over every Stage 11 guarded file (incl. Stage 10B)."""
    out = {}
    for relpath in PROTECTED_FILES_11:
        path = project_root / relpath
        out[relpath] = {
            "sha256": s10b._sha256_file(path),
            "mtime": os.path.getmtime(path),
            "size": os.path.getsize(path),
        }
    return out


def _verify_fingerprint11(before: dict, project_root: Path) -> list[str]:
    violations = []
    after = _fingerprint11(project_root)
    for relpath, expected in before.items():
        if after.get(relpath) != expected:
            violations.append(f"protected artifact changed: {relpath}")
    return violations


def cold_ssl_model(feature_set: FeatureSet, seed: int = SSL_AUG_SEED) -> SSLTransformerMMoE:
    """Fresh, deterministic seed-42 SSLTransformerMMoE (cold init; no loads).

    The two-view augmentation module is attached with the Stage 7 CPU-side
    deterministic generator convention (seed 42, per-view streams
    ``seed + 1_000_003 * (step + 1)``). A fresh module per client/round makes
    every client's corruption stream identical at stream start and fully
    reproducible.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return SSLTransformerMMoE(
        feature_set,
        ssl_config=SSLConfig(projection_dim=SSL_PROJECTION_DIM, temperature=SSL_TEMPERATURE),
        augmentations=FeatureCorruptionAugmentation(
            feature_set, corruption_rate=SSL_CORRUPTION_RATE, seed=SSL_AUG_SEED
        ),
    )


def train_one_client_ssl(
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
    """ONE client-local SSL round: fresh model/optimizer, ONE local epoch.

    Mirrors ``run_stage10b.train_one_client`` exactly — same loader factory,
    sampler convention, strict global-state load, fresh AdamW, sample
    accounting and parameter-change verification — with the joint SSL
    objective replacing the supervised-only loss.
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
    sampler = client_sampler(loader)
    if sampler.epoch != round_number - 1:
        _fail(f"{client_id}: sampler epoch {sampler.epoch} != {round_number - 1} (round {round_number})")
    sampler_kind = type(getattr(loader, "sampler", None)).__name__

    local_model = cold_ssl_model(dataset.feature_set, seed=cfg.seed)
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
    totals = {"total": 0.0, "click": 0.0, "install": 0.0, "contrastive": 0.0}
    seen = 0
    n_batches = 0
    for batch in loader:
        batch = s10b._to_device(batch, device)
        optimizer.zero_grad()
        output = local_model(batch, return_diagnostics=False)  # two corrupted views
        loss, components = ssl_joint_loss(
            output,
            batch,
            alpha=SSL_ALPHA,
            lambda_click=cfg.lambda_click,
            lambda_install=cfg.lambda_install,
            temperature=SSL_TEMPERATURE,
            return_diagnostics=True,
        )
        if not torch.isfinite(loss):
            _fail(
                f"{client_id}: non-finite joint loss at round {round_number} "
                f"batch {n_batches + 1}"
            )
        loss.backward()
        optimizer.step()

        batch_n = int(batch["install"].shape[0])
        totals["total"] += float(loss.item()) * batch_n
        totals["click"] += float(components["click"].item()) * batch_n
        totals["install"] += float(components["install"].item()) * batch_n
        totals["contrastive"] += float(components["contrastive"].item()) * batch_n
        seen += batch_n
        n_batches += 1
        if log_every and n_batches % log_every == 0:
            print(
                f"    r{round_number} {client_id} batch {n_batches} | running joint "
                f"{totals['total'] / seen:.4f} (contrastive {totals['contrastive'] / seen:.4f})",
                flush=True,
            )

    if seen != n_expected:
        _fail(f"{client_id}: local epoch saw {seen:,} rows, partition says {n_expected:,}")
    local_state = s10b._cpu_state(local_model)
    if s10b._count_nonfinite(local_state):
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
        "local_contrastive_loss": totals["contrastive"] / seen,
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
def global_validation_ssl(model: SSLTransformerMMoE, dataset, val_indices, device, batch_size: int):
    """Central evaluation on the frozen global validation split only.

    Single-view supervised path (``supervised_forward`` = exact Stage 6
    behavior): no augmentation, no projection, identical metric contract to
    the Stage 10B validation — required for protocol fairness.
    """
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
        batch = s10b._to_device(batch, device)
        output = model.supervised_forward(batch, return_diagnostics=False)
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
        "stage_number": STAGE_NUMBER,
        "protocol": cfg.to_dict(),
        "kind": kind,
        "round": int(round_number),
        "protocol_rounds": cfg.rounds,
        "best_validation_install_logloss": float(best_logloss),
        "best_round": best_round,
        "model_type": MODEL_TYPE,
        "parameter_count": EXPECTED_SSL_PARAM_COUNT,
        "state_entry_count": EXPECTED_SSL_STATE_ENTRIES,
        "ssl": {
            "alpha": SSL_ALPHA,
            "temperature": SSL_TEMPERATURE,
            "projection_dim": SSL_PROJECTION_DIM,
            "corruption_rate": SSL_CORRUPTION_RATE,
            "augmentation": "FeatureCorruptionAugmentation (Stage 7, CPU generators)",
            "contrastive_loss": "symmetric NT-Xent (nt_xent_loss)",
            "joint_loss": "ssl_joint_loss (Stage 7)",
            "eval_path": "supervised_forward (Stage 6 behavior, no augmentation)",
        },
        "model_state_dict": {k: v.cpu() for k, v in state.items()},
        "history": history,
        "seed": cfg.seed,
        "partition_hash": s10b.EXPECTED_PARTITION_HASH,
        "partition_npz": s10b.NPZ_RELPATH,
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


# ---------------------------------------------------------------------------
# Preflight: data layer + SSL integration smoke (real data, one CUDA batch)
# ---------------------------------------------------------------------------

def verify_data_layer_11(dataset, split, project_root: Path) -> dict:
    """Stage 10B data verification PLUS client mutual-disjointness (#7)."""
    counts, positions = s10b.verify_data_layer(dataset, split, project_root, None)
    composed = np.concatenate([split.train_indices[positions[cid]] for cid in CLIENT_IDS])
    if not np.array_equal(np.unique(composed), np.sort(composed)):
        _fail("client rows are not mutually disjoint (duplicate actual row indices)")
    if composed.size != s10b.EXPECTED_TOTAL_TRAIN_ROWS:
        _fail(f"composed client rows {composed.size:,} != {s10b.EXPECTED_TOTAL_TRAIN_ROWS:,}")
    print("  [OK] client rows mutually disjoint (10 clients, 45 pairs, 0 shared rows)",
          flush=True)
    return counts


def ssl_integration_smoke(
    dataset: ProcessedRecSysDataset,
    split,
    positions: dict[str, np.ndarray],
    cfg,
    device: torch.device,
    project_root: Path,
) -> dict:
    """Real-data preflight: architecture facts + one-batch/one-step CUDA smoke.

    Covers preflight checks 12-30 of the Stage 11 spec. Read-only with
    respect to every protected artifact; the only file written is a temporary
    checkpoint round-trip file inside the Stage 11 output directory, deleted
    again before returning.
    """
    results: dict[str, object] = {}
    fs = dataset.feature_set

    # ---- #28 cold seed-42 SSL initialization is deterministic --------------
    model_a = cold_ssl_model(fs)
    model_b = cold_ssl_model(fs)
    sd_a = s10b._cpu_state(model_a)
    sd_b = s10b._cpu_state(model_b)
    deterministic = all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    results["cold_init_deterministic"] = deterministic
    print(f"  [OK] cold seed-42 SSL init deterministic across two creations: {deterministic}",
          flush=True)
    if not deterministic:
        _fail("cold SSL initialization is not deterministic")

    # ---- #12/#13/#14 architecture facts ------------------------------------
    n_params = sum(p.numel() for p in model_a.parameters())
    n_params_mmoe = sum(p.numel() for p in model_a.mmoe.parameters())
    n_entries = len(model_a.state_dict())
    n_entries_mmoe = len(model_a.mmoe.state_dict())
    print(f"  [OK] parameter count: {n_params:,} total "
          f"(mmoe.* frozen backbone+MMoE {n_params_mmoe:,})", flush=True)
    print(f"  [OK] state entries: {n_entries} total ({n_entries_mmoe} mmoe.* "
          f"+ {n_entries - n_entries_mmoe} projection.*)", flush=True)
    results["param_count_total"] = n_params
    results["param_count_mmoe"] = n_params_mmoe
    results["state_entries_total"] = n_entries
    if n_params != EXPECTED_SSL_PARAM_COUNT or n_params_mmoe != EXPECTED_FROZEN_PARAM_COUNT:
        _fail(f"parameter count {n_params:,}/{n_params_mmoe:,} != expected "
              f"{EXPECTED_SSL_PARAM_COUNT:,}/{EXPECTED_FROZEN_PARAM_COUNT:,}")
    if n_entries != EXPECTED_SSL_STATE_ENTRIES or n_entries_mmoe != EXPECTED_MM_STATE_ENTRIES:
        _fail(f"state entries {n_entries}/{n_entries_mmoe} != expected "
              f"{EXPECTED_SSL_STATE_ENTRIES}/{EXPECTED_MM_STATE_ENTRIES}")
    if s10b._count_nonfinite(sd_a):
        _fail("non-finite parameters in the cold SSL model")          # #14
    print("  [OK] all model parameters finite", flush=True)

    # ---- #29 cold init does NOT load Stage 6 / Stage 10B -------------------
    # Both protected checkpoints store a plain TransformerMMoE state (138
    # entries WITHOUT the "mmoe." prefix); the SSL model nests it under
    # "mmoe.*", so compare the corresponding 138 sub-module entries.
    mmoe_entries = {k: v for k, v in sd_a.items() if k.startswith("mmoe.")}
    projection_entries = {k for k in sd_a if k.startswith("projection.")}
    if len(mmoe_entries) != EXPECTED_MM_STATE_ENTRIES or len(projection_entries) != 4:
        _fail("unexpected SSL state layout for checkpoint comparison")
    for label, relpath in (
        ("Stage 6", "artifacts/checkpoints/centralized_transformer_mmoe_best.pt"),
        ("Stage 10B best", s10b.BEST_CKPT_RELPATH),
    ):
        ref = torch.load(project_root / relpath, map_location="cpu", weights_only=False)
        ref_sd = ref["model_state_dict"]
        if len(ref_sd) != EXPECTED_MM_STATE_ENTRIES:
            _fail(f"{label} checkpoint has {len(ref_sd)} entries, expected "
                  f"{EXPECTED_MM_STATE_ENTRIES}")
        if set(ref_sd) != {k[len("mmoe."):] for k in mmoe_entries}:
            _fail(f"{label} checkpoint keys do not correspond to the SSL mmoe.* entries")
        differing = sum(1 for k in ref_sd
                        if not torch.equal(mmoe_entries["mmoe." + k], ref_sd[k]))
        if differing != EXPECTED_MM_STATE_ENTRIES:
            _fail(f"cold SSL init coincides with the {label} checkpoint "
                  f"({differing}/{EXPECTED_MM_STATE_ENTRIES} entries differ)")
        print(f"  [OK] cold init: all {differing}/{EXPECTED_MM_STATE_ENTRIES} mmoe.* "
              f"entries differ from the {label} checkpoint (not loaded); the 4 "
              f"projection.* entries have no counterpart there", flush=True)
        del ref

    # ---- one real batch (client_02, sampler epoch 0) -----------------------
    # ---- #25/#26 sampler epoch convention ----------------------------------
    loader0 = make_client_loader(
        dataset, split.train_indices, positions["client_02"],
        batch_size=cfg.local_batch_size, shuffle=True, seed=cfg.seed,
        epoch=0, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    sampler0 = client_sampler(loader0)
    if sampler0.epoch != 0:
        _fail(f"round-1 sampler epoch is {sampler0.epoch}, expected 0")   # #25
    first_batch0 = s10b._to_device(next(iter(loader0)), device)
    loader5 = make_client_loader(
        dataset, split.train_indices, positions["client_02"],
        batch_size=cfg.local_batch_size, shuffle=True, seed=cfg.seed,
        epoch=5, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    sampler5 = client_sampler(loader5)
    if sampler5.epoch != 5:
        _fail(f"epoch-5 sampler epoch is {sampler5.epoch}, expected 5")   # #26
    first_batch5 = s10b._to_device(next(iter(loader5)), device)
    arrangement_changes = not torch.equal(
        first_batch0["categorical"], first_batch5["categorical"]
    )
    if not arrangement_changes:
        _fail("sampler epoch change did not alter the batch arrangement")
    print(f"  [OK] sampler epoch: round 1 -> 0, later rounds set correctly "
          f"(epoch 5 arrangement differs from epoch 0)", flush=True)
    results["sampler_epoch_round1"] = int(sampler0.epoch)
    results["sampler_epoch_respects_round"] = True
    del loader5, first_batch5

    batch = first_batch0
    bsz = int(batch["install"].shape[0])

    # ---- #15/#16 two-view augmentation behavior ----------------------------
    aug = model_a.augmentations
    view1 = aug.augment(batch)
    view2 = aug.augment(batch)
    labels_preserved = (
        torch.equal(view1["click"], batch["click"])
        and torch.equal(view1["install"], batch["install"])
        and torch.equal(view2["click"], batch["click"])
        and torch.equal(view2["install"], batch["install"])
    )
    missing_preserved = (
        torch.equal(view1["missing"], batch["missing"])
        and torch.equal(view2["missing"], batch["missing"])
    )
    views_differ = not (
        torch.equal(view1["categorical"], view2["categorical"])
        and torch.equal(view1["numerical"], view2["numerical"])
        and torch.equal(view1["binary"], view2["binary"])
    )
    if not (labels_preserved and missing_preserved and views_differ):
        _fail(f"augmentation check failed: labels {labels_preserved}, "
              f"missing {missing_preserved}, views differ {views_differ}")
    print(f"  [OK] two-view augmentation (rate {SSL_CORRUPTION_RATE}): labels and "
          f"missing indicators preserved; the two views differ; no in-place "
          f"mutation (fresh tensors from torch.where)", flush=True)
    results["augmentation_labels_preserved"] = True
    results["augmentation_views_differ"] = True

    # ---- #17/#18/#19 dual-view forward shapes ------------------------------
    model_a = model_a.to(device)
    output = model_a(batch, return_diagnostics=True)
    z1, z2 = output["z1"], output["z2"]
    out1, out2 = output["view1"], output["view2"]
    if tuple(z1.shape) != (bsz, SSL_PROJECTION_DIM) or tuple(z2.shape) != (bsz, SSL_PROJECTION_DIM):
        _fail(f"projection shape {tuple(z1.shape)} != {(bsz, SSL_PROJECTION_DIM)}")  # #17
    norm1 = z1.norm(dim=-1)
    norm2 = z2.norm(dim=-1)
    norms_ok = bool(
        torch.allclose(norm1, torch.ones_like(norm1), atol=1e-5)
        and torch.allclose(norm2, torch.ones_like(norm2), atol=1e-5)
    )
    if not norms_ok:
        _fail("projection vectors are not L2-normalized")                 # #18
    if tuple(out1["click_logit"].shape) != (bsz,) or tuple(out2["install_logit"].shape) != (bsz,):
        _fail(f"task logits shape != ({bsz},)")                            # #19
    print(f"  [OK] dual-view forward: z1/z2 [{bsz},{SSL_PROJECTION_DIM}] with unit "
          f"L2 norm; click/install logits [{bsz}]", flush=True)
    results["projection_shape_ok"] = True
    results["projection_l2_norm_ok"] = True
    results["logits_shape_ok"] = True

    # ---- #20/#21/#22 loss components ---------------------------------------
    loss, comp = ssl_joint_loss(
        output, batch, alpha=SSL_ALPHA,
        lambda_click=cfg.lambda_click, lambda_install=cfg.lambda_install,
        temperature=SSL_TEMPERATURE, return_diagnostics=True,
    )
    sup = float(comp["supervised"].item())
    con = float(comp["contrastive"].item())
    tot = float(loss.item())
    if not (math.isfinite(sup) and math.isfinite(con) and math.isfinite(tot)):
        _fail(f"non-finite loss components: sup {sup}, contrastive {con}")  # #20/#21
    recombined = SSL_ALPHA * sup + (1.0 - SSL_ALPHA) * con
    if abs(tot - recombined) > 1e-6 * max(1.0, abs(recombined)):
        _fail(f"joint loss {tot} != {SSL_ALPHA} * {sup} + "
              f"{1.0 - SSL_ALPHA} * {con} = {recombined}")                # #22
    print(f"  [OK] joint loss = {SSL_ALPHA} * supervised + {1.0 - SSL_ALPHA} * "
          f"contrastive: {tot:.6f} = {SSL_ALPHA}*{sup:.6f} + "
          f"{1.0 - SSL_ALPHA}*{con:.6f} (finite)", flush=True)
    results["losses_finite"] = True
    results["joint_recombination_ok"] = True
    results["smoke_supervised_loss"] = sup
    results["smoke_contrastive_loss"] = con

    # ---- #23/#24 one local step: params update; both paths backprop --------
    optimizer = torch.optim.AdamW(
        model_a.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    before_step = s10b._cpu_state(model_a)
    optimizer.zero_grad()
    loss.backward()
    grad_names = {n for n, p in model_a.named_parameters() if p.grad is not None}
    all_named = {n for n, _ in model_a.named_parameters()}
    towers_gates_experts = {n for n in all_named
                            if n.startswith(("mmoe.towers", "mmoe.gates", "mmoe.experts"))}
    projection = {n for n in all_named if n.startswith("projection")}
    backbone = {n for n in all_named if n.startswith("mmoe.backbone")}
    if grad_names != all_named:
        missing = sorted(all_named - grad_names)[:5]
        _fail(f"joint backward left {len(all_named - grad_names)} parameters without "
              f"gradients (e.g. {missing})")
    # supervised-only pass: heads+backbone receive gradients
    optimizer.zero_grad()
    sup_out = model_a.supervised_forward(batch, return_diagnostics=False)
    sup_only = mmoe_loss(sup_out, batch,
                         lambda_click=cfg.lambda_click,
                         lambda_install=cfg.lambda_install)["total"]
    sup_only.backward()
    sup_names = {n for n, p in model_a.named_parameters() if p.grad is not None}
    # contrastive-only pass: projection+backbone receive gradients; heads do NOT
    optimizer.zero_grad()
    con_out = model_a.forward_views(aug.augment(batch), aug.augment(batch))
    con_only = nt_xent_loss(con_out["z1"], con_out["z2"], temperature=SSL_TEMPERATURE)
    con_only.backward()
    con_names = {n for n, p in model_a.named_parameters() if p.grad is not None}
    optimizer.step()
    after_step = s10b._cpu_state(model_a)
    params_updated = [k for k in before_step if not torch.equal(before_step[k], after_step[k])]
    if not params_updated:
        _fail("one local training step changed no parameters")            # #23
    path_checks = (
        towers_gates_experts <= sup_names          # supervised reaches task heads
        and backbone <= sup_names                  # ... and the shared backbone
        and not (projection & sup_names)           # supervised-only path never uses it
        and projection <= con_names                # contrastive reaches projection
        and backbone <= con_names                  # ... and the shared backbone
        and not (towers_gates_experts & con_names) # contrastive NEVER touches task heads
    )
    if not path_checks:
        detail = {
            "sup_reaches_heads": towers_gates_experts <= sup_names,
            "sup_reaches_backbone": backbone <= sup_names,
            "sup_avoids_projection": not (projection & sup_names),
            "con_reaches_projection": projection <= con_names,
            "con_reaches_backbone": backbone <= con_names,
            "con_avoids_heads": not (towers_gates_experts & con_names),
        }
        _fail(f"supervised/contrastive gradient-path separation check failed: {detail}")
    print(f"  [OK] one local AdamW step updated {len(params_updated)}/{len(before_step)} "
          f"state entries", flush=True)
    print(f"  [OK] gradient paths: joint objective -> all {len(all_named)} parameters; "
          f"supervised-only -> task heads + backbone (projection head untouched); "
          f"contrastive-only -> projection + backbone ONLY (task heads untouched: "
          f"projection is a pure add-on)", flush=True)
    results["one_step_updates_params"] = len(params_updated)
    results["both_paths_contribute"] = True
    del optimizer, model_a

    # ---- #27 FedAvg accepts SSL states -------------------------------------
    import copy

    state_a = copy.deepcopy(sd_a)
    state_b = copy.deepcopy(sd_a)
    probe_key = "projection.2.weight"
    state_b[probe_key] = state_b[probe_key] + 1.0  # known perturbation
    counts = [621_801, 59_999]  # real client_00/client_02 sizes as weights
    weights = fedavg_weights_from_counts(counts)
    if abs(float(sum(weights)) - 1.0) > 1e-9:
        _fail("FedAvg weights do not sum to 1")
    merged = fedavg_state_dicts([state_a, state_b], counts)
    if len(merged) != EXPECTED_SSL_STATE_ENTRIES:
        _fail(f"FedAvg output has {len(merged)} entries, expected {EXPECTED_SSL_STATE_ENTRIES}")
    got_probe = float(merged[probe_key][0, 0].item())
    orig_probe = float(state_a[probe_key][0, 0].item())
    pert_probe = float(state_b[probe_key][0, 0].item())
    expected_probe = float(weights[0]) * orig_probe + float(weights[1]) * pert_probe
    if abs(got_probe - expected_probe) > 1e-6:
        _fail(f"FedAvg weighting wrong on {probe_key}: {got_probe} != {expected_probe}")
    if s10b._count_nonfinite(merged):
        _fail("FedAvg produced non-finite parameters")
    print(f"  [OK] sample-weighted FedAvg accepts SSL state dicts: {len(merged)} entries, "
          f"weights sum 1.0, weighted mean verified exactly on a probe entry", flush=True)
    results["fedavg_ssl_ok"] = True

    # ---- #30 checkpoint save/reload bit-identical --------------------------
    stage_dir = project_root / STAGE11_DIR_RELPATH
    roundtrip_path = stage_dir / "preflight_roundtrip_tmp.pt"
    payload = checkpoint_payload(
        state=sd_a, round_number=1, best_logloss=float("inf"), best_round=None,
        history=[], cfg=cfg, split=split, feature_set=fs, kind="preflight_roundtrip",
    )
    s10b._atomic_torch_save(payload, roundtrip_path)
    reloaded = torch.load(roundtrip_path, map_location="cpu", weights_only=False)
    if reloaded.get("stage") != PROTOCOL_ID:
        _fail("round-trip payload is not a Stage 11 checkpoint")
    fresh = cold_ssl_model(fs)
    load_result = fresh.load_state_dict(reloaded["model_state_dict"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        _fail("round-trip strict reload reported missing/unexpected keys")
    fresh_sd = s10b._cpu_state(fresh)
    exact = all(torch.equal(fresh_sd[k], reloaded["model_state_dict"][k]) for k in fresh_sd)
    if not exact:
        _fail("checkpoint round-trip is not bit-identical")
    roundtrip_path.unlink(missing_ok=True)
    if roundtrip_path.exists():
        _fail("temporary preflight checkpoint file was not removed")
    print(f"  [OK] checkpoint save/reload bit-identical ({EXPECTED_SSL_STATE_ENTRIES} "
          f"entries, strict reload into a fresh cold model); temp file removed", flush=True)
    results["checkpoint_roundtrip_ok"] = True
    del fresh

    # ---- #8 official test is never loaded ----------------------------------
    print("  official test data: NOT loaded, NOT evaluated (no test code path in "
          "this script)", flush=True)
    results["official_test_accessed"] = False
    return results


def preflight_11(project_root: Path, dataset, split, cfg, device) -> dict:
    """Integrity (1-11, 31), data layer (2-7) and SSL integration smoke (12-30)."""
    print("=== pre-run integrity verification ===", flush=True)
    snapshot = s10b.snapshot_protected_artifacts(project_root)
    violations = verify_protected_artifacts(snapshot, project_root)
    if violations:
        _fail("protected artifacts do not match frozen values: " + "; ".join(violations))
    print(f"  Stage 2 artifact hash:     {snapshot['stage2_artifact_hash']} OK", flush=True)
    print(f"  split train hash:          {snapshot['split_train_index_hash']} OK", flush=True)
    print(f"  split validation hash:     {snapshot['split_validation_index_hash']} OK", flush=True)
    print(f"  Stage 6 checkpoint sha256: {snapshot['stage6_checkpoint_sha256'][:16]}... OK", flush=True)
    fingerprint = _fingerprint11(project_root)
    n10b = len(STAGE10B_OUTPUTS)
    print(f"  protected files sha256+size+mtime recorded: {len(fingerprint)} "
          f"(Stage 1-9 guards + {n10b} Stage 10B baseline outputs)", flush=True)
    results: dict[str, object] = {"fingerprint_file_count": len(fingerprint)}

    print("\n=== data layer verification ===", flush=True)
    counts = verify_data_layer_11(dataset, split, project_root)
    positions = s10b.load_partition_positions(project_root / s10b.NPZ_RELPATH)

    print("\n=== SSL integration smoke (real data, one CUDA batch) ===", flush=True)
    results["ssl_smoke"] = ssl_integration_smoke(
        dataset, split, positions, cfg, device, project_root
    )

    # ---- #31 fingerprints unchanged after the preflight itself ------------
    violations = _verify_fingerprint11(fingerprint, project_root)
    violations += verify_protected_artifacts(snapshot, project_root)
    if violations:
        for violation in violations:
            print(f"  VIOLATION: {violation}", flush=True)
        _fail("protected artifacts changed during the preflight")
    print(f"  [OK] protected artifacts unchanged through the preflight: all "
          f"{len(fingerprint)} guarded files byte-identical (sha256+size+mtime)",
          flush=True)

    results["client_counts"] = counts
    results["integrity_snapshot"] = snapshot
    results["fingerprint_before"] = fingerprint
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="'auto' (CUDA if available) or 'cpu'")
    parser.add_argument("--log_every", type=int, default=200,
                        help="client progress log interval (batches)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from stage11_latest.pt after an interrupted run")
    parser.add_argument("--allow_restart", action="store_true",
                        help="discard any existing Stage 11 outputs and start fresh")
    parser.add_argument("--preflight_only", action="store_true",
                        help="run integrity + data verification + SSL integration smoke, "
                             "then stop (no training)")
    args = parser.parse_args()

    cfg = get_federated_config()
    stage_dir = PROJECT_ROOT / STAGE11_DIR_RELPATH
    best_ckpt = PROJECT_ROOT / BEST_CKPT_RELPATH
    final_ckpt = PROJECT_ROOT / FINAL_CKPT_RELPATH
    latest_ckpt = PROJECT_ROOT / LATEST_CKPT_RELPATH
    result_json = PROJECT_ROOT / RESULT_JSON_RELPATH

    print("=== STAGE 11: FEDERATED TRANSFORMER + MMOE + SSL (real data) ===", flush=True)
    print(f"start: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", flush=True)
    print(f"protocol: 10 clients, alpha {cfg.alpha}, seed {cfg.seed}, {cfg.rounds} rounds, "
          f"1 local epoch, batch {cfg.local_batch_size}, AdamW lr {cfg.learning_rate} "
          f"wd {cfg.weight_decay}, sample_weighted_fedavg, cold seed-42 SSL init; "
          f"joint objective {SSL_ALPHA}*(BCE click + BCE install) + "
          f"{1.0 - SSL_ALPHA}*NT-Xent(T={SSL_TEMPERATURE}) over two corrupted views "
          f"(rate {SSL_CORRUPTION_RATE})", flush=True)

    if final_ckpt.exists() or best_ckpt.exists():
        if args.resume and latest_ckpt.exists():
            pass  # resume path below
        elif args.allow_restart:
            for path in (best_ckpt, final_ckpt, latest_ckpt, result_json):
                path.unlink(missing_ok=True)
            print("  existing Stage 11 outputs discarded (--allow_restart)", flush=True)
        else:
            print(f"\nSTOP: Stage 11 outputs already exist under {stage_dir}")
            print("Use --resume to continue an interrupted run, or --allow_restart to "
                  "discard them.")
            return 1

    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

    # ------------------------------------------------------------------
    # 0. Integrity + environment + data layer (+ SSL smoke in preflight)
    # ------------------------------------------------------------------
    feature_set = FeatureSet.load(PROJECT_ROOT / "artifacts" / "preprocessing")
    if feature_set.artifact_hash != s10b.STAGE2_ARTIFACT_HASH:
        _fail(f"Stage 2 artifact hash {feature_set.artifact_hash} "
              f"!= {s10b.STAGE2_ARTIFACT_HASH}")
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

    preflight_results = preflight_11(PROJECT_ROOT, dataset=dataset, split=split,
                                     cfg=cfg, device=device)
    counts = preflight_results["client_counts"]
    positions = s10b.load_partition_positions(PROJECT_ROOT / s10b.NPZ_RELPATH)
    integrity_snapshot = preflight_results["integrity_snapshot"]
    fingerprint_before = preflight_results["fingerprint_before"]

    if args.preflight_only:
        print("\nRESULT: PREFLIGHT ONLY - stopping before the Stage 11 run as requested")
        return 0

    # ------------------------------------------------------------------
    # 1. Global model: cold seed-42 SSL init (or deterministic resume)
    # ------------------------------------------------------------------
    started_all = time.perf_counter()
    monitor = s10b.RSSMonitor(interval=10.0)
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
            _fail("resume state is not a Stage 11 payload")
        start_round = int(resume_payload["round"]) + 1
        if start_round > cfg.rounds:
            _fail("resume state says all rounds are already complete")
        global_model = cold_ssl_model(feature_set, seed=cfg.seed).to(device)
        global_model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        history = list(resume_payload["history"])
        best_round = resume_payload["best_round"]
        best_val_install_logloss = float(resume_payload["best_validation_install_logloss"])
        print(f"\n=== RESUME: continuing from round {start_round} "
              f"(best so far: round {best_round}, Install LogLoss "
              f"{best_val_install_logloss:.6f}) ===", flush=True)
    else:
        print("\n=== 1. cold seed-42 SSL global initialization (no checkpoint loads) ===",
              flush=True)
        global_model = cold_ssl_model(feature_set, seed=cfg.seed).to(device)
        if sum(p.numel() for p in global_model.parameters()) != EXPECTED_SSL_PARAM_COUNT:
            _fail("global parameter count does not match the frozen SSL architecture")
        if len(global_model.state_dict()) != EXPECTED_SSL_STATE_ENTRIES:
            _fail("global state entry count does not match the frozen SSL architecture")
        initial_probe = cold_ssl_model(feature_set, seed=cfg.seed)
        global_sd_cpu = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        if any(not torch.equal(initial_probe.state_dict()[k], global_sd_cpu[k])
               for k in global_sd_cpu):
            _fail("seed-42 SSL initialization is not deterministic across two creations")
        del initial_probe
        if s10b._count_nonfinite(global_sd_cpu):
            _fail("non-finite initial global parameters")
        print(f"  parameter count {EXPECTED_SSL_PARAM_COUNT:,}; state entries "
              f"{len(global_sd_cpu)}", flush=True)
        print("  determinism: two seed-42 SSL creations identical: True", flush=True)
        print("  cold init: no checkpoint loaded (Stage 6 / Stage 8 / Stage 10B untouched)",
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
            record, state = train_one_client_ssl(
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
                f"  r{round_number} {cid} done: {record['samples']:,} rows | joint loss "
                f"{record['local_total_loss']:.4f} (click {record['local_click_loss']:.4f}, "
                f"install {record['local_install_loss']:.4f}, contrastive "
                f"{record['local_contrastive_loss']:.4f}) | {record['duration_seconds']:.1f}s",
                flush=True,
            )

        # ---- per-round participation + integrity checks ----------------
        if len(client_records) != 10:
            _fail(f"round {round_number}: {len(client_records)} clients participated, "
                  f"expected 10")
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
        if len(aggregated) != EXPECTED_SSL_STATE_ENTRIES:
            _fail(f"round {round_number}: aggregated state has {len(aggregated)} entries, "
                  f"expected {EXPECTED_SSL_STATE_ENTRIES}")

        global_model.load_state_dict(aggregated, strict=True)
        if s10b._count_nonfinite({k: v.detach().cpu() for k, v in global_model.state_dict().items()}):
            _fail(f"round {round_number}: non-finite global parameters after aggregation")

        # ---- global validation (frozen split only) ----------------------
        val_metrics = global_validation_ssl(
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
                ),
                best_ckpt,
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
            ),
            latest_ckpt,
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
            "ssl": {
                "alpha": SSL_ALPHA,
                "temperature": SSL_TEMPERATURE,
                "projection_dim": SSL_PROJECTION_DIM,
                "corruption_rate": SSL_CORRUPTION_RATE,
            },
            "seed": cfg.seed,
            "partition_hash": s10b.EXPECTED_PARTITION_HASH,
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
        s10b._atomic_json_write(result, result_json)

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
        _fail("best checkpoint payload is not a Stage 11 best checkpoint")
    fresh = cold_ssl_model(feature_set, seed=cfg.seed)
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

    reloaded_metrics = global_validation_ssl(
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
    violations += _verify_fingerprint11(fingerprint_before, PROJECT_ROOT)
    if violations:
        for violation in violations:
            print(f"  VIOLATION: {violation}", flush=True)
        return 1
    print(f"  protected artifacts unchanged: all {len(fingerprint_before)} guarded files "
          f"byte-identical (sha256+size+mtime; includes Stage 10B baseline outputs)",
          flush=True)

    final_result = json.loads(result_json.read_text(encoding="utf-8"))
    final_result["status"] = "complete"
    final_result["reload_verified"] = True
    final_result["reloaded_metrics_match"] = True
    final_result["peak_rss_mib_overall"] = peak_rss_total
    final_result["completed_at"] = datetime.now(timezone.utc).isoformat()
    s10b._atomic_json_write(final_result, result_json)

    print("\n=== STAGE 11 COMPLETE ===", flush=True)
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
