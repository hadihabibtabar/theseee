"""Stage 8: sampler-protocol diagnostic (historical vs corrected reshuffle).

Background: the frozen Stage 6 run INTENDED per-epoch reshuffling via
``Trainer.fit`` -> ``sampler.set_epoch(...)``, but ``make_subset_loader``
attaches the ``PartShuffledBatchSampler`` as the DataLoader's
``batch_sampler``; ``loader.sampler`` is therefore a plain ``SequentialSampler``
without ``set_epoch`` and the call was a silent no-op. All three Stage 6
epochs reused the epoch-1 batch arrangement.

This module provides read-only, deterministic utilities to:

1. reproduce the historical arrangement (Protocol H): one fixed batch
   sequence derived from ``seed + 0`` regardless of epoch;
2. produce the corrected arrangement (Protocol C): deterministic,
   epoch-dependent batch sequences (``seed + epoch``);
3. fingerprint batch arrangements from ACTUAL batch indices (first/last
   sample of every batch + a hashlib digest over the exact index sequence),
   so the comparison is proven from data, not just code inspection;
4. run a tiny controlled training comparison between the two protocols
   (same initialization, seed, data, loss) and report trajectory deltas.

No function here loads the official test set; no protected artifact is
touched. This is a diagnostic only — it must NOT be used to claim that one
protocol outperforms the other.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from recsys23_fedrec.data import ProcessedRecSysDataset, collate_train
from recsys23_fedrec.preprocessing.config import SEED
from recsys23_fedrec.training.loaders import SubsetDataset
from recsys23_fedrec.training.sampler import PartShuffledBatchSampler

__all__ = [
    "ArrangementFingerprint",
    "protocol_arrangement",
    "fingerprint_from_batches",
    "fingerprint_historical",
    "fingerprint_corrected",
    "TinyComparisonReport",
    "run_tiny_sampler_comparison",
]


# ---------------------------------------------------------------------------
# Batch arrangements (proven from actual indices)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArrangementFingerprint:
    """Cryptographic + human-checkable fingerprint of a batch arrangement."""

    epoch: int
    protocol: str  # "H" (historical) or "C" (corrected)
    n_batches: int
    batch_size: int
    n_samples: int
    first_batch: list[int]      # exact indices of batch 0 (bounded)
    first_batch_first: int      # first index of batch 0
    first_batch_last: int       # last index of batch 0
    last_batch_first: int       # first index of the last batch
    arrangement_sha256: str     # digest over ALL batch indices, exact order

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "protocol": self.protocol,
            "n_batches": self.n_batches,
            "batch_size": self.batch_size,
            "n_samples": self.n_samples,
            "first_batch": self.first_batch,
            "first_batch_first": self.first_batch_first,
            "first_batch_last": self.first_batch_last,
            "last_batch_first": self.last_batch_first,
            "arrangement_sha256": self.arrangement_sha256,
        }


def _arrangement_digest(batches: list[list[int]]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(np.asarray(batch, dtype=np.int64).tobytes())
    return digest.hexdigest()


def protocol_arrangement(
    subset_indices: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    protocol: str,
) -> list[list[int]]:
    """Materialize the exact batch-index arrangement for one epoch.

    Protocol "H" (historical Stage 6): the sampler was constructed once with
    its default ``epoch=0`` and ``set_epoch`` never reached it, so EVERY
    epoch used the ``seed + 0`` arrangement.

    Protocol "C" (corrected Stage 8): ``set_epoch(epoch - 1)`` is applied, so
    each epoch uses ``seed + (epoch - 1)``.
    """
    sampler_epoch = 0 if protocol == "H" else epoch - 1
    sampler = PartShuffledBatchSampler(
        subset_indices=np.asarray(subset_indices, dtype=np.int64),
        batch_size=batch_size,
        seed=seed,
        epoch=sampler_epoch,
    )
    return [list(map(int, batch)) for batch in sampler]


def fingerprint_from_batches(
    batches: list[list[int]], *, epoch: int, protocol: str
) -> ArrangementFingerprint:
    """Fingerprint an actual batch arrangement (exact indices, exact order)."""
    if not batches:
        raise ValueError("cannot fingerprint an empty arrangement")
    first_batch = batches[0][:16]  # bounded excerpt for the report
    return ArrangementFingerprint(
        epoch=epoch,
        protocol=protocol,
        n_batches=len(batches),
        batch_size=len(batches[0]),
        n_samples=sum(len(b) for b in batches),
        first_batch=first_batch,
        first_batch_first=batches[0][0],
        first_batch_last=batches[0][-1],
        last_batch_first=batches[-1][0],
        arrangement_sha256=_arrangement_digest(batches),
    )


def fingerprint_historical(
    subset_indices: np.ndarray, *, batch_size: int, seed: int, epoch: int
) -> ArrangementFingerprint:
    batches = protocol_arrangement(
        subset_indices, batch_size=batch_size, seed=seed, epoch=epoch, protocol="H"
    )
    return fingerprint_from_batches(batches, epoch=epoch, protocol="H")


def fingerprint_corrected(
    subset_indices: np.ndarray, *, batch_size: int, seed: int, epoch: int
) -> ArrangementFingerprint:
    batches = protocol_arrangement(
        subset_indices, batch_size=batch_size, seed=seed, epoch=epoch, protocol="C"
    )
    return fingerprint_from_batches(batches, epoch=epoch, protocol="C")


# ---------------------------------------------------------------------------
# Tiny controlled training comparison
# ---------------------------------------------------------------------------
@dataclass
class TinyComparisonReport:
    """Outcome of the tiny historical-vs-corrected training comparison."""

    protocol: str
    batch_fingerprints: dict[int, dict] = field(default_factory=dict)
    epoch_losses: dict[int, dict[str, float]] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    param_delta_l2: float | None = None       # vs its own initialization
    param_delta_vs_other: float | None = None  # vs the other protocol's model
    deterministic_repeat: bool | None = None

    def to_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "batch_fingerprints": self.batch_fingerprints,
            "epoch_losses": self.epoch_losses,
            "val_metrics": self.val_metrics,
            "param_delta_l2": self.param_delta_l2,
            "param_delta_vs_other": self.param_delta_vs_other,
            "deterministic_repeat": self.deterministic_repeat,
        }


def _tiny_batches(
    dataset: ProcessedRecSysDataset,
    subset_indices: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    protocol: str,
    n_batches: int,
) -> Iterator[tuple[list[list[int]], list[dict]]]:
    """Yield (batches, samples) for the first ``n_batches`` of one epoch."""
    arrangement = protocol_arrangement(
        subset_indices, batch_size=batch_size, seed=seed, epoch=epoch, protocol=protocol
    )[:n_batches]
    subset = SubsetDataset(dataset, subset_indices)
    for batch_positions in arrangement:
        yield arrangement, [subset[int(pos)] for pos in batch_positions]


def _supervised_forward(model: torch.nn.Module, batch: dict) -> dict:
    """Single-view forward with per-type signatures (mirrors the runner).

    ``TransformerBaseline.forward`` uses ``return_tokens`` (Stage 4), the
    SSL wrappers expose ``supervised_forward`` (Stage 7/8), and the plain
    ``TransformerMMoE`` takes ``return_diagnostics`` (Stage 5).
    """
    from recsys23_fedrec.models import TransformerBaseline

    if isinstance(model, TransformerBaseline):
        return model(batch)
    if hasattr(model, "supervised_forward"):
        return model.supervised_forward(batch, return_diagnostics=False)
    return model(batch, return_diagnostics=False)


def _train_tiny(
    config,  # ExperimentConfig
    model: torch.nn.Module,
    dataset: ProcessedRecSysDataset,
    subset_indices: np.ndarray,
    *,
    batch_size: int,
    n_batches: int,
    epochs: int,
    protocol: str,
    val_indices: np.ndarray,
    device: torch.device,
) -> tuple[TinyComparisonReport, dict]:
    """Train a tiny model under one protocol; return (report, model state)."""
    from recsys23_fedrec.experiments.runner import compute_joint_loss

    report = TinyComparisonReport(protocol=protocol)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    init_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

    for epoch in range(1, epochs + 1):
        totals = {"total": 0.0, "install": 0.0, "click": 0.0, "contrastive": 0.0}
        n_steps = 0
        fingerprint: ArrangementFingerprint | None = None
        for arrangement, samples in _tiny_batches(
            dataset,
            subset_indices,
            batch_size=batch_size,
            seed=config.seed,
            epoch=epoch,
            protocol=protocol,
            n_batches=n_batches,
        ):
            if fingerprint is None:
                fingerprint = fingerprint_from_batches(arrangement, epoch=epoch, protocol=protocol)
                report.batch_fingerprints[epoch] = fingerprint.to_dict()
            batch = collate_train(samples)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            if config.use_ssl:
                output = model(batch)
            else:
                output = _supervised_forward(model, batch)
            losses = compute_joint_loss(output, batch, config)
            losses["total"].backward()
            optimizer.step()
            totals["total"] += float(losses["total"])
            totals["install"] += float(losses["install"])
            if losses.get("click") is not None:
                totals["click"] += float(losses["click"])
            if "contrastive" in losses:
                totals["contrastive"] += float(losses["contrastive"])
            n_steps += 1
        report.epoch_losses[epoch] = {k: v / max(n_steps, 1) for k, v in totals.items()}

    # ---- validation on the same tiny validation slice -------------------
    from recsys23_fedrec.training.metrics import binary_logloss, roc_auc

    model.eval()
    install_logits, install_targets = [], []
    with torch.no_grad():
        subset = SubsetDataset(dataset, val_indices)
        for start in range(0, len(val_indices), batch_size):
            stop = min(start + batch_size, len(val_indices))
            samples = [subset[local] for local in range(start, stop)]
            batch = collate_train(samples)
            batch = {k: v.to(device) for k, v in batch.items()}
            output = _supervised_forward(model, batch)
            install_logits.append(output["install_logit"].float().cpu())
            install_targets.append(batch["install"].float().cpu())
    all_logits = torch.cat(install_logits)
    all_targets = torch.cat(install_targets)
    report.val_metrics = {
        "n_val_samples": int(len(all_targets)),
        "val_install_logloss": float(binary_logloss(all_logits, all_targets)),
        "val_install_auc": float(roc_auc(all_logits, all_targets)),
    }

    final_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    report.param_delta_l2 = float(
        torch.sqrt(
            sum(
                (final_state[n].float() - init_state[n].float()).pow(2).sum()
                for n in init_state
            )
        )
    )
    return report, final_state


def run_tiny_sampler_comparison(
    config,
    feature_set,
    dataset: ProcessedRecSysDataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    *,
    batch_size: int = 1024,
    n_batches: int = 4,
    epochs: int = 2,
    device: torch.device | str = "cpu",
    seed: int = SEED,
) -> dict:
    """Compare historical vs corrected sampler on a tiny controlled run.

    Both variants use identical initialization (same ``torch.manual_seed``),
    optimizer, data, and loss; ONLY the per-epoch batch arrangement differs.
    Returns a dict with both reports + direct deltas; purely informational.
    """
    from recsys23_fedrec.experiments.runner import build_model

    device = torch.device(device) if isinstance(device, str) else device
    protocol_reports: dict[str, TinyComparisonReport] = {}
    protocol_states: dict[str, dict] = {}

    for protocol in ("H", "C"):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = build_model(config, feature_set).to(device)
        report, state = _train_tiny(
            config,
            model,
            dataset,
            train_indices,
            batch_size=batch_size,
            n_batches=n_batches,
            epochs=epochs,
            protocol=protocol,
            val_indices=val_indices,
            device=device,
        )
        protocol_reports[protocol] = report
        protocol_states[protocol] = state

    # ---- cross-protocol parameter delta ---------------------------------
    h_state, c_state = protocol_states["H"], protocol_states["C"]
    cross_delta = float(
        torch.sqrt(
            sum(
                (h_state[n].float() - c_state[n].float()).pow(2).sum()
                for n in h_state
            )
        )
    )
    for report in protocol_reports.values():
        report.param_delta_vs_other = cross_delta

    # ---- determinism: repeat Protocol H once and compare -----------------
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    repeat_model = build_model(config, feature_set).to(device)
    repeat_report, repeat_state = _train_tiny(
        config,
        repeat_model,
        dataset,
        train_indices,
        batch_size=batch_size,
        n_batches=n_batches,
        epochs=epochs,
        protocol="H",
        val_indices=val_indices,
        device=device,
    )
    repeat_equal = all(
        torch.equal(repeat_state[n], h_state[n]) for n in h_state
    )
    protocol_reports["H"].deterministic_repeat = repeat_equal
    repeat_report.param_delta_vs_other = cross_delta  # keep semantics uniform

    return {
        "protocols": {k: v.to_dict() for k, v in protocol_reports.items()},
        "cross_protocol_param_delta_l2": cross_delta,
        "relative_to_init_h": protocol_reports["H"].param_delta_l2,
        "relative_to_init_c": protocol_reports["C"].param_delta_l2,
        "repeat_of_H_matches_H": repeat_equal,
        "repeat_report": repeat_report.to_dict(),
    }
