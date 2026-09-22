"""Stage 8: reusable experiment runner for the controlled A/B/C/D runs.

Responsibilities per experiment (spec 8.8):

1. seed the project RNGs, 2. build the model from :class:`ExperimentConfig`,
3. build train/validation subset loaders over the hash-verified split,
4. train with the controlled protocol, 5. validate every epoch,
6. select best epoch by validation Install LogLoss, 7. save a dedicated
atomic checkpoint, 8. record metrics, 9. verify checkpoint reload, 10. write
the experiment result JSON.

The runner reuses the Stage 6 data layer (``make_subset_loader``), metrics
(``binary_logloss``/``roc_auc``), and loss functions verbatim — no metric
redefinition. Two fair-protocol notes:

* Per-epoch reshuffle: the Stage 6 Trainer attempted a per-epoch sampler
  reset via ``loader.sampler``, but ``make_subset_loader`` attaches the
  sampler as ``batch_sampler`` so the reset never fired; all Stage 6 epochs
  used the epoch-1 batch arrangement. This runner calls ``set_epoch`` on the
  batch_sampler directly — the intended Stage 6 behavior.
* Experiment B is NOT retrained here by default: the canonical Stage 6
  checkpoint/result is reused (source = Stage 6 canonical baseline).

The official test set is never touched by this runner.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn

from recsys23_fedrec.data import ProcessedRecSysDataset
from recsys23_fedrec.data.feature_schema import FeatureSet
from recsys23_fedrec.experiments.config import (
    EXPERIMENT_A,
    EXPERIMENT_B,
    EXPERIMENT_C,
    EXPERIMENT_D,
    ExperimentConfig,
)
from recsys23_fedrec.models import (
    SSLConfig,
    SSLTransformerBaseline,
    SSLTransformerMMoE,
    TransformerBaseline,
    TransformerMMoE,
)
from recsys23_fedrec.models.mmoe import mmoe_loss
from recsys23_fedrec.models.transformer import install_loss
from recsys23_fedrec.ssl import (
    FeatureCorruptionAugmentation,
    nt_xent_loss,
)
from recsys23_fedrec.training.loaders import make_subset_loader
from recsys23_fedrec.training.metrics import binary_logloss, roc_auc
from recsys23_fedrec.training.split import load_split

__all__ = ["EpochRecord", "ExperimentResult", "ExperimentRunner", "build_model"]


# ---------------------------------------------------------------------------
# Model construction (dispatch on the experiment config)
# ---------------------------------------------------------------------------
def build_model(config: ExperimentConfig, feature_set: FeatureSet) -> nn.Module:
    """Construct the experiment's model from the frozen architecture defaults.

    All architectures share the frozen Stage 4 backbone (128/8/6/128/0.1);
    MMoE adds the frozen Stage 5 head; SSL adds the Stage 7 projection head
    and the feature-corruption augmentation (rate from the config).
    """
    torch.manual_seed(config.seed)
    if config.use_ssl:
        augmentation = FeatureCorruptionAugmentation(
            feature_set, corruption_rate=config.corruption_rate, seed=config.seed
        )
        ssl_config = SSLConfig(projection_dim=64, temperature=config.temperature)
    else:
        augmentation, ssl_config = None, None

    if config.experiment_name == EXPERIMENT_A:
        return TransformerBaseline(feature_set)  # install head only
    if config.experiment_name == EXPERIMENT_B:
        return TransformerMMoE(feature_set)
    if config.experiment_name == EXPERIMENT_C:
        return SSLTransformerBaseline(
            feature_set, ssl_config=ssl_config, augmentations=augmentation
        )
    if config.experiment_name == EXPERIMENT_D:
        return SSLTransformerMMoE(
            feature_set, ssl_config=ssl_config, augmentations=augmentation
        )
    raise ValueError(f"unsupported experiment {config.experiment_name!r}")


def _supervised_keys(config: ExperimentConfig) -> tuple[str, ...]:
    if config.use_mmoe:
        return ("click_logit", "install_logit")
    return ("install_logit",)


def compute_supervised_loss(
    output: Mapping[str, Tensor], batch: Mapping[str, Tensor], config: ExperimentConfig
) -> dict[str, Tensor]:
    """Dispatch the correct supervised loss for the experiment (raw logits).

    A: Stage 4 ``install_loss`` (BCEWithLogitsLoss on the install logit).
    B/D: Stage 5 ``mmoe_loss`` (click + install). C: Stage 4 install loss on
    each view, averaged. Never applies sigmoid; never reweights classes.
    """
    if config.use_mmoe:
        if "view1" in output:  # D: supervised term averaged over both views
            sup1 = mmoe_loss(output["view1"], batch)
            sup2 = mmoe_loss(output["view2"], batch)
            return {
                "click": 0.5 * (sup1["click"] + sup2["click"]),
                "install": 0.5 * (sup1["install"] + sup2["install"]),
                "total": 0.5 * (sup1["total"] + sup2["total"]),
            }
        return mmoe_loss(output, batch)
    if "view1" in output:  # C: install loss averaged over both views
        l1 = install_loss(output["view1"], batch)
        l2 = install_loss(output["view2"], batch)
        return {"click": None, "install": 0.5 * (l1 + l2), "total": 0.5 * (l1 + l2)}
    single = install_loss(output, batch)
    return {"click": None, "install": single, "total": single}


def compute_contrastive_loss(output: Mapping[str, Tensor], config: ExperimentConfig) -> Tensor:
    """NT-Xent over the dual-view projections (C and D only)."""
    if not config.use_ssl:
        raise ValueError(f"{config.experiment_name} does not use SSL")
    return nt_xent_loss(output["z1"], output["z2"], temperature=config.temperature)


def compute_joint_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    config: ExperimentConfig,
) -> dict[str, Tensor]:
    """``alpha * supervised + (1 - alpha) * contrastive`` (SSL experiments).

    For non-SSL experiments the joint loss is the supervised loss itself.
    """
    sup = compute_supervised_loss(output, batch, config)
    if not config.use_ssl:
        return sup
    contrastive = compute_contrastive_loss(output, config)
    total = config.alpha * sup["total"] + (1.0 - config.alpha) * contrastive
    return {
        "click": sup.get("click"),
        "install": sup["install"],
        "total": total,
        "contrastive": contrastive,
    }


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class EpochRecord:
    """Per-epoch metrics (spec 8.9)."""

    epoch: int
    train_total_loss: float
    train_supervised_loss: float
    train_contrastive_loss: float | None
    train_click_loss: float | None
    train_install_loss: float
    val_install_logloss: float
    val_install_auc: float
    val_click_logloss: float | None
    val_click_auc: float | None
    duration_seconds: float
    learning_rate: float
    n_train_batches: int
    n_val_batches: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ExperimentResult:
    """Serialized outcome of one experiment run."""

    experiment_name: str
    config: dict
    epochs: list[dict]
    best_epoch: int | None
    best_val_install_logloss: float | None
    best_val_install_auc: float | None
    best_val_click_logloss: float | None
    best_val_click_auc: float | None
    total_seconds: float
    checkpoint_path: str
    split_hash: str
    reload_verified: bool
    source: str = "stage8_fresh_run"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class ExperimentRunner:
    """Train/validate/checkpoint one controlled Stage 8 experiment."""

    def __init__(
        self,
        config: ExperimentConfig,
        feature_set: FeatureSet,
        train_dataset: ProcessedRecSysDataset,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        checkpoint_path: str | Path,
        *,
        split_hash: str,
        device: torch.device | str = "auto",
        results_path: str | Path | None = None,
        preprocessing_hash: str | None = None,
        validation_index_hash: str | None = None,
        log_every: int = 0,
    ) -> None:
        self.config = config
        self.feature_set = feature_set
        self.preprocessing_hash = preprocessing_hash
        self.log_every = int(log_every)
        self._validation_index_hash = validation_index_hash
        self.train_dataset = train_dataset
        self.train_indices = np.asarray(train_indices, dtype=np.int64)
        self.val_indices = np.asarray(val_indices, dtype=np.int64)
        self.checkpoint_path = Path(checkpoint_path)
        self.results_path = Path(results_path) if results_path else None
        self.split_hash = split_hash
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )

        self.model = build_model(config, feature_set).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.history: list[EpochRecord] = []
        self.best_val_install_logloss = float("inf")
        self.best_epoch: int | None = None
        self._best_state: dict | None = None

    # ------------------------------------------------------------------
    # Data loaders (Stage 6 subset loaders, unchanged)
    # ------------------------------------------------------------------
    def _train_loader(self, epoch: int) -> torch.utils.data.DataLoader:
        loader = make_subset_loader(
            self.train_dataset,
            self.train_indices,
            batch_size=self.config.batch_size,
            shuffle=True,
            seed=self.config.seed,
            pin_memory=(self.device.type == "cuda"),
        )
        # Per-epoch reshuffle: set the epoch on the batch_sampler directly.
        sampler = getattr(loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch - 1)
        return loader

    def _val_loader(self) -> torch.utils.data.DataLoader:
        return make_subset_loader(
            self.train_dataset,
            self.val_indices,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=(self.device.type == "cuda"),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _to_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    @staticmethod
    def _forward_supervised(
        model: nn.Module, batch: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        """Single-view supervised forward with per-type signatures.

        ``TransformerBaseline.forward`` uses ``return_tokens`` (Stage 4
        signature); the MMoE/SSL models accept ``return_diagnostics``.
        """
        if isinstance(model, TransformerBaseline):
            return model(batch)
        return model(batch, return_diagnostics=False)

    # ------------------------------------------------------------------
    # Training pass
    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        totals = {"total": 0.0, "supervised": 0.0, "install": 0.0, "contrastive": 0.0, "click": 0.0}
        n_batches = 0
        for batch in self._train_loader(epoch):
            batch = self._to_device(batch, self.device)
            self.optimizer.zero_grad()
            if self.config.use_ssl:
                output = self.model(batch)  # dual-view forward
            else:
                output = self._forward_supervised(self.model, batch)
            losses = compute_joint_loss(output, batch, self.config)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(
                    f"non-finite training loss at epoch {epoch} batch {n_batches + 1} "
                    f"({self.config.experiment_name}): {losses} — stopping"
                )
            losses["total"].backward()
            self.optimizer.step()

            totals["total"] += float(losses["total"].item())
            totals["supervised"] += float(losses["install"].item()) + (
                float(losses["click"].item()) if losses.get("click") is not None else 0.0
            )
            totals["install"] += float(losses["install"].item())
            if losses.get("click") is not None:
                totals["click"] += float(losses["click"].item())
            if "contrastive" in losses:
                totals["contrastive"] += float(losses["contrastive"].item())
            n_batches += 1
            if self.log_every and n_batches % self.log_every == 0:
                print(
                    f"  epoch {epoch} progress: batch {n_batches} | "
                    f"running total {totals['total'] / n_batches:.4f}",
                    flush=True,
                )
        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Validation pass
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        click_logits: list[Tensor] = []
        install_logits: list[Tensor] = []
        click_targets: list[Tensor] = []
        install_targets: list[Tensor] = []
        click_sum = install_sum = 0.0
        n_seen = 0
        for batch in self._val_loader():
            batch = self._to_device(batch, self.device)
            if self.config.use_ssl:
                output = self.model.supervised_forward(batch, return_diagnostics=False)
            else:
                output = self._forward_supervised(self.model, batch)
            batch_n = int(batch["install"].shape[0])
            install_sum += binary_logloss(output["install_logit"], batch["install"]).item() * batch_n
            if self.config.use_mmoe:
                click_sum += binary_logloss(output["click_logit"], batch["click"]).item() * batch_n
            n_seen += batch_n
            install_logits.append(output["install_logit"].detach().float().cpu())
            install_targets.append(batch["install"].detach().float().cpu())
            if self.config.use_mmoe:
                click_logits.append(output["click_logit"].detach().float().cpu())
                click_targets.append(batch["click"].detach().float().cpu())

        if n_seen == 0:
            raise ValueError("validation loader produced no batches")
        if not torch.isfinite(torch.cat(install_logits)).all():
            raise RuntimeError("non-finite validation install logits — stopping")
        metrics = {
            "val_install_logloss": install_sum / n_seen,
            "val_install_auc": float(roc_auc(torch.cat(install_logits), torch.cat(install_targets))),
            "val_click_logloss": None,
            "val_click_auc": None,
        }
        if self.config.use_mmoe:
            metrics["val_click_logloss"] = click_sum / n_seen
            metrics["val_click_auc"] = float(
                roc_auc(torch.cat(click_logits), torch.cat(click_targets))
            )
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing (atomic write; never touches other files)
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, val_install_auc: float | None = None) -> Path:
        arch = getattr(self.model, "config", None)
        architecture_config = (
            dataclasses.asdict(arch)
            if dataclasses.is_dataclass(arch)
            else {"model_type": self.config.model_type}
        )
        payload = {
            "experiment_name": self.config.experiment_name,
            "experiment": self.config.experiment_name.split("_")[1].upper(),
            "stage": "8-controlled-experiments",
            "stage_number": 8,
            "model_type": self.config.model_type,
            "architecture_config": architecture_config,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optimizer_config": {
                "optimizer": self.config.optimizer,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
            },
            "batch_size": self.config.batch_size,
            "epoch": int(epoch),
            "best_validation_install_logloss": float(self.best_val_install_logloss),
            "val_install_auc": val_install_auc,
            "training_config": self.config.to_dict(),
            "seed": self.config.seed,
            "alpha": self.config.alpha,
            "corruption_rate": self.config.corruption_rate,
            "temperature": self.config.temperature,
            "split_hash": self.split_hash,
            "train_index_hash": self.split_hash,
            "validation_index_hash": self._validation_index_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "sampler_protocol": (
                "stage8-corrected-per-epoch-reshuffle: set_epoch on "
                "PartShuffledBatchSampler (epoch e uses seed + e - 1)"
            ),
            "n_train_rows": int(len(self.train_indices)),
            "n_val_rows": int(len(self.val_indices)),
            "n_train_batches": math.ceil(len(self.train_indices) / self.config.batch_size),
            "n_val_batches": math.ceil(len(self.val_indices) / self.config.batch_size),
            "device": str(self.device),
            "checkpoint_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.checkpoint_path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.checkpoint_path)  # atomic on POSIX & Windows
        finally:
            if os.path.exists(tmp_name):  # pragma: no cover - only on failure
                os.unlink(tmp_name)
        return self.checkpoint_path

    # ------------------------------------------------------------------
    # Reload verification
    # ------------------------------------------------------------------
    def verify_checkpoint_reload(self) -> bool:
        """Load the checkpoint into a FRESH model and compare one val batch."""
        checkpoint = torch.load(self.checkpoint_path, weights_only=False, map_location="cpu")
        fresh = build_model(self.config, self.feature_set)
        fresh.load_state_dict(checkpoint["model_state_dict"])
        fresh = fresh.to(self.device).eval()
        self.model.eval()
        with torch.no_grad():
            for batch in self._val_loader():
                batch = self._to_device(batch, self.device)
                if self.config.use_ssl:
                    out_saved = self.model.supervised_forward(batch, return_diagnostics=False)
                    out_fresh = fresh.supervised_forward(batch, return_diagnostics=False)
                else:
                    out_saved = self._forward_supervised(self.model, batch)
                    out_fresh = self._forward_supervised(fresh, batch)
                for key in _supervised_keys(self.config):
                    if not torch.isfinite(out_fresh[key]).all():
                        return False
                    if not torch.allclose(out_fresh[key], out_saved[key], atol=1e-5):
                        return False
                break  # one deterministic validation batch
        return True

    # ------------------------------------------------------------------
    # Full experiment
    # ------------------------------------------------------------------
    def fit(self) -> ExperimentResult:
        started = time.perf_counter()
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.perf_counter()
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            duration = time.perf_counter() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            record = EpochRecord(
                epoch=epoch,
                train_total_loss=train_metrics["total"],
                train_supervised_loss=train_metrics["supervised"],
                train_contrastive_loss=(
                    train_metrics["contrastive"] if self.config.use_ssl else None
                ),
                train_click_loss=train_metrics["click"] if self.config.use_mmoe else None,
                train_install_loss=train_metrics["install"],
                val_install_logloss=val_metrics["val_install_logloss"],
                val_install_auc=val_metrics["val_install_auc"],
                val_click_logloss=val_metrics["val_click_logloss"],
                val_click_auc=val_metrics["val_click_auc"],
                duration_seconds=duration,
                learning_rate=lr,
                n_train_batches=math.ceil(len(self.train_indices) / self.config.batch_size),
                n_val_batches=math.ceil(len(self.val_indices) / self.config.batch_size),
            )
            self.history.append(record)
            self._print_epoch(record)

            improved = record.val_install_logloss < self.best_val_install_logloss
            if improved:
                self.best_val_install_logloss = record.val_install_logloss
                self.best_epoch = epoch
                self._best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.save_checkpoint(epoch, val_install_auc=record.val_install_auc)
                print("  -> new best validation install logloss; checkpoint saved")
            elif self.config.patience and epoch - (self.best_epoch or 0) >= self.config.patience:
                print(
                    f"early stopping at epoch {epoch} "
                    f"(no improvement for {epoch - (self.best_epoch or 0)} epochs)"
                )
                break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        best = next(
            (r for r in self.history if r.epoch == self.best_epoch), None
        )
        result = ExperimentResult(
            experiment_name=self.config.experiment_name,
            config=self.config.to_dict(),
            epochs=[r.to_dict() for r in self.history],
            best_epoch=self.best_epoch,
            best_val_install_logloss=self.best_val_install_logloss
            if self.best_epoch is not None
            else None,
            best_val_install_auc=best.val_install_auc if best else None,
            best_val_click_logloss=best.val_click_logloss if best else None,
            best_val_click_auc=best.val_click_auc if best else None,
            total_seconds=time.perf_counter() - started,
            checkpoint_path=str(self.checkpoint_path),
            split_hash=self.split_hash,
            reload_verified=False,
        )
        result.reload_verified = self.verify_checkpoint_reload()
        if self.results_path is not None:
            result.save(self.results_path)
            print(f"result written: {self.results_path}")
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _print_epoch(log: EpochRecord) -> None:
        parts = [
            f"epoch {log.epoch:02d} | {log.duration_seconds:7.1f}s | lr {log.learning_rate:.2e}",
            f"train total {log.train_total_loss:.4f} (sup {log.train_supervised_loss:.4f}",
            f"install {log.train_install_loss:.4f}",
        ]
        if log.train_click_loss is not None:
            parts.append(f"click {log.train_click_loss:.4f}")
        if log.train_contrastive_loss is not None:
            parts.append(f"contrastive {log.train_contrastive_loss:.4f}")
        parts.append(")")
        parts.append(
            f"| val logloss install {log.val_install_logloss:.4f} "
            f"AUC {log.val_install_auc:.4f}"
        )
        if log.val_click_logloss is not None:
            parts.append(
                f"click {log.val_click_logloss:.4f} AUC {log.val_click_auc:.4f}"
            )
        print(" | ".join(parts))
