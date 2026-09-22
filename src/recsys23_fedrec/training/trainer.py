"""Centralized supervised trainer for the Transformer + MMoE model (Stage 6).

Clean single-process training loop over the Stage 3 memory-bounded Parquet
loader with index-subset loaders (Stage 6 split). Responsibilities:

* epoch loop with model.train()/eval() discipline, zero_grad, backward, step
* multi-task MMoE loss (raw logits -> BCEWithLogitsLoss, no sigmoid)
* validation pass with stable log-loss + exact ROC-AUC per task
* checkpointing on best validation Install LogLoss (primary metric) with
  early stopping (patience on the same metric)
* reproducibility: project seed, per-epoch deterministic part-shuffled
  batches, full config + split hash stored in every checkpoint

The official test set is never touched here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn

from recsys23_fedrec.models import TransformerMMoE, mmoe_loss
from recsys23_fedrec.training.config import TrainingConfig
from recsys23_fedrec.training.metrics import binary_logloss, roc_auc

__all__ = ["EpochLog", "Trainer", "resolve_device"]


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class EpochLog:
    """One epoch's training + validation summary."""

    epoch: int
    train_total_loss: float
    train_click_loss: float
    train_install_loss: float
    val_click_logloss: float
    val_install_logloss: float
    val_click_auc: float
    val_install_auc: float
    duration_seconds: float
    learning_rate: float
    n_train_batches: int
    n_val_batches: int


@dataclass
class Trainer:
    """Train/validate/checkpoint a TransformerMMoE on index-subset loaders."""

    model: TransformerMMoE
    config: TrainingConfig
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    checkpoint_path: str | Path
    split_hash: str = ""
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    patience: int = 2

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.checkpoint_path = Path(self.checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_val_install_logloss = float("inf")
        self.best_epoch: int | None = None
        self.epochs_without_improvement = 0
        self.history: list[EpochLog] = []
        self._best_state: dict | None = None

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    # ------------------------------------------------------------------
    # Training pass
    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """One training pass over the train loader; returns loss aggregates."""
        self.model.train()
        totals = {"total": 0.0, "click": 0.0, "install": 0.0}
        n_batches = 0
        for batch in self.train_loader:
            batch = self._to_device(batch, self.device)
            self.optimizer.zero_grad()
            output = self.model(batch, return_diagnostics=False)
            losses = mmoe_loss(
                output,
                batch,
                lambda_click=self.config.lambda_click,
                lambda_install=self.config.lambda_install,
            )
            losses["total"].backward()
            self.optimizer.step()
            for key in totals:
                totals[key] += losses[key].item()
            n_batches += 1
        return {key: value / max(n_batches, 1) for key, value in totals.items()}

    # ------------------------------------------------------------------
    # Validation pass
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Full validation pass; returns per-task log-loss + AUC aggregates.

        Log-loss is aggregated sample-weighted (sum of per-sample losses /
        total samples), so the small final partial batch cannot bias the
        reported metric. AUC is computed once over all validation logits.
        """
        self.model.eval()
        click_logits: list[Tensor] = []
        install_logits: list[Tensor] = []
        click_targets: list[Tensor] = []
        install_targets: list[Tensor] = []
        click_loss_sum = 0.0
        install_loss_sum = 0.0
        n_seen = 0
        for batch in self.val_loader:
            batch = self._to_device(batch, self.device)
            output = self.model(batch, return_diagnostics=False)
            batch_n = int(batch["click"].shape[0])
            click_loss_sum += binary_logloss(output["click_logit"], batch["click"]).item() * batch_n
            install_loss_sum += (
                binary_logloss(output["install_logit"], batch["install"]).item() * batch_n
            )
            n_seen += batch_n
            click_logits.append(output["click_logit"].detach().float().cpu())
            install_logits.append(output["install_logit"].detach().float().cpu())
            click_targets.append(batch["click"].detach().float().cpu())
            install_targets.append(batch["install"].detach().float().cpu())

        all_click_logits = torch.cat(click_logits)
        all_install_logits = torch.cat(install_logits)
        all_click_targets = torch.cat(click_targets)
        all_install_targets = torch.cat(install_targets)
        if n_seen == 0:
            raise ValueError("validation loader produced no batches")
        return {
            "val_click_logloss": click_loss_sum / n_seen,
            "val_install_logloss": install_loss_sum / n_seen,
            "val_click_auc": float(roc_auc(all_click_logits, all_click_targets)),
            "val_install_auc": float(roc_auc(all_install_logits, all_install_targets)),
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int) -> Path:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": int(epoch),
            "best_validation_install_logloss": float(self.best_val_install_logloss),
            "training_config": self.config.to_dict(),
            "seed": self.config.seed,
            "split_hash": self.split_hash,
            "stage": "6-centralized",
        }
        torch.save(payload, self.checkpoint_path)
        return self.checkpoint_path

    # ------------------------------------------------------------------
    # Full fit
    # ------------------------------------------------------------------
    def fit(self, max_epochs: int | None = None) -> dict:
        """Run the full training loop; returns the summary dictionary."""
        epochs = max_epochs or self.config.epochs
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()

            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch - 1)  # 0-based for the RNG stream

            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            duration = time.perf_counter() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            log = EpochLog(
                epoch=epoch,
                train_total_loss=train_metrics["total"],
                train_click_loss=train_metrics["click"],
                train_install_loss=train_metrics["install"],
                val_click_logloss=val_metrics["val_click_logloss"],
                val_install_logloss=val_metrics["val_install_logloss"],
                val_click_auc=val_metrics["val_click_auc"],
                val_install_auc=val_metrics["val_install_auc"],
                duration_seconds=duration,
                learning_rate=lr,
                n_train_batches=len(self.train_loader),
                n_val_batches=len(self.val_loader),
            )
            self.history.append(log)
            self._print_epoch(log)

            improved = log.val_install_logloss < self.best_val_install_logloss
            if improved:
                self.best_val_install_logloss = log.val_install_logloss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.save_checkpoint(epoch)
                print(f"  -> new best validation install logloss; checkpoint saved")
            else:
                self.epochs_without_improvement += 1

            if (
                self.patience is not None
                and self.epochs_without_improvement >= self.patience
                and epoch >= self.best_epoch + self.patience
            ):
                print(
                    f"early stopping at epoch {epoch} "
                    f"(no improvement for {self.epochs_without_improvement} epochs)"
                )
                break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return {
            "epochs_completed": len(self.history),
            "best_epoch": self.best_epoch,
            "best_val_install_logloss": self.best_val_install_logloss,
            "best_val_install_auc": next(
                (h.val_install_auc for h in self.history if h.epoch == self.best_epoch), None
            ),
            "best_val_click_logloss": next(
                (h.val_click_logloss for h in self.history if h.epoch == self.best_epoch), None
            ),
            "best_val_click_auc": next(
                (h.val_click_auc for h in self.history if h.epoch == self.best_epoch), None
            ),
            "total_seconds": time.perf_counter() - started,
            "checkpoint_path": str(self.checkpoint_path),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _print_epoch(log: EpochLog) -> None:
        print(
            f"epoch {log.epoch:02d} | {log.duration_seconds:7.1f}s | lr {log.learning_rate:.2e} | "
            f"train total {log.train_total_loss:.4f} "
            f"(click {log.train_click_loss:.4f}, install {log.train_install_loss:.4f}) | "
            f"val logloss click {log.val_click_logloss:.4f} install {log.val_install_logloss:.4f} | "
            f"val AUC click {log.val_click_auc:.4f} install {log.val_install_auc:.4f}"
        )
