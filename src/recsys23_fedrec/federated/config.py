"""Stage 10A: frozen protocol configuration for the federated baseline.

One frozen protocol (Stage 10 spec). Values are validated strictly in
``__post_init__`` because this stage *freezes* the baseline: changing
protocol-critical fields (10 clients, 1 local epoch, all-client
participation, sample-weighted FedAvg) is a new protocol version, not a
parameter tweak.

Serialization follows the existing ``to_dict``/``from_dict`` convention of
:class:`~recsys23_fedrec.experiments.config.ExperimentConfig` so later stages
can embed the full protocol in checkpoints/reports without a parallel
configuration system.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from recsys23_fedrec.preprocessing.config import SEED

__all__ = [
    "FederatedConfig",
    "FEDAVG_METHOD",
    "PARTICIPATION_ALL",
    "LOCAL_EPOCHS",
    "ROUNDS",
    "LOCAL_BATCH_SIZE",
    "get_federated_config",
]

#: Aggregation method identifier recorded in manifests/checkpoints.
FEDAVG_METHOD = "sample_weighted_fedavg"
#: Stage 10 baseline participation policy: every client, every round.
PARTICIPATION_ALL = "all_clients_every_round"
#: Local epochs per client per round.
LOCAL_EPOCHS = 1
#: Global FedAvg rounds for the baseline.
ROUNDS = 10
#: Local batch size (identical to the centralized Stage 6/8 protocol).
LOCAL_BATCH_SIZE = 1024


@dataclass
class FederatedConfig:
    """Frozen Stage 10 baseline federated protocol.

    Attributes:
        num_clients: exactly 10 simulated clients (Stage 9 protocol).
        alpha: Dirichlet concentration of the frozen partition.
        partition_seed: seed that generated the frozen partition.
        seed: project seed for client-local training order.
        participation: ``all_clients_every_round`` for this baseline.
        local_epochs: exactly 1 local epoch per client per round.
        rounds: exactly 10 FedAvg rounds.
        local_batch_size: 1024, matching the centralized protocol.
        optimizer / learning_rate / weight_decay: AdamW 3e-4 / 1e-2.
        lambda_click / lambda_install: 1.0 / 1.0 (Stage 5/6 objective).
        aggregation: ``sample_weighted_fedavg``.
        selection_metric: global validation Install LogLoss.
        use_ssl: False — the Stage 10 baseline is purely supervised.
        local_optimizer_state_persistence: False — a fresh local optimizer
            is created per client per round (see protocol report).
        local_sampler_epoch_mode: ``round_minus_one`` — the client batch
            sampler receives ``set_epoch(round - 1)`` so round 1 reuses the
            sampler arrangement of the centralized epoch-1 convention
            (``seed + e - 1``) and every round sees a different,
            reproducible arrangement.
    """

    num_clients: int = 10
    alpha: float = 0.5
    partition_seed: int = 42
    seed: int = SEED
    participation: str = PARTICIPATION_ALL
    local_epochs: int = LOCAL_EPOCHS
    rounds: int = ROUNDS
    local_batch_size: int = LOCAL_BATCH_SIZE
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    lambda_click: float = 1.0
    lambda_install: float = 1.0
    aggregation: str = FEDAVG_METHOD
    selection_metric: str = "global_validation_install_logloss"
    use_ssl: bool = False
    local_optimizer_state_persistence: bool = False
    local_sampler_epoch_mode: str = "round_minus_one"

    def __post_init__(self) -> None:
        if self.num_clients != 10:
            raise ValueError(
                f"the Stage 10 protocol requires exactly 10 clients, got {self.num_clients}"
            )
        if not self.alpha > 0.0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.participation != PARTICIPATION_ALL:
            raise ValueError(
                f"the Stage 10 baseline requires {PARTICIPATION_ALL!r}, got {self.participation!r}"
            )
        if self.local_epochs != LOCAL_EPOCHS:
            raise ValueError(
                f"the Stage 10 baseline requires local_epochs={LOCAL_EPOCHS}, got {self.local_epochs}"
            )
        if self.rounds <= 0:
            raise ValueError(f"rounds must be positive, got {self.rounds}")
        if self.local_batch_size <= 0:
            raise ValueError(f"local_batch_size must be positive, got {self.local_batch_size}")
        if self.optimizer != "adamw":
            raise ValueError(f"the Stage 10 baseline requires adamw, got {self.optimizer!r}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.aggregation != FEDAVG_METHOD:
            raise ValueError(
                f"the Stage 10 baseline requires {FEDAVG_METHOD!r}, got {self.aggregation!r}"
            )
        if self.selection_metric != "global_validation_install_logloss":
            raise ValueError(f"unknown selection metric {self.selection_metric!r}")
        if self.use_ssl:
            raise ValueError("the Stage 10 baseline must not use SSL")
        if self.local_sampler_epoch_mode != "round_minus_one":
            raise ValueError(
                f"unknown local_sampler_epoch_mode {self.local_sampler_epoch_mode!r}"
            )

    # -- serialization (same convention as ExperimentConfig) --------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FederatedConfig":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dict(payload).items() if k in field_names})

    @classmethod
    def from_json_file(cls, path: str | Path) -> "FederatedConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def get_federated_config() -> FederatedConfig:
    """The frozen Stage 10 baseline protocol (no parameters to vary)."""
    return FederatedConfig()
