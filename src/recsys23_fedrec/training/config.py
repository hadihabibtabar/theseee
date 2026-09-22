"""Centralized training configuration (Stage 6).

Plain serializable dataclass; stored inside checkpoints so every experiment
is auditable. Model architecture hyperparameters are NOT duplicated here —
the frozen Stage 4/5 defaults apply (d_model=128, nhead=8, num_layers=6,
dim_feedforward=128, dropout=0.1, num_experts=8).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from recsys23_fedrec.preprocessing.config import SEED

__all__ = ["TrainingConfig"]


@dataclass
class TrainingConfig:
    """Centralized supervised training configuration (Stage 6 defaults)."""

    seed: int = SEED
    batch_size: int = 1024
    epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    optimizer: str = "adamw"
    lambda_click: float = 1.0
    lambda_install: float = 1.0
    early_stopping_patience: int = 2
    device: str = "auto"  # "auto" -> cuda if available else cpu

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainingConfig":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dict(payload).items() if k in field_names})

    @classmethod
    def from_json_file(cls, path: str | Path) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
