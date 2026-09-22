"""Stage 8 experiment configuration (extends the Stage 6 config conventions).

One frozen protocol for all four controlled experiments (spec 8.4); the
experiment name selects architecture and objective. Model-architecture
hyperparameters are NOT duplicated here — the frozen Stage 4/5 defaults
(d_model 128, nhead 8, 6 layers, FFN 128, dropout 0.1, 8 experts) and the
Stage 7 SSL defaults (projection 64, temperature 0.2) apply.

Serialization follows the existing ``to_dict``/``from_dict`` convention of
:class:`~recsys23_fedrec.training.config.TrainingConfig`; every checkpoint
stores the full experiment config for reproducibility.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from recsys23_fedrec.preprocessing.config import SEED

__all__ = ["ExperimentConfig", "EXPERIMENTS", "get_experiment_config", "DEFAULT_ALPHA"]

#: SSL joint-loss weight for the supervised term (Stage 7/8 spec).
DEFAULT_ALPHA = 0.6

#: Experiment A — Transformer supervised, install only (Stage 4 architecture).
EXPERIMENT_A = "experiment_a_transformer_supervised"
#: Experiment B — Transformer + MMoE (canonical Stage 6 baseline).
EXPERIMENT_B = "experiment_b_transformer_mmoe"
#: Experiment C — Transformer + SSL (no MMoE), joint alpha = 0.6.
EXPERIMENT_C = "experiment_c_transformer_ssl"
#: Experiment D — Transformer + MMoE + SSL, joint alpha = 0.6.
EXPERIMENT_D = "experiment_d_transformer_mmoe_ssl"

EXPERIMENTS: tuple[str, ...] = (EXPERIMENT_A, EXPERIMENT_B, EXPERIMENT_C, EXPERIMENT_D)


@dataclass
class ExperimentConfig:
    """Controlled configuration for one Stage 8 experiment.

    Attributes mirror spec 8.7. ``use_mmoe``/``use_ssl`` are derived from the
    experiment name and validated to match it, so configuration serialization
    and dispatch can never drift apart.
    """

    experiment_name: str
    model_type: str
    use_mmoe: bool
    use_ssl: bool
    alpha: float = DEFAULT_ALPHA
    corruption_rate: float = 0.15
    temperature: float = 0.2
    seed: int = SEED
    batch_size: int = 1024
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 3
    patience: int = 2
    device: str = "auto"
    lambda_click: float = 1.0
    lambda_install: float = 1.0

    def __post_init__(self) -> None:
        if self.experiment_name not in EXPERIMENTS:
            raise ValueError(
                f"unknown experiment {self.experiment_name!r}; expected one of {EXPERIMENTS}"
            )
        expected_mmoe = self.experiment_name in (EXPERIMENT_B, EXPERIMENT_D)
        expected_ssl = self.experiment_name in (EXPERIMENT_C, EXPERIMENT_D)
        if self.use_mmoe != expected_mmoe:
            raise ValueError(
                f"{self.experiment_name}: use_mmoe={self.use_mmoe} contradicts the "
                f"experiment definition (expected {expected_mmoe})"
            )
        if self.use_ssl != expected_ssl:
            raise ValueError(
                f"{self.experiment_name}: use_ssl={self.use_ssl} contradicts the "
                f"experiment definition (expected {expected_ssl})"
            )
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if not 0.0 <= self.corruption_rate <= 1.0:
            raise ValueError(f"corruption_rate must be in [0, 1], got {self.corruption_rate}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.model_type not in set(_EXPECTED_MODEL_TYPES.values()):
            raise ValueError(
                f"unknown model_type {self.model_type!r}; expected one of "
                f"{sorted(_EXPECTED_MODEL_TYPES.values())}"
            )
        expected_model = _EXPECTED_MODEL_TYPES[self.experiment_name]
        if self.model_type != expected_model:
            raise ValueError(
                f"{self.experiment_name}: model_type {self.model_type!r} contradicts "
                f"the experiment definition (expected {expected_model!r})"
            )
        if self.batch_size <= 0 or self.epochs <= 0 or self.patience < 0:
            raise ValueError("batch_size/epochs must be positive; patience non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

    # -- serialization (same convention as TrainingConfig) ----------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentConfig":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dict(payload).items() if k in field_names})

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


#: The exact architecture string per experiment (validated in __post_init__).
_EXPECTED_MODEL_TYPES: dict[str, str] = {
    EXPERIMENT_A: "transformer_baseline",
    EXPERIMENT_B: "transformer_mmoe",
    EXPERIMENT_C: "ssl_transformer_baseline",
    EXPERIMENT_D: "ssl_transformer_mmoe",
}


def get_experiment_config(
    experiment_name: str,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> ExperimentConfig:
    """Build the frozen-protocol config for one of the four experiments."""
    if experiment_name not in EXPERIMENTS:
        raise ValueError(
            f"unknown experiment {experiment_name!r}; expected one of {EXPERIMENTS}"
        )
    model_type = _EXPECTED_MODEL_TYPES[experiment_name]
    config = ExperimentConfig(
        experiment_name=experiment_name,
        model_type=model_type,
        use_mmoe=experiment_name in (EXPERIMENT_B, EXPERIMENT_D),
        use_ssl=experiment_name in (EXPERIMENT_C, EXPERIMENT_D),
    )
    if epochs is not None:
        config.epochs = epochs
    if batch_size is not None:
        config.batch_size = batch_size
    if device is not None:
        config.device = device
    return config
