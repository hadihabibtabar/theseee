"""Stage 8: controlled centralized experiments (A/B/C/D)."""

from .config import (
    EXPERIMENT_A,
    EXPERIMENT_B,
    EXPERIMENT_C,
    EXPERIMENT_D,
    EXPERIMENTS,
    ExperimentConfig,
    get_experiment_config,
)
from .runner import ExperimentResult, ExperimentRunner

__all__ = [
    "ExperimentConfig",
    "EXPERIMENTS",
    "EXPERIMENT_A",
    "EXPERIMENT_B",
    "EXPERIMENT_C",
    "EXPERIMENT_D",
    "get_experiment_config",
    "ExperimentResult",
    "ExperimentRunner",
]
