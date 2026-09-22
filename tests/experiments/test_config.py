"""Stage 8 ExperimentConfig tests (spec 8.12)."""

from __future__ import annotations

import json

import pytest

from recsys23_fedrec.experiments.config import (
    DEFAULT_ALPHA,
    EXPERIMENTS,
    EXPERIMENT_A,
    EXPERIMENT_B,
    EXPERIMENT_C,
    EXPERIMENT_D,
    ExperimentConfig,
    get_experiment_config,
)
from recsys23_fedrec.models import (
    SSLTransformerBaseline,
    SSLTransformerMMoE,
    TransformerBaseline,
    TransformerMMoE,
)


def test_all_four_experiments_defined():
    assert EXPERIMENTS == (
        EXPERIMENT_A,
        EXPERIMENT_B,
        EXPERIMENT_C,
        EXPERIMENT_D,
    )


@pytest.mark.parametrize(
    "name, mmoe, ssl",
    [
        (EXPERIMENT_A, False, False),
        (EXPERIMENT_B, True, False),
        (EXPERIMENT_C, False, True),
        (EXPERIMENT_D, True, True),
    ],
)
def test_config_dispatch(name, mmoe, ssl):
    config = get_experiment_config(name)
    assert config.use_mmoe is mmoe
    assert config.use_ssl is ssl
    assert config.model_type == {
        EXPERIMENT_A: "transformer_baseline",
        EXPERIMENT_B: "transformer_mmoe",
        EXPERIMENT_C: "ssl_transformer_baseline",
        EXPERIMENT_D: "ssl_transformer_mmoe",
    }[name]


def test_frozen_protocol_defaults():
    config = get_experiment_config(EXPERIMENT_C)
    assert config.seed == 42
    assert config.batch_size == 1024
    assert config.epochs == 3
    assert config.learning_rate == 3e-4
    assert config.weight_decay == 1e-2
    assert config.optimizer == "adamw"
    assert config.alpha == 0.6
    assert config.corruption_rate == 0.15
    assert config.temperature == 0.2
    assert config.patience == 2
    assert config.lambda_click == 1.0
    assert config.lambda_install == 1.0


def test_contradictory_flags_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(
            experiment_name=EXPERIMENT_A, model_type="transformer_baseline",
            use_mmoe=True, use_ssl=False,
        )
    with pytest.raises(ValueError):
        ExperimentConfig(
            experiment_name=EXPERIMENT_D, model_type="ssl_transformer_mmoe",
            use_mmoe=True, use_ssl=False,
        )
    with pytest.raises(ValueError):
        ExperimentConfig(
            experiment_name=EXPERIMENT_C, model_type="ssl_transformer_baseline",
            use_mmoe=False, use_ssl=False,
        )


def test_wrong_model_type_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(
            experiment_name=EXPERIMENT_B, model_type="transformer_baseline",
            use_mmoe=True, use_ssl=False,
        )


def test_unknown_experiment_rejected():
    with pytest.raises(ValueError):
        get_experiment_config("experiment_e_transformer_xl")
    with pytest.raises(ValueError):
        ExperimentConfig(
            experiment_name="bogus", model_type="transformer_mmoe",
            use_mmoe=True, use_ssl=False,
        )


def test_invalid_values_rejected():
    base = dict(experiment_name=EXPERIMENT_C, model_type="ssl_transformer_baseline",
                use_mmoe=False, use_ssl=True)
    for field, value in (
        ("alpha", 1.5), ("alpha", -0.1), ("corruption_rate", 2.0),
        ("temperature", 0.0), ("batch_size", 0), ("epochs", 0),
        ("learning_rate", 0.0), ("patience", -1),
    ):
        with pytest.raises(ValueError):
            ExperimentConfig(**base, **{field: value})


def test_serialization_roundtrip(tmp_path):
    config = get_experiment_config(EXPERIMENT_D)
    payload = config.to_dict()
    assert payload["experiment_name"] == EXPERIMENT_D
    assert json.loads(config.to_json())["alpha"] == DEFAULT_ALPHA
    restored = ExperimentConfig.from_dict(payload)
    assert restored == config
    path = tmp_path / "config.json"
    path.write_text(config.to_json())
    assert ExperimentConfig.from_json_file(path) == config


def test_overrides_applied():
    config = get_experiment_config(EXPERIMENT_A, epochs=1, batch_size=128, device="cpu")
    assert config.epochs == 1 and config.batch_size == 128 and config.device == "cpu"


# ---------------------------------------------------------------------------
# Model dispatch correctness (spec 8.12: correct model selected for A/B/C/D)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name, cls",
    [
        (EXPERIMENT_A, TransformerBaseline),
        (EXPERIMENT_B, TransformerMMoE),
        (EXPERIMENT_C, SSLTransformerBaseline),
        (EXPERIMENT_D, SSLTransformerMMoE),
    ],
)
def test_build_model_dispatch(feature_set, name, cls):
    from recsys23_fedrec.experiments.runner import build_model

    model = build_model(get_experiment_config(name), feature_set)
    assert isinstance(model, cls)


def test_ssl_augmentation_only_for_c_and_d(feature_set):
    from recsys23_fedrec.experiments.runner import build_model

    for name in (EXPERIMENT_A, EXPERIMENT_B):
        model = build_model(get_experiment_config(name), feature_set)
        assert not hasattr(model, "projection"), f"{name} must not carry a projection head"
        aug = getattr(model, "augmentations", None)
        if aug is None and hasattr(model, "mmoe"):
            aug = getattr(model.mmoe, "augmentations", None)
        assert aug is None, f"{name} must not carry an augmentation module"
    for name in (EXPERIMENT_C, EXPERIMENT_D):
        model = build_model(get_experiment_config(name), feature_set)
        assert hasattr(model, "projection")
        assert model.augmentations is not None
        assert model.augmentations.corruption_rate == 0.15
        assert model.ssl_config.temperature == 0.2


def test_backbone_identical_across_experiments(feature_set):
    """All four experiments share the same frozen Stage 4 backbone config."""
    from recsys23_fedrec.experiments.runner import build_model
    from recsys23_fedrec.models import TransformerBaseline, TransformerMMoE

    for name in EXPERIMENTS:
        model = build_model(get_experiment_config(name), feature_set)
        if isinstance(model, TransformerBaseline):
            backbone = model
        elif isinstance(model, TransformerMMoE):
            backbone = model.backbone
        elif hasattr(model, "mmoe"):
            backbone = model.mmoe.backbone
        else:
            backbone = model.base
        assert backbone.config.d_model == 128
        assert backbone.config.num_layers == 6
        assert backbone.config.nhead == 8
        assert backbone.config.dim_feedforward == 128
        assert backbone.config.dropout == 0.1
