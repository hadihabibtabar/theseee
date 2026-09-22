"""Fixtures for the Stage 7 SSL tests (via shared helpers)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recsys23_fedrec.data import FeatureSet, ProcessedRecSysDataset, collate_train  # noqa: E402
from recsys23_fedrec.models import SSLConfig, SSLTransformerMMoE, TransformerConfig  # noqa: E402
from recsys23_fedrec.ssl import FeatureCorruptionAugmentation  # noqa: E402
from tests.fixture_helpers import (  # noqa: E402
    build_processed_fixture,
    load_feature_set,
    load_real_feature_set,
)


@pytest.fixture()
def processed_fixture(tmp_path):
    return build_processed_fixture(tmp_path)


@pytest.fixture()
def feature_set(processed_fixture):
    return load_feature_set(processed_fixture)


@pytest.fixture()
def real_feature_set():
    return load_real_feature_set()


@pytest.fixture()
def synthetic_batch(feature_set, processed_fixture) -> dict[str, torch.Tensor]:
    """A small collated train batch from the synthetic Stage 3 fixture."""
    dataset = ProcessedRecSysDataset(
        feature_set, processed_fixture["train_dir"], split="train"
    )
    return collate_train([dataset[i] for i in range(8)])


@pytest.fixture()
def small_ssl_model(feature_set) -> SSLTransformerMMoE:
    """Compact SSL model over the synthetic fixture schema (fast tests)."""
    torch.manual_seed(42)
    return SSLTransformerMMoE(
        feature_set,
        ssl_config=SSLConfig(projection_dim=64, temperature=0.2),
        backbone_config=TransformerConfig(
            d_model=32, nhead=4, num_layers=2, dim_feedforward=32, dropout=0.1
        ),
        augmentations=FeatureCorruptionAugmentation(feature_set, corruption_rate=0.15, seed=42),
    )
