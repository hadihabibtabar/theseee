"""Fixtures for the Stage 6 training tests (via shared helpers)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

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
