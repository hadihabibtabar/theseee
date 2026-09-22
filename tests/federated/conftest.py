"""Fixtures for the federated-layer tests.

Reuses the Stage 2→3 synthetic fixture from ``tests/data/conftest.py`` so
client-loader tests exercise the exact same preprocessing artifacts and
Parquet dataset code paths as the data-layer tests, with no real data.
"""

from tests.data.conftest import *  # noqa: F401,F403
