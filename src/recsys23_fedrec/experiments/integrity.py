"""Stage 8: protected-artifact integrity verification.

Stage 8 must not modify the frozen Stage 1–7 artifacts. This module records a
snapshot of the protected state (hashes, sizes, mtimes) and re-verifies it
after experiments. It performs READ-ONLY operations only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

__all__ = [
    "PROTECTED_VALUES",
    "STAGE6_CHECKPOINT_RELPATH",
    "SPLIT_JSON_RELPATH",
    "snapshot_protected_artifacts",
    "verify_protected_artifacts",
]

#: Canonical Stage 6 checkpoint (must never be overwritten in Stage 8).
STAGE6_CHECKPOINT_RELPATH = "artifacts/checkpoints/centralized_transformer_mmoe_best.pt"
#: Canonical centralized split metadata (hash-verified by load_split as well).
SPLIT_JSON_RELPATH = "artifacts/splits/centralized_split.json"

#: Frozen values from Stages 1–7 (verified at Stage 8 start).
PROTECTED_VALUES = {
    "stage2_artifact_hash": "c3a62caf13326617",
    "split_train_index_hash": "dd37605169615442",
    "split_validation_index_hash": "3cd8370ebdecb305",
    "raw_train_bytes": 1_919_412_753,
    "raw_test_bytes": 88_684_100,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_protected_artifacts(project_root: str | Path = ".") -> dict[str, Any]:
    """Read-only snapshot of every protected artifact (values + mtimes)."""
    root = Path(project_root)
    snapshot: dict[str, Any] = {}

    checkpoint = root / STAGE6_CHECKPOINT_RELPATH
    snapshot["stage6_checkpoint_sha256"] = _sha256_file(checkpoint)
    snapshot["stage6_checkpoint_mtime"] = os.path.getmtime(checkpoint)

    split_meta = json.loads((root / SPLIT_JSON_RELPATH).read_text(encoding="utf-8"))
    snapshot["split_train_index_hash"] = split_meta["train_index_hash"]
    snapshot["split_validation_index_hash"] = split_meta["validation_index_hash"]
    snapshot["split_total_rows"] = split_meta["total_rows"]

    sys.path.insert(0, str(root / "src"))
    from recsys23_fedrec.data import FeatureSet  # noqa: E402  (local import: path setup)

    snapshot["stage2_artifact_hash"] = FeatureSet.load(root / "artifacts" / "preprocessing").artifact_hash

    snapshot["raw_train_bytes"] = sum(
        os.path.getsize(root / "train" / name) for name in os.listdir(root / "train")
    )
    snapshot["raw_test_bytes"] = sum(
        os.path.getsize(root / "test" / name) for name in os.listdir(root / "test")
    )
    return snapshot


def verify_protected_artifacts(
    snapshot: dict[str, Any], project_root: str | Path = "."
) -> list[str]:
    """Compare a snapshot against the frozen values + the live state.

    Returns a list of violations (empty == all protected artifacts intact).
    Call pattern: snapshot before the experiment, verify after (the function
    re-snapshots the live state internally and diffs against the given one).
    """
    violations: list[str] = []
    expected = dict(PROTECTED_VALUES)
    expected["stage6_checkpoint_sha256"] = (
        "25f3267eba572bf99e72c19eee2351236fe65327f71af7c7a8e39fae8e1d8617"
    )

    # Frozen-value checks against the GIVEN snapshot (taken before the run).
    for key, value in expected.items():
        if snapshot.get(key) != value:
            violations.append(
                f"pre-run snapshot {key}={snapshot.get(key)!r} != frozen {value!r}"
            )

    # Live-state diff: nothing protected may have changed since the snapshot.
    live = snapshot_protected_artifacts(project_root)
    for key, before in snapshot.items():
        after = live.get(key)
        if before != after:
            violations.append(f"protected artifact changed during run: {key} {before!r} -> {after!r}")
    return violations
