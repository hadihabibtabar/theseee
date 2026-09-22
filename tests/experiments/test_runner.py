"""Stage 8 runner tests: losses, checkpoint metadata, integrity (spec 8.12)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from recsys23_fedrec.data import ProcessedRecSysDataset, collate_train
from recsys23_fedrec.experiments.config import (
    EXPERIMENT_A,
    EXPERIMENT_B,
    EXPERIMENT_C,
    EXPERIMENT_D,
    ExperimentConfig,
    get_experiment_config,
)
from recsys23_fedrec.experiments.integrity import (
    PROTECTED_VALUES,
    snapshot_protected_artifacts,
    verify_protected_artifacts,
)
from recsys23_fedrec.experiments.runner import (
    ExperimentRunner,
    build_model,
    compute_contrastive_loss,
    compute_joint_loss,
    compute_supervised_loss,
)


# ---------------------------------------------------------------------------
# Fixtures: tiny runner over the synthetic fixture data
# ---------------------------------------------------------------------------
@pytest.fixture()
def tiny_setup(feature_set, processed_fixture):
    dataset = ProcessedRecSysDataset(
        feature_set, processed_fixture["train_dir"], split="train"
    )
    n = len(dataset)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    val_idx = np.sort(perm[:60])
    train_idx = np.sort(perm[60:])
    return dataset, train_idx, val_idx


def _make_runner(feature_set, tiny_setup, name, tmp_path, **overrides):
    dataset, train_idx, val_idx = tiny_setup
    config = get_experiment_config(name, **overrides)
    runner = ExperimentRunner(
        config=config,
        feature_set=feature_set,
        train_dataset=dataset,
        train_indices=train_idx,
        val_indices=val_idx,
        checkpoint_path=tmp_path / f"{name}.pt",
        split_hash="dd37605169615442",
        device="cpu",
    )
    return runner


# ---------------------------------------------------------------------------
# Loss dispatch (spec 8.12: correct loss selected)
# ---------------------------------------------------------------------------
def test_loss_dispatch_per_experiment(feature_set, tiny_setup):
    dataset, _, _ = tiny_setup
    batch = collate_train([dataset[i] for i in range(8)])
    for name in (EXPERIMENT_A, EXPERIMENT_B, EXPERIMENT_C, EXPERIMENT_D):
        config = get_experiment_config(name)
        model = build_model(config, feature_set)
        model.eval()
        with torch.no_grad():
            if config.use_ssl:
                out = model(batch)
                sup = compute_supervised_loss(out, batch, config)
                con = compute_contrastive_loss(out, config)
                joint = compute_joint_loss(out, batch, config)
                expected_total = config.alpha * sup["total"] + (1 - config.alpha) * con
                assert torch.allclose(joint["total"], expected_total, atol=1e-6)
            else:
                out = model(batch)
                sup = compute_supervised_loss(out, batch, config)
                joint = sup
        assert torch.isfinite(joint["total"])
        # A/C have no click loss; B/D do
        has_click = sup.get("click") is not None
        assert has_click == config.use_mmoe


def test_ssl_disabled_for_a_and_b(feature_set, tiny_setup):
    for name in (EXPERIMENT_A, EXPERIMENT_B):
        config = get_experiment_config(name)
        assert not config.use_ssl
        with pytest.raises(ValueError):
            compute_contrastive_loss({"z1": torch.randn(4, 8)}, config)


def test_no_sigmoid_squashing(feature_set, tiny_setup):
    """Raw logits: BCEWithLogitsLoss applied to unbounded values."""
    dataset, _, _ = tiny_setup
    batch = collate_train([dataset[i] for i in range(8)])
    big = torch.tensor([50.0, -50.0, 30.0, -30.0, 10.0, -10.0, 0.0, 5.0])
    # A: single install logit; B: both task logits at extreme values
    sup_a = compute_supervised_loss({"install_logit": big}, batch, get_experiment_config(EXPERIMENT_A))
    assert torch.isfinite(sup_a["total"])  # BCEWithLogits stable at ±50
    sup_b = compute_supervised_loss(
        {"install_logit": big, "click_logit": -big}, batch, get_experiment_config(EXPERIMENT_B)
    )
    assert torch.isfinite(sup_b["total"])


# ---------------------------------------------------------------------------
# Training loop behavior (tiny fixture scale)
# ---------------------------------------------------------------------------
def test_runner_fit_updates_and_selects_best(feature_set, tiny_setup, tmp_path):
    runner = _make_runner(feature_set, tiny_setup, EXPERIMENT_C, tmp_path, epochs=2, batch_size=32)
    before = {n: p.detach().clone() for n, p in runner.model.named_parameters()}
    result = runner.fit()
    changed = [
        n for n, p in runner.model.named_parameters()
        if not torch.equal(p.detach(), before[n])
    ]
    assert changed, "fit must update parameters"
    assert result.best_epoch in (1, 2)
    assert math.isfinite(result.best_val_install_logloss)
    assert 0.0 <= result.best_val_install_auc <= 1.0
    assert result.best_val_click_logloss is None  # C has no click task
    assert result.reload_verified is True
    assert len(result.epochs) == 2
    per_epoch = result.epochs[0]
    assert per_epoch["train_contrastive_loss"] is not None
    assert per_epoch["train_click_loss"] is None
    assert per_epoch["val_click_logloss"] is None


def test_runner_mmoe_reports_click(feature_set, tiny_setup, tmp_path):
    runner = _make_runner(feature_set, tiny_setup, EXPERIMENT_B, tmp_path, epochs=1, batch_size=32)
    result = runner.fit()
    assert result.best_val_click_logloss is not None
    assert result.best_val_click_auc is not None
    per_epoch = result.epochs[0]
    assert per_epoch["train_click_loss"] is not None
    assert per_epoch["train_contrastive_loss"] is None  # B has no SSL


def test_runner_single_task_no_click(feature_set, tiny_setup, tmp_path):
    runner = _make_runner(feature_set, tiny_setup, EXPERIMENT_A, tmp_path, epochs=1, batch_size=32)
    result = runner.fit()
    assert result.best_val_click_logloss is None
    assert result.best_val_click_auc is None
    assert result.epochs[0]["train_contrastive_loss"] is None


# ---------------------------------------------------------------------------
# Checkpoint metadata + atomicity
# ---------------------------------------------------------------------------
def test_checkpoint_metadata_complete(feature_set, tiny_setup, tmp_path):
    runner = _make_runner(feature_set, tiny_setup, EXPERIMENT_D, tmp_path, epochs=1, batch_size=32)
    runner.fit()
    payload = torch.load(runner.checkpoint_path, weights_only=False, map_location="cpu")
    for key in (
        "experiment_name", "model_type", "model_state_dict", "optimizer_state_dict",
        "epoch", "best_validation_install_logloss", "training_config", "seed",
        "alpha", "corruption_rate", "temperature", "split_hash", "stage",
        # Stage 8 full-run metadata additions
        "experiment", "stage_number", "architecture_config", "parameter_count",
        "optimizer_config", "batch_size", "val_install_auc", "train_index_hash",
        "validation_index_hash", "preprocessing_hash", "sampler_protocol",
        "n_train_rows", "n_val_rows", "n_train_batches", "n_val_batches",
        "device", "checkpoint_timestamp",
    ):
        assert key in payload, key
    assert payload["experiment_name"] == EXPERIMENT_D
    assert payload["alpha"] == 0.6
    assert payload["corruption_rate"] == 0.15
    assert payload["temperature"] == 0.2
    assert payload["seed"] == 42
    assert payload["split_hash"] == "dd37605169615442"
    assert payload["stage_number"] == 8
    assert payload["stage"].startswith("8")
    assert payload["sampler_protocol"].startswith("stage8-corrected")
    assert payload["parameter_count"] > 0
    assert payload["val_install_auc"] is not None  # best epoch had validation
    assert 0.0 <= payload["val_install_auc"] <= 1.0
    assert payload["checkpoint_timestamp"]  # ISO timestamp recorded
    assert payload["training_config"]["epochs"] == 1  # the actual run config


def test_checkpoint_write_is_atomic(feature_set, tiny_setup, tmp_path):
    runner = _make_runner(feature_set, tiny_setup, EXPERIMENT_A, tmp_path, epochs=1, batch_size=32)
    runner.fit()
    leftovers = [p for p in tmp_path.glob("*.tmp") if p.is_file()]
    assert not leftovers, f"temporary checkpoint files left behind: {leftovers}"
    assert runner.checkpoint_path.is_file()


def test_result_serialization(feature_set, tiny_setup, tmp_path):
    results_path = tmp_path / "results" / "exp.json"
    runner = _make_runner(
        feature_set, tiny_setup, EXPERIMENT_A, tmp_path, epochs=1, batch_size=32
    )
    runner.results_path = results_path
    result = runner.fit()
    data = json.loads(results_path.read_text(encoding="utf-8"))
    assert data["experiment_name"] == EXPERIMENT_A
    assert data["best_epoch"] == result.best_epoch
    assert data["config"]["experiment_name"] == EXPERIMENT_A
    assert data["split_hash"] == "dd37605169615442"
    assert data["reload_verified"] is True
    assert data["epochs"][0]["n_train_batches"] > 0
    assert data["epochs"][0]["n_val_batches"] > 0


def test_nonfinite_training_loss_raises(feature_set, tiny_setup, tmp_path):
    """A NaN/Inf loss must stop the run loudly (Stage 8 monitoring)."""
    runner = _make_runner(feature_set, tiny_setup, EXPERIMENT_A, tmp_path, epochs=1, batch_size=32)
    import recsys23_fedrec.experiments.runner as runner_mod

    original = runner_mod.compute_joint_loss
    runner_mod.compute_joint_loss = lambda *a, **k: {
        "total": torch.tensor(float("nan")), "install": torch.tensor(float("nan")), "click": None
    }
    try:
        with pytest.raises(RuntimeError, match="non-finite training loss"):
            runner.train_one_epoch(epoch=1)
    finally:
        runner_mod.compute_joint_loss = original


def test_runner_does_not_touch_foreign_checkpoints(feature_set, tiny_setup, tmp_path):
    """A runner writing its own checkpoint must leave other files untouched."""
    sentinel = tmp_path / "unrelated.pt"
    torch.save({"keep": True}, sentinel)
    before = sentinel.read_bytes()
    runner = _make_runner(feature_set, tiny_setup, EXPERIMENT_B, tmp_path, epochs=1, batch_size=32)
    runner.fit()
    assert sentinel.read_bytes() == before


# ---------------------------------------------------------------------------
# Integrity guards (spec 8.12 + 8.15)
# ---------------------------------------------------------------------------
def test_protected_values_constants():
    assert PROTECTED_VALUES["stage2_artifact_hash"] == "c3a62caf13326617"
    assert PROTECTED_VALUES["split_train_index_hash"] == "dd37605169615442"
    assert PROTECTED_VALUES["split_validation_index_hash"] == "3cd8370ebdecb305"
    assert PROTECTED_VALUES["raw_train_bytes"] == 1_919_412_753
    assert PROTECTED_VALUES["raw_test_bytes"] == 88_684_100


def test_verify_protected_artifacts_detects_change(tmp_path):
    file = tmp_path / "guarded.bin"
    file.write_bytes(b"original")
    original = {"guarded.bin": file.read_bytes()}

    # Simulate the verify flow with a fake snapshot function:
    from recsys23_fedrec.experiments import integrity as integrity_mod

    calls = {"n": 0}

    def fake_snapshot(_root):
        calls["n"] += 1
        if calls["n"] == 1:
            return dict(original)  # pre-run snapshot
        return {"guarded.bin": b"TAMPERED"}  # post-run state

    real_snapshot = integrity_mod.snapshot_protected_artifacts
    integrity_mod.snapshot_protected_artifacts = fake_snapshot
    try:
        violations = integrity_mod.verify_protected_artifacts(
            original, project_root=tmp_path
        )
    finally:
        integrity_mod.snapshot_protected_artifacts = real_snapshot
    assert violations, "tampering must be detected"


def test_verify_protected_artifacts_clean_when_unchanged(tmp_path):
    file = tmp_path / "guarded.bin"
    file.write_bytes(b"same")
    snapshot = {"guarded.bin": b"same"}
    from recsys23_fedrec.experiments import integrity as integrity_mod

    real_snapshot = integrity_mod.snapshot_protected_artifacts
    integrity_mod.snapshot_protected_artifacts = lambda _root: dict(snapshot)
    try:
        # Pass only live-diff keys (frozen-value checks expect Stage 8 keys,
        # which the monkeypatched snapshot does not provide; verify the diff
        # logic by injecting matching values).
        snapshot_full = dict(PROTECTED_VALUES)
        snapshot_full["stage6_checkpoint_sha256"] = (
            "25f3267eba572bf99e72c19eee2351236fe65327f71af7c7a8e39fae8e1d8617"
        )
        snapshot_full["stage6_checkpoint_mtime"] = 123.0
        snapshot_full["guarded.bin"] = b"same"
        integrity_mod.snapshot_protected_artifacts = lambda _root: dict(snapshot_full)
        violations = integrity_mod.verify_protected_artifacts(
            snapshot_full, project_root=tmp_path
        )
    finally:
        integrity_mod.snapshot_protected_artifacts = real_snapshot
    assert violations == []
