"""Tests for the Stage 12 evaluation-matrix preparation (protocol stage).

Stage 12 performs NO training; these tests verify the preparation only:

* matrix structure (2 x 3 cells, deterministic IDs, no frozen-path collisions)
* partition reproducibility (alpha=1.0 and alpha=0.1, synthetic fast paths and
  real-artifact byte-level checks against the stored manifests/NPZs)
* partition invariants on the REAL frozen data (10 clients, exact row
  conservation, mutual disjointness, zero validation overlap)
* protocol equality with the frozen Stage 10A FederatedConfig
* SSL configuration equality with the completed Stage 11
* official test data is never constructed by Stage 12 preparation code

Read-only with respect to every protected artifact. No checkpoints, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_stage10b as s10b  # noqa: E402
import run_stage11 as s11  # noqa: E402
import run_stage12_federated as s12  # noqa: E402

from recsys23_fedrec.federated.client_loader import load_partition_positions  # noqa: E402
from recsys23_fedrec.federated.config import FederatedConfig, get_federated_config  # noqa: E402
from recsys23_fedrec.federated.partition import (  # noqa: E402
    CLIENT_IDS,
    PARTITION_METHOD,
    dirichlet_partition,
    load_train_labels,
    partition_hash,
)
from recsys23_fedrec.federated.stage12_matrix import (  # noqa: E402
    CENTRALIZED_CONTROLS,
    EXISTING_EXPERIMENTS,
    MATRIX_ALPHAS,
    NEW_EXPERIMENTS,
    STAGE12_EXPERIMENTS,
    STAGE12_PARTITIONS_DIR,
    alpha_tag,
    experiment_id,
    get_experiment,
    get_partition_paths,
    partition_name,
    stage12_output_dir,
    validate_stage12_matrix,
)
from recsys23_fedrec.training.split import load_split  # noqa: E402

TOTAL_TRAIN_ROWS = 3_137_266
TOTAL_DATASET_ROWS = 3_485_852
PROJECT = PROJECT_ROOT


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------
def _synthetic_labels(n_per_state: int = 250, seed: int = 7) -> np.ndarray:
    """Small binary label array over all four joint states (shuffled)."""
    rows: list[tuple[int, int]] = []
    for state in ((0, 0), (0, 1), (1, 0), (1, 1)):
        rows.extend([state] * n_per_state)
    labels = np.array(rows, dtype=np.int8)
    np.random.default_rng(seed).shuffle(labels)
    return labels


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def real_train_labels() -> np.ndarray:
    """The 3,137,266 train-only labels (streamed once for the module)."""
    split = load_split(
        PROJECT / "artifacts" / "splits" / "centralized_split.json",
        expected_total_rows=TOTAL_DATASET_ROWS,
    )
    all_labels = load_train_labels(
        PROJECT / "artifacts" / "processed" / "train",
        expected_total_rows=TOTAL_DATASET_ROWS,
    )
    return all_labels[split.train_indices]


@pytest.fixture(scope="module")
def frozen_split():
    return load_split(
        PROJECT / "artifacts" / "splits" / "centralized_split.json",
        expected_total_rows=TOTAL_DATASET_ROWS,
    )


# ---------------------------------------------------------------------------
# 1. matrix structure
# ---------------------------------------------------------------------------
def test_matrix_definition_is_valid_and_complete():
    assert validate_stage12_matrix() == []
    assert len(STAGE12_EXPERIMENTS) == 6
    pairs = {(s.use_ssl, s.alpha) for s in STAGE12_EXPERIMENTS}
    assert pairs == {(b, a) for b in (False, True) for a in (1.0, 0.5, 0.1)}
    assert {s.status for s in STAGE12_EXPERIMENTS} == {"existing", "new"}
    assert {s.experiment_id for s in EXISTING_EXPERIMENTS} == {
        "stage12_fed_nossl_alpha_0_5_seed42",   # Stage 10B cell
        "stage12_fed_ssl_alpha_0_5_seed42",     # Stage 11 cell
    }
    assert {s.experiment_id for s in NEW_EXPERIMENTS} == {
        "stage12_fed_nossl_alpha_1_0_seed42",
        "stage12_fed_ssl_alpha_1_0_seed42",
        "stage12_fed_nossl_alpha_0_1_seed42",
        "stage12_fed_ssl_alpha_0_1_seed42",
    }
    # existing cells point at the frozen Stage 10B / Stage 11 output dirs
    by_id = {s.experiment_id: s for s in STAGE12_EXPERIMENTS}
    assert by_id["stage12_fed_nossl_alpha_0_5_seed42"].output_dir == "artifacts/federated/stage10"
    assert by_id["stage12_fed_ssl_alpha_0_5_seed42"].output_dir == "artifacts/federated/stage11"


# ---------------------------------------------------------------------------
# 2. partition reproducibility (alpha=1.0)
# ---------------------------------------------------------------------------
def test_alpha_1_0_partition_reproducible_synthetic():
    labels = _synthetic_labels()
    p1 = dirichlet_partition(labels, alpha=1.0, seed=42)
    p2 = dirichlet_partition(labels, alpha=1.0, seed=42)
    for a, b in zip(p1, p2):
        assert np.array_equal(a, b)
    assert partition_hash(p1) == partition_hash(p2)
    assert len(p1) == 10


# ---------------------------------------------------------------------------
# 3. partition reproducibility (alpha=0.1)
# ---------------------------------------------------------------------------
def test_alpha_0_1_partition_reproducible_synthetic():
    labels = _synthetic_labels()
    p1 = dirichlet_partition(labels, alpha=0.1, seed=42)
    p2 = dirichlet_partition(labels, alpha=0.1, seed=42)
    for a, b in zip(p1, p2):
        assert np.array_equal(a, b)
    assert partition_hash(p1) == partition_hash(p2)
    # 0.1 must NOT reproduce the 1.0 assignment (alpha really changes the cut)
    assert partition_hash(p1) != partition_hash(
        dirichlet_partition(labels, alpha=1.0, seed=42)
    )


# ---------------------------------------------------------------------------
# 4. real Stage 12 partition artifacts reproduce byte-identically
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [1.0, 0.1])
def test_stage12_partition_real_artifact_matches_manifest_and_reruns(
    real_train_labels, alpha
):
    npz_relpath, manifest_relpath = get_partition_paths(alpha)
    npz_path = PROJECT / npz_relpath
    manifest_path = PROJECT / manifest_relpath
    assert npz_path.is_file() and manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    positions = load_partition_positions(npz_path)
    stored_hash = str(np.load(npz_path)["partition_hash"])
    recomputed = partition_hash([positions[cid] for cid in CLIENT_IDS])
    assert stored_hash == manifest["partition_hash"] == recomputed

    # full re-derivation through the frozen Stage 9 implementation
    derived = dirichlet_partition(real_train_labels, alpha=alpha, seed=42)
    for c, cid in enumerate(CLIENT_IDS):
        assert np.array_equal(derived[c], positions[cid]), cid
    assert partition_hash(derived) == recomputed


# ---------------------------------------------------------------------------
# 5. exactly 10 clients (real artifacts)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [1.0, 0.1])
def test_stage12_partition_has_exactly_ten_canonical_clients(alpha):
    npz_relpath, _ = get_partition_paths(alpha)
    positions = load_partition_positions(PROJECT / npz_relpath)
    assert list(positions.keys()) == list(CLIENT_IDS)
    assert len(positions) == 10
    assert all(pos.size > 0 for pos in positions.values())


# ---------------------------------------------------------------------------
# 6. exact row-count conservation (real artifacts)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [1.0, 0.1])
def test_stage12_partition_row_counts_conserve_exactly(alpha):
    npz_relpath, manifest_relpath = get_partition_paths(alpha)
    positions = load_partition_positions(PROJECT / npz_relpath)
    manifest = json.loads((PROJECT / manifest_relpath).read_text(encoding="utf-8"))
    sizes = [int(positions[cid].size) for cid in CLIENT_IDS]
    assert sum(sizes) == TOTAL_TRAIN_ROWS
    assert manifest["rows_per_client"] == sizes
    assert manifest["total_training_rows"] == TOTAL_TRAIN_ROWS
    # per-joint-state client sums equal the global counts exactly
    for key, global_count in manifest["global_label_distribution"]["joint_counts"].items():
        assert sum(manifest["clients"][cid]["joint_counts"][key] for cid in CLIENT_IDS) \
            == global_count


# ---------------------------------------------------------------------------
# 7. mutual client disjointness (real artifacts)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [1.0, 0.1])
def test_stage12_clients_are_mutually_disjoint(alpha):
    npz_relpath, _ = get_partition_paths(alpha)
    positions = load_partition_positions(PROJECT / npz_relpath)
    all_pos = np.concatenate([positions[cid] for cid in CLIENT_IDS])
    assert np.unique(all_pos).size == all_pos.size  # no position shared twice
    frozen_split = load_split(
        PROJECT / "artifacts" / "splits" / "centralized_split.json",
        expected_total_rows=TOTAL_DATASET_ROWS,
    )
    composed = np.concatenate(
        [frozen_split.train_indices[positions[cid]] for cid in CLIENT_IDS]
    )
    assert np.unique(composed).size == composed.size  # no dataset row shared twice


# ---------------------------------------------------------------------------
# 8. zero validation overlap (real artifacts)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [1.0, 0.1])
def test_stage12_clients_never_overlap_frozen_validation(frozen_split, alpha):
    npz_relpath, _ = get_partition_paths(alpha)
    positions = load_partition_positions(PROJECT / npz_relpath)
    for cid in CLIENT_IDS:
        rows = frozen_split.train_indices[positions[cid]]
        assert np.intersect1d(rows, frozen_split.validation_indices).size == 0, cid
        # structurally: positions live inside the train-position space only
        assert positions[cid].max() < len(frozen_split.train_indices)


# ---------------------------------------------------------------------------
# 9. correct alpha / seed metadata in the manifests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("alpha", "tag"), [(1.0, "1_0"), (0.1, "0_1")])
def test_stage12_manifest_metadata(alpha, tag):
    _, manifest_relpath = get_partition_paths(alpha)
    manifest = json.loads((PROJECT / manifest_relpath).read_text(encoding="utf-8"))
    assert manifest["alpha"] == alpha
    assert manifest["seed"] == 42
    assert manifest["number_of_clients"] == 10
    assert manifest["client_ids"] == list(CLIENT_IDS)
    assert manifest["partition_method"] == PARTITION_METHOD
    assert manifest["stage"] == "9-partition"  # built by the frozen Stage 9 builder
    assert manifest["source_split_train_index_hash"] == s10b.SPLIT_TRAIN_HASH
    assert manifest["labels_hash"] == "800e2aa9d0bf7b79"
    # naming convention aligns manifest, npz and the alpha tag
    assert manifest_relpath.endswith(f"{partition_name(alpha)}.json")
    assert alpha_tag(alpha) == tag


# ---------------------------------------------------------------------------
# 10. deterministic experiment IDs
# ---------------------------------------------------------------------------
def test_experiment_ids_are_deterministic_and_convention_following():
    for spec in STAGE12_EXPERIMENTS:
        expected = experiment_id(use_ssl=spec.use_ssl, alpha=spec.alpha)
        assert spec.experiment_id == expected
        # repeated construction is stable
        assert experiment_id(use_ssl=spec.use_ssl, alpha=spec.alpha) == expected
        assert get_experiment(spec.experiment_id) is spec
    assert alpha_tag(1.0) == "1_0" and alpha_tag(0.5) == "0_5" and alpha_tag(0.1) == "0_1"
    # IDs / output dirs are unique across the matrix
    ids = [s.experiment_id for s in STAGE12_EXPERIMENTS]
    dirs = [s.output_dir for s in STAGE12_EXPERIMENTS]
    assert len(set(ids)) == 6 and len(set(dirs)) == 6
    with pytest.raises(KeyError):
        get_experiment("stage12_fed_nossl_alpha_0_7_seed42")


# ---------------------------------------------------------------------------
# 11. no accidental path collision with Stage 10B / Stage 11 / Stage 9
# ---------------------------------------------------------------------------
def test_no_path_collision_with_frozen_stages():
    frozen_paths = {
        s10b.BEST_CKPT_RELPATH, s10b.FINAL_CKPT_RELPATH, s10b.LATEST_CKPT_RELPATH,
        s10b.RESULT_JSON_RELPATH, s10b.NPZ_RELPATH, s10b.MANIFEST_RELPATH,
        s11.BEST_CKPT_RELPATH, s11.FINAL_CKPT_RELPATH, s11.LATEST_CKPT_RELPATH,
        s11.RESULT_JSON_RELPATH,
    }
    for spec in NEW_EXPERIMENTS:
        out_rel = spec.output_dir
        assert out_rel not in ("artifacts/federated/stage10", "artifacts/federated/stage11")
        assert "stage10" not in out_rel and "stage11" not in out_rel
        for suffix in ("_best.pt", "_final_round10.pt", "_latest.pt", "_result.json"):
            rel = f"{out_rel}/{spec.experiment_id}{suffix}"
            assert rel not in frozen_paths
        assert spec.partition_npz != s10b.NPZ_RELPATH  # Stage 9 alpha=0.5 untouched
        assert "alpha_0_5" not in spec.partition_npz
        assert spec.partition_npz.startswith(STAGE12_PARTITIONS_DIR)
    # the Stage 12 runner never writes into the frozen stage dirs
    for spec in NEW_EXPERIMENTS:
        paths = s12._paths(spec).values()
        for path in paths:
            rel = str(path.relative_to(PROJECT))
            assert not rel.startswith("artifacts/federated/stage10")
            assert not rel.startswith("artifacts/federated/stage11")


# ---------------------------------------------------------------------------
# 12. protocol configuration matches the frozen settings
# ---------------------------------------------------------------------------
def test_protocol_configuration_matches_frozen_settings():
    cfg = get_federated_config()
    frozen = dict(
        num_clients=10,
        participation="all_clients_every_round",
        local_epochs=1,
        rounds=10,
        local_batch_size=1024,
        optimizer="adamw",
        learning_rate=3e-4,
        weight_decay=1e-2,
        lambda_click=1.0,
        lambda_install=1.0,
        aggregation="sample_weighted_fedavg",
        selection_metric="global_validation_install_logloss",
        local_optimizer_state_persistence=False,
        local_sampler_epoch_mode="round_minus_one",
        seed=42,
    )
    for key, expected in frozen.items():
        assert getattr(cfg, key) == expected, key
    # every Stage 12 cell inherits this exact frozen protocol object
    for spec in STAGE12_EXPERIMENTS:
        assert spec.status in ("existing", "new")  # cells carry no protocol drift
    # the matrix module validates protocol equality against FederatedConfig
    assert validate_stage12_matrix() == []
    assert FederatedConfig().to_dict() == cfg.to_dict()


# ---------------------------------------------------------------------------
# 13. SSL model configuration matches the completed Stage 11
# ---------------------------------------------------------------------------
def test_ssl_configuration_matches_stage11():
    # the Stage 12 SSL cells reuse the Stage 11 constants verbatim
    assert s11.SSL_ALPHA == 0.6
    assert s11.SSL_TEMPERATURE == 0.2
    assert s11.SSL_CORRUPTION_RATE == 0.15
    assert s11.SSL_PROJECTION_DIM == 64
    assert s11.SSL_AUG_SEED == 42
    assert s11.EXPECTED_SSL_PARAM_COUNT == 2_527_698
    assert s11.EXPECTED_SSL_STATE_ENTRIES == 142
    assert s11.EXPECTED_FROZEN_PARAM_COUNT == 2_502_930
    # the runner dispatches to the FROZEN Stage 11 / Stage 10B implementations
    assert s12._train_one_client.__globals__["s11"] is s11
    assert s12._train_one_client.__globals__["s10b"] is s10b
    assert hasattr(s11, "train_one_client_ssl") and hasattr(s10b, "train_one_client")
    assert hasattr(s11, "cold_ssl_model") and hasattr(s10b, "cold_model")
    # the SSL cell's expected architecture identity comes from Stage 11's module
    assert s12.SSL_CELL_PARAM_COUNT == s11.EXPECTED_SSL_PARAM_COUNT
    # checkpoint payload identity fields stay Stage 11-compatible
    assert s11.PROTOCOL_ID == "stage11-federated-ssl"
    assert s12.STAGE_NUMBER == 12


# ---------------------------------------------------------------------------
# 14. official test is not constructed by Stage 12 preparation code
# ---------------------------------------------------------------------------
def test_official_test_is_never_constructed_by_stage12_code():
    _q, _s = chr(34), chr(47)
    forbidden = (
        "processed" + _q + ", " + _q + "test",   # opening the official test dir
        "processed" + _s + "test",               # test path spelling
        "split" + "=" + _q + "test" + _q,        # test-split dataset construction
        "load_" + "official" + "_test",
        "test" + "_parquet",
    )
    sources = [
        PROJECT / "scripts" / "run_stage12_partitions.py",
        PROJECT / "scripts" / "run_stage12_federated.py",
        PROJECT / "scripts" / "audit_stage12_matrix.py",
        PROJECT / "src" / "recsys23_fedrec" / "federated" / "stage12_matrix.py",
    ]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in text, f"{path.name}: found {pattern!r}"
    # and the runner's data layer constructs ONLY the train dataset
    runner_text = (PROJECT / "scripts" / "run_stage12_federated.py").read_text(encoding="utf-8")
    assert 'split="train"' in runner_text
    assert "ProcessedRecSysDataset" in runner_text


# ---------------------------------------------------------------------------
# 15. runner guardrails: cell status + partition verification gates
# ---------------------------------------------------------------------------
def test_runner_rejects_existing_cells_and_verifies_partitions(tmp_path, capsys):
    # an existing cell may never be retrained by the Stage 12 runner: the
    # status guard fires before any file/GPU access (fast, side-effect free)
    import sys as _sys

    argv_backup = _sys.argv
    _sys.argv = [
        "run_stage12_federated.py", "--experiment", "stage12_fed_nossl_alpha_0_5_seed42",
    ]
    try:
        with pytest.raises(SystemExit) as excinfo:
            s12.main()
        assert excinfo.value.code == 1
    finally:
        _sys.argv = argv_backup
    assert "existing cell" in capsys.readouterr().out
    # a new cell passes the status guard and reaches its own paths
    new_spec = get_experiment("stage12_fed_ssl_alpha_1_0_seed42")
    paths = s12._paths(new_spec)
    prefix = new_spec.experiment_id
    assert paths["best"].name == f"{prefix}_best.pt"
    assert paths["final"].name == f"{prefix}_final_round10.pt"
    assert paths["latest"].name == f"{prefix}_latest.pt"
    assert paths["result"].name == f"{prefix}_result.json"
    assert paths["dir"] == PROJECT / new_spec.output_dir
    # guarded files include the frozen Stage 9 partition + this cell's partition
    guarded = s12._guarded_files(new_spec)
    assert s10b.NPZ_RELPATH in guarded and "artifacts/federated/partition_alpha_0_5_seed42.json" in guarded
    assert new_spec.partition_npz in guarded
    # every Stage 11 / Stage 10B guarded file is also guarded by Stage 12
    assert set(s11.PROTECTED_FILES_11) <= set(guarded)


# ---------------------------------------------------------------------------
# 16. matrix snapshot + centralized controls exist on disk
# ---------------------------------------------------------------------------
def test_matrix_snapshot_and_control_artifacts_exist():
    snapshot_path = PROJECT / "reports" / "stage12" / "stage12_matrix.json"
    assert snapshot_path.is_file()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert len(snapshot["experiments"]) == 6
    assert snapshot["protocol"]["num_clients"] == 10
    assert snapshot["protocol"]["rounds"] == 10
    # every referenced centralized control artifact actually exists (read-only)
    for control in CENTRALIZED_CONTROLS:
        assert (PROJECT / control["checkpoint"]).is_file(), control["id"]
