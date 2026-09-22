"""Stage 9: reproducible Non-IID Dirichlet partition of the training rows.

Exactly 10 **simulated federated clients** with controlled Non-IID data
distributions are produced from the centralized training rows only
(3,137,266 rows). The centralized validation split (348,586 rows) stays
globally held out and the official test set is never touched: this module
operates purely on index arrays of the *train* portion of the Stage 3
Parquet dataset.

Index-space convention
----------------------
Client index arrays are **positions into the sorted centralized train index
array** (``0 .. 3,137,265``), so the partition is defined purely over "the
training rows as an ordered list". The later federated DataLoader composes
``dataset_row_indices = train_indices[client_positions]`` (the frozen split
NPZ supplies ``train_indices``). Validation/test rows never enter this space.

Mechanism (horizontal FL: same schema, same task, different local samples)
-------------------------------------------------------------------------
The Non-IID target is the **joint task-relevant label state**
``(is_clicked, is_installed)`` with the four observed states
``(0,0) (0,1) (1,0) (1,1)``. For each state, the rows exhibiting it are
split across the clients **proportionally to independent Dirichlet draws**
(concentration ``alpha`` per client), using a largest-remainder correction
so integer counts sum exactly to the state's row count. Chunks are cut
contiguously in ascending dataset order, clients walked in id order.

Determinism: one ``numpy.random.default_rng(seed)`` (PCG64) is consumed in
a fixed order — joint states in sorted canonical order
``(0,0), (0,1), (1,0), (1,1)`` — so identical seed + alpha + labels yield a
byte-identical partition. The primary Stage 9 partition uses
``alpha=0.5, seed=42``; ``alpha=1.0`` and ``alpha=0.1`` can be generated
reproducibly later without changing this code.

No rebalancing, mixing, or smoothing is applied after allocation — client
imbalance (size and label skew) is the intended, preserved Non-IID signal.

Memory: only label arrays (``[N, 2]`` int8) and int64 index arrays are
materialized — never feature rows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from recsys23_fedrec.preprocessing.config import SEED

__all__ = [
    "NUM_CLIENTS",
    "CLIENT_IDS",
    "JOINT_STATES",
    "PLANNED_ALPHAS",
    "PARTITION_FORMAT_VERSION",
    "PARTITION_METHOD",
    "PartitionError",
    "dirichlet_partition",
    "load_train_labels",
    "train_labels_hash",
    "partition_hash",
    "summarize_partition",
    "build_manifest",
    "save_partition_arrays",
]

#: Stage 9 protocol: exactly ten simulated federated clients.
NUM_CLIENTS = 10
#: Canonical client identifiers (``client_00`` .. ``client_09``).
CLIENT_IDS: tuple[str, ...] = tuple(f"client_{i:02d}" for i in range(NUM_CLIENTS))
#: Joint label states in the fixed canonical iteration order.
JOINT_STATES: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
#: Concentration parameters planned for later reproducible experiments
#: (only alpha=0.5 seed=42 is generated in Stage 9).
PLANNED_ALPHAS: tuple[float, ...] = (1.0, 0.5, 0.1)
#: Primary Stage 9 partition parameters.
PRIMARY_ALPHA = 0.5

PARTITION_FORMAT_VERSION = 1
PARTITION_METHOD = "dirichlet_joint_label_proportional_v1"

class PartitionError(ValueError):
    """Partition request is invalid or cannot produce 10 non-empty clients."""


def _index_hash(arrays: list[np.ndarray]) -> str:
    """Deterministic 16-hex hash over a sequence of int64 index arrays."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a, dtype=np.int64)
        h.update(len(a).to_bytes(8, "little", signed=False))
        h.update(a.tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def load_train_labels(
    processed_train_dir: str | Path,
    *,
    expected_total_rows: int | None = None,
) -> np.ndarray:
    """Read ``(is_clicked, is_installed)`` for every processed train row.

    Streams each ``part-*.parquet`` one at a time, reading only the two
    label columns, and returns an ``[N, 2]`` int8 array in dataset order.
    The Parquet parts carry the Stage 2 artifact hash in their key/value
    metadata; if present it is verified so the labels provably come from
    the frozen preprocessing artifact.

    Raises:
        PartitionError: part count/row count mismatch or artifact-hash
            mismatch.
    """
    import pyarrow.parquet as pq

    from recsys23_fedrec.data.dataset import SCHEMA_HASH_KEY

    parts = sorted(Path(processed_train_dir).glob("part-*.parquet"))
    if not parts:
        raise PartitionError(f"no processed train Parquet parts in {processed_train_dir}")

    from recsys23_fedrec.data import FeatureSet  # local import: avoids cycle at module load

    # Verify the Parquet parts against the frozen Stage 2 artifact when both
    # the linkage metadata and the artifact store are available (the real
    # repo layout); tiny synthetic test directories without an artifact skip
    # this check gracefully.
    expected_hash: str | None = None
    try:
        expected_hash = FeatureSet.load(
            Path(processed_train_dir).parents[1] / "preprocessing"
        ).artifact_hash
    except FileNotFoundError:
        expected_hash = None

    clicks: list[np.ndarray] = []
    installs: list[np.ndarray] = []
    for path in parts:
        meta = pq.read_metadata(path).metadata or {}
        kv = {k.decode("utf-8"): v.decode("utf-8") for k, v in meta.items()}
        linked = kv.get(SCHEMA_HASH_KEY)
        if expected_hash is not None and linked is not None and linked != expected_hash:
            raise PartitionError(
                f"{path.name}: artifact hash {linked!r} does not match the loaded "
                f"preprocessing artifact {expected_hash!r}"
            )
        table = pq.read_table(path, columns=["is_clicked", "is_installed"])
        clicks.append(table.column("is_clicked").to_numpy())
        installs.append(table.column("is_installed").to_numpy())

    click = np.concatenate(clicks).astype(np.int8, copy=False)
    install = np.concatenate(installs).astype(np.int8, copy=False)
    labels = np.stack([click, install], axis=1)

    if not np.isin(labels, (0, 1)).all():
        raise PartitionError("labels must be binary {0, 1}")
    if expected_total_rows is not None and len(labels) != expected_total_rows:
        raise PartitionError(
            f"train label rows {len(labels):,} != expected {expected_total_rows:,}"
        )
    return labels


def train_labels_hash(labels: np.ndarray) -> str:  # noqa: N802 (documented name)
    """Deterministic 16-hex identity of the label array used for partitioning."""
    flat = np.ascontiguousarray(labels, dtype=np.int8).reshape(-1)
    h = hashlib.sha256()
    h.update(len(labels).to_bytes(8, "little", signed=False))
    h.update(flat.tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Partition core
# ---------------------------------------------------------------------------
def dirichlet_partition(
    labels: np.ndarray,
    *,
    alpha: float,
    seed: int = SEED,
    num_clients: int = NUM_CLIENTS,
) -> list[np.ndarray]:
    """Partition training rows across clients (see module docstring).

    Args:
        labels: ``[N, 2]`` array of ``(is_clicked, is_installed)`` in
            dataset order (train rows only).
        alpha: Dirichlet concentration (> 0). Smaller values concentrate
            each joint state in fewer clients (stronger Non-IID).
        seed: project seed driving the PCG64 generator.
        num_clients: must be exactly :data:`NUM_CLIENTS` for the Stage 9
            protocol.

    Returns:
        List of ``num_clients`` sorted int64 arrays of dataset-row indices;
        together they cover ``range(N)`` exactly once. Client ``c``'s array
        pairs with :data:`CLIENT_IDS` ``[c]``.

    Raises:
        PartitionError: invalid alpha, wrong client count, empty resulting
            client, or non-binary labels.
    """
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise PartitionError(f"alpha must be a positive finite number, got {alpha!r}")
    if num_clients != NUM_CLIENTS:
        raise PartitionError(
            f"the Stage 9 protocol requires exactly {NUM_CLIENTS} clients, got {num_clients}"
        )
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise PartitionError(f"labels must have shape [N, 2], got {labels.shape}")
    if len(labels) == 0:
        raise PartitionError("cannot partition zero rows")
    if not np.isin(labels, (0, 1)).all():
        raise PartitionError("labels must be binary {0, 1}")
    if len(labels) < num_clients:
        raise PartitionError(
            f"{len(labels)} rows cannot populate {num_clients} non-empty clients"
        )

    n = len(labels)
    assignments = np.full(n, -1, dtype=np.int64)
    rng = np.random.default_rng(seed)

    click = labels[:, 0]
    install = labels[:, 1]
    for state in JOINT_STATES:  # fixed canonical order (determinism)
        positions = np.flatnonzero((click == state[0]) & (install == state[1]))
        k = positions.size
        if k == 0:
            continue  # state absent from the data: nothing to allocate
        props = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        # Largest-remainder rounding so integer counts sum exactly to k.
        counts = np.floor(props * k).astype(np.int64)
        remainder = int(k - counts.sum())
        if remainder > 0:
            frac_order = np.argsort(-(props * k - counts), kind="stable")
            counts[frac_order[:remainder]] += 1
        # Contiguous chunks in ascending dataset order, clients in id order.
        start = 0
        for client in range(num_clients):
            stop = start + int(counts[client])
            assignments[positions[start:stop]] = client
            start = stop

    client_indices: list[np.ndarray] = []
    for client in range(num_clients):
        rows = np.sort(np.flatnonzero(assignments == client))
        if rows.size == 0:
            raise PartitionError(
                f"alpha={alpha} produced an empty client_{client:02d}; increase alpha "
                "or reduce the number of clients (empty clients are rejected)"
            )
        client_indices.append(rows)

    # Coverage proof: every row assigned exactly once.
    total = sum(len(rows) for rows in client_indices)
    if total != n:
        raise PartitionError(f"internal error: allocated {total:,} of {n:,} rows")
    return client_indices


def partition_hash(client_indices: list[np.ndarray]) -> str:
    """Deterministic 16-hex identifier of the exact client index assignment."""
    arrays = [np.asarray(rows, dtype=np.int64) for rows in client_indices]
    if len(arrays) != NUM_CLIENTS:
        raise PartitionError(f"expected {NUM_CLIENTS} client index arrays, got {len(arrays)}")
    return _index_hash(arrays)


# ---------------------------------------------------------------------------
# Summary / manifest
# ---------------------------------------------------------------------------
def _joint_counts(labels: np.ndarray, rows: np.ndarray | None) -> dict[str, int]:
    if rows is None:
        sub = labels
    else:
        sub = labels[rows]
    out: dict[str, int] = {}
    for s in JOINT_STATES:
        key = f"click={s[0]},install={s[1]}"
        out[key] = int(np.count_nonzero((sub[:, 0] == s[0]) & (sub[:, 1] == s[1])))
    return out


def _client_summary(labels: np.ndarray, rows: np.ndarray, total: int) -> dict:
    sub = labels[rows]
    n = len(rows)
    joint = _joint_counts(labels, rows)
    return {
        "rows": n,
        "row_percentage": round(100.0 * n / total, 4),
        "click_count": int(sub[:, 0].sum()),
        "click_rate": round(float(sub[:, 0].mean()), 6),
        "install_count": int(sub[:, 1].sum()),
        "install_rate": round(float(sub[:, 1].mean()), 6),
        "joint_counts": joint,
        "joint_proportions": {k: round(v / n, 6) for k, v in joint.items()},
    }


def summarize_partition(
    client_indices: list[np.ndarray], labels: np.ndarray
) -> dict:
    """Descriptive statistics + a transparent heterogeneity measure.

    Heterogeneity: per-client *absolute deviation* of each joint-state
    proportion and row share from the global value (no ranking, no fancy
    index — auditable arithmetic).
    """
    if len(client_indices) != NUM_CLIENTS:
        raise PartitionError(f"expected {NUM_CLIENTS} clients, got {len(client_indices)}")
    total = len(labels)
    sizes = np.array([len(rows) for rows in client_indices], dtype=np.int64)

    global_joint = _joint_counts(labels, None)
    global_props = {k: v / total for k, v in global_joint.items()}
    clients = {
        CLIENT_IDS[c]: _client_summary(labels, client_indices[c], total)
        for c in range(NUM_CLIENTS)
    }

    deviations = {}
    for cid, summary in clients.items():
        devs = {
            f"delta_{key}": round(abs(summary["joint_proportions"][key] - global_props[key]), 6)
            for key in global_props
        }
        devs["delta_row_share"] = round(abs(summary["rows"] / total - 1.0 / NUM_CLIENTS), 6)
        deviations[cid] = devs
    all_joint_devs = [d[f"delta_{k}"] for d in deviations.values() for k in global_props]
    heterogeneity = {
        "definition": (
            "per-client absolute deviation of joint-state proportions and row share "
            "from the global values (sum over the 4 joint states of |client - global|)"
        ),
        "per_client_joint_abs_deviation_sum": {
            cid: round(sum(d[f"delta_{k}"] for k in global_props), 6)
            for cid, d in deviations.items()
        },
        "mean_absolute_joint_proportion_deviation": round(float(np.mean(all_joint_devs)), 6),
        "max_absolute_joint_proportion_deviation": round(float(np.max(all_joint_devs)), 6),
        "min_max_client_size": [int(sizes.min()), int(sizes.max())],
    }
    return {
        "total_rows": total,
        "global_label_distribution": {
            "click_rate": round(float(labels[:, 0].mean()), 6),
            "install_rate": round(float(labels[:, 1].mean()), 6),
            "joint_counts": global_joint,
            "joint_proportions": {k: round(v, 6) for k, v in global_props.items()},
        },
        "clients": clients,
        "heterogeneity": heterogeneity,
        "deviations": deviations,
    }


def build_manifest(
    client_indices: list[np.ndarray],
    labels: np.ndarray,
    *,
    alpha: float,
    seed: int,
    source_split_path: str,
    source_split_train_index_hash: str,
    labels_hash: str | None = None,
    notes: str | None = None,
) -> dict:
    """Assemble the full audit/reproduction manifest for one partition."""
    if labels_hash is None:
        labels_hash = train_labels_hash(labels)
    summary = summarize_partition(client_indices, labels)
    manifest = {
        "format_version": PARTITION_FORMAT_VERSION,
        "stage": "9-partition",
        "description": (
            "10 simulated federated clients with controlled Non-IID data "
            "distributions (horizontal FL: same feature schema, same task, "
            "different local samples)"
        ),
        "partition_method": PARTITION_METHOD,
        "partition_mechanism": (
            "for each joint label state (is_clicked, is_installed) in fixed order "
            "(0,0),(0,1),(1,0),(1,1): rows are split across clients proportionally "
            "to one Dirichlet(num_clients, alpha) draw per state (largest-remainder "
            "integer correction, counts sum exactly to the state row count); chunks "
            "are cut contiguously in ascending dataset order with clients in id order"
        ),
        "seed": int(seed),
        "alpha": float(alpha),
        "number_of_clients": NUM_CLIENTS,
        "client_ids": list(CLIENT_IDS),
        "source_split_path": source_split_path,
        "source_split_train_index_hash": source_split_train_index_hash,
        "labels_source": "artifacts/processed/train (is_clicked, is_installed columns)",
        "labels_hash": labels_hash,
        "total_training_rows": summary["total_rows"],
        "rows_per_client": [int(len(rows)) for rows in client_indices],
        "row_percentages": [summary["clients"][cid]["row_percentage"] for cid in CLIENT_IDS],
        "global_label_distribution": summary["global_label_distribution"],
        "clients": summary["clients"],
        "heterogeneity": summary["heterogeneity"],
        "client_index_hashes": {
            cid: _index_hash([client_indices[c]]) for c, cid in enumerate(CLIENT_IDS)
        },
        "partition_hash": partition_hash(client_indices),
        "rng": "numpy.random.default_rng (PCG64)",
        "determinism_note": (
            "identical seed + alpha + labels reproduce byte-identical client "
            "assignments; no post-hoc rebalancing is applied"
        ),
    }
    if notes:
        manifest["notes"] = notes
    return manifest


def save_partition_arrays(
    client_indices: list[np.ndarray],
    out_path: str | Path,
    *,
    partition_hash_value: str,
) -> Path:
    """Persist the compact client index mapping (NPZ, sorted int64 arrays).

    This is the mapping the later federated DataLoader consumes — raw rows
    are never duplicated on disk.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {cid: np.asarray(client_indices[c], dtype=np.int64) for c, cid in enumerate(CLIENT_IDS)}
    np.savez_compressed(out_path, **payload, partition_hash=np.array(partition_hash_value))
    return out_path
