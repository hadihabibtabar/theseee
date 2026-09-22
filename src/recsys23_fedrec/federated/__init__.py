"""Stage 9 federated layer: simulated-client data partitioning.

Public API::

    dirichlet_partition(labels, alpha=0.5, seed=42)   # 10 client index arrays
    load_train_labels("artifacts/processed/train")    # [N,2] (click, install)
    build_manifest(...) / summarize_partition(...)    # audit + statistics
    partition_hash(client_indices)                    # deterministic identity
"""

from .client_loader import (
    client_sampler,
    compose_client_row_indices,
    load_partition_positions,
    make_client_loader,
)
from .config import (
    FEDAVG_METHOD,
    LOCAL_BATCH_SIZE,
    LOCAL_EPOCHS,
    PARTICIPATION_ALL,
    ROUNDS,
    FederatedConfig,
    get_federated_config,
)
from .fedavg import FedAvgError, fedavg_state_dicts, fedavg_weights_from_counts
from .partition import (
    CLIENT_IDS,
    JOINT_STATES,
    NUM_CLIENTS,
    PARTITION_FORMAT_VERSION,
    PARTITION_METHOD,
    PLANNED_ALPHAS,
    PRIMARY_ALPHA,
    PartitionError,
    build_manifest,
    dirichlet_partition,
    load_train_labels,
    partition_hash,
    save_partition_arrays,
    summarize_partition,
    train_labels_hash,
)
from .stage12_matrix import (
    CENTRALIZED_CONTROLS,
    EXISTING_EXPERIMENTS,
    MATRIX_ALPHAS,
    NEW_EXPERIMENTS,
    STAGE12_EXPERIMENTS,
    Stage12ExperimentSpec,
    alpha_tag,
    experiment_id,
    get_experiment,
    get_partition_paths,
    matrix_summary,
    partition_name,
    stage12_output_dir,
    validate_stage12_matrix,
)

__all__ = [
    "load_partition_positions",
    "compose_client_row_indices",
    "make_client_loader",
    "client_sampler",
    "FederatedConfig",
    "get_federated_config",
    "FEDAVG_METHOD",
    "PARTICIPATION_ALL",
    "LOCAL_EPOCHS",
    "ROUNDS",
    "LOCAL_BATCH_SIZE",
    "FedAvgError",
    "fedavg_state_dicts",
    "fedavg_weights_from_counts",
    "NUM_CLIENTS",
    "CLIENT_IDS",
    "JOINT_STATES",
    "PLANNED_ALPHAS",
    "PRIMARY_ALPHA",
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
    "STAGE12_EXPERIMENTS",
    "Stage12ExperimentSpec",
    "MATRIX_ALPHAS",
    "EXISTING_EXPERIMENTS",
    "NEW_EXPERIMENTS",
    "CENTRALIZED_CONTROLS",
    "experiment_id",
    "partition_name",
    "alpha_tag",
    "get_experiment",
    "get_partition_paths",
    "stage12_output_dir",
    "validate_stage12_matrix",
    "matrix_summary",
]
