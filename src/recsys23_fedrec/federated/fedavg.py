"""Stage 10A: standalone sample-weighted FedAvg aggregation utility.

Pure tensor mathematics over state dicts — **no training loop, no model
construction, no optimizer**. The Stage 10 baseline aggregates model states
with the standard sample-weighted FedAvg (McMahan et al., 2017):

    theta_global_next = sum_k (n_k / sum_j n_j) * theta_client_k

where ``n_k`` is the number of training samples assigned to client k.

Aggregation policy (verified against the actual TransformerMMoE
implementation, whose state dict is 138 entries, all ``torch.float32``; the
model contains no BatchNorm and therefore no running statistics or
``num_batches_tracked`` buffers):

* floating tensors: sample-weighted average, computed in float64 and cast
  back to the source dtype (protects against accumulation error over 10
  clients);
* non-floating integer/boolean tensors: this implementation **rejects** them
  by default (``on_non_floating="error"``). The Stage 10 model has none, so
  any such entry means the state dict does not belong to the frozen
  architecture. ``"first"`` is available for future architectures that
  legitimately carry counter buffers (e.g. BatchNorm ``num_batches_tracked``)
  where the conventional policy is to take the first client's value;
* inputs are never mutated: the weighted sum is accumulated into freshly
  allocated tensors;
* determinism: the weighted sum is accumulated in **client order
  (client_00 first)**, which makes the floating-point result exactly
  reproducible for identical inputs regardless of dict iteration order.

Incompatibility (missing/extra keys, shape/dtype mismatch) raises
:class:`FedAvgError`. Invalid client sample counts (n_k <= 0) raise
:class:`FedAvgError`.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

import torch

__all__ = ["FedAvgError", "fedavg_state_dicts", "fedavg_weights_from_counts"]


class FedAvgError(ValueError):
    """State dicts are incompatible or the sample counts are invalid."""


def fedavg_weights_from_counts(sample_counts: Sequence[int]) -> list[float]:
    """Exact normalized weights ``n_k / sum_j n_j`` with full validation."""
    counts = [int(c) for c in sample_counts]
    if len(counts) == 0:
        raise FedAvgError("no clients given")
    for k, c in enumerate(counts):
        if c <= 0:
            raise FedAvgError(f"client {k} has non-positive sample count {c}")
    total = sum(counts)
    if total <= 0:
        raise FedAvgError(f"total sample count is non-positive: {total}")
    return [c / total for c in counts]


def _validate_compatible(
    states: Sequence[Mapping[str, torch.Tensor]], reference_keys: list[str]
) -> None:
    for idx, state in enumerate(states):
        if not isinstance(state, Mapping):
            raise FedAvgError(f"client {idx} state is not a mapping")
        keys = list(state.keys())
        if keys != reference_keys:
            missing = sorted(set(reference_keys) - set(keys))
            extra = sorted(set(keys) - set(reference_keys))
            raise FedAvgError(
                f"client {idx} state keys differ from the first client "
                f"(missing={missing[:5]}, extra={extra[:5]})"
            )


def fedavg_state_dicts(
    states: Sequence[Mapping[str, torch.Tensor]],
    sample_counts: Sequence[int],
    *,
    on_non_floating: str = "error",
) -> "OrderedDict[str, torch.Tensor]":
    """Sample-weighted FedAvg over compatible state dicts.

    Args:
        states: one state dict per participating client (client order is
            significant for float reproducibility: accumulation starts from
            client_00's state).
        sample_counts: ``n_k`` per client (training rows assigned by the
            Stage 9 partition); must be positive.
        on_non_floating: ``"error"`` (default) rejects non-floating tensors;
            ``"first"`` copies the first client's value unchanged (the
            conventional policy for counter buffers such as BatchNorm
            ``num_batches_tracked``).

    Returns:
        A new ``OrderedDict`` with the aggregated state; inputs are never
        mutated and result tensors are freshly allocated.

    Raises:
        FedAvgError: empty input, unequal list lengths, invalid sample
            counts, incompatible keys/shapes/dtypes, or non-floating tensors
            under the default policy.
    """
    if on_non_floating not in ("error", "first"):
        raise FedAvgError(f"unknown on_non_floating policy {on_non_floating!r}")
    states = list(states)
    counts = [int(c) for c in sample_counts]
    if len(states) == 0:
        raise FedAvgError("no client states given")
    if len(states) != len(counts):
        raise FedAvgError(
            f"{len(states)} states but {len(counts)} sample counts"
        )

    weights = fedavg_weights_from_counts(counts)

    reference = states[0]
    if not isinstance(reference, Mapping):
        raise FedAvgError("client 0 state is not a mapping")
    reference_keys = list(reference.keys())
    if len(reference_keys) == 0:
        raise FedAvgError("reference state dict is empty")
    _validate_compatible(states, reference_keys)

    # Shape/dtype validation up front so an incompatible client fails before
    # any arithmetic happens.
    for idx, state in enumerate(states):
        for key in reference_keys:
            tensor = state[key]
            if not isinstance(tensor, torch.Tensor):
                raise FedAvgError(
                    f"client {idx} entry {key!r} is {type(tensor).__name__}, not a Tensor"
                )
            ref = reference[key]
            if tuple(tensor.shape) != tuple(ref.shape):
                raise FedAvgError(
                    f"client {idx} entry {key!r} shape {tuple(tensor.shape)} != "
                    f"reference {tuple(ref.shape)}"
                )
            if tensor.dtype != ref.dtype:
                raise FedAvgError(
                    f"client {idx} entry {key!r} dtype {tensor.dtype} != "
                    f"reference {ref.dtype}"
                )

    aggregated: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key in reference_keys:
        ref = reference[key]
        if not ref.dtype.is_floating_point:
            if on_non_floating == "first":
                # Fresh tensor; never alias the caller's storage.
                aggregated[key] = ref.detach().clone()
                continue
            raise FedAvgError(
                f"entry {key!r} has non-floating dtype {ref.dtype}; the frozen "
                "Stage 10 architecture has none — refusing to aggregate"
            )
        # float64 accumulation (client order 0..K-1), then cast back.
        acc = states[0][key].to(torch.float64) * weights[0]
        for idx in range(1, len(states)):
            acc = acc + states[idx][key].to(torch.float64) * weights[idx]
        aggregated[key] = acc.to(ref.dtype)
    return aggregated
