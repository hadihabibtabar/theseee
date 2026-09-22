"""Stage 7 SSL: NT-Xent / InfoNCE contrastive loss on normalized projections.

Setup: for a batch of ``B`` original impressions two views each are encoded,
giving ``2B`` L2-normalized projection vectors ``z``. Each view's positive is
the other view of the SAME impression; every other view in the 2B batch is a
negative; the view itself is never its own negative.

Standard symmetric formulation (Chen et al., SimCLR 2020; Oord et al., 2018)::

    l(i, j) = -log( exp(sim(z_i, z_j) / tau)
                    / sum_{k != i} exp(sim(z_i, z_k) / tau) )

    L_NT-Xent = (1 / 2B) * sum_{i=1}^{2B} l(i, p(i))    with p(i) = (i + B) mod 2B

with cosine similarity ``sim(z_i, z_j) = z_i . z_j`` (the projections are
L2-normalized, so the dot product IS the cosine similarity) and
``tau = temperature`` (default 0.2). Averaging over all ``2B`` rows makes the
objective exactly symmetric.

Numerical stability: the log-partition is computed with
``log_softmax`` over a row whose own (diagonal) entry is masked to ``-inf``
— no manual ``exp``/``log`` subtraction, no overflow for large similarities.

The loss is completely self-supervised: it never sees Click/Install labels.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor

__all__ = ["ContrastiveLossError", "DEFAULT_TEMPERATURE", "l2_normalize", "nt_xent_loss", "contrastive_loss"]

#: Default NT-Xent temperature (Stage 7 spec).
DEFAULT_TEMPERATURE = 0.2


class ContrastiveLossError(ValueError):
    """Invalid input to the contrastive loss (shape, dtype, or finiteness)."""


def l2_normalize(z: Tensor, eps: float = 1e-12) -> Tensor:
    """L2-normalize each row of ``z`` (over the last dim).

    ``clamp_min(eps)`` guards the all-zero vector (norm 0) against
    division by zero. Applying this to an already-normalized tensor is an
    (almost exact) no-op, so double normalization is harmless.
    """
    return z / z.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def _validate_projections(z1: Tensor, z2: Tensor) -> int:
    """Common shape/dtype/finiteness validation; returns the batch size B."""
    for name, z in (("z1", z1), ("z2", z2)):
        if not z.is_floating_point():
            raise ContrastiveLossError(f"{name} must be a float tensor, got {z.dtype}")
        if z.dim() != 2:
            raise ContrastiveLossError(
                f"{name} must be 2-D [batch, projection_dim], got shape {tuple(z.shape)}"
            )
    if z1.shape != z2.shape:
        raise ContrastiveLossError(
            f"z1 and z2 must have identical shapes, got {tuple(z1.shape)} vs {tuple(z2.shape)}"
        )
    b = z1.shape[0]
    if b < 2:
        raise ContrastiveLossError(
            f"contrastive loss requires batch size >= 2 (positive pairs + negatives), got {b}"
        )
    if z1.shape[1] < 1:
        raise ContrastiveLossError(f"projection dimension must be >= 1, got {z1.shape[1]}")
    if not (torch.isfinite(z1).all() and torch.isfinite(z2).all()):
        raise ContrastiveLossError("projections contain NaN/Inf; check the upstream model")
    return int(b)


def nt_xent_loss(
    z1: Tensor,
    z2: Tensor,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    return_diagnostics: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Symmetric NT-Xent / InfoNCE loss over ``B`` positive pairs.

    Args:
        z1: projections of view 1, ``[B, D]`` (any magnitude; normalized
            defensively inside — idempotent for already-normalized input).
        z2: projections of view 2, ``[B, D]``; row ``i`` of ``z2`` is the
            positive partner of row ``i`` of ``z1`` (alignment by construction:
            both views of impression ``i`` are produced from the same sample).
        temperature: softmax temperature ``tau > 0`` (default 0.2).
        return_diagnostics: additionally return the similarity matrix and
            per-view positive log-probabilities (detached; debug use).

    Returns:
        Scalar loss (mean over all ``2B`` views) — or ``(loss, diagnostics)``.
    """
    if not (temperature > 0.0) or math.isnan(temperature) or math.isinf(temperature):
        raise ContrastiveLossError(f"temperature must be a positive finite float, got {temperature}")
    b = _validate_projections(z1, z2)
    device = z1.device

    # ---- 2B views, cosine-similarity matrix ---------------------------
    z = torch.cat([z1, z2], dim=0)                       # [2B, D]
    z = l2_normalize(z)
    sim = z @ z.T                                        # [2B, 2B] cosine similarities
    sim = sim / temperature

    # ---- stable row-wise log-softmax, self excluded --------------------
    eye = torch.eye(2 * b, dtype=torch.bool, device=device)
    # exp(-inf) == 0, so the diagonal contributes nothing to the partition.
    log_prob = torch.log_softmax(sim.masked_fill(eye, float("-inf")), dim=1)  # [2B, 2B]

    # ---- positive index: view i pairs with view (i + B) mod 2B ---------
    view_index = torch.arange(2 * b, device=device)
    positive_index = (view_index + b) % (2 * b)
    positive_log_prob = log_prob[view_index, positive_index]                  # [2B]
    loss = -positive_log_prob.mean()

    if not torch.isfinite(loss):
        raise ContrastiveLossError(
            "contrastive loss is not finite; verify temperature and upstream projections"
        )
    if return_diagnostics:
        diagnostics = {
            "similarity_matrix": sim.detach(),
            "positive_log_prob": positive_log_prob.detach(),
        }
        return loss, diagnostics
    return loss


def contrastive_loss(
    output: Mapping[str, Tensor],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Tensor:
    """Convenience wrapper over :func:`nt_xent_loss` for model outputs.

    Accepts either ``{"z1": ..., "z2": ...}`` (SSL forward with two views) or
    ``{"z1_view1": ..., "z1_view2": ...}`` (dual-encoder forward).
    """
    if "z1" in output and "z2" in output:
        z1, z2 = output["z1"], output["z2"]
    elif "z1_view1" in output and "z1_view2" in output:
        z1, z2 = output["z1_view1"], output["z1_view2"]
    else:
        keys = sorted(output.keys())
        raise ContrastiveLossError(
            f"expected projection keys 'z1'/'z2' (or 'z1_view1'/'z1_view2') in output, got {keys}"
        )
    return nt_xent_loss(z1, z2, temperature=temperature)
