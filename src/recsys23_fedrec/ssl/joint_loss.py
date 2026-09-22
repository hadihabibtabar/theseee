"""Stage 7 SSL: joint supervised + contrastive objective.

``L_total = alpha * L_supervised + (1 - alpha) * L_contrastive``

Stage 7 SSL configuration: ``alpha = 0.6`` → ``0.6 * L_sup + 0.4 * L_contrastive``.

``L_supervised`` is exactly the Stage 5 MMoE objective (raw logits ->
``BCEWithLogitsLoss``, no sigmoid, no class weighting):

    L_sup = lambda_click * BCE(click_logit, click)
          + lambda_install * BCE(install_logit, install)

``L_contrastive`` is the symmetric NT-Xent loss over the two augmented views'
normalized projections; it never sees Click/Install labels. Keeping the joint
loss in one auditable module makes Stage 8 experiment variants (alpha sweeps)
configurable without touching the loss semantics.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor

from recsys23_fedrec.models import mmoe_loss
from recsys23_fedrec.ssl.contrastive_loss import (
    DEFAULT_TEMPERATURE,
    ContrastiveLossError,
    nt_xent_loss,
)

__all__ = ["DEFAULT_ALPHA", "JointLossResult", "ssl_joint_loss"]


class JointLossError(ValueError):
    """Invalid joint-loss configuration (alpha out of range)."""


#: Default supervised-loss weight for SSL joint training (Stage 7 spec).
DEFAULT_ALPHA = 0.6


def ssl_joint_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    *,
    alpha: float = DEFAULT_ALPHA,
    lambda_click: float = 1.0,
    lambda_install: float = 1.0,
    temperature: float = DEFAULT_TEMPERATURE,
    return_diagnostics: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Joint SSL objective for one training batch.

    Args:
        output: dual-view model output containing ``z1``/``z2`` projections
            (plus, for supervised training, ``view1``/``view2`` sub-dicts with
            ``click_logit``/``install_logit``; a single-view ``click_logit``/
            ``install_logit`` at the top level is also accepted).
        batch: original (uncorrupted) Stage 3 batch with ``click``/``install``
            labels — augmentation never alters labels, so views share them.
        alpha: supervised-loss weight in [0, 1] (contrastive weight = 1-alpha).
        lambda_click / lambda_install: Stage 5 MMoE loss weights.
        temperature: NT-Xent temperature (default 0.2).
        return_diagnostics: also return the supervised/contrastive components.

    Returns:
        Scalar joint loss — or ``(loss, components)`` with keys
        ``supervised``, ``contrastive``, ``click``, ``install``.
    """
    if not 0.0 <= alpha <= 1.0 or math.isnan(alpha):
        raise JointLossError(f"alpha must be in [0, 1], got {alpha}")

    # ---- supervised part: raw logits, BCEWithLogitsLoss (no sigmoid) ----
    if "view1" in output and "view2" in output:
        sup1 = mmoe_loss(
            output["view1"], batch,
            lambda_click=lambda_click, lambda_install=lambda_install,
        )
        sup2 = mmoe_loss(
            output["view2"], batch,
            lambda_click=lambda_click, lambda_install=lambda_install,
        )
        supervised = 0.5 * (sup1["total"] + sup2["total"])
        components: dict[str, Tensor] = {
            "click": 0.5 * (sup1["click"] + sup2["click"]),
            "install": 0.5 * (sup1["install"] + sup2["install"]),
        }
    elif "click_logit" in output and "install_logit" in output:
        sup = mmoe_loss(
            output, batch,
            lambda_click=lambda_click, lambda_install=lambda_install,
        )
        supervised = sup["total"]
        components = {"click": sup["click"], "install": sup["install"]}
    else:
        raise ContrastiveLossError(
            "output must provide view1/view2 sub-dicts with task logits "
            "(dual-view) or top-level click_logit/install_logit (single-view)"
        )

    # ---- contrastive part: NT-Xent over the two views' projections ------
    contrastive = nt_xent_loss(output["z1"], output["z2"], temperature=temperature)

    total = alpha * supervised + (1.0 - alpha) * contrastive
    if not torch.isfinite(total):
        raise ContrastiveLossError("joint loss is not finite; check model outputs")
    components.update({"supervised": supervised, "contrastive": contrastive})
    if return_diagnostics:
        return total, components
    return total
