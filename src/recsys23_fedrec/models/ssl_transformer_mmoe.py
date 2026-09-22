"""Stage 7: SSL-capable Transformer + MMoE (contrastive path on CLS).

Composition over the FROZEN Stage 4/5 architecture — the Transformer
backbone, MMoE experts/gates/towers and their semantics are reused verbatim
from :class:`~recsys23_fedrec.models.mmoe.TransformerMMoE`; nothing about the
Stage 6 model is rewritten. The only new component is the contrastive
projection head, which exists purely for SSL and never replaces the CLS
representation the MMoE heads consume::

    original batch x
        │
        ├── view1 = augment(x)   (independent corruption mask)
        └── view2 = augment(x)   (independent corruption mask)

    x^(v) → Transformer backbone → CLS h^(v) ∈ R^128
              ├──────────────────────→ MMoE → click_logit, install_logit
              └──→ projection MLP (128→128→GELU→64) → z^(v) (L2-normalized)

Dual-view forward returns ``z1``/``z2`` (contrastive) plus ``view1``/``view2``
sub-dicts with the task logits (supervised). The single-view
:meth:`supervised_forward` is exactly the Stage 6 behavior and is the path
used for supervised-only fine-tuning/evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from recsys23_fedrec.data.feature_schema import FeatureSet
from recsys23_fedrec.models.mmoe import MMoEConfig, TransformerMMoE
from recsys23_fedrec.models.transformer import TransformerBaseline, TransformerConfig

__all__ = ["SSLConfig", "SSLTransformerMMoE"]


class SSLConfigError(ValueError):
    """Invalid SSL model configuration."""


@dataclass(frozen=True)
class SSLConfig:
    """Contrastive-path hyperparameters (Stage 7 defaults per the spec)."""

    projection_dim: int = 64
    temperature: float = 0.2

    def __post_init__(self) -> None:
        if self.projection_dim <= 0:
            raise SSLConfigError(
                f"projection_dim must be positive, got {self.projection_dim}"
            )
        if not 0.0 < self.temperature < float("inf"):
            raise SSLConfigError(
                f"temperature must be positive, got {self.temperature}"
            )


class SSLTransformerMMoE(nn.Module):
    """Transformer + MMoE with an SSL projection head on the CLS output.

    Args:
        feature_set: frozen Stage 2 artifact (same contract as Stage 5/6).
        ssl_config: :class:`SSLConfig` (projection dim, temperature).
        backbone_config / mmoe_config: frozen Stage 4/5 hyperparameters.
        augmentations: optional pre-built augmentation module. When given,
            :meth:`forward` derives the two views from it (seed-consistent
            with the rest of the project); when omitted, only explicit-batch
            forwards (:meth:`forward_views`, :meth:`supervised_forward`) and
            projection computation are available.

    The projection head is ``Linear(128, 128) -> GELU -> Linear(128, 64)``
    followed by L2 normalization at the loss boundary. It is an ADD-ON:
    the MMoE heads keep consuming the original 128-d CLS representation.
    """

    def __init__(
        self,
        feature_set: FeatureSet,
        *,
        ssl_config: SSLConfig | None = None,
        backbone_config: TransformerConfig | None = None,
        mmoe_config: MMoEConfig | None = None,
        augmentations: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.feature_set = feature_set
        self.ssl_config = ssl_config or SSLConfig()
        self.augmentations = augmentations

        # Frozen Stage 5 model: backbone + MMoE heads, reused verbatim.
        self.mmoe = TransformerMMoE(
            feature_set,
            backbone_config=backbone_config,
            mmoe_config=mmoe_config,
        )
        d = self.mmoe.backbone.config.d_model

        # ---- projection head: 128 -> 128 -> GELU -> 64 --------------------
        self.projection = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, self.ssl_config.projection_dim),
        )
        nn.init.normal_(self.projection[0].weight, std=0.02)
        nn.init.zeros_(self.projection[0].bias)
        nn.init.normal_(self.projection[2].weight, std=0.02)
        nn.init.zeros_(self.projection[2].bias)

    # ------------------------------------------------------------------
    # Components (public for tests / analysis)
    # ------------------------------------------------------------------
    @property
    def backbone(self) -> TransformerBaseline:
        """The frozen Stage 4 Transformer backbone."""
        return self.mmoe.backbone

    def project(self, cls_repr: Tensor) -> Tensor:
        """CLS [B, d] -> L2-normalized projection z [B, projection_dim]."""
        z = self.projection(cls_repr)
        return F.normalize(z, p=2, dim=-1, eps=1e-12)

    # ------------------------------------------------------------------
    # Encoder reuse: tokenized once, two CLS passes
    # ------------------------------------------------------------------
    def forward_views(
        self,
        view1: Mapping[str, Tensor],
        view2: Mapping[str, Tensor],
        *,
        return_diagnostics: bool = False,
    ) -> dict[str, Tensor]:
        """Two ALREADY-AUGMENTED batches -> contrastive + supervised outputs.

        The two views are tokenized together as one [2B, 92, d] sequence and
        pass through the Transformer encoder as two independent rows (the
        clean interpretation of "each view processed independently"), then
        split back into per-view CLS representations. Each CLS feeds BOTH
        the MMoE heads (supervised) and the projection head (contrastive).
        """
        batch_size = int(view1["categorical"].shape[0])
        merged = {
            "categorical": torch.cat([view1["categorical"], view2["categorical"]], dim=0),
            "numerical": torch.cat([view1["numerical"], view2["numerical"]], dim=0),
            "binary": torch.cat([view1["binary"], view2["binary"]], dim=0),
            "missing": torch.cat([view1["missing"], view2["missing"]], dim=0),
        }
        encoded = self.mmoe.backbone.encode(merged)     # [2B, 93, d]
        cls_all = encoded[:, 0, :]                      # [2B, d]

        cls1, cls2 = cls_all[:batch_size], cls_all[batch_size:]
        z1, z2 = self.project(cls1), self.project(cls2)  # [B, projection_dim]
        out1 = self.mmoe.head_from_cls(cls1)             # raw logits, [B]
        out2 = self.mmoe.head_from_cls(cls2)

        output: dict[str, Tensor] = {
            "z1": z1,
            "z2": z2,
            "view1": out1,
            "view2": out2,
        }
        if return_diagnostics:
            output.update(
                {
                    "cls_view1": cls1,
                    "cls_view2": cls2,
                    "cls_all": cls_all,
                }
            )
        return output

    # ------------------------------------------------------------------
    def forward(
        self,
        batch: Mapping[str, Tensor],
        *,
        return_diagnostics: bool = False,
    ) -> dict[str, Tensor]:
        """Original batch -> dual-view SSL output (requires augmentations).

        Two INDEPENDENTLY corrupted views are generated with the configured
        augmentation module and pushed through :meth:`forward_views`.
        """
        if self.augmentations is None:
            raise SSLConfigError(
                "forward(batch) requires an augmentation module; pass one at "
                "construction or use forward_views/supervised_forward"
            )
        view1 = self.augmentations.augment(batch)
        view2 = self.augmentations.augment(batch)
        return self.forward_views(view1, view2, return_diagnostics=return_diagnostics)

    # ------------------------------------------------------------------
    def supervised_forward(
        self,
        batch: Mapping[str, Tensor],
        *,
        return_diagnostics: bool = True,
    ) -> dict[str, Tensor]:
        """Single-view supervised pass — exactly the Stage 6 behavior.

        No augmentation, no projection computation. The output contract
        (``click_logit``/``install_logit`` raw logits [B], diagnostics incl.
        ``cls`` [B, d]) is identical to
        :meth:`~recsys23_fedrec.models.mmoe.TransformerMMoE.forward`.
        """
        return self.mmoe(batch, return_diagnostics=return_diagnostics)
