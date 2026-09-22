"""Stage 8: SSL-capable single-task Transformer (projection head on CLS).

Mirrors :class:`~recsys23_fedrec.models.ssl_transformer_mmoe.SSLTransformerMMoE`
but wraps the Stage 4 Transformer with its install head (no MMoE), isolating
the SSL contribution without multi-task learning (Experiment C).

Composition — the Stage 4 backbone + install head are reused verbatim; the
projection head is an add-on for contrastive learning only::

    x^(v) → Transformer backbone → CLS h^(v) ∈ R^128
              ├──────────────────────→ install head → install_logit
              └──→ projection MLP (128→128→GELU→64) → z^(v) (L2-normalized)

Dual-view forward returns ``z1``/``z2`` plus ``view1``/``view2`` sub-dicts
with the raw install logit. :meth:`supervised_forward` is the exact Stage 4
path. The projection architecture is intentionally identical to the MMoE SSL
variant (same 128→128→GELU→64 head) so Experiments C and D differ only in the
presence of MMoE.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from recsys23_fedrec.data.feature_schema import FeatureSet
from recsys23_fedrec.models.ssl_transformer_mmoe import SSLConfig, SSLConfigError
from recsys23_fedrec.models.transformer import TransformerBaseline, TransformerConfig

__all__ = ["SSLTransformerBaseline"]


class SSLTransformerBaseline(nn.Module):
    """Stage 4 Transformer (install head) + SSL projection head.

    Args:
        feature_set: frozen Stage 2 artifact (same contract as Stage 4).
        ssl_config: :class:`SSLConfig` (projection dim, temperature).
        backbone_config: optional explicit Stage 4 config (defaults frozen).
        augmentations: optional augmentation module; required for the
            dual-view :meth:`forward`, not for :meth:`supervised_forward`.
    """

    def __init__(
        self,
        feature_set: FeatureSet,
        *,
        ssl_config: SSLConfig | None = None,
        backbone_config: TransformerConfig | None = None,
        augmentations: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.feature_set = feature_set
        self.ssl_config = ssl_config or SSLConfig()
        self.augmentations = augmentations

        backbone_config = backbone_config or TransformerConfig()
        self.base = TransformerBaseline(
            feature_set,
            d_model=backbone_config.d_model,
            nhead=backbone_config.nhead,
            num_layers=backbone_config.num_layers,
            dim_feedforward=backbone_config.dim_feedforward,
            dropout=backbone_config.dropout,
            include_install_head=True,
        )
        d = self.base.config.d_model

        # ---- projection head: 128 -> 128 -> GELU -> 64 (identical to MMoE SSL)
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
    @property
    def backbone(self) -> TransformerBaseline:
        """The frozen Stage 4 Transformer backbone."""
        return self.base

    def project(self, cls_repr: Tensor) -> Tensor:
        """CLS [B, d] -> L2-normalized projection z [B, projection_dim]."""
        z = self.projection(cls_repr)
        return F.normalize(z, p=2, dim=-1, eps=1e-12)

    def install_logit_from_cls(self, cls_repr: Tensor) -> Tensor:
        """Frozen Stage 4 install head applied to a CLS representation."""
        return self.base.head(cls_repr).squeeze(-1)          # [B]

    # ------------------------------------------------------------------
    def forward_views(
        self,
        view1: Mapping[str, Tensor],
        view2: Mapping[str, Tensor],
        *,
        return_diagnostics: bool = False,
    ) -> dict[str, Tensor]:
        """Two ALREADY-AUGMENTED batches -> contrastive + supervised outputs.

        Both views are tokenized together as one [2B, 92, d] sequence (two
        independent encoder rows), then split back into per-view CLS
        representations; each CLS feeds the install head and the projection.
        """
        batch_size = int(view1["categorical"].shape[0])
        merged = {
            "categorical": torch.cat([view1["categorical"], view2["categorical"]], dim=0),
            "numerical": torch.cat([view1["numerical"], view2["numerical"]], dim=0),
            "binary": torch.cat([view1["binary"], view2["binary"]], dim=0),
            "missing": torch.cat([view1["missing"], view2["missing"]], dim=0),
        }
        encoded = self.base.encode(merged)              # [2B, 93, d]
        cls_all = encoded[:, 0, :]                      # [2B, d]
        cls1, cls2 = cls_all[:batch_size], cls_all[batch_size:]

        z1, z2 = self.project(cls1), self.project(cls2)  # [B, projection_dim]
        out1 = {"install_logit": self.install_logit_from_cls(cls1)}
        out2 = {"install_logit": self.install_logit_from_cls(cls2)}

        output: dict[str, Tensor] = {"z1": z1, "z2": z2, "view1": out1, "view2": out2}
        if return_diagnostics:
            output.update({"cls_view1": cls1, "cls_view2": cls2})
        return output

    # ------------------------------------------------------------------
    def forward(
        self,
        batch: Mapping[str, Tensor],
        *,
        return_diagnostics: bool = False,
    ) -> dict[str, Tensor]:
        """Original batch -> dual-view SSL output (requires augmentations)."""
        if self.augmentations is None:
            raise SSLConfigError(
                "forward(batch) requires an augmentation module; pass one at "
                "construction or use supervised_forward"
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
        """Single-view supervised pass — exactly the Stage 4 behavior."""
        encoded = self.base.encode(batch)               # [B, 93, d]
        cls_repr = encoded[:, 0, :]                     # [B, d]
        output: dict[str, Tensor] = {
            "install_logit": self.install_logit_from_cls(cls_repr),
        }
        if return_diagnostics:
            output["cls"] = cls_repr
        return output
