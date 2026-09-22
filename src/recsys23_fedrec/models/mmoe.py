"""Stage 5: Transformer + MMoE multi-task model (Click + Install).

Extends the frozen Stage 4 architecture with a Multi-gate Mixture-of-Experts
layer on the shared CLS representation. The backbone (per-feature tokenization,
CLS, positional embedding, 6-layer Transformer encoder) is **reused verbatim**
from :class:`~recsys23_fedrec.models.transformer.TransformerBaseline` via its
public :meth:`~...TransformerBaseline.encode` path — no tokenization is
duplicated here and Stage 4 semantics are untouched.

Architecture (addition only)::

    CLS [B, 128]
      -> 8 shared experts (each 128 -> 128, GELU, Dropout)  -> [B, 8, 128]
      -> Click gate   (Linear 128->8 + softmax)             -> [B, 8]
      -> Install gate (Linear 128->8 + softmax)             -> [B, 8]
      -> per-sample weighted expert aggregation (einsum)
           click_repr [B, 128], install_repr [B, 128]
      -> Click tower (128 -> 128 -> 1)   -> click_logit [B]
      -> Install tower (128 -> 128 -> 1) -> install_logit [B]

Scientific note: MMoE reproduces the known multi-task architecture concept
(Ma et al., KDD 2018); it is not claimed as a novel contribution. This
implementation is a baseline for later controlled experiments (SSL, FL).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from recsys23_fedrec.data.feature_schema import FeatureSet
from recsys23_fedrec.models.transformer import (
    TransformerBaseline,
    TransformerConfig,
)

__all__ = ["MMoEConfig", "TransformerMMoE", "mmoe_loss"]


class MMoEConfigError(ValueError):
    """Invalid MMoE configuration."""


@dataclass(frozen=True)
class MMoEConfig:
    """MMoE hyperparameters (Stage 5 defaults per the spec)."""

    num_experts: int = 8
    num_tasks: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise MMoEConfigError(f"num_experts must be positive, got {self.num_experts}")
        if self.num_tasks <= 0:
            raise MMoEConfigError(f"num_tasks must be positive, got {self.num_tasks}")
        if not 0.0 <= self.dropout < 1.0:
            raise MMoEConfigError(f"dropout must be in [0, 1), got {self.dropout}")


def _build_tower(d_model: int, dropout: float) -> nn.Sequential:
    """Task tower: Linear(d, d) -> GELU -> Dropout -> Linear(d, 1)."""
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, 1),
    )


class TransformerMMoE(nn.Module):
    """Stage 4 Transformer backbone + MMoE head for Click and Install.

    Args:
        feature_set: frozen Stage 2 artifact (same object the Stage 4
            baseline consumes; vocabularies/order are not re-derived).
        backbone_config: optional explicit Stage 4 config. Defaults to the
            frozen Stage 4 hyperparameters (128/8/6/128/0.1).
        mmoe_config: optional :class:`MMoEConfig` (experts, tasks, dropout).

    The experts are shared across tasks; each task has its own gate and its
    own tower. Outputs are raw logits (no sigmoid anywhere in the forward
    path).
    """

    def __init__(
        self,
        feature_set: FeatureSet,
        *,
        backbone_config: TransformerConfig | None = None,
        mmoe_config: MMoEConfig | None = None,
    ) -> None:
        super().__init__()
        self.feature_set = feature_set
        self.mmoe_config = mmoe_config or MMoEConfig()
        backbone_config = backbone_config or TransformerConfig()

        # ---- Stage 4 backbone, reused verbatim (headless) -----------------
        self.backbone = TransformerBaseline(
            feature_set,
            d_model=backbone_config.d_model,
            nhead=backbone_config.nhead,
            num_layers=backbone_config.num_layers,
            dim_feedforward=backbone_config.dim_feedforward,
            dropout=backbone_config.dropout,
            include_install_head=False,
        )
        d = self.backbone.config.d_model

        # ---- 8 shared experts: 128 -> 128, GELU, Dropout ------------------
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d, d),
                    nn.GELU(),
                    nn.Dropout(self.mmoe_config.dropout),
                )
                for _ in range(self.mmoe_config.num_experts)
            ]
        )

        # ---- one gate per task (shared-expert routing) --------------------
        self.gates = nn.ModuleList(
            [nn.Linear(d, self.mmoe_config.num_experts) for _ in range(self.mmoe_config.num_tasks)]
        )

        # ---- one independent tower per task -------------------------------
        self.towers = nn.ModuleList(
            [_build_tower(d, self.mmoe_config.dropout) for _ in range(self.mmoe_config.num_tasks)]
        )

        self._init_mmoe_parameters()

    # ------------------------------------------------------------------
    @property
    def task_names(self) -> tuple[str, ...]:
        return ("click", "install")

    def _init_mmoe_parameters(self) -> None:
        """Init consistent with the Stage 4 parameter convention."""
        for expert in self.experts:
            nn.init.normal_(expert[0].weight, std=0.02)
            nn.init.zeros_(expert[0].bias)
        for gate in self.gates:
            nn.init.normal_(gate.weight, std=0.02)
            nn.init.zeros_(gate.bias)
        for tower in self.towers:
            nn.init.normal_(tower[0].weight, std=0.02)
            nn.init.zeros_(tower[0].bias)
            nn.init.normal_(tower[3].weight, std=0.02)
            nn.init.zeros_(tower[3].bias)

    # ------------------------------------------------------------------
    # Components (public for tests / analysis)
    # ------------------------------------------------------------------
    def compute_expert_outputs(self, cls_repr: Tensor) -> Tensor:
        """CLS [B, d] -> stacked shared expert outputs [B, num_experts, d]."""
        return torch.stack([expert(cls_repr) for expert in self.experts], dim=1)

    def compute_gate(self, task_index: int, cls_repr: Tensor) -> Tensor:
        """Per-sample task gate: CLS [B, d] -> softmax weights [B, num_experts]."""
        logits = self.gates[task_index](cls_repr)
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def aggregate_experts(gate_weights: Tensor, expert_outputs: Tensor) -> Tensor:
        """Weighted expert sum: ([B, E], [B, E, d]) -> [B, d].

        Vectorized einsum; no Python loop over samples.
        """
        return torch.einsum("be,bed->bd", gate_weights, expert_outputs)

    # ------------------------------------------------------------------
    def head_from_cls(
        self,
        cls_repr: Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> dict[str, Tensor]:
        """MMoE heads from a PRECOMPUTED CLS representation.

        Reuse path for models that own the encoder pass themselves
        (e.g. the Stage 7 SSL model projects/encodes views first and then
        applies this head). Semantics are identical to :meth:`forward`:
        raw logits, no sigmoid, task order click=0 / install=1.
        """
        expert_outputs = self.compute_expert_outputs(cls_repr)  # [B, E, d]

        # Task order: 0 = click, 1 = install (Stage 5 spec).
        gate_click = self.compute_gate(0, cls_repr)     # [B, E]
        gate_install = self.compute_gate(1, cls_repr)   # [B, E]
        repr_click = self.aggregate_experts(gate_click, expert_outputs)      # [B, d]
        repr_install = self.aggregate_experts(gate_install, expert_outputs)  # [B, d]

        click_logit = self.towers[0](repr_click).squeeze(-1)      # [B]
        install_logit = self.towers[1](repr_install).squeeze(-1)  # [B]

        output: dict[str, Tensor] = {
            "click_logit": click_logit,
            "install_logit": install_logit,
        }
        if return_diagnostics:
            output.update(
                {
                    "cls": cls_repr,
                    "expert_outputs": expert_outputs,
                    "click_gate": gate_click,
                    "install_gate": gate_install,
                    "click_repr": repr_click,
                    "install_repr": repr_install,
                }
            )
        return output

    # ------------------------------------------------------------------
    def forward(
        self,
        batch: Mapping[str, Tensor],
        *,
        return_diagnostics: bool = True,
    ) -> dict[str, Tensor]:
        """Stage 3 batch -> multi-task output dict.

        Always returns ``click_logit`` and ``install_logit`` (raw logits,
        shape [B]). With ``return_diagnostics=True`` (default) also returns
        ``cls``, ``expert_outputs``, ``click_gate``, ``install_gate``,
        ``click_repr``, ``install_repr`` for testing/analysis; set False in
        memory-critical training loops.
        """
        encoded = self.backbone.encode(batch)          # [B, 93, d]
        cls_repr = encoded[:, 0, :]                    # [B, d]
        return self.head_from_cls(cls_repr, return_diagnostics=return_diagnostics)


def mmoe_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    *,
    lambda_click: float = 1.0,
    lambda_install: float = 1.0,
) -> dict[str, Tensor]:
    """Stage 5 multi-task loss.

    ``L = lambda_click * BCE(click_logit, click) + lambda_install * BCE(install_logit, install)``

    Raw logits in, no sigmoid applied here (BCEWithLogitsLoss fuses it), no
    class weighting, no focal loss. Returns ``{"click": ..., "install": ...,
    "total": ...}`` so components stay observable in experiments.
    """
    click = nn.functional.binary_cross_entropy_with_logits(
        output["click_logit"], batch["click"]
    )
    install = nn.functional.binary_cross_entropy_with_logits(
        output["install_logit"], batch["install"]
    )
    total = lambda_click * click + lambda_install * install
    return {"click": click, "install": install, "total": total}
