"""Stage 4 Transformer baseline for the RecSys23 federated-installation project.

Single-task (is_installed) Transformer over per-feature tokens:

* every one of the 92 model features (30 categorical + 38 numerical +
  11 binary + 13 missing indicators) becomes its own ``d_model`` token,
* a learnable CLS token is prepended (sequence length 93),
* a learnable positional embedding carries positional information,
* a standard PyTorch Transformer encoder (6 layers, 8 heads) processes it,
* the CLS position feeds a small MLP producing one raw install logit.

The model performs NO preprocessing: categorical IDs are already encoded and
numerical values already normalized by the Stage 2 artifact (the single
source of truth for vocabulary sizes and feature order, supplied via
:class:`~recsys23_fedrec.data.feature_schema.FeatureSet`).

Public output contract::

    {"install_logit": Tensor[B], "cls": Tensor[B, d_model]}

``cls`` is returned so later stages (MMoE, SSL) can reuse the representation.
No sigmoid is applied — the raw logit pairs with BCEWithLogitsLoss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from recsys23_fedrec.data.feature_schema import FeatureSet

__all__ = [
    "TransformerConfig",
    "TransformerBaseline",
    "CategoricalIdError",
    "install_loss",
]


class CategoricalIdError(ValueError):
    """A categorical ID violates ``0 <= id < vocabulary_size``."""


@dataclass(frozen=True)
class TransformerConfig:
    """Transformer hyperparameters (Stage 4 defaults per the spec)."""

    d_model: int = 128
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.dim_feedforward <= 0:
            raise ValueError(
                f"dim_feedforward must be positive, got {self.dim_feedforward}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class TransformerBaseline(nn.Module):
    """Per-feature-token Transformer for install prediction (Stage 4).

    Args:
        feature_set: frozen Stage 2 artifact (feature order + vocabulary
            sizes). Never hard-coded here.
        **hyperparameters: see :class:`TransformerConfig` (d_model, nhead,
            num_layers, dim_feedforward, dropout).

    Derived (never hard-coded): ``num_feature_tokens`` = 30 + 38 + 11 + 13
    from the schema; ``sequence_length`` = 1 + num_feature_tokens (CLS).
    """

    def __init__(
        self,
        feature_set: FeatureSet,
        *,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        include_install_head: bool = True,
    ) -> None:
        super().__init__()
        self.feature_set = feature_set
        self.config = TransformerConfig(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        d = self.config.d_model

        # ---- per-feature vocabulary sizes from the Stage 2 artifact ------
        self.vocabularies: tuple[int, ...] = tuple(
            int(v) for v in feature_set.vocabulary_sizes
        )
        if not self.vocabularies or any(v <= 0 for v in self.vocabularies):
            raise ValueError(
                f"Stage 2 artifact has invalid vocabulary sizes: {self.vocabularies}"
            )
        self.num_categorical = len(self.vocabularies)
        self.num_numerical = len(feature_set.numerical_names)
        self.num_binary = len(feature_set.binary_names)
        self.num_missing = len(feature_set.missing_indicator_names)
        self.num_feature_tokens = (
            self.num_categorical + self.num_numerical + self.num_binary + self.num_missing
        )
        self.sequence_length = 1 + self.num_feature_tokens  # + CLS

        # ---- one embedding table per categorical feature ------------------
        self.embeddings = nn.ModuleList(
            [nn.Embedding(v, d) for v in self.vocabularies]
        )

        # ---- per-feature learnable affine projections --------------------
        # One token per scalar feature: x_j * w_j + b_j with w_j, b_j in R^d.
        # Binary/missing indicators use the same one-token-per-feature
        # projection (a 2-value embedding is equivalent but this keeps a
        # single uniform mechanism and handles {0,1} floats directly).
        self.numerical_weight = nn.Parameter(torch.empty(self.num_numerical, d))
        self.numerical_bias = nn.Parameter(torch.empty(self.num_numerical, d))
        self.binary_weight = nn.Parameter(torch.empty(self.num_binary, d))
        self.binary_bias = nn.Parameter(torch.empty(self.num_binary, d))
        self.missing_weight = nn.Parameter(torch.empty(self.num_missing, d))
        self.missing_bias = nn.Parameter(torch.empty(self.num_missing, d))

        # ---- CLS + learnable positional embedding ------------------------
        self.cls_token = nn.Parameter(torch.empty(1, 1, d))
        self.positional_embedding = nn.Parameter(
            torch.empty(1, self.sequence_length, d)
        )

        # ---- Transformer encoder (PyTorch standard components) -----------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.num_layers
        )

        # ---- install prediction head: d -> d -> 1 (raw logit) ------------
        # Optional so later stages (MMoE) can reuse the tokenization +
        # encoder backbone unchanged while replacing the single-task head.
        self.include_install_head = include_install_head
        if include_install_head:
            self.head = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, 1),
            )
        else:
            self.head = None

        self._init_parameters()

    # ------------------------------------------------------------------
    def _init_parameters(self) -> None:
        """Deterministic-init compatible init (call after torch.manual_seed)."""
        d = self.config.d_model
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)
        for w, b in (
            (self.numerical_weight, self.numerical_bias),
            (self.binary_weight, self.binary_bias),
            (self.missing_weight, self.missing_bias),
        ):
            nn.init.normal_(w, std=0.02)
            nn.init.zeros_(b)
        if self.include_install_head:
            nn.init.normal_(self.head[0].weight, std=0.02)
            nn.init.zeros_(self.head[0].bias)
            nn.init.normal_(self.head[2].weight, std=0.02)
            nn.init.zeros_(self.head[2].bias)
        # nn.Embedding / nn.Linear inside encoder keep PyTorch defaults,
        # which are seeded by the global torch RNG (project seed = 42).

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def _validate_categorical(self, categorical: Tensor) -> None:
        """Raise :class:`CategoricalIdError` unless every ID is in range.

        Vectorized: one min/max per column, compared against the per-feature
        vocabulary sizes from the Stage 2 artifact. Error messages name the
        offending feature.
        """
        if categorical.dim() != 2:
            raise CategoricalIdError(
                f"categorical must be [B, {self.num_categorical}], "
                f"got shape {tuple(categorical.shape)}"
            )
        if categorical.shape[1] != self.num_categorical:
            raise CategoricalIdError(
                f"categorical must have {self.num_categorical} columns "
                f"(Stage 2 artifact), got {categorical.shape[1]}"
            )
        if categorical.numel() == 0:
            return
        vocab = torch.tensor(self.vocabularies, dtype=torch.int64, device=categorical.device)
        min_ids = categorical.amin(dim=0)
        max_ids = categorical.amax(dim=0)
        neg = (min_ids < 0).nonzero().flatten().tolist()
        if neg:
            names = [self.feature_set.categorical_names[j] for j in neg]
            raise CategoricalIdError(
                f"negative categorical IDs for features {names} "
                f"(min values {[int(min_ids[j]) for j in neg]})"
            )
        over = (max_ids >= vocab).nonzero().flatten().tolist()
        if over:
            details = [
                f"{self.feature_set.categorical_names[j]} (column {j}): "
                f"max id {int(max_ids[j])} >= vocabulary size {int(vocab[j])}"
                for j in over
            ]
            raise CategoricalIdError(
                "categorical ID out of vocabulary range: " + "; ".join(details)
            )

    # ------------------------------------------------------------------
    # Tokenization (also the debug/test path for intermediate shapes)
    # ------------------------------------------------------------------
    def tokenize(
        self, batch: Mapping[str, Tensor], *, validate_inputs: bool = True
    ) -> dict[str, Tensor]:
        """Map a Stage 3 batch to per-group token tensors.

        Returns dict with:
            ``categorical`` [B, 30, d], ``numerical`` [B, 38, d],
            ``binary`` [B, 11, d], ``missing`` [B, 13, d],
            ``combined`` [B, 92, d] (concat in artifact order, pre-CLS).
        """
        categorical = batch["categorical"]
        if validate_inputs:
            self._validate_categorical(categorical)

        # 30 per-feature embedding tokens: [B, 30, d]
        cat_tokens = torch.stack(
            [self.embeddings[j](categorical[:, j]) for j in range(self.num_categorical)],
            dim=1,
        )
        # Per-feature affine projection: x_j * w_j + b_j -> [B, n, d]
        numerical = batch["numerical"].unsqueeze(-1) * self.numerical_weight.unsqueeze(0) + self.numerical_bias.unsqueeze(0)
        binary = batch["binary"].unsqueeze(-1) * self.binary_weight.unsqueeze(0) + self.binary_bias.unsqueeze(0)
        missing = batch["missing"].unsqueeze(-1) * self.missing_weight.unsqueeze(0) + self.missing_bias.unsqueeze(0)

        combined = torch.cat([cat_tokens, numerical, binary, missing], dim=1)
        return {
            "categorical": cat_tokens,
            "numerical": numerical,
            "binary": binary,
            "missing": missing,
            "combined": combined,
        }

    # ------------------------------------------------------------------
    def encode(
        self,
        batch: Mapping[str, Tensor],
        *,
        validate_inputs: bool = True,
    ) -> Tensor:
        """Stage 3 batch -> encoder output ``[B, sequence_length, d]``.

        Public reuse path: runs tokenization + CLS + positional embedding +
        the Transformer encoder without any task head. MMoE (Stage 5) builds
        on this and extracts ``[:, 0, :]`` as the shared CLS representation.
        """
        tokens = self.tokenize(batch, validate_inputs=validate_inputs)  # [B, 92, d]
        batch_size = tokens["combined"].shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)                 # [B, 1, d]
        sequence = torch.cat([cls, tokens["combined"]], dim=1)         # [B, 93, d]
        sequence = sequence + self.positional_embedding                 # [B, 93, d]
        return self.encoder(sequence)                                   # [B, 93, d]

    def forward(
        self,
        batch: Mapping[str, Tensor],
        *,
        return_tokens: bool = False,
    ) -> dict[str, Tensor]:
        """Stage 3 batch -> ``{"install_logit": [B], "cls": [B, d]}``.

        ``return_tokens=True`` additionally returns the full encoder output
        ``[B, sequence_length, d]`` (debug path for shape verification).
        """
        encoded = self.encode(batch)                         # [B, 93, d]
        cls_output = encoded[:, 0, :]                        # [B, d]
        if self.head is not None:
            install_logit = self.head(cls_output).squeeze(-1)  # [B]
        else:
            install_logit = None

        output = {"cls": cls_output}
        if install_logit is not None:
            output["install_logit"] = install_logit
        if return_tokens:
            output["encoded_sequence"] = encoded
        return output


def install_loss(output: Mapping[str, Tensor], batch: Mapping[str, Tensor]) -> Tensor:
    """Minimal Stage 4 loss: BCEWithLogitsLoss(install_logit, install).

    Takes raw logits (no sigmoid here — BCEWithLogitsLoss fuses it).
    Click and SSL losses are deliberately absent until later stages.
    """
    return nn.functional.binary_cross_entropy_with_logits(
        output["install_logit"], batch["install"]
    )
