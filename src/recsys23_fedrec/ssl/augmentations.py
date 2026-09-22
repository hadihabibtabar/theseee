"""Stage 7 SSL: feature-corruption augmentation for contrastive views.

For each original impression ``x`` the augmentation produces a corrupted copy
``x^(v)`` that is fed to the Transformer as an independent view. Two
independently sampled corruptions of the same impression form a positive
pair for the NT-Xent objective (see ``contrastive_loss.py``).

Augmentation definition ( operates on the ALREADY-preprocessed Stage 2
representation; raw CSVs and preprocessing artifacts are never touched or
refit):

====================  ==========================================  =========
Feature group         Corruption                                  Neutral
====================  ==========================================  =========
categorical [30]      replace token with UNK                      index 0
numerical    [38]     replace value with neutral                  0.0
                      (0 in the already-standardized space)
binary       [11]     replace value with neutral                  0.0
missing      [13]     PRESERVED by default (semantic meaning:     unchanged
                      "value was missing" must not be lied about);
                      optional corruption to 0.0 via flag
====================  ==========================================  =========

Each feature entry of each sample is corrupted independently with
probability ``corruption_rate`` (default 0.15). View 1 and view 2 draw
**independent** corruption masks, so ``augment(x)`` normally yields two
different views of the same impression.

Labels are never part of the corruption: ``click`` / ``install`` pass
through unchanged (the views share the original label tensors), so the
supervised loss on either view trains toward the true impression label.

Reproducibility: the project has exactly one seed convention
(``preprocessing.config.SEED = 42``). The augmentation uses per-call
``torch.Generator`` objects derived from that seed (``SEED + view_offset +
step``); it never relies on ambient global RNG state, so identical seeds
reproduce identical views regardless of surrounding random consumption.
An explicit ``torch.Generator`` may be passed for full control (tests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from recsys23_fedrec.data.feature_schema import FeatureSet
from recsys23_fedrec.preprocessing.config import SEED

__all__ = ["CorruptionMasks", "FeatureCorruptionAugmentation"]

#: Neutral token for corrupted categorical entries: index 0 == UNK.
CATEGORICAL_UNK_INDEX = 0
#: Neutral value for corrupted numerical entries: 0 in the standardized space.
NUMERICAL_NEUTRAL_VALUE = 0.0
#: Neutral value for corrupted binary entries.
BINARY_NEUTRAL_VALUE = 0.0
#: Neutral value for corrupted missing indicators (only when enabled).
MISSING_NEUTRAL_VALUE = 0.0


@dataclass(frozen=True)
class CorruptionMasks:
    """Boolean corruption masks for one augmented view (debug/testing).

    ``True`` == the entry was replaced by its neutral representation.
    Shapes follow the batch contract: [B, num_features] per group.
    """

    categorical: Tensor
    numerical: Tensor
    binary: Tensor
    missing: Tensor


class FeatureCorruptionAugmentation(nn.Module):
    """Feature-corruption augmentation over preprocessed Stage 3 batches.

    Args:
        feature_set: frozen Stage 2 artifact (feature-group sizes come from
            the artifact, never hard-coded).
        corruption_rate: per-entry Bernoulli corruption probability.
        corrupt_missing: also corrupt missing indicators (default ``False``;
            their semantic meaning is preserved unless explicitly enabled).
        seed: project seed for the internal generator stream (one convention;
            do not invent a second seed system).

    The augmentation never modifies the input batch in place: every output
    tensor is a fresh tensor. Label keys (``click`` / ``install``) pass
    through as the original tensors, guaranteeing identical labels.
    """

    def __init__(
        self,
        feature_set: FeatureSet,
        *,
        corruption_rate: float = 0.15,
        corrupt_missing: bool = False,
        seed: int = SEED,
    ) -> None:
        super().__init__()
        if not 0.0 <= corruption_rate <= 1.0 or math.isnan(corruption_rate):
            raise ValueError(
                f"corruption_rate must be in [0, 1], got {corruption_rate}"
            )
        self.feature_set = feature_set
        self.corruption_rate = float(corruption_rate)
        self.corrupt_missing = bool(corrupt_missing)
        self.seed = int(seed)
        self._step = 0  # advances per generated view; keeps streams distinct

    # ------------------------------------------------------------------
    # Mask sampling
    # ------------------------------------------------------------------
    def _sample_mask(
        self,
        shape: tuple[int, ...],
        generator: torch.Generator,
        device: torch.device,
    ) -> Tensor:
        """Independent Bernoulli(corruption_rate) mask of ``shape`` on ``device``.

        Draws happen on the CPU (masks are tiny [B, n] booleans) and are
        moved to ``device`` afterwards: ``torch.bernoulli`` requires the
        generator device to match the tensor device, and CPU-side sampling
        keeps the randomness identical across CPU/CUDA runs.
        """
        if self.corruption_rate == 0.0:
            return torch.zeros(shape, dtype=torch.bool, device=device)
        if self.corruption_rate == 1.0:
            return torch.ones(shape, dtype=torch.bool, device=device)
        probs = torch.full(shape, self.corruption_rate, dtype=torch.float32)
        return torch.bernoulli(probs, generator=generator).to(torch.bool).to(device)

    def _next_generator(self) -> torch.Generator:
        """Fresh CPU generator from the single project seed convention."""
        generator = torch.Generator(device="cpu")
        # Distinct, deterministic stream per generated view.
        generator.manual_seed(self.seed + 1_000_003 * (self._step + 1))
        self._step += 1
        return generator

    # ------------------------------------------------------------------
    # Corruption
    # ------------------------------------------------------------------
    def _corrupt_group(
        self,
        values: Tensor,
        mask: Tensor,
        neutral: float,
    ) -> Tensor:
        """Return a NEW tensor with ``values[mask] = neutral`` (no in-place)."""
        return torch.where(
            mask, torch.full_like(values, neutral), values
        )

    def augment(
        self,
        batch: Mapping[str, Tensor],
        *,
        generator: torch.Generator | None = None,
        return_mask: bool = False,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], CorruptionMasks]:
        """Generate one corrupted view of ``batch``.

        Args:
            batch: Stage 3 batch contract (``categorical`` int64 [B, 30],
                ``numerical``/``binary``/``missing`` float32 [B, n], optional
                ``click``/``install`` labels).
            generator: optional explicit ``torch.Generator`` (CPU) for the
                Bernoulli draws; when omitted, a deterministic per-call
                generator derived from the project seed is used.
            return_mask: also return the :class:`CorruptionMasks`.

        Returns:
            The view dict (same keys as the batch; fresh feature tensors,
            original label tensors) — optionally with masks attached.
        """
        categorical = batch["categorical"]
        device = categorical.device
        if generator is None:
            generator = self._next_generator()

        cat_mask = self._sample_mask(categorical.shape, generator, device)
        num_mask = self._sample_mask(batch["numerical"].shape, generator, device)
        bin_mask = self._sample_mask(batch["binary"].shape, generator, device)
        if self.corrupt_missing:
            mis_mask = self._sample_mask(batch["missing"].shape, generator, device)
        else:
            # Masks always describe what was ACTUALLY corrupted: with missing
            # preservation enabled (default) nothing is corrupted there.
            mis_mask = torch.zeros(
                batch["missing"].shape, dtype=torch.bool, device=device
            )

        view: dict[str, Tensor] = {
            "categorical": self._corrupt_group(
                categorical, cat_mask, CATEGORICAL_UNK_INDEX
            ),
            "numerical": self._corrupt_group(
                batch["numerical"], num_mask, NUMERICAL_NEUTRAL_VALUE
            ),
            "binary": self._corrupt_group(
                batch["binary"], bin_mask, BINARY_NEUTRAL_VALUE
            ),
            "missing": batch["missing"]
            if not self.corrupt_missing
            else self._corrupt_group(
                batch["missing"], mis_mask, MISSING_NEUTRAL_VALUE
            ),
        }
        # Labels are NEVER altered: pass the original tensors through.
        for label_key in ("click", "install"):
            if label_key in batch:
                view[label_key] = batch[label_key]

        if not return_mask:
            return view
        masks = CorruptionMasks(
            categorical=cat_mask, numerical=num_mask, binary=bin_mask, missing=mis_mask
        )
        return view, masks

    # Alias matching the spec's ``augment(x) -> x_v`` phrasing.
    forward = augment

    # ------------------------------------------------------------------
    def generate_two_views(
        self, batch: Mapping[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Two INDEPENDENTLY corrupted views of the same impression.

        Uses two distinct deterministic generator streams, so the masks of
        view 1 and view 2 differ with probability ~1 (for rate in (0, 1)).
        """
        view1 = self.augment(batch)
        view2 = self.augment(batch)
        return view1, view2

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"corruption_rate={self.corruption_rate}, "
            f"corrupt_missing={self.corrupt_missing}, seed={self.seed}"
        )
