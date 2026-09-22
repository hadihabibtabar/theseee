"""Stage 7 augmentation tests: feature-corruption views (spec 7.8)."""

from __future__ import annotations

import pytest
import torch

from recsys23_fedrec.ssl import (
    BINARY_NEUTRAL_VALUE,
    CATEGORICAL_UNK_INDEX,
    NUMERICAL_NEUTRAL_VALUE,
    FeatureCorruptionAugmentation,
)
from recsys23_fedrec.ssl.augmentations import CorruptionMasks

pytestmark = pytest.mark.filterwarnings(
    "ignore: TypedStorage is deprecated:UserWarning"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_batch(b=8, n_cat=4, n_num=5, n_bin=3, n_mis=2, seed=42):
    g = torch.Generator().manual_seed(seed)
    return {
        "categorical": torch.randint(1, 7, (b, n_cat), generator=g),
        "numerical": torch.randn(b, n_num, generator=g),
        "binary": torch.randint(0, 2, (b, n_bin), generator=g).float(),
        "missing": torch.randint(0, 2, (b, n_mis), generator=g).float(),
        "click": torch.randint(0, 2, (b,), generator=g).float(),
        "install": torch.randint(0, 2, (b,), generator=g).float(),
    }


@pytest.fixture()
def aug(feature_set):
    return FeatureCorruptionAugmentation(feature_set, corruption_rate=0.15, seed=42)


# ---------------------------------------------------------------------------
# Shape / non-mutation / labels
# ---------------------------------------------------------------------------
def test_output_shape_equals_input_shape(aug, synthetic_batch):
    view, masks = aug.augment(synthetic_batch, return_mask=True)
    for group in ("categorical", "numerical", "binary", "missing"):
        assert view[group].shape == synthetic_batch[group].shape
        assert getattr(masks, group).shape == synthetic_batch[group].shape


def test_input_tensors_not_modified_in_place(aug, synthetic_batch):
    before = {k: v.clone() for k, v in synthetic_batch.items()}
    aug.augment(synthetic_batch, return_mask=True)
    aug.generate_two_views(synthetic_batch)
    for key, tensor in synthetic_batch.items():
        assert torch.equal(tensor, before[key]), f"input batch mutated: {key}"


def test_labels_unchanged_between_views(aug, synthetic_batch):
    view1, view2 = aug.generate_two_views(synthetic_batch)
    for key in ("click", "install"):
        assert torch.equal(view1[key], synthetic_batch[key])
        assert torch.equal(view2[key], synthetic_batch[key])


def test_masks_do_not_touch_label_keys(aug, synthetic_batch):
    view, masks = aug.augment(synthetic_batch, return_mask=True)
    assert "click" in view and "install" in view
    assert not hasattr(masks, "click") and not hasattr(masks, "install")


# ---------------------------------------------------------------------------
# Corruption semantics
# ---------------------------------------------------------------------------
def test_categorical_corruption_produces_valid_unk(feature_set):
    # UNK is index 0; vocabularies from the artifact always include it.
    aug = FeatureCorruptionAugmentation(feature_set, corruption_rate=1.0, seed=42)
    batch = _make_batch(n_cat=len(feature_set.categorical_names))
    batch["categorical"] = torch.randint(
        1, 5, (16, len(feature_set.categorical_names))
    )
    view, masks = aug.augment(batch, return_mask=True)
    assert masks.categorical.all(), "rate 1.0 must corrupt every categorical entry"
    assert bool((view["categorical"][masks.categorical] == CATEGORICAL_UNK_INDEX).all())
    # corrupted IDs remain inside every vocabulary (UNK == 0 is always valid)
    assert int(view["categorical"].min()) >= 0


def test_numerical_corruption_produces_documented_neutral(feature_set):
    aug = FeatureCorruptionAugmentation(feature_set, corruption_rate=1.0, seed=42)
    batch = _make_batch(n_num=len(feature_set.numerical_names))
    batch["numerical"] = torch.randn(16, len(feature_set.numerical_names)) * 3 + 1.5
    view, masks = aug.augment(batch, return_mask=True)
    assert masks.numerical.all()
    assert bool((view["numerical"][masks.numerical] == NUMERICAL_NEUTRAL_VALUE).all())


def test_binary_corruption_produces_neutral_value(feature_set):
    aug = FeatureCorruptionAugmentation(feature_set, corruption_rate=1.0, seed=42)
    batch = _make_batch(n_bin=len(feature_set.binary_names))
    batch["binary"] = torch.ones(16, len(feature_set.binary_names))
    view, masks = aug.augment(batch, return_mask=True)
    assert masks.binary.all()
    assert bool((view["binary"][masks.binary] == BINARY_NEUTRAL_VALUE).all())


def test_missing_indicators_preserved_by_default(feature_set):
    aug = FeatureCorruptionAugmentation(feature_set, corruption_rate=1.0, seed=42)
    batch = _make_batch(n_mis=len(feature_set.missing_indicator_names))
    batch["missing"] = torch.randint(0, 2, (16, len(feature_set.missing_indicator_names))).float()
    view, masks = aug.augment(batch, return_mask=True)
    assert torch.equal(view["missing"], batch["missing"]), (
        "missing indicators must be preserved by default (documented behavior)"
    )
    assert not masks.missing.any()


def test_missing_indicators_corruptable_when_enabled(feature_set):
    aug = FeatureCorruptionAugmentation(
        feature_set, corruption_rate=1.0, corrupt_missing=True, seed=42
    )
    batch = _make_batch(n_mis=len(feature_set.missing_indicator_names))
    batch["missing"] = torch.ones(16, len(feature_set.missing_indicator_names))
    view, masks = aug.augment(batch, return_mask=True)
    assert masks.missing.all()
    assert bool((view["missing"] == 0.0).all())


def test_zero_rate_is_identity(feature_set):
    aug = FeatureCorruptionAugmentation(feature_set, corruption_rate=0.0, seed=42)
    batch = _make_batch()
    view = aug.augment(batch)
    for key in ("categorical", "numerical", "binary", "missing"):
        assert torch.equal(view[key], batch[key])


def test_uncorrupted_entries_keep_original_values(feature_set):
    """Where mask is False the output equals the input (torch.where exactness)."""
    aug = FeatureCorruptionAugmentation(feature_set, corruption_rate=0.5, seed=7)
    batch = _make_batch(b=64)
    view, masks = aug.augment(batch, return_mask=True)
    keep = ~masks.categorical
    assert bool((view["categorical"][keep] == batch["categorical"][keep]).all())
    keep = ~masks.numerical
    assert bool((view["numerical"][keep] == batch["numerical"][keep]).all())


# ---------------------------------------------------------------------------
# Rate correctness / independence / determinism
# ---------------------------------------------------------------------------
def test_corruption_rate_approximately_correct(feature_set):
    rate = 0.15
    aug = FeatureCorruptionAugmentation(feature_set, corruption_rate=rate, seed=42)
    n = 4096
    batch = _make_batch(b=n)
    view, masks = aug.augment(batch, return_mask=True)
    for group in ("categorical", "numerical", "binary"):
        observed = getattr(masks, group).float().mean().item()
        # binomial stdev over 4096*cols draws is << 0.01; tolerance 0.02
        assert abs(observed - rate) < 0.02, f"{group}: observed {observed:.4f}"


def test_two_views_independently_sampled(feature_set):
    """Same input -> two views with (almost surely) different masks/values."""
    aug = FeatureCorruptionAugmentation(feature_set, corruption_rate=0.5, seed=42)
    batch = _make_batch(b=256)
    view1, view2 = aug.generate_two_views(batch)
    # with rate 0.5 over ~256*12 entries, identical corrupted sets are
    # astronomically unlikely; require actual differences in all groups
    for group in ("categorical", "numerical", "binary"):
        assert not torch.equal(view1[group], view2[group]), group


def test_deterministic_under_project_seed(feature_set):
    aug1 = FeatureCorruptionAugmentation(feature_set, corruption_rate=0.3, seed=42)
    aug2 = FeatureCorruptionAugmentation(feature_set, corruption_rate=0.3, seed=42)
    batch = _make_batch(b=32)
    v1a, v1b = aug1.generate_two_views(batch)
    v2a, v2b = aug2.generate_two_views(batch)
    for group in ("categorical", "numerical", "binary", "missing"):
        assert torch.equal(v1a[group], v2a[group])
        assert torch.equal(v1b[group], v2b[group])
    # fresh instances (reset step counter) reproduce the same stream
    aug3 = FeatureCorruptionAugmentation(feature_set, corruption_rate=0.3, seed=42)
    v3a, _ = aug3.generate_two_views(batch)
    assert torch.equal(v1a["categorical"], v3a["categorical"])


def test_explicit_generator_supported(feature_set):
    aug = FeatureCorruptionAugmentation(feature_set, corruption_rate=0.5, seed=42)
    batch = _make_batch(b=16)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    v1 = aug.augment(batch, generator=g1)
    v2 = aug.augment(batch, generator=g2)
    for group in ("categorical", "numerical", "binary"):
        assert torch.equal(v1[group], v2[group])


# ---------------------------------------------------------------------------
# Device behavior
# ---------------------------------------------------------------------------
def test_cpu_behavior(feature_set):
    aug = FeatureCorruptionAugmentation(feature_set, corruption_rate=0.25, seed=42)
    batch = _make_batch(b=16)
    view, masks = aug.augment(batch, return_mask=True)
    assert view["categorical"].device == batch["categorical"].device
    assert masks.categorical.dtype == torch.bool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_behavior(feature_set):
    aug = FeatureCorruptionAugmentation(feature_set, corruption_rate=0.25, seed=42)
    batch = _make_batch(b=16)
    cuda_batch = {k: v.cuda() for k, v in batch.items()}
    view = aug.augment(cuda_batch)
    assert view["categorical"].device.type == "cuda"
    assert view["numerical"].device.type == "cuda"
    assert torch.equal(view["click"], cuda_batch["click"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu_for_same_seed(feature_set):
    """CPU-side mask sampling makes CPU/CUDA views identical for one seed.

    Two FRESH instances are used: one instance deliberately advances its
    generator stream per generated view (independent views), so parity must
    be checked across identical configurations, not sequential calls.
    """
    batch = _make_batch(b=32)
    aug_cpu = FeatureCorruptionAugmentation(feature_set, corruption_rate=0.3, seed=42)
    aug_cuda = FeatureCorruptionAugmentation(feature_set, corruption_rate=0.3, seed=42)
    view_cpu, mask_cpu = aug_cpu.augment(batch, return_mask=True)
    view_cuda, mask_cuda = aug_cuda.augment(
        {k: v.cuda() for k, v in batch.items()}, return_mask=True
    )
    for group in ("categorical", "numerical", "binary", "missing"):
        assert torch.equal(view_cpu[group], view_cuda[group].cpu())
        assert torch.equal(getattr(mask_cpu, group), getattr(mask_cuda, group).cpu())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_invalid_rate_rejected(feature_set):
    with pytest.raises(ValueError):
        FeatureCorruptionAugmentation(feature_set, corruption_rate=1.5)
    with pytest.raises(ValueError):
        FeatureCorruptionAugmentation(feature_set, corruption_rate=-0.1)


def test_mask_type_alias():
    assert CorruptionMasks is not None
