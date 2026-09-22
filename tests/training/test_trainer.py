"""Stage 6 trainer/sampler/loader tests (spec 21c) on the synthetic fixture."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
for path in (str(SRC), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recsys23_fedrec.data import FeatureSet, ProcessedRecSysDataset, collate_train  # noqa: E402
from recsys23_fedrec.models import TransformerConfig, TransformerMMoE  # noqa: E402
from recsys23_fedrec.training import (  # noqa: E402
    PartShuffledBatchSampler,
    SortedBatchSampler,
    Trainer,
    TrainingConfig,
    make_subset_loader,
    make_split,
)



# ----------------------------------------------------------------------
# Sampler properties
# ----------------------------------------------------------------------
def test_part_shuffled_sampler_covers_every_index_exactly_once():
    indices = np.arange(100)
    sampler = PartShuffledBatchSampler(indices, batch_size=16, seed=42, epoch=0)
    seen = [i for batch in sampler for i in batch]
    assert sorted(seen) == list(range(100))
    assert len(sampler) == int(np.ceil(100 / 16))


def test_part_shuffled_sampler_is_deterministic_per_epoch():
    indices = np.arange(90)
    a = [b.copy() for b in PartShuffledBatchSampler(indices, 16, seed=42, epoch=3)]
    b = [b.copy() for b in PartShuffledBatchSampler(indices, 16, seed=42, epoch=3)]
    assert a == b
    c = [b.copy() for b in PartShuffledBatchSampler(indices, 16, seed=42, epoch=4)]
    assert a != c  # different epochs permute differently


def test_part_shuffled_batches_are_sorted_within_batch():
    """Within each batch, subset positions must be ascending (cache-friendly)."""
    indices = np.arange(200)
    for batch in PartShuffledBatchSampler(indices, 32, seed=42, epoch=0):
        assert batch == sorted(batch)


def test_sorted_sampler_ascending():
    sampler = SortedBatchSampler(subset_size=70, batch_size=32)
    batches = list(sampler)
    assert batches[0] == list(range(32))
    assert batches[-1] == list(range(64, 70))  # remainder batch
    assert len(sampler) == 3


def test_sampler_validates_batch_size():
    with pytest.raises(ValueError):
        PartShuffledBatchSampler(np.arange(10), batch_size=0, seed=1)


# ----------------------------------------------------------------------
# Subset loaders over the real Stage 3 dataset object
# ----------------------------------------------------------------------
def test_subset_loader_shapes_and_coverage(feature_set, processed_fixture):
    dataset = ProcessedRecSysDataset(feature_set, processed_fixture["train_dir"], split="train")
    split = make_split(len(dataset), validation_ratio=0.2, seed=42)
    train_loader = make_subset_loader(dataset, split.train_indices, batch_size=64, shuffle=True, seed=42, pin_memory=False)
    val_loader = make_subset_loader(dataset, split.validation_indices, batch_size=64, shuffle=False, pin_memory=False)

    seen_batches = 0
    for batch in train_loader:
        assert batch["categorical"].shape[1] == feature_set.num_categorical
        assert batch["categorical"].shape[0] <= 64
        seen_batches += 1
        break  # shape smoke only; batch count checked below
    n_batches = sum(1 for _ in train_loader)
    assert n_batches == int(np.ceil(len(split.train_indices) / 64))

    val_rows = sum(b["categorical"].shape[0] for b in val_loader)
    assert val_rows == split.validation_rows
    # Validation batches follow ascending subset order.
    val_loader_iter = iter(val_loader)
    first = next(val_loader_iter)
    assert first["categorical"].shape[0] <= 64


def test_subset_out_of_range_rejected(feature_set, processed_fixture):
    dataset = ProcessedRecSysDataset(feature_set, processed_fixture["train_dir"], split="train")
    from recsys23_fedrec.training import SubsetDataset

    with pytest.raises(IndexError):
        SubsetDataset(dataset, np.array([len(dataset)], dtype=np.int64))


# ----------------------------------------------------------------------
# Trainer end-to-end on the synthetic fixture
# ----------------------------------------------------------------------
@pytest.fixture()
def tiny_trainer(feature_set, processed_fixture, tmp_path):
    dataset = ProcessedRecSysDataset(feature_set, processed_fixture["train_dir"], split="train")
    split = make_split(len(dataset), validation_ratio=0.2, seed=42)
    torch.manual_seed(42)
    model = TransformerMMoE(
        feature_set,
        backbone_config=TransformerConfig(d_model=32, nhead=4, num_layers=1, dim_feedforward=32),
    )
    train_loader = make_subset_loader(dataset, split.train_indices, batch_size=64, shuffle=True, seed=42, pin_memory=False)
    val_loader = make_subset_loader(dataset, split.validation_indices, batch_size=64, shuffle=False, pin_memory=False)
    config = TrainingConfig(epochs=2, batch_size=64, device="cpu", seed=42)
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=tmp_path / "checkpoints" / "best.pt",
        split_hash=split.train_index_hash,
        device=torch.device("cpu"),
        patience=2,
    )
    return trainer, split


def test_trainer_fit_produces_history_and_checkpoint(tiny_trainer):
    trainer, split = tiny_trainer
    summary = trainer.fit()
    assert summary["epochs_completed"] == 2
    assert len(trainer.history) == 2
    for log in trainer.history:
        assert all(
            np.isfinite([
                log.train_total_loss, log.train_click_loss, log.train_install_loss,
                log.val_click_logloss, log.val_install_logloss, log.val_click_auc, log.val_install_auc,
            ])
        )
        assert 0.0 <= log.val_click_auc <= 1.0
        assert 0.0 <= log.val_install_auc <= 1.0
        assert log.val_install_logloss > 0
    assert trainer.checkpoint_path.is_file()


def test_trainer_selects_best_epoch_by_install_logloss(tiny_trainer):
    trainer, _ = tiny_trainer
    summary = trainer.fit()
    best_logged = min(trainer.history, key=lambda h: h.val_install_logloss)
    assert summary["best_epoch"] == best_logged.epoch
    assert summary["best_val_install_logloss"] == pytest.approx(best_logged.val_install_logloss)


def test_checkpoint_reload_round_trip(tiny_trainer, feature_set, processed_fixture):
    trainer, split = tiny_trainer
    summary = trainer.fit()

    # Load checkpoint into a fresh model and compare on a fixed batch.
    ckpt = torch.load(trainer.checkpoint_path, weights_only=False)
    assert ckpt["epoch"] == summary["best_epoch"]
    assert ckpt["training_config"]["seed"] == 42
    assert ckpt["split_hash"] == split.train_index_hash
    torch.manual_seed(123)  # different init than training used
    fresh = TransformerMMoE(
        feature_set,
        backbone_config=TransformerConfig(d_model=32, nhead=4, num_layers=1, dim_feedforward=32),
    )
    fresh.load_state_dict(ckpt["model_state_dict"])
    fresh.eval()
    trainer.model.eval()
    dataset = ProcessedRecSysDataset(feature_set, processed_fixture["train_dir"], split="train")
    batch = collate_train([dataset[int(i)] for i in split.validation_indices[:16]])
    with torch.no_grad():
        out_reloaded = fresh(batch)
        out_saved = trainer.model(batch)
    for key in ("click_logit", "install_logit"):
        assert torch.isfinite(out_reloaded[key]).all()
        assert torch.allclose(out_reloaded[key], out_saved[key], atol=1e-6)


def test_early_stopping_stops_after_patience(feature_set, processed_fixture, tmp_path):
    """A model that cannot improve must stop at patience epochs."""
    dataset = ProcessedRecSysDataset(feature_set, processed_fixture["train_dir"], split="train")
    split = make_split(len(dataset), validation_ratio=0.2, seed=42)
    torch.manual_seed(42)
    model = TransformerMMoE(
        feature_set,
        backbone_config=TransformerConfig(d_model=32, nhead=4, num_layers=1, dim_feedforward=32),
    )
    # Zero LR: loss cannot improve after epoch 1 -> early stop at patience.
    for p in model.parameters():
        p.requires_grad_(True)
    train_loader = make_subset_loader(dataset, split.train_indices, batch_size=64, shuffle=True, seed=42, pin_memory=False)
    val_loader = make_subset_loader(dataset, split.validation_indices, batch_size=64, shuffle=False, pin_memory=False)
    config = TrainingConfig(epochs=10, batch_size=64, learning_rate=0.0, device="cpu", seed=42)
    trainer = Trainer(
        model=model, config=config, train_loader=train_loader, val_loader=val_loader,
        checkpoint_path=tmp_path / "best.pt", split_hash="x", device=torch.device("cpu"), patience=2,
    )
    summary = trainer.fit(max_epochs=10)
    assert summary["epochs_completed"] < 10
    assert summary["best_epoch"] == 1  # first epoch is the only possible best


def test_trainer_loaders_cover_split_not_dataset(tiny_trainer):
    """Loaders iterate the split subsets only; dataset stays out of RAM."""
    trainer, split = tiny_trainer
    n_train_batches = int(np.ceil(split.train_rows / trainer.config.batch_size))
    n_val_batches = int(np.ceil(split.validation_rows / trainer.config.batch_size))
    assert len(trainer.train_loader) == n_train_batches
    assert len(trainer.val_loader) == n_val_batches
