"""Stage 7 smoke test: tiny SSL training path on a REAL preprocessed batch.

Verifies the full contrastive + supervised mechanism end-to-end (spec 7.9):

 1. load real preprocessed batch (Stage 3 memory-bounded Parquet loader)
 2. generate view 1 / view 2 (independent feature corruption)
 3. run both views through the frozen Transformer backbone
 4. produce z1 / z2 via the projection head
 5. compute contrastive loss (NT-Xent, tau=0.2)
 6. compute supervised loss (Stage 5 MMoE objective, raw logits)
 7. compute joint loss (alpha 0.6)
 8. backward + optimizer step
 9. verify at least one trainable parameter changed
10. verify all losses are finite (no NaN/Inf)

Runs on CUDA when available. Uses only the memory-bounded Stage 3 loader and
a handful of batches; never touches the official test set for metrics and
never writes any checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch

from recsys23_fedrec.data import (
    FeatureSet,
    ProcessedRecSysDataset,
    create_dataloader,
)
from recsys23_fedrec.models import (
    MMoEConfig,
    SSLConfig,
    SSLTransformerMMoE,
    TransformerConfig,
)
from recsys23_fedrec.preprocessing.config import SEED
from recsys23_fedrec.ssl import (
    DEFAULT_ALPHA,
    DEFAULT_TEMPERATURE,
    FeatureCorruptionAugmentation,
    ssl_joint_loss,
)
from recsys23_fedrec.training.config import TrainingConfig
from recsys23_fedrec.training.sampler import PartShuffledBatchSampler


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 7 SSL smoke test")
    parser.add_argument("--artifacts_dir", default=str(PROJECT_ROOT / "artifacts"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_batches", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--corruption_rate", type=float, default=0.15)
    parser.add_argument("--skip_param_update", action="store_true",
                        help="skip the optimizer-step check (update-only mode)")
    args = parser.parse_args()

    started = time.perf_counter()
    device = resolve_device(args.device)
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ---- real data through the memory-bounded Stage 3 loader -----------
    feature_set = FeatureSet.load(Path(args.artifacts_dir) / "preprocessing")
    dataset = ProcessedRecSysDataset(
        feature_set, Path(args.artifacts_dir) / "processed" / "train", split="train"
    )
    print(f"real train dataset: {dataset.length:,} rows, {dataset.num_parts} parts")

    import numpy as np

    from recsys23_fedrec.data.collate import collate_train

    sampler = PartShuffledBatchSampler(
        np.arange(dataset.length, dtype=np.int64), batch_size=args.batch_size, seed=SEED
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, num_workers=0, collate_fn=collate_train
    )

    # ---- model + augmentation (frozen architecture, real artifact) -----
    model = SSLTransformerMMoE(
        feature_set,
        ssl_config=SSLConfig(projection_dim=64, temperature=DEFAULT_TEMPERATURE),
        backbone_config=TransformerConfig(),  # frozen 128/8/6/128/0.1
        mmoe_config=MMoEConfig(),             # frozen 8 experts
        augmentations=FeatureCorruptionAugmentation(
            feature_set, corruption_rate=args.corruption_rate, seed=SEED
        ),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SSL model parameters: {n_params:,} (Stage 6 backbone 2,502,930 + projection head)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    config = TrainingConfig()  # report reference only
    before_state = {
        n: p.detach().clone() for n, p in model.named_parameters()
    }

    # ---- tiny training loop --------------------------------------------
    model.train()
    all_losses: list[dict[str, float]] = []
    step = 0
    for batch in loader:
        if step >= args.n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch)  # dual-view: z1/z2 + view1/view2 logits
        total, comp = ssl_joint_loss(
            out, batch, alpha=DEFAULT_ALPHA, temperature=DEFAULT_TEMPERATURE,
            return_diagnostics=True,
        )
        total.backward()
        optimizer.step()

        finite = all(
            torch.isfinite(v) for v in (total, comp["supervised"], comp["contrastive"])
        )
        print(
            f"step {step}: joint {float(total):.4f} "
            f"(supervised {float(comp['supervised']):.4f} "
            f"[click {float(comp['click']):.4f} / install {float(comp['install']):.4f}] "
            f"| contrastive {float(comp['contrastive']):.4f}) | finite={bool(finite)}"
        )
        assert finite, "non-finite loss in smoke run"
        all_losses.append({k: float(v) for k, v in comp.items()})
        step += 1

    assert step == args.n_batches, f"expected {args.n_batches} steps, got {step}"

    # ---- verify optimizer actually updated parameters ------------------
    if not args.skip_param_update:
        changed = [
            n for n, p in model.named_parameters()
            if not torch.equal(p.detach(), before_state[n])
        ]
        print(f"parameters updated by optimizer: {len(changed)}")
        assert changed, "optimizer updated no parameters"
        for group in ("projection", "mmoe.backbone", "mmoe.experts", "mmoe.towers"):
            hit = next((n for n in changed if n.startswith(group)), None)
            print(f"  {group}: {'updated' if hit else 'NOT updated'}")

    # ---- projection sanity on one batch --------------------------------
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch, return_diagnostics=True)
        z_norms = out["z1"].norm(dim=-1)
        print(
            f"projection norms: mean {float(z_norms.mean()):.4f} "
            f"(should be ~1.0 after L2 normalization)"
        )
        assert torch.allclose(z_norms, torch.ones_like(z_norms), atol=1e-4)

    elapsed = time.perf_counter() - started
    if device.type == "cuda":
        print(f"peak GPU allocated: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")
    print(f"\nSTAGE 7 SSL SMOKE RUN PASSED in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
