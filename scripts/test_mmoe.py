"""Real-data test for the Stage 5 Transformer + MMoE model.

Consumes Stage 3 batches from the processed Parquet training data (never the
raw CSVs, never the full dataset in RAM): forward shapes/diagnostics at batch
sizes 32 and 1024, gate normalization per sample, multi-task loss, backward/
gradient coverage across backbone/experts/gates/towers, a 3-step optimization
sanity check, parameter breakdown for later federated communication analysis,
CUDA forward/backward, and bounded memory. The official test split is only
opened to confirm schema compatibility (never trained on).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recsys23_fedrec.data import (  # noqa: E402
    FeatureSet,
    ProcessedRecSysDataset,
    collate_test,
    create_dataloader,
)
from recsys23_fedrec.models import (  # noqa: E402
    TransformerMMoE,
    mmoe_loss,
)

SEED = 42  # project seed (recsys23_fedrec.preprocessing.config.SEED)
failures: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("PASS  " if condition else "FAIL  ") + message)
    if not condition:
        failures.append(message)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def rss_mib() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts_dir", default=str(PROJECT_ROOT / "artifacts"))
    args = parser.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    torch.manual_seed(SEED)

    # ------------------------------------------------------------------
    section("A. Construction from the real Stage 2 artifact")
    feature_set = FeatureSet.load(artifacts_dir)
    model = TransformerMMoE(feature_set)
    check(model.mmoe_config.num_experts == 8, f"num_experts == 8 (got {model.mmoe_config.num_experts})")
    check(model.mmoe_config.num_tasks == 2, f"num_tasks == 2 (got {model.mmoe_config.num_tasks})")
    check(model.backbone.config.d_model == 128, f"d_model == 128 (got {model.backbone.config.d_model})")
    check(len(model.experts) == 8 and len(model.gates) == 2 and len(model.towers) == 2,
          "8 shared experts, 2 gates, 2 towers")
    vocab = [e.num_embeddings for e in model.backbone.embeddings]
    check(vocab == [int(v) for v in feature_set.vocabulary_sizes],
          "Stage 4 categorical embedding vocabularies unchanged")
    check(model.backbone.sequence_length == 93, f"sequence length == 93 (got {model.backbone.sequence_length})")

    # ------------------------------------------------------------------
    section("21. Parameter count and breakdown (federated communication cost input)")
    groups = {
        "transformer/tokenization (backbone)": dict(model.backbone.named_parameters()),
        "MMoE experts": {},
        "gates": {},
        "click tower": {},
        "install tower": {},
    }
    counts = {}
    for name, params in groups.items():
        counts[name] = sum(p.numel() for p in params.values()) if isinstance(params, dict) else 0
    # Experts/gates/towers individually.
    counts["MMoE experts"] = sum(p.numel() for e in model.experts for p in e.parameters())
    counts["gates"] = sum(p.numel() for g in model.gates for p in g.parameters())
    counts["click tower"] = sum(p.numel() for p in model.towers[0].parameters())
    counts["install tower"] = sum(p.numel() for p in model.towers[1].parameters())
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    counted = sum(counts.values())
    for name, count in counts.items():
        print(f"  {name:38s}: {count:>10,}")
    print(f"  {'TOTAL':38s}: {total:>10,}")
    check(counted == total, f"breakdown covers all parameters ({counted:,} == {total:,})")
    check(trainable == total, "all parameters trainable")
    stage4_backbone = counts["transformer/tokenization (backbone)"]
    print(f"  (Stage 4 baseline was 2,352,129; MMoE adds {total - stage4_backbone + stage4_backbone - stage4_backbone:,} head params on the headless backbone)")

    # ------------------------------------------------------------------
    section("20. Real-data forward pass (Stage 3 DataLoader)")
    rss_start = rss_mib()
    train_ds = ProcessedRecSysDataset(feature_set, artifacts_dir / "processed" / "train", split="train")
    print(f"dataset length: {len(train_ds):,} rows, {train_ds.num_parts} parts")
    print(f"RSS after dataset construction: {rss_mib():.1f} MiB (start {rss_start:.1f} MiB)")

    model.eval()
    for batch_size in (32, 1024):
        loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        batch = next(iter(loader))
        with torch.no_grad():
            out = model(batch)
            check(out["click_logit"].shape == (batch_size,), f"B={batch_size}: click_logit [{batch_size}]")
            check(out["install_logit"].shape == (batch_size,), f"B={batch_size}: install_logit [{batch_size}]")
            check(out["cls"].shape == (batch_size, 128), f"B={batch_size}: cls [{batch_size}, 128]")
            check(out["expert_outputs"].shape == (batch_size, 8, 128), f"B={batch_size}: expert_outputs [{batch_size}, 8, 128]")
            check(out["click_gate"].shape == (batch_size, 8), f"B={batch_size}: click_gate [{batch_size}, 8]")
            check(out["install_gate"].shape == (batch_size, 8), f"B={batch_size}: install_gate [{batch_size}, 8]")
            check(out["click_repr"].shape == (batch_size, 128), f"B={batch_size}: click_repr [{batch_size}, 128]")
            check(out["install_repr"].shape == (batch_size, 128), f"B={batch_size}: install_repr [{batch_size}, 128]")
            finite = all(torch.isfinite(v).all() for v in out.values())
            check(finite, f"B={batch_size}: all outputs finite")
            # 12. Gate normalization, per sample.
            for key in ("click_gate", "install_gate"):
                sums = out[key].sum(dim=-1)
                check(bool(torch.allclose(sums, torch.ones(batch_size), atol=1e-6)),
                      f"B={batch_size}: every {key} row sums to 1")
                check(bool(((out[key] >= -1e-6) & (out[key] <= 1 + 1e-6)).all()),
                      f"B={batch_size}: {key} weights within [0, 1]")
            # Sample-specific routing evidence on real data.
            row_spread = (out["click_gate"].amax(dim=-1) - out["click_gate"].amin(dim=-1)).mean()
            pair_diff = (out["click_gate"] - out["install_gate"]).abs().mean()
            print(f"  mean per-sample click-gate max-min spread: {row_spread:.4f}")
            print(f"  mean |click_gate - install_gate|:          {pair_diff:.4f}")

            # 8. Multi-task loss on real data.
            losses = mmoe_loss(out, batch)
            check(losses["click"].dim() == 0 and torch.isfinite(losses["click"]),
                  f"B={batch_size}: click BCEWithLogitsLoss finite scalar ({losses['click'].item():.4f})")
            check(losses["install"].dim() == 0 and torch.isfinite(losses["install"]),
                  f"B={batch_size}: install BCEWithLogitsLoss finite scalar ({losses['install'].item():.4f})")
            check(losses["total"].dim() == 0 and torch.isfinite(losses["total"]),
                  f"B={batch_size}: total loss finite scalar ({losses['total'].item():.4f})")
        batch_1024 = batch if batch_size == 1024 else None

    # ------------------------------------------------------------------
    section("16. Backward / gradient coverage (real batch, B=1024)")
    batch = batch_1024
    model.zero_grad(set_to_none=True)
    losses = mmoe_loss(model(batch), batch)
    t0 = time.perf_counter()
    losses["total"].backward()
    backward_s = time.perf_counter() - t0
    named = dict(model.named_parameters())
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    grads = [(n, p.grad) for n, p in named.items() if p.grad is not None]
    check(len(grads) == n_trainable, f"gradients exist for all {n_trainable} parameter tensors")
    check(all(torch.isfinite(g).all() for _, g in grads), "all gradients finite")
    n_zero = sum(1 for _, g in grads if g.abs().sum() == 0)
    print(f"backward time: {backward_s:.2f}s; zero-grad parameters: {n_zero}")
    # Component spot-checks.
    check(model.backbone.cls_token.grad is not None and model.backbone.cls_token.grad.abs().sum() > 0,
          "backbone CLS token has non-zero gradient")
    check(model.experts[0][0].weight.grad.abs().sum() > 0, "shared expert 0 has non-zero gradient")
    check(model.gates[0].weight.grad.abs().sum() > 0 and model.gates[1].weight.grad.abs().sum() > 0,
          "both gates have non-zero gradient")
    check(model.towers[0][3].weight.grad.abs().sum() > 0 and model.towers[1][3].weight.grad.abs().sum() > 0,
          "both towers have non-zero gradient")

    # ------------------------------------------------------------------
    section("17. Tiny optimization sanity (3 steps, NOT training)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses_seen = []
    ref_expert = model.experts[0][0].weight.detach().clone()
    ref_gate = model.gates[1].weight.detach().clone()
    it = iter(create_dataloader(train_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False))
    for step in range(3):
        b = next(it)
        optimizer.zero_grad()
        out = model(b)
        l = mmoe_loss(out, b)
        losses_seen.append((l["click"].item(), l["install"].item(), l["total"].item()))
        l["total"].backward()
        optimizer.step()
    changed = not torch.equal(ref_expert, model.experts[0][0].weight.detach()) and not torch.equal(
        ref_gate, model.gates[1].weight.detach()
    )
    finite_losses = all(all(torch.isfinite(torch.tensor(t)) for t in triple) for triple in losses_seen)
    check(finite_losses, f"3-step (click, install, total) losses finite: "
                         f"{[(round(c, 4), round(i, 4), round(t, 4)) for c, i, t in losses_seen]}")
    check(changed, "experts and gates changed after optimizer steps")
    print("note: sanity check only - not evidence of model quality")

    # ------------------------------------------------------------------
    section("14. Memory sanity (no_grad inference over 25 batches)")
    rss_mark = rss_mib()
    it = iter(create_dataloader(train_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False))
    peak = rss_mark
    n = 0
    model.eval()
    with torch.no_grad():
        for b in it:
            _ = model(b, return_diagnostics=False)
            n += 1
            peak = max(peak, rss_mib())
            if n >= 25:
                break
    print(f"RSS before 25 batches: {rss_mark:.1f} MiB; peak during: {peak:.1f} MiB (delta {peak - rss_mark:+.1f} MiB)")
    check(peak < 2500, "peak RSS stays bounded")

    # ------------------------------------------------------------------
    section("18. CUDA forward/backward (real batch)")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_d = model.to(device)
        b = next(iter(create_dataloader(train_ds, batch_size=256, shuffle=False,
                                        num_workers=0, pin_memory=True)))
        batch_d = {k: v.to(device) for k, v in b.items()}
        out = model_d(batch_d)
        check(bool(torch.isfinite(out["click_logit"]).all()) and bool(torch.isfinite(out["install_logit"]).all()),
              f"forward on {device}: finite click/install logits")
        losses = mmoe_loss(out, batch_d)
        check(torch.isfinite(losses["total"]), f"loss on {device}: finite ({losses['total'].item():.4f})")
        losses["total"].backward()
        grads_d = [p.grad for p in model_d.parameters() if p.grad is not None]
        check(bool(grads_d) and all(torch.isfinite(g).all() for g in grads_d),
              f"backward on {device}: finite gradients")
        model_d.zero_grad(set_to_none=True)
        model.cpu()
        print(f"device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA skipped: not available")

    # ------------------------------------------------------------------
    section("15. Official test split: schema compatibility only (no training)")
    test_ds = ProcessedRecSysDataset(feature_set, artifacts_dir / "processed" / "test", split="test")
    check(len(test_ds) == 160_973, f"test dataset opens with expected length (got {len(test_ds):,})")
    tb = collate_test([test_ds[i] for i in range(64)])
    model.eval()
    with torch.no_grad():
        out = model(tb)
    check(out["click_logit"].shape == (64,) and out["install_logit"].shape == (64,),
          "model accepts label-free test-split batch (schema identical)")
    print("official test set used ONLY for schema compatibility; not fitted or tuned on")

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: ALL REAL-DATA MMOE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
