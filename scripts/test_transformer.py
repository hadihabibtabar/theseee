"""Real-data test for the Stage 4 Transformer baseline.

Consumes Stage 3 batches from the processed Parquet training data (never the
raw CSVs, never the full dataset in RAM) and verifies the model end-to-end:
forward shapes/finite outputs at batch sizes 32 and 1024, intermediate token
shapes, BCEWithLogitsLoss, backward/gradient sanity, a 3-step tiny
optimization sanity check, CPU/CUDA execution, deterministic construction,
parameter counts, and bounded memory. The official test split is only
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
    CategoricalIdError,
    TransformerBaseline,
    install_loss,
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
    model = TransformerBaseline(feature_set)
    check(model.num_categorical == 30 and model.num_numerical == 38
          and model.num_binary == 11 and model.num_missing == 13,
          f"token groups from schema: cat={model.num_categorical}, num={model.num_numerical}, "
          f"bin={model.num_binary}, mis={model.num_missing}")
    check(model.num_feature_tokens == 92, f"num_feature_tokens == 92 (got {model.num_feature_tokens})")
    check(model.sequence_length == 93, f"sequence_length == 1 + 92 == 93 (got {model.sequence_length})")
    vocab = [emb.num_embeddings for emb in model.embeddings]
    expected_vocab = [int(v) for v in feature_set.vocabulary_sizes]
    check(vocab == expected_vocab, "all 30 embedding tables match Stage 2 vocabulary sizes")
    print(f"embedding sizes: {vocab}")

    # ------------------------------------------------------------------
    section("13. Parameter count")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total parameters:     {total:,}")
    print(f"trainable parameters: {trainable:,}")
    check(trainable == total, "all parameters are trainable (no frozen components in Stage 4)")
    for group, count in (
        ("categorical embeddings", sum(e.weight.numel() for e in model.embeddings)),
        ("numerical projections", model.numerical_weight.numel() + model.numerical_bias.numel()),
        ("binary projections", model.binary_weight.numel() + model.binary_bias.numel()),
        ("missing projections", model.missing_weight.numel() + model.missing_bias.numel()),
        ("CLS + positional", model.cls_token.numel() + model.positional_embedding.numel()),
        ("encoder", sum(p.numel() for p in model.encoder.parameters())),
        ("head", sum(p.numel() for p in model.head.parameters())),
    ):
        print(f"  {group:24s}: {count:,}")

    # ------------------------------------------------------------------
    section("B/C/D. Real-data forward pass (Stage 3 DataLoader)")
    rss_start = rss_mib()
    train_ds = ProcessedRecSysDataset(feature_set, artifacts_dir / "processed" / "train", split="train")
    print(f"dataset length: {len(train_ds):,} rows, {train_ds.num_parts} parts")
    print(f"RSS after dataset construction: {rss_mib():.1f} MiB (start {rss_start:.1f} MiB)")

    model.eval()
    for batch_size in (32, 1024):
        loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        batch = next(iter(loader))
        with torch.no_grad():
            output = model(batch)
            check(output["install_logit"].shape == (batch_size,),
                  f"B={batch_size}: install_logit.shape == [{batch_size}]")
            check(output["cls"].shape == (batch_size, 128),
                  f"B={batch_size}: cls.shape == [{batch_size}, 128]")
            check(bool(torch.isfinite(output["install_logit"]).all())
                  and bool(torch.isfinite(output["cls"]).all()),
                  f"B={batch_size}: outputs finite")

            # C. Intermediate shapes (debug path)
            tokens = model.tokenize(batch, validate_inputs=False)
            d = model.config.d_model
            check(tokens["categorical"].shape == (batch_size, 30, d),
                  f"B={batch_size}: categorical tokens [B,30,{d}]")
            check(tokens["numerical"].shape == (batch_size, 38, d),
                  f"B={batch_size}: numerical tokens [B,38,{d}]")
            check(tokens["binary"].shape == (batch_size, 11, d),
                  f"B={batch_size}: binary tokens [B,11,{d}]")
            check(tokens["missing"].shape == (batch_size, 13, d),
                  f"B={batch_size}: missing tokens [B,13,{d}]")
            check(tokens["combined"].shape == (batch_size, 92, d),
                  f"B={batch_size}: combined [B,92,{d}]")
            encoded = model(batch, return_tokens=True)["encoded_sequence"]
            check(encoded.shape == (batch_size, 93, d),
                  f"B={batch_size}: encoder output [B,93,{d}]")

            # D. Loss (on a fresh graph-bearing forward)
            output_g = model(batch)
        loss = install_loss(output_g, batch)
        check(loss.dim() == 0 and bool(torch.isfinite(loss)),
              f"B={batch_size}: BCEWithLogitsLoss finite scalar ({loss.item():.4f})")

    # ------------------------------------------------------------------
    section("E. Backward / gradient sanity (real batch, B=1024)")
    model.zero_grad(set_to_none=True)
    loss = install_loss(model(batch), batch)
    t0 = time.perf_counter()
    loss.backward()
    backward_s = time.perf_counter() - t0
    grads = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    check(len(grads) == n_trainable, f"gradients exist for all {n_trainable} trainable parameters")
    check(all(torch.isfinite(g).all() for _, g in grads), "all gradients finite (no NaN/Inf)")
    n_zero = sum(1 for _, g in grads if g.abs().sum() == 0)
    print(f"backward time: {backward_s:.2f}s; parameters with exactly-zero grad: {n_zero}")

    # ------------------------------------------------------------------
    section("F. Tiny optimization sanity (3 steps, NOT training)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    ref_param = model.head[2].weight.detach().clone()
    it = iter(create_dataloader(train_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False))
    for step in range(3):
        b = next(it)
        optimizer.zero_grad()
        l = install_loss(model(b), b)
        losses.append(l.item())
        l.backward()
        optimizer.step()
    changed = not torch.equal(ref_param, model.head[2].weight.detach())
    check(all(torch.isfinite(torch.tensor(losses))), f"3-step losses finite: {[round(x, 4) for x in losses]}")
    check(changed, "head parameters changed after optimizer steps")
    print("note: sanity check only - not evidence of model quality")

    # ------------------------------------------------------------------
    section("14. Memory sanity (Stage 3 bounded DataLoader, no_grad inference)")
    rss_mark = rss_mib()
    it = iter(create_dataloader(train_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False))
    peak = rss_mark
    n = 0
    model.eval()
    with torch.no_grad():
        for b in it:
            _ = model(b)
            n += 1
            peak = max(peak, rss_mib())
            if n >= 25:
                break
    print(f"RSS before 25 forward batches: {rss_mark:.1f} MiB")
    print(f"RSS peak during:               {peak:.1f} MiB (delta {peak - rss_mark:+.1f} MiB)")
    check(peak < 2500, "peak RSS stays bounded (dataset never fully materialized)")

    # ------------------------------------------------------------------
    section("I. Deterministic construction (project seed 42)")
    # Both builds fresh: `model` above has been mutated by the optimizer.
    torch.manual_seed(SEED)
    m1 = TransformerBaseline(feature_set)
    torch.manual_seed(SEED)
    m2 = TransformerBaseline(feature_set)
    same = all(
        torch.equal(p1, p2)
        for (_, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters())
    )
    check(same, "two constructions under seed 42 produce identical parameters")

    # ------------------------------------------------------------------
    section("G/H. CPU + CUDA")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_d = model.to(device)
        b = next(iter(create_dataloader(train_ds, batch_size=256, shuffle=False,
                                        num_workers=0, pin_memory=True)))
        batch_d = {k: v.to(device) for k, v in b.items()}
        out = model_d(batch_d)
        check(bool(torch.isfinite(out["install_logit"]).all()), f"forward on {device}: finite install_logit")
        l = install_loss(out, batch_d)
        l.backward()
        grads_d = [p.grad for p in model_d.parameters() if p.grad is not None]
        check(bool(grads_d) and all(torch.isfinite(g).all() for g in grads_d),
              f"backward on {device}: finite gradients")
        model_d.zero_grad(set_to_none=True)
        model.cpu()
        print(f"device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA skipped: not available (explicit skip per Stage 4 spec)")

    # ------------------------------------------------------------------
    section("15. Official test split: schema compatibility only (no training)")
    test_ds = ProcessedRecSysDataset(feature_set, artifacts_dir / "processed" / "test", split="test")
    check(len(test_ds) == 160_973, f"test dataset opens with expected length (got {len(test_ds):,})")
    tb = collate_test([test_ds[i] for i in range(64)])
    model.eval()
    with torch.no_grad():
        out = model(tb)  # model back on CPU after the CUDA block
    check(out["install_logit"].shape == (64,) and bool(torch.isfinite(out["install_logit"]).all()),
          "model accepts label-free test-split batch (schema identical)")
    print("official test set used ONLY for schema compatibility; not fitted or tuned on")

    # ------------------------------------------------------------------
    section("8. Categorical ID contract on real data")
    b = next(iter(create_dataloader(train_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)))
    bad = b["categorical"].clone()
    bad[0, 5] = feature_set.vocabulary_sizes[5]  # f_6 vocab overflow
    try:
        model({**b, "categorical": bad})
        check(False, "out-of-range real-data ID raised CategoricalIdError")
    except CategoricalIdError as exc:
        check(True, f"out-of-range ID raises CategoricalIdError: {exc}")

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: ALL REAL-DATA TRANSFORMER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
