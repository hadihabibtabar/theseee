# -*- coding: utf-8 -*-
"""Generate thesis-ready convergence figures 4-2..4-5 from the FINAL Stage 12 artifacts.

READ-ONLY with respect to all experiment artifacts: the only data source is
reports/stage12/final_federated_matrix.json (cross-verified against the raw
Stage 12 / Stage 10B / Stage 11 result JSONs). No training, no metric invented,
no smoothing/interpolation. Writes only the 8 figure files in this directory.

Every validation is a hard assert: if any check fails, nothing is plotted.
"""
import json
import math
import os
import sys

import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MATRIX = os.path.join(ROOT, "reports", "stage12", "final_federated_matrix.json")

RAW_RESULT_JSONS = {
    ("nossl_a10",): "artifacts/federated/stage12/stage12_fed_nossl_alpha_1_0_seed42/stage12_fed_nossl_alpha_1_0_seed42_result.json",
    ("ssl_a10",): "artifacts/federated/stage12/stage12_fed_ssl_alpha_1_0_seed42/stage12_fed_ssl_alpha_1_0_seed42_result.json",
    ("nossl_a05",): "artifacts/federated/stage10/stage10b_result.json",
    ("ssl_a05",): "artifacts/federated/stage11/stage11_result.json",
    ("nossl_a01",): "artifacts/federated/stage12/stage12_fed_nossl_alpha_0_1_seed42/stage12_fed_nossl_alpha_0_1_seed42_result.json",
    ("ssl_a01",): "artifacts/federated/stage12/stage12_fed_ssl_alpha_0_1_seed42/stage12_fed_ssl_alpha_0_1_seed42_result.json",
}

def fa(text):
    """Shape+bidi Persian text for matplotlib."""
    return get_display(arabic_reshaper.reshape(text))

def load_cells():
    with open(MATRIX, encoding="utf-8") as f:
        d = json.load(f)
    cells = d["experiments"]
    # --- check 1: exactly six cells with the exact expected identities -------
    expected = {("nossl_a10", 1.0, False), ("ssl_a10", 1.0, True),
                ("nossl_a05", 0.5, False), ("ssl_a05", 0.5, True),
                ("nossl_a01", 0.1, False), ("ssl_a01", 0.1, True)}
    got = {(c["key"], float(c["alpha"]), bool(c["ssl"])) for c in cells}
    assert len(cells) == 6 and got == expected, f"cell identity mismatch: {got}"
    # effective identity must not come from the runner-default protocol quirk
    for c in cells:
        assert c.get("effective_use_ssl") == bool(c["ssl"]) and float(c["effective_alpha"]) == float(c["alpha"]), c["key"]
        assert c["status"] == "complete" and int(c["rounds_completed"]) == 10, c["key"]
    # --- check 2: 10 rounds, numbered 1..10, no gaps, all metrics present ----
    series = {}
    for c in cells:
        pr = sorted(c["per_round"], key=lambda r: r["round"])
        assert [r["round"] for r in pr] == list(range(1, 11)), f"{c['key']}: rounds != 1..10"
        for r in pr:
            for m in ("install_logloss", "install_auc", "click_logloss", "click_auc"):
                v = r[m]
                assert isinstance(v, (int, float)) and math.isfinite(v), f"{c['key']} r{r['round']} {m}"
        series[c["key"]] = {
            "alpha": float(c["alpha"]), "ssl": bool(c["ssl"]),
            "rounds": [r["round"] for r in pr],
            "install_logloss": [r["install_logloss"] for r in pr],
            "install_auc": [r["install_auc"] for r in pr],
            "click_logloss": [r["click_logloss"] for r in pr],
            "click_auc": [r["click_auc"] for r in pr],
        }
    # --- check 3: cross-verify ALL 60x4 values against raw result JSONs ------
    for (key,), rel in RAW_RESULT_JSONS.items():
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            raw = json.load(f)
        hist = sorted(raw["history"], key=lambda r: r["round"])
        assert len(hist) == 10
        for i, rr in enumerate(hist):
            vm = rr["validation_metrics"]
            for src, dst in (("val_install_logloss", "install_logloss"),
                             ("val_install_auc", "install_auc"),
                             ("val_click_logloss", "click_logloss"),
                             ("val_click_auc", "click_auc")):
                assert series[key][dst][i] == vm[src], f"{key} r{i+1} {dst} mismatch vs {rel}"
    # --- check 4: known reference values (spec section 12 sanity checks) -----
    ref = {
        ("nossl_a10", "install_logloss"): 0.262026822, ("nossl_a10", "install_auc"): 0.908208787,
        ("ssl_a10", "install_logloss"): 0.274393036, ("ssl_a10", "install_auc"): 0.897050083,
        ("nossl_a05", "install_logloss"): 0.26553604847797624, ("nossl_a05", "install_auc"): 0.908881783,
        ("nossl_a05", "click_logloss"): 0.271194215, ("nossl_a05", "click_auc"): 0.907656491,
        ("ssl_a05", "install_logloss"): 0.273903786, ("ssl_a05", "install_auc"): 0.899003267,
        ("nossl_a01", "install_logloss"): 0.287310072, ("nossl_a01", "install_auc"): 0.898739874,
        ("ssl_a01", "install_logloss"): 0.303964521, ("ssl_a01", "install_auc"): 0.889158964,
    }
    for (key, m), want in ref.items():
        best = min(series[key][m]) if m.endswith("logloss") else None
        if m.endswith("logloss"):
            # reference is the best-round (min) value; assert exact source match
            idx = min(range(10), key=lambda i: series[key][m][i])
            assert abs(series[key][m][idx] - want) < 5e-9, f"{key} {m} best {series[key][m][idx]} != {want}"
        else:
            idx = min(range(10), key=lambda i: series[key]["install_logloss"][i])
            assert abs(series[key][m][idx] - want) < 5e-9, f"{key} {m} at best round {series[key][m][idx]} != {want}"
    # SSL alpha=0.1 full trajectory sanity (rounded to 6 dp in the spec)
    traj = [(0.389630, 0.8132), (0.306231, 0.8670), (0.317884, 0.8758), (0.320017, 0.8784),
            (0.312523, 0.8829), (0.313369, 0.8833), (0.308537, 0.8858), (0.305642, 0.8874),
            (0.303965, 0.8892), (0.307488, 0.8893)]
    for i, (ll, auc) in enumerate(traj):
        assert abs(series["ssl_a01"]["install_logloss"][i] - ll) < 5e-7, f"ssl_a01 r{i+1} LL"
        assert abs(series["ssl_a01"]["install_auc"][i] - auc) < 5e-5, f"ssl_a01 r{i+1} AUC"
    return series

ORDER = ["nossl_a10", "ssl_a10", "nossl_a05", "ssl_a05", "nossl_a01", "ssl_a01"]
STYLE = {  # linestyle encodes alpha, marker encodes SSL (consistent across figures)
    "nossl_a10": dict(linestyle="-",  marker="o", label_fa="بدون SSL، α=1.0"),
    "ssl_a10":   dict(linestyle="-",  marker="s", label_fa="SSL، α=1.0"),
    "nossl_a05": dict(linestyle="--", marker="o", label_fa="بدون SSL، α=0.5"),
    "ssl_a05":   dict(linestyle="--", marker="s", label_fa="SSL، α=0.5"),
    "nossl_a01": dict(linestyle="-.", marker="o", label_fa="بدون SSL، α=0.1"),
    "ssl_a01":   dict(linestyle="-.", marker="s", label_fa="SSL، α=0.1"),
}
FIGS = [
    ("fig_4_2_install_logloss_convergence", "install_logloss",
     "Install LogLoss در ۱۰ دور یادگیری فدرال", "LogLoss نصب"),
    ("fig_4_3_install_auc_convergence", "install_auc",
     "Install AUC در ۱۰ دور یادگیری فدرال", "AUC نصب"),
    ("fig_4_4_click_logloss_convergence", "click_logloss",
     "Click LogLoss در ۱۰ دور یادگیری فدرال", "LogLoss کلیک"),
    ("fig_4_5_click_auc_convergence", "click_auc",
     "Click AUC در ۱۰ دور یادگیری فدرال", "AUC کلیک"),
]

def main():
    series = load_cells()
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = [f for f in ("Tahoma", "Noto Sans", "Noto Sans CJK JP", "DejaVu Sans") if f in names]
    assert family, "no Persian-capable font found"
    plt.rcParams.update({
        "font.family": family,
        "axes.unicode_minus": False,
        "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
        "legend.fontsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "figure.dpi": 100, "savefig.dpi": 300,
    })
    x = list(range(1, 11))
    for stem, metric, title_fa, ylabel_fa in FIGS:
        fig, ax = plt.subplots(figsize=(10.0, 6.2))
        for key in ORDER:
            st = STYLE[key]
            ax.plot(x, series[key][metric], linewidth=2.0, markersize=6.5,
                    marker=st["marker"], linestyle=st["linestyle"], label=fa(st["label_fa"]))
        ax.set_xticks(x)
        ax.set_xlim(0.5, 10.5)
        ax.set_xlabel(fa("دور یادگیری فدرال"))
        ax.set_ylabel(fa(ylabel_fa))
        ax.set_title(fa(title_fa))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)
        fig.tight_layout()
        png = os.path.join(HERE, stem + ".png")
        pdf = os.path.join(HERE, stem + ".pdf")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        # --- per-figure sanity: exactly 60 points per metric across 6 curves --
        n = sum(len(series[k][metric]) for k in ORDER)
        assert n == 60, f"{metric}: {n} != 60"
        print(f"OK {stem}: 60 points, y-range [{min(min(series[k][metric]) for k in ORDER):.6f}, "
              f"{max(max(series[k][metric]) for k in ORDER):.6f}]")
    print("ALL FIGURES GENERATED; all validation asserts passed.")

if __name__ == "__main__":
    main()
