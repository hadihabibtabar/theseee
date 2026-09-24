"""Read-only plotting for the final federated experimental matrix.

Source of truth : reports/stage12/final_federated_matrix.json
Outputs         : reports/stage12/figures/fig_4_1..fig_4_6 (.png + .pdf)

Safety: this script only READS the finalized result JSON. It trains nothing,
touches no artifact under artifacts/, and modifies no existing report.
All plotted numbers are taken verbatim from the JSON (no hard-coded metrics).

Style constraints (per protocol): matplotlib defaults only - no seaborn, no
custom colors (default color cycle), no custom stylesheet, no subplots, no
axis reversal, no log axes, no smoothing, no error bars (single seed).
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; no display, no interaction
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "reports" / "stage12" / "final_federated_matrix.json"
OUT = ROOT / "reports" / "stage12" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

with SRC.open("r", encoding="utf-8") as f:
    data = json.load(f)

# ---------------------------------------------------------------- load cells
cells = {}
for exp in data["experiments"]:
    key = (float(exp["effective_alpha"]), bool(exp["effective_use_ssl"]))
    cells[key] = exp

expected_keys = {
    (1.0, False): "stage12_fed_nossl_alpha_1_0_seed42",
    (1.0, True): "stage12_fed_ssl_alpha_1_0_seed42",
    (0.5, False): "stage12_fed_nossl_alpha_0_5_seed42",   # Stage 10B
    (0.5, True): "stage12_fed_ssl_alpha_0_5_seed42",      # Stage 11
    (0.1, False): "stage12_fed_nossl_alpha_0_1_seed42",
    (0.1, True): "stage12_fed_ssl_alpha_0_1_seed42",
}
assert set(cells) == set(expected_keys), f"cell mismatch: {sorted(cells)}"
for k, name in expected_keys.items():
    assert cells[k]["experiment"] == name, (k, cells[k]["experiment"])
    assert cells[k]["status"] == "complete" and cells[k]["complete"], k

# X order: canonical Dirichlet ordering 1.0, 0.5, 0.1 (heterogeneity increases
# as alpha decreases); axis is explicitly labeled "Dirichlet alpha".
ALPHAS = [1.0, 0.5, 0.1]
ROUND_LIST = list(range(1, 11))


def traj(alpha, ssl, field):
    """Exact per-round values for one cell; asserts 10 consecutive rounds."""
    exp = cells[(alpha, ssl)]
    per = {int(r["round"]): float(r[field]) for r in exp["per_round"]}
    assert sorted(per) == ROUND_LIST, (alpha, ssl, sorted(per))
    vals = [per[r] for r in ROUND_LIST]
    assert all(math.isfinite(v) for v in vals), (alpha, ssl, field)
    return vals


def best_round(alpha, ssl):
    return int(cells[(alpha, ssl)]["best"]["best_round"])


def best_v(alpha, ssl):
    """Best-round metrics from the JSON, as (install_ll, install_auc, click_ll, click_auc)."""
    b = cells[(alpha, ssl)]["best"]
    return (float(b["install_logloss"]), float(b["install_auc"]),
            float(b["click_logloss"]), float(b["click_auc"]))


def save(fig, stem):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png.name} and {pdf.name}")


def alpha_axis(ax):
    ax.set_xticks(ALPHAS)
    ax.set_xticklabels([f"{a:g}" for a in ALPHAS])
    ax.set_xlabel("Dirichlet alpha")
    ax.grid(True)
    ax.set_axisbelow(True)


def label_series(alpha, ssl):
    stage = cells[(alpha, ssl)]["stage"]
    return f"{'SSL' if ssl else 'No-SSL'}, alpha={alpha:g} ({stage})"


# ------------------------------------------------- Fig 1: Install LL vs alpha
fig, ax = plt.subplots(figsize=(7.0, 5.0))
for ssl in (False, True):
    ax.plot(ALPHAS, [best_v(a, ssl)[0] for a in ALPHAS],
            marker="o", label="SSL" if ssl else "No-SSL")
ax.set_xlabel("Dirichlet alpha")
ax.set_ylabel("Validation Install LogLoss")
ax.set_title("Final validation Install LogLoss vs Dirichlet alpha\n"
             "(best checkpoint per experiment, validation only, seed 42)")
alpha_axis(ax)
ax.legend()
save(fig, "fig_4_1_install_logloss_vs_alpha")

# ------------------------------------------------- Fig 2: Install AUC vs alpha
fig, ax = plt.subplots(figsize=(7.0, 5.0))
for ssl in (False, True):
    ax.plot(ALPHAS, [best_v(a, ssl)[1] for a in ALPHAS],
            marker="o", label="SSL" if ssl else "No-SSL")
ax.set_xlabel("Dirichlet alpha")
ax.set_ylabel("Validation Install AUC")
ax.set_title("Final validation Install AUC vs Dirichlet alpha\n"
             "(best checkpoint per experiment, validation only, seed 42)")
alpha_axis(ax)
ax.legend()
save(fig, "fig_4_2_install_auc_vs_alpha")

# --------------------------------------------- Figs 3/4: convergence (10 rounds)
def convergence_fig(ssl, stem):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for alpha in ALPHAS:
        ys = traj(alpha, ssl, "install_logloss")
        ln, = ax.plot(ROUND_LIST, ys, marker="o", label=label_series(alpha, ssl))
        br = best_round(alpha, ssl)
        # Mark the selected round (lowest validation Install LogLoss) only;
        # no error bars / CI: single seed, no smoothing, no interpolation.
        ax.scatter([br], [ys[br - 1]], s=140, facecolors="none",
                   edgecolors=ln.get_color(), linewidths=1.8, zorder=5,
                   label="selected round (min. validation Install LogLoss)"
                   if alpha == ALPHAS[0] else None)
    ax.set_xticks(ROUND_LIST)
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Validation Install LogLoss")
    ax.set_title(("No-SSL" if not ssl else "SSL") +
                 " convergence: validation Install LogLoss per round\n"
                 "(circled = selected best checkpoint, validation only, seed 42)")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend()
    save(fig, stem)


convergence_fig(False, "fig_4_3_nossl_convergence")
convergence_fig(True, "fig_4_4_ssl_convergence")

# --------------------------------------- Fig 5: matched pairs (grouped bars)
import numpy as np  # only for bar x-offsets, not for data manipulation

fig, ax = plt.subplots(figsize=(7.0, 5.0))
x = np.arange(len(ALPHAS))
width = 0.35
for off, ssl in ((-width / 2, False), (width / 2, True)):
    vals = [best_v(a, ssl)[0] for a in ALPHAS]
    bars = ax.bar(x + off, vals, width, label="SSL" if ssl else "No-SSL")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.6f}",
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"{a:g}" for a in ALPHAS])
ax.set_xlabel("Dirichlet alpha")
ax.set_ylabel("Validation Install LogLoss")
ax.set_ylim(0, 0.35)  # bars start at zero so bar lengths stay proportional
ax.set_title("Matched pairs: validation Install LogLoss, No-SSL vs SSL\n"
             "(best checkpoint per experiment, validation only, seed 42)")
ax.grid(True, axis="y")
ax.set_axisbelow(True)
ax.legend()
save(fig, "fig_4_5_ssl_vs_nossl_install_logloss")

# ------------------------------------------------ Fig 6: Click LogLoss vs alpha
fig, ax = plt.subplots(figsize=(7.0, 5.0))
for ssl in (False, True):
    vals = [best_v(a, ssl)[3] for a in ALPHAS]
    ax.plot(ALPHAS, vals, marker="o", label="SSL" if ssl else "No-SSL")
ax.set_xlabel("Dirichlet alpha")
ax.set_ylabel("Validation Click LogLoss")
ax.set_title("Final validation Click LogLoss (auxiliary task) vs Dirichlet alpha\n"
             "(best checkpoint per experiment, validation only, seed 42)")
alpha_axis(ax)
ax.legend()
save(fig, "fig_4_6_click_logloss_vs_alpha")

print("All six figures generated from", SRC)
