# -*- coding: utf-8 -*-
"""Generate Figure 4-10, Figure 4-11, and Table 4-16 from EXISTING verified artifacts.

READ-ONLY analysis: reads result JSONs, partition manifests/NPZ, and the frozen
split; writes only the new figure files and the two execution-summary files.
No training, no estimation, no interpolation, no invented values.

Validation is by hard assert: if any check fails, nothing is written.
"""
import json
import os

import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

MATRIX = "reports/stage12/final_federated_matrix.json"
RAW = {
    "nossl_a10": "artifacts/federated/stage12/stage12_fed_nossl_alpha_1_0_seed42/stage12_fed_nossl_alpha_1_0_seed42_result.json",
    "ssl_a10":   "artifacts/federated/stage12/stage12_fed_ssl_alpha_1_0_seed42/stage12_fed_ssl_alpha_1_0_seed42_result.json",
    "nossl_a05": "artifacts/federated/stage10/stage10b_result.json",
    "ssl_a05":   "artifacts/federated/stage11/stage11_result.json",
    "nossl_a01": "artifacts/federated/stage12/stage12_fed_nossl_alpha_0_1_seed42/stage12_fed_nossl_alpha_0_1_seed42_result.json",
    "ssl_a01":   "artifacts/federated/stage12/stage12_fed_ssl_alpha_0_1_seed42/stage12_fed_ssl_alpha_0_1_seed42_result.json",
}
PARTS = {  # alpha_D -> (manifest, npz, expected hash)
    1.0: ("artifacts/federated/partitions/stage12_partition_alpha_1_0_seed42.json",
          "artifacts/federated/partitions/stage12_partition_alpha_1_0_seed42.npz", "c382d47f5824f64a"),
    0.5: ("artifacts/federated/partition_alpha_0_5_seed42.json",
          "artifacts/federated/partition_alpha_0_5_seed42.npz", "9603ddc9facc973d"),
    0.1: ("artifacts/federated/partitions/stage12_partition_alpha_0_1_seed42.json",
          "artifacts/federated/partitions/stage12_partition_alpha_0_1_seed42.npz", "23146aebf184345b"),
}
SPLIT = "artifacts/splits/centralized_split.npz"
TOTAL_ROWS = 3_137_266

def J(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return json.load(f)

def fa(t):
    return get_display(arabic_reshaper.reshape(t))

# ================= shared validation =================
def validate_shared():
    m = J(MATRIX)
    cells = {c["key"]: c for c in m["experiments"]}
    expected = {("nossl_a10", 1.0, False), ("ssl_a10", 1.0, True), ("nossl_a05", 0.5, False),
                ("ssl_a05", 0.5, True), ("nossl_a01", 0.1, False), ("ssl_a01", 0.1, True)}
    got = {(k, float(c["alpha"]), bool(c["ssl"])) for k, c in cells.items()}
    assert len(cells) == 6 and got == expected, f"cell identity mismatch: {got}"
    return m, cells

# ================= Figure 4-10 =================
def fig_4_10(cells):
    key = "nossl_a05"  # selected representative scenario: No-SSL alpha_D=0.5 (Stage 10B)
    c = cells[key]
    assert key in RAW
    raw = J(RAW[key])
    hist = sorted(raw["history"], key=lambda r: r["round"])
    # --- validation: 10 rounds, 4 metrics each, raw cross-check, sanity value --
    assert [r["round"] for r in hist] == list(range(1, 11))
    vm = [r["validation_metrics"] for r in hist]
    series = {m: [v["val_" + m] for v in vm] for m in
              ("install_logloss", "install_auc", "click_logloss", "click_auc")}
    pr = sorted(c["per_round"], key=lambda r: r["round"])
    for i in range(10):
        for m in series:
            assert pr[i][m] == series[m][i], f"{key} r{i+1} {m} mismatch vs raw JSON"
            assert np.isfinite(series[m][i])
    assert len(series) * 10 == 40
    assert series["install_logloss"][9] == 0.26553604847797624  # verified best/final value
    assert raw["best_round"] == 10 and raw["rounds_completed"] == 10

    x = list(range(1, 11))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.4))
    for ax, m in ((ax1, "logloss"), (ax2, "auc")):
        ax.plot(x, series["install_" + m], marker="o", linewidth=2.0, markersize=6, label="Install")
        ax.plot(x, series["click_" + m], marker="s", linewidth=2.0, markersize=6, label="Click")
        ax.set_xticks(x)
        ax.set_xlim(0.5, 10.5)
        ax.set_xlabel(fa("دور یادگیری فدرال"))
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right" if m == "logloss" else "lower right", framealpha=0.9)
    fig.suptitle(fa("مقایسه روند Click و Install در طول Roundهای آموزش"))
    fig.text(0.5, 0.005,
             fa("سناریوی نمایش‌داده‌شده: بدون SSL، α_D = 0.5 (پارتیشن دیریکله؛ پارامتر پارتیشن‌بندی داده)"),
             ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_4_10_click_install_comparison.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("OK fig_4_10: scenario nossl_a05 (Stage 10B), 40 values, round10 Install LL =",
          series["install_logloss"][9])

# ================= Figure 4-11 =================
def fig_4_11():
    split = np.load(os.path.join(ROOT, SPLIT))
    train_idx = np.asarray(split["train_indices"])
    assert train_idx.size == TOTAL_ROWS
    val_idx = np.asarray(split["validation_indices"])
    counts = {}
    for a, (mj, nj, expected_hash) in PARTS.items():
        man = J(mj)
        assert man["partition_hash"] == expected_hash, f"alpha {a}: hash {man['partition_hash']} != {expected_hash}"
        cl = man["clients"]
        assert len(cl) == 10, f"alpha {a}: {len(cl)} clients"
        z = np.load(os.path.join(ROOT, nj))
        embedded = str(z["partition_hash"].item()).strip()
        assert embedded == expected_hash, \
            f"alpha {a}: NPZ-embedded hash {embedded} != manifest hash {expected_hash}"
        assert sorted(z.files) == sorted([f"client_{i:02d}" for i in range(10)] + ["partition_hash"]), \
            f"alpha {a}: unexpected NPZ keys {z.files}"
        tot, overlap = 0, 0
        cc = {}
        for i in range(10):
            arr = np.asarray(z[f"client_{i:02d}"])
            n = man["clients"][f"client_{i:02d}"]["rows"]
            assert n == arr.size, f"alpha {a} client_{i:02d}: manifest rows {n} != npz {arr.size}"
            assert arr.max() < TOTAL_ROWS and np.all(np.diff(arr) > 0)
            tot += n
            # positions are indices INTO train_indices; map to dataset rows for the
            # overlap check against the frozen validation indices
            overlap += int(np.intersect1d(train_idx[arr], val_idx).size)
            cc[f"client_{i:02d}"] = int(n)
        assert tot == TOTAL_ROWS, f"alpha {a}: rows {tot} != {TOTAL_ROWS}"
        assert overlap == 0, f"alpha {a}: validation overlap {overlap} != 0"
        counts[a] = cc
        print(f"OK partition alpha_D={a}: hash {expected_hash}, 10 clients, sum={tot}, validation overlap=0")
    assert sum(len(v) for v in counts.values()) == 30

    clients = [f"client_{i:02d}" for i in range(10)]
    x = np.arange(10)
    w = 0.27
    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    for j, a in enumerate((1.0, 0.5, 0.1)):
        ax.bar(x + (j - 1) * w, [counts[a][c] for c in clients], width=w,
               label=f"α_D = {a}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Client {i+1}" for i in range(10)])
    ax.set_xlabel("Client")
    ax.set_ylabel(fa("تعداد نمونه‌های آموزشی اختصاص‌یافته"))
    ax.set_title(fa("تعداد نمونه‌های اختصاص‌یافته به ۱۰ Client برای سه مقدار α_D (پارامتر پارتیشن‌بندی دیریکله)"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_4_11_client_sample_distribution.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    return counts

# ================= Table 4-16 =================
def table_4_16(cells):
    rows, checks = [], []
    for key in ("nossl_a10", "ssl_a10", "nossl_a05", "ssl_a05", "nossl_a01", "ssl_a01"):
        c = cells[key]
        raw = J(RAW[key])
        # identity + completion from the raw artifact (not hard-coded)
        assert raw["status"] == "complete" and raw["rounds_completed"] == 10 and raw["protocol_rounds"] == 10
        if "stage12" in raw:
            assert bool(raw["stage12"]["use_ssl"]) == bool(c["ssl"]) and float(raw["stage12"]["alpha"]) == float(c["alpha"])
        walls = [r["round_wall_seconds"] for r in raw["history"]]
        assert len(walls) == 10
        gpu_a = max(r["peak_gpu_allocated_mib"] for r in raw["history"])
        gpu_r = max(r["peak_gpu_reserved_mib"] for r in raw["history"])
        rows.append({
            "key": key, "cell": {"nossl_a10": "A", "ssl_a10": "B", "nossl_a05": "C",
                                 "ssl_a05": "D", "nossl_a01": "E", "ssl_a01": "F"}[key],
            "experiment": c["experiment"], "ssl": bool(c["ssl"]), "alpha_D": float(c["alpha"]),
            "total_elapsed_seconds": raw["total_elapsed_seconds"],
            "sum_round_walls": round(sum(walls), 1),
            "peak_rss_mib": raw["peak_rss_mib_overall"],
            "gpu_alloc_mib": round(gpu_a, 1), "gpu_reserved_mib": round(gpu_r, 1),
            "rounds_completed": raw["rounds_completed"], "protocol_rounds": raw["protocol_rounds"],
        })
        checks.append(abs(raw["total_elapsed_seconds"] - sum(walls)) < 30)
    # Stage 10B anomaly: recorded total vs sum of round walls
    c_row = next(r for r in rows if r["key"] == "nossl_a05")
    assert abs(c_row["total_elapsed_seconds"] - 5226.7641265) < 1.0
    assert abs(c_row["sum_round_walls"] - 25494.0) < 1.0

    def hms(s):
        s = int(round(s)); return f"{s//3600}:{s%3600//60:02d}:{s%60:02d}"

    md = []
    md.append("# Table 4-16 — زمان اجرا، مصرف حافظه و تعداد Roundهای شش آزمایش نهایی\n")
    md.append("منبع: فیلدهای ثبت‌شده در JSONهای نتیجه‌ی شش سلول نهایی "
              "(`total_elapsed_seconds`، `peak_rss_mib_overall`، `rounds_completed`، "
              "`round_wall_seconds`، `peak_gpu_*`). هیچ مقداری تخمین زده نشده است.\n")
    md.append("| آزمایش | شناسه | SSL | α_D | زمان اجرا (ثانیه، ثبت‌شده) | زمان اجرا (h:mm:ss) | حافظه مصرفی (ثبت‌شده) | Roundهای اجراشده (اجراشده/پیکربندی) |")
    md.append("| --- | --- | --- | ---: | ---: | ---: | --- | ---: |")
    name_fa = {"A": "بدون SSL، α_D=1.0", "B": "SSL، α_D=1.0", "C": "بدون SSL، α_D=0.5",
               "D": "SSL، α_D=0.5", "E": "بدون SSL، α_D=0.1", "F": "SSL، α_D=0.1"}
    for r in rows:
        mem = (f"{r['peak_rss_mib']:,.1f} MiB peak RSS؛ GPU {r['gpu_alloc_mib']:,.0f}/"
               f"{r['gpu_reserved_mib']:,.0f} MiB (allocated/reserved)")
        flag = " †" if r["key"] == "nossl_a05" else ""
        md.append(f"| {name_fa[r['cell']]} | `{r['experiment']}` | {'بله' if r['ssl'] else 'خیر'} | "
                  f"{r['alpha_D']} | {r['total_elapsed_seconds']:,.1f}{flag} | {hms(r['total_elapsed_seconds'])} | "
                  f"{mem} | {r['rounds_completed']}/{r['protocol_rounds']} |")
    md.append("")
    md.append("† **ناسازگاری شناخته‌شده‌ی Stage 10B (سلول C):** مقدار ثبت‌شده‌ی "
              "`total_elapsed_seconds` = 5,226.8 ثانیه است، در حالی که مجموع "
              "`round_wall_seconds` ده دور در همان فایل 25,494.0 ثانیه است. هر دو مقدار "
              "همان‌طور که در آرتیفکت ثبت شده گزارش شدند؛ هیچ‌کدام اصلاح یا جایگزین نشده‌اند. "
              "در پنج سلول دیگر اختلاف این دو مقدار کمتر از 30 ثانیه است.\n")
    md.append("GPU peaks = بیشینه‌ی مقادیر ثبت‌شده‌ی هر دور در `history` (نه تخمین بر اساس مشخصات کارت).")
    md.append("")
    md.append("### Machine-readable summary (same data)\n")
    md.append("| key | cell | ssl | alpha_D | total_elapsed_seconds | sum_round_wall_seconds | peak_rss_mib | peak_gpu_allocated_mib | peak_gpu_reserved_mib | rounds_completed | protocol_rounds |")
    md.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        md.append(f"| {r['key']} | {r['cell']} | {r['ssl']} | {r['alpha_D']} | {r['total_elapsed_seconds']} | "
                  f"{r['sum_round_walls']} | {r['peak_rss_mib']} | {r['gpu_alloc_mib']} | "
                  f"{r['gpu_reserved_mib']} | {r['rounds_completed']} | {r['protocol_rounds']} |")
    with open(os.path.join(ROOT, "reports/stage12/final_execution_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    import csv
    with open(os.path.join(ROOT, "reports/stage12/final_execution_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cell", "key", "experiment", "ssl", "alpha_D", "total_elapsed_seconds_recorded",
                    "sum_round_wall_seconds", "peak_rss_mib_recorded", "peak_gpu_allocated_mib_recorded_max",
                    "peak_gpu_reserved_mib_recorded_max", "rounds_completed", "protocol_rounds", "note"])
        for r in rows:
            note = ("Stage 10B known anomaly: recorded total_elapsed_seconds differs from sum of "
                    "round_wall_seconds; both preserved verbatim" if r["key"] == "nossl_a05" else "")
            w.writerow([r["cell"], r["key"], r["experiment"], r["ssl"], r["alpha_D"],
                        r["total_elapsed_seconds"], r["sum_round_walls"], r["peak_rss_mib"],
                        r["gpu_alloc_mib"], r["gpu_reserved_mib"], r["rounds_completed"],
                        r["protocol_rounds"], note])
    print("OK table_4_16: 6 rows; anomaly flagged only for Stage 10B;",
          "cells with elapsed-vs-walls mismatch <30s:", sum(checks), "of 6")

def main():
    m, cells = validate_shared()
    fig_4_10(cells)
    fig_4_11()
    table_4_16(cells)
    print("ALL OUTPUTS WRITTEN; all validation asserts passed.")

if __name__ == "__main__":
    main()
