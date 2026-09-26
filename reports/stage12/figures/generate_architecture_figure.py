"""Draw the full proposed-method architecture figure (READ-ONLY stage).

End-to-end pipeline: ShareChat data -> preprocessing -> federated loop with
Transformer+MMoE(+SSL) clients -> Click/Install outputs. Every architectural
fact (92 feature tokens + CLS, d_model=128, 6-layer/8-head encoder, 8 shared
MMoE experts, projection head 128->128->GELU->64, NT-Xent tau=0.2, joint loss
0.6/0.4, corruption 0.15, sample-weighted FedAvg over 10 clients) is taken
verbatim from the frozen sources:

  src/recsys23_fedrec/models/transformer.py          (Stage 4)
  src/recsys23_fedrec/models/mmoe.py                 (Stage 5)
  src/recsys23_fedrec/models/ssl_transformer_mmoe.py (Stage 7)
  src/recsys23_fedrec/ssl/augmentations.py           (Stage 7)
  src/recsys23_fedrec/ssl/contrastive_loss.py        (Stage 7)
  src/recsys23_fedrec/ssl/joint_loss.py              (Stage 7)
  src/recsys23_fedrec/federated/partition.py         (Stage 9)
  src/recsys23_fedrec/federated/fedavg.py            (Stage 10)

No training, no artifact modification; outputs go only to
reports/stage12/figures/. Neutral rendering only: black/white/grayscale,
matplotlib defaults otherwise (no style sheet).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    def T(s: str) -> str:
        """Shape + bidi-reorder Persian text for matplotlib."""
        return get_display(arabic_reshaper.reshape(s))
except ImportError:  # graceful degradation: raw strings
    def T(s: str) -> str:
        return s

# ------------------------------------------------------------------ fonts
# Noto Sans is unavailable on this machine; Tahoma has full Persian coverage.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "stage12" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ canvas
fig, ax = plt.subplots(figsize=(16.0, 11.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_axis_off()

FS_TITLE = 17
FS = 11
FS_SMALL = 8.8


def box(x, y, w, h, title, note=None, fs=FS, nfs=FS_SMALL, lw=1.1):
    """Rounded box with an optional second (note) line."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.3",
        linewidth=lw, facecolor="white", edgecolor="black", zorder=2,
    ))
    if note is None:
        ax.text(x + w / 2, y + h / 2, T(title), fontsize=fs, ha="center",
                va="center", fontweight="bold", zorder=3)
    else:
        ax.text(x + w / 2, y + h * 0.68, T(title), fontsize=fs, ha="center",
                va="center", fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.26, T(note), fontsize=nfs, ha="center",
                va="center", color="0.35", zorder=3)


def arrow(x1, y1, x2, y2, lw=1.4, ms=14, connectionstyle=None, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
        linewidth=lw, color="black", zorder=3, connectionstyle=connectionstyle,
    ))


def note(x, y, s, fs=FS_SMALL, ha="center", color="0.35"):
    ax.text(x, y, T(s), fontsize=fs, ha=ha, va="center", color=color, zorder=3)


# ==================================================================
# Title
# ==================================================================
ax.text(50, 96.5,
        T("معماری کامل روش پیشنهادی: از داده‌های ShareChat تا خروجی‌های Click و Install — با مسیر SSL و حلقه‌ی یادگیری فدرال"),
        fontsize=FS_TITLE, ha="center", va="center", fontweight="bold")

# ==================================================================
# Band 1 — data pipeline (left to right), y 76..84
# ==================================================================
box(2, 76, 17, 8, "داده‌های ShareChat",
    "رکوردهای نمایش با برچسب‌های Click و Install")
box(21.5, 76, 17, 8, "پیش‌پردازش فریز‌شده (Stage 2)",
    "کدگذاری دسته‌ای و نرمال‌سازی عددی — بدون استفاده از مجموعه‌ی آزمون رسمی")
box(41, 76, 17, 8, "92 ویژگی مدل",
    "30 دسته‌ای + 38 عددی + 11 باینری + 13 شاخص مقدار گمشده")
box(60.5, 76, 17, 8, "تقسیم Dirichlet با α = 1.0 / 0.5 / 0.1",
    "10 کلاینت شبیه‌سازی‌شده، بذر 42، 3,137,266 ردیف آموزش")
box(80, 76, 17, 8, "کلاینت k از 10 کلاینت",
    "داده‌ی محلی با توزیع غیرهمگن Non-IID")

for x1 in (19, 38.5, 58, 77.5):
    arrow(x1, 80, x1 + 2.5, 80)

# clients feed the model container below
arrow(88.5, 76, 88.5, 66.8)

# ==================================================================
# Band 2 — client model container, y 24..66
# ==================================================================
ax.add_patch(FancyBboxPatch(
    (28, 24), 70.5, 42, boxstyle="round,pad=0.3",
    linewidth=1.6, facecolor="white", edgecolor="black", zorder=1,
))
ax.text(63, 63.2,
        T("مدل کلاینت: Transformer + MMoE با مسیر SSL  (SSLTransformerMMoE)"),
        fontsize=FS, ha="center", va="center", fontweight="bold", zorder=3)
note(63, 60.2,
     "مسیر SSL فقط در سلول‌های SSL فعال است؛ در سلول‌های No-SSL تنها خطای نظارت‌شده استفاده می‌شود.")

# main supervised row, y 33..40
box(30.5, 33, 14, 7, "توکن‌سازی ویژگی‌به‌ویژگی", "92 توکن + CLS = دنباله‌ی 93 تایی")
box(47, 33, 14, 7, "بک‌بون Transformer مشترک", "d=128، 6 لایه، 8 هد توجه")
box(63.5, 33, 13, 7, "نمایش CLS", "بردار 128 بُعدی")
box(79, 33, 17, 7, "MMoE چندوظیفه‌ای", "8 متخصص مشترک + دروازه و برج برای هر وظیفه")
arrow(44.5, 36.5, 47, 36.5)
arrow(61, 36.5, 63.5, 36.5)
arrow(76.5, 36.5, 79, 36.5)

# SSL row, y 47..54
box(47, 47, 14, 7, "افزودن نویز به ویژگی‌ها (نرخ 0.15)", "دو نمای مستقل از هر نمونه")
box(63.5, 47, 13, 7, "هد پروجکشن کنتراستیو", "128 → 128 → GELU → 64 و نرمال‌سازی L2")
box(79, 47, 17, 7, "NT-Xent (دمای τ = 0.2)", "جفت مثبت: دو نمای یک نمونه")
arrow(61, 50.5, 63.5, 50.5)
arrow(76.5, 50.5, 79, 50.5)
# views pass through the shared backbone
arrow(54, 47, 54, 40)
# CLS of each view feeds the projection head
arrow(70, 40, 70, 47)

# outputs row, y 24.5..30.5
box(47, 24.5, 13, 6, "خروجی Click", "logit")
box(63.5, 24.5, 13, 6, "خروجی Install", "logit")
arrow(83, 33, 55, 30.8)
arrow(87, 33, 70, 30.8)

# joint loss box (bottom right of the container)
box(79, 24.5, 17, 6, "خطای تلفیقی آموزش", "0.6 × L_sup + 0.4 × L_NT-Xent")
# NT-Xent -> joint loss, routed as an elbow AROUND the MMoE box (right side)
arrow(96.4, 50.5, 97.7, 50.5, style="-")
arrow(97.7, 50.5, 97.7, 27.5, style="-")
arrow(97.7, 27.5, 96.5, 27.5)

# ==================================================================
# Band 3 — federated loop, y 4..18
# ==================================================================
box(30.5, 6, 30, 8, "سرور: FedAvg وزن‌دار با نمونه", "w ← Σ (n_k / n) · w_k روی پارامترهای مدل")
box(66, 6, 30, 8, "پخش مدل سراسری به کلاینت‌ها", "شروع سرد با بذر 42؛ تکرار در 10 راند")

arrow(40, 24, 40, 14.4)   # upload
note(42.5, 19.5, "آپلود وزن‌های محلی", ha="left")
arrow(60.5, 10, 66, 10)   # server -> broadcast
arrow(81, 14.4, 81, 24)   # broadcast up to clients
note(83.5, 19.5, "پخش w سراسری", ha="left")

# left footnote: fixed federated protocol (bottom strip, clear of all boxes)
ax.text(2, 2.6,
        T("پروتکل ثابت حلقه‌ی فدرال: شرکت هر 10 کلاینت در هر راند؛ 1 epoch محلی؛ batch = 1024؛ AdamW با lr = 3e-4 و weight decay = 0.01؛\n"
          "تجمیع FedAvg وزن‌دار با نمونه؛ انتخاب بهترین چک‌پوینت بر اساس Install LogLoss مجموعه‌ی اعتبارسنجی."),
        fontsize=8.8, ha="left", va="center", color="0.15")

# ==================================================================
# Save
# ==================================================================
png = OUT / "architecture_overview.png"
pdf = OUT / "architecture_overview.pdf"
fig.savefig(png, dpi=200, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)
print(f"wrote {png}")
print(f"wrote {pdf}")
