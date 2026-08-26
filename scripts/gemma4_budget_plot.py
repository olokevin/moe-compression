#!/usr/bin/env python3
"""Gemma4-26B-A4B dynamic-channel-selection curves, in the presentation style.

Emits ``gemma4_mmlu.{pdf,png}`` and ``gemma4_arc_c.{pdf,png}``, restyled to match
``fig_probe_curve_*.pdf`` (same rcParams, palette, figure size, spine/grid treatment,
legend placement) and sharing the probe figures' x-axis:

    x = **FFN active parameter reduction**  (%, increasing left to right)

exactly as in fig_probe_curve_*.pdf. As there, the axis is a deliberate *mixed*
convention (documented on the probe figure too):

  * solid  (rank-by-up)   -> plotted at the REALIZED active-param reduction
    ``100*(1 - active_param_ratio)`` -- gate+down shrink, up_proj kept full.
  * dashed (rank-by-inter) -> plotted at the NOMINAL channel-cut, because gate+up
    stay full width and only down_proj shrinks, so its used-parameter cut is small;
    plotting at nominal keeps the curve's shape visible (same rationale as the
    oracle_mag ceiling on the probe figure).

Colour encodes the case, linestyle encodes the scoring signal (a 2x2):
  * Case 1 (routed-only) = blue      Case 2 (joint) = amber
  * rank-by-up = solid, filled sq.   rank-by-intermediate = dashed, hollow circle

Data source: docs/report/week9.md section 2 "Gemma4-26B-A4B", Case 1 / Case 2 tables.
The two Case-2 rank-by-up mid points (60%, 75% nominal) are not in the week9.md
table (only 90% is tabulated) -- they were recovered from the prior gemma4_*.png by
calibrated marker extraction (the calibration reproduced every documented point to
+/-0.005), so this reproduces the original figures' data in the new style.

Usage::

    python scripts/gemma4_budget_plot.py
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- palette + rcParams: identical to probe_curve_plot.py ------------------- #
INK = "#1c2330"
MUTED = "#68717f"
BLUE = "#2f6fdb"
AMBER = "#e08a1e"
GREEN = "#2e8b6f"
RED = "#c8402f"
GREY = "#9aa4b2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": "#cfd6e0", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 10.5, "axes.labelsize": 13.0,
    "xtick.labelsize": 12.0, "ytick.labelsize": 12.0,
    "legend.fontsize": 11.0, "figure.titlesize": 11.5,
    "savefig.bbox": "tight", "axes.grid": True,
    "grid.color": "#eef1f6", "grid.linewidth": 0.8,
})

# --------------------------------------------------------------------------- #
# Data (accuracy as a fraction; plotted x100). x-values are already converted to
# "FFN active parameter reduction" %: solid = 100*(1-active_param_ratio) from the
# active-param-ratio column; dashed = nominal channel cut.
# --------------------------------------------------------------------------- #
# Case 1 (routed-only) -- reduce only the routed experts, always-on MLP stays full.
C1_UP = {              # rank-by-up (solid): x = active-param reduction
    28.1: {"mmlu": 0.766, "arc": 0.688},   # active-param ratio 0.719 (nominal 60%)
    35.1: {"mmlu": 0.744, "arc": 0.665},   # active-param ratio 0.649 (nominal 75%)
    42.2: {"mmlu": 0.667, "arc": 0.536},   # active-param ratio 0.578 (nominal 90%)
}
C1_INTER = {           # rank-by-intermediate (dashed): x = nominal cut
    60: {"mmlu": 0.775, "arc": 0.701},
    75: {"mmlu": 0.776, "arc": 0.697},
    90: {"mmlu": 0.742, "arc": 0.654},
}
# Case 2 (joint) -- pool always-on MLP channels with routed channels into one budget.
C2_UP = {              # rank-by-up (solid): x = active-param reduction
    38.7: {"mmlu": 0.749, "arc": 0.667},   # active-param ratio 0.613 (nominal 60%, read from prior fig)
    48.2: {"mmlu": 0.663, "arc": 0.500},   # active-param ratio 0.518 (nominal 75%, read from prior fig)
    58.0: {"mmlu": 0.254, "arc": 0.177},   # active-param ratio 0.420 (nominal 90%, week9.md)
}
C2_INTER = {           # rank-by-intermediate (dashed): x = nominal cut
    60: {"mmlu": 0.772, "arc": 0.699},
    75: {"mmlu": 0.760, "arc": 0.650},
    90: {"mmlu": 0.587, "arc": 0.438},
}
FULL = {"mmlu": 0.776, "arc": 0.702}       # dense / full-model baseline (Gemma4-26B-A4B -pt)


def _series(d, metric):
    """Sorted (x, y%) for one series and one metric."""
    pts = sorted((x, v[metric] * 100.0) for x, v in d.items())
    return [p[0] for p in pts], [p[1] for p in pts]


def _panel(out_dir, fname, metric, *, ylabel, ylim, title, full):
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.2))

    # Case 2 (joint, amber) first so Case 1 (blue) draws on top.
    x, y = _series(C2_INTER, metric)
    ax.plot(x, y, color=AMBER, lw=2.0, marker="o", ms=6.5, ls="--", mfc="white")
    x, y = _series(C2_UP, metric)
    ax.plot(x, y, color=AMBER, lw=2.4, marker="s", ms=6.5, ls="-")

    x, y = _series(C1_INTER, metric)
    ax.plot(x, y, color=BLUE, lw=2.0, marker="o", ms=6.5, ls="--", mfc="white", zorder=5)
    x, y = _series(C1_UP, metric)
    ax.plot(x, y, color=BLUE, lw=2.6, marker="s", ms=6.5, ls="-", zorder=6)

    ax.axhline(full, color=GREY, lw=1.4, ls=":")
    ax.text(96.0, full + 0.6, f"full model = {full:.1f}", color=MUTED,
            fontsize=12.0, ha="right", va="bottom")

    ax.set_xlabel("FFN active parameter reduction")
    ax.set_ylabel(ylabel)
    ticks = [30, 40, 50, 60, 70, 80, 90]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{n}%" for n in ticks])
    ax.set_xlim(28, 97)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", color=INK, weight="bold")

    # legend: colour = case, linestyle = scoring signal (mirrors the original 2x2)
    handles = [
        Line2D([], [], color=BLUE, lw=2.4, label="Case 1 (routed-only)"),
        Line2D([], [], color=AMBER, lw=2.4, label="Case 2 (joint)"),
        Line2D([], [], color=MUTED, lw=2.0, ls="-", marker="s", ms=6.5,
               label="rank-by-up (solid · active-param)"),
        Line2D([], [], color=MUTED, lw=2.0, ls="--", marker="o", ms=6.5,
               mfc="white", label="rank-by-intermediate (dashed · nominal)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{fname}.{ext}"), dpi=400)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out-dir", default=os.path.join(repo, "docs/presentation/figs"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    _panel(args.out_dir, "gemma4_mmlu", "mmlu",
           ylabel="MMLU accuracy  (%)", ylim=(20, 84),
           title="Gemma4-26B-A4B — MMLU", full=FULL["mmlu"] * 100.0)
    _panel(args.out_dir, "gemma4_arc_c", "arc",
           ylabel="ARC-Challenge accuracy  (%)", ylim=(14, 74),
           title="Gemma4-26B-A4B — ARC-Challenge", full=FULL["arc"] * 100.0)
    print(f"[plot] wrote gemma4_mmlu + gemma4_arc_c to {args.out_dir}")


if __name__ == "__main__":
    main()
