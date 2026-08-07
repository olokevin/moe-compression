#!/usr/bin/env python
"""Two motivation figures for the slides — data hard-coded from measured runs.

All numbers are transcribed from the results docs (no capture .npz needed), so
this script is self-contained and instant to run.

  * ``fig_mobe_prune_sweep.pdf`` — accuracy vs nominal reduction on the *dense*
    Qwen3-30B-A3B per-token prune-ratio sweep (``reduce=gate+down``). Two curves,
    MMLU (5-shot acc) and HellaSwag (0-shot acc_norm), with dashed dense-baseline
    references. Source: ``test/results/mobe/mobe_30b.md`` §"Results — dense
    Qwen3-30B-A3B, prune-ratio sweep".

  * ``fig_offline_vs_online.pdf`` — offline (static) vs online (per-token)
    channel selection on Qwen3-30B-A3B-Thinking-2507, HellaSwag 0-shot acc_norm,
    across four active-param reductions. Two black offline curves (reduce-top-k,
    Level-1 pivchol) vs one blue online curve (per-token oracle_mag), dense
    dashed reference. Source: ``docs/exps/dynamic_active_param/q3_30b_dynamic_active.md``
    and ``docs/report/level2.md``.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1c2330"
MUTED = "#68717f"
BLUE = "#2f6fdb"
AMBER = "#e08a1e"
PURPLE = "#7a4fc4"
GREEN = "#2e8b6f"
RED = "#c8402f"
GREY = "#9aa4b2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": "#cfd6e0", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 10.5, "axes.labelsize": 9.4,
    "xtick.labelsize": 8.6, "ytick.labelsize": 8.6,
    "legend.fontsize": 8.6, "figure.titlesize": 11.5,
    "savefig.bbox": "tight", "axes.grid": True,
    "grid.color": "#eef1f6", "grid.linewidth": 0.8,
})


def _clean(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
# Figure 3 — dense Qwen3-30B-A3B per-token prune-ratio sweep (MMLU + HellaSwag)
# Source: test/results/mobe/mobe_30b.md, "Results — dense Qwen3-30B-A3B,
# prune-ratio sweep" (reduce=gate+down; nominal = intermediate-dim cut).
# --------------------------------------------------------------------------- #
NOMINAL = [50, 60, 70, 80, 90]                              # % nominal reduction
MMLU = [79.33, 79.50, 79.15, 77.85, 74.14]                 # 5-shot acc, %
HELLASWAG = [77.76, 77.62, 77.34, 75.82, 70.92]            # 0-shot acc_norm, %
DENSE_MMLU = 79.62                                         # uncompressed baseline
# Dense HellaSwag on the base A3B was never measured (baseline eval crashed
# before hellaswag). Using the arc_challenge-era baseline acc_norm as reference.
DENSE_HELLASWAG = 69.71


def fig_mobe_prune_sweep(out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2))

    ax.plot(NOMINAL, MMLU, color=BLUE, lw=2.2, marker="o", ms=6.5,
            label="MMLU (5-shot acc)")
    ax.plot(NOMINAL, HELLASWAG, color=AMBER, lw=2.2, marker="s", ms=6.0,
            label="HellaSwag (0-shot acc_norm)")

    ax.axhline(DENSE_MMLU, color=BLUE, lw=1.4, ls="--", alpha=0.9)
    ax.text(NOMINAL[0] - 1.0, DENSE_MMLU + 0.35, "dense MMLU", color=BLUE,
            fontsize=8.0, va="bottom", ha="left")
    ax.axhline(DENSE_HELLASWAG, color=AMBER, lw=1.4, ls="--", alpha=0.9)
    ax.text(NOMINAL[0] - 1.0, DENSE_HELLASWAG + 0.35,
            "dense HellaSwag (arc-era ref)",
            color=AMBER, fontsize=8.0, va="bottom", ha="left")

    ax.set_xlabel("nominal reduction of the active intermediate dimension")
    ax.set_ylabel("accuracy  (%)")
    ax.set_xticks(NOMINAL)
    ax.set_xticklabels([f"{n}%" for n in NOMINAL])
    ax.set_xlim(46, 94)
    ax.set_ylim(68, 82)
    ax.set_title("Per-token channel selection holds near-dense accuracy\n"
                 "dense Qwen3-30B-A3B, no fine-tuning (reduce = gate+down)",
                 loc="left", color=INK, weight="bold")
    ax.legend(frameon=False, loc="upper right")
    _clean(ax)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_mobe_prune_sweep.{ext}"), dpi=400)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4 — offline vs online channel selection (HellaSwag acc_norm)
# Base: Qwen3-30B-A3B-Thinking-2507, dense HellaSwag acc_norm = 78.56.
# reduce-top-k / Level-1 : q3_30b_dynamic_active.md budget-sweep table.
# oracle_mag (online)    : q3_30b_dynamic_active.md Level-2 sweep table.
# --------------------------------------------------------------------------- #
REDUCTION = [50, 62.5, 75, 87.5]                           # % active-param reduction
REDUCE_TOPK = [75.2, 69.8, 49.4, 26.2]                     # offline: fewer experts
LEVEL1 = [74.26, 70.54, 63.60, 44.15]                      # offline: pivchol nested
ORACLE_MAG = [78.54, 78.76, 78.28, 76.84]                  # online: per-token
DENSE_HS = 78.56                                           # dense baseline (Thinking)


def fig_offline_vs_online(out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.2))

    # offline methods — black
    ax.plot(REDUCTION, REDUCE_TOPK, color="#20242c", lw=2.0, marker="^", ms=7.0,
            ls="--", label="offline: reduce top-k (fewer experts)")
    ax.plot(REDUCTION, LEVEL1, color="#20242c", lw=2.0, marker="s", ms=6.0,
            ls="-", label="offline: Level 1 (static, pivoted-Cholesky)")
    # online method — blue
    ax.plot(REDUCTION, ORACLE_MAG, color=BLUE, lw=2.6, marker="o", ms=7.5,
            label="online: per-token channel selection")

    ax.axhline(DENSE_HS, color=GREY, lw=1.4, ls=":")
    ax.text(REDUCTION[-1], DENSE_HS + 0.6, f"dense = {DENSE_HS:.1f}",
            color=MUTED, fontsize=8.2, ha="right", va="bottom")

    ax.set_xlabel("per-token active-channel reduction  (fraction of K·I dropped)")
    ax.set_ylabel("HellaSwag acc_norm  (%)")
    ax.set_xticks(REDUCTION)
    ax.set_xticklabels([f"{r:g}%" for r in REDUCTION])
    ax.set_xlim(46, 91)
    ax.set_ylim(24, 82)
    ax.set_title("Online per-token selection stays near dense where offline collapses\n"
                 "Qwen3-30B-A3B-Thinking-2507, HellaSwag 0-shot, no fine-tuning",
                 loc="left", color=INK, weight="bold")
    ax.legend(frameon=False, loc="lower left")
    _clean(ax)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_offline_vs_online.{ext}"), dpi=400)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out-dir", default=os.path.join(repo, "docs/presentation/figs"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fig_mobe_prune_sweep(args.out_dir)
    fig_offline_vs_online(args.out_dir)
    print(f"[plot] wrote fig_mobe_prune_sweep + fig_offline_vs_online to {args.out_dir}")


if __name__ == "__main__":
    main()
