#!/usr/bin/env python
"""Two motivation figures for the slides — data hard-coded from measured runs.

All numbers are transcribed from the results docs (no capture .npz needed), so
this script is self-contained and instant to run.

  * ``fig_prune_sweep_mmlu.pdf`` / ``fig_prune_sweep_hellaswag.pdf`` — accuracy vs
    nominal reduction on the *dense* Qwen3-30B-A3B per-token prune-ratio sweep
    (``reduce=gate+down``), one metric per figure, each with its dashed
    dense-baseline reference. Source: ``test/results/mobe/mobe_30b.md`` §"Results —
    dense Qwen3-30B-A3B, prune-ratio sweep".

  * ``fig_offline_vs_online.pdf`` / ``fig_offline_vs_online_mmlu.pdf`` — offline
    (static) vs online (per-token) channel selection on Qwen3-30B-A3B-Thinking-2507,
    HellaSwag 0-shot acc_norm and MMLU 5-shot acc, across active-param reductions.
    Two offline curves (reduce-top-k, Level-1 pivchol) vs two blue online curves
    (realized per-token sweep at the real cut, oracle_mag ceiling at the nominal
    cut), dense dashed reference. Source:
    ``docs/exps/dynamic_active_param/q3_30b_dynamic_active.md`` and
    ``test/results/mobe/mobe_30b.md``.
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
    "axes.titlesize": 10.5, "axes.labelsize": 13.0,
    "xtick.labelsize": 12.0, "ytick.labelsize": 12.0,
    "legend.fontsize": 12.0, "figure.titlesize": 11.5,
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
DENSE_HELLASWAG = 77.8                                     # uncompressed baseline


def _prune_sweep_panel(out_dir, fname, ys, color, marker, series_label,
                       dense_val, dense_label, dense_color, title_metric,
                       dense_ls="--", oracle_x=None, oracle_ys=None,
                       oracle_label=None, plot_series=True,
                       dense_text_below=False):
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.2))

    if plot_series:
        ax.plot(NOMINAL, ys, color=color, lw=2.2, marker=marker, ms=6.8,
                label=series_label)
    # optional per-token oracle_mag ceiling (same nominal channel-reduction axis)
    if oracle_ys is not None:
        ax.plot(oracle_x, oracle_ys, color=color, lw=2.2, marker="o", ms=6.5,
                ls="--", mfc="white", label=oracle_label)
    ax.axhline(dense_val, color=dense_color, lw=1.4, ls=dense_ls, alpha=0.9)
    _dy, _va = (-0.35, "top") if dense_text_below else (0.35, "bottom")
    ax.text(NOMINAL[0] - 1.0, dense_val + _dy, dense_label, color=dense_color,
            fontsize=12.0, va=_va, ha="left")

    ax.set_xlabel("channel reduction ratio")
    ax.set_ylabel("accuracy  (%)")
    ax.set_xticks(NOMINAL)
    ax.set_xticklabels([f"{n}%" for n in NOMINAL])
    ax.set_xlim(46, 94)
    ax.set_ylim(50, 82)
    ax.set_title(title_metric, loc="left", color=INK, weight="bold")
    ax.legend(frameon=False, loc="lower left")
    _clean(ax)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{fname}.{ext}"), dpi=400)
    plt.close(fig)


def fig_mobe_prune_sweep(out_dir):
    # dense reference in muted grey dotted so it does not clash with the blue
    # realized/oracle pair (mirrors fig_offline_vs_online).
    _prune_sweep_panel(
        out_dir, "fig_prune_sweep_mmlu", MMLU, BLUE, "o",
        "MMLU (5-shot acc)", 79.5, "dense MMLU 79.5", MUTED,
        "MMLU", dense_ls=":", oracle_x=MMLU_ORACLE_X, oracle_ys=MMLU_ORACLE,
        oracle_label="MMLU (5-shot acc)", plot_series=False,
        dense_text_below=True)
    _prune_sweep_panel(
        out_dir, "fig_prune_sweep_hellaswag", HELLASWAG, AMBER, "s",
        "HellaSwag (0-shot acc_norm)", DENSE_HELLASWAG,
        "dense HellaSwag", AMBER, "HellaSwag")


# --------------------------------------------------------------------------- #
# Figure 4 — offline vs online channel selection (HellaSwag + MMLU)
# Base: Qwen3-30B-A3B(-Thinking-2507). reduce-top-k / Level-1 / oracle_mag from
# q3_30b_dynamic_active.md; realized per-token sweep from mobe_30b.md.
# Two offline curves (reduce-top-k red, Level-1 pivchol black) at their NOMINAL
# cut (they shrink all three FFN matrices, so nominal = whole-FFN cut). Two blue
# online views of per-token selection:
#   * dashed = per-token oracle_mag ceiling, at the NOMINAL channel reduction.
#   * solid  = realized reduce=gate+down per-token sweep (up_proj kept full →
#     47-of-48 layers cut), at the REAL model-wide active-param cut.
# --------------------------------------------------------------------------- #
REDUCTION = [50, 62.5, 75, 87.5]                           # nominal (down_proj-dim) cut
OFFLINE_X = REDUCTION                                      # offline cuts all 3 matrices

# realized per-token method (mobe_30b.md prune sweep): nominal 50/60/70/80/90%
# → model-wide active cut (up_proj full, 47-of-48 layers) −32.64…−58.75%.
PRUNE_REAL_X = [32.64, 39.17, 45.69, 52.22, 58.75]

# --- HellaSwag 0-shot acc_norm ---
HS_REDUCE_TOPK = [75.2, 69.8, 49.4, 26.2]                  # offline: fewer experts
HS_LEVEL1 = [74.26, 70.54, 63.60, 44.15]                   # offline: pivchol nested
HS_ORACLE = [78.54, 78.76, 78.28, 76.84]                   # online oracle @ nominal
HS_ORACLE_X = REDUCTION
HS_PRUNE = [77.76, 77.62, 77.34, 75.82, 72.92]             # online realized @ real cut
DENSE_HS = 78.56                                           # dense baseline (Thinking)

# --- MMLU 5-shot acc (q3_30b_dynamic_active.md "MMLU 5-shot (acc)" table) ---
MMLU_REDUCE_TOPK = [74.1, 65.1, 34.9, 24.4]                # offline: fewer experts
MMLU_LEVEL1 = [77.85, 76.16, 70.81, 45.51]                 # offline: pivchol nested
MMLU_ORACLE = [80.22, 80.53, 79.48]                        # online oracle @ nominal
MMLU_ORACLE_X = [50, 75, 87.5]                             # no 62.5% oracle point
MMLU_PRUNE = [79.33, 79.50, 79.15, 77.85, 76.14]           # online realized @ real cut
DENSE_MMLU_HS = 79.5                                       # dense 5-shot MMLU


def _offline_vs_online_panel(out_dir, fname, *, reduce_topk, level1, oracle,
                             oracle_x, prune, dense, ylabel, ylim, title):
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.2))

    # offline methods
    ax.plot(OFFLINE_X, reduce_topk, color="#20242c", lw=2.2, marker="^", ms=7.0,
            ls="-", label="Activate fewer experts")
    ax.plot(OFFLINE_X, level1, color=RED, lw=2.0, marker="s", ms=6.0,
            ls="-", label="fixed channel ranking")
    # online method — blue.
    #  solid : realized per-token gate+down selection, at the REAL model-wide cut.
    #  dashed: per-token oracle_mag ceiling, at the NOMINAL channel reduction.
    ax.plot(PRUNE_REAL_X, prune, color=BLUE, lw=2.6, marker="o", ms=7.5,
            ls="-", label="dynamic channel ranking (realized)")
    ax.plot(oracle_x, oracle, color=BLUE, lw=2.2, marker="o", ms=6.5,
            ls="--", mfc="white", label="dynamic channel ranking (nominal)")

    ax.axhline(dense, color=GREY, lw=1.4, ls=":")
    ax.text(OFFLINE_X[-1], dense + 0.6, f"dense = {dense:.1f}",
            color=MUTED, fontsize=12.0, ha="right", va="bottom")

    ax.set_xlabel("whole-FFN active-parameter reduction  (gate + up + down)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([30, 40, 50, 60, 70, 80, 90])
    ax.set_xticklabels([f"{n}%" for n in (30, 40, 50, 60, 70, 80, 90)])
    ax.set_xlim(28, 93)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", color=INK, weight="bold")
    ax.legend(frameon=False, loc="lower left")
    _clean(ax)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{fname}.{ext}"), dpi=400)
    plt.close(fig)


def fig_offline_vs_online(out_dir):
    _offline_vs_online_panel(
        out_dir, "fig_offline_vs_online",
        reduce_topk=HS_REDUCE_TOPK, level1=HS_LEVEL1,
        oracle=HS_ORACLE, oracle_x=HS_ORACLE_X, prune=HS_PRUNE, dense=DENSE_HS,
        ylabel="HellaSwag acc_norm  (%)", ylim=(24, 82),
        title=("Online per-token selection stays near dense where offline collapses\n"
               "Qwen3-30B-A3B-Thinking-2507, HellaSwag 0-shot, no fine-tuning"))
    _offline_vs_online_panel(
        out_dir, "fig_offline_vs_online_mmlu",
        reduce_topk=MMLU_REDUCE_TOPK, level1=MMLU_LEVEL1,
        oracle=MMLU_ORACLE, oracle_x=MMLU_ORACLE_X, prune=MMLU_PRUNE,
        dense=DENSE_MMLU_HS,
        ylabel="MMLU acc  (5-shot, %)", ylim=(20, 84),
        title=("Online per-token selection stays near dense where offline collapses\n"
               "Qwen3-30B-A3B-Thinking-2507, MMLU 5-shot, no fine-tuning"))


def main():
    ap = argparse.ArgumentParser()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out-dir", default=os.path.join(repo, "docs/presentation/figs"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fig_mobe_prune_sweep(args.out_dir)
    fig_offline_vs_online(args.out_dir)
    print(f"[plot] wrote fig_prune_sweep_mmlu + fig_prune_sweep_hellaswag "
          f"+ fig_offline_vs_online + fig_offline_vs_online_mmlu to {args.out_dir}")


if __name__ == "__main__":
    main()
