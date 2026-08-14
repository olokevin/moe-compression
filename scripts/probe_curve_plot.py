#!/usr/bin/env python3
"""Plot `input_sparse` accuracy vs used-parameter cut, in the presentation style.

Emits ``fig_probe_curve_hellaswag.{pdf,png}`` and ``fig_probe_curve_mmlu.{pdf,png}``,
styled to match ``docs/presentation/figs/fig_offline_vs_online_mmlu.pdf`` (same
rcParams, palette, figure size, spine/grid treatment and legend placement as
``scripts/presentation_motivation_plot.py``).

Each figure puts the best-practice `input_sparse` curve (blue) against the offline
baselines it has to beat -- Level-1 `pivchol` (red) and activate-fewer-experts
(near-black) -- plus the full-width `oracle_mag_noW` scoring ceiling and the dense
dashed reference. The x axis is the **used-parameter** cut (scoring reads + compute
reads), which is the only frame on which all of these are comparable.

Results are read from ``docs/results/btt_dynamic/probe_curve.json`` when present so
the numbers live in one place; the literals below are the fallback and are annotated
with provenance. Points that have not been measured yet are simply absent from the
curve -- nothing is interpolated or invented.

Usage::

    python scripts/probe_curve_plot.py
    python scripts/probe_curve_plot.py --results docs/results/btt_dynamic/probe_curve.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- palette + rcParams: identical to presentation_motivation_plot.py ------- #
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
    "legend.fontsize": 12.0, "figure.titlesize": 11.5,
    "savefig.bbox": "tight", "axes.grid": True,
    "grid.color": "#eef1f6", "grid.linewidth": 0.8,
})

# --------------------------------------------------------------------------- #
# input_sparse best practice: bits=16, use_gate=true, input_alloc=router,
# lam=1.0, k_min=0, one global (rho_input, rho_channel) per budget, solved by
# scripts/probe_split_solve.py. x = used-param cut % = 100*(1 - rho_ch - 2*rho_in/3).
# Sources: docs/exps/dynamic_active_param/efficient_scorer.md leaderboard.
# `None` = not yet measured; such points are dropped, never interpolated.
# --------------------------------------------------------------------------- #
PROBE_HS = {
    63.3: None,     # solved 0.2500/0.20003 -- eval running
    68.3: None,     # solved 0.2400/0.15670 -- eval running
    73.3: None,     # solved 0.1875/0.14170 -- eval running
    75.0: 74.08,    # measured (probe_router_split_opt_hellaswag)
    77.5: None,     # solved 0.1875/0.10000 -- eval running
    80.0: None,     # solved 0.1575/0.09500 -- eval running
}
PROBE_MMLU = {
    63.3: None,
    68.3: None,
    73.3: None,
    75.0: None,     # eval running (A100-3)
    77.5: None,
    80.0: None,
}

# Off-best-practice reference: the hand-picked rho_input=0.25 rows that used
# input_alloc=uniform (rows 4-6), plus row 7 which is 0.25/0.10 WITH router.
# Kept as a second blue curve because it is what the doc's headline quotes.
HANDPICKED_HS = {63.3: 76.72, 68.3: 76.47, 73.3: 74.64}   # 73.3 is row 7 (+router)
HANDPICKED_MMLU = {63.3: 79.10, 68.3: 78.63, 73.3: 77.67}

# Full-width scoring ceiling: oracle_mag_noW, gate+up at full width (row 1).
ORACLE_NOW_HS = {29.2: 77.11}
ORACLE_NOW_MMLU = {29.2: 79.44}

# Offline baselines (no per-token scoring pass), from q3_30b_dynamic_active.md.
LEVEL1_HS = {50.0: 74.26, 62.5: 70.54, 75.0: 63.60, 87.5: 44.15}
LEVEL1_MMLU = {50.0: 77.85, 62.5: 76.16, 75.0: 70.81, 87.5: 45.51}
TOPK_HS = {50.0: 75.2, 62.5: 69.8, 75.0: 49.4, 87.5: 26.2}
TOPK_MMLU = {50.0: 74.1, 62.5: 65.1, 75.0: 34.9, 87.5: 24.4}

DENSE_HS = 78.56          # Qwen3-30B-A3B-Thinking-2507, HellaSwag 0-shot acc_norm
DENSE_MMLU = 80.91        # full MMLU 5-shot acc (measured; see the doc)


def _xy(d):
    """Sorted (x, y) with unmeasured points dropped."""
    pts = sorted((x, y) for x, y in d.items() if y is not None)
    return [p[0] for p in pts], [p[1] for p in pts]


def _panel(out_dir, fname, *, probe, handpicked, oracle, level1, topk, dense,
           ylabel, ylim, title):
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.2))

    # offline baselines first so the online curves draw on top
    x, y = _xy(topk)
    ax.plot(x, y, color="#20242c", lw=2.2, marker="^", ms=7.0, ls="-",
            label="Activate fewer experts (offline)")
    x, y = _xy(level1)
    ax.plot(x, y, color=RED, lw=2.0, marker="s", ms=6.0, ls="-",
            label="Fixed channel ranking (offline)")

    # full-width scoring ceiling -- a single point, drawn as a marker
    x, y = _xy(oracle)
    if x:
        ax.plot(x, y, color=GREEN, marker="*", ms=15.0, ls="none",
                label="oracle_mag_noW (full-width scoring)")

    # off-best-practice hand-picked rho_input=0.25 rows
    x, y = _xy(handpicked)
    if x:
        ax.plot(x, y, color=BLUE, lw=2.0, marker="o", ms=6.5, ls="--",
                mfc="white", label=r"input_sparse, hand-picked $\rho_{in}$=0.25")

    # the best-practice curve
    x, y = _xy(probe)
    if x:
        ax.plot(x, y, color=BLUE, lw=2.6, marker="o", ms=7.5, ls="-",
                label=r"input_sparse, solved split (best practice)")

    ax.axhline(dense, color=GREY, lw=1.4, ls=":")
    ax.text(88.0, dense + 0.6, f"dense = {dense:.1f}", color=MUTED,
            fontsize=12.0, ha="right", va="bottom")

    ax.set_xlabel("used-parameter reduction  (scoring + compute)")
    ax.set_ylabel(ylabel)
    ticks = [30, 40, 50, 60, 70, 80, 90]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{n}%" for n in ticks])
    ax.set_xlim(26, 91)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", color=INK, weight="bold")
    ax.legend(frameon=False, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{fname}.{ext}"), dpi=400)
    plt.close(fig)
    n = len(_xy(probe)[0])
    print(f"[plot] {fname}: {n}/{len(probe)} best-practice points measured")


def main():
    ap = argparse.ArgumentParser()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out-dir", default=os.path.join(repo, "docs/presentation/figs"))
    ap.add_argument("--results",
                    default=os.path.join(repo, "docs/results/btt_dynamic/probe_curve.json"),
                    help="optional JSON of measured points; overrides the literals")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    hs, mmlu = dict(PROBE_HS), dict(PROBE_MMLU)
    if os.path.exists(args.results):
        d = json.load(open(args.results))
        for cut, v in d.get("hellaswag", {}).items():
            hs[float(cut)] = v
        for cut, v in d.get("mmlu", {}).items():
            mmlu[float(cut)] = v
        print(f"[plot] merged measured points from {args.results}")

    _panel(
        args.out_dir, "fig_probe_curve_hellaswag",
        probe=hs, handpicked=HANDPICKED_HS, oracle=ORACLE_NOW_HS,
        level1=LEVEL1_HS, topk=TOPK_HS, dense=DENSE_HS,
        ylabel="HellaSwag acc_norm  (%)", ylim=(24, 82),
        title=("Cheap per-token scoring holds accuracy far past where offline collapses\n"
               "Qwen3-30B-A3B-Thinking-2507, HellaSwag 0-shot, no fine-tuning"))
    _panel(
        args.out_dir, "fig_probe_curve_mmlu",
        probe=mmlu, handpicked=HANDPICKED_MMLU, oracle=ORACLE_NOW_MMLU,
        level1=LEVEL1_MMLU, topk=TOPK_MMLU, dense=DENSE_MMLU,
        ylabel="MMLU acc  (5-shot, %)", ylim=(20, 84),
        title=("Cheap per-token scoring holds accuracy far past where offline collapses\n"
               "Qwen3-30B-A3B-Thinking-2507, MMLU 5-shot, no fine-tuning"))
    print(f"[plot] wrote fig_probe_curve_hellaswag + fig_probe_curve_mmlu to {args.out_dir}")


if __name__ == "__main__":
    main()
