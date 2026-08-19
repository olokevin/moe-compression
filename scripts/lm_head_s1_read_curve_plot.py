#!/usr/bin/env python3
"""Plot ARC-C accuracy against the S1 read fraction, for both Qwen3 heads.

Answers "how few of the head's parameters can a token touch before the answer
degrades?" -- one panel per model, x = reads/token as a % of ``V*D``, y = ARC-C
``acc_norm``. The dense head is the dashed reference; the ~25%-of-parameters
*structural* baselines (low-rank, row pruning) are drawn as isolated markers because
they live on the stored-parameter axis and there is no ladder for them.

Numbers are read from the sweep JSONs -- nothing is typed in, nothing is
interpolated. A variant that has not been evaluated is simply absent from the curve.

    python scripts/lm_head_s1_read_curve_plot.py
    python scripts/lm_head_s1_read_curve_plot.py --task hellaswag
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- palette + rcParams: same house style as presentation_motivation_plot.py -- #
INK = "#1c2330"
MUTED = "#68717f"
BLUE = "#2f6fdb"
AMBER = "#e08a1e"
RED = "#c8402f"
GREY = "#9aa4b2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": "#cfd6e0", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 11.5, "axes.labelsize": 12.5,
    "xtick.labelsize": 11.0, "ytick.labelsize": 11.0,
    "legend.fontsize": 10.0, "figure.titlesize": 12.5,
    "savefig.bbox": "tight", "axes.grid": True,
    "grid.color": "#eef1f6", "grid.linewidth": 0.8,
})

MODELS = [
    ("Qwen3-30B-A3B", 2048, [
        "results_eval/lm_head_s1_30b_lowread_arc.json",
        "results_eval/lm_head_s1_30b_c4arc.json",
        "results_eval/lm_head_s1_30b_hs.json",
    ]),
    ("Qwen3-0.6B", 1024, [
        "results_eval/lm_head_s1_0_6b_lowread_arc.json",
        "results_eval/lm_head_s1_0_6b_tasks.json",
    ]),
]

# Structural baselines: these sit on the STORED-parameter axis, so they are plotted as
# single markers at their storage fraction, not as a read ladder.
BASELINES = {"f2_lr25": ("low-rank", RED, "s"), "b1p_t32k": ("row pruning", AMBER, "^"),
             "b1a_t4k": ("static sparse reads", GREY, "v")}


def score(row, task):
    """(accuracy %, stderr %) or (None, None)."""
    raw = row.get(f"{task}_raw")
    if not isinstance(raw, dict):
        return None, None
    inner = raw.get(task) if isinstance(raw.get(task), dict) else raw
    for k in ("acc_norm,none", "acc,none"):
        if k in inner:
            se = inner.get(k.replace(",none", "_stderr,none"))
            return 100 * inner[k], (100 * se if se is not None else None)
    return None, None


def collect(paths, task):
    """variant -> (read %, score). Later files do not overwrite earlier ones."""
    out, dense, dense_se = {}, None, None
    for p in paths:
        if not os.path.exists(p):
            continue
        for row in json.load(open(p))["rows"]:
            s, se = score(row, task)
            if s is None:
                continue
            v = row["variant"]
            if v == "dense":
                if dense is None:
                    dense, dense_se = s, se
                continue
            if v in out:
                continue
            rf = row.get("read_param_frac", row.get("read_frac"))
            out[v] = (100 * rf, s)
    return out, dense, dense_se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="arc_challenge")
    ap.add_argument("--out", default="docs/exps/lm_head/figures/fig_s1_read_curve")
    ap.add_argument("--linx", action="store_true",
                    help="linear read axis (default is log: the ladder is geometric)")
    a = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    label = {"arc_challenge": "ARC-Challenge", "hellaswag": "HellaSwag"}.get(a.task, a.task)

    for ax, (name, D, paths) in zip(axes, MODELS):
        pts, dense, dense_se = collect(paths, a.task)
        # Ablation variants share a read budget with s1_r25_n8k; they are separate
        # points on a different question, so they must not join the budget curve.
        s1 = sorted((rf, sc) for v, (rf, sc) in pts.items()
                    if v.startswith("s1") and v not in ("s1_freq_n8k", "s1_r25_n8k_inf",
                                                        "s1_r25_n8k_static", "s1_r25_n8k_raw"))
        if dense is not None:
            # +/-1 stderr band: ARC-C on 1172 items has stderr ~1.4 pt, so every
            # wiggle inside this band is sampling noise, not a trend.
            if dense_se:
                ax.axhspan(dense - dense_se, dense + dense_se, color=BLUE, alpha=0.08,
                           zorder=1, lw=0)
            ax.axhline(dense, ls="--", lw=1.6, color=INK, zorder=2)
            lab = f" dense {dense:.2f}" + (f" $\\pm$ {dense_se:.2f} " if dense_se else " ")
            # left-anchored: the curve reaches dense at the RIGHT of the axis, so a
            # right-anchored label sits on top of it
            ax.text(0.015, dense, lab, color=INK, fontsize=9.5,
                    va="bottom", ha="left", transform=ax.get_yaxis_transform(),
                    bbox=dict(fc="white", ec="none", pad=1.0))
        if s1:
            ax.plot([p[0] for p in s1], [p[1] for p in s1], "-o", color=BLUE, lw=2.0,
                    ms=5.5, label="S1 screen-and-refine", zorder=5)
        for v, (nm, col, mk) in BASELINES.items():
            if v in pts:
                rf, sc = pts[v]
                ax.plot([rf], [sc], mk, color=col, ms=8.0, label=nm, zorder=4,
                        mew=0, alpha=0.95)
        if not a.linx:
            # the ladder is geometric (r0 and N both halve), so log is the honest axis
            ax.set_xscale("log")
            ax.set_xticks([2, 3, 5, 10, 25])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"{name}   (head 151936$\\times${D})", color=INK)
        ax.set_xlabel("head parameters read per token  (% of $V \\times D$)")
        ax.set_ylabel(f"{label} accuracy (%)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, loc="lower right")

    fig.suptitle(f"{label} vs how much of the lm_head a token reads", color=INK)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}_{a.task}.{ext}", dpi=200)
    print(f"wrote {a.out}_{a.task}.png / .pdf")
    for ax, (name, _, paths) in zip(axes, MODELS):
        pts, dense, dense_se = collect(paths, a.task)
        print(f"\n{name}: dense {dense:.2f} +/- {dense_se:.2f}")
        for v, (rf, sc) in sorted(pts.items(), key=lambda kv: -kv[1][0]):
            d = sc - dense
            flag = "" if not dense_se else ("  within 1 se" if abs(d) <= dense_se else "")
            print(f"   {v:<22} reads={rf:6.2f}%  {label}={sc:6.2f}  ({d:+.2f}){flag}")


if __name__ == "__main__":
    main()
