#!/usr/bin/env python
"""Plot the low-rank-scorer recall investigation (``lowrank_scorer_recall.py``).

Reads one or more ``recall.json`` files and emits, into ``--fig-dir``:

  * ``recall_vs_cost.png``  — the Pareto view: top-B agreement with the
    `oracle_mag_noW` selection against the compute spent purely to *decide*
    (in units of one full-width matmul), with the `oracle_up` reference (cost
    1.0) and the random baseline (recall = rho) as the two anchors.
  * ``mass_vs_cost.png``   — same axes but for captured oracle score *mass*,
    which is the quantity that should track accuracy: picking a different
    channel of equal magnitude is harmless, missing a dominant one is not.
  * ``svd_vs_btt.png``     — SVD (m=n=1) against the block grids at equal cost,
    i.e. does the extra effective rank BTT buys actually help on real weights.
  * ``by_depth.png``       — per-layer spread for a few headline variants.

CPU-only, no model needed.
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(paths):
    rows, ratios = [], None
    for p in paths:
        d = json.load(open(p))
        rows += d["rows"]
        ratios = d["ratios"] if ratios is None else ratios
    return rows, ratios


def _agg(rows):
    """name -> dict of layer-averaged metrics (+ the per-layer values)."""
    by = defaultdict(list)
    for r in rows:
        by[r["name"]].append(r)
    out = {}
    for name, rs in by.items():
        keys = [k for k in rs[0] if k.startswith(("recall@", "mass@", "random_recall@"))]
        d = {
            "cost": rs[0]["cost_frac_of_one_matmul"],
            "m": rs[0]["m"], "n": rs[0]["n"], "rank": rs[0]["rank"],
            "use_gate": rs[0]["use_gate"],
            "spearman": float(np.mean([r["spearman"] for r in rs])),
            "layers": sorted(r["layer"] for r in rs),
        }
        for k in keys:
            d[k] = float(np.mean([r[k] for r in rs]))
            d[k + "__per_layer"] = {r["layer"]: r[k] for r in rs}
        out[name] = d
    return out


def _family(name):
    if name == "oracle_up_ref":
        return "oracle_up (full up_proj)"
    base = name.replace("_uponly", "")
    grid = "SVD (1x1)" if base.startswith("svd") else base.split("_r")[0].replace("btt_m", "BTT ").replace("n", "x")
    return grid + (" · up only" if name.endswith("_uponly") else " · up+gate")


_STYLE = {
    "SVD (1x1) · up+gate":  dict(color="#1f77b4", marker="o", ls="-"),
    "SVD (1x1) · up only":  dict(color="#1f77b4", marker="o", ls="--", mfc="white"),
    "BTT 2x2 · up+gate":    dict(color="#d62728", marker="s", ls="-"),
    "BTT 2x2 · up only":    dict(color="#d62728", marker="s", ls="--", mfc="white"),
    "BTT 4x2 · up+gate":    dict(color="#2ca02c", marker="^", ls="-"),
    "BTT 4x2 · up only":    dict(color="#2ca02c", marker="^", ls="--", mfc="white"),
}


def _curve_plot(ax, agg, metric, rho, ylabel):
    fams = defaultdict(list)
    for name, d in agg.items():
        if name == "oracle_up_ref":
            continue
        fams[_family(name)].append((d["cost"], d[f"{metric}@rho{rho}"], d["rank"]))
    for fam in sorted(fams):
        pts = sorted(fams[fam])
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        st = _STYLE.get(fam, dict(marker="x"))
        ax.plot(x, y, label=fam, ms=5, lw=1.4, **st)

    o = agg.get("oracle_up_ref")
    if o is not None:
        ax.plot([o["cost"]], [o[f"{metric}@rho{rho}"]], marker="*", ms=16,
                color="black", ls="none", label="oracle_up (full up_proj)")
    if metric == "recall":
        ax.axhline(rho, color="gray", ls=":", lw=1.2,
                   label=f"random (= rho = {rho})")
    ax.axhline(1.0, color="green", ls=":", lw=1.0, alpha=0.6,
               label="oracle_mag_noW (perfect)")
    ax.set_xscale("log")
    ax.set_xlabel("scorer cost  (fraction of one full-width matmul)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"budget rho = {rho}  (keep {rho:.0%} of K·I channels)")
    ax.grid(alpha=0.3)


# Measured HellaSwag acc_norm from Investigation 3 (all 6 configs, −75% nominal).
_MEASURED_ACC = {
    "svd_r16": 54.95, "svd_r32": 63.80, "btt_m2n2_r32": 66.83,
    "svd_r16_uponly": 60.71, "svd_r32_uponly": 63.94,
    "btt_m2n2_r32_uponly": 65.97,
}


def plot_recall_vs_accuracy(agg, fig_dir, measured=None, rho=0.25):
    """Measured HellaSwag acc_norm against the recall/mass the diagnostic predicted.

    The point of this figure: recall is only a *partial* proxy for accuracy. The
    BTT points sit **above** the SVD trend despite scoring lower recall, which is
    why the equal-cost accuracy control matters and why Investigation 1's
    "SVD wins" verdict could not be acted on directly.
    """
    measured = _MEASURED_ACC if measured is None else measured
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, metric in zip(axes, ["recall", "mass"]):
        pts = {"svd": [], "btt": []}
        for name, acc in measured.items():
            if name not in agg:
                continue
            x = agg[name][f"{metric}@rho{rho}"]
            pts["btt" if name.startswith("btt") else "svd"].append((x, acc))
            ax.annotate(name.replace("btt_m2n2_", "btt ").replace("_uponly", " (up)"),
                        (x, acc), textcoords="offset points", xytext=(6, -4),
                        fontsize=7, color="dimgray")
        ax.plot(*zip(*sorted(pts["svd"])), "o", ms=8, color="#1f77b4",
                ls="-", lw=1.0, alpha=0.8, label="SVD (1x1)")
        ax.plot(*zip(*sorted(pts["btt"])), "s", ms=9, color="#d62728",
                ls="-", lw=1.0, alpha=0.8, label="BTT 2x2")
        if "oracle_up_ref" in agg:
            ax.plot([agg["oracle_up_ref"][f"{metric}@rho{rho}"]], [75.31],
                    "*", ms=16, color="black", label="oracle_up (75.31)")
        ax.axhline(78.36, color="green", ls=":", lw=1.2, label="oracle_mag_noW (78.36)")
        ax.axhline(63.60, color="orange", ls="--", lw=1.2, label="Level-1 pivchol (63.60)")
        ax.axhline(49.4, color="brown", ls="-.", lw=1.0, label="reduce-top-k 8→2 (49.4)")
        ax.set_xlabel(f"{metric}@rho={rho}  (predicted by the cheap diagnostic)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("HellaSwag acc_norm (measured)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle("Does the cheap diagnostic predict accuracy?  "
                 "BTT beats SVD on accuracy despite *lower* recall", fontsize=11)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    p = os.path.join(fig_dir, "recall_vs_accuracy.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {p}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", default=[
        os.path.join(_REPO, "docs/results/btt_dynamic/recall.json"),
        os.path.join(_REPO, "docs/results/btt_dynamic/recall_hirank.json"),
    ])
    ap.add_argument("--fig-dir",
                    default=os.path.join(_REPO, "docs/exps/dynamic_active_param/figures/btt_dynamic"))
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    rows, ratios = _load([p for p in args.json if os.path.exists(p)])
    agg = _agg(rows)
    print(f"[plot] {len(rows)} rows, {len(agg)} variants, ratios={ratios}")

    # ---- 1/2: recall & mass vs cost, one panel per budget -------------------
    for metric, ylab, fname in [
        ("recall", "top-B recall vs oracle_mag_noW", "recall_vs_cost.png"),
        ("mass", "captured oracle score mass (/ oracle's own)", "mass_vs_cost.png"),
    ]:
        fig, axes = plt.subplots(1, len(ratios), figsize=(5.2 * len(ratios), 4.4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, rho in zip(axes, ratios):
            _curve_plot(ax, agg, metric, rho, ylab if ax is axes[0] else "")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
        fig.suptitle(
            f"Cheap low-rank scorers: {ylab}\n"
            "Qwen3-30B-A3B, 4 MoE layers x 8192 C4 tokens (layer-averaged)",
            fontsize=11)
        fig.tight_layout(rect=(0, 0.11, 1, 0.94))
        p = os.path.join(args.fig_dir, fname)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {p}")

    # ---- 3: SVD vs BTT at equal cost ---------------------------------------
    rho = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for fam in ["SVD (1x1) · up+gate", "BTT 2x2 · up+gate", "BTT 4x2 · up+gate"]:
        pts = sorted((d["cost"], d[f"recall@rho{rho}"], d["rank"])
                     for n, d in agg.items() if _family(n) == fam)
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ms=6, lw=1.5,
                label=fam, **_STYLE.get(fam, {}))
        for c, r, rk in pts:
            ax.annotate(f"r{rk}", (c, r), textcoords="offset points",
                        xytext=(0, -12), fontsize=7, ha="center", color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("scorer cost (fraction of one full-width matmul)")
    ax.set_ylabel(f"top-B recall @ rho={rho}")
    ax.set_title("Does the block grid buy anything over a plain SVD?\n"
                 "(equal cost => compare vertically)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(args.fig_dir, "svd_vs_btt.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {p}")

    # ---- 4: per-layer spread ----------------------------------------------
    picks = [n for n in ["svd_r16", "svd_r32", "svd_r128", "btt_m2n2_r32",
                         "svd_r32_uponly", "oracle_up_ref"] if n in agg]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for n in picks:
        per = agg[n][f"recall@rho{rho}__per_layer"]
        ls = sorted(per)
        ax.plot(ls, [per[l] for l in ls], marker="o", ms=5, lw=1.4,
                label=f"{n} (cost {agg[n]['cost']:.3f})")
    ax.axhline(rho, color="gray", ls=":", label=f"random (={rho})")
    ax.set_xlabel("layer index")
    ax.set_ylabel(f"top-B recall @ rho={rho}")
    ax.set_title("Recall by depth — is any layer easier to proxy?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(args.fig_dir, "by_depth.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {p}")

    # ---- 5: measured accuracy vs predicted recall/mass ---------------------
    plot_recall_vs_accuracy(agg, args.fig_dir)

    # ---- summary table to stdout (markdown, for the doc) -------------------
    print("\n| variant | cost | spearman | " +
          " | ".join(f"recall@{r} | mass@{r}" for r in ratios) + " |")
    print("|---|---|---|" + "---|" * (2 * len(ratios)))
    for n in sorted(agg, key=lambda k: agg[k]["cost"]):
        d = agg[n]
        cells = " | ".join(f"{d[f'recall@rho{r}']:.3f} | {d[f'mass@rho{r}']:.3f}"
                           for r in ratios)
        print(f"| `{n}` | {d['cost']:.4f} | {d['spearman']:.3f} | {cells} |")


if __name__ == "__main__":
    main()
