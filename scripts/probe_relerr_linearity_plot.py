#!/usr/bin/env python
"""Plot per-layer ``rel_err`` vs measured HellaSwag accuracy (the ladder's basis).

Two panels:
  left  — per-layer scatter + OLS line, restricted to the fixed-budget family
          (rho=0.125, selector varying), which is the regime every scorer
          comparison lives in. One colour per captured layer.
  right — the layer-averaged fit, both families overlaid, showing that they are
          two different rulers (a mis-selection at fixed budget costs ~3.8x more
          accuracy per unit rel_err than an honestly smaller budget).

Consumes ``docs/results/idea_pilot/relerr_linearity.json`` from
``probe_relerr_linearity.py``. No GPU.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

SRC = os.path.join(_REPO, "docs/results/idea_pilot/relerr_linearity.json")
OUT_DIR = os.path.join(_REPO, "docs/exps/dynamic_active_param/figures/btt_dynamic")


def main():
    d = json.load(open(SRC))
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.4))

    # ---- left: per-layer, fixed-budget family ----------------------------
    layers = sorted(d["layers"], key=int)
    cmap = plt.get_cmap("viridis")
    for i, lname in enumerate(layers):
        rec = d["layers"][lname].get("fixed_budget")
        if not rec:
            continue
        x = np.array([p["rel_err"] for p in rec["points"]])
        y = np.array([p["acc"] for p in rec["points"]])
        c = cmap(i / max(len(layers) - 1, 1))
        ax0.scatter(x, y, color=c, s=34, zorder=3,
                    label=f"L{lname}  slope {rec['slope']:.1f}, R²={rec['r2']:.2f}")
        xs = np.linspace(x.min(), x.max(), 50)
        ax0.plot(xs, rec["slope"] * xs + rec["intercept"], color=c, lw=1.4, alpha=.85)
    ax0.set_xlabel("layer rel_err  ‖y_full − y_kept‖ / ‖y_full‖")
    ax0.set_ylabel("HellaSwag acc_norm (measured, end-to-end)")
    ax0.set_title("Per-layer rel_err vs final accuracy\n"
                  "(fixed budget ρ=0.125, selector varying)", fontsize=10)
    ax0.legend(fontsize=7, loc="lower left")
    ax0.grid(alpha=.3)

    # ---- right: layer-averaged, both families ----------------------------
    for fam, colour, marker in (("fixed_rho0.125", "tab:red", "o"),
                                ("budget_ladder", "tab:blue", "s")):
        rec = d.get("families", {}).get(fam)
        if not rec:
            continue
        x = np.array([p["rel_err"] for p in rec["points"]])
        y = np.array([p["acc"] for p in rec["points"]])
        ax1.scatter(x, y, color=colour, marker=marker, s=46, zorder=3,
                    label=f"{fam}\n  slope {rec['slope']:.1f} pt/unit, "
                          f"R²={rec['r2']:.3f}")
        xs = np.linspace(0, max(x.max(), 0.6), 50)
        ax1.plot(xs, rec["slope"] * xs + rec["intercept"], color=colour,
                 lw=1.6, alpha=.85)
        for p in rec["points"]:
            ax1.annotate(p["label"].replace("@0.125", "").replace("probe_", ""),
                         (p["rel_err"], p["acc"]), fontsize=6,
                         xytext=(3, -8), textcoords="offset points", color=colour)
    # held-out reuse-probe rows: predicted from the fit BEFORE being measured
    for i, h in enumerate(d.get("heldout", [])):
        ax1.scatter([h["rel_err"]], [h["measured"]], marker="D", s=54,
                    facecolor="none", edgecolor="darkgreen", lw=1.8, zorder=6,
                    label="held-out (pre-registered)" if i == 0 else None)
        ax1.plot([h["rel_err"]] * 2, [h["predicted"], h["measured"]],
                 c="darkgreen", lw=1.2, alpha=.7)
    if d.get("heldout"):
        ax1.annotate(f"held-out MAE {d['heldout_mae']:.2f}pt\n"
                     f"(measurement stderr ≈0.43pt)",
                     (0.97, 0.93), xycoords="axes fraction", ha="right",
                     va="top", fontsize=7.5, color="darkgreen")
    ax1.axhline(78.56, ls=":", c="k", lw=1, label="dense 78.56")
    ax1.set_xlabel("layer-averaged rel_err")
    ax1.set_ylabel("HellaSwag acc_norm")
    ax1.set_title("The two families are different rulers\n"
                  "(mis-selection costs ~3.8× more per unit rel_err)", fontsize=10)
    ax1.legend(fontsize=7, loc="lower left")
    ax1.grid(alpha=.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"relerr_linearity.{ext}")
        fig.savefig(p, dpi=170, bbox_inches="tight")
        print(f"[done] wrote {p}")


if __name__ == "__main__":
    main()
