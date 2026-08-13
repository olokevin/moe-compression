#!/usr/bin/env python
"""Plot the all-layer ``rel_err(p, rho)`` surface and the allocation it implies.

Three panels:
  left   — rel_err across depth at a fixed ``(p, rho)``: how unequal the layers
           are. This is the entire justification for a non-uniform schedule.
  middle — the trade-off curve, mean rel_err vs mean used-param fraction, for every
           uniform grid point plus the per-layer optima. The optima sit *below*
           the uniform frontier, and by how much is the achievable gain.
  right   — the solved schedule at the -75% target: which ``(p, rho)`` each layer
           gets, over depth.

Consumes ``docs/results/idea_pilot/layer_surface.json``. No GPU.
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

from src.dynamic_active_param.sparse_probe import used_param_fraction

SRC = os.environ.get("SURFACE_JSON", os.path.join(
    _REPO, "docs/results/idea_pilot/layer_surface_8k_solved.json"))
OUT_DIR = os.path.join(_REPO, "docs/exps/dynamic_active_param/figures/btt_dynamic")
SLOPE = 26.4          # HellaSwag pt per unit rel_err, fixed-budget family


def main():
    d = json.load(open(SRC))
    S = np.array(d["surface"])                     # (L, n_p, n_rho)
    ps, rhos, layers = d["ps"], d["rhos"], d["layers"]
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # ---- left: spread across depth ---------------------------------------
    cmap = plt.get_cmap("plasma")
    for j, r in enumerate(rhos):
        pi = ps.index(0.25) if 0.25 in ps else len(ps) // 2
        ax0.plot(layers, S[:, pi, j], lw=1.3, color=cmap(j / max(len(rhos) - 1, 1)),
                 label=f"ρ={r:.4g}")
    ax0.set_xlabel("MoE layer index")
    ax0.set_ylabel("rel_err")
    v = S[:, ps.index(0.25) if 0.25 in ps else len(ps) // 2, rhos.index(0.125)]
    ax0.set_title(f"Layers are not interchangeable (p=0.25)\n"
                  f"at ρ=0.125: min {v.min():.3f}, max {v.max():.3f} "
                  f"({v.max()/v.min():.2f}×)", fontsize=10)
    ax0.legend(fontsize=7, ncol=2)
    ax0.grid(alpha=.3)

    # ---- middle: uniform frontier vs per-layer optima --------------------
    cost = np.array([[used_param_fraction(p, r) for r in rhos] for p in ps])
    for i, p in enumerate(ps):
        ax1.plot(cost[i], S[:, i, :].mean(axis=0), "o-", ms=3.5, lw=1,
                 alpha=.75, label=f"uniform p={p:g}")
    xs, ys, lbl = [], [], []
    for key, sol in sorted(d["solutions"].items()):
        xs.append(sol["achieved_mean_kept"])
        ys.append(sol["mean_rel_err"])
        lbl.append(f"{100*(1-sol['target_kept']):.0f}% cut")
    ax1.scatter(xs, ys, marker="*", s=230, color="crimson", zorder=5,
                label="per-layer optimum")
    for x, y, t in zip(xs, ys, lbl):
        ax1.annotate(t, (x, y), fontsize=7, xytext=(5, 6),
                     textcoords="offset points", color="crimson")
    ax1.set_xlabel("mean used-param fraction   ρ_channel + 2·ρ_input/3  (scoring + compute)")
    ax1.set_ylabel("mean rel_err over 48 layers")
    wt = d.get("slope_weighted")
    ax1.set_title("Per-layer allocation vs the uniform frontier\n"
                  + ("(stars: slope-weighted solve, scored on unweighted err)" if wt
                     else "(stars: UNWEIGHTED solve — measured worse than uniform)"),
                  fontsize=10)
    ax1.legend(fontsize=6.5, ncol=2)
    ax1.grid(alpha=.3)

    # ---- right: the solved schedule at the tightest target ---------------
    key = sorted(d["solutions"])[0]
    sol = d["solutions"][key]
    sched = sorted(sol["schedule"], key=lambda e: e["layer"])
    lay = [e["layer"] for e in sched]
    ax2.step(lay, [e["p"] for e in sched], where="mid", lw=1.6,
             color="tab:blue", label="input read p")
    ax2.step(lay, [e["rho"] for e in sched], where="mid", lw=1.6,
             color="tab:orange", label="channel keep ρ")
    ax2.axhline(sol["uniform_best"]["p"], ls=":", c="tab:blue", lw=1,
                label=f"best uniform p={sol['uniform_best']['p']:g}")
    ax2.axhline(sol["uniform_best"]["rho"], ls=":", c="tab:orange", lw=1,
                label=f"best uniform ρ={sol['uniform_best']['rho']:g}")
    gain = sol["gain_rel_err"]
    ax2.set_xlabel("MoE layer index")
    ax2.set_ylabel("fraction")
    wtag = ("slope-weighted" if d.get("slope_weighted") else "unweighted")
    ax2.set_title(f"Solved schedule @ {100*(1-sol['target_kept']):.1f}% used-param cut\n"
                  f"({wtag}; predicted {SLOPE*gain:+.2f} pt on this objective)",
                  fontsize=10)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"layer_surface.{ext}")
        fig.savefig(p, dpi=170, bbox_inches="tight")
        print(f"[done] wrote {p}")


if __name__ == "__main__":
    main()
