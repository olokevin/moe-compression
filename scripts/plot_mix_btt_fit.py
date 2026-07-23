#!/usr/bin/env python
"""Plot per-layer fitting loss curves for the mix_btt experiment.

Parses the six fit logs under run_results/A100-New/run_logs/ (3 fit cells ×
{gate_proj, down_proj}) and produces loss-curve figures under
docs/results/mix_btt/figs/.

Two log formats are handled:
  * activation-space cells (act_nl, act_nl_mix): lines from mix_btt.fit_mix_btt_layer
      "[mixbtt-fit <layer>] step <s>: full_mse=<..> rel=<..>"   (every 200 steps, 3000 iters)
  * weight-space cell (ws): lines from moe_basis.fit.fit_layer_basis
      "[fit <layer>] step <s>: mse=<..> rel_err=<..>"           (every 1000 steps, 30000 iters)
"""
import re
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, "run_results", "A100-New", "run_logs")
OUTDIR = os.path.join(ROOT, "docs", "results", "mix_btt", "figs")
os.makedirs(OUTDIR, exist_ok=True)

# (cell label, projection, log filename)
RUNS = [
    ("ws",         "gate_proj", "mixbtt_ws_0722-015614.log"),
    ("act_nl",     "gate_proj", "mixbtt_act_nl_0722-030841.log"),
    ("act_nl_mix", "gate_proj", "mixbtt_act_nl_mix_0722-030848.log"),
    ("ws",         "down_proj", "mixbtt_down_ws_0722-053945.log"),
    ("act_nl",     "down_proj", "mixbtt_down_act_nl_0722-053952.log"),
    ("act_nl_mix", "down_proj", "mixbtt_down_act_nl_mix_0722-053959.log"),
]

CELL_TITLE = {
    "ws":         "weight-space (MoBE, lr=0.07/30k)",
    "act_nl":     "act-space nl-only (fixed I, lr=1e-3/3k)",
    "act_nl_mix": "act-space nl+mix (learn α, lr=1e-3/3k)",
}

# Match both the activation-space and weight-space log lines.
PAT = re.compile(
    r"\[(?:mixbtt-fit|fit) model\.layers\.(\d+)\.mlp\.(\w+)\] step (\d+): "
    r"(?:full_mse|mse)=([0-9.eE+-]+) rel(?:_err)?=([0-9.eE+-]+)"
)


def parse(path):
    """Return {layer_idx: (steps[], rel[], mse[])}."""
    curves = defaultdict(lambda: ([], [], []))
    with open(path) as fh:
        for line in fh:
            m = PAT.search(line)
            if not m:
                continue
            layer = int(m.group(1))
            step = int(m.group(3))
            mse = float(m.group(4))
            rel = float(m.group(5))
            s, r, ms = curves[layer]
            s.append(step)
            r.append(rel)
            ms.append(mse)
    return curves


def plot_cell(ax, curves, title):
    layers = sorted(curves)
    cmap = plt.get_cmap("viridis")
    for li in layers:
        steps, rel, _ = curves[li]
        ax.plot(steps, rel, color=cmap(li / max(layers)), lw=0.8, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("fit step")
    ax.set_ylabel("reconstruction rel-err  ‖Wx−Ŵx‖/‖Wx‖")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=max(layers)))
    return sm


def main():
    data = {}
    for cell, proj, fname in RUNS:
        path = os.path.join(LOGDIR, fname)
        if not os.path.exists(path):
            print(f"MISSING {path}")
            continue
        data[(cell, proj)] = parse(path)
        n = len(data[(cell, proj)])
        print(f"parsed {cell:11s} {proj}: {n} layers")

    cells = ["ws", "act_nl", "act_nl_mix"]

    # --- Figure 1 & 2: per-projection 3-panel grid of all-layer curves ---
    for proj in ("gate_proj", "down_proj"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=False)
        sm = None
        for ax, cell in zip(axes, cells):
            curves = data.get((cell, proj))
            if not curves:
                ax.set_visible(False)
                continue
            sm = plot_cell(ax, curves, CELL_TITLE[cell])
        fig.suptitle(f"mix_btt fit loss curves — {proj} (Qwen3-8B, −33%, one line per layer)",
                     fontsize=12)
        if sm is not None:
            cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
            cbar.set_label("layer index (0=shallow → 35=deep)")
        out = os.path.join(OUTDIR, f"fit_curves_{proj}.png")
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)

    # --- Figure 3: final rel-err per layer, all cells/projections overlaid ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    style = {"ws": ("weight-space", "o", "#d62728"),
             "act_nl": ("act nl-only", "s", "#1f77b4"),
             "act_nl_mix": ("act nl+mix", "^", "#2ca02c")}
    for ax, proj in zip(axes, ("gate_proj", "down_proj")):
        for cell in cells:
            curves = data.get((cell, proj))
            if not curves:
                continue
            layers = sorted(curves)
            finals = [curves[li][1][-1] for li in layers]  # last rel per layer
            lbl, mk, col = style[cell]
            ax.plot(layers, finals, marker=mk, ms=3, lw=1.0, color=col, label=lbl)
        ax.set_yscale("log")
        ax.set_xlabel("layer index")
        ax.set_ylabel("final reconstruction rel-err")
        ax.set_title(proj, fontsize=11)
        ax.grid(True, which="both", ls=":", alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("mix_btt — final fit rel-err by layer (lower = better reconstruction)",
                 fontsize=12)
    out = os.path.join(OUTDIR, "final_rel_by_layer.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    # --- print a quick summary table ---
    print("\nmean / median final rel-err across 36 layers:")
    for proj in ("gate_proj", "down_proj"):
        for cell in cells:
            curves = data.get((cell, proj))
            if not curves:
                continue
            finals = np.array([curves[li][1][-1] for li in sorted(curves)])
            print(f"  {proj:10s} {cell:11s}  mean={finals.mean():.4f}  "
                  f"median={np.median(finals):.4f}  max={finals.max():.4f}")


if __name__ == "__main__":
    main()
