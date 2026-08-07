#!/usr/bin/env python
"""Plot the two thesis figures for slides 6-8 from the single-expert capture.

Consumes ``docs/results/presentation/expert_activation.npz`` (from
``scripts/expert_activation_capture.py``). Thesis:

    "A token doesn't need 8 whole experts — it needs a sparse, token-specific
     subset of channels across those experts."

Two single-panel figures:

  * ``fig_sparse_suffices.pdf`` (slide 7) — the Fig-4 analog on Qwen3-30B-A3B.
    Linear-y histogram of one expert's SwiGLU activation magnitude, with the
    bottom-``rho`` (by |h|) shaded as "deactivated" vs the surviving tail.
    Establishes: a small fraction of neurons carries essentially all the output.

  * ``fig_token_specific.pdf`` (slide 8) — the extension the reference lacks.
    For one expert, per-neuron keep-frequency (as a 0%–100% ratio, sorted desc):
    each token keeps its own top-``budget`` channels, and we plot how often each
    neuron lands in that kept set. Almost nothing is always-on or always-off; the
    curve clusters at the budget ρ. Establishes: which channels matter is decided
    per token, so no fixed within-expert keep-set works.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

INK = "#1c2330"
MUTED = "#68717f"
BLUE = "#2f6fdb"
AMBER = "#e08a1e"
PURPLE = "#7a4fc4"
GREEN = "#2e8b6f"
RED = "#c8402f"
GREY = "#9aa4b2"
COLD = "#c7d0dc"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": "#cfd6e0", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 9.2, "axes.labelsize": 8.4,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "legend.fontsize": 7.4, "figure.titlesize": 10.5,
    "savefig.bbox": "tight", "axes.grid": True,
    "grid.color": "#eef1f6", "grid.linewidth": 0.7,
})


def _clean(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)


def _primary_target(d):
    """The (layer, expert) used for the single-expert panels: prefer layer 0."""
    tg = d["targets"]
    for li, e in tg:
        if li == 0:
            return int(li), int(e)
    return int(tg[0, 0]), int(tg[0, 1])


def fig_sparse_suffices(d, out_dir, stats, rho=0.95):
    """Slide 7 — the Fig-4 analog: sparse activation suffices, one expert.

    Single panel: the magnitude histogram of one expert's SwiGLU output, with the
    bottom-``rho`` (by |h|) shaded as "deactivated" against the surviving tail.
    Linear y-axis (plain count).
    """
    li, e = _primary_target(d)
    H = d[f"h_L{li}_e{e}"].astype(np.float32)         # (T, I) SwiGLU output
    T, I = H.shape
    flat = H.reshape(-1)
    mag = np.abs(flat)
    # global threshold that deactivates the bottom `rho` of activations by |h|
    thr = np.quantile(mag, rho)
    stats[f"L{li}e{e}_T"] = int(T)
    stats[f"L{li}e{e}_thr_r{rho}"] = float(thr)
    near0 = float(np.mean(mag < 0.003))
    stats[f"L{li}e{e}_frac_near_zero"] = near0

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.6))

    # ---- magnitude histogram, linear-y, split at the sparsity threshold ----
    lo = float(np.quantile(flat, 0.001)); hi = float(np.quantile(flat, 0.999))
    bins = np.linspace(min(lo, -hi), max(hi, -lo), 121)
    counts, edges = np.histogram(flat, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kept = np.abs(centers) >= thr
    ax.bar(centers[~kept], counts[~kept], width=np.diff(edges)[0],
           color=BLUE, alpha=0.9, label=f"deactivated (bottom {rho * 100:.0f}%)")
    ax.bar(centers[kept], counts[kept], width=np.diff(edges)[0],
           color=AMBER, alpha=0.95, label="kept (surviving tail)")
    ax.axvline(thr, color=INK, lw=0.9, ls=":")
    ax.axvline(-thr, color=INK, lw=0.9, ls=":")
    ax.set_xlabel("SwiGLU activation output  $h_j = \\mathrm{SiLU}(gate_j\\!\\cdot\\! x)\\,(up_j\\!\\cdot\\! x)$")
    ax.set_ylabel("count")
    ax.text(0.03, 0.94,
            f"{near0 * 100:.0f}% of activations are ~0\n"
            f"(|h| < 0.003)", transform=ax.transAxes, fontsize=7.2,
            color=INK, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f4f6fa",
                      edgecolor="#d9dfe8", lw=0.6))
    ax.legend(frameon=False, loc="upper right")
    _clean(ax)

    ax.set_title(
        f"One expert's activations are long-tailed — a few neurons carry it\n"
        f"expert {e}, layer {li}  ·  Qwen3-30B-A3B, WikiText-2, {T:,} tokens",
        loc="left", color=INK, weight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_sparse_suffices.{ext}"), dpi=400)
    plt.close(fig)


def fig_token_specific(d, out_dir, stats, budget_frac=0.25):
    """Slide 8 — within one expert, no fixed keep-set works.

    Single panel: per-neuron keep-frequency for **one** expert. Each token routed
    to this expert keeps its own top-``budget_frac`` channels (here 25%); we plot,
    for each of the expert's ``I`` neurons, the fraction of those tokens that keep
    it — as a ratio 0%–100% on the y-axis, neurons sorted by that frequency. If a
    fixed within-expert keep-set worked, a ``budget_frac`` slice would sit at
    ~100% and the rest at ~0%; instead the curve is smooth and clusters at the
    budget line ρ, so which channels fire is decided per token.
    """
    li, e = _primary_target(d)
    H = np.abs(d[f"h_L{li}_e{e}"].astype(np.float32))  # (T, I) |SwiGLU|
    cn = d[f"colnorm_L{li}_e{e}"].astype(np.float32)   # (I,) ||W_down[:,j]||
    score = H * cn[None, :]                             # full oracle_mag score, g cancels
    T, I = score.shape
    B = max(1, int(round(budget_frac * I)))
    # per-token top-B keep mask (within this expert)
    order = np.argsort(-score, axis=1)
    keep = np.zeros_like(score, dtype=bool)
    rows = np.arange(T)[:, None]
    keep[rows, order[:, :B]] = True

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.6))

    freq = np.sort(keep.mean(0))[::-1] * 100.0         # (I,) %, sorted desc
    x = np.arange(I)
    ax.fill_between(x, 0, freq, color=PURPLE, alpha=0.28, step="mid")
    ax.plot(x, freq, color=PURPLE, lw=1.8)
    ax.axhline(budget_frac * 100, color=RED, lw=1.4, ls="--")
    ax.text(I * 0.985, budget_frac * 100 + 3.0,
            f"per-token budget  $\\rho$ = {budget_frac * 100:.0f}%",
            fontsize=7.4, color=RED, ha="right", va="bottom")

    freq01 = keep.mean(0)
    always = float(np.mean(freq01 > 0.95)); never = float(np.mean(freq01 < 0.05))
    stats[f"L{li}e{e}_freq_always_on"] = always
    stats[f"L{li}e{e}_freq_never"] = never
    stats[f"L{li}e{e}_budget_frac"] = float(budget_frac)
    ax.text(0.97, 0.94,
            f"{always * 100:.1f}% of neurons kept >95% of the time\n"
            f"{never * 100:.0f}% kept <5% of the time\n"
            f"the rest cluster at the budget $\\rho$",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.0, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f4f6fa",
                      edgecolor="#d9dfe8", lw=0.6))
    ax.set_xlabel("neuron of this expert, sorted by keep-frequency")
    ax.set_ylabel("% of the expert's tokens that keep the neuron")
    ax.set_xlim(0, I); ax.set_ylim(0, 100)
    _clean(ax)

    ax.set_title(
        f"No channel is always on or always off — kept set is token-specific\n"
        f"expert {e}, layer {li}  ·  Qwen3-30B-A3B  ·  per-token budget "
        f"ρ = {budget_frac * 100:.0f}%",
        loc="left", color=INK, weight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_token_specific.{ext}"), dpi=400)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--npz", default=os.path.join(
        repo, "docs/results/presentation/expert_activation.npz"))
    ap.add_argument("--out-dir", default=os.path.join(repo, "docs/presentation/figs"))
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--budget", type=float, default=0.25)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(args.npz)
    stats = {}
    fig_sparse_suffices(d, args.out_dir, stats, rho=args.rho)
    fig_token_specific(d, args.out_dir, stats, budget_frac=args.budget)

    sp = os.path.join(args.out_dir, "stats_activation.json")
    with open(sp, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(f"[plot] wrote figures + {sp}")
    for k, v in sorted(stats.items()):
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
