#!/usr/bin/env python
"""Plot oracle_mag activation-frequency + per-token score profiles.

Consumes the ``.npz`` produced by ``scripts/oracle_mag_freq_capture.py`` and
writes the figures for the two investigations, plus a small stats JSON, into
``docs/exps/dynamic_active_param/figures/oracle_mag/``.

Investigation 1 — activation frequency of each intermediate channel:
  * ``freq_grid_r<rho>.png``   — per-layer (expert × channel) heatmap grid.
  * ``freq_layer_sorted.png``  — layer × sorted-channel-frequency (sparsity by depth).
  * ``freq_hot_by_depth.png``  — hot-channel fraction / mean freq vs depth.
  * a few full-resolution single-layer heatmaps.

Investigation 2 — per-token magnitude concentration:
  * ``token_sorted_curves.png``  — sampled per-token sorted-score curves (shallow/
    mid/deep layers), showing heterogeneity across tokens + the mean curve.
  * ``token_concentration.png``  — participation-ratio + top-B-mass distributions
    by depth (how spread vs peaked each token's magnitude is, and its trend).
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _depth_frac(layer_indices):
    li = layer_indices.astype(np.float64)
    if li.max() > li.min():
        return (li - li.min()) / (li.max() - li.min())
    return np.linspace(0, 1, len(li))


def plot_freq_grid(d, rho_idx, out_dir):
    freq = d["freq"][rho_idx]                     # (L, E, I)
    L, E, I = freq.shape
    rho = float(d["ratios"][rho_idx])
    layer_idx = d["layer_indices"]

    ncol = 6
    nrow = int(np.ceil(L / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.3, nrow * 1.9))
    axes = np.atleast_1d(axes).ravel()
    im = None
    for pos in range(L):
        ax = axes[pos]
        im = ax.imshow(freq[pos], aspect="auto", cmap="magma", vmin=0.0, vmax=1.0,
                       interpolation="nearest")
        ax.set_title(f"L{int(layer_idx[pos])}", fontsize=6, pad=1)
        ax.set_xticks([]); ax.set_yticks([])
    for pos in range(L, len(axes)):
        axes[pos].axis("off")
    fig.suptitle(
        f"oracle_mag channel keep-frequency (kept | routed), ρ={rho:.3f}  "
        f"— rows=experts (E={E}), cols=channels (I={I})", fontsize=10)
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.01)
    cbar.set_label("keep frequency", fontsize=8)
    p = os.path.join(out_dir, f"freq_grid_r{rho:.3f}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_single_layers(d, rho_idx, out_dir, layers_pos):
    freq = d["freq"][rho_idx]
    rho = float(d["ratios"][rho_idx])
    layer_idx = d["layer_indices"]
    paths = []
    for pos in layers_pos:
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(freq[pos], aspect="auto", cmap="magma", vmin=0.0, vmax=1.0,
                       interpolation="nearest")
        ax.set_xlabel("intermediate channel j"); ax.set_ylabel("expert e")
        ax.set_title(f"oracle_mag keep-frequency — layer {int(layer_idx[pos])}, ρ={rho:.3f}")
        fig.colorbar(im, ax=ax, label="keep frequency (kept | routed)")
        p = os.path.join(out_dir, f"freq_layer{int(layer_idx[pos])}_r{rho:.3f}.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def plot_freq_layer_sorted(d, rho_idx, out_dir):
    """Layer × sorted per-channel frequency: each row is a layer's E*I channel
    frequencies sorted descending → shows what fraction of channels are ever hot."""
    freq = d["freq"][rho_idx]                     # (L,E,I)
    L = freq.shape[0]
    rho = float(d["ratios"][rho_idx])
    flat = freq.reshape(L, -1).astype(np.float32)
    flat_sorted = -np.sort(-flat, axis=1)         # descending per layer
    # downsample channel axis for display
    KI = flat_sorted.shape[1]
    ncol = 512
    idx = np.linspace(0, KI - 1, ncol).astype(int)
    disp = flat_sorted[:, idx]
    depth = d["layer_indices"]
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(disp, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   extent=[0, 100, L - 0.5, -0.5])
    ax.set_xlabel("channel percentile (sorted by keep-frequency, descending)")
    ax.set_ylabel("MoE layer position (shallow → deep)")
    ax.set_yticks(np.arange(0, L, 4))
    ax.set_yticklabels([str(int(depth[p])) for p in range(0, L, 4)])
    ax.set_title(f"Per-channel keep-frequency sorted within layer, ρ={rho:.3f}")
    fig.colorbar(im, ax=ax, label="keep frequency")
    p = os.path.join(out_dir, f"freq_layer_sorted_r{rho:.3f}.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_hot_by_depth(d, out_dir, hot_thresh=0.5):
    """Fraction of channels that are 'hot' (freq>thresh) and mean freq vs depth,
    for every budget."""
    ratios = d["ratios"]
    depth = d["layer_indices"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for bi, rho in enumerate(ratios):
        freq = d["freq"][bi].reshape(d["freq"].shape[1], -1).astype(np.float32)
        hot_frac = (freq > hot_thresh).mean(axis=1)
        mean_freq = freq.mean(axis=1)
        ax1.plot(depth, hot_frac, marker="o", ms=2.5, label=f"ρ={float(rho):.3f}")
        ax2.plot(depth, mean_freq, marker="o", ms=2.5, label=f"ρ={float(rho):.3f}")
    ax1.set_xlabel("layer index"); ax1.set_ylabel(f"fraction of channels with freq > {hot_thresh}")
    ax1.set_title("Hot-channel fraction vs depth"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax2.set_xlabel("layer index"); ax2.set_ylabel("mean keep-frequency")
    ax2.set_title("Mean channel keep-frequency vs depth"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    p = os.path.join(out_dir, "freq_hot_by_depth.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_token_curves(d, out_dir, layers_pos, n_show=60):
    """Sampled per-token sorted-score curves (normalized by per-token sum) for a
    few layers, overlaid with the mean curve — heterogeneity across tokens."""
    sc = d["sample_curves"].astype(np.float32)     # (L, S, KI)
    mean_curve = d["mean_curve"].astype(np.float32)
    layer_idx = d["layer_indices"]
    KI = sc.shape[2]
    xs = np.arange(1, KI + 1)
    fig, axes = plt.subplots(1, len(layers_pos), figsize=(5 * len(layers_pos), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, pos in zip(axes, layers_pos):
        curves = sc[pos]                            # (S, KI)
        valid = curves.sum(axis=1) > 0
        curves = curves[valid][:n_show]
        # normalize each token's curve by its own sum -> compare shape
        norm = curves / np.clip(curves.sum(axis=1, keepdims=True), 1e-30, None)
        for c in norm:
            ax.plot(xs, c, color="steelblue", alpha=0.10, lw=0.6)
        mc = mean_curve[pos] / max(1e-30, mean_curve[pos].sum())
        ax.plot(xs, mc, color="crimson", lw=1.6, label="mean over all tokens")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("channel rank (sorted desc)")
        ax.set_title(f"layer {int(layer_idx[pos])}")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("normalized oracle_mag score (÷ per-token sum)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Per-token sorted oracle_mag score profiles (each thin line = one token)",
                 fontsize=11)
    p = os.path.join(out_dir, "token_sorted_curves.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_concentration(d, out_dir):
    """Participation ratio (effective #channels) and top-B mass distributions by depth."""
    pr = d["pr"]                                    # (L, Ttot)
    mass = d["topB_mass"]                           # (nb, L, Ttot)
    depth = d["layer_indices"]
    ratios = d["ratios"]
    L = pr.shape[0]
    KI = int(d["KI"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # participation ratio: median + 10-90 band vs depth
    pr_med = np.median(pr, axis=1)
    pr_lo = np.percentile(pr, 10, axis=1)
    pr_hi = np.percentile(pr, 90, axis=1)
    ax1.plot(depth, pr_med, color="navy", marker="o", ms=3, label="median")
    ax1.fill_between(depth, pr_lo, pr_hi, color="navy", alpha=0.18, label="10–90 pct")
    ax1.axhline(KI, color="gray", ls="--", lw=1, label=f"K·I = {KI} (all)")
    ax1.set_xlabel("layer index")
    ax1.set_ylabel("participation ratio  (Σs)²/Σs²  = effective #channels")
    ax1.set_yscale("log")
    ax1.set_title("Per-token magnitude concentration vs depth")
    ax1.grid(alpha=0.3, which="both"); ax1.legend(fontsize=8)

    # top-B mass: median vs depth per budget
    for bi, rho in enumerate(ratios):
        m_med = np.median(mass[bi], axis=1)
        m_lo = np.percentile(mass[bi], 10, axis=1)
        m_hi = np.percentile(mass[bi], 90, axis=1)
        line, = ax2.plot(depth, m_med, marker="o", ms=3, label=f"top-B mass, ρ={float(rho):.3f}")
        ax2.fill_between(depth, m_lo, m_hi, color=line.get_color(), alpha=0.12)
    ax2.set_xlabel("layer index")
    ax2.set_ylabel("fraction of total per-token score mass kept in top-B")
    ax2.set_ylim(0, 1.01)
    ax2.set_title("Score mass captured by the kept budget vs depth")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    p = os.path.join(out_dir, "token_concentration.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs/results/level2/oracle_mag_freq.npz"))
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs/exps/dynamic_active_param/figures/oracle_mag"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    raw = np.load(args.npz, allow_pickle=True)
    d = {k: raw[k] for k in raw.files}
    L = int(d["L"])
    ratios = d["ratios"]
    # primary budget = the loosest (largest rho, first in default list); index by max
    rho_primary = int(np.argmax(ratios))
    layer_idx = d["layer_indices"]
    # representative layer positions: shallow / mid / deep
    reps = sorted(set([1, L // 2, L - 2]))

    made = []
    for bi in range(len(ratios)):
        made.append(plot_freq_grid(d, bi, args.out_dir))
        made.append(plot_freq_layer_sorted(d, bi, args.out_dir))
    made += plot_single_layers(d, rho_primary, args.out_dir, reps)
    made.append(plot_hot_by_depth(d, args.out_dir))
    made.append(plot_token_curves(d, args.out_dir, reps))
    made.append(plot_concentration(d, args.out_dir))

    # --- stats JSON ---------------------------------------------------------
    stats = {"n_tokens": int(d["n_tokens"]), "K": int(d["K"]), "I": int(d["I"]),
             "E": int(d["E"]), "L": L, "KI": int(d["KI"]),
             "ratios": [float(x) for x in ratios],
             "budgets": [int(x) for x in d["budgets"]],
             "by_budget": [], "by_depth": {}}
    for bi, rho in enumerate(ratios):
        freq = d["freq"][bi].reshape(L, -1).astype(np.float32)
        stats["by_budget"].append({
            "rho": float(rho),
            "mean_freq": float(freq.mean()),
            "frac_never_kept": float((freq <= 1e-6).mean()),
            "frac_hot_gt0.5": float((freq > 0.5).mean()),
            "frac_always_kept_gt0.95": float((freq > 0.95).mean()),
        })
    pr = d["pr"]
    stats["participation_ratio"] = {
        "overall_median": float(np.median(pr)),
        "shallow_median": float(np.median(pr[reps[0]])),
        "mid_median": float(np.median(pr[reps[1]])),
        "deep_median": float(np.median(pr[reps[-1]])),
        "KI": int(d["KI"]),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    made.append(os.path.join(args.out_dir, "stats.json"))

    print("[plot] wrote:")
    for p in made:
        print("  ", os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


if __name__ == "__main__":
    main()
