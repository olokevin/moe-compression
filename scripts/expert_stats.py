#!/usr/bin/env python
"""Compute and plot per-expert statistics for Qwen3-30B-A3B.

For the layers in ``--layers`` (default 0, 15, 31, 47) this produces four
figures and a JSON of summary stats:

  1. Ridge leverage score per expert, sorted descending
     (x = channel rank, y = leverage). One subplot per layer, all 128 experts
     overlaid. Read straight from ``expert_scores['leverage']``.

  2. Normalized spectral distribution  pi_i = sigma_i^2 / sum_j sigma_j^2,
     where sigma are the singular values of each expert's stacked weight
     matrix  W = [gate_proj || up_proj || down_proj^T]  of shape
     (I channels, 3*hidden). One subplot per layer, all 128 experts overlaid.

  3. Expert effective rank  erank = exp(-sum_i pi_i ln pi_i)  (spectral entropy
     of the same pi), x = expert index, one curve per layer.

  4. Expert contribution score, read straight from
     ``expert_scores['expert_out_token_contrib']``, x = expert index, one curve
     per layer.

Expert weights are read lazily from the HF safetensors shards (one layer at a
time) so the full 30B model is never materialized.
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import torch
from safetensors import safe_open


def find_snapshot(model_id: str) -> str:
    """Resolve the local HF snapshot dir for a model id."""
    if os.path.isdir(model_id) and glob.glob(os.path.join(model_id, "*.safetensors")):
        return model_id
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    folder = "models--" + model_id.replace("/", "--")
    snaps = sorted(glob.glob(os.path.join(cache, folder, "snapshots", "*")))
    if not snaps:
        raise FileNotFoundError(f"No snapshot found for {model_id} under {cache}")
    return snaps[-1]


def load_weight_map(snapshot: str) -> dict:
    """Map tensor name -> shard file. Handles single- or multi-shard."""
    index = os.path.join(snapshot, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            return json.load(f)["weight_map"]
    # single shard
    shard = glob.glob(os.path.join(snapshot, "*.safetensors"))[0]
    with safe_open(shard, framework="pt") as f:
        return {k: os.path.basename(shard) for k in f.keys()}


def get_expert_stack(snapshot, weight_map, layer, eid, device):
    """Return stacked expert weight W of shape (I, 3*hidden) on ``device``.

    W = [gate_proj || up_proj || down_proj^T]; every row is one intermediate
    channel, so rank(W) <= I and we get exactly I singular values.
    """
    base = f"model.layers.{layer}.mlp.experts.{eid}"
    parts = []
    for name, transpose in (("gate_proj", False), ("up_proj", False), ("down_proj", True)):
        key = f"{base}.{name}.weight"
        shard = os.path.join(snapshot, weight_map[key])
        with safe_open(shard, framework="pt", device="cpu") as f:
            w = f.get_tensor(key)  # gate/up: (I, hidden); down: (hidden, I)
        if transpose:
            w = w.t()  # (hidden, I) -> (I, hidden)
        parts.append(w)
    W = torch.cat(parts, dim=1).to(device=device, dtype=torch.float32)
    return W


def spectral_stats_for_layer(snapshot, weight_map, layer, num_experts, device):
    """Return (pi array [E, r], effective_rank array [E]) for one layer."""
    pis, eranks = [], []
    for eid in range(num_experts):
        W = get_expert_stack(snapshot, weight_map, layer, eid, device)
        # singular values via SVD (no U/V needed)
        sv = torch.linalg.svdvals(W)
        s2 = sv.pow(2)
        pi = (s2 / s2.sum().clamp_min(torch.finfo(s2.dtype).tiny)).cpu().numpy()
        pis.append(pi)
        # spectral entropy -> effective rank
        nz = pi[pi > 0]
        entropy = -np.sum(nz * np.log(nz))
        eranks.append(float(np.exp(entropy)))
        del W, sv, s2
    return np.stack(pis), np.array(eranks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    p.add_argument("--scores-dir", required=True,
                   help="dir containing expert_scores.pth")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="layer indices; default = all layers in the model")
    p.add_argument("--num-layers", type=int, default=48,
                   help="total layers, used when --layers is not given")
    p.add_argument("--out-dir", default="docs/results/stats/figures")
    p.add_argument("--stats-json", default="docs/results/stats/expert_stats_data.json")
    p.add_argument("--highlight-layers", type=int, nargs="+", default=[0, 15, 31, 47],
                   help="layers for the standalone 1xN subset figures")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.stats_json) or ".", exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    # ---- Load precomputed scores -----------------------------------------
    es = torch.load(os.path.join(args.scores_dir, "expert_scores.pth"),
                    map_location="cpu")
    leverage = es["leverage"]                      # {layer: (E, I)}
    contrib = es["expert_out_token_contrib"]       # {layer: (E,)}

    # Default to every MoE layer present in the score dict.
    if args.layers is not None:
        layers = args.layers
    else:
        layers = sorted(int(k) for k in leverage.keys())
    E, I = leverage[layers[0]].shape
    print(f"num_experts={E}, intermediate={I}, num_layers={len(layers)}, layers={layers}")

    # ---- Compute spectral stats from expert weights ----------------------
    # The SVD sweep is the expensive part (~13 min for 48 layers). Cache the
    # per-layer pi/erank arrays to an npz so plot tweaks don't recompute.
    cache_path = os.path.join(args.out_dir, "spectral_cache.npz")
    pi_by_layer, erank_by_layer = {}, {}
    cached = None
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if set(int(l) for l in cached["layers"]) >= set(layers):
            print(f"loaded spectral cache from {cache_path}")
            for L in layers:
                pi_by_layer[L] = cached[f"pi_{L}"]
                erank_by_layer[L] = cached[f"erank_{L}"]

    if not pi_by_layer:
        snapshot = find_snapshot(args.model_id)
        weight_map = load_weight_map(snapshot)
        print(f"snapshot: {snapshot}")
        for L in layers:
            print(f"[layer {L}] SVD over {E} experts ...", flush=True)
            pi, erank = spectral_stats_for_layer(snapshot, weight_map, L, E, device)
            pi_by_layer[L], erank_by_layer[L] = pi, erank
        save = {"layers": np.array(layers)}
        for L in layers:
            save[f"pi_{L}"] = pi_by_layer[L]
            save[f"erank_{L}"] = erank_by_layer[L]
        os.makedirs(args.out_dir, exist_ok=True)
        np.savez_compressed(cache_path, **save)
        print(f"wrote spectral cache to {cache_path}")

    # ---- Persist raw numbers ---------------------------------------------
    stats = {
        "model_id": args.model_id,
        "num_experts": int(E),
        "intermediate_size": int(I),
        "layers": layers,
        "effective_rank": {str(L): erank_by_layer[L].tolist() for L in layers},
        "contribution": {str(L): contrib[L].cpu().numpy().tolist() for L in layers},
        "effrank_summary": {
            str(L): {
                "mean": float(erank_by_layer[L].mean()),
                "min": float(erank_by_layer[L].min()),
                "max": float(erank_by_layer[L].max()),
                "std": float(erank_by_layer[L].std()),
            } for L in layers
        },
        "leverage_summary": {
            str(L): {
                "top1_mean": float(leverage[L].numpy().max(axis=1).mean()),
                "gini_note": "see figures for descending curves",
            } for L in layers
        },
    }
    with open(args.stats_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {args.stats_json}")

    nL = len(layers)
    cmap = plt.get_cmap("viridis")
    # Grid geometry for the per-layer small multiples (~4:3 aspect).
    ncols = min(8, nL)
    nrows = math.ceil(nL / ncols)

    def _grid(fig_scale=(3.0, 2.4)):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(fig_scale[0] * ncols, fig_scale[1] * nrows),
                                 squeeze=False)
        return fig, axes.ravel()

    # ---- Fig 1: ridge leverage, descending, per expert -------------------
    # One subplot per layer, all 128 experts overlaid (no mean curve).
    fig, flat = _grid()
    x = np.arange(I)
    for k, L in enumerate(layers):
        ax = flat[k]
        lev_sorted = -np.sort(-leverage[L].numpy(), axis=1)   # descending per expert
        for e in range(E):
            ax.plot(x, lev_sorted[e], color=cmap(e / E), lw=0.3, alpha=0.35)
        ax.set_title(f"L{L}", fontsize=8)
        ax.tick_params(labelsize=6)
    for k in range(nL, len(flat)):
        flat[k].axis("off")
    fig.suptitle("Per-expert ridge leverage score, descending "
                 "(x=channel rank, y=score; 128 experts overlaid)", y=1.005)
    fig.supxlabel("channel rank (descending)")
    fig.supylabel("ridge leverage score")
    fig.tight_layout()
    f1 = os.path.join(args.out_dir, "ridge_leverage_descending.png")
    fig.savefig(f1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {f1}")

    # ---- Fig 2: normalized spectral distribution -------------------------
    fig, flat = _grid()
    r = pi_by_layer[layers[0]].shape[1]
    x = np.arange(r)
    for k, L in enumerate(layers):
        ax = flat[k]
        pi = pi_by_layer[L]                        # (E, r), descending from svdvals
        for e in range(E):
            ax.plot(x, pi[e], color=cmap(e / E), lw=0.3, alpha=0.35)
        ax.set_yscale("log")
        ax.set_title(f"L{L}", fontsize=8)
        ax.tick_params(labelsize=6)
    for k in range(nL, len(flat)):
        flat[k].axis("off")
    fig.suptitle(r"Per-expert normalized spectral distribution "
                 r"$\pi_i=\sigma_i^2/\sum_j\sigma_j^2$ (128 experts overlaid)", y=1.005)
    fig.supxlabel("singular value index i")
    fig.supylabel(r"$\pi_i$")
    fig.tight_layout()
    f2 = os.path.join(args.out_dir, "spectral_distribution.png")
    fig.savefig(f2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {f2}")

    # ---- Subset figures: highlight layers only (1xN, no mean curve) ------
    hl = [L for L in args.highlight_layers if L in leverage and L in pi_by_layer]
    if hl:
        nH = len(hl)
        # 1a) ridge leverage, descending
        fig, axes = plt.subplots(1, nH, figsize=(5 * nH, 4), squeeze=False)
        x = np.arange(I)
        for ax, L in zip(axes[0], hl):
            lev_sorted = -np.sort(-leverage[L].numpy(), axis=1)
            for e in range(E):
                ax.plot(x, lev_sorted[e], color=cmap(e / E), lw=0.4, alpha=0.35)
            ax.set_title(f"Layer {L}")
            ax.set_xlabel("channel rank (descending)")
            ax.set_ylabel("ridge leverage score")
        fig.suptitle("Per-expert ridge leverage score, descending "
                     f"(layers {', '.join(map(str, hl))}; 128 experts overlaid)", y=1.02)
        fig.tight_layout()
        f1s = os.path.join(args.out_dir, "ridge_leverage_descending_subset.png")
        fig.savefig(f1s, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {f1s}")

        # 1b) normalized spectral distribution
        fig, axes = plt.subplots(1, nH, figsize=(5 * nH, 4), squeeze=False)
        x = np.arange(pi_by_layer[hl[0]].shape[1])
        for ax, L in zip(axes[0], hl):
            pi = pi_by_layer[L]
            for e in range(E):
                ax.plot(x, pi[e], color=cmap(e / E), lw=0.4, alpha=0.35)
            ax.set_yscale("log")
            ax.set_title(f"Layer {L}")
            ax.set_xlabel("singular value index i")
            ax.set_ylabel(r"$\pi_i=\sigma_i^2/\sum_j\sigma_j^2$")
        fig.suptitle("Per-expert normalized spectral distribution "
                     f"(layers {', '.join(map(str, hl))}; 128 experts overlaid)", y=1.02)
        fig.tight_layout()
        f2s = os.path.join(args.out_dir, "spectral_distribution_subset.png")
        fig.savefig(f2s, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {f2s}")

    # ---- Fig 3: effective rank, layer x expert heatmap -------------------
    erank_mat = np.stack([erank_by_layer[L] for L in layers])   # (nL, E)
    fig, (ax, axm) = plt.subplots(
        1, 2, figsize=(14, 0.16 * nL + 2), gridspec_kw={"width_ratios": [4, 1]})
    im = ax.imshow(erank_mat, aspect="auto", cmap="viridis", origin="lower",
                   extent=[0, E, layers[0] - 0.5, layers[-1] + 0.5]
                   if layers == list(range(layers[0], layers[-1] + 1)) else None)
    ax.set_xlabel("expert index")
    ax.set_ylabel("layer index")
    ax.set_title(f"Expert effective rank  exp(-Σ πᵢ ln πᵢ)   (r_max = {I})")
    fig.colorbar(im, ax=ax, label="effective rank")
    # marginal: mean effective rank vs depth
    axm.plot(erank_mat.mean(1), np.arange(nL), color="navy")
    axm.fill_betweenx(np.arange(nL), erank_mat.min(1), erank_mat.max(1),
                      color="navy", alpha=0.15)
    axm.set_yticks(np.arange(0, nL, max(1, nL // 12)))
    axm.set_yticklabels([str(layers[i]) for i in range(0, nL, max(1, nL // 12))],
                        fontsize=7)
    axm.set_xlabel("mean±range")
    axm.set_title("over experts", fontsize=9)
    fig.tight_layout()
    f3 = os.path.join(args.out_dir, "effective_rank.png")
    fig.savefig(f3, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {f3}")

    # ---- Fig 4: contribution score, layer x expert heatmap ---------------
    contrib_mat = np.stack([contrib[L].cpu().numpy() for L in layers])   # (nL, E)
    # Values are <= 0 with a few large-magnitude spikes; clip the color scale to
    # a robust percentile so mid-layer structure isn't washed out by outliers.
    vmax = float(np.percentile(np.abs(contrib_mat), 99.0))
    fig, (ax, axm) = plt.subplots(
        1, 2, figsize=(14, 0.16 * nL + 2), gridspec_kw={"width_ratios": [4, 1]})
    im = ax.imshow(contrib_mat, aspect="auto", cmap="magma_r", origin="lower",
                   vmin=-vmax, vmax=0.0)
    ax.figure.text(0.01, 0.005,
                   f"color clipped at 99th pct (|contrib|={vmax:.1e}); "
                   f"true min={contrib_mat.min():.1e}", fontsize=7)
    ax.set_xlabel("expert index")
    ax.set_ylabel("layer position (0 = layer {})".format(layers[0]))
    ax.set_yticks(np.arange(0, nL, max(1, nL // 12)))
    ax.set_yticklabels([str(layers[i]) for i in range(0, nL, max(1, nL // 12))],
                       fontsize=7)
    ax.set_title("Expert contribution score (expert_out_token_contrib; "
                 "more negative = more important)")
    fig.colorbar(im, ax=ax, label="contribution")
    # marginal: mean contribution vs depth
    axm.plot(contrib_mat.mean(1), np.arange(nL), color="crimson", label="mean")
    axm.plot(contrib_mat.min(1), np.arange(nL), color="black", lw=0.8,
             alpha=0.6, label="min")
    axm.set_yticks(np.arange(0, nL, max(1, nL // 12)))
    axm.set_yticklabels([str(layers[i]) for i in range(0, nL, max(1, nL // 12))],
                        fontsize=7)
    axm.set_xlabel("contribution")
    axm.legend(fontsize=7)
    fig.tight_layout()
    f4 = os.path.join(args.out_dir, "contribution_score.png")
    fig.savefig(f4, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {f4}")

    print("DONE")


if __name__ == "__main__":
    main()
