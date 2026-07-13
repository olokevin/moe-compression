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
    p.add_argument("--layers", type=int, nargs="+", default=[0, 15, 31, 47])
    p.add_argument("--out-dir", default="docs/results/figures")
    p.add_argument("--stats-json", default="docs/results/expert_stats_data.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.stats_json) or ".", exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    layers = args.layers

    # ---- Load precomputed scores -----------------------------------------
    es = torch.load(os.path.join(args.scores_dir, "expert_scores.pth"),
                    map_location="cpu")
    leverage = es["leverage"]                      # {layer: (E, I)}
    contrib = es["expert_out_token_contrib"]       # {layer: (E,)}
    E, I = leverage[layers[0]].shape
    print(f"num_experts={E}, intermediate={I}, layers={layers}")

    # ---- Compute spectral stats from expert weights ----------------------
    snapshot = find_snapshot(args.model_id)
    weight_map = load_weight_map(snapshot)
    print(f"snapshot: {snapshot}")

    pi_by_layer, erank_by_layer = {}, {}
    for L in layers:
        print(f"[layer {L}] SVD over {E} experts ...", flush=True)
        pi, erank = spectral_stats_for_layer(snapshot, weight_map, L, E, device)
        pi_by_layer[L], erank_by_layer[L] = pi, erank

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

    # ---- Fig 1: ridge leverage, descending, per expert -------------------
    fig, axes = plt.subplots(1, nL, figsize=(5 * nL, 4), squeeze=False)
    for ax, L in zip(axes[0], layers):
        lev = leverage[L].numpy()                 # (E, I)
        lev_sorted = -np.sort(-lev, axis=1)       # descending per expert
        x = np.arange(I)
        for e in range(E):
            ax.plot(x, lev_sorted[e], color=cmap(e / E), lw=0.4, alpha=0.35)
        ax.plot(x, lev_sorted.mean(0), color="crimson", lw=1.8, label="mean over experts")
        ax.set_title(f"Layer {L}")
        ax.set_xlabel("channel rank (descending)")
        ax.set_ylabel("ridge leverage score")
        ax.legend(fontsize=8)
    fig.suptitle("Per-expert ridge leverage score (128 experts overlaid)", y=1.02)
    fig.tight_layout()
    f1 = os.path.join(args.out_dir, "ridge_leverage_descending.png")
    fig.savefig(f1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {f1}")

    # ---- Fig 2: normalized spectral distribution -------------------------
    fig, axes = plt.subplots(1, nL, figsize=(5 * nL, 4), squeeze=False)
    for ax, L in zip(axes[0], layers):
        pi = pi_by_layer[L]                        # (E, r) already descending from svdvals
        r = pi.shape[1]
        x = np.arange(r)
        for e in range(E):
            ax.plot(x, pi[e], color=cmap(e / E), lw=0.4, alpha=0.35)
        ax.plot(x, pi.mean(0), color="crimson", lw=1.8, label="mean over experts")
        ax.set_yscale("log")
        ax.set_title(f"Layer {L}")
        ax.set_xlabel("singular value index i")
        ax.set_ylabel(r"$\pi_i=\sigma_i^2/\sum_j\sigma_j^2$")
        ax.legend(fontsize=8)
    fig.suptitle("Per-expert normalized spectral distribution (128 experts overlaid)", y=1.02)
    fig.tight_layout()
    f2 = os.path.join(args.out_dir, "spectral_distribution.png")
    fig.savefig(f2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {f2}")

    # ---- Fig 3: effective rank per expert --------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(E)
    for i, L in enumerate(layers):
        ax.plot(x, erank_by_layer[L], lw=1.0, alpha=0.85,
                color=cmap(i / max(1, nL - 1)) if nL > 1 else "navy",
                label=f"layer {L} (mean {erank_by_layer[L].mean():.1f})")
    ax.set_xlabel("expert index")
    ax.set_ylabel("effective rank  exp(-Σ πᵢ ln πᵢ)")
    ax.set_title(f"Expert effective rank (r_max = {I})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f3 = os.path.join(args.out_dir, "effective_rank.png")
    fig.savefig(f3, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {f3}")

    # ---- Fig 4: contribution score per expert ----------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, L in enumerate(layers):
        c = contrib[L].cpu().numpy()
        ax.plot(x, c, lw=1.0, alpha=0.85,
                color=cmap(i / max(1, nL - 1)) if nL > 1 else "navy",
                label=f"layer {L}")
    ax.set_xlabel("expert index")
    ax.set_ylabel("expert contribution (expert_out_token_contrib)")
    ax.set_title("Expert contribution score")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f4 = os.path.join(args.out_dir, "contribution_score.png")
    fig.savefig(f4, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {f4}")

    print("DONE")


if __name__ == "__main__":
    main()
