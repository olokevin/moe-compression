#!/usr/bin/env python
"""Render the MoE heatmap figures from ``scripts/heatmap_capture.py`` output.

Produces, under ``docs/exps/heatmap/``:

  * ``fig_expert_channel_L{li}.png/pdf`` and a combined ``fig_expert_channel_grid``
    — Part 1: per-expert (E x I) mean-|inter| channel heatmap, one per target
    layer (0,11,23,47).
  * ``fig_token_{pos}_{label}.png/pdf`` — Part 2: for each selected special/
    content token, a 3-panel figure: hidden state (input of up_proj, L x H),
    intermediate (input of down_proj, L x I), and router probabilities /
    activated experts (L x E).

Runs on plain numpy + matplotlib (no model / torch needed) — do it locally after
pulling ``heatmap.npz`` + ``heatmap_meta.json`` back from the A100.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "figure.dpi": 130,
})

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(_REPO, "docs/exps/heatmap")


def pretty_token(t):
    """BPE token -> human-readable label (GPT-2 byte markers demangled)."""
    if t is None:
        return "?"
    s = t.replace("\u0120", "\u2423").replace("\u010a", "\\n").replace("\u0109", "\\t")
    return s


def _save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"[plot] wrote {name}.png/.pdf", flush=True)


def rownorm(mat, eps=1e-12):
    """Normalise each row by its max magnitude -> [0,1] comparability across rows."""
    m = np.abs(mat)
    return m / (m.max(axis=1, keepdims=True) + eps)


# ---------------------------------------------------------------------------
# Part 1 — one raw (tokens x channels) heatmap per expert (top-N routed)
# ---------------------------------------------------------------------------
def plot_part1(d):
    p1_layers = list(d["p1_layers"])
    targets = [tuple(int(v) for v in t) for t in d["p1_targets"]]   # (li, e)
    route = d["p1_route"]                                           # (nL, E)
    by_layer = {}
    for (li, e) in targets:
        by_layer.setdefault(li, []).append(e)

    print("\n[part1 stats]")
    for li in p1_layers:
        experts = by_layer[li]
        n = len(experts)
        ncol = min(4, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.7 * nrow),
                                 squeeze=False)
        axes = axes.reshape(-1)
        for j, e in enumerate(experts):
            m = d[f"p1_L{li}_e{e}"].astype(np.float32)              # (tokens, I)
            ax = axes[j]
            if m.size == 0:
                ax.axis("off")
                continue
            vmax = np.percentile(m[m > 0], 99.0) if (m > 0).any() else 1.0
            im = ax.imshow(m, aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                           interpolation="nearest")
            ax.set_title(f"expert {e}  ({m.shape[0]} tok, rank {j})", fontsize=10)
            ax.set_xlabel("channel (%d)" % m.shape[1])
            if j % ncol == 0:
                ax.set_ylabel("routed token")
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
            # per-channel activation frequency for the report
            act = (m > 0.1 * vmax).mean(axis=0)                     # frac tokens active per channel
            if j == 0:
                dens = (act > 0.5).mean()                           # channels active for >50% tokens
                print(f"  L{li} top-expert {e}: {m.shape[0]} tok, "
                      f"channels active(>50% tok)={dens*100:.1f}%, max|inter|={m.max():.3e}")
        for j in range(n, len(axes)):
            axes[j].axis("off")
        used = int((route[p1_layers.index(li)] > 0).sum())
        fig.suptitle(f"Layer {li}: per-expert intermediate |SwiGLU| (input of down_proj), "
                     f"top-{n} most-routed experts  [{used}/{route.shape[1]} experts used]",
                     y=1.01, fontsize=12)
        fig.tight_layout()
        _save(fig, f"fig_expert_channel_L{li}")


# ---------------------------------------------------------------------------
# Part 2 — per-token layer traces
# ---------------------------------------------------------------------------
def choose_tokens(meta, explicit):
    is_sp = meta["is_special"]
    if explicit:
        return sorted(set(int(x) for x in explicit.split(",")))
    sp = [i for i, s in enumerate(is_sp) if s]
    ct = [i for i, s in enumerate(is_sp) if not s]
    chosen = []
    chosen += sp[:2]
    if sp:
        chosen.append(sp[-1])
    if ct:
        chosen += [ct[0], ct[len(ct) // 2], ct[-1]]
    return sorted(set(chosen))[:6]


def plot_part2(d, meta, explicit):
    layer_idx = list(d["layer_indices"])
    hidden = d["p2_hidden"]        # (L,T,H)
    inter = d["p2_inter_mag"]      # (L,T,I)
    probs = d["p2_probs"]          # (L,T,E)
    sel = d["p2_sel"]              # (L,T,K)
    tokens = meta["tokens"]
    L = hidden.shape[0]
    chosen = choose_tokens(meta, explicit)
    print(f"\n[part2] chosen token positions: {chosen}")
    yt = np.arange(0, L, max(1, L // 12))

    for pos in chosen:
        lab = pretty_token(tokens[pos])
        sp = " [SPECIAL]" if meta["is_special"][pos] else ""
        h = rownorm(hidden[:, pos, :])          # (L,H)
        it = rownorm(inter[:, pos, :])          # (L,I)
        pr = probs[:, pos, :]                   # (L,E)
        pvmax = np.percentile(pr, 99.9)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2),
                                 gridspec_kw={"width_ratios": [2048, 768, 128]})
        # hidden state (input of up_proj)
        im0 = axes[0].imshow(h, aspect="auto", cmap="magma", vmin=0, vmax=1,
                             interpolation="nearest")
        axes[0].set_title("hidden state |x|  (input of up_proj)")
        axes[0].set_xlabel("hidden dim (%d)" % hidden.shape[2])
        axes[0].set_ylabel("layer")
        fig.colorbar(im0, ax=axes[0], fraction=0.035, pad=0.02).set_label("row-norm |x|")
        # intermediate (input of down_proj)
        im1 = axes[1].imshow(it, aspect="auto", cmap="magma", vmin=0, vmax=1,
                             interpolation="nearest")
        axes[1].set_title("intermediate  (input of down_proj)")
        axes[1].set_xlabel("expert channel (%d)" % inter.shape[2])
        fig.colorbar(im1, ax=axes[1], fraction=0.035, pad=0.02).set_label("row-norm g-wtd |inter|")
        # router probs / activated experts
        im2 = axes[2].imshow(pr, aspect="auto", cmap="viridis", vmin=0, vmax=pvmax,
                             interpolation="nearest")
        axes[2].set_title("router prob  (activated experts)")
        axes[2].set_xlabel("expert (%d)" % probs.shape[2])
        fig.colorbar(im2, ax=axes[2], fraction=0.035, pad=0.02).set_label("softmax prob")
        for ax in axes:
            ax.set_yticks(yt)
            ax.set_yticklabels([str(layer_idx[i]) for i in yt])
        fig.suptitle(f"Token #{pos} = '{lab}'{sp}   (prompt trace across {L} MoE layers)",
                     y=1.02, fontsize=12)
        fig.tight_layout()
        safe = "".join(c if c.isalnum() else "_" for c in lab)[:16] or "tok"
        _save(fig, f"fig_token_{pos:02d}_{safe}")

    # routing overlap across tokens: how many distinct experts each layer uses for
    # the whole prompt, and per-token routing similarity summary
    print("\n[part2 stats]")
    T = hidden.shape[1]
    for pos in chosen:
        lab = pretty_token(tokens[pos])
        # top-1 expert per layer for this token
        top1 = sel[:, pos, 0]
        print(f"  tok#{pos:>2} '{lab}': distinct top-1 experts across layers="
              f"{len(np.unique(top1))}/{L}")
    # expert reuse across tokens: at each layer, union of selected experts / (T*K possible)
    union_frac = []
    for li in range(L):
        u = np.unique(sel[li].reshape(-1)).size
        union_frac.append(u)
    print(f"  distinct experts fired across all {T} tokens per layer: "
          f"min={min(union_frac)} max={max(union_frac)} (of {probs.shape[2]})")


# ---------------------------------------------------------------------------
# Part 3 — different tokens at the SAME layer
# ---------------------------------------------------------------------------
def plot_part3(d, meta, layers_arg):
    layer_idx = list(d["layer_indices"])
    pos_of = {li: p for p, li in enumerate(layer_idx)}
    want = [int(x) for x in layers_arg.split(",") if int(x) in pos_of]
    hidden = d["p2_hidden"]        # (L,T,H)
    inter = d["p2_inter_mag"]      # (L,T,I)
    tokens = meta["tokens"]
    is_sp = meta["is_special"]
    T = hidden.shape[1]
    labels = [f"#{i} {pretty_token(tokens[i])}" + ("*" if is_sp[i] else "")
              for i in range(T)]
    print(f"\n[part3] layers={want}")

    for li in want:
        p = pos_of[li]
        h = np.abs(hidden[p])       # (T,H)
        it = np.abs(inter[p])       # (T,I)
        hvmax = np.percentile(h[h > 0], 99.0) if (h > 0).any() else 1.0
        ivmax = np.percentile(it[it > 0], 99.0) if (it > 0).any() else 1.0
        fig, axes = plt.subplots(1, 2, figsize=(15, max(4.5, 0.22 * T)),
                                 gridspec_kw={"width_ratios": [hidden.shape[2], inter.shape[2]]})
        im0 = axes[0].imshow(h, aspect="auto", cmap="magma", vmin=0, vmax=hvmax,
                             interpolation="nearest")
        axes[0].set_title("hidden state |x|  (input of up_proj)")
        axes[0].set_xlabel("hidden dim (%d)" % hidden.shape[2])
        fig.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.02).set_label("|x|")
        im1 = axes[1].imshow(it, aspect="auto", cmap="magma", vmin=0, vmax=ivmax,
                             interpolation="nearest")
        axes[1].set_title("intermediate  (input of down_proj)")
        axes[1].set_xlabel("expert channel (%d)" % inter.shape[2])
        fig.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.02).set_label("g-wtd |inter|")
        for ax in axes:
            ax.set_yticks(range(T))
        axes[0].set_yticklabels(labels, fontsize=7)
        axes[1].set_yticklabels([])
        fig.suptitle(f"Layer {li}: hidden state & intermediate across the {T} prompt tokens "
                     f"(* = special token)", y=1.005, fontsize=12)
        fig.tight_layout()
        _save(fig, f"fig_tokens_at_L{li}")
        # per-token hidden norm to show token-to-token scale differences
        hn = np.linalg.norm(hidden[p], axis=1)
        j = int(hn.argmax())
        print(f"  L{li}: hidden-norm range [{hn.min():.2f},{hn.max():.2f}], "
              f"largest at tok#{j} '{pretty_token(tokens[j])}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(_REPO, "docs/results/heatmap/heatmap.npz"))
    ap.add_argument("--meta", default=None, help="defaults to <npz stem>_meta.json")
    ap.add_argument("--tokens", default=None, help="explicit comma-list of token positions")
    ap.add_argument("--p3-layers", default="0,11,23,35,47",
                    help="Part 3: layers to compare tokens at")
    args = ap.parse_args()
    meta_path = args.meta or (os.path.splitext(args.npz)[0] + "_meta.json")
    d = np.load(args.npz, allow_pickle=True)
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"[plot] model={meta['model']} L={meta['L']} K={meta['K']} "
          f"E={meta['E']} I={meta['I']} H={meta['H']} T={meta['T']}")
    print(f"[plot] prompt={meta['prompt']!r} chat_template={meta['used_chat_template']}")
    plot_part1(d)
    plot_part2(d, meta, args.tokens)
    plot_part3(d, meta, args.p3_layers)


if __name__ == "__main__":
    main()
