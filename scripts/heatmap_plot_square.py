#!/usr/bin/env python
"""Square-cell variant of ``scripts/heatmap_plot.py``.

Same data (``docs/results/heatmap/heatmap.npz`` + ``heatmap_meta.json``), but
every heatmap is drawn with ``aspect="equal"`` so that **each matrix/vector
element renders as a square, not a stretched rectangle**.

Because the layer-trace matrices are very wide (Part 2: ``48 layers x 2048``
hidden dims, Part 3: ``32 tokens x 2048``), square cells would otherwise force a
~40-64:1 thin strip. Instead the wide column axis is **wrapped into 256-wide
chunks** and stacked vertically -- e.g. a ``48 x 2048`` matrix becomes 8 stacked
``48 x 256`` square-celled panels that together cover the whole width.

Outputs go to ``docs/exps/heatmap/sq/`` with the same base filenames as the
original figures, so the original (rectangular-cell) figures and ``heatmap.md``
stay untouched. Referenced from ``docs/exps/heatmap/heatmap_new.md``.

Runs on plain numpy + matplotlib (no model / torch needed).
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
OUTDIR = os.path.join(_REPO, "docs/exps/heatmap/sq")

CHUNK = 256   # columns per wrapped square-celled panel


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
    print(f"[plot] wrote sq/{name}.png/.pdf", flush=True)


def rownorm(mat, eps=1e-12):
    """Normalise each row by its max magnitude -> [0,1] comparability across rows."""
    m = np.abs(mat)
    return m / (m.max(axis=1, keepdims=True) + eps)


def chunks(ncol, w=CHUNK):
    """Column slices [(c0,c1), ...] wrapping ncol into <=w-wide pieces."""
    return [(c0, min(c0 + w, ncol)) for c0 in range(0, ncol, w)]


# ---------------------------------------------------------------------------
# Part 1 — one raw (tokens x channels) heatmap per expert, square cells.
# ---------------------------------------------------------------------------
def plot_part1(d):
    p1_layers = list(d["p1_layers"])
    targets = [tuple(int(v) for v in t) for t in d["p1_targets"]]   # (li, e)
    route = d["p1_route"]                                           # (nL, E)
    by_layer = {}
    for (li, e) in targets:
        by_layer.setdefault(li, []).append(e)

    for li in p1_layers:
        experts = by_layer[li]
        n = len(experts)
        ncol = min(4, n)
        nrow = int(np.ceil(n / ncol))
        # tokens (rows) >= channels (cols=768) here, so a single square-celled
        # panel per expert is already tall-not-wide; no wrapping needed.
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 8.2 * nrow),
                                 squeeze=False)
        axes = axes.reshape(-1)
        for j, e in enumerate(experts):
            m = d[f"p1_L{li}_e{e}"].astype(np.float32)              # (tokens, I)
            ax = axes[j]
            if m.size == 0:
                ax.axis("off")
                continue
            vmax = np.percentile(m[m > 0], 99.0) if (m > 0).any() else 1.0
            im = ax.imshow(m, aspect="equal", cmap="magma", vmin=0, vmax=vmax,
                           interpolation="nearest")
            ax.set_title(f"expert {e}  ({m.shape[0]} tok, rank {j})", fontsize=10)
            ax.set_xlabel("channel (%d)" % m.shape[1])
            if j % ncol == 0:
                ax.set_ylabel("routed token")
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        for j in range(n, len(axes)):
            axes[j].axis("off")
        used = int((route[p1_layers.index(li)] > 0).sum())
        fig.suptitle(f"Layer {li}: per-expert intermediate |SwiGLU| (input of down_proj), "
                     f"top-{n} most-routed experts  [{used}/{route.shape[1]} experts used]",
                     y=1.005, fontsize=12)
        fig.tight_layout()
        _save(fig, f"fig_expert_channel_L{li}")


# ---------------------------------------------------------------------------
# Shared: draw a list of (matrix, label, cmap, vmin, vmax) "blocks" as a single
# column of square-celled panels, wrapping each matrix's columns into CHUNK
# widths. y-axis is shared (layers or tokens).
# ---------------------------------------------------------------------------
def plot_wrapped(blocks, yticks, yticklabels, ylabel, suptitle, name,
                 ytick_fontsize=8, cell=0.030):
    # one panel per column-chunk, stacked vertically
    panels = []   # (block_idx, mat_chunk, c0, c1)
    for bi, blk in enumerate(blocks):
        mat = blk["mat"]
        for (c0, c1) in chunks(mat.shape[1]):
            panels.append((bi, mat[:, c0:c1], c0, c1))
    nrows = len(panels)

    nrows_data = blocks[0]["mat"].shape[0]
    # per-panel height so that CHUNK cols x nrows_data rows come out ~square;
    # width fixed so 256 square cells span the axes. ``cell`` (in) sets the
    # square-cell edge -- bump it up when there are many y-tick labels to fit.
    fig_w = CHUNK * cell + 2.2                       # + labels + colorbar
    row_h = nrows_data * cell + 0.55                 # + title strip
    fig_h = nrows * row_h + 0.8

    fig, axes = plt.subplots(nrows, 1, figsize=(fig_w, fig_h),
                             constrained_layout=True)
    if nrows == 1:
        axes = [axes]

    block_axes = {bi: [] for bi in range(len(blocks))}
    block_im = {}
    for ax, (bi, chunk, c0, c1) in zip(axes, panels):
        blk = blocks[bi]
        im = ax.imshow(chunk, aspect="equal", cmap=blk["cmap"],
                       vmin=blk["vmin"], vmax=blk["vmax"], interpolation="nearest")
        block_axes[bi].append(ax)
        block_im[bi] = im
        ax.set_title(f"{blk['label']}  [{c0}:{c1}]", fontsize=8, pad=2)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=ytick_fontsize)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)

    for bi, blk in enumerate(blocks):
        cb = fig.colorbar(block_im[bi], ax=block_axes[bi], fraction=0.02,
                          pad=0.01, location="right")
        cb.set_label(blk["cbar"], fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle(suptitle, fontsize=12)
    _save(fig, name)


# ---------------------------------------------------------------------------
# Part 2 — per-token layer traces, square cells (wide axes wrapped).
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
    tokens = meta["tokens"]
    L = hidden.shape[0]
    chosen = choose_tokens(meta, explicit)
    print(f"\n[part2] chosen token positions: {chosen}")
    yt = np.arange(0, L, max(1, L // 12))
    ytl = [str(layer_idx[i]) for i in yt]

    for pos in chosen:
        lab = pretty_token(tokens[pos])
        sp = " [SPECIAL]" if meta["is_special"][pos] else ""
        h = rownorm(hidden[:, pos, :])          # (L,H)
        it = rownorm(inter[:, pos, :])          # (L,I)
        pr = probs[:, pos, :]                   # (L,E)
        pvmax = float(np.percentile(pr, 99.9))
        blocks = [
            {"mat": h, "label": "hidden |x| (input of up_proj)", "cmap": "magma",
             "vmin": 0, "vmax": 1, "cbar": "row-norm |x|"},
            {"mat": it, "label": "intermediate (input of down_proj)", "cmap": "magma",
             "vmin": 0, "vmax": 1, "cbar": "row-norm g-wtd |inter|"},
            {"mat": pr, "label": "router prob (activated experts)", "cmap": "viridis",
             "vmin": 0, "vmax": pvmax, "cbar": "softmax prob"},
        ]
        safe = "".join(c if c.isalnum() else "_" for c in lab)[:16] or "tok"
        plot_wrapped(
            blocks, yt, ytl, "layer",
            f"Token #{pos} = '{lab}'{sp}   (trace across {L} MoE layers, "
            f"wrapped into {CHUNK}-wide square-cell panels)",
            f"fig_token_{pos:02d}_{safe}", ytick_fontsize=8)


# ---------------------------------------------------------------------------
# Part 3 — different tokens at the SAME layer, square cells (wide axes wrapped).
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
        hvmax = float(np.percentile(h[h > 0], 99.0)) if (h > 0).any() else 1.0
        ivmax = float(np.percentile(it[it > 0], 99.0)) if (it > 0).any() else 1.0
        blocks = [
            {"mat": h, "label": "hidden state |x| (input of up_proj)", "cmap": "magma",
             "vmin": 0, "vmax": hvmax, "cbar": "|x|"},
            {"mat": it, "label": "intermediate (input of down_proj)", "cmap": "magma",
             "vmin": 0, "vmax": ivmax, "cbar": "g-wtd |inter|"},
        ]
        plot_wrapped(
            blocks, list(range(T)), labels, "",
            f"Layer {li}: hidden state & intermediate across the {T} prompt tokens "
            f"(* = special)  -- wrapped into {CHUNK}-wide square-cell panels",
            f"fig_tokens_at_L{li}", ytick_fontsize=5, cell=0.070)


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
    plot_part1(d)
    plot_part2(d, meta, args.tokens)
    plot_part3(d, meta, args.p3_layers)


if __name__ == "__main__":
    main()
