#!/usr/bin/env python
"""Draw the two illustrative (schematic) figures for the midpoint slides.

No model needed — these are hand-laid diagrams, drawn with matplotlib patches so
they stay vector-crisp in the deck and restyle in one place.

  * ``fig_channel_activation.pdf/.png`` (slide 9) — two tokens of a sequence
    flowing into the same MoE layer: each picks its own K experts *and*, inside
    those experts, its own channels.  The shared expert with a different lit
    channel set is the whole point of the figure.
  * ``fig_framework.pdf/.png`` (slide 11) — the two-phase / one-layer-ahead
    pipeline: layer i computes on partially-loaded gate/down while layer i+1's
    full-width up_proj produces the score heat map that drives the fetch.

Output: ``docs/presentation/figs/``.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# shared style
# ---------------------------------------------------------------------------
INK = "#1c2330"          # primary text / strokes
MUTED = "#8b95a6"        # secondary text
COLD = "#dfe4ec"         # inactive parameter cell
COLD_E = "#f4f6fa"       # inactive expert backdrop
BLUE = "#2f6fdb"         # alias for the primary accent (framework figure)
HOT_A = "#2f6fdb"        # token A accent (blue)
HOT_B = "#e08a1e"        # token B accent (amber)
SHARED = "#7a4fc4"       # accent for the shared expert
PANEL = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})


def _mesh(ax, x0, y0, w, h, nch, nd, lit=None, base=COLD, hot=HOT_A,
          axis="col", gap=0.10):
    """Draw a weight block as a mesh of parameter tiles, lighting the ``lit`` channels.

    ``axis="col"``: channels run along x (one channel = a vertical stripe) — this
    is ``down_proj`` (d x I), whose channels are its *columns*.
    ``axis="row"``: channels run along y — this is ``gate_proj``/``up_proj``
    (I x d), whose channels are their *rows*.  Drawing them differently is the
    honest picture and shows why a channel is "a gate row + an up row + a down
    column".
    """
    lit = set() if lit is None else set(lit)
    nx, ny = (nch, nd) if axis == "col" else (nd, nch)
    cw, ch = w / nx, h / ny
    for i in range(nx):
        for j in range(ny):
            idx = i if axis == "col" else j
            c = hot if idx in lit else base
            if c == "none":
                continue
            ax.add_patch(Rectangle(
                (x0 + i * cw + gap * cw, y0 + j * ch + gap * ch),
                cw * (1 - 2 * gap), ch * (1 - 2 * gap),
                facecolor=c, edgecolor="none", zorder=3))


def _expert_row(ax, x0, y0, w, h, nch, nd, lits, hots, active, label=None,
                sub=None, name_fs=5.6):
    """One expert drawn as gate | up | down side by side inside a rounded card.

    ``lits``/``hots`` are parallel lists so a *shared* expert can show two
    tokens' channel subsets in one box (the crux of the slide-9 figure).
    """
    edge = hots[0] if (active and len(hots) == 1) else (SHARED if active else "#c9d0dc")
    lwid = 2.0 if active else 0.7
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0.006,rounding_size=0.014",
        facecolor=PANEL if active else COLD_E, edgecolor=edge, lw=lwid, zorder=2))

    padx, pady = 0.030 * w, 0.150 * h
    bw = (w - 4 * padx) / 3.0
    bh = h - 2 * pady
    for bi, name in enumerate(("gate", "up", "down")):
        bx = x0 + padx + bi * (bw + padx)
        by = y0 + pady
        axis = "col" if name == "down" else "row"
        ax.add_patch(Rectangle((bx, by), bw, bh, facecolor="none",
                               edgecolor="#dbe1ea", lw=0.5, zorder=2))
        _mesh(ax, bx, by, bw, bh, nch, nd, lit=None, base=COLD if active else "#e8ecf3",
              axis=axis)
        if active:
            for lit, hot in zip(lits, hots):
                _mesh(ax, bx, by, bw, bh, nch, nd, lit=lit, base="none",
                      hot=hot, axis=axis)
        ax.text(bx + bw / 2, by + bh + 0.012, name, ha="center", va="bottom",
                fontsize=name_fs, color=INK if active else MUTED, zorder=4)
    if label:
        ax.text(x0 - 0.012, y0 + h / 2, label, ha="right", va="center",
                fontsize=7.2, color=edge if active else MUTED,
                weight="bold" if active else "normal", zorder=4)
    if sub:
        ax.text(x0 + w + 0.010, y0 + h / 2, sub, ha="left", va="center",
                fontsize=6.0, color=edge if active else MUTED, zorder=4)


def fig_channel_activation(out_dir):
    """Slide 9: two tokens, same MoE layer, different experts AND channels.

    Layout: one horizontal band per token, each band showing the *same* three
    experts, so the eye compares top vs bottom at fixed x — the shared expert
    (E0) sits in the same column in both bands with a different lit subset.
    """
    fig, ax = plt.subplots(figsize=(10.0, 5.1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    NCH, ND = 10, 5                     # channels per block, hidden-dim tiles
    picks = {                            # 3 of 10 channels per (token, expert)
        ("A", 0): [0, 4, 9],             # shared expert, token A's subset
        ("A", 2): [1, 5, 8],
        ("B", 0): [2, 5, 7],             # shared expert, token B's subset (differs)
        ("B", 1): [3, 6, 9],
    }
    bands = {
        "A": dict(y=0.535, hot=HOT_A, word="Where", experts=[0, 2]),
        "B": dict(y=0.140, hot=HOT_B, word="is", experts=[0, 1]),
    }

    EX0, EXW, GAP, EXH = 0.300, 0.148, 0.033, 0.270
    xs = {e: EX0 + i * (EXW + GAP) for i, e in enumerate((0, 1, 2))}

    for key, band in bands.items():
        by = band["y"]
        hot = band["hot"]
        cy = by + EXH / 2

        # ---- the input sequence: same tokens, this band's token highlighted -
        # both bands show "where is …"; the band's own token is coloured, the
        # rest greyed — so the eye reads one sequence processed token by token.
        seq = [("where", 0.062), ("is", 0.036), ("…", 0.024)]
        active_word = band["word"].lower()
        sx = 0.014
        for w, ww in seq:
            on = (w == active_word)
            ax.add_patch(FancyBboxPatch(
                (sx, cy - 0.050), ww, 0.100,
                boxstyle="round,pad=0.004,rounding_size=0.016",
                facecolor=hot if on else "#eef1f5", edgecolor="none",
                alpha=0.16 if on else 1.0, zorder=2))
            ax.add_patch(FancyBboxPatch(
                (sx, cy - 0.050), ww, 0.100,
                boxstyle="round,pad=0.004,rounding_size=0.016",
                facecolor="none", edgecolor=hot if on else "#c8cfdb",
                lw=1.7 if on else 0.8, zorder=3))
            ax.text(sx + ww / 2, cy, w, ha="center", va="center",
                    fontsize=10.5 if on else 8.5,
                    color=hot if on else MUTED,
                    weight="bold" if on else "normal", zorder=4)
            sx += ww + 0.006

        # ---- router ---------------------------------------------------------
        rx, rw = 0.186, 0.048
        ax.add_patch(FancyBboxPatch(
            (rx, by + 0.012), rw, EXH - 0.024,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor="#eef1f7", edgecolor="#c8cfdb", lw=0.8, zorder=2))
        ax.text(rx + rw / 2, cy, "router", rotation=90, ha="center", va="center",
                fontsize=7.0, color=INK, zorder=4)
        ax.add_patch(FancyArrowPatch(
            (sx + 0.002, cy), (rx - 0.004, cy), arrowstyle="-|>", mutation_scale=10,
            lw=1.4, color=hot, zorder=4, shrinkA=0, shrinkB=0))

        # ---- the three experts, same order in both bands --------------------
        for e in (0, 1, 2):
            bx = xs[e]
            active = e in band["experts"]
            shared = e == 0
            if not active:
                _expert_row(ax, bx, by, EXW, EXH, NCH, ND, [], [], False)
                ax.text(bx + EXW / 2, by - 0.026, f"E{e}", ha="center", va="top",
                        fontsize=6.6, color=MUTED, zorder=4)
                continue
            _expert_row(ax, bx, by, EXW, EXH, NCH, ND,
                        [picks[(key, e)]], [hot], True)
            # a thicker outer ring marks "the router picked this expert"
            ax.add_patch(FancyBboxPatch(
                (bx, by), EXW, EXH,
                boxstyle="round,pad=0.006,rounding_size=0.014",
                facecolor="none", edgecolor=SHARED if shared else hot,
                lw=2.2, zorder=5))
            ax.text(bx + EXW / 2, by - 0.026, f"E{e}", ha="center", va="top",
                    fontsize=6.6, color=SHARED if shared else hot,
                    weight="bold", zorder=4)
            # Route wires run *behind* the expert cards (zorder 1) on two
            # separate parallel lanes. Keeping each lane at a constant height
            # stops the far wire (which passes behind a nearer card) from
            # reading as a continuation of the near wire's arrowhead.
            near = bx <= min(xs[q] for q in band["experts"]) + 1e-9
            lane = cy + (0.042 if near else -0.042)
            ax.add_patch(FancyArrowPatch(
                (rx + rw + 0.004, lane), (bx - 0.007, lane),
                arrowstyle="-|>", mutation_scale=9, lw=1.25, color=hot,
                alpha=0.9, zorder=1, shrinkA=0, shrinkB=0))

    # ---- the punchline: E0 is shared, its channels are not ------------------
    e0x = xs[0]
    ax.add_patch(FancyBboxPatch(
        (e0x - 0.013, bands["B"]["y"] - 0.048), EXW + 0.026,
        (bands["A"]["y"] + EXH + 0.024) - (bands["B"]["y"] - 0.048),
        boxstyle="round,pad=0.002,rounding_size=0.014",
        facecolor="none", edgecolor=SHARED, lw=1.6, ls=(0, (4, 2.5)), zorder=7))
    ax.text(e0x + EXW / 2, bands["A"]["y"] + EXH + 0.036,
            "same expert,\ndifferent channels", ha="center", va="bottom",
            fontsize=7.6, color=SHARED, weight="bold", zorder=7,
            linespacing=1.25)

    # ---- "…" after the experts: there are more than the three drawn --------
    ellipsis_x = xs[2] + EXW + GAP + 0.010
    for band in bands.values():
        ax.text(ellipsis_x, band["y"] + EXH / 2, "…", ha="left", va="center",
                fontsize=14, color=MUTED, weight="bold", zorder=4)

    # ---- legend ------------------------------------------------------------
    ly = 0.030
    items = [(HOT_A, 'active for "Where"'),
             (HOT_B, 'active for "is"'),
             (COLD, "inactive (not read)")]
    for i, (c, t) in enumerate(items):
        lx = 0.300 + i * 0.200
        ax.add_patch(Rectangle((lx, ly), 0.014, 0.023, facecolor=c,
                               edgecolor="none", zorder=4))
        ax.text(lx + 0.020, ly + 0.0115, t, fontsize=6.8, va="center", color=INK)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_channel_activation.{ext}"), dpi=400)
    plt.close(fig)


def fig_framework(out_dir, seed=3):
    """Slide 11: one-layer-ahead scoring, drawn as two stacked layers.

    TOP lane = layer i actually computing on its sparse channel budget (gate and
    down hold only the B kept channels; up runs full width because it produces
    the scores). BOTTOM lane = layer i+1 being scored and prefetched concurrently
    (up -> score -> mask -> fetch -> staged gate/down), so its params are resident
    before it runs.
    """
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    NCH, B, WIDE = 32, 3, 12                     # heatmap channels, kept budget, tiles
    # A peaky score profile, like the measured per-token |up.x| distributions.
    score = rng.gamma(0.5, 1.0, NCH)
    score[rng.choice(NCH, 5, replace=False)] += rng.uniform(2.4, 4.2, 5)
    score = score / score.max()
    keep = np.sort(np.argsort(score)[::-1][:5])

    TOP, BOT = 0.590, 0.150                      # lane baselines
    LH = 0.300                                   # lane content height
    BLUE_D = "#2b5fb8"
    LBL = 0.150                                  # left margin for lane labels

    # ---- lane backdrops + big layer labels ---------------------------------
    ax.add_patch(FancyBboxPatch(
        (0.030, TOP - 0.048), 0.940, LH + 0.096,
        boxstyle="round,pad=0.006,rounding_size=0.016",
        facecolor="#f7f9fc", edgecolor="#c8cfdb", lw=1.0, zorder=1))
    ax.add_patch(FancyBboxPatch(
        (0.030, BOT - 0.048), 0.940, LH + 0.096,
        boxstyle="round,pad=0.006,rounding_size=0.016",
        facecolor="#fffdf6", edgecolor="#e6dcc4", lw=1.0, zorder=1))
    ax.text(0.052, TOP + LH / 2 + 0.028, "Layer $i$", fontsize=12.5,
            color=BLUE_D, weight="bold", va="center", ha="left")
    ax.text(0.052, TOP + LH / 2 - 0.024, "compute on its\nsparse channels",
            fontsize=6.6, color=MUTED, va="center", ha="left", linespacing=1.3)
    ax.text(0.052, BOT + LH / 2 + 0.028, "Layer $i\\!+\\!1$", fontsize=12.5,
            color="#a8792a", weight="bold", va="center", ha="left")
    ax.text(0.052, BOT + LH / 2 - 0.024, "score & prefetch\nconcurrently",
            fontsize=6.6, color=MUTED, va="center", ha="left", linespacing=1.3)

    # ======================= TOP LANE: layer i compute ======================
    # gate/down hold only B channels; up is full width (3 hot + 9 grey) because
    # it is computed to produce the scores that pick those B channels.
    ccy = TOP + LH / 2
    nb = 0.088
    blocks = [("gate", B, "row", "#5f8fe0"),
              ("up", WIDE, "row", "#c99b45"),     # full width, only B highlighted
              ("down", B, "col", "#5f8fe0")]
    gx = LBL + 0.055
    step = 0.205
    for bi, (name, nch, axis, hot) in enumerate(blocks):
        bx = gx + bi * step
        ax.add_patch(Rectangle((bx, TOP), nb, LH, facecolor="none",
                               edgecolor="#cdd5e1", lw=0.7, ls=(0, (2.2, 2.2)),
                               zorder=2))
        if name == "up":
            # all 12 rows present: top 3 activated (same mask M as gate/down),
            # the other 9 fetched full-width only to compute the scores (grey).
            # rows count from the bottom, so the top B rows are range(WIDE-B, WIDE).
            _mesh(ax, bx, TOP, nb, LH, WIDE, 9, lit=range(WIDE), base=COLD,
                  hot=COLD, axis="row")
            _mesh(ax, bx, TOP, nb, LH, WIDE, 9, lit=range(WIDE - B, WIDE),
                  base="none", hot=hot, axis="row")
            sub = "3 active + 9 for scores"
        elif axis == "row":
            hgt = LH * nch / WIDE
            _mesh(ax, bx, TOP + LH - hgt, nb, hgt, nch, 9,
                  lit=range(nch), hot=hot, axis="row")
            sub = f"{nch}/{WIDE} channels"
        else:
            _mesh(ax, bx, TOP, nb * nch / WIDE, LH, nch, 9,
                  lit=range(nch), hot=hot, axis="col")
            sub = f"{nch}/{WIDE} channels"
        lab = (name + "$_{[\\mathcal{M}]}$") if name != "up" else "up$_{\\mathrm{full}}$"
        ax.text(bx + nb / 2, TOP + LH + 0.014, lab, ha="center", va="bottom",
                fontsize=7.4, color=INK, weight="bold")
        ax.text(bx + nb / 2, TOP - 0.014, sub, ha="center", va="top",
                fontsize=6.2, color=BLUE_D if name != "up" else "#a8792a")
    for bi, sym in enumerate(("$\\odot$", "$\\rightarrow$")):
        sx = gx + bi * step + nb
        ax.text(sx + (step - nb) / 2, ccy, sym, ha="center", va="center",
                fontsize=13 if bi == 0 else 12, color=INK, zorder=5)
    ax.text(gx + 3 * step - 0.02, ccy, "$y$", ha="left", va="center",
            fontsize=10, color=INK)

    # ======================= BOTTOM LANE: layer i+1 score & prefetch ========
    cy = BOT + LH / 2

    # (1) up_proj at full width — the built-in channel router
    ux, uw = LBL + 0.030, 0.076
    _mesh(ax, ux, BOT, uw, LH, WIDE, 9, lit=range(WIDE), hot="#c99b45", axis="row")
    ax.add_patch(Rectangle((ux, BOT), uw, LH, facecolor="none",
                           edgecolor="#d9cfb4", lw=0.7, zorder=4))
    ax.text(ux + uw / 2, BOT + LH + 0.014, "up$_{\\mathrm{full}}$",
            ha="center", va="bottom", fontsize=7.4, color=INK, weight="bold")

    # (2) the per-channel score heat map
    hx, hw, hh = ux + uw + 0.045, 0.210, 0.095
    hy = cy - hh / 2
    ax.imshow(score.reshape(1, -1), aspect="auto", cmap="inferno",
              extent=(hx, hx + hw, hy, hy + hh), zorder=3, vmin=0, vmax=1)
    ax.add_patch(Rectangle((hx, hy), hw, hh, facecolor="none",
                           edgecolor="#b9c2d0", lw=0.7, zorder=4))
    ax.text(hx + hw / 2, hy + hh + 0.018,
            "$s_j=|\\,\\mathrm{up}_j\\!\\cdot\\! x\\,|$  per channel",
            ha="center", va="bottom", fontsize=6.8, color=INK)
    cw = hw / NCH
    my = hy - 0.046
    for c in keep:
        ax.add_patch(Rectangle((hx + c * cw, my), cw, 0.024,
                               facecolor=BLUE, edgecolor="none", zorder=4))
    ax.text(hx + hw / 2, my - 0.012, "top-$B$ mask $\\mathcal{M}$",
            ha="center", va="top", fontsize=6.6, color=BLUE, weight="bold")

    # (3) memory: fetch only the masked rows/cols
    mx, mw = hx + hw + 0.048, 0.100
    ax.add_patch(FancyBboxPatch(
        (mx, BOT), mw, LH, boxstyle="round,pad=0.005,rounding_size=0.014",
        facecolor="#eef1f7", edgecolor="#b9c2d0", lw=0.8, zorder=2))
    for j in range(9):
        c = BLUE if j in (1, 4, 7) else COLD
        ax.add_patch(Rectangle((mx + 0.013, BOT + 0.016 + j * 0.031),
                               mw - 0.026, 0.021, facecolor=c,
                               edgecolor="none", zorder=3))
    ax.text(mx + mw / 2, BOT + LH + 0.014, "CPU DRAM / HBM", ha="center",
            va="bottom", fontsize=6.8, color=INK)
    ax.text(mx + mw / 2, BOT - 0.014, "fetch only $\\mathcal{M}$'s\nrows & cols",
            ha="center", va="top", fontsize=6.3, color=BLUE, linespacing=1.3)

    for a, b, col in ((ux + uw, hx, "#a8792a"), (hx + hw, mx, BLUE)):
        ax.add_patch(FancyArrowPatch(
            (a + 0.008, cy), (b - 0.007, cy), arrowstyle="-|>",
            mutation_scale=10, lw=1.4, color=col, zorder=5, shrinkA=0, shrinkB=0))

    # (4) staged slice ready for when layer i+1 runs
    px, pw = mx + mw + 0.045, 0.098
    ax.add_patch(FancyBboxPatch(
        (px, BOT + 0.045), pw, LH - 0.090,
        boxstyle="round,pad=0.005,rounding_size=0.014",
        facecolor="#e8f0fd", edgecolor=BLUE, lw=1.2, zorder=2))
    ax.text(px + pw / 2, cy + 0.026,
            "gate$_{[\\mathcal{M}]}$,\ndown$_{[\\mathcal{M}]}$",
            ha="center", va="center", fontsize=7.2, color=BLUE_D, weight="bold",
            linespacing=1.2)
    ax.text(px + pw / 2, cy - 0.036, "staged", ha="center", va="center",
            fontsize=6.3, color=BLUE_D)
    ax.add_patch(FancyArrowPatch(
        (mx + mw + 0.008, cy), (px - 0.007, cy), arrowstyle="-|>",
        mutation_scale=10, lw=1.4, color=BLUE, zorder=5, shrinkA=0, shrinkB=0))

    ax.text(0.500, 0.028,
            "prefetch overlaps layer $i$'s compute  $\\Rightarrow$  "
            "dynamic selection with no wait on the mask",
            ha="center", va="center", fontsize=7.8, color=BLUE_D, weight="bold")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_framework.{ext}"), dpi=400)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs/presentation/figs"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fig_channel_activation(args.out_dir)
    fig_framework(args.out_dir)
    print(f"[illus] wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
