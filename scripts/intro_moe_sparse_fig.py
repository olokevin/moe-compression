#!/usr/bin/env python
"""Generate a schematic figure illustrating MoE vs. general sparse activation.

One figure with four panels in a row. Each panel is a tall (width 1 : height 2)
rectangle representing the full parameter tensor. Horizontal grey lines cut it
into 4 parts (experts) -- except panel 2, which has no dividers. Activation is
row-based (blue rows). Saved to docs/presentation/figs as
fig_intro_moe_sparse.{png,pdf}:

  (a) 4 horizontal parts, 1 part (blue) fully active.
  (b) no dividers, random rows active across the whole tensor (general sparsity).
  (c) copy of (a).
  (d) same layout as (a) but only ~25% of the rows inside the active part are on.

No captions or in-figure text -- pure schematics.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Style / palette
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "presentation", "figs")
OUT_DIR = os.path.abspath(OUT_DIR)

BLUE = "#3B7DD8"      # activated
BLUE_EDGE = "#245BA6"
GREY = "#D3D6DB"      # inactive fill
DIVIDER = "#9AA0A8"   # grey part-divider lines
BORDER = "#333333"    # outer rectangle border

N_PARTS = 4           # experts, stacked as horizontal bands
ROWS_PER_PART = 8     # rows within each part
N_ROWS = N_PARTS * ROWS_PER_PART
ACTIVE_PART = 2       # 0-indexed part (from bottom) that is "on"
ROW_KEEP_FRAC = 0.25  # fraction of rows kept in the row-activation panel

# Panel 2 (general sparse): each row is its own expert, labelled E_0..E_{n-1}.
N_SPARSE_EXPERTS = 8

# Outer rectangle geometry: width 1, height 2 (tall portrait block).
W = 1.0
H = 2.0
PART_H = H / N_PARTS
ROW_H = H / N_ROWS

PANEL_W, PANEL_H = 1.5, 3.0   # inches per panel; matches 1:2 aspect
DPI = 300


LABEL_PAD = 0.42      # x-room reserved to the left for the E_i labels

def _init_ax(ax, labels=False):
    # Reserve the same left pad on every panel so all rectangles render at the
    # identical physical size under aspect="equal" (panel 2 just leaves it blank).
    ax.set_xlim(-LABEL_PAD, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")


def _save(fig, name):
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.04,
                    transparent=True)
        print("wrote", path)
    plt.close(fig)


def _base(ax):
    """Grey background filling the whole tensor."""
    ax.add_patch(Rectangle((0, 0), W, H, facecolor=GREY, edgecolor="none",
                            zorder=1))


def _dividers(ax):
    """3 horizontal grey lines splitting the rectangle into 4 parts."""
    for p in range(1, N_PARTS):
        y = p * PART_H
        ax.plot([0, W], [y, y], color=DIVIDER, linewidth=1.4, zorder=4)


def _outer_border(ax):
    """Draw the crisp outer rectangle enclosing the whole tensor."""
    ax.add_patch(Rectangle((0, 0), W, H, fill=False,
                            edgecolor=BORDER, linewidth=2.2, zorder=5))


def _expert_labels(ax):
    """Label the 4 parts E_0 (top) .. E_3 (bottom) to the left of the block."""
    for p in range(N_PARTS):
        # top-most band is E_0
        yc = H - (p + 0.5) * PART_H
        ax.text(-0.12, yc, rf"$E_{{{p}}}$", ha="right", va="center",
                fontsize=13, zorder=6)


def _blue_row(ax, r):
    """Activate row r (full width) in blue."""
    ax.add_patch(Rectangle((0, r * ROW_H), W, ROW_H, facecolor=BLUE,
                            edgecolor=BLUE_EDGE, linewidth=0.4, zorder=3))


# ---------------------------------------------------------------------------
# (a) / (c)  MoE: one full part active
# ---------------------------------------------------------------------------
def draw_moe_experts(ax):
    _init_ax(ax, labels=True)
    _base(ax)
    # fully activate every row of the active part
    for r in range(ACTIVE_PART * ROWS_PER_PART, (ACTIVE_PART + 1) * ROWS_PER_PART):
        _blue_row(ax, r)
    _dividers(ax)
    _outer_border(ax)
    _expert_labels(ax)


# ---------------------------------------------------------------------------
# (b)  General sparse: each fine row is its own expert; random rows active.
#      Only the top three rows are labelled E_0, E_1, E_2 (as an illustration).
# ---------------------------------------------------------------------------
def draw_sparse_rows(ax, keep_frac=0.35, seed=7):
    _init_ax(ax, labels=True)
    _base(ax)
    rng = np.random.default_rng(seed)
    active = rng.random(N_ROWS) < keep_frac
    for r in range(N_ROWS):
        if active[r]:
            _blue_row(ax, r)
    # thin dividers between every fine row-expert
    for r in range(1, N_ROWS):
        y = r * ROW_H
        ax.plot([0, W], [y, y], color=DIVIDER, linewidth=0.4, zorder=4)
    _outer_border(ax)
    # label only the top three rows E_0, E_1, E_2 (top row is E_0); rows are
    # thin, so use a small font sized to the row height
    for k in range(3):
        yc = H - (k + 0.5) * ROW_H
        ax.text(-0.10, yc, rf"$E_{{{k}}}$", ha="right", va="center",
                fontsize=6, zorder=6)
    # a small "..." to signal the remaining rows are also experts
    ax.text(-0.10, H - 4.2 * ROW_H, r"$\vdots$", ha="right", va="center",
            fontsize=8, zorder=6)


# ---------------------------------------------------------------------------
# (d)  MoE + row sparsity: only ~25% of rows within the active part are active
# ---------------------------------------------------------------------------
def draw_moe_row_activation(ax, keep_frac=ROW_KEEP_FRAC, seed=3):
    _init_ax(ax, labels=True)
    _base(ax)
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(keep_frac * ROWS_PER_PART)))
    local = rng.choice(ROWS_PER_PART, size=n_keep, replace=False)
    for lr in local:
        _blue_row(ax, ACTIVE_PART * ROWS_PER_PART + lr)
    _dividers(ax)
    _outer_border(ax)
    _expert_labels(ax)


# titles below each panel (line-wrapped so they don't collide)
TITLES = ["experts", "channel\nexperts", "activate\nwhole expert",
          "activate\nchannel experts"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(4 * PANEL_W, PANEL_H))
    draw_moe_experts(axes[0])
    draw_sparse_rows(axes[1])
    draw_moe_experts(axes[2])
    draw_moe_row_activation(axes[3])
    # centre each title under the outer rectangle (which spans x in [0, W])
    for ax, title in zip(axes, TITLES):
        ax.text(W / 2, -0.05 * H, title, ha="center", va="top",
                fontsize=12, linespacing=1.1, zorder=6)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.12, wspace=0.35)
    _save(fig, "fig_intro_moe_sparse")


if __name__ == "__main__":
    main()
