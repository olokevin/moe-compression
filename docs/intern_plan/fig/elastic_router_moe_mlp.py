"""
Conceptual figure for intern_plan.md: elastic training on the MoE-MLP
intermediate-dimension axis only (no width, no depth).

  - ROUTER drawn as a literal 2-layer MLP. Its 3 outputs feed a Gumbel-Softmax;
    each output dim maps to one discrete dim *level* (2048 / 2560 / 3072) and the
    Gumbel-Softmax emits a probability per level. The selected level (3072, with
    prob z = 0.5) is highlighted.
  - LLM FORWARD: a layered transformer stack on the left; we zoom into ONE MoE
    layer and show several experts (… denotes many). Each expert keeps a DIFFERENT
    top-3072 set of rows (W_up) / columns (W_down) — its own importance ranking.
    The MoE output is multiplied by z = 0.5.
  - LOSS = KD(teacher || z * student) + budget loss. Because the student output
    is scaled by z, the loss backprops into BOTH the LLM weights and the router.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'figure.dpi': 150,
})

# ---- palette ------------------------------------------------------------------
BLUE   = '#2563eb'
RED    = '#dc2626'
GREEN  = '#16a34a'
GRAY   = '#cbd5e1'
GRAY_E = '#94a3b8'
INK    = '#1f2937'
PANEL  = '#f8fafc'
PANEL_E= '#cbd5e1'
PURPLE = '#7c3aed'
PURP_L = '#ede9fe'

fig, ax = plt.subplots(figsize=(17.0, 11.0))
ax.set_xlim(0, 17)
ax.set_ylim(0, 11.8)
ax.axis('off')


# ---- helpers ------------------------------------------------------------------
def rbox(x, y, w, h, fc, ec, lw=1.4, z=2, alpha=1.0, rounding=0.06):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.0,rounding_size={rounding}",
                       fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


def rect(x, y, w, h, fc, ec='white', lw=0.8, z=3, alpha=1.0):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha))


def node(x, y, r=0.15, fc='white', ec=INK, lw=1.5, z=6):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z))


def vdots(x, y, color='#94a3b8', dy=0.16, s=22, z=8):
    """Vertical ellipsis drawn as three dots (font-independent)."""
    for k in (-1, 0, 1):
        ax.scatter([x], [y + k * dy], s=s, c=color, marker='o', zorder=z,
                   edgecolors='none')


def hdots(x, y, color='#94a3b8', dx=0.16, s=22, z=8):
    """Horizontal ellipsis drawn as three dots."""
    for k in (-1, 0, 1):
        ax.scatter([x + k * dx], [y], s=s, c=color, marker='o', zorder=z,
                   edgecolors='none')


def fwd_arrow(p0, p1, color=INK, lw=2.0, z=6, style='-|>', mut=16, conn=None, ls='-'):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mut,
                        lw=lw, color=color, zorder=z, linestyle=ls,
                        connectionstyle=conn, shrinkA=0, shrinkB=0, capstyle='round')
    ax.add_patch(a)
    return a


def txt(x, y, s, size=14, color=INK, weight='normal', ha='center', va='center',
        z=8, style='normal', rotation=0):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, style=style, rotation=rotation)


# ================================================================================
# 1. ROUTER  — literal 2-layer MLP -> Gumbel-Softmax -> 3 levels w/ probabilities
# ================================================================================
router_band = rbox(0.30, 8.95, 9.55, 2.55, fc='#f7fef9', ec=GREEN, lw=1.4, z=1,
                   alpha=0.55)
txt(2.55, 11.22, "ROUTER  —  2-layer MLP", size=16, weight='bold', color=GREEN)

# budget input label
txt(0.92, 11.05, "budget", size=11.5, color='#6b7280', style='italic')
txt(0.92, 10.78, "(one-hot)", size=10.5, color='#6b7280', style='italic')

# layer node positions
in_x,  in_ys  = 0.92, [9.55, 10.05, 10.55]
hid_x, hid_ys = 1.95, [9.40, 9.85, 10.30, 10.75]
out_ys = [10.45, 9.95, 9.45]          # aligns with the 3 levels
out_x  = 2.98

# edges in -> hidden, hidden -> out
for x0, y0 in zip([in_x] * 3, in_ys):
    for y1 in hid_ys:
        ax.plot([x0, hid_x], [y0, y1], color='#cbd5e1', lw=0.7, zorder=4)
for y0 in hid_ys:
    for y1 in out_ys:
        ax.plot([hid_x, out_x], [y0, y1], color='#cbd5e1', lw=0.7, zorder=4)

# nodes
for i, y in enumerate(in_ys):
    node(in_x, y, fc=(GREEN if i == 1 else 'white'),
         ec=(GREEN if i == 1 else GRAY_E))
for y in hid_ys:
    node(hid_x, y, fc='white', ec=INK)
for y in out_ys:
    node(out_x, y, fc='white', ec=INK)
txt(in_x, 9.05, "input", size=10.5, color='#6b7280')
txt(hid_x, 9.05, "hidden", size=10.5, color='#6b7280')
txt(out_x, 9.05, "3 logits", size=10.5, color='#6b7280')

# Gumbel-Softmax block (spans the 3 output rows)
gb_x, gb_y, gb_w, gb_h = 3.70, 9.18, 1.30, 1.55
rbox(gb_x, gb_y, gb_w, gb_h, fc='white', ec=PURPLE, lw=1.7, z=3)
txt(gb_x + gb_w / 2, gb_y + gb_h / 2 + 0.18, "Gumbel-", size=12.5, weight='bold', color=PURPLE)
txt(gb_x + gb_w / 2, gb_y + gb_h / 2 - 0.14, "Softmax", size=12.5, weight='bold', color=PURPLE)
txt(gb_x + gb_w / 2, gb_y + gb_h / 2 - 0.46, r"($\tau$)", size=11, color=PURPLE)

# out nodes -> gumbel
for y in out_ys:
    fwd_arrow((out_x + 0.16, y), (gb_x, y), color='#94a3b8', lw=1.4, mut=11)

# 3 discrete levels with probabilities (no bars), selected highlighted
levels = [("2048 dims", "p = 0.20", False),
          ("2560 dims", "p = 0.30", False),
          ("3072 dims", "p = 0.50", True)]
lv_x, lv_w, lv_h = 5.30, 2.30, 0.50
for (name, prob, sel), y in zip(levels, out_ys):
    fwd_arrow((gb_x + gb_w, y), (lv_x, y), color=PURPLE, lw=1.5, mut=11)
    rbox(lv_x, y - lv_h / 2, lv_w, lv_h,
         fc=(PURP_L if sel else 'white'), ec=(PURPLE if sel else PANEL_E),
         lw=(2.0 if sel else 1.2), z=4)
    txt(lv_x + 0.78, y, name, size=12.5, weight=('bold' if sel else 'normal'),
        color=(PURPLE if sel else INK), z=5)
    txt(lv_x + 1.78, y, prob, size=12, weight=('bold' if sel else 'normal'),
        color=(PURPLE if sel else '#6b7280'), z=5)
txt(lv_x + lv_w / 2, 11.05, "level  →  probability", size=12, weight='bold',
    color=INK)

# selected box
sel_x, sel_y, sel_w, sel_h = 8.00, 9.30, 1.70, 1.30
rbox(sel_x, sel_y, sel_w, sel_h, fc='#f5f3ff', ec=PURPLE, lw=2.0, z=4)
txt(sel_x + sel_w / 2, sel_y + sel_h - 0.28, "selected", size=11, color='#6b7280',
    style='italic')
txt(sel_x + sel_w / 2, sel_y + sel_h - 0.66, "3072", size=15.5, weight='bold', color=PURPLE)
txt(sel_x + sel_w / 2, sel_y + 0.28, "z = 0.50", size=13, weight='bold', color=PURPLE)
fwd_arrow((lv_x + lv_w, 9.45), (sel_x, sel_y + 0.45), color=PURPLE, lw=1.8, mut=12,
          conn="arc3,rad=-0.15")


# ================================================================================
# 2. LLM FORWARD : layered stack (left)  +  zoom into ONE MoE layer (right)
# ================================================================================
mx, my, mw, mh = 0.35, 0.45, 9.55, 8.05
rbox(mx, my, mw, mh, fc='white', ec=INK, lw=1.6, z=1)
txt(mx + 0.35, my + mh - 0.32, "LLM forward", size=16, weight='bold', color=INK,
    ha='left')

# ---- 2a. stack of DECODER LAYERS (left); one is expanded to self-attn + MoE ----
ds_x, ds_w = 0.78, 2.02
layer_h = 0.62          # plain decoder layer height
exp_h = 2.05            # expanded (selected) decoder layer height
gap = 0.20


def decoder_layer(ly):
    rect(ds_x, ly, ds_w, layer_h, fc='#eef2ff', ec='#a5b4fc', lw=1.2, z=3)
    txt(ds_x + ds_w / 2, ly + layer_h / 2, "Decoder layer", size=11.5,
        color='#4f46e5', z=4)


# bottom -> top
y = 1.05
decoder_layer(y);                       y += layer_h + gap        # L1
decoder_layer(y);                       y += layer_h + gap        # L2

# --- expanded / selected decoder layer: contains Self-Attn + MoE ---
ey = y
rect(ds_x, ey, ds_w, exp_h, fc='white', ec=PURPLE, lw=1.9, z=3)
txt(ds_x + ds_w / 2, ey + exp_h - 0.22, "Decoder layer", size=11.5,
    weight='bold', color=PURPLE, z=5)
sub_x, sub_w = ds_x + 0.16, ds_w - 0.32
# self-attention sub-block
sa_h = 0.62
sa_y = ey + exp_h - 1.06
rect(sub_x, sa_y, sub_w, sa_h, fc='#e0e7ff', ec='#6366f1', lw=1.2, z=4)
txt(sub_x + sub_w / 2, sa_y + sa_h / 2, "Self-Attn", size=11, color='#4338ca', z=5)
# MoE sub-block (highlighted — this is what we zoom into)
moe_h = 0.66
moe_y = ey + 0.20
rect(sub_x, moe_y, sub_w, moe_h, fc=PURPLE, ec=PURPLE, lw=1.3, z=4)
txt(sub_x + sub_w / 2, moe_y + moe_h / 2, "MoE", size=12, weight='bold',
    color='white', z=5)
moe_cy = moe_y + moe_h / 2
moe_right_x = sub_x + sub_w
y = ey + exp_h + gap

decoder_layer(y);                       y += layer_h + gap        # L3

stack_top = y - gap
hdots(ds_x + ds_w / 2, stack_top + 0.24, s=16, dx=0.14)
txt(ds_x + ds_w / 2, stack_top + 0.50, "more layers", size=11, color='#94a3b8',
    style='italic')

# ---- 2b. zoom panel : inside the MoE block ----
zx, zy, zw, zh = 3.35, 0.85, 6.35, 6.30
rbox(zx, zy, zw, zh, fc='#fbfaff', ec=PURPLE, lw=1.4, z=2)
txt(zx + zw / 2, zy + zh - 0.30, "inside the MoE block", size=14, weight='bold',
    color=PURPLE)

# zoom bracket from the highlighted MoE sub-block to the panel
fwd_arrow((moe_right_x, moe_cy), (zx, zy + zh - 0.55), color=PURPLE, lw=1.8,
          style='-|>', mut=14, conn="arc3,rad=-0.18")
txt((moe_right_x + zx) / 2 + 0.10, (moe_cy + zy + zh) / 2 + 0.05, "zoom", size=11,
    color=PURPLE, style='italic')

# column headers inside the zoom
up_x, up_w = 4.95, 0.85
dn_x, dn_w = 7.05, 1.70
txt(up_x + up_w / 2, zy + zh - 0.78, "W_up\n(keep rows)", size=11.5, weight='bold',
    color=PURPLE)
txt(dn_x + dn_w / 2, zy + zh - 0.78, "W_down\n(keep cols)", size=11.5, weight='bold',
    color=PURPLE)

# experts (several shown; ⋮ denotes many)
n_units = 8
rows = [
    ("Expert 1", 4.95, {0, 1, 2, 4, 5, 7}),
    ("Expert 2", 3.95, {0, 2, 3, 4, 6, 7}),
    ("DOTS",     3.18, None),
    ("Expert E", 2.30, {1, 2, 3, 5, 6, 7}),
]
out_bus_x = 9.20
merge_ys = []
for name, cy, keep in rows:
    if keep is None:
        vdots(4.05, cy, dy=0.14)
        vdots(up_x + up_w / 2, cy, dy=0.14)
        vdots(dn_x + dn_w / 2, cy, dy=0.14)
        txt(zx + zw / 2 + 0.95, cy - 0.02, "many experts (showing a few)", size=10.5,
            color='#94a3b8', style='italic')
        continue
    merge_ys.append(cy)
    txt(4.05, cy, name, size=12, weight='bold', color=PURPLE, rotation=90)
    fwd_arrow((4.35, cy), (up_x, cy), color='#94a3b8', lw=1.4, mut=10)

    # W_up rows
    uh = 1.05
    row_h = uh / n_units
    for i in range(n_units):
        ry = cy - uh / 2 + i * row_h
        rect(up_x, ry, up_w, row_h * 0.82,
             fc=(PURPLE if i in keep else GRAY), ec='white', lw=0.6, z=4)
    fwd_arrow((up_x + up_w, cy), (dn_x, cy), color='#94a3b8', lw=1.4, mut=10)

    # W_down cols
    dh = 0.72
    col_w = dn_w / n_units
    for i in range(n_units):
        cx = dn_x + i * col_w
        rect(cx, cy - dh / 2, col_w * 0.82, dh,
             fc=(PURPLE if i in keep else GRAY), ec='white', lw=0.6, z=4)
    fwd_arrow((dn_x + dn_w, cy), (out_bus_x, cy), color=PURPLE, lw=1.6, mut=11)

txt(zx + zw / 2, zy + 0.30,
    "same dim budget (6/8 ≈ 3072/4096) — different neurons per expert (own ranking)",
    size=10.5, color=PURPLE, style='italic')

# merge bus -> scaling node
ax.plot([out_bus_x, out_bus_x], [min(merge_ys), max(merge_ys)], color=PURPLE,
        lw=1.7, zorder=5)
mid_y = sum(merge_ys) / len(merge_ys)


# ================================================================================
# 3. SCALING node  ( x z = 0.5 )  — the differentiable link to the router
# ================================================================================
sc_x, sc_y, sc_r = 10.55, mid_y, 0.40
fwd_arrow((out_bus_x, mid_y), (sc_x - sc_r, sc_y), color=PURPLE, lw=2.0, mut=13)
ax.add_patch(Circle((sc_x, sc_y), sc_r, fc='white', ec=PURPLE, lw=2.2, zorder=6))
txt(sc_x, sc_y + 0.08, "×", size=20, weight='bold', color=PURPLE, z=7)
txt(sc_x, sc_y - 0.20, "z=.5", size=9.5, weight='bold', color=PURPLE, z=7)
txt(sc_x, sc_y + 0.78, "scale output by z", size=11, weight='bold', color=PURPLE)

# selected box -> experts (mask) and -> scaling node (z), both dashed
fwd_arrow((sel_x + 0.3, sel_y), (6.55, zy + zh + 0.02), color=PURPLE, lw=1.8,
          mut=14, conn="arc3,rad=0.22", ls=(0, (5, 2)))
txt(7.65, 8.62, "apply: keep top-3072", size=11.5, weight='bold',
    color=PURPLE, ha='center')
txt(7.65, 8.34, "(per-expert ranking)", size=10.5, color=PURPLE, ha='center',
    style='italic')
fwd_arrow((sel_x + sel_w, sel_y + 0.5), (sc_x + 0.10, sc_y + sc_r), color=PURPLE,
          lw=1.7, mut=12, conn="arc3,rad=0.32", ls=(0, (5, 2)))
txt(11.35, 6.7, "z = 0.5", size=11, weight='bold', color=PURPLE, ha='center',
    rotation=-72)


# ================================================================================
# 4. LOSS  (right)
# ================================================================================
rbox(11.55, 7.05, 4.95, 1.35, fc='white', ec=INK, lw=1.5, z=3)
txt(11.55 + 4.95 / 2, 7.92, "Teacher = full MoE MLP", size=14, weight='bold', color=INK)
txt(11.55 + 4.95 / 2, 7.46, "all 4096 dims, no scaling", size=11.5, color='#6b7280',
    style='italic')

lx, ly, lw_, lh = 11.55, 3.20, 4.95, 3.20
rbox(lx, ly, lw_, lh, fc='#fef2f2', ec=RED, lw=1.9, z=3)
txt(lx + lw_ / 2, ly + lh - 0.42, "LOSS", size=18, weight='bold', color=RED)
txt(lx + lw_ / 2, ly + lh - 1.14, r"$\mathcal{L}=$KD$(\,$teacher $\Vert$ $z\!\cdot\!$student$\,)$",
    size=14, color=INK)
txt(lx + lw_ / 2, ly + lh - 1.58, "(distillation loss)", size=11, color='#9ca3af',
    style='italic')
txt(lx + lw_ / 2, ly + lh - 2.20, r"$+\;\;\Vert\,$Cost$(3072)-$Target$\,\Vert$",
    size=14, color=INK)
txt(lx + lw_ / 2, ly + lh - 2.64, "(budget loss)", size=11, color='#9ca3af',
    style='italic')

# forward arrows into loss
fwd_arrow((sc_x + sc_r, sc_y + 0.05), (lx, ly + lh - 1.1), color=PURPLE, lw=2.2,
          conn="arc3,rad=-0.12")
txt(11.05, 4.35, "z·student", size=11, color=PURPLE, ha='center', style='italic')
fwd_arrow((11.55 + 4.95 / 2, 7.05), (11.55 + 4.95 / 2, ly + lh), color=INK, lw=1.9)
txt(14.55, 6.72, "teacher logits", size=11, color='#374151', ha='center')


# ================================================================================
# 5. BACKWARD  (loss updates LLM weights AND router params, jointly)
# ================================================================================
# loss -> RIGHT END of the LLM forward box (dip below the forward clutter)
fwd_arrow((lx, ly + 0.30), (mx + mw, 1.55), color=RED, lw=2.4, mut=17,
          conn="arc3,rad=-0.32")
txt(10.72, 0.92, "update LLM\nweights", size=11.5, weight='bold', color=RED,
    ha='center')
txt(10.72, 0.42, r"$\partial\mathcal{L}/\partial\theta$", size=12, color=RED,
    ha='center')

# loss -> RIGHT END of the ROUTER band (rise up the corridor left of the teacher)
router_right_x = 0.30 + 9.55          # right edge of the router band
fwd_arrow((lx, ly + lh - 0.55), (router_right_x, 11.02),
          color=RED, lw=2.4, mut=17, conn="arc3,rad=0.30")
txt(11.05, 9.05, "update router", size=11.5, weight='bold', color=RED, ha='center')
txt(11.05, 8.70, r"$\partial\mathcal{L}/\partial\phi$  (through $z$)", size=11.5,
    color=RED, ha='center')


# ---- title --------------------------------------------------------------------
# (kept compact; main message is in the backward label + section headers)

plt.tight_layout()
out = '/Users/yequan/Documents/Macro/Intern Plan/fig/elastic_router_moe_mlp.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("saved", out)
