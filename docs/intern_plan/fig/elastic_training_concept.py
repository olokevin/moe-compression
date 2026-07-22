"""
Conceptual figure for intern_plan.md section 3.2 "How Elastic Training Works".

Layout (top -> bottom, with feedback on the right):
  - TOP:    Router (per-axis 2-layer MLP). Takes a budget target (one-hot),
            emits a per-axis config (the kept size on each axis).
  - MIDDLE: The LLM rendered in THIS config. One model block shows the full
            model (outline = teacher) and the active submodel (blue = student);
            masked parameters / layers are gray. The three compressible axes
            are annotated: Depth, Width (hidden dim), MoE MLP intermediate dim.
  - RIGHT:  Loss = KD(teacher || student) + budget loss.
  - BACK:   Dashed red gradient arrows flow the loss back to BOTH the model
            weights and the router, updating them jointly.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({
    'font.size': 13,
    'font.family': 'serif',
    'figure.dpi': 150,
})

# ---- palette (matches existing project figures) -------------------------------
BLUE   = '#2563eb'   # kept / active (student submodel)
BLUE_L = '#bfdbfe'   # kept, light fill
RED    = '#dc2626'   # loss + gradient feedback
GREEN  = '#16a34a'   # router
GRAY   = '#cbd5e1'   # masked units
GRAY_E = '#94a3b8'   # masked edge
INK    = '#1f2937'   # text
PANEL  = '#f8fafc'   # light panel fill
PANEL_E= '#cbd5e1'   # panel edge
TEAL   = '#0d9488'
PURPLE = '#7c3aed'

fig, ax = plt.subplots(figsize=(14.5, 9.6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
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


def fwd_arrow(p0, p1, color=INK, lw=2.0, z=6, style='-|>', mut=16, conn=None):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mut,
                        lw=lw, color=color, zorder=z,
                        connectionstyle=conn, shrinkA=0, shrinkB=0,
                        capstyle='round')
    ax.add_patch(a)
    return a


def txt(x, y, s, size=13, color=INK, weight='normal', ha='center', va='center',
        z=8, style='normal', rotation=0):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, style=style, rotation=rotation)


# ================================================================================
# 1. BUDGET INPUT  ->  ROUTER  (top)
# ================================================================================
# budget input
rbox(0.20, 9.10, 2.55, 1.25, fc='white', ec=GREEN, lw=1.6, z=3)
txt(1.47, 10.00, "Budget target", size=13, weight='bold', color=GREEN)
txt(1.47, 9.60, "(one-hot)", size=11, color='#6b7280', style='italic')
txt(1.47, 9.32, r"e.g.  c.f. = 1.5", size=12, color=INK)

# router box
rbox(3.35, 8.95, 6.55, 1.55, fc='#ecfdf5', ec=GREEN, lw=1.8, z=3)
txt(6.62, 10.18, "ROUTER", size=18, weight='bold', color=GREEN)
txt(6.62, 9.72, "one tiny 2-layer MLP per axis", size=12.5, color=INK)
txt(6.62, 9.36, "Gumbel-Softmax  →  differentiable per-axis config",
    size=11.5, color='#047857', style='italic')

fwd_arrow((2.75, 9.72), (3.35, 9.72), color=GREEN, lw=2.4)

# ================================================================================
# 2. CONFIG OUTPUT panel  (between router and model)
# ================================================================================
cfg_x, cfg_y, cfg_w, cfg_h = 3.45, 6.95, 6.30, 1.72
rbox(cfg_x, cfg_y, cfg_w, cfg_h, fc=PANEL, ec=PANEL_E, lw=1.3, z=3)
txt(cfg_x + cfg_w / 2, cfg_y + cfg_h - 0.30, "config  (how much to keep per axis)",
    size=12.5, weight='bold', color=INK)

cfg_rows = [
    ("Depth", "42 / 48 layers", BLUE),
    ("Width  (hidden)", "3584 / 4096", TEAL),
    ("MoE MLP dim", "3072 / 4096", PURPLE),
]
chip_x, chip_w = cfg_x + 4.10, 1.95
ry = cfg_y + cfg_h - 0.74
for name, val, col in cfg_rows:
    txt(cfg_x + 0.32, ry, name, size=12.5, ha='left', color=INK)
    rbox(chip_x, ry - 0.18, chip_w, 0.38, fc='white', ec=col, lw=1.5, z=4)
    txt(chip_x + chip_w / 2, ry, val, size=12, weight='bold', color=col, z=5)
    ry -= 0.45

# router -> config
fwd_arrow((6.62, 8.95), (6.62, cfg_y + cfg_h), color=GREEN, lw=2.4)
# config -> model (these are the masks)
fwd_arrow((5.0, cfg_y), (5.0, 6.58), color=GRAY_E, lw=2.6)
txt(5.95, 6.76, "applied as masks", size=11, color='#6b7280', style='italic', ha='left')


# ================================================================================
# 3. THE LLM in this config  (the model block)
# ================================================================================
mx, my, mw, mh = 0.55, 0.55, 9.30, 6.00
rbox(mx, my, mw, mh, fc='white', ec=INK, lw=1.6, z=1)
txt(mx + mw / 2, my + mh - 0.32, "LLM forward in this config",
    size=15, weight='bold', color=INK)
txt(mx + mw / 2, my + mh - 0.66,
    "outline = full model (teacher)    blue = active submodel (student)    gray = masked",
    size=10.5, color='#6b7280', style='italic')

# ---- 3a. DEPTH stack (left) — show 5 layers ------------------------------------
ds_x, ds_w = 1.40, 2.25
layer_h = 0.74
gap = 0.22
n_layers = 5
skip_idx = {3}                       # one skipped layer
base_y = my + 0.50
zoom_layer = 1                       # which active layer we "zoom" into

layer_centers = {}
for i in range(n_layers):
    ly = base_y + i * (layer_h + gap)
    layer_centers[i] = ly + layer_h / 2
    if i in skip_idx:
        rect(ds_x, ly, ds_w, layer_h, fc=GRAY, ec=GRAY_E, lw=1.0, z=3)
        txt(ds_x + ds_w / 2, ly + layer_h / 2, "Layer\n(skipped)", size=12.5,
            color='#64748b', z=4)
    else:
        emph = (i == zoom_layer)
        rect(ds_x, ly, ds_w, layer_h, fc=(BLUE if emph else BLUE_L),
             ec=BLUE, lw=1.4, z=3)
        txt(ds_x + ds_w / 2, ly + layer_h / 2, "Transformer\nLayer", size=13.5,
            color=('white' if emph else BLUE), weight=('bold' if emph else 'normal'),
            z=4)

# residual bypass around the skipped layer
sy = base_y + sorted(skip_idx)[0] * (layer_h + gap)
fwd_arrow((ds_x - 0.14, sy - gap - 0.02), (ds_x - 0.14, sy + layer_h + gap + 0.02),
          color='#64748b', lw=1.8, style='-|>', mut=13,
          conn="arc3,rad=-0.55")
txt(ds_x - 0.66, sy + layer_h / 2, "residual\nbypass", size=10, color='#64748b',
    ha='center')

# depth axis annotation (vertical double arrow to the right of stack)
ax_x = ds_x + ds_w + 0.32
stack_top = base_y + n_layers * (layer_h + gap) - gap
fwd_arrow((ax_x, base_y), (ax_x, stack_top), color=BLUE, lw=1.8, style='<|-|>', mut=13)
txt(ax_x + 0.22, (base_y + stack_top) / 2, "Depth",
    size=13.5, weight='bold', color=BLUE, ha='center', rotation=90)
txt(ax_x + 0.58, (base_y + stack_top) / 2, "42 / 48",
    size=11, color=BLUE, ha='center', rotation=90)

# ---- 3b. ZOOM of one layer (right): width strip + MoE + expert MLP --------------
zx, zy, zw, zh = 4.55, my + 0.50, 5.15, 4.65
rbox(zx, zy, zw, zh, fc='#fbfdff', ec='#dbeafe', lw=1.3, z=2)
txt(zx + zw / 2, zy + zh - 0.28, "inside one active layer", size=12,
    weight='bold', color=BLUE)

# bracket from the zoomed layer to this panel
zl_y = layer_centers[zoom_layer]
fwd_arrow((ds_x + ds_w, zl_y), (zx, zy + zh - 0.55), color=BLUE, lw=1.5,
          style='-|>', mut=12, conn="arc3,rad=-0.15")

# --- Width strip (token hidden vector) ---
wstrip_x, wstrip_y = zx + 0.55, zy + zh - 1.30
cell_w, cell_h = 0.45, 0.45
n_width = 8
mask_width = {7}                     # rightmost channel masked
for j in range(n_width):
    cx = wstrip_x + j * (cell_w + 0.045)
    if j in mask_width:
        rect(cx, wstrip_y, cell_w, cell_h, fc=GRAY, ec=GRAY_E, lw=0.9, z=4)
    else:
        rect(cx, wstrip_y, cell_w, cell_h, fc=TEAL, ec='white', lw=0.9, z=4)
txt(wstrip_x, wstrip_y + cell_h + 0.26, "Width (hidden dim)", size=12,
    weight='bold', color=TEAL, ha='left')
wend = wstrip_x + n_width * (cell_w + 0.045)
txt(wend + 0.06, wstrip_y + cell_h / 2, "3584\n/4096", size=10.5, color=TEAL, ha='left')

# --- MoE block: experts ---
moe_y = zy + 0.55
moe_h = 2.40
exp_w = 1.18
exp_xs = [zx + 0.55, zx + 0.55 + exp_w + 0.32, zx + 0.55 + 2 * (exp_w + 0.32)]
txt(zx + 0.55, moe_y + moe_h + 0.20, "MoE experts", size=12, weight='bold',
    color=INK, ha='left')

# arrow width -> moe
fwd_arrow((wstrip_x + 1.4, wstrip_y), (wstrip_x + 1.4, moe_y + moe_h + 0.02),
          color='#94a3b8', lw=1.5, style='-|>', mut=11)

for k, ex in enumerate(exp_xs):
    if k == 0:
        # this expert is expanded -> intermediate neuron column
        rbox(ex, moe_y, exp_w, moe_h, fc='white', ec=PURPLE, lw=1.7, z=3)
        txt(ex + exp_w / 2, moe_y + moe_h - 0.24, "expert 1", size=11,
            weight='bold', color=PURPLE)
        ncol_x = ex + exp_w / 2 - 0.15
        ncol_y0 = moe_y + 0.30
        nh = 0.205
        n_neur = 8
        mask_neur = {6, 7}           # 2 of 8 masked
        for t in range(n_neur):
            ny = ncol_y0 + t * (nh + 0.02)
            if t in mask_neur:
                rect(ncol_x, ny, 0.30, nh, fc=GRAY, ec=GRAY_E, lw=0.7, z=5)
            else:
                rect(ncol_x, ny, 0.30, nh, fc='#a78bfa', ec='white', lw=0.7, z=5)
    else:
        rbox(ex, moe_y, exp_w, moe_h, fc='#f5f3ff', ec='#c4b5fd', lw=1.3, z=3)
        txt(ex + exp_w / 2, moe_y + moe_h - 0.24, f"expert {k+1}", size=11,
            color=PURPLE)
        txt(ex + exp_w / 2, moe_y + moe_h / 2 - 0.1, "…", size=18, color='#a78bfa')

# callout: MoE MLP intermediate dim (points at expert-1 neuron column)
ann_x = exp_xs[0] + exp_w + 0.10
fwd_arrow((ann_x + 1.10, moe_y + moe_h / 2), (exp_xs[0] + exp_w / 2 + 0.22,
          moe_y + moe_h / 2 - 0.1), color=PURPLE, lw=1.5, style='-|>', mut=11)
txt(ann_x + 1.20, moe_y + moe_h / 2 + 0.30, "MoE MLP", size=12, weight='bold',
    color=PURPLE, ha='left')
txt(ann_x + 1.20, moe_y + moe_h / 2 - 0.02, "intermediate dim", size=12,
    weight='bold', color=PURPLE, ha='left')
txt(ann_x + 1.20, moe_y + moe_h / 2 - 0.34, "3072 / 4096", size=10.5,
    color=PURPLE, ha='left')


# ================================================================================
# 4. LOSS  (right)
# ================================================================================
# teacher node (full model, no masks)
rbox(11.05, 7.50, 4.55, 1.35, fc='white', ec=INK, lw=1.5, z=3)
txt(11.05 + 4.55 / 2, 8.42, "Teacher = full model", size=13.5, weight='bold', color=INK)
txt(11.05 + 4.55 / 2, 8.00, "same weights, no masks", size=11.5, color='#6b7280',
    style='italic')

# loss box
lx, ly, lw_, lh = 11.05, 3.80, 4.55, 2.90
rbox(lx, ly, lw_, lh, fc='#fef2f2', ec=RED, lw=1.9, z=3)
txt(lx + lw_ / 2, ly + lh - 0.38, "LOSS", size=18, weight='bold', color=RED)
txt(lx + lw_ / 2, ly + lh - 1.04, r"$\mathcal{L}\;=\;$KD$(\,$teacher $\Vert$ student$\,)$",
    size=13.5, color=INK)
txt(lx + lw_ / 2, ly + lh - 1.48, "(distillation loss)", size=10.5, color='#9ca3af',
    style='italic')
txt(lx + lw_ / 2, ly + lh - 2.08, r"$+\;\;\Vert\,$Cost$(c)-$Target$\,\Vert$",
    size=13.5, color=INK)
txt(lx + lw_ / 2, ly + lh - 2.52, "(budget loss)", size=10.5, color='#9ca3af',
    style='italic')

# forward: student logits (from model) -> KD loss
fwd_arrow((mx + mw, 3.1), (lx, ly + lh - 1.05), color=BLUE, lw=2.4,
          conn="arc3,rad=-0.10")
txt(10.42, 2.92, "student\nlogits", size=11, color=BLUE, ha='center')

# forward: teacher -> KD loss
fwd_arrow((11.05 + 4.55 / 2, 7.50), (11.05 + 4.55 / 2, ly + lh), color=INK, lw=1.9)
txt(13.98, 7.16, "teacher logits", size=11, color='#374151', ha='center')

# forward: config / cost -> budget loss
fwd_arrow((cfg_x + cfg_w, cfg_y + 0.45), (lx + 0.2, ly + 0.70), color='#6b7280',
          lw=1.7, conn="arc3,rad=-0.22", style='-|>', mut=14)
txt(10.78, 5.95, "cost(c)", size=11, color='#6b7280', ha='center', style='italic')


# ================================================================================
# 5. BACKWARD  (gradients update model + router jointly)
# ================================================================================
# loss -> model weights (curve along the bottom)
fwd_arrow((lx + 0.4, ly), (mx + mw - 0.5, my + 0.12), color=RED, lw=2.4,
          style='-|>', mut=17, conn="arc3,rad=0.28")
txt(8.0, 0.22, r"backprop: update model weights  $\partial\mathcal{L}/\partial\theta$",
    size=12, weight='bold', color=RED, ha='center')
txt(8.0, 0.50, "model weights and router are trained together",
    size=10.5, color='#9ca3af', ha='center', style='italic')

# loss -> router : route up the corridor LEFT of the teacher box
fwd_arrow((lx + 0.10, ly + lh - 0.20), (9.92, 9.20), color=RED, lw=2.4,
          style='-|>', mut=17, conn="arc3,rad=-0.30")
txt(10.62, 8.82, r"update router  $\partial\mathcal{L}/\partial\phi$",
    size=11.5, weight='bold', color=RED, ha='center', rotation=68)
txt(11.05, 8.82, "(KD via Gumbel-Softmax + budget)", size=9.5,
    color='#ef4444', ha='center', style='italic', rotation=68)


# ---- title --------------------------------------------------------------------
txt(8.0, 10.78, "Elastic Training (Nemotron / Star Elastic): one model → many nested submodels",
    size=17, weight='bold', color=INK)

plt.tight_layout()
out = '/Users/yequan/Documents/Macro/Intern Plan/fig/elastic_training_concept.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("saved", out)
