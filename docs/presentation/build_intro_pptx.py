#!/usr/bin/env python3
"""Build the broader-audience intro slides as a standalone deck.

Two new slides, same design system as ``build_pptx.py`` (navy/gold palette,
Times New Roman), meant to be shown *before* the midpoint deck:

  1. Dense Model vs Mixture-of-Experts — side-by-side block diagrams
     (sourced from the GShard figure on the Hugging Face MoE blog).
  2. The Challenge: MoE on Edge — a 3-tier memory hierarchy (DRAM / SRAM /
     compute) with a per-layer fetch-vs-compute latency comparison showing
     decode is memory-bound, then the two optimization challenges.

The original ``Yequan_26_Midpoint.pptx`` is NOT modified.
Output: ``Yequan_26_Intro.pptx``.

Run from ``docs/presentation``:  python build_intro_pptx.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
OUT = HERE / "Yequan_26_Intro.pptx"

# ---------------------------------------------------------------------------
# Palette (identical to build_pptx.py)
# ---------------------------------------------------------------------------
NAVY = RGBColor(0x0E, 0x2A, 0x47)
NAVY2 = RGBColor(0x14, 0x3A, 0x5E)
GOLD = RGBColor(0xE0, 0xA3, 0x2E)
TEAL = RGBColor(0x1C, 0x72, 0x93)
INK = RGBColor(0x22, 0x2A, 0x33)
MUTE = RGBColor(0x5C, 0x6B, 0x7A)
LIGHT = RGBColor(0xF4, 0xF6, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ROWALT = RGBColor(0xEA, 0xEF, 0xF4)
RED = RGBColor(0xB4, 0x32, 0x32)

FONT = "Times New Roman"

# tuples for PIL (RGB)
P_NAVY = (0x0E, 0x2A, 0x47)
P_NAVY2 = (0x14, 0x3A, 0x5E)
P_GOLD = (0xE0, 0xA3, 0x2E)
P_TEAL = (0x1C, 0x72, 0x93)
P_INK = (0x22, 0x2A, 0x33)
P_MUTE = (0x5C, 0x6B, 0x7A)
P_LIGHT = (0xF4, 0xF6, 0xF9)
P_WHITE = (0xFF, 0xFF, 0xFF)
P_RED = (0xB4, 0x32, 0x32)
P_GREEN = (0x2E, 0x8B, 0x3D)
P_LINE = (0xC7, 0xD1, 0xDB)

SW, SH = 13.333, 7.5
MARGIN = 0.6
CONTENT_TOP = 1.40
CONTENT_BOT = 6.82
CONTENT_L = MARGIN
CONTENT_R = SW - MARGIN
CONTENT_W = CONTENT_R - CONTENT_L

LOGO = FIGS / "amazon_logo.png"
LOGO_X, LOGO_Y, LOGO_W, LOGO_H = 0.21, 6.90, 1.60, 0.482

BODY_SZ = 20
TITLE_SZ = 30
SUBTITLE_SZ = 20

prs = Presentation()
prs.slide_width = Emu(int(SW * 914400))
prs.slide_height = Emu(int(SH * 914400))
BLANK = prs.slide_layouts[6]

_slide_no = 0


# ---------------------------------------------------------------------------
# Low-level helpers (identical to build_pptx.py)
# ---------------------------------------------------------------------------
def _set_font(run, size=BODY_SZ, bold=False, italic=False, color=INK, name=FONT):
    f = run.font
    f.name = name
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", name)


def add_bg(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def add_rect(slide, x, y, w, h, color, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(int(x * 914400)),
                                Emu(int(y * 914400)), Emu(int(w * 914400)),
                                Emu(int(h * 914400)))
    sp.fill.solid()
    sp.fill.fore_color.rgb = color
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(1)
    sp.shadow.inherit = False
    return sp


def add_text(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             wrap=True, space_after=6, line_spacing=1.0):
    tb = slide.shapes.add_textbox(Emu(int(x * 914400)), Emu(int(y * 914400)),
                                  Emu(int(w * 914400)), Emu(int(h * 914400)))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = Emu(int(0.05 * 914400))
    tf.margin_right = Emu(int(0.05 * 914400))
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.space_before = Pt(0)
        p.line_spacing = line_spacing
        if isinstance(para, str):
            para = [(para, {})]
        for text, kw in para:
            r = p.add_run()
            r.text = text
            _set_font(r, **kw)
    return tb


def add_logo(slide, x=LOGO_X):
    slide.shapes.add_picture(str(LOGO), Emu(int(x * 914400)),
                             Emu(int(LOGO_Y * 914400)), Emu(int(LOGO_W * 914400)),
                             Emu(int(LOGO_H * 914400)))


def add_footer(slide, title_short):
    global _slide_no
    _slide_no += 1
    add_logo(slide)
    add_rect(slide, SW - 1.15, SH - 0.5, 0.12, 0.26, GOLD)
    add_text(slide, SW - 0.95, SH - 0.54, 0.75, 0.35,
             [[(str(_slide_no), dict(size=12, color=MUTE, bold=True))]],
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)
    add_text(slide, 2.0, SH - 0.52, 9.0, 0.35,
             [[("Per-Token Adaptive Channel Activation for MoE", dict(size=11, color=MUTE, italic=True))]],
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)


def content_slide(title, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    add_bg(s, LIGHT)
    add_rect(s, MARGIN, 0.34, 0.14, 0.62, GOLD)
    add_text(s, MARGIN + 0.30, 0.22, SW - 2 * MARGIN - 0.30, 0.86,
             [[(title, dict(size=TITLE_SZ, bold=True, color=NAVY))]],
             anchor=MSO_ANCHOR.MIDDLE)
    top = CONTENT_TOP
    if subtitle:
        add_text(s, MARGIN, 1.18, CONTENT_W, 0.5,
                 [[(subtitle, dict(size=SUBTITLE_SZ, bold=True, color=TEAL))]],
                 anchor=MSO_ANCHOR.TOP)
        top = 1.82
    add_footer(s, title)
    return s, top


def bullets(slide, x, y, w, h, items, size=BODY_SZ, space_after=8, line_spacing=1.0):
    tb = slide.shapes.add_textbox(Emu(int(x * 914400)), Emu(int(y * 914400)),
                                  Emu(int(w * 914400)), Emu(int(h * 914400)))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.space_before = Pt(0)
        p.line_spacing = line_spacing
        if len(item) == 3:
            content, level, kw = item
        else:
            content, level = item
            kw = {}
        p.level = level
        pPr = p._p.get_or_add_pPr()
        buChar = pPr.makeelement(qn("a:buChar"), {"char": "▪" if level == 0 else "–"})
        buFont = pPr.makeelement(qn("a:buFont"), {"typeface": FONT})
        buClr = pPr.makeelement(qn("a:buClr"), {})
        srgb = buClr.makeelement(qn("a:srgbClr"), {"val": "E0A32E" if level == 0 else "1C7293"})
        buClr.append(srgb)
        pPr.append(buClr)
        pPr.append(buFont)
        pPr.append(buChar)
        indent = 0.28
        pPr.set("indent", str(-int(indent * 914400)))
        pPr.set("marL", str(int((indent + level * 0.3) * 914400)))
        runs = content if isinstance(content, list) else [(content, {})]
        for text, rkw in runs:
            r = p.add_run()
            r.text = text
            merged = dict(size=size, color=INK)
            merged.update(kw)
            merged.update(rkw)
            _set_font(r, **merged)
    return tb


def add_picture_hw(slide, png_path, x, y, w, h):
    slide.shapes.add_picture(str(png_path), Emu(int(x * 914400)),
                             Emu(int(y * 914400)), Emu(int(w * 914400)),
                             Emu(int(h * 914400)))


def add_figure(slide, png_path, y, max_w=CONTENT_W, max_h=3.5, center_x=None):
    im = Image.open(png_path)
    ar = im.width / im.height
    w = max_w
    h = w / ar
    if h > max_h:
        h = max_h
        w = h * ar
    x = (SW - w) / 2 if center_x is None else center_x
    slide.shapes.add_picture(str(png_path), Emu(int(x * 914400)),
                             Emu(int(y * 914400)), Emu(int(w * 914400)),
                             Emu(int(h * 914400)))
    return w, h


# ---------------------------------------------------------------------------
# Fonts for PIL figure generation
# ---------------------------------------------------------------------------
def _font(size, bold=False):
    names = (["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"])
    roots = [
        "/usr/share/fonts/truetype/dejavu/",
        "/usr/share/fonts/dejavu-sans-fonts/",
        "/usr/share/fonts/dejavu/",
        "/usr/share/fonts/gnu-free/",
    ]
    for r in roots:
        for n in names:
            p = Path(r) / n
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _autocrop(im, pad=8, bg=(255, 255, 255)):
    rgb = im.convert("RGB")
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, bg))
    bbox = diff.getbbox()
    if bbox:
        l, t, r, b = bbox
        return im.crop((max(0, l - pad), max(0, t - pad),
                        min(im.size[0], r + pad), min(im.size[1], b + pad)))
    return im


# ---------------------------------------------------------------------------
# Prepare the dense / MoE block diagrams from the downloaded GShard figure
# ---------------------------------------------------------------------------
def prepare_block_figures():
    src = FIGS / "web" / "02_moe_block.png"
    dense_out = FIGS / "fig_intro_dense_block.png"
    moe_out = FIGS / "fig_intro_moe_block.png"
    if not src.exists():
        print(f"  WARNING: {src} missing — run the download step first")
        return dense_out, moe_out
    im = Image.open(src).convert("RGB")
    dense = _autocrop(im.crop((30, 150, 228, 525)))
    moe = _autocrop(im.crop((300, 60, 488, 648)))
    dense.save(str(dense_out))
    moe.save(str(moe_out))
    print(f"  block figures: dense {dense.size}, moe {moe.size}")
    return dense_out, moe_out


# ---------------------------------------------------------------------------
# Memory-hierarchy + latency figure (drawn with PIL)
# ---------------------------------------------------------------------------
def draw_memory_hierarchy(path: Path):
    S = 2  # supersample for crisp text
    W, H = 1010 * S, 760 * S
    img = Image.new("RGB", (W, H), P_WHITE)
    d = ImageDraw.Draw(img)

    def F(sz, bold=False):
        return _font(sz * S, bold)

    def rr(x0, y0, x1, y1, r, **kw):
        d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=r * S, **kw)

    def tx(x, y, s, fill, fnt, anchor="lt"):
        d.text((x * S, y * S), s, fill=fill, font=fnt, anchor=anchor)

    def ln(x0, y0, x1, y1, fill, width=2):
        d.line([(x0 * S, y0 * S), (x1 * S, y1 * S)], fill=fill, width=width * S)

    cx = 420

    # ---- Section A: 3-tier hierarchy -------------------------------------
    tx(W / (2 * S), 14, "One MoE layer at decode  (batch = 1, per token)",
       P_NAVY, F(27, True), "mt")

    # tiers: compute (top, narrow) -> SRAM -> DRAM (bottom, widest)
    tiers = [
        # (y0, y1, half_width, fill, title, sub)
        (70, 132, 150, P_GOLD, "Compute cores", "does the math — 75.5 MFLOP / layer"),
        (168, 246, 250, P_TEAL, "SRAM / on-chip", "active params this token — K = 8 experts"),
        (282, 372, 360, P_NAVY, "DRAM (off-chip / host)", "TOTAL params live here — 30B  =  60 GB"),
    ]
    for y0, y1, hw, fill, title, sub in tiers:
        rr(cx - hw, y0, cx + hw, y1, 12, fill=fill)
        tw = "black" if fill == P_GOLD else P_WHITE
        subc = P_INK if fill == P_GOLD else (225, 233, 240)
        tx(cx, (y0 + y1) / 2 - 13, title, tw, F(20, True), "mm")
        tx(cx, (y0 + y1) / 2 + 12, sub, subc, F(15), "mm")

    # bandwidth arrows on the right edge of the pyramid
    ax = cx + 360 + 40
    # DRAM -> SRAM (the bottleneck)
    ln(ax, 300, ax, 210, P_RED, 5)
    d.polygon([((ax - 8) * S, 214 * S), ((ax + 8) * S, 214 * S), (ax * S, 200 * S)], fill=P_RED)
    tx(ax + 16, 232, "load weights", P_RED, F(16, True), "lm")
    tx(ax + 16, 254, "≈ 2 TB/s (HBM)", P_RED, F(15), "lm")
    tx(ax + 16, 276, "the bottleneck", P_RED, F(14, True), "lm")
    # SRAM -> compute (free)
    gx = ax - 110
    ln(gx, 190, gx, 120, P_GREEN, 4)
    d.polygon([((gx - 8) * S, 124 * S), ((gx + 8) * S, 124 * S), (gx * S, 110 * S)], fill=P_GREEN)
    tx(gx + 14, 148, "compute", P_GREEN, F(14, True), "lm")
    tx(gx + 14, 168, "(fast)", P_GREEN, F(13), "lm")

    # divider
    ln(40, 405, W / S - 40, 405, P_LINE, 2)

    # ---- Section B: fetch vs compute latency ------------------------------
    tx(40, 420, "Latency to process one layer:", P_NAVY, F(22, True), "lt")

    bx0 = 60
    bx1 = 820           # max bar right
    full = bx1 - bx0
    # Load bar (memory) — dominant
    ly = 462
    rr(bx0, ly, bx1, ly + 44, 8, fill=P_RED)
    tx((bx0 + bx1) / 2, ly + 22, "Load 75.5 MB  →  37.8 µs", P_WHITE, F(20, True), "mm")
    tx(bx1 + 14, ly + 22, "@ 2 TB/s", P_MUTE, F(15), "lm")

    # Compute bar — tiny (0.24 / 37.8 of full, floored to a visible sliver)
    cy = ly + 60
    comp_w = max(8, int(full * 0.24 / 37.8))
    rr(bx0, cy, bx0 + comp_w, cy + 44, 6, fill=P_GREEN)
    tx(bx0 + comp_w + 14, cy + 22,
       "Compute 75.5 MFLOP  →  0.24 µs   @ 312 TFLOPS", P_INK, F(20, True), "lm")

    # conclusion band
    by = cy + 66
    rr(40, by, W / S - 40, by + 58, 10, fill=P_NAVY)
    rr(40, by, 54, by + 58, 4, fill=P_GOLD)
    tx(72, by + 29,
       "Decode is  ~156×  memory-bound — compute is essentially free",
       P_GOLD, F(22, True), "lm")

    # edge note
    ey = by + 74
    tx(40, ey, "On edge (PCIe Gen4, 16 GB/s):", P_INK, F(18, True), "lt")
    tx(40, ey + 26,
       "loading the same 75.5 MB takes 4.7 ms/layer — the wall is bandwidth, not FLOPs.",
       P_INK, F(17), "lt")

    img = img.resize((W // S, H // S), Image.LANCZOS)
    img = _autocrop(img, pad=6)
    img.save(str(path))
    print(f"  memory hierarchy figure: {img.size}")


# ---------------------------------------------------------------------------
# Build figures
# ---------------------------------------------------------------------------
print("Preparing figures...")
DENSE_FIG, MOE_FIG = prepare_block_figures()
HIER_FIG = FIGS / "fig_intro_memory_hierarchy.png"
draw_memory_hierarchy(HIER_FIG)


# ===========================================================================
# SLIDE 1 — Dense vs MoE
# ===========================================================================
s, top = content_slide("Dense Model  vs  Mixture-of-Experts (MoE)",
                       "Two ways to build a transformer — MoE is why modern LLMs scale")

# Two block diagrams at a shared scale so the boxes are the same size.
dense_im = Image.open(DENSE_FIG)
moe_im = Image.open(MOE_FIG)
FIG_H = 3.25                                   # MoE (taller) height in inches
scale = FIG_H / moe_im.height                  # inches per source pixel (shared)
moe_w, moe_h = moe_im.width * scale, moe_im.height * scale
dense_w, dense_h = dense_im.width * scale, dense_im.height * scale

diag_top = top + 0.40
# Both diagrams sit near the centre with a "vs" chip between them; the
# descriptive text is pushed to the outer edges so nothing overlaps the chip.
lc = CONTENT_L + CONTENT_W * 0.34         # dense diagram centre
rc = CONTENT_L + CONTENT_W * 0.66         # MoE diagram centre
dense_x = lc - dense_w / 2
moe_x = rc - moe_w / 2
add_picture_hw(s, DENSE_FIG, dense_x, diag_top, dense_w, dense_h)
add_picture_hw(s, MOE_FIG, moe_x, diag_top, moe_w, moe_h)

# titles above each diagram (centred on the diagram, not the half)
add_text(s, lc - 2.2, top, 4.4, 0.4,
         [[("Dense Transformer", dict(size=20, bold=True, color=NAVY))]],
         align=PP_ALIGN.CENTER)
add_text(s, rc - 2.2, top, 4.4, 0.4,
         [[("Mixture-of-Experts", dict(size=20, bold=True, color=NAVY))]],
         align=PP_ALIGN.CENTER)

# "vs" chip centred in the gap between the two diagrams
vs_cx = (dense_x + dense_w + moe_x) / 2
add_rect(s, vs_cx - 0.32, diag_top + 1.3, 0.64, 0.5, GOLD)
add_text(s, vs_cx - 0.32, diag_top + 1.3, 0.64, 0.5,
         [[("vs", dict(size=22, bold=True, color=NAVY))]],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# dense annotation — far left, right-aligned so it hugs the dense diagram
add_text(s, CONTENT_L, diag_top + 0.55, dense_x - CONTENT_L - 0.2, 2.6,
         [[("Every token flows", dict(size=15, color=INK))],
          [("through the same", dict(size=15, color=INK))],
          [("single FFN.", dict(size=15, color=INK))],
          [("", dict(size=6))],
          [("100% of params", dict(size=15, bold=True, color=TEAL))],
          [("active per token.", dict(size=15, bold=True, color=TEAL))]],
         align=PP_ALIGN.RIGHT, space_after=2, line_spacing=1.05)

# MoE annotation — far right, left-aligned so it hugs the MoE diagram
add_text(s, moe_x + moe_w + 0.2, diag_top + 0.55,
         CONTENT_R - (moe_x + moe_w) - 0.2, 3.0,
         [[("A router sends each", dict(size=15, color=INK))],
          [("token to K of N", dict(size=15, color=INK))],
          [("expert FFNs.", dict(size=15, color=INK))],
          [("", dict(size=6))],
          [("Many experts,", dict(size=15, bold=True, color=TEAL))],
          [("few fire per token.", dict(size=15, bold=True, color=TEAL))]],
         align=PP_ALIGN.LEFT, space_after=2, line_spacing=1.05)

# bottom callout band
cb_y = diag_top + FIG_H + 0.16
add_rect(s, CONTENT_L, cb_y, CONTENT_W, 0.92, NAVY)
add_rect(s, CONTENT_L, cb_y, 0.14, 0.92, GOLD)
add_text(s, CONTENT_L + 0.35, cb_y + 0.09, CONTENT_W - 0.7, 0.76,
         [[("Dense:", dict(size=16, bold=True, color=GOLD)),
           ("  quality and cost both grow with parameter count — scaling gets expensive fast.", dict(size=16, color=WHITE))],
          [("MoE:", dict(size=16, bold=True, color=GOLD)),
           ("  quality scales with ", dict(size=16, color=WHITE)),
           ("total", dict(size=16, bold=True, italic=True, color=WHITE)),
           (" params, cost only with ", dict(size=16, color=WHITE)),
           ("active", dict(size=16, bold=True, italic=True, color=WHITE)),
           (" params — Qwen3-30B fires just 3B of 30B per token.", dict(size=16, color=WHITE))]],
         space_after=4, line_spacing=1.03)
# tiny source credit — in the gap between the callout band and the footer
add_text(s, SW - 5.8, cb_y + 0.96, 5.2, 0.22,
         [[("Diagram: GShard (Lepikhin et al.), via the Hugging Face MoE guide.",
            dict(size=9, italic=True, color=MUTE))]], align=PP_ALIGN.RIGHT)


# ===========================================================================
# SLIDE 2 — MoE on Edge: the memory-hierarchy challenge
# ===========================================================================
s, top = content_slide("The Challenge: Running MoE on Edge Devices",
                       "At decode, the bottleneck is memory bandwidth — not compute")

# LEFT: memory hierarchy + latency figure
fw, fh = add_figure(s, HIER_FIG, top + 0.02, max_w=6.9, max_h=4.75,
                    center_x=CONTENT_L - 0.05)

# RIGHT: explanation + two challenges
tx0 = CONTENT_L + fw + 0.2
tw = CONTENT_R - tx0

bullets(s, tx0, top + 0.02, tw, 2.0, [
    ([("Decoding is one token at a time.", dict(bold=True, color=NAVY)),
      (" Each token streams its experts from DRAM.", {})], 0),
    ([("Fetching weights dwarfs computing with them", dict(bold=True, color=TEAL)),
      (" — the GPU idles waiting on memory.", {})], 0),
    ([("On edge (phone, L4, Cor3),", dict(bold=True, color=NAVY)),
      (" DRAM is small and slow — both hurt.", {})], 0),
], size=15, space_after=7, line_spacing=1.03)

add_text(s, tx0, top + 1.95, tw, 0.32,
         [[("→ Two ways to make MoE fit and run on edge:",
            dict(size=16, bold=True, color=NAVY))]])

# Challenge 1 card
c1y = top + 2.36
c1h = 0.92
add_rect(s, tx0, c1y, tw, c1h, WHITE, line=RGBColor(0xD5, 0xDE, 0xE7))
add_rect(s, tx0, c1y, 0.12, c1h, GOLD)
add_text(s, tx0 + 0.28, c1y + 0.09, tw - 0.42, c1h - 0.15,
         [[("① Reduce ", dict(size=16, bold=True, color=NAVY)),
           ("total", dict(size=16, bold=True, italic=True, color=TEAL)),
           (" parameters", dict(size=16, bold=True, color=NAVY))],
          [("Shrink the model so it fits in device DRAM.", dict(size=13, color=INK))],
          [("30B = 60 GB, but edge DRAM is only 8–24 GB.", dict(size=12, color=MUTE))]],
         space_after=2, line_spacing=1.0)

# Challenge 2 card
c2y = c1y + c1h + 0.14
c2h = 0.92
add_rect(s, tx0, c2y, tw, c2h, WHITE, line=RGBColor(0xD5, 0xDE, 0xE7))
add_rect(s, tx0, c2y, 0.12, c2h, GOLD)
add_text(s, tx0 + 0.28, c2y + 0.09, tw - 0.42, c2h - 0.15,
         [[("② Reduce ", dict(size=16, bold=True, color=NAVY)),
           ("active", dict(size=16, bold=True, italic=True, color=TEAL)),
           (" parameters", dict(size=16, bold=True, color=NAVY))],
          [("Load fewer bytes DRAM→SRAM each token.", dict(size=13, color=INK))],
          [("Less bandwidth + less compute = lower latency.", dict(size=12, color=MUTE))]],
         space_after=2, line_spacing=1.0)

add_text(s, tx0, c2y + c2h + 0.06, tw, 0.3,
         [[("This work targets ", dict(size=13, color=INK)),
           ("both", dict(size=13, bold=True, color=NAVY)),
           (".", dict(size=13, color=INK))]])


# ===========================================================================
prs.save(str(OUT))
print(f"\nSaved {OUT} with {len(prs.slides)} slides")
