#!/usr/bin/env python
"""Plot the measured figures for the midpoint slides.

Consumes ``docs/results/presentation/pres_exps.npz`` (from
``scripts/presentation_capture.py``) plus, for the routing-load panel and the
keep-frequency spread, the 69,764-token ``docs/results/level2/oracle_mag_freq.npz``
and the weight-spectrum cache ``docs/exps/stats/figures/spectral_cache.npz``.

Writes to ``docs/presentation/figs/``:

  * ``fig_expert_overlap.pdf``   (slide 6) — why experts learn overlapping
    features: the router does not partition token space, so experts end up
    spanning a shared subspace, and the ones a token actually gets are the most
    redundant of all.
  * ``fig_leverage_spectrum.pdf`` (slide 7) — ridge-leverage spectra per expert
    by depth: a few channels carry the mass, the tail is near-zero.
  * ``fig_fixed_fails.pdf``      (slide 8/17) — why a fixed channel set cannot
    work: per-token keep masks for 3 experts + the union of channels any token
    activated, versus prefill length.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

INK = "#1c2330"
MUTED = "#68717f"
BLUE = "#2f6fdb"
AMBER = "#e08a1e"
PURPLE = "#7a4fc4"
GREEN = "#2e8b6f"
RED = "#c8402f"
GREY = "#9aa4b2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": "#cfd6e0", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 9, "axes.labelsize": 8.2,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "legend.fontsize": 7.2, "figure.titlesize": 10,
    "savefig.bbox": "tight", "axes.grid": True,
    "grid.color": "#eef1f6", "grid.linewidth": 0.7,
})
LAYER_C = {0: BLUE, 1: AMBER, 2: PURPLE}


def _clean(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)


def fig_expert_overlap(d, freq_npz, out_dir, stats):
    """Slide 6: what pre-training actually leaves behind — the measured story.

    The three panels are deliberately *not* all confirmations. (a) and (b) show
    the training procedure gives experts no reason to specialise and their weight
    subspaces do overlap; (c) shows the overlap nevertheless does NOT make whole
    experts interchangeable. That negative result is the argument for the method:
    the exploitable redundancy has to be looked for *inside* an expert, which is
    what the next slide measures.
    """
    L = d["layer_indices"]
    ranks = d["a1_ranks"].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.35))
    fig.subplots_adjust(wspace=0.42)

    # ---- (a) the router does not partition token space ---------------------
    ax = axes[0]
    ctr = np.concatenate([d["a3_centroid_cos"][i][np.isfinite(d["a3_centroid_cos"][i])]
                          for i in range(len(L))])
    tokc = np.concatenate([d["a3_token_cos"][i] for i in range(len(L))])
    ax.hist(tokc, bins=70, range=(-0.3, 1.0), density=True, histtype="stepfilled",
            color=GREY, alpha=0.40, lw=0,
            label="two individual tokens\n(reference: same distribution)")
    ax.hist(ctr, bins=70, range=(-0.3, 1.0), density=True, histtype="step",
            lw=2.0, color=RED, label="two experts' mean routed input")
    for v, c in ((float(tokc.mean()), MUTED), (float(ctr.mean()), RED)):
        ax.axvline(v, color=c, lw=1.2, ls="--")
    ylim = ax.get_ylim()[1]
    ax.text(float(ctr.mean()) + 0.025, ylim * 0.50, f"mean {ctr.mean():.2f}",
            fontsize=7.4, color=RED, weight="bold", ha="left")
    ax.text(float(tokc.mean()) - 0.025, ylim * 0.50, f"mean {tokc.mean():.2f}",
            fontsize=7.4, color=MUTED, ha="right", weight="bold")
    stats["centroid_cos_mean"] = float(ctr.mean())
    stats["token_cos_mean"] = float(tokc.mean())
    ax.set_xlabel("cosine similarity of inputs")
    ax.set_ylabel("density")
    ax.set_title("(a) Experts are fed the same kind of token", loc="left",
                 color=INK, weight="bold")
    ax.legend(frameon=False, loc="upper left")
    _clean(ax)

    # ---- (b) leave-one-out shared weight subspace -------------------------
    ax = axes[1]
    for i, li in enumerate(L):
        r = d["a1_up_proj_real"][i]
        ax.plot(ranks, r.mean(0), color=LAYER_C[i], lw=1.9, label=f"layer {li}")
        ax.fill_between(ranks, np.percentile(r, 25, 0), np.percentile(r, 75, 0),
                        color=LAYER_C[i], alpha=0.13, lw=0)
    ax.plot(ranks, d["a1_up_proj_ctrl"][0].mean(0), color=GREY, lw=1.7, ls="--",
            label="random-weight control")
    j = int(np.argmin(abs(ranks - 70)))
    rr = float(d["a1_up_proj_real"][0][:, j].mean())
    cc = float(d["a1_up_proj_ctrl"][0][:, j].mean())
    stats["a1_r70_real_L1"], stats["a1_r70_ctrl_L1"] = rr, cc
    ax.annotate(f"{rr / cc:.1f}× the chance\nlevel at rank {int(ranks[j])}",
                xy=(ranks[j], rr), xytext=(2.0, 0.42), fontsize=7.2, color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.9, color=INK,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_xscale("log")
    ax.set_xlabel("rank $r$ of a basis built from the OTHER 127 experts")
    ax.set_ylabel("fraction of an expert's\n$W_{up}$ energy explained")
    ax.set_title("(b) Their weights do share a subspace", loc="left",
                 color=INK, weight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, 1.0)
    _clean(ax)

    # ---- (c) …but whole experts are still not interchangeable -------------
    ax = axes[2]
    xs = np.arange(len(L))
    sub = d["a5_sub_damage"]                              # (n_layer, 3, T)
    own = sub[:, 0].mean(1); notr = sub[:, 1].mean(1); rnd = sub[:, 2].mean(1)
    w = 0.26
    ax.bar(xs - w, own, w, color=RED, label="another expert the router chose")
    ax.bar(xs, notr, w, color=AMBER, label="an expert it did not choose")
    ax.bar(xs + w, rnd, w, color=GREY, label="random-weight expert")
    ax.axhline(1.0, color=INK, lw=1.2, ls="--")
    ax.text(-0.42, 1.02, "100% = as bad as outputting zero",
            fontsize=6.6, color=INK, ha="left", va="bottom")
    for i in range(len(L)):
        stats[f"sub_own_L{L[i]}"] = float(own[i])
        stats[f"sub_random_L{L[i]}"] = float(rnd[i])
    r2 = d["a4_r2_real"][:, -1].mean()
    ax.text(0.5, 0.055,
            f"least-squares fit of a whole expert's output\n"
            f"from all 127 others: $R^2$ = {r2:.2f}",
            transform=ax.transAxes, fontsize=6.8, color=INK, va="bottom",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#f4f6fa",
                      edgecolor="#d9dfe8", lw=0.6))
    stats["a4_r2_all127_mean"] = float(r2)
    ax.set_xticks(xs); ax.set_xticklabels([f"layer {li}" for li in L])
    ax.set_ylabel("relative output error when a token is\nserved by a substitute expert")
    ax.set_ylim(0, 1.95)
    ax.set_title("(c) …yet no expert can stand in for another", loc="left",
                 color=INK, weight="bold")
    ax.legend(frameon=False, loc="upper right", ncol=1, fontsize=6.8)
    _clean(ax)

    fig.suptitle(
        "The router gives no specialisation pressure, but the resulting redundancy is "
        "NOT expert-level $\\Rightarrow$ look inside the expert   "
        "(Qwen3-30B-A3B, E=128, K=8)", y=1.05, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_expert_overlap.{ext}"), dpi=400)
    plt.close(fig)


def fig_channel_granularity(d, freq_npz, out_dir, stats):
    """Companion to slide 8: the redundancy that IS there is per-token, per-channel.

    Contrasts the three granularities measured: whole expert (not redundant),
    static channel set (not redundant either — the union grows), and per-token
    channel set (highly redundant — 7/8 of channels can go).
    """
    L = d["layer_indices"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.05))

    # (a) how much of a token's score mass a small channel prefix captures
    ax = axes[0]
    if freq_npz is not None:
        mass = freq_npz["topB_mass"]                     # (n_rho, L, T)
        ratios = freq_npz["ratios"]
        for ri, rho in enumerate(ratios):
            v = mass[ri].reshape(-1)
            ax.hist(v, bins=80, range=(0, 1), density=True, histtype="step",
                    lw=1.8, color=[BLUE, AMBER, PURPLE][ri],
                    label=f"top {rho * 100:.1f}% of channels")
            stats[f"topB_mass_median_r{rho:.3f}"] = float(np.median(v))
        ax.set_xlabel("fraction of a token's total channel-output magnitude captured")
        ax.set_ylabel("density")
        ax.set_title("(a) Per token, a few channels carry the output", loc="left",
                     color=INK, weight="bold")
        ax.legend(frameon=False, loc="upper left")
    _clean(ax)

    # (b) the three granularities side by side, each with the evidence for it
    ax = axes[1]
    ipos = len(L) // 2
    ri = int(np.argmin(abs(d["c_ratios"] - 0.125)))
    union = float(d["c_union_oracle_mag"][ipos, ri].mean(0)[-1])
    expert_r2 = float(d["a4_r2_real"][:, -1].mean())
    # The per-token bar is the measured oracle_mag operating point at rho=0.125
    # (HellaSwag acc_norm 76.84 vs dense 78.56); the other two are this run's
    # own measurements. Accuracy numbers come from the Level-2 sweep, not here.
    bars = [
        ("whole expert", expert_r2, GREY,
         "$R^2$ of fitting one\nexpert from 127"),
        ("static channel set", 1.0 - union, AMBER,
         "channels no token\ntouched in 2048"),
        ("per-token channel set", 0.875, BLUE,
         "measured: $-$1.7 pt\nHellaSwag"),
    ]
    for i, (lab, v, c, note) in enumerate(bars):
        ax.bar(i, v, 0.56, color=c)
        ax.text(i, v + 0.022, f"{v * 100:.0f}%", ha="center", fontsize=9.0,
                color=c, weight="bold")
        ax.text(i, v + 0.105, note, ha="center", fontsize=6.2, color=MUTED,
                linespacing=1.3, va="bottom")
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=7.4)
    ax.set_ylabel("fraction of an expert's channels\nthat can be skipped per token")
    ax.set_ylim(0, 1.16)
    ax.set_title("(b) Only per-token selection finds real slack", loc="left",
                 color=INK, weight="bold")
    stats["droppable_expert_level"] = expert_r2
    stats["droppable_static_channel"] = 1.0 - union
    stats["droppable_per_token"] = 0.875
    _clean(ax)

    fig.suptitle("Where the redundancy actually is: granularity decides",
                 y=1.05, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_channel_granularity.{ext}"), dpi=400)
    plt.close(fig)


def fig_load_balance(freq_npz, out_dir, stats):
    """Companion to slide 6: the aux loss does NOT deliver balanced routing.

    Uses the 69,764-token capture (192 tokens is far too few to say anything
    about routing load). The measured picture is strongly skewed — a long tail of
    barely-used experts and 66 that never fire at all on this calibration set —
    which is the "no load balancing" limitation stated on slide 6, and another
    reason whole-expert accounting is the wrong unit.
    """
    if freq_npz is None:
        return
    route = freq_npz["route"]                      # (L, E) route counts
    n_tok = int(freq_npz["n_tokens"]); K = int(freq_npz["K"])
    E = route.shape[1]
    frac = route / (n_tok * K)                     # share of all assignments
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.95))

    ax = axes[0]
    ideal = 1.0 / E
    for i, li in enumerate((1, 24, 46)):
        ax.hist(frac[li] / ideal, bins=50, range=(0, 3.0), histtype="step",
                lw=1.7, color=LAYER_C[i], label=f"layer {li}")
    ax.axvline(1.0, color=INK, lw=1.2, ls="--")
    ax.text(1.06, ax.get_ylim()[1] * 0.62, "perfect balance\nwould be here",
            fontsize=6.6, color=INK, linespacing=1.3)
    q = np.array([np.mean(frac[li] / ideal < 0.25) for li in (1, 24, 46)]).mean()
    ax.text(0.97, 0.42, f"{q * 100:.0f}% of experts get\n<¼ of a uniform share",
            transform=ax.transAxes, fontsize=6.8, color=RED, ha="right",
            weight="bold", linespacing=1.35)
    stats["frac_experts_under_quarter_share"] = float(q)
    ax.set_xlabel("expert load / uniform load")
    ax.set_ylabel("# experts")
    ax.set_title("(a) Routing load is strongly skewed", loc="left",
                 color=INK, weight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=7.0)
    _clean(ax)

    ax = axes[1]
    cv = (route.std(1) / route.mean(1))
    dead = (route == 0).sum(1)
    ax.plot(np.arange(route.shape[0]), cv, color=BLUE, lw=1.8)
    ax.axhline(0.0, color=GREY, lw=1.0, ls=":")
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("load coefficient of variation", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax.set_ylim(0, None)
    ax2 = ax.twinx()
    ax2.plot(np.arange(route.shape[0]), dead, color=AMBER, lw=1.5, ls="--")
    ax2.set_ylabel("# experts never routed", color=AMBER)
    ax2.tick_params(axis="y", colors=AMBER)
    ax2.grid(False)
    ax.set_title("(b) Skew and dead experts at every depth", loc="left",
                 color=INK, weight="bold")
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=1.8, label="load CV (0 = balanced)"),
                       Line2D([], [], color=AMBER, lw=1.5, ls="--",
                              label=f"never-routed experts ({int(dead.sum())} total)")],
              frameon=True, facecolor="white", edgecolor="#dfe4ec", framealpha=0.94,
              loc="lower center", fontsize=6.8)
    _clean(ax, ("top",))
    stats["load_cv_mean"] = float(cv.mean())
    stats["load_dead_total"] = int(dead.sum())

    fig.suptitle(f"Routing load over {n_tok:,} C4 tokens (E={E}, K={K}): the "
                 f"load-balancing loss does not equalise usage", y=1.06, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_load_balance.{ext}"), dpi=400)
    plt.close(fig)


def fig_leverage_spectrum(lev, out_dir, stats):
    """Slide 7: ridge-leverage spectra — a few channels carry the reconstruction.

    ``lev`` maps layer index -> (E, I) ridge-leverage array, i.e. the *measured*
    ``diag((C+λI)^{-1} C)`` from the C4 calibration run, not an SVD proxy.
    """
    if not lev:
        return
    layers = sorted(lev)
    show = [0, 15, 31, 47]
    show = [li for li in show if li in lev][:4]
    cols = {li: c for li, c in zip(show, (BLUE, AMBER, PURPLE, GREEN))}
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.2))
    fig.subplots_adjust(wspace=0.40)

    # ---- (a) sorted leverage curves, normalised per expert -----------------
    # Linear y with the head zoomed: the last few channels sit at ~1e-12, so a
    # log axis spends its whole range on a tail that carries no mass and flattens
    # the head structure that actually matters.
    ax = axes[0]
    for li in show:
        a = lev[li]
        srt = np.sort(a, axis=1)[:, ::-1]
        pi = srt / srt.sum(1, keepdims=True)
        med = np.median(pi, 0)
        ax.plot(np.arange(1, med.size + 1), med * 100, color=cols[li], lw=1.9,
                label=f"layer {li}")
        ax.fill_between(np.arange(1, med.size + 1),
                        np.percentile(pi, 10, 0) * 100,
                        np.percentile(pi, 90, 0) * 100,
                        color=cols[li], alpha=0.12, lw=0)
        stats[f"lev_top1_share_L{li}"] = float(med[0])
    ax.set_xscale("log")
    ax.set_xlim(1, lev[layers[0]].shape[1])
    ax.set_ylim(0, None)
    ax.set_xlabel("channel rank (descending)")
    ax.set_ylabel("share of an expert's total\nridge leverage  (%)")
    ax.set_title("(a) Early layers: a few channels carry it", loc="left",
                 color=INK, weight="bold")
    ax.legend(frameon=False, loc="upper right")
    _clean(ax)

    # ---- (b) mass captured by the top-r channels ---------------------------
    ax = axes[1]
    I = lev[layers[0]].shape[1]
    for li in show:
        a = lev[li]
        srt = np.sort(a, axis=1)[:, ::-1]
        pi = srt / srt.sum(1, keepdims=True)
        cum = np.cumsum(pi, axis=1).mean(0)
        ax.plot(np.arange(1, I + 1) / I, cum, color=cols[li], lw=1.9,
                label=f"layer {li}")
        stats[f"lev_top8pct_L{li}"] = float(cum[int(0.0833 * I) - 1])
    ax.axvline(0.125, color=RED, lw=1.3, ls="--")
    ax.text(0.135, 0.12, "top 12.5%\nof channels", fontsize=6.8, color=RED)
    ax.set_xlabel("fraction of channels kept (highest leverage first)")
    ax.set_ylabel("fraction of leverage mass captured")
    ax.set_ylim(0, 1.02)
    ax.set_title("(b) …so most of the mass is in a small prefix", loc="left",
                 color=INK, weight="bold")
    ax.legend(frameon=False, loc="lower right")
    _clean(ax)

    # ---- (c) effective rank vs depth --------------------------------------
    ax = axes[2]
    er, p10, p90 = [], [], []
    for li in layers:
        a = lev[li]
        pi = a / a.sum(1, keepdims=True).clip(1e-30)
        e = np.exp(-(pi * np.log(pi + 1e-30)).sum(1))
        er.append(e.mean()); p10.append(np.percentile(e, 10)); p90.append(np.percentile(e, 90))
    er = np.array(er)
    ax.plot(layers, er, color=BLUE, lw=1.9)
    ax.fill_between(layers, p10, p90, color=BLUE, alpha=0.15, lw=0)
    ax.axhline(I, color=INK, lw=1.1, ls="--")
    ax.text(len(layers) * 0.42, I * 0.965, f"full width I = {I}", fontsize=6.8,
            color=INK, va="top")
    lo = int(np.argmin(er))
    ax.annotate(f"L{layers[lo]}: {er[lo]:.0f}", xy=(layers[lo], er[lo]),
                xytext=(layers[lo] + 5, er[lo] - 60), fontsize=7.2, color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.9, color=INK))
    stats["erank_min_layer"] = int(layers[lo])
    stats["erank_min"] = float(er[lo])
    stats["erank_max"] = float(er.max())
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("effective # of load-bearing\nchannels per expert")
    ax.set_ylim(0, I * 1.06)
    ax.set_title("(c) Early layers compress most", loc="left",
                 color=INK, weight="bold")
    _clean(ax)

    fig.suptitle("Ridge leverage of expert channels (C4 calibration, "
                 "Qwen3-30B-A3B): the load is unevenly spread across channels",
                 y=1.06, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_leverage_spectrum.{ext}"), dpi=400)
    plt.close(fig)


def fig_fixed_fails(d, freq_npz, out_dir, stats, rho=0.125, layer=24):
    """Slide 8/17: per-token masks + the union any static keep-set would need."""
    L = list(d["layer_indices"])
    li = layer if layer in L else L[len(L) // 2]
    masks = d[f"b_mask_L{li}_r{rho:.3f}"]            # (n_e, n_tok, I)
    experts = d[f"b_experts_L{li}"]
    route = d[f"b_route_L{li}"]
    ne, ntok, I = masks.shape
    ck = d["c_ckpts"].astype(float)
    rr = d["c_ratios"]
    ri = int(np.argmin(abs(rr - rho)))
    ipos = L.index(li)

    fig = plt.figure(figsize=(11.4, 3.5))
    gs = fig.add_gridspec(1, ne + 2, width_ratios=[1] * ne + [0.18, 1.45],
                          wspace=0.30)

    # ---- (a) three experts: rows = tokens, cols = channels ----------------
    NCH_SHOW = 220                                   # a readable channel window
    for k in range(ne):
        ax = fig.add_subplot(gs[0, k])
        m = masks[k][:, :NCH_SHOW]
        ax.imshow(m, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(f"expert {int(experts[k])}", fontsize=8, color=INK, pad=3)
        ax.set_xlabel("channel", labelpad=1.5)
        if k == 0:
            ax.set_ylabel(f"token (in arrival order)")
        else:
            ax.set_yticklabels([])
        ax.grid(False)
        ax.tick_params(labelsize=6.4)
        # per-expert Jaccard between consecutive tokens' keep sets
        a, b = masks[k][:-1].astype(bool), masks[k][1:].astype(bool)
        jac = (a & b).sum(1) / np.maximum((a | b).sum(1), 1)
        stats[f"jaccard_L{li}_e{int(experts[k])}_r{rho}"] = float(jac.mean())
        ax.text(0.5, -0.20, f"consecutive-token overlap {jac.mean() * 100:.0f}%",
                transform=ax.transAxes, ha="center", fontsize=6.6, color=MUTED)

    fig.text(0.090, 1.00,
             f"(a) Which channels a token keeps is re-decided every token "
             f"— layer {li}, budget $\\rho$={rho:g}\n"
             f"      (rows = consecutive tokens routed to that expert; "
             f"first {NCH_SHOW} of I={I} channels)",
             fontsize=8.4, color=INK, weight="bold", ha="left", va="bottom",
             linespacing=1.5)

    # ---- (b) union coverage vs prefill length -----------------------------
    ax = fig.add_subplot(gs[0, ne + 1])
    for i, lay in enumerate(L):
        u = d["c_union_oracle_mag"][i, ri]           # (n_seq, n_ckpt)
        ax.plot(ck, u.mean(0), color=LAYER_C[i], lw=1.9, label=f"layer {lay}")
        ax.fill_between(ck, u.min(0), u.max(0), color=LAYER_C[i], alpha=0.13, lw=0)
    ax.axhline(rho, color=RED, lw=1.4, ls="--")
    ax.text(1.15, rho - 0.030, f"any ONE token needs only {rho:g}",
            fontsize=7.2, color=RED, weight="bold", va="top")
    u_end = float(d["c_union_oracle_mag"][ipos, ri].mean(0)[-1])
    stats[f"union_r{rho}_L{li}_T{int(ck[-1])}"] = u_end
    ax.annotate(f"but {u_end * 100:.0f}% after {int(ck[-1])} tokens",
                xy=(ck[-1], u_end), xytext=(ck[-1] * 0.030, u_end + 0.15),
                fontsize=7.2, color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.9, color=INK))
    ax.set_xscale("log")
    ax.set_xlabel("tokens seen (prefill length)")
    ax.set_ylabel("fraction of an expert's channels\never activated (running union)")
    ax.set_ylim(0, 1.04)
    ax.set_title("(b) A static keep-set must hold nearly everything",
                 loc="left", color=INK, weight="bold", pad=6)
    ax.legend(frameon=False, loc="center right", fontsize=7.0)
    _clean(ax)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_fixed_fails.{ext}"), dpi=400)
    plt.close(fig)


def fig_union_budgets(d, out_dir, stats):
    """Companion: union coverage at all three budgets, both criteria."""
    L = list(d["layer_indices"])
    ck = d["c_ckpts"].astype(float)
    rr = d["c_ratios"]
    fig, axes = plt.subplots(1, len(rr), figsize=(10.4, 2.95), sharey=True)
    for ri, rho in enumerate(rr):
        ax = axes[ri]
        for i, lay in enumerate(L):
            ax.plot(ck, d["c_union_oracle_mag"][i, ri].mean(0), color=LAYER_C[i],
                    lw=1.8, label=f"layer {lay}")
        ax.axhline(float(rho), color=RED, lw=1.3, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("tokens seen")
        ax.set_title(f"budget $\\rho$ = {rho:g}", loc="left", color=INK,
                     weight="bold")
        if ri == 0:
            ax.set_ylabel("union of activated channels")
            ax.legend(frameon=False, loc="lower right")
        ax.set_ylim(0, 1.0)
        _clean(ax)
    fig.suptitle("Per-token sparsity does not become static sparsity: the union "
                 "grows with every token", y=1.06, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_union_budgets.{ext}"), dpi=400)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--npz", default=os.path.join(repo, "docs/results/presentation/pres_exps.npz"))
    ap.add_argument("--freq-npz", default=os.path.join(repo, "docs/results/level2/oracle_mag_freq.npz"))
    ap.add_argument("--scores", default=os.path.join(
        repo, "docs/results/presentation/expert_scores_50p.pth"),
        help="expert_scores.pth carrying the measured 'leverage' tensors")
    ap.add_argument("--out-dir", default=os.path.join(repo, "docs/presentation/figs"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(args.npz)
    freq = np.load(args.freq_npz) if os.path.exists(args.freq_npz) else None

    lev = {}
    if os.path.exists(args.scores):
        import torch
        sc = torch.load(args.scores, map_location="cpu")
        for li, t in sc.get("leverage", {}).items():
            lev[int(li)] = t.float().numpy()

    stats = {}
    fig_expert_overlap(d, freq, args.out_dir, stats)
    fig_channel_granularity(d, freq, args.out_dir, stats)
    fig_load_balance(freq, args.out_dir, stats)
    fig_leverage_spectrum(lev, args.out_dir, stats)
    fig_fixed_fails(d, freq, args.out_dir, stats)
    fig_union_budgets(d, args.out_dir, stats)

    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(f"[plot] wrote figures + stats.json to {args.out_dir}")
    for k, v in sorted(stats.items()):
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
