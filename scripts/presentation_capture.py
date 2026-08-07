#!/usr/bin/env python
"""Capture the three midpoint-presentation experiments in one 30B model load.

Writes a single ``.npz`` consumed by ``scripts/presentation_plot.py``.  Heavy
(loads the full sharded Qwen3-30B-A3B), so run on the A100 via launch-on-a100.

**Exp A — why experts learn overlapping features** (slide 6).  Three orthogonal
views, on a handful of layers:

  A1 *weight-space overlap*: build a shared basis from the **other** E-1 experts'
     rows (leave-one-out eigenbasis of ``sum_{e'!=e} W_e'^T W_e'``) and measure
     how much of expert ``e``'s own Frobenius energy it already explains, vs
     rank ``r``.  A Gaussian control with matched per-expert norms gives the
     "no shared structure" reference curve.
  A2 *functional overlap*: run every expert on the same sampled tokens and take
     pairwise cosine similarity of their outputs (real weights vs a
     norm-matched Gaussian control).  Also the similarity restricted to the K
     experts a token actually routes to.
  A3 *router gives no specialization pressure*: per-expert mean of the hidden
     states routed to it (the "input centroid"); pairwise cosine similarity of
     those centroids says whether experts even see different token
     distributions.  Plus the routing load histogram (load-balancing loss keeps
     it near-uniform, so the gradient each expert receives is near-identical).

**Exp B — token x channel keep masks** (slide 8/17).  For a few heavily-routed
experts at one layer, the exact ``oracle_mag`` per-token keep mask over the
first tokens that route to them.  Different rows (tokens) light up different
columns (channels): the direct picture of why a fixed keep-set cannot work.

**Exp C — union coverage vs prefill length** (slide 8/17).  Replay a prefill and
track, per expert, the cumulative union of channels any token has activated.
Per token only rho of an expert's channels fire, but the union any *static*
resident set would have to hold climbs toward 1 within a few hundred tokens.

Scoring mirrors ``src/dynamic_active_param/block.py`` exactly for both
``oracle_mag`` (g*|inter|*||W_down[:,j]||) and ``oracle_up`` (g*|up|*||W_down||).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from src.base.datasets import load_datasets
from src.base.shared_utils.safe_isinstance import (
    _get_experts,
    _get_moe_block,
    _get_moe_intermediate_size,
    _get_num_hidden_layers,
    _get_topk,
)


def _build_moe_layer_map(model):
    """Ordered list of (layer_idx, moe_block) for every MoE layer."""
    pairs = []
    for layer_idx in range(_get_num_hidden_layers(model)):
        block = _get_moe_block(model, layer_idx)
        if _get_experts(block) is None:
            continue
        pairs.append((layer_idx, block))
    return pairs


def _eigh_desc(C):
    """Descending eigenbasis of a symmetric PSD matrix, computed on CPU.

    A 2048x2048 eigh is small, but these GPUs still hold 30B shards and cuSOLVER
    workspace allocation there is a known crash source, so keep linalg on CPU and
    move only the resulting basis back.
    """
    _, V = torch.linalg.eigh(C.double().cpu())
    return V.flip(-1).float().to(C.device)  # columns ordered by descending eigenvalue


def _loo_energy_curve(W_list, e, C_all, ranks):
    """Fraction of expert ``e``'s energy inside the top-r basis of the *others*.

    ``C_all = sum_e W_e^T W_e``; the leave-one-out Gram is a subtraction, so we
    pay one eigh per probed expert instead of rebuilding the stack.
    """
    We = W_list[e]                                       # (I, d) float32
    C_loo = C_all - We.T @ We
    V = _eigh_desc(C_loo)                                # (d, d)
    proj = We @ V                                        # (I, d) energy per basis dir
    per_dir = (proj * proj).sum(dim=0)                   # (d,)
    cum = torch.cumsum(per_dir, dim=0) / per_dir.sum().clamp_min(1e-30)
    return cum[ranks - 1].to("cpu", torch.float32).numpy()


def exp_a_weight_overlap(moe_pairs, layers_pos, ranks, n_probe, seed, out):
    """A1: leave-one-out shared-basis energy curves, real weights + Gaussian control."""
    gen = torch.Generator().manual_seed(seed)
    for mat in ("up_proj", "gate_proj"):
        real, ctrl = [], []
        for pos in layers_pos:
            _, block = moe_pairs[pos]
            experts = _get_experts(block)
            dev = next(block.parameters()).device
            W = [getattr(el, mat).weight.detach().float() for el in experts]   # (I,d)
            E = len(W)
            C_all = torch.zeros((W[0].shape[1],) * 2, dtype=torch.float32, device=dev)
            for We in W:
                C_all += We.T @ We
            probe = torch.randperm(E, generator=gen)[:n_probe].tolist()
            real.append(np.stack([_loo_energy_curve(W, e, C_all, ranks) for e in probe]))

            # Gaussian control: same shapes, matched per-expert Frobenius norms.
            norms = [float(We.norm()) for We in W]
            Wr = []
            for e in range(E):
                g = torch.randn(W[0].shape, generator=gen, dtype=torch.float32).to(dev)
                Wr.append(g * (norms[e] / float(g.norm())))
            C_r = torch.zeros_like(C_all)
            for We in Wr:
                C_r += We.T @ We
            ctrl.append(np.stack([_loo_energy_curve(Wr, e, C_r, ranks) for e in probe]))
            print(f"[expA1] {mat} layer_pos={pos} done", flush=True)
        out[f"a1_{mat}_real"] = np.stack(real)            # (n_layers, n_probe, n_ranks)
        out[f"a1_{mat}_ctrl"] = np.stack(ctrl)
    out["a1_ranks"] = ranks.cpu().numpy().astype(np.int32)


def _pairwise_cos_upper(Y):
    """Upper-triangular pairwise cosine similarities of rows of ``Y`` (n, d)."""
    Yn = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-30)
    S = Yn @ Yn.T
    iu = torch.triu_indices(S.shape[0], S.shape[0], offset=1, device=S.device)
    return S[iu[0], iu[1]]


def exp_a_functional(moe_pairs, layers_pos, tok_bank, seed, out):
    """A2 + A3: expert-output similarity, input-centroid similarity, routing load."""
    gen = torch.Generator().manual_seed(seed + 1)
    out_real, out_ctrl, out_routed, cent_cos, loads = [], [], [], [], []
    for pos in layers_pos:
        _, block = moe_pairs[pos]
        experts = _get_experts(block)
        dev = next(block.parameters()).device
        wdtype = next(block.parameters()).dtype
        x = tok_bank[pos].to(dev, torch.float32)         # (T, d) sampled hidden states
        xb = x.to(wdtype)                                # module inputs must match weights
        T = x.shape[0]
        E = len(experts)

        # every expert's output on every sampled token: (E, T, d)
        Y = torch.stack([el(xb).detach().float() for el in experts], dim=0)
        # mean-over-token pairwise cosine between experts (functional overlap)
        Ym = Y.reshape(E, -1)
        out_real.append(_pairwise_cos_upper(Ym).to("cpu", torch.float32).numpy())

        # Gaussian control: norm-matched random experts, same tokens.
        ctrl_rows = []
        for el in experts:
            ys = []
            for mat in ("gate_proj", "up_proj", "down_proj"):
                Wc = getattr(el, mat).weight.detach().float()
                g = torch.randn(Wc.shape, generator=gen, dtype=torch.float32).to(dev)
                ys.append(g * (float(Wc.norm()) / float(g.norm())))
            h = F.silu(x @ ys[0].T) * (x @ ys[1].T)
            ctrl_rows.append((h @ ys[2].T).reshape(-1))
        out_ctrl.append(_pairwise_cos_upper(torch.stack(ctrl_rows)).to("cpu", torch.float32).numpy())

        # A2b: similarity among the K experts a token actually routes to.
        logits = block.gate(xb)
        probs = F.softmax(logits, dim=1, dtype=torch.float)
        g, sel = torch.topk(probs, block.top_k, dim=-1)
        if getattr(block, "norm_topk_prob", False):
            g = g / g.sum(dim=-1, keepdim=True)
        per_tok = []
        for t in range(T):
            per_tok.append(_pairwise_cos_upper(Y[sel[t], t]).mean())
        out_routed.append(torch.stack(per_tok).to("cpu", torch.float32).numpy())

        # A3: per-expert input centroid (mean hidden state routed to it) + load.
        # Stored as a fixed (E,E) matrix with NaN for under-sampled experts, so
        # every layer has the same shape regardless of how many experts fired.
        onehot = torch.zeros((T, E), dtype=torch.float32, device=dev)
        onehot.scatter_(1, sel, 1.0)
        cnt = onehot.sum(dim=0)                          # (E,) routing load
        cent = (onehot.T @ x) / cnt.clamp_min(1.0).unsqueeze(1)
        cn = cent / cent.norm(dim=1, keepdim=True).clamp_min(1e-30)
        S = (cn @ cn.T).to("cpu", torch.float32).numpy()
        bad = (cnt < 2).to("cpu").numpy()
        S[bad, :] = np.nan
        S[:, bad] = np.nan
        np.fill_diagonal(S, np.nan)
        cent_cos.append(S)
        loads.append(cnt.to("cpu", torch.float32).numpy())
        print(f"[expA2] layer_pos={pos} T={T} done", flush=True)

    out["a2_out_cos_real"] = np.stack(out_real)
    out["a2_out_cos_ctrl"] = np.stack(out_ctrl)
    out["a2_routed_cos"] = np.stack(out_routed)
    out["a3_centroid_cos"] = np.stack(cent_cos)
    out["a3_load"] = np.stack(loads)


def exp_a_redundancy(moe_pairs, layers_pos, tok_bank, subset_sizes, seed, out):
    """A4 + A5: how much of an expert's *function* the other experts already cover.

    A4 *reconstruction R^2*: treat each expert's output on the shared token batch
    as a vector in R^{T*d} and least-squares fit probe expert ``e`` from a random
    subset of ``m`` other experts.  ``R^2`` is then literally "the fraction of
    this expert's function that already lives in the span of its peers".  A
    norm-matched Gaussian control gives the m/(T*d) chance level.

    A5 *substitution damage*: for each token, how wrong is it to serve the
    token's top-1 expert with a *different* expert?  Compared three ways — one of
    the token's own other routed experts, an expert the router did not pick, and
    a random-weight expert.  If the router truly specialised, its own picks would
    be the *least* interchangeable; the opposite is the redundancy signature.
    """
    gen = torch.Generator().manual_seed(seed + 7)
    r2_real, r2_ctrl, sub = [], [], []
    for pos in layers_pos:
        _, block = moe_pairs[pos]
        experts = _get_experts(block)
        dev = next(block.parameters()).device
        wdtype = next(block.parameters()).dtype
        x = tok_bank[pos].to(dev, torch.float32)
        xb = x.to(wdtype)
        T, dmod = x.shape
        E = len(experts)

        Y = torch.stack([el(xb).detach().float() for el in experts], 0)   # (E,T,d)
        Yf = Y.reshape(E, -1)                                            # (E, T*d)

        Yc = []
        for el in experts:
            ws = []
            for mat in ("gate_proj", "up_proj", "down_proj"):
                Wc = getattr(el, mat).weight.detach().float()
                g = torch.randn(Wc.shape, generator=gen, dtype=torch.float32).to(dev)
                ws.append(g * (float(Wc.norm()) / float(g.norm())))
            h = F.silu(x @ ws[0].T) * (x @ ws[1].T)
            Yc.append((h @ ws[2].T).reshape(-1))
        Ycf = torch.stack(Yc)

        def r2_curve(M):
            """Mean R^2 of reconstructing a probe expert from m random peers.

            Solved via normal equations on the E x E Gram matrix of expert
            outputs: the design matrix is (T*d, m) but only its Gram enters the
            solution, so one E x E product replaces hundreds of tall lstsq calls.
            """
            G = (M @ M.T).double().cpu()                     # (E, E) Gram
            vals = np.zeros((len(subset_sizes),), dtype=np.float32)
            probes = torch.randperm(E, generator=gen)[:8].tolist()
            for mi, m in enumerate(subset_sizes):
                acc = []
                for e in probes:
                    others = [q for q in range(E) if q != e]
                    pick = [others[j] for j in
                            torch.randperm(len(others), generator=gen)[:m].tolist()]
                    idx = torch.tensor(pick)
                    AtA = G[idx][:, idx]
                    Aty = G[idx, e]
                    yy = float(G[e, e])
                    if yy <= 1e-30:
                        acc.append(0.0)
                        continue
                    # ridge-stabilise: these Grams are near-singular at large m
                    lam = 1e-10 * float(torch.diagonal(AtA).mean())
                    c = torch.linalg.solve(
                        AtA + lam * torch.eye(m, dtype=AtA.dtype), Aty)
                    ss = yy - 2.0 * float(c @ Aty) + float(c @ AtA @ c)
                    acc.append(float(np.clip(1.0 - ss / yy, 0.0, 1.0)))
                vals[mi] = float(np.mean(acc))
            return vals

        r2_real.append(r2_curve(Yf))
        r2_ctrl.append(r2_curve(Ycf))

        # ---- A5 substitution damage ----------------------------------------
        probs = F.softmax(block.gate(xb), dim=1, dtype=torch.float)
        g, sel = torch.topk(probs, block.top_k, dim=-1)
        rows = np.zeros((3, T), dtype=np.float32)
        for t in range(T):
            e1 = int(sel[t, 0])
            y1 = Y[e1, t]
            den = max(float(y1.norm()), 1e-30)
            own = [int(q) for q in sel[t, 1:]]
            notr = [q for q in range(E) if q not in set(int(v) for v in sel[t])]
            pick_not = [notr[j] for j in
                        torch.randperm(len(notr), generator=gen)[:len(own)].tolist()]
            rows[0, t] = float(np.mean([float((Y[q, t] - y1).norm()) / den for q in own]))
            rows[1, t] = float(np.mean([float((Y[q, t] - y1).norm()) / den for q in pick_not]))
            rows[2, t] = float((Yc[e1].reshape(T, dmod)[t] - y1).norm()) / den
        sub.append(rows)
        print(f"[expA4] layer_pos={pos} done", flush=True)

    out["a4_r2_real"] = np.stack(r2_real)          # (n_layer, n_subset)
    out["a4_r2_ctrl"] = np.stack(r2_ctrl)
    out["a4_subset_sizes"] = np.array(subset_sizes, dtype=np.int32)
    out["a5_sub_damage"] = np.stack(sub)           # (n_layer, 3, T)


def exp_a_channel_twins(moe_pairs, layers_pos, n_probe, seed, out):
    """A6: is the overlap at *channel* granularity rather than expert granularity?

    For each channel of a probe expert (a row of ``up_proj``, i.e. the input
    direction that switches that channel on), find its nearest neighbour among
    **all channels of all other experts** by absolute cosine.  A high
    near-duplicate rate means "expert e's channel j" is a feature many experts
    also learned — overlapping features at channel granularity — even though the
    whole-expert output is not reconstructable (see A4).

    Reported against two references: a norm-matched Gaussian control (the chance
    level for this many candidate vectors in d dimensions) and the nearest
    neighbour *within* the same expert.
    """
    gen = torch.Generator().manual_seed(seed + 11)
    cross, ctrl, within = [], [], []
    for pos in layers_pos:
        _, block = moe_pairs[pos]
        experts = _get_experts(block)
        dev = next(block.parameters()).device
        W = torch.stack([el.up_proj.weight.detach().float() for el in experts])  # (E,I,d)
        E, I, dm = W.shape
        Wn = W / W.norm(dim=2, keepdim=True).clamp_min(1e-30)
        flat = Wn.reshape(E * I, dm)

        Wr = torch.randn(W.shape, generator=gen, dtype=torch.float32).to(dev)
        Wrn = Wr / Wr.norm(dim=2, keepdim=True).clamp_min(1e-30)
        flat_r = Wrn.reshape(E * I, dm)

        probes = torch.randperm(E, generator=gen)[:n_probe].tolist()
        cr, ct, wi = [], [], []
        for e in probes:
            for M, sink in ((flat, cr), (flat_r, ct)):
                S = (M[e * I:(e + 1) * I] @ M.T).abs()       # (I, E*I)
                S[:, e * I:(e + 1) * I] = -1.0               # exclude own expert
                sink.append(S.max(dim=1).values.to("cpu", torch.float32).numpy())
            Sw = (Wn[e] @ Wn[e].T).abs()
            Sw.fill_diagonal_(-1.0)
            wi.append(Sw.max(dim=1).values.to("cpu", torch.float32).numpy())
        cross.append(np.stack(cr)); ctrl.append(np.stack(ct)); within.append(np.stack(wi))
        print(f"[expA6] layer_pos={pos} done", flush=True)

    out["a6_nn_cross"] = np.stack(cross)          # (n_layer, n_probe, I)
    out["a6_nn_ctrl"] = np.stack(ctrl)
    out["a6_nn_within"] = np.stack(within)


def exp_a_channel_r2(moe_pairs, layers_pos, tok_bank, n_probe, pool, seed, out):
    """A7: reconstruct a *channel's* output contribution from other experts' channels.

    The channel-granularity twin of A4.  Channel ``(e,j)``'s contribution to the
    block output for token ``t`` is ``h_{e,j}(t) * W_down^{(e)}[:,j]``; we fit that
    (T*d)-vector from a random pool of channels drawn from *other* experts, and
    report R^2.  If channel-level R^2 is high where expert-level R^2 was ~0, the
    redundancy is real but only visible at channel granularity — which is exactly
    the claim that the effective expert unit is a channel.
    """
    gen = torch.Generator().manual_seed(seed + 13)
    r2, r2c = [], []
    for pos in layers_pos:
        _, block = moe_pairs[pos]
        experts = _get_experts(block)
        dev = next(block.parameters()).device
        wdtype = next(block.parameters()).dtype
        x = tok_bank[pos].to(dev, torch.float32)
        xb = x.to(wdtype)
        T = x.shape[0]
        E = len(experts)
        I = experts[0].down_proj.weight.shape[1]

        def chan_vecs(e, cols):
            """(len(cols), T*d) per-channel output contributions of expert e."""
            el = experts[e]
            h = (F.silu(el.gate_proj(xb)) * el.up_proj(xb)).float()      # (T,I)
            Wd = el.down_proj.weight.detach().float()                    # (d,I)
            # channel j's contribution is the outer product h[:,j] x Wd[:,j]
            return torch.stack([torch.outer(h[:, j], Wd[:, j]).reshape(-1)
                                for j in cols])

        probes = torch.randperm(E, generator=gen)[:n_probe].tolist()
        vals, valsc = [], []
        for e in probes:
            cols = torch.randperm(I, generator=gen)[:12].tolist()
            Y = chan_vecs(e, cols)                                       # (12, T*d)
            others = [q for q in range(E) if q != e]
            pick_e = [others[j] for j in
                      torch.randperm(len(others), generator=gen)[:pool].tolist()]
            A = torch.cat([chan_vecs(q, torch.randperm(I, generator=gen)[:4].tolist())
                           for q in pick_e], 0)                          # (4*pool, T*d)
            G = (A @ A.T).double().cpu()
            lam = 1e-8 * float(torch.diagonal(G).mean())
            G = G + lam * torch.eye(G.shape[0], dtype=G.dtype)
            for k in range(Y.shape[0]):
                y = Y[k]
                Aty = (A @ y).double().cpu()
                yy = float(y @ y)
                if yy <= 1e-30:
                    continue
                c = torch.linalg.solve(G, Aty)
                ss = yy - 2.0 * float(c @ Aty) + float(c @ G @ c)
                vals.append(float(np.clip(1.0 - ss / yy, 0.0, 1.0)))
            # control: same fit but against random-direction channels
            Ar = torch.randn(A.shape, generator=gen, dtype=torch.float32).to(dev)
            Ar = Ar * (A.norm(dim=1, keepdim=True) / Ar.norm(dim=1, keepdim=True))
            Gr = (Ar @ Ar.T).double().cpu()
            Gr = Gr + 1e-8 * float(torch.diagonal(Gr).mean()) * torch.eye(
                Gr.shape[0], dtype=Gr.dtype)
            for k in range(Y.shape[0]):
                y = Y[k]
                Aty = (Ar @ y).double().cpu()
                yy = float(y @ y)
                if yy <= 1e-30:
                    continue
                c = torch.linalg.solve(Gr, Aty)
                ss = yy - 2.0 * float(c @ Aty) + float(c @ Gr @ c)
                valsc.append(float(np.clip(1.0 - ss / yy, 0.0, 1.0)))
        r2.append(np.array(vals, dtype=np.float32))
        r2c.append(np.array(valsc, dtype=np.float32))
        print(f"[expA7] layer_pos={pos} n={len(vals)} done", flush=True)

    n = min(len(v) for v in r2)
    out["a7_chan_r2"] = np.stack([v[:n] for v in r2])
    nc = min(len(v) for v in r2c)
    out["a7_chan_r2_ctrl"] = np.stack([v[:nc] for v in r2c])
    out["a7_pool"] = np.array([pool * 4], dtype=np.int32)


def exp_a_centroid_control(moe_pairs, layers_pos, tok_bank, out):
    """Control for the centroid panel: cosine between individual token states.

    The residual stream has a dominant shared direction, so *any* two averages of
    it are somewhat aligned. This gives the "same token distribution" reference
    that the expert-centroid similarity must beat to count as specialisation.
    """
    ref = []
    for pos in layers_pos:
        x = tok_bank[pos].float()
        xn = x / x.norm(dim=1, keepdim=True).clamp_min(1e-30)
        S = xn @ xn.T
        iu = torch.triu_indices(S.shape[0], S.shape[0], offset=1)
        ref.append(S[iu[0], iu[1]].numpy())
    out["a3_token_cos"] = np.stack(ref)


def score_channels(block, x, col_norm, K, I, criterion):
    """Per-channel oracle score, mirroring ``block.py::_cross_expert_keep``.

    Returns ``(sel, flat_score)`` with ``sel`` (T,K) and ``flat_score`` (T,K*I).
    """
    dev = x.device
    assert block.top_k == K
    x = x.to(next(block.parameters()).dtype)
    probs = F.softmax(block.gate(x), dim=1, dtype=torch.float)
    g, sel = torch.topk(probs, block.top_k, dim=-1)
    if getattr(block, "norm_topk_prob", False):
        g = g / g.sum(dim=-1, keepdim=True)
    g = g.to(torch.float32)

    T = x.shape[0]
    sig = torch.zeros((T, K, I), dtype=torch.float32, device=dev)
    expert_mask = F.one_hot(sel, num_classes=block.num_experts).permute(2, 1, 0)
    hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for eid_t in hit:
        eid = int(eid_t)
        el = block.experts[eid]
        slot, top_x = torch.where(expert_mask[eid].squeeze(0))
        cur = x[top_x]
        if criterion == "oracle_up":
            h = el.up_proj(cur)
        else:
            h = el.act_fn(el.gate_proj(cur)) * el.up_proj(cur)
        sig[top_x, slot] = h.to(torch.float32)

    score = g.unsqueeze(-1) * sig.abs() * col_norm[sel]
    return sel, score.reshape(T, K * I)


def exp_bc_masks(moe_pairs, layers_pos, tok_seqs, ratios, n_mask_experts,
                 mask_tokens, ckpts, criteria, out):
    """Exp B (token x channel masks) + Exp C (union coverage vs prefill length)."""
    K = None
    for crit in criteria:
        union_all, per_tok_all, nseen_all = [], [], []
        for pos in layers_pos:
            _, block = moe_pairs[pos]
            experts = _get_experts(block)
            dev = next(block.parameters()).device
            I = experts[0].down_proj.weight.shape[1]
            K = block.top_k
            E = len(experts)
            col_norm = torch.stack(
                [el.down_proj.weight.detach().float().norm(dim=0) for el in experts]
            ).to(dev)

            # (n_ratio, n_seq, n_ckpt) mean union fraction over routed experts
            union = np.zeros((len(ratios), len(tok_seqs), len(ckpts)), dtype=np.float32)
            per_tok = np.zeros((len(ratios), len(tok_seqs)), dtype=np.float32)
            nseen = np.zeros((len(tok_seqs), len(ckpts)), dtype=np.float32)

            for si, xs in enumerate(tok_seqs):
                x = xs[pos].to(dev)
                sel, flat = score_channels(block, x, col_norm, K, I, crit)
                T = x.shape[0]
                for ri, rho in enumerate(ratios):
                    B = min(max(1, int(round(rho * K * I))), K * I)
                    idx = torch.topk(flat, B, dim=1, sorted=False).indices
                    keep = torch.zeros_like(flat, dtype=torch.bool)
                    keep.scatter_(1, idx, True)
                    keep = keep.reshape(T, K, I)
                    per_tok[ri, si] = keep.float().sum().item() / (T * K * I)

                    # cumulative union of channels each expert has ever activated
                    acc = torch.zeros((E, I), dtype=torch.uint8, device=dev)
                    routed = torch.zeros(E, dtype=torch.bool, device=dev)
                    ci, t0 = 0, 0
                    for ck in ckpts:
                        t1 = min(int(ck), T)
                        if t1 > t0:
                            s = sel[t0:t1].reshape(-1)
                            k = keep[t0:t1].reshape(-1, I).to(torch.uint8)
                            acc.index_reduce_(0, s, k, "amax")
                            routed[s] = True
                        frac = acc[routed].float().mean(dim=1)
                        union[ri, si, ci] = float(frac.mean()) if frac.numel() else 0.0
                        if ri == 0:
                            nseen[si, ci] = float(routed.sum())
                        ci += 1
                        t0 = t1

                    # Exp B: keep masks for the most-routed experts (first seq only)
                    if si == 0 and crit == "oracle_mag":
                        cnt = torch.bincount(sel.reshape(-1), minlength=E)
                        top_e = torch.topk(cnt, n_mask_experts).indices.tolist()
                        masks, tok_ids = [], []
                        for e in top_e:
                            hit_t, hit_k = torch.where(sel == e)
                            n = min(mask_tokens, hit_t.numel())
                            masks.append(
                                keep[hit_t[:n], hit_k[:n]].to("cpu", torch.uint8).numpy()
                            )
                            tok_ids.append(hit_t[:n].to("cpu").numpy())
                        out[f"b_mask_L{pos}_r{rho:.3f}"] = np.stack(masks)   # (n_e, n_tok, I)
                        out[f"b_tok_L{pos}_r{rho:.3f}"] = np.stack(tok_ids)
                        out[f"b_experts_L{pos}"] = np.array(top_e, dtype=np.int32)
                        out[f"b_route_L{pos}"] = cnt.to("cpu", torch.float32).numpy()
                print(f"[expBC] {crit} layer_pos={pos} seq={si} T={T} done", flush=True)

            union_all.append(union)
            per_tok_all.append(per_tok)
            nseen_all.append(nseen)
        out[f"c_union_{crit}"] = np.stack(union_all)          # (n_layer, n_rho, n_seq, n_ckpt)
        out[f"c_pertok_{crit}"] = np.stack(per_tok_all)
        out[f"c_nexperts_{crit}"] = np.stack(nseen_all)
    out["c_ckpts"] = np.array(ckpts, dtype=np.int32)
    out["c_ratios"] = np.array(ratios, dtype=np.float32)


def collect_hidden(model, moe_pairs, layers_pos, tok, texts, seq_len, in_dev):
    """Run texts through the model, returning each probed layer's MoE input.

    One dict per text: ``{layer_pos: (T, d) cpu float32}`` with padding stripped.
    """
    grab, store = {}, {}

    def make_hook(pos):
        def hook(module, inputs, kwargs):
            x = inputs[0] if inputs else kwargs.get("hidden_states")
            if x is not None and pos in grab:
                store[pos] = x.detach().reshape(-1, x.shape[-1]).float().cpu()
        return hook

    handles = [moe_pairs[pos][1].register_forward_pre_hook(make_hook(pos), with_kwargs=True)
               for pos in layers_pos]
    grab.update({p: True for p in layers_pos})

    outs = []
    with torch.no_grad():
        for txt in texts:
            enc = tok(txt, max_length=seq_len, truncation=True, return_tensors="pt")
            enc = {k: v.to(in_dev) for k, v in enc.items()}
            store.clear()
            model(**enc, use_cache=False)
            outs.append({p: store[p].clone() for p in layers_pos})
    for h in handles:
        h.remove()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--layers", default="1,24,46", help="model layer indices to probe")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--n-seqs", type=int, default=6, help="prefill sequences for Exp C")
    ap.add_argument("--doc-pool", type=int, default=4000,
                    help="#calib docs scanned to pick the longest --n-seqs prefills")
    ap.add_argument("--ratios", default="0.5,0.25,0.125")
    ap.add_argument("--probe-experts", type=int, default=12,
                    help="#experts probed per layer for the leave-one-out basis (Exp A1)")
    ap.add_argument("--rank-max", type=int, default=1024)
    ap.add_argument("--func-tokens", type=int, default=192, help="tokens for Exp A2/A3")
    ap.add_argument("--mask-experts", type=int, default=3)
    ap.add_argument("--mask-tokens", type=int, default=160)
    ap.add_argument("--criteria", default="oracle_mag,oracle_up")
    ap.add_argument("--dataset", default="c4")
    ap.add_argument("--out", default=os.path.join(_REPO, "docs/results/presentation/pres_exps.npz"))
    ap.add_argument("--per-gpu-mem", default=os.environ.get("PER_GPU_MEM", "34GiB"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
        max_memory={i: args.per_gpu_mem for i in range(torch.cuda.device_count())},
    )
    model.eval()

    K = _get_topk(model)
    I = _get_moe_intermediate_size(model)
    moe_pairs = _build_moe_layer_map(model)
    want = [int(v) for v in args.layers.split(",")]
    pos_of = {li: p for p, (li, _) in enumerate(moe_pairs)}
    layers_pos = [pos_of[li] for li in want if li in pos_of]
    ratios = [float(v) for v in args.ratios.split(",")]
    criteria = [c for c in args.criteria.split(",") if c]
    print(f"[cap] L={len(moe_pairs)} K={K} I={I} probing layers={want} pos={layers_pos}", flush=True)

    # Exp C needs genuinely long prefills; C4 validation is mostly short docs
    # (median ~250 tok), so scan a pool and keep the longest few.
    pool = load_datasets(args.dataset, tok, max_samples=args.doc_pool, max_length=args.seq_len)
    pool = [str(t) for t in pool if t]
    lens = [len(tok(t, truncation=True, max_length=args.seq_len).input_ids) for t in pool]
    order = np.argsort(lens)[::-1][: args.n_seqs]
    texts = [pool[i] for i in order]
    print(f"[cap] prefill doc lengths: {[lens[i] for i in order]}", flush=True)
    in_dev = model.get_input_embeddings().weight.device

    print("[cap] collecting hidden states", flush=True)
    tok_seqs = collect_hidden(model, moe_pairs, layers_pos, tok, texts, args.seq_len, in_dev)
    seq_T = min(v[layers_pos[0]].shape[0] for v in tok_seqs)
    ckpts = sorted(set(
        [int(round(v)) for v in np.unique(np.geomspace(1, seq_T, 40).round())] + [seq_T]
    ))
    print(f"[cap] seq_T={seq_T} ckpts={len(ckpts)}", flush=True)

    out = {}
    meta = dict(
        layer_indices=np.array(want, dtype=np.int32), K=K, I=I,
        E=len(_get_experts(moe_pairs[0][1])), seq_T=seq_T,
        n_seqs=len(tok_seqs), func_tokens=args.func_tokens, model=args.model,
    )

    def checkpoint(tag):
        """Save after each experiment so one late failure never costs the rest."""
        np.savez_compressed(args.out, **out, **meta)
        print(f"[cap] checkpoint after {tag} -> {args.out}", flush=True)

    # Exp A2/A3 reuse the first sequence's hidden states, subsampled.
    bank = {p: tok_seqs[0][p][: args.func_tokens] for p in layers_pos}
    ranks = torch.unique(torch.tensor(
        np.geomspace(1, args.rank_max, 32).round().astype(np.int64)))
    exp_a_weight_overlap(moe_pairs, layers_pos, ranks, args.probe_experts, 0, out)
    checkpoint("expA1")
    exp_a_functional(moe_pairs, layers_pos, bank, 0, out)
    exp_a_centroid_control(moe_pairs, layers_pos, bank, out)
    checkpoint("expA2/A3")
    subset_sizes = [int(v) for v in (1, 2, 4, 8, 16, 32, 64, 96, 127)]
    exp_a_redundancy(moe_pairs, layers_pos, bank, subset_sizes, 0, out)
    checkpoint("expA4/A5")
    # Channel granularity: the same two questions (weight twins, functional
    # reconstruction) asked of channels instead of whole experts.
    exp_a_channel_twins(moe_pairs, layers_pos, 6, 0, out)
    small = {p: bank[p][:64] for p in layers_pos}     # T*d vectors get large
    exp_a_channel_r2(moe_pairs, layers_pos, small, 4, 24, 0, out)
    checkpoint("expA6/A7")
    exp_bc_masks(moe_pairs, layers_pos, tok_seqs, ratios, args.mask_experts,
                 args.mask_tokens, ckpts, criteria, out)
    checkpoint("expB/C")
    print(f"[cap] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
