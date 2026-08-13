#!/usr/bin/env python
"""Scorer frontier at a **hard byte budget**: can a proxy costing <10% of one
expert matrix rank channels as well as the exact SwiGLU intermediate?

Target (frozen). `oracle_mag_noW` ranks a token's pooled `K*I` channels by
``g_e*|SiLU(gate)*up|`` and keeps the global top-B; at rho=0.125 it scores 77.11
HellaSwag acc_norm (dense 78.56) but needs full-width gate+up just to decide, so
the realized whole-FFN active cut is only -29.2%. The goal here is a scorer whose
**per-token bytes are <= 0.10 of one expert (I,H) matrix** and whose selection is
accuracy-equivalent, which would make the realized cut -84.2%.

Where the previous pilot (``idea_pilot_scorers.py``) left the frontier, averaged
over layers 6/22/38/46 at rho=0.125:

    quant_w4        cB 0.500   recall 0.915   <- accuracy-equivalent, 5x over budget
    insp0.5_q4      cB 0.250   recall 0.819
    insp0.25_q4     cB 0.125   recall 0.663   <- current cheapest usable point
    quant_w2        cB 0.250   recall 0.506   (bits alone is the *expensive* axis)
    oracle_up       cB 1.000   recall 0.564   (-> 71.30 HS, the deployable baseline)

So the open problem is a **5x byte reduction at iso-recall 0.9**, i.e. an
effective 0.8 bits/weight for gate+up. RTN cannot go there; this script tests
representation models that can, plus the two allocation levers that are free.

Variant groups (``--groups``):

  ``ref``    reproduce known points (oracle_up, quant_w4, insp0.25_q4) so this
             harness is comparable to ``docs/results/idea_pilot/pilot_scorers.json``.
  ``rtn``    the bits x input-keep grid extended down to cB<=0.1.
  ``inspw``  **norm-weighted input sparsification.** Plain Prox-style stage 1 keeps
             the largest ``|x_i|``; but coordinate i moves the score by
             ``x_i * w[j,i]``, so the right currency is ``|x_i| * rms_j(w[.,i])``.
             Offline column stats, free at run time.
  ``asym``   **asymmetric gate/up allocation.** The score is a product pushed
             through SiLU, so the two branches need not carry equal precision.
  ``pq``     **product quantization.** Splits H into S=H/d subspaces and codes each
             row's subvector by one of C centroids: ``log2(C)/d`` bits/weight, which
             reaches 0.25-1.0 bits/weight where RTN bottoms out at ~2. Inner products
             become S table lookups against a per-token LUT built once per *layer*
             and shared by all K co-activated experts and all I rows -- the
             amortization is an MoE-native win. Unlike the dead spectral/basis
             family this is not a rank truncation: every row keeps its own code
             word per subspace, so fine row structure survives.
  ``pqinsp`` PQ composed with **subspace-structured** input sparsity (keep the top
             fraction of subspaces by ``||x_s||``), which is the form of input
             sparsity that actually removes PQ bytes -- an unstructured coordinate
             mask cannot skip a code that spans d coordinates.
  ``had``    **Hadamard-rotated RTN/PQ.** An orthonormal FWHT on H (folded into the
             weights offline, ~H log H per token online) gaussianizes weight groups
             and is what makes 2-3 bit quantization work elsewhere. It necessarily
             *destroys* input sparsity, so it competes with ``rtn``/``inspw``
             rather than composing.
  ``sign``   1-bit sign x per-group scale, the RTN floor.
  ``radapt`` **router-adaptive precision.** Score is ``g_e * |...|``, so channels of
             low-``g_e`` experts rarely reach the top-B; spend bits per expert by
             router rank instead of uniformly. Free, MoE-only.

Diagnostics (``--diag``): the share of the oracle top-B contributed by each router
rank (is the ``radapt`` lever real?), row clusterability (can coarse-to-fine bounds
work?), and PQ reconstruction error vs bits.

Metrics mirror ``lowrank_scorer_recall.py`` / ``idea_pilot_scorers.py`` exactly
(recall/mass/spearman at budget, plus the relaxed-candidate cascade), so numbers
are directly comparable. Each row also carries ``total_bytes``, the honest
end-to-end quantity: ``c_proxy + 2*lam*rho + rho`` in units of one expert matrix
against a dense expert's 3.

Reads the cached captures from ``lowrank_scorer_recall.py``; one GPU, no model load.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts.idea_pilot_scorers import _route, _spearman, quantize_rtn


# ==========================================================================
# representation models
# ==========================================================================

def rtn_bits_per_weight(bits, group, scale_bits=16):
    """Honest bits/weight for group-wise RTN: payload + the fp16 group scale."""
    return bits + scale_bits / group


def sign_quant(W, group=128):
    """1-bit sign with a per-group mean-|w| scale (the RTN floor)."""
    E, I, H = W.shape
    g = group if group and H % group == 0 else H
    Wg_ = W.reshape(E, I, H // g, g)
    scale = Wg_.abs().mean(dim=-1, keepdim=True)
    return (torch.sign(Wg_) * scale).reshape(E, I, H)


def fwht(x, chunk=None):
    """Orthonormal fast Walsh-Hadamard transform along the last axis (power of 2)."""
    n = x.shape[-1]
    assert n & (n - 1) == 0, "FWHT needs a power-of-two length"
    if chunk is not None and x.dim() > 1 and x.shape[0] > chunk:
        return torch.cat([fwht(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)], 0)
    lead = x.shape[:-1]
    y = x.clone()
    h = 1
    while h < n:
        y = y.reshape(*lead, n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2)
        h *= 2
    return y.reshape(*lead, n) / (n ** 0.5)


def fit_pq(W, subdim, C, iters=20, sample=8192, seed=0, budget=2 ** 26):
    """Product-quantize ``W (E,I,H)`` along H.

    One codebook per subspace, **shared across all experts and rows** (E*I ~ 1e5
    subvectors in d dims, so the clusters are well populated). Returns the
    dequantized proxy and the relative Frobenius error. Cost model:
    ``log2(C)/subdim`` bits per weight; the codebook itself is
    ``S*C*subdim`` scalars, i.e. ``C*16/(E*I)`` bits/weight amortized (~0.003).

    k-means runs on all subspaces simultaneously (they are independent and
    identically shaped), which is what keeps this a few seconds per matrix.
    """
    E, I, H = W.shape
    assert H % subdim == 0, f"subdim {subdim} must divide H {H}"
    S, N = H // subdim, E * I
    V = W.reshape(N, S, subdim)
    rows = max(256, min(N, budget // max(1, S * C)))

    gcpu = torch.Generator().manual_seed(seed)
    sidx = torch.randperm(N, generator=gcpu)[:min(sample, N)].to(W.device)
    Vs = V[sidx]                                              # (n,S,d)
    init = torch.randperm(Vs.shape[0], generator=gcpu)[:C].to(W.device)
    cb = Vs[init].permute(1, 0, 2).contiguous()               # (S,C,d)

    vsq = (Vs * Vs).sum(-1, keepdim=True)
    for _ in range(iters):
        d2 = vsq - 2 * torch.einsum("nsd,scd->nsc", Vs, cb) + (cb * cb).sum(-1)[None]
        a = d2.argmin(-1)                                     # (n,S)
        at = a.t()                                            # (S,n)
        num = torch.zeros_like(cb).scatter_add_(
            1, at.unsqueeze(-1).expand(-1, -1, subdim), Vs.permute(1, 0, 2))
        den = torch.zeros(S, C, device=W.device, dtype=W.dtype).scatter_add_(
            1, at, torch.ones_like(at, dtype=W.dtype))
        cb = torch.where(den.unsqueeze(-1) > 0, num / den.clamp_min(1).unsqueeze(-1), cb)

    cbsq = (cb * cb).sum(-1)[None]
    Wq = torch.empty_like(W).reshape(N, S, subdim)
    sr = torch.arange(S, device=W.device)
    err = num_el = 0.0
    for s0 in range(0, N, rows):
        Vc = V[s0:s0 + rows]
        code = (cbsq - 2 * torch.einsum("nsd,scd->nsc", Vc, cb)).argmin(-1)
        rec = cb[sr, code]
        Wq[s0:s0 + rows] = rec
        err += float(((rec - Vc) ** 2).sum())
        num_el += float((Vc ** 2).sum())
    return Wq.reshape(E, I, H), (err / max(num_el, 1e-30)) ** 0.5


# ==========================================================================
# input sparsification
# ==========================================================================

def topk_mask_input(x, keep_frac, weight=None):
    """Keep the ``keep_frac`` largest entries of x by ``|x_i|*weight_i``, zero rest.

    ``weight=None`` is the plain Prox criterion. A column-statistic weight makes
    the criterion match what actually perturbs the *ranking* of channels.
    """
    if keep_frac >= 1.0:
        return x
    k = max(1, int(round(keep_frac * x.shape[-1])))
    crit = x.abs() if weight is None else x.abs() * weight
    idx = crit.topk(k, dim=-1).indices
    out = torch.zeros_like(x)
    return out.scatter_(-1, idx, x.gather(-1, idx))


def subspace_sparsify(x, keep_frac, subdim):
    """Zero all but the ``keep_frac`` highest-energy subspaces of width ``subdim``.

    The PQ-compatible form of input sparsity: a code word spans ``subdim``
    coordinates, so bytes are only saved when whole subspaces are skipped.
    """
    if keep_frac >= 1.0:
        return x
    t, H = x.shape
    S = H // subdim
    xs = x.reshape(t, S, subdim)
    k = max(1, int(round(keep_frac * S)))
    keep = xs.norm(dim=-1).topk(k, dim=-1).indices                   # (t,k)
    m = torch.zeros(t, S, device=x.device, dtype=x.dtype).scatter_(1, keep, 1.0)
    return (xs * m.unsqueeze(-1)).reshape(t, H)


# ==========================================================================
# variant construction
# ==========================================================================

class Variant:
    """A scorer: proxy weights for up/gate, an input transform, and a cost.

    ``cost_bytes``/``cost_flops`` are per-token, in units of one full-width
    ``(I,H)`` expert matrix (a dense expert FFN is 3). ``rank`` optionally holds a
    per-expert bit schedule for router-adaptive precision.
    """

    def __init__(self, name, wu, wg, cost_bytes, cost_flops, kind,
                 in_mode=None, in_keep=1.0, in_weight=None, subdim=None,
                 rotate=False, extra=None, per_rank=None):
        self.name, self.wu, self.wg = name, wu, wg
        self.cost_bytes, self.cost_flops, self.kind = cost_bytes, cost_flops, kind
        self.in_mode, self.in_keep, self.in_weight = in_mode, in_keep, in_weight
        self.subdim, self.rotate = subdim, rotate
        self.extra = extra or {}
        self.per_rank = per_rank

    def prepare_input(self, x):
        if self.rotate:
            x = fwht(x)
        if self.in_mode == "topk":
            return topk_mask_input(x, self.in_keep, self.in_weight)
        if self.in_mode == "subspace":
            return subspace_sparsify(x, self.in_keep, self.subdim)
        return x

    def meta(self):
        return dict(kind=self.kind, cost_bytes=self.cost_bytes,
                    cost_flops=self.cost_flops, use_gate=self.wg is not None,
                    input_keep=self.in_keep, input_mode=self.in_mode or "dense",
                    rotate=self.rotate, **self.extra)


def build_variants(Wu, Wg, groups, args, log):
    """Yield ``Variant``s lazily so only one proxy pair is resident at a time."""
    H = Wu.shape[-1]
    # offline column statistics for the norm-weighted input criterion
    cw = (Wu.pow(2).mean(dim=(0, 1)) + Wg.pow(2).mean(dim=(0, 1))).sqrt()
    cw = cw / cw.mean()

    def rtn_pair(bits, group):
        return quantize_rtn(Wu, bits, group), quantize_rtn(Wg, bits, group)

    def cb_rtn(bits, group, keep=1.0, both=True):
        bpw = rtn_bits_per_weight(bits, group) / 16.0
        return (2 if both else 1) * bpw * keep

    # ---- ref: reproduce the published points ------------------------------
    if "ref" in groups:
        qu, qg = rtn_pair(4, args.qgroup)
        yield Variant("quant_w4", qu, qg, cb_rtn(4, args.qgroup), 2.0, "rtn")
        yield Variant("insp0.25_q4", qu, qg, cb_rtn(4, args.qgroup, 0.25), 0.5,
                      "rtn+insp", in_mode="topk", in_keep=0.25)
        del qu, qg
        torch.cuda.empty_cache()

    # ---- rtn: bits x input-keep, pushed down to the budget ----------------
    if "rtn" in groups:
        for bits in args.bits:
            qu, qg = rtn_pair(bits, args.qgroup)
            for keep in args.keeps:
                yield Variant(f"insp{keep}_q{bits}", qu, qg,
                              cb_rtn(bits, args.qgroup, keep), 2.0 * keep,
                              "rtn+insp", in_mode="topk", in_keep=keep,
                              extra=dict(bits=bits))
            del qu, qg
            torch.cuda.empty_cache()

    # ---- inspw: same grid, norm-weighted coordinate choice ---------------
    if "inspw" in groups:
        for bits in args.bits:
            qu, qg = rtn_pair(bits, args.qgroup)
            for keep in args.keeps:
                if keep >= 1.0:
                    continue
                yield Variant(f"inspW{keep}_q{bits}", qu, qg,
                              cb_rtn(bits, args.qgroup, keep), 2.0 * keep,
                              "rtn+inspW", in_mode="topk", in_keep=keep,
                              in_weight=cw, extra=dict(bits=bits))
            del qu, qg
            torch.cuda.empty_cache()

    # ---- asym: unequal precision between the two branches ----------------
    if "asym" in groups:
        for bu, bg in args.asym_bits:
            qu = quantize_rtn(Wu, bu, args.qgroup)
            qg = quantize_rtn(Wg, bg, args.qgroup)
            for keep in args.asym_keeps:
                cb = keep * (rtn_bits_per_weight(bu, args.qgroup)
                             + rtn_bits_per_weight(bg, args.qgroup)) / 16.0
                yield Variant(f"insp{keep}_u{bu}g{bg}", qu, qg, cb, 2.0 * keep,
                              "rtn+asym", in_mode="topk", in_keep=keep,
                              in_weight=cw if args.asym_normw else None,
                              extra=dict(bits_up=bu, bits_gate=bg))
            del qu, qg
            torch.cuda.empty_cache()

    # ---- pq / pqinsp -----------------------------------------------------
    if "pq" in groups or "pqinsp" in groups:
        for subdim, C in args.pq:
            bpw = np.log2(C) / subdim
            qu, eu = fit_pq(Wu, subdim, C, args.pq_iters, seed=args.seed)
            qg, eg = fit_pq(Wg, subdim, C, args.pq_iters, seed=args.seed)
            log(f"  pq d={subdim} C={C}: {bpw:.3f} bits/w  relerr up={eu:.4f} gate={eg:.4f}")
            ex = dict(subdim=subdim, codebook=C, bits_per_weight=float(bpw),
                      pq_relerr_up=eu, pq_relerr_gate=eg)
            if "pq" in groups:
                yield Variant(f"pq_d{subdim}c{C}", qu, qg, 2 * bpw / 16.0, 2.0,
                              "pq", subdim=subdim, extra=ex)
            if "pqinsp" in groups:
                for keep in args.pq_keeps:
                    yield Variant(f"pqsub{keep}_d{subdim}c{C}", qu, qg,
                                  2 * bpw / 16.0 * keep, 2.0 * keep, "pq+subinsp",
                                  in_mode="subspace", in_keep=keep, subdim=subdim,
                                  extra=ex)
            del qu, qg
            torch.cuda.empty_cache()

    # ---- had: Hadamard-rotated RTN and PQ --------------------------------
    if "had" in groups:
        Ru = fwht(Wu, chunk=8)
        Rg = fwht(Wg, chunk=8)
        for bits in args.had_bits:
            qu = quantize_rtn(Ru, bits, args.qgroup)
            qg = quantize_rtn(Rg, bits, args.qgroup)
            yield Variant(f"had_q{bits}", qu, qg, cb_rtn(bits, args.qgroup), 2.0,
                          "had+rtn", rotate=True, extra=dict(bits=bits))
            del qu, qg
            torch.cuda.empty_cache()
        for subdim, C in args.had_pq:
            bpw = np.log2(C) / subdim
            qu, eu = fit_pq(Ru, subdim, C, args.pq_iters, seed=args.seed)
            qg, eg = fit_pq(Rg, subdim, C, args.pq_iters, seed=args.seed)
            log(f"  had+pq d={subdim} C={C}: relerr up={eu:.4f} gate={eg:.4f}")
            yield Variant(f"had_pq_d{subdim}c{C}", qu, qg, 2 * bpw / 16.0, 2.0,
                          "had+pq", rotate=True, subdim=subdim,
                          extra=dict(subdim=subdim, codebook=C,
                                     bits_per_weight=float(bpw),
                                     pq_relerr_up=eu, pq_relerr_gate=eg))
            del qu, qg
            torch.cuda.empty_cache()
        del Ru, Rg
        torch.cuda.empty_cache()

    # ---- sign: the 1-bit floor ------------------------------------------
    if "sign" in groups:
        for group in args.sign_groups:
            qu, qg = sign_quant(Wu, group), sign_quant(Wg, group)
            cb = 2 * rtn_bits_per_weight(1, group) / 16.0
            yield Variant(f"sign_g{group}", qu, qg, cb, 2.0, "sign",
                          extra=dict(sign_group=group))
            for keep in args.sign_keeps:
                yield Variant(f"insp{keep}_sign_g{group}", qu, qg, cb * keep,
                              2.0 * keep, "sign+insp", in_mode="topk",
                              in_keep=keep, in_weight=cw,
                              extra=dict(sign_group=group))
            del qu, qg
            torch.cuda.empty_cache()

    # ---- radapt: bits by router rank ------------------------------------
    # The score carries the router weight g_e as a factor, so channels of
    # low-g_e experts rarely reach the pooled top-B. Spending the byte budget by
    # router rank -- down to *not scoring* the tail experts at all (bit-width 0,
    # which forfeits their channels) -- is a lever that exists only in an MoE.
    if "radapt" in groups:
        for sched in args.radapt:
            uniq = sorted({b for b in sched if b > 0})
            qus = {b: quantize_rtn(Wu, b, args.qgroup) for b in uniq}
            qgs = {b: quantize_rtn(Wg, b, args.qgroup) for b in uniq}
            base = "radapt_" + "-".join(str(b) for b in sched)
            cb0 = float(np.mean([2 * rtn_bits_per_weight(b, args.qgroup) / 16.0
                                 if b > 0 else 0.0 for b in sched]))
            cf0 = float(np.mean([2.0 if b > 0 else 0.0 for b in sched]))
            for keep in args.radapt_keeps:
                nm = base if keep >= 1.0 else f"{base}_insp{keep}"
                yield Variant(nm, qus, qgs, cb0 * keep, cf0 * keep, "radapt",
                              in_mode=None if keep >= 1.0 else "topk",
                              in_keep=keep, per_rank=sched,
                              extra=dict(schedule=list(sched),
                                         n_scored=sum(1 for b in sched if b > 0)))
            del qus, qgs
            torch.cuda.empty_cache()


# ==========================================================================
# diagnostics
# ==========================================================================

def router_rank_share(oracle, g, sel, K, I, budgets, ratios):
    """Share of the oracle top-B drawn from the expert at each router rank.

    If this is steeply skewed, byte budget should be spent per expert by router
    rank (``radapt``) rather than uniformly -- a lever that exists only in an MoE.
    """
    t = oracle.shape[0]
    rank_of_slot = g.argsort(dim=-1, descending=True).argsort(dim=-1)   # (t,K)
    out = {}
    for r, B in zip(ratios, budgets):
        idx = oracle.topk(B, dim=1, sorted=False).indices               # (t,B)
        slot = idx // I
        rk = rank_of_slot.gather(1, slot)
        share = torch.bincount(rk.reshape(-1), minlength=K).float() / (t * B)
        # mass share matters more than count share: it is the fraction of the
        # oracle top-B *score* that a rank-restricted scorer could still reach.
        val = oracle.gather(1, idx)
        msum = torch.zeros(K, device=oracle.device).index_add_(
            0, rk.reshape(-1), val.reshape(-1))
        out[f"rho{r}"] = [round(float(v), 4) for v in share]
        out[f"rho{r}_mass"] = [round(float(v), 4) for v in msum / msum.sum()]
        out[f"rho{r}_mass_cum"] = [round(float(v), 4)
                                   for v in (msum / msum.sum()).cumsum(0)]
    return out


def input_energy_retention(X, cw, keeps, dev, chunk=2048):
    """Fraction of ``x``'s energy kept by top-k coordinate selection.

    Reported for the plain ``|x_i|`` criterion and for the norm-weighted
    ``|x_i|*rms_i(W)`` one. Input truncation perturbs every channel score by
    ``<w_j, x_drop>``, whose scale is set by the *weighted* residual energy, so
    this curve is the ceiling on how far the cheap axis can be pushed.
    """
    out = {}
    tot = totw = 0.0
    keptw = {k: 0.0 for k in keeps}
    kept = {k: 0.0 for k in keeps}
    for s0 in range(0, X.shape[0], chunk):
        x = X[s0:s0 + chunk].to(dev)
        xw = x * cw
        tot += float((x ** 2).sum())
        totw += float((xw ** 2).sum())
        for k in keeps:
            if k >= 1.0:
                kept[k] += float((x ** 2).sum())
                keptw[k] += float((xw ** 2).sum())
                continue
            n = max(1, int(round(k * x.shape[-1])))
            kept[k] += float((x.abs().topk(n, -1).values ** 2).sum())
            # weighted criterion, weighted energy: what the score error sees
            idx = (x.abs() * cw).topk(n, -1).indices
            keptw[k] += float((xw.gather(-1, idx) ** 2).sum())
    out["plain_energy"] = {str(k): round(kept[k] / tot, 4) for k in keeps}
    out["normw_weighted_energy"] = {str(k): round(keptw[k] / totw, 4) for k in keeps}
    return out


def row_clusterability(W, ks=(8, 16, 64), sample=8192, iters=15, seed=0):
    """How well do whole rows cluster? (Do coarse-to-fine bounds have a chance?)

    Reports mean cosine of a row to its centroid. Values near the random-baseline
    ``1/sqrt(dim)`` mean the rows are mutually near-orthogonal and any
    group-summary bound is vacuous.
    """
    E, I, H = W.shape
    V = W.reshape(E * I, H)
    gcpu = torch.Generator().manual_seed(seed)
    V = V[torch.randperm(V.shape[0], generator=gcpu)[:sample].to(W.device)]
    Vn = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    out = {"random_baseline": round(1.0 / H ** 0.5, 5)}
    for k in ks:
        cb = Vn[torch.randperm(Vn.shape[0], generator=gcpu)[:k].to(W.device)].clone()
        for _ in range(iters):
            a = (Vn @ cb.t()).argmax(-1)
            num = torch.zeros_like(cb).index_add_(0, a, Vn)
            n = num.norm(dim=-1, keepdim=True)
            cb = torch.where(n > 0, num / n.clamp_min(1e-12), cb)
        cos = (Vn * cb[(Vn @ cb.t()).argmax(-1)]).sum(-1)
        out[f"k{k}"] = round(float(cos.mean()), 5)
    return out


# ==========================================================================
# main
# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="46")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="cap tokens actually scored (0 = all in the capture)")
    ap.add_argument("--ratios", default="0.25,0.125")
    ap.add_argument("--lams", default="1.1,1.25,1.5,2.0")
    ap.add_argument("--groups", default="ref,rtn,inspw,asym,pq,pqinsp,had,sign,radapt")
    ap.add_argument("--bits", default="4,3,2")
    ap.add_argument("--keeps", default="1.0,0.5,0.3,0.25,0.2,0.15,0.1")
    ap.add_argument("--qgroup", type=int, default=128)
    ap.add_argument("--asym-bits", default="4:2,4:3,3:2,8:2,8:4",
                    help="up:gate bit pairs")
    ap.add_argument("--asym-keeps", default="1.0,0.5,0.25")
    ap.add_argument("--asym-normw", type=int, default=1)
    ap.add_argument("--pq", default="4:16,8:16,8:256,16:16,16:256,32:256",
                    help="subdim:codebook pairs")
    ap.add_argument("--pq-keeps", default="0.5,0.25")
    ap.add_argument("--pq-iters", type=int, default=20)
    ap.add_argument("--had-bits", default="4,3,2")
    ap.add_argument("--had-pq", default="8:16,16:256")
    ap.add_argument("--sign-groups", default="128,2048")
    ap.add_argument("--sign-keeps", default="0.5")
    ap.add_argument("--radapt", default="3-3-3-3-0-0-0-0,3-3-3-3-3-3-0-0,"
                                        "4-4-3-3-2-2-2-2,4-4-4-2-2-2-0-0,3-3-0-0-0-0-0-0")
    ap.add_argument("--radapt-keeps", default="1.0,0.5,0.25")
    ap.add_argument("--diag", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capture-dir", default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--out", default=os.path.join(_REPO, "docs/results/idea_pilot/probe_frontier.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=1024)
    args = ap.parse_args()

    args.bits = [int(x) for x in args.bits.split(",") if x]
    args.keeps = [float(x) for x in args.keeps.split(",") if x]
    args.asym_bits = [tuple(int(v) for v in p.split(":")) for p in args.asym_bits.split(",") if p]
    args.asym_keeps = [float(x) for x in args.asym_keeps.split(",") if x]
    args.pq = [tuple(int(v) for v in p.split(":")) for p in args.pq.split(",") if p]
    args.pq_keeps = [float(x) for x in args.pq_keeps.split(",") if x]
    args.had_bits = [int(x) for x in args.had_bits.split(",") if x]
    args.had_pq = [tuple(int(v) for v in p.split(":")) for p in args.had_pq.split(",") if p]
    args.sign_groups = [int(x) for x in args.sign_groups.split(",") if x]
    args.sign_keeps = [float(x) for x in args.sign_keeps.split(",") if x]
    args.radapt = [[int(v) for v in s.split("-")] for s in args.radapt.split(",") if s]
    args.radapt_keeps = [float(x) for x in args.radapt_keeps.split(",") if x]

    layers = [int(x) for x in args.layers.split(",")]
    ratios = [float(x) for x in args.ratios.split(",")]
    lams = [float(x) for x in args.lams.split(",")]
    groups = set(args.groups.split(","))
    dev = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def log(m):
        print(m, flush=True)

    rows, diags = [], {}
    for layer in layers:
        cap_path = os.path.join(args.capture_dir, f"capture_L{layer}_t{args.tokens}.pt")
        if not os.path.exists(cap_path):
            log(f"[skip] no capture {cap_path}")
            continue
        cap = torch.load(cap_path, map_location="cpu")
        X, gate_w, Wg, Wu = cap["X"], cap["gate_w"], cap["Wg"], cap["Wu"]
        K, norm_topk = cap["top_k"], cap["norm_topk"]
        E, I, H = Wu.shape
        if args.max_tokens:
            X = X[:args.max_tokens]
        T, KI = X.shape[0], K * I
        budgets = [max(1, min(int(round(r * KI)), KI)) for r in ratios]
        log(f"\n[layer {layer}] T={T} E={E} I={I} H={H} K={K} B={budgets}")

        g, sel = _route(X, gate_w, K, norm_topk, dev)
        Wu_d, Wg_d = Wu.to(dev), Wg.to(dev)
        del cap, Wu, Wg

        # ---- cache the oracle once: every variant is scored against it ----
        oracle = torch.empty((T, KI), dtype=torch.float32, device=dev)
        hits_all = []
        for s0 in range(0, T, args.chunk):
            x = X[s0:s0 + args.chunk].to(dev)
            t = x.shape[0]
            sc_ = sel[s0:s0 + args.chunk].to(dev)
            gc_ = g[s0:s0 + args.chunk].to(dev)
            hits = []
            for e in torch.unique(sc_):
                tok, slot = torch.where(sc_ == int(e))
                hits.append((int(e), slot, tok))
            hits_all.append(hits)
            gate_t = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
            up_t = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
            for e, slot, tok in hits:
                cur = x[tok]
                gate_t[tok, slot] = cur @ Wg_d[e].t()
                up_t[tok, slot] = cur @ Wu_d[e].t()
            oracle[s0:s0 + t] = (gc_.unsqueeze(-1)
                                 * (F.silu(gate_t) * up_t).abs()).reshape(t, KI)
            del gate_t, up_t
        log(f"[layer {layer}] oracle cached ({oracle.numel() * 4 / 2**30:.2f} GiB)")

        o_top = [oracle.topk(B, dim=1, sorted=False).indices for B in budgets]
        o_mask = []
        for ti in o_top:
            m = torch.zeros_like(oracle, dtype=torch.bool)
            o_mask.append(m.scatter_(1, ti, True))
        o_topmass = [oracle.gather(1, ti).sum(dim=1).clamp_min(1e-30) for ti in o_top]
        del o_top

        if args.diag:
            cwd = (Wu_d.pow(2).mean(dim=(0, 1)) + Wg_d.pow(2).mean(dim=(0, 1))).sqrt()
            cwd = cwd / cwd.mean()
            d = {"router_rank_share": router_rank_share(
                    oracle, g.to(dev), sel.to(dev), K, I, budgets, ratios),
                 "input_energy": input_energy_retention(X, cwd, args.keeps, dev),
                 "col_norm_spread": {
                     "cv": round(float(cwd.std() / cwd.mean()), 4),
                     "p99_over_median": round(
                         float(cwd.quantile(0.99) / cwd.median()), 4)},
                 "row_clusterability_up": row_clusterability(Wu_d, seed=args.seed),
                 "row_clusterability_gate": row_clusterability(Wg_d, seed=args.seed)}
            diags[f"L{layer}"] = d
            log(f"[layer {layer}] router-rank share of top-B: {d['router_rank_share']}")
            log(f"[layer {layer}] input energy kept: {d['input_energy']}")
            log(f"[layer {layer}] col-norm spread: {d['col_norm_spread']}")
            log(f"[layer {layer}] row clusterability up: {d['row_clusterability_up']}")

        # ---- score every variant ------------------------------------------
        def score_variant(v):
            hit = np.zeros(len(budgets))
            mass = np.zeros(len(budgets))
            cover = np.zeros((len(budgets), len(lams)))
            cmass = np.zeros((len(budgets), len(lams)))
            spear = 0.0
            for ci, s0 in enumerate(range(0, T, args.chunk)):
                x = X[s0:s0 + args.chunk].to(dev)
                t = x.shape[0]
                gc_ = g[s0:s0 + args.chunk].to(dev)
                xs = v.prepare_input(x)
                up_p = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                gate_p = (torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                          if v.wg is not None else None)
                if v.per_rank is not None:
                    # bit-width assigned to each (token, slot) by its router rank
                    rank_of_slot = gc_.argsort(dim=-1, descending=True).argsort(dim=-1)
                    sched = torch.tensor(v.per_rank, device=dev)
                    bits_of_slot = sched[rank_of_slot]                   # (t,K)
                for e, slot, tok in hits_all[ci]:
                    cur = xs[tok]
                    if v.per_rank is None:
                        up_p[tok, slot] = cur @ v.wu[e].t()
                        if gate_p is not None:
                            gate_p[tok, slot] = cur @ v.wg[e].t()
                        continue
                    # a rank-0 bit-width means "do not score this expert at all"
                    bs = bits_of_slot[tok, slot]
                    for b in sorted(set(v.per_rank)):
                        if b == 0:
                            continue
                        m = bs == b
                        if not bool(m.any()):
                            continue
                        tk, sl, cu = tok[m], slot[m], cur[m]
                        up_p[tk, sl] = cu @ v.wu[b][e].t()
                        gate_p[tk, sl] = cu @ v.wg[b][e].t()
                if gate_p is None:
                    score = (gc_.unsqueeze(-1) * up_p.abs()).reshape(t, KI)
                else:
                    score = (gc_.unsqueeze(-1)
                             * (F.silu(gate_p) * up_p).abs()).reshape(t, KI)
                del up_p, gate_p
                orc = oracle[s0:s0 + t]
                for bi, B in enumerate(budgets):
                    om, tm = o_mask[bi][s0:s0 + t], o_topmass[bi][s0:s0 + t]
                    p_idx = score.topk(B, dim=1, sorted=False).indices
                    hit[bi] += float((om.gather(1, p_idx).sum(1).float() / B).sum())
                    mass[bi] += float((orc.gather(1, p_idx).sum(1) / tm).sum())
                    for li, lam in enumerate(lams):
                        C = min(KI, max(B, int(round(lam * B))))
                        c_idx = score.topk(C, dim=1, sorted=False).indices
                        f_local = orc.gather(1, c_idx).topk(B, dim=1, sorted=False).indices
                        f_idx = c_idx.gather(1, f_local)
                        cover[bi, li] += float(
                            (om.gather(1, f_idx).sum(1).float() / B).sum())
                        cmass[bi, li] += float((orc.gather(1, f_idx).sum(1) / tm).sum())
                spear += _spearman(score, orc) * t
                del score
            md = v.meta()
            row = {"layer": layer, "name": v.name, "n_tokens": T,
                   "spearman": spear / T, **md}
            for bi, (r, B) in enumerate(zip(ratios, budgets)):
                row[f"recall@rho{r}"] = hit[bi] / T
                row[f"mass@rho{r}"] = mass[bi] / T
                row[f"random_recall@rho{r}"] = B / KI
                # bytes for the whole FFN: proxy + exact gather of up/gate/down
                row[f"ffn_bytes@rho{r}"] = (r + md["cost_bytes"] / 3.0)
                row[f"ffn_flops@rho{r}"] = (r + md["cost_flops"] / 3.0)
                for li, lam in enumerate(lams):
                    row[f"casc_recall@rho{r}_lam{lam}"] = cover[bi, li] / T
                    row[f"casc_mass@rho{r}_lam{lam}"] = cmass[bi, li] / T
                    # cascade: exact up/gate on lam*B candidates, down on B kept
                    row[f"ffn_bytes@rho{r}_lam{lam}"] = (
                        md["cost_bytes"] + 2 * lam * r + r) / 3.0
            rows.append(row)
            r0 = ratios[-1]
            log(f"  {v.name:24s} cB={md['cost_bytes']:.4f} cF={md['cost_flops']:.3f} "
                f"sp={row['spearman']:.3f} rec@{r0}={row[f'recall@rho{r0}']:.3f} "
                f"mass={row[f'mass@rho{r0}']:.3f} "
                f"casc1.25={row[f'casc_recall@rho{r0}_lam1.25']:.3f} "
                f"ffnB={row[f'ffn_bytes@rho{r0}']:.3f}")

        # reference: oracle_up -- exact up_proj only, the strongest *deployable*
        # signal in the repo (71.30 HS at rho=0.125) and the number to beat.
        score_variant(Variant("oracle_up_ref", Wu_d, None, 1.0, 1.0, "oracle",
                              extra=dict(note="exact up-only signal")))

        for v in build_variants(Wu_d, Wg_d, groups, args, log):
            score_variant(v)
            del v
            torch.cuda.empty_cache()

        del oracle, o_mask, o_topmass, Wu_d, Wg_d, hits_all
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "ratios": ratios, "lams": lams,
                   "diagnostics": diags, "args": {k: str(v) for k, v in vars(args).items()}},
                  f, indent=2)
    log(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
