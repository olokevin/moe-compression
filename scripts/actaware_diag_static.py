#!/usr/bin/env python
"""Diagnostic: how much of a low-rank scorer's ranking is *static* (token-independent)?

Why this is the load-bearing question for the whole activation-aware family. The
screen (``actaware_scorer_screen.py``) reports the score metric's effective rank as
**~1.7** at layer 46: one direction of ``Sigma^{1/2} M Sigma^{1/2}`` carries most of
the energy that moves channel scores. A rank-r sketch therefore spends its first and
largest component on the *common-mode* part of ``x`` — the part that is nearly the
same for every token. But the ranking that a scorer must reproduce is
**per-token**: this repo already measured that a static per-channel prior is worth
almost nothing (``probe_prefilter_diag.py``: forbidding the bottom 25% of channels
by held-out keep-frequency already loses 12-18% of the oracle top-B mass; and
``expert-redundancy-is-not-expert-level``: all the exploitable slack is per-token).

So a low-rank scorer can look good on ``recall`` while being *structurally* unable
to reproduce the per-token signal, because recall is dominated by the easy, always-hot
channels. This script separates the two contributions, for any scorer:

  ``static_recall``   recall of a scorer that ignores ``x`` entirely and ranks
                      channels by their mean oracle score (the strongest possible
                      static ranking, fit on held-out tokens). This is the floor
                      that *any* method gets for free.
  ``centered_spear``  Spearman between scorer and oracle **after removing each
                      channel's mean over tokens**, i.e. agreement on the purely
                      per-token part of the signal. This is the quantity a scorer
                      must have to beat the static floor.

If a low-rank scorer's ``static_recall`` is close to its full ``recall`` while its
``centered_spear`` is near zero, the mechanism is dead no matter the rank — exactly
the diagnosis the rank-saturation in ``btt_dynamic.md`` (recall stuck at ~0.47) hints
at but does not prove. The quantized probe should behave the opposite way.

Runs on the cached ``_wd`` captures; one GPU, no model load.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts.actaware_scorer_screen import (
    build_per_expert, build_shared_basis, input_gram, weight_gram,
)
from scripts.idea_pilot_scorers import _route, quantize_rtn
from scripts.probe_frontier import rtn_bits_per_weight, topk_mask_input


def centered_spearman(a, b):
    """Spearman(a, b) computed on the token-centered scores.

    Each channel's mean over the token axis is removed from both sides first, so the
    statistic measures agreement on *variation across tokens* rather than on the
    static channel profile. Computed over the pooled (token, channel) matrix of a
    fixed global channel slot, which is why this needs the global (E,I) indexing.
    """
    ac = a - a.mean(0, keepdim=True)
    bc = b - b.mean(0, keepdim=True)
    ra = ac.argsort(0).argsort(0).float()
    rb = bc.argsort(0).argsort(0).float()
    ra = ra - ra.mean(0, keepdim=True)
    rb = rb - rb.mean(0, keepdim=True)
    num = (ra * rb).sum(0)
    den = ra.norm(dim=0) * rb.norm(dim=0)
    return float((num / den.clamp_min(1e-12)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="46")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--fit-tokens", type=int, default=4096)
    ap.add_argument("--score-tokens", type=int, default=1024)
    ap.add_argument("--ratios", default="0.25,0.125")
    ap.add_argument("--ranks", default="32,128")
    ap.add_argument("--qgroup", type=int, default=128)
    ap.add_argument("--capture-dir", default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--out", default=os.path.join(_REPO, "docs/results/actaware/static_diag.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()

    layers = [int(v) for v in args.layers.split(",") if v]
    ratios = [float(v) for v in args.ratios.split(",") if v]
    ranks = [int(v) for v in args.ranks.split(",") if v]
    dev = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def log(m):
        print(m, flush=True)

    out = {"layers": {}}
    for layer in layers:
        p = os.path.join(args.capture_dir, f"capture_L{layer}_t{args.tokens}_wd.pt")
        if not os.path.exists(p):
            log(f"[skip] no capture {p}")
            continue
        cap = torch.load(p, map_location="cpu")
        Xall, gate_w = cap["X"], cap["gate_w"]
        Wg_c, Wu_c = cap["Wg"], cap["Wu"]
        K, norm_topk = cap["top_k"], cap["norm_topk"]
        E, I, H = Wu_c.shape
        del cap

        Xfit = Xall[:args.fit_tokens]
        X = Xall[args.fit_tokens:args.fit_tokens + args.score_tokens]
        T, KI = X.shape[0], K * I
        budgets = [max(1, min(int(round(r * KI)), KI)) for r in ratios]
        log(f"\n[layer {layer}] fit={Xfit.shape[0]} score={T} B={budgets}")

        Sigma = input_gram(Xfit, dev)
        M = weight_gram([Wu_c, Wg_c], dev)
        Wu, Wg = Wu_c.to(dev), Wg_c.to(dev)
        del Wu_c, Wg_c

        # ---- static prior: mean oracle score per GLOBAL (expert, channel) -----
        # Fit on the held-out slice. This is the best static ranking available:
        # it knows each channel's average importance but nothing about the token.
        gf, self_ = _route(Xfit, gate_w, K, norm_topk, dev)
        prior_sum = torch.zeros((E, I), dtype=torch.float64, device=dev)
        prior_cnt = torch.zeros((E,), dtype=torch.float64, device=dev)
        for s0 in range(0, Xfit.shape[0], args.chunk):
            x = Xfit[s0:s0 + args.chunk].to(dev)
            sc_ = self_[s0:s0 + args.chunk].to(dev)
            gc_ = gf[s0:s0 + args.chunk].to(dev)
            for e in torch.unique(sc_):
                e = int(e)
                tok, slot = torch.where(sc_ == e)
                cur = x[tok]
                v = (F.silu(cur @ Wg[e].t().float()) * (cur @ Wu[e].t().float())).abs()
                prior_sum[e] += (v * gc_[tok, slot].unsqueeze(-1)).sum(0).double()
                prior_cnt[e] += tok.numel()
        prior = (prior_sum / prior_cnt.clamp_min(1).unsqueeze(-1)).float()   # (E,I)

        g, sel = _route(X, gate_w, K, norm_topk, dev)

        # ---- oracle on the scored slice --------------------------------------
        oracle = torch.empty((T, KI), dtype=torch.float32, device=dev)
        hits_all = []
        for ci, s0 in enumerate(range(0, T, args.chunk)):
            x = X[s0:s0 + args.chunk].to(dev)
            t = x.shape[0]
            sc_ = sel[s0:s0 + args.chunk].to(dev)
            gc_ = g[s0:s0 + args.chunk].to(dev)
            hits = []
            for e in torch.unique(sc_):
                tok, slot = torch.where(sc_ == int(e))
                hits.append((int(e), slot, tok))
            hits_all.append(hits)
            v = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
            for e, slot, tok in hits:
                cur = x[tok]
                v[tok, slot] = (F.silu(cur @ Wg[e].t().float())
                                * (cur @ Wu[e].t().float())).abs()
            oracle[s0:s0 + t] = (gc_.unsqueeze(-1) * v).reshape(t, KI)
        o_mask = []
        for B in budgets:
            ti = oracle.topk(B, dim=1, sorted=False).indices
            o_mask.append(torch.zeros_like(oracle, dtype=torch.bool).scatter_(1, ti, True))

        # ---- scorers to diagnose ---------------------------------------------
        def sc_static(x, hits, t, gc_):
            """Ignores x: the held-out mean score of the routed (expert, channel)."""
            s = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
            for e, slot, tok in hits:
                s[tok, slot] = prior[e].unsqueeze(0).expand(tok.numel(), -1)
            return gc_.unsqueeze(-1) * s

        def make_shared(r, mode):
            P, pull = build_shared_basis(Sigma, M, r, mode, 1e-6)
            P, pull = P.float(), pull.float()
            Au = torch.einsum("eih,hr->eir", Wu.float(), pull)
            Ag = torch.einsum("eih,hr->eir", Wg.float(), pull)

            def f(x, hits, t, gc_):
                h = x.float() @ P.t()
                up = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                gt = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                for e, slot, tok in hits:
                    hh = h[tok]
                    up[tok, slot] = hh @ Au[e].t()
                    gt[tok, slot] = hh @ Ag[e].t()
                return gc_.unsqueeze(-1) * (F.silu(gt) * up).abs()
            return f

        def make_awsvd(r):
            Au, Ru = build_per_expert(Wu, Sigma, r, "awsvd", 1e-6)
            Ag, Rg = build_per_expert(Wg, Sigma, r, "awsvd", 1e-6)

            def f(x, hits, t, gc_):
                xf = x.float()
                up = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                gt = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                for e, slot, tok in hits:
                    cur = xf[tok]
                    up[tok, slot] = (cur @ Ru[e].t()) @ Au[e].t()
                    gt[tok, slot] = (cur @ Rg[e].t()) @ Ag[e].t()
                return gc_.unsqueeze(-1) * (F.silu(gt) * up).abs()
            return f

        qu = quantize_rtn(Wu.float(), 3, args.qgroup)
        qg = quantize_rtn(Wg.float(), 3, args.qgroup)

        def sc_probe(x, hits, t, gc_):
            xs = topk_mask_input(x, 0.25)
            up = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
            gt = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
            for e, slot, tok in hits:
                cur = xs[tok]
                up[tok, slot] = cur @ qu[e].t()
                gt[tok, slot] = cur @ qg[e].t()
            return gc_.unsqueeze(-1) * (F.silu(gt) * up).abs()

        scorers = [("static_prior", sc_static), ("probe_q3_k25", sc_probe)]
        for r in ranks:
            scorers.append((f"actbasis_r{r}", make_shared(r, "act")))
            scorers.append((f"awsvd_r{r}", make_awsvd(r)))

        res = {}
        for name, fn in scorers:
            hit = [0.0] * len(budgets)
            cs = 0.0
            for ci, s0 in enumerate(range(0, T, args.chunk)):
                x = X[s0:s0 + args.chunk].to(dev)
                t = x.shape[0]
                gc_ = g[s0:s0 + args.chunk].to(dev)
                score = fn(x, hits_all[ci], t, gc_).reshape(t, KI)
                orc = oracle[s0:s0 + t]
                for bi, B in enumerate(budgets):
                    idx = score.topk(B, dim=1, sorted=False).indices
                    hit[bi] += float(
                        (o_mask[bi][s0:s0 + t].gather(1, idx).sum(1).float() / B).sum())
                # centered agreement, on this chunk's tokens (chunk >= 256 tokens,
                # so the per-channel mean is a usable estimate)
                cs += centered_spearman(score, orc) * t
                del score
            res[name] = {"centered_spearman": cs / T}
            for bi, r in enumerate(ratios):
                res[name][f"recall@rho{r}"] = hit[bi] / T
            log(f"  {name:20s} " + "  ".join(
                f"rec@{r}={res[name][f'recall@rho{r}']:.3f}" for r in ratios)
                + f"  centered_spearman={res[name]['centered_spearman']:.4f}")

        st = res["static_prior"]
        log("\n  --- how much of each scorer's recall is above the static floor? ---")
        for name, v in res.items():
            for r in ratios:
                k = f"recall@rho{r}"
                log(f"  {name:20s} rho={r}: recall={v[k]:.3f} "
                    f"static={st[k]:.3f} excess={v[k] - st[k]:+.3f}")
        out["layers"][f"L{layer}"] = res

        del Wu, Wg, qu, qg, oracle, o_mask, hits_all, Sigma, M, prior
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# spectrum provenance: is the top data-optimal direction just the mean token?
# --------------------------------------------------------------------------
# Run as: python -m scripts.actaware_diag_static --mean-diag 1 ... (see main),
# or directly via this helper. Reported in actaware_scorer.md; the measured answer
# is yes, cos^2(E[x], top-1 eigvec of Sigma^{1/2} M Sigma^{1/2}) >= 0.998 on all
# four layers, and E[x] alone accounts for lambda_1 to within 0.01 of tr(C).

def mean_direction_diag(X, Sigma, M, dev, ranks=(1, 8, 32)):
    """How much of the score-moving energy is the *mean token*, and is it the top
    eigendirection of the metric ``C = Sigma^{1/2} M Sigma^{1/2}``?

    ``E_j <w_j, dx>^2 = dx^T M dx / I``, so ``tr(Sigma M)`` is the total
    score-moving energy of the data and ``mu^T M mu`` is the part a scorer would get
    from replacing every token by the average token. If the latter is ~= the top
    eigenvalue share, a rank-1 data-optimal sketch *is* the static prior.
    """
    import torch
    mu = X.to(dev, torch.float64).mean(0)
    tot = float(torch.einsum("ij,ji->", Sigma, M))
    Ssq = psd_sqrt(Sigma, 1e-6)
    ev, Q = torch.linalg.eigh(Ssq @ M @ Ssq)
    order = ev.argsort(descending=True)
    ev, Q = ev[order], Q[:, order]
    mun = mu / mu.norm().clamp_min(1e-30)
    out = {
        "eff_rank_score": tot / float(ev[0]),
        "lam1_share": float(ev[0]) / tot,
        "score_energy_from_mean": float(mu @ M @ mu) / tot,
        "mean_share_of_input_energy": float(mu @ mu) / float(Sigma.diagonal().sum()),
    }
    for r in ranks:
        Bq, _ = torch.linalg.qr(Ssq @ Q[:, :r])
        out[f"cos2_mean_in_top{r}"] = float((Bq.t() @ mun).pow(2).sum())
    return out
