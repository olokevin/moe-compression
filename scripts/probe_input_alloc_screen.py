#!/usr/bin/env python
"""Which input coordinates should the probe read — and should every expert get the
same number of them?

The probe ranks a token's pooled ``K*I`` channels by ``g_e*|SiLU(g̃ate)⊙ũp|`` after
reading only a fraction of ``x``'s coordinates. Everything measured so far gives
each of the token's K experts the *same* coordinate budget. But the pooled score
carries a ``g_e`` factor, so a coordinate read on a dominated expert moves the
ranking less than the same read on the top-routed one. That suggests spending the
pooled read budget unevenly. Terms compared here, all at an **identical** pooled
budget of ``K*round(p*H)`` coordinate-reads per token:

  ``uniform``   every expert reads the token's top-``p*H`` coordinates by ``|x|``.
                (What every measured probe row does today.)
  ``router``    rank the ``(slot, coord)`` pairs by ``g_e*|x_i|``, pooled top-N.
  ``router2``   rank by ``g_e^2*|x_i|``, motivated by Level-1's ``g^2*sigma``
                scoring: an expert's channels both carry a ``g_e`` factor and are
                more numerous near the global threshold in proportion to ``g_e``.
  ``colnorm``   uniform budget, but rank coordinates by ``|x_i|*rms_j(W[:,i])``
                — the currency that actually perturbs a score. (Re-measured here
                on the *output-error* metric; the recall screen found it a wash.)

Metric is block-output ``rel_err`` — the currency the ladder validated against
measured accuracy (R^2 0.985 at fixed budget, ``probe_relerr_linearity.py``), not
index recall, which does not order selectors across families.

Reads the cached ``_wd`` captures. One GPU, no model load, minutes.
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

from scripts.idea_pilot_scorers import _route, quantize_rtn
from src.dynamic_active_param.sparse_probe import (
    allocate_input_reads,
    descending_abs_ranks,
    sparsify_input_by_count,
    sparsify_input_topk,
    used_param_fraction,
)

TERMS = ("uniform", "router", "router2", "colnorm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,22,38,46")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--rho-inputs", "--ps", dest="ps", default="0.25",
                    help="rho_input grid (input coordinates read for scoring)")
    ap.add_argument("--rho-channels", "--rhos", dest="rhos",
                    default="0.10,0.125,0.15,0.20",
                    help="rho_channel grid (channels kept for compute)")
    ap.add_argument("--terms", default=",".join(TERMS))
    ap.add_argument("--bits", type=int, default=16,
                    help="16 = reuse the served weights (no extra storage)")
    ap.add_argument("--qgroup", type=int, default=128)
    ap.add_argument("--capture-dir", default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--out", default=os.path.join(
        _REPO, "docs/results/idea_pilot/input_alloc.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    layers = [int(v) for v in args.layers.split(",")]
    ps = [float(v) for v in args.ps.split(",")]
    rhos = [float(v) for v in args.rhos.split(",")]
    terms = [t for t in args.terms.split(",") if t]
    dev = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for layer in layers:
        path = os.path.join(args.capture_dir, f"capture_L{layer}_t{args.tokens}_wd.pt")
        if not os.path.exists(path):
            print(f"[skip] no capture {path}", flush=True)
            continue
        cap = torch.load(path, map_location="cpu")
        X, gate_w = cap["X"], cap["gate_w"]
        Wg, Wu, Wd = cap["Wg"], cap["Wu"], cap["Wd"]
        K, norm_topk = cap["top_k"], cap["norm_topk"]
        E, I, H = Wu.shape
        if args.max_tokens:
            X = X[:args.max_tokens]
        T = X.shape[0]
        g_all, sel_all = _route(X, gate_w, K, norm_topk, dev)
        Wu_d, Wg_d, Wd_d = Wu.to(dev).float(), Wg.to(dev).float(), Wd.to(dev).float()
        del cap, Wu, Wg, Wd
        if args.bits < 16:
            Wu_p = quantize_rtn(Wu_d, args.bits, args.qgroup)
            Wg_p = quantize_rtn(Wg_d, args.bits, args.qgroup)
        else:
            Wu_p, Wg_p = Wu_d, Wg_d          # reuse: probe IS the served weight
        # rms_j(W[:, i]) pooled over both branches and all experts
        col_rms = torch.cat([Wu_d, Wg_d], dim=1).pow(2).mean(dim=(0, 1)).sqrt()
        print(f"\n[layer {layer}] T={T} E={E} I={I} H={H} K={K}", flush=True)

        for p in ps:
            acc = {(t, r): 0.0 for t in terms for r in rhos}
            nread = {t: 0.0 for t in terms}
            for s0 in range(0, T, args.chunk):
                x = X[s0:s0 + args.chunk].to(dev).float()
                t_n = x.shape[0]
                g = g_all[s0:s0 + args.chunk].to(dev)
                sel = sel_all[s0:s0 + args.chunk].to(dev)
                hits = []
                for e in torch.unique(sel):
                    tok, slot = torch.where(sel == int(e))
                    hits.append((int(e), slot, tok))

                # exact intermediate + true block output (shared by all terms)
                inter = torch.zeros((t_n, K, I), dtype=torch.float32, device=dev)
                for e, slot, tok in hits:
                    cur = x[tok]
                    inter[tok, slot] = (F.silu(cur @ Wg_d[e].t()) * (cur @ Wu_d[e].t()))
                y_full = torch.zeros((t_n, H), dtype=torch.float32, device=dev)
                for e, slot, tok in hits:
                    y_full.index_add_(0, tok, (inter[tok, slot] @ Wd_d[e].t())
                                      * g[tok, slot].unsqueeze(-1))
                fnorm = y_full.norm(dim=1).clamp_min(1e-30)

                ranks, sorted_abs = descending_abs_ranks(x)
                for term in terms:
                    if term == "uniform":
                        xs_shared = sparsify_input_topk(x, p)
                        nk = None
                    elif term == "colnorm":
                        k = max(1, int(round(p * H)))
                        idx = (x.abs() * col_rms).topk(k, dim=-1).indices
                        xs_shared = torch.zeros_like(x).scatter_(
                            -1, idx, x.gather(-1, idx))
                        nk = None
                    else:
                        beta = 1.0 if term == "router" else 2.0
                        nk = allocate_input_reads(sorted_abs, g, p, beta)
                        xs_shared = None
                    if nk is not None:
                        nread[term] += float(nk.sum())
                    else:
                        nread[term] += float(t_n * K * max(1, int(round(p * H))))

                    proxy = torch.zeros((t_n, K, I), dtype=torch.float32, device=dev)
                    for e, slot, tok in hits:
                        cs = (xs_shared[tok] if xs_shared is not None
                              else sparsify_input_by_count(x[tok], ranks[tok],
                                                           nk[tok, slot]))
                        uh = cs @ Wu_p[e].t()
                        gh = cs @ Wg_p[e].t()
                        proxy[tok, slot] = (F.silu(gh) * uh).abs()
                    score = g.unsqueeze(-1) * proxy
                    flat = score.reshape(t_n, K * I)
                    for rho in rhos:
                        B = max(1, min(int(round(rho * K * I)), K * I))
                        ii = flat.topk(B, dim=1, sorted=False).indices
                        m = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, ii, True)
                        dropped = inter * (~m.reshape(t_n, K, I))
                        y_err = torch.zeros((t_n, H), dtype=torch.float32, device=dev)
                        for e, slot, tok in hits:
                            y_err.index_add_(0, tok, (dropped[tok, slot] @ Wd_d[e].t())
                                             * g[tok, slot].unsqueeze(-1))
                        acc[(term, rho)] += float((y_err.norm(dim=1) / fnorm).sum())
                del inter, y_full

            for term in terms:
                row = {"layer": layer, "term": term, "p": p, "n_tokens": T,
                       "bits": args.bits,
                       "reads_per_token": nread[term] / T}
                for rho in rhos:
                    row[f"rel_err@rho{rho}"] = acc[(term, rho)] / T
                    row[f"kept@rho{rho}"] = used_param_fraction(p, rho)
                rows.append(row)
                print(f"  {term:9s} p={p} reads/token={row['reads_per_token']:7.1f}  "
                      + "  ".join(f"rho{r}:{row[f'rel_err@rho{r}']:.4f}" for r in rhos),
                      flush=True)

        del Wu_d, Wg_d, Wd_d, Wu_p, Wg_p
        torch.cuda.empty_cache()

    # ---- layer-averaged summary + verdict ---------------------------------
    print("\n[summary] layer-averaged rel_err (lower is better)")
    summ = {}
    for p in ps:
        print(f"\n  p={p}")
        print(f"    {'term':10s}" + "".join(f"{'rho'+str(r):>12s}" for r in rhos))
        base = None
        for term in terms:
            sub = [r for r in rows if r["term"] == term and r["p"] == p]
            if not sub:
                continue
            vals = [float(np.mean([r[f"rel_err@rho{rho}"] for r in sub])) for rho in rhos]
            summ[(p, term)] = vals
            if term == "uniform":
                base = vals
            print(f"    {term:10s}" + "".join(f"{v:12.4f}" for v in vals))
        if base:
            print(f"    {'-- delta vs uniform (pt of HellaSwag, slope -26.4) --':}")
            for term in terms:
                if term == "uniform" or (p, term) not in summ:
                    continue
                d = [(base[i] - summ[(p, term)][i]) for i in range(len(rhos))]
                print(f"    {term:10s}" + "".join(f"{26.4*x:+12.2f}" for x in d))

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "ps": ps, "rhos": rhos, "terms": terms,
                   "bits": args.bits}, f, indent=2)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
