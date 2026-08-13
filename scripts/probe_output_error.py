#!/usr/bin/env python
"""Block-output error ladder: convert a selector's channel choice into predicted accuracy.

The recall screens (``probe_frontier.py``) rank selectors by *index agreement* with
the oracle's top-B. That is the wrong currency for accuracy: missing a channel
whose ``down_proj`` column barely moves the output is free, while missing a
dominant one is not. What the model actually feels is

    rel_err(token) = || y_full − y_kept || / || y_full ||

where ``y = sum_e g_e · W_down^(e) (m_e ⊙ inter_e)`` is the MoE block output.

This is worth measuring because the repo already has downstream accuracy for
several selectors at several budgets, so ``rel_err -> accuracy`` can be *fitted*
rather than extrapolated from two points:

    oracle_mag  rho=0.5/0.375/0.25/0.125  ->  HS 78.54 / 78.76 / 78.28 / 76.84
    oracle_up   rho=0.125                 ->  HS 71.30
    dense                                 ->  HS 78.56

With that curve, a new selector's HellaSwag score is predicted from a few minutes
on one GPU instead of ~11.6 GPU-h, which is what makes it possible to screen a
design space at all. The doc lists this ladder as TBD
(``docs/exps/dynamic_active_param/q3_30b_dynamic_active.md``).

Needs the ``_wd`` captures from ``probe_capture.py`` (the older captures have no
``down_proj``). One GPU, no model load.
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
from scripts.probe_frontier import rtn_bits_per_weight, topk_mask_input


def build_selectors(specs, Wu, Wg, qgroup, dev, log):
    """Yield ``(name, meta, scorer)``; ``scorer(x, g, hits, t) -> (t,K,I) score``.

    ``exact``/``up_only`` read the true projections (the reference ladder whose
    downstream accuracy is already measured); ``probe`` variants read b-bit weights
    on a top-|x| subset of coordinates. The ``||W_down[:,j]||`` factor that
    distinguishes ``oracle_mag`` from ``oracle_mag_noW`` is applied by the caller
    (it needs the per-token expert selection), flagged by ``needs_colnorm``.
    """
    E, I, H = Wu.shape
    I_H = float(H)          # <w_j, x_sub> has variance ||w_j||^2 ||x_sub||^2 / H

    def make_exact(use_gate):
        def f(x, g, hits, t, K):
            up = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
            gate = (torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                    if use_gate else None)
            for e, slot, tok in hits:
                cur = x[tok]
                up[tok, slot] = cur @ Wu[e].t().float()
                if use_gate:
                    gate[tok, slot] = cur @ Wg[e].t().float()
            s = (F.silu(gate) * up).abs() if use_gate else up.abs()
            return g.unsqueeze(-1) * s
        return f

    for spec in specs:
        kind = spec["kind"]
        if kind == "exact":
            yield spec["name"], spec, make_exact(True)
        elif kind == "up_only":
            yield spec["name"], spec, make_exact(False)
        elif kind == "probe":
            bits, keep = spec["bits"], spec["input_keep"]
            debias, absg = spec.get("debias", False), spec.get("abs_gate", False)
            qu = quantize_rtn(Wu.float(), bits, qgroup)
            qg = quantize_rtn(Wg.float(), bits, qgroup)
            cb = 2 * rtn_bits_per_weight(bits, qgroup) / 16.0 * keep
            # Per-channel row norms of the weight and of the quantization residual.
            # Two floats per channel (~0.05% of a matrix) and the only offline
            # state the debiased variant needs.
            rn_u, rn_g = Wu.float().norm(dim=2), Wg.float().norm(dim=2)   # (E,I)
            dn_u = (Wu.float() - qu).norm(dim=2)
            dn_g = (Wg.float() - qg).norm(dim=2)
            spec = dict(spec, cost_bytes=cb)
            log(f"  built probe bits={bits} keep={keep} debias={debias} "
                f"abs_gate={absg} cB={cb:.4f}")

            def f(x, g, hits, t, K, qu=qu, qg=qg, keep=keep, debias=debias,
                  absg=absg, rn_u=rn_u, rn_g=rn_g, dn_u=dn_u, dn_g=dn_g):
                xs = topk_mask_input(x, keep)
                # dropped / kept input energy: free, we already sorted |x| above
                e_drop = ((x - xs) ** 2).sum(-1)                    # (t,)
                e_keep = (xs ** 2).sum(-1)
                up = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                gate = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                for e, slot, tok in hits:
                    cur = xs[tok]
                    uh, gh = cur @ qu[e].t(), cur @ qg[e].t()
                    if debias:
                        # E[u_hat^2] = u^2 + sigma^2 with
                        #   sigma^2 = (||w_j||^2 ||x_drop||^2 + ||dw_j||^2 ||x_keep||^2)/H
                        # so u_hat^2 - sigma^2 is an unbiased estimate of u^2. The
                        # score is a *product* of two noisy factors, so shrinking
                        # each one re-ranks even though sigma is near-uniform over
                        # channels: it demotes channels kept alive only by noise.
                        ed = e_drop[tok].unsqueeze(-1)
                        ek = e_keep[tok].unsqueeze(-1)
                        su2 = (rn_u[e].unsqueeze(0) ** 2 * ed
                               + dn_u[e].unsqueeze(0) ** 2 * ek) / I_H
                        sg2 = (rn_g[e].unsqueeze(0) ** 2 * ed
                               + dn_g[e].unsqueeze(0) ** 2 * ek) / I_H
                        uh = (uh * uh - su2).clamp_min(0).sqrt() * torch.sign(uh)
                        gh = gh * (1.0 - sg2 / (gh * gh).clamp_min(1e-30)).clamp_min(0)
                    up[tok, slot] = uh
                    gate[tok, slot] = gh
                act = gate.abs() if absg else F.silu(gate)
                return g.unsqueeze(-1) * (act * up).abs()
            yield spec["name"], spec, f
        elif kind == "random":
            def f(x, g, hits, t, K):
                return torch.rand((t, K, I), device=dev)
            yield spec["name"], spec, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,22,38,46")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--ratios", default="0.5,0.375,0.25,0.125")
    ap.add_argument("--probes", default="3:0.25,3:0.5,4:1.0,3:1.0,2:1.0,4:0.25",
                    help="bits:input_keep probe variants")
    ap.add_argument("--qgroup", type=int, default=128)
    ap.add_argument("--serve-bits", type=int, default=0,
                    help="if >0, quantize up/gate/down to this many bits FIRST, so "
                         "'exact' means the served weights. A probe at the same "
                         "bit-width is then literally reading the served weights: "
                         "no extra storage, and its only error is input sparsity.")
    ap.add_argument("--refine", default="",
                    help="comma-separated probe refinements to also score: "
                         "db (closed-form debias), cn (x ||W_down[:,j]||), "
                         "dbcn, absg (|gate| instead of SiLU)")
    ap.add_argument("--capture-dir", default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--out", default=os.path.join(_REPO, "docs/results/idea_pilot/output_error.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    ratios = [float(x) for x in args.ratios.split(",")]
    probes = [tuple(p.split(":")) for p in args.probes.split(",") if p]
    dev = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def log(m):
        print(m, flush=True)

    rows = []
    for layer in layers:
        p = os.path.join(args.capture_dir, f"capture_L{layer}_t{args.tokens}_wd.pt")
        if not os.path.exists(p):
            log(f"[skip] no capture {p}")
            continue
        cap = torch.load(p, map_location="cpu")
        X, gate_w = cap["X"], cap["gate_w"]
        Wg, Wu, Wd = cap["Wg"], cap["Wu"], cap["Wd"]
        K, norm_topk = cap["top_k"], cap["norm_topk"]
        E, I, H = Wu.shape
        if args.max_tokens:
            X = X[:args.max_tokens]
        T = X.shape[0]
        budgets = [max(1, min(int(round(r * K * I)), K * I)) for r in ratios]
        log(f"\n[layer {layer}] T={T} E={E} I={I} H={H} K={K} B={budgets}")

        g, sel = _route(X, gate_w, K, norm_topk, dev)
        Wu_d, Wg_d, Wd_d = Wu.to(dev), Wg.to(dev), Wd.to(dev)
        del cap, Wu, Wg, Wd
        if args.serve_bits:
            # Serve the model at serve_bits. RTN is idempotent on its own output
            # (same group maxima -> same scale), so a probe built at this bit-width
            # reads exactly these tensors and contributes no quantization error of
            # its own -- only input sparsity remains.
            sb = args.serve_bits
            Wu_d = quantize_rtn(Wu_d.float(), sb, args.qgroup)
            Wg_d = quantize_rtn(Wg_d.float(), sb, args.qgroup)
            Wd_d = quantize_rtn(Wd_d.float().transpose(1, 2), sb,
                                args.qgroup).transpose(1, 2).contiguous()
            log(f"[layer {layer}] serving at {sb} bits (probe adds no storage)")
        # ||W_down[:, j]|| per (expert, channel) -- the oracle_mag weighting
        col_norm_full = Wd_d.float().norm(dim=1)                    # (E, I)

        specs = [
            dict(name="oracle_mag", kind="exact", cost_bytes=2.0, needs_colnorm=True),
            dict(name="oracle_mag_noW", kind="exact", cost_bytes=2.0),
            dict(name="oracle_up", kind="up_only", cost_bytes=1.0, needs_colnorm=True),
            dict(name="random", kind="random", cost_bytes=0.0),
        ]
        for bits, keep in probes:
            b, k = int(bits), float(keep)
            specs.append(dict(name=f"probe_q{b}_k{k}", kind="probe",
                              bits=b, input_keep=k))
            for tag, kw in (("db", dict(debias=True)),
                            ("cn", dict(needs_colnorm=True)),
                            ("dbcn", dict(debias=True, needs_colnorm=True)),
                            ("absg", dict(abs_gate=True))):
                if tag in args.refine.split(","):
                    specs.append(dict(name=f"probe_q{b}_k{k}_{tag}", kind="probe",
                                      bits=b, input_keep=k, **kw))

        for name, meta, scorer in build_selectors(specs, Wu_d, Wg_d, args.qgroup,
                                                  dev, log):
            err = np.zeros(len(ratios))
            fullsq = 0.0
            errsq = np.zeros(len(ratios))
            for s0 in range(0, T, args.chunk):
                x = X[s0:s0 + args.chunk].to(dev)
                t = x.shape[0]
                gc_ = g[s0:s0 + args.chunk].to(dev)
                sc_ = sel[s0:s0 + args.chunk].to(dev)
                hits = []
                for e in torch.unique(sc_):
                    tok, slot = torch.where(sc_ == int(e))
                    hits.append((int(e), slot, tok))

                # exact intermediate and the full block output
                inter = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                for e, slot, tok in hits:
                    cur = x[tok]
                    inter[tok, slot] = (F.silu(cur @ Wg_d[e].t().float())
                                        * (cur @ Wu_d[e].t().float()))
                y_full = torch.zeros((t, H), dtype=torch.float32, device=dev)
                for e, slot, tok in hits:
                    y_full.index_add_(
                        0, tok,
                        (inter[tok, slot] @ Wd_d[e].t().float())
                        * gc_[tok, slot].unsqueeze(-1))
                fullsq += float((y_full ** 2).sum())

                score = scorer(x, gc_, hits, t, K)
                if meta.get("needs_colnorm"):
                    score = score * col_norm_full[sc_]
                flat = score.reshape(t, K * I)
                for bi, B in enumerate(budgets):
                    idx = flat.topk(B, dim=1, sorted=False).indices
                    m = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, idx, True)
                    dropped = (inter * (~m.reshape(t, K, I))).contiguous()
                    y_err = torch.zeros((t, H), dtype=torch.float32, device=dev)
                    for e, slot, tok in hits:
                        y_err.index_add_(
                            0, tok,
                            (dropped[tok, slot] @ Wd_d[e].t().float())
                            * gc_[tok, slot].unsqueeze(-1))
                    errsq[bi] += float((y_err ** 2).sum())
                    err[bi] += float((y_err.norm(dim=1)
                                      / y_full.norm(dim=1).clamp_min(1e-30)).sum())
                del inter, y_full, score, flat

            row = {"layer": layer, "name": name,
                   "cost_bytes": meta.get("cost_bytes"), "n_tokens": T}
            for bi, r in enumerate(ratios):
                row[f"rel_err@rho{r}"] = err[bi] / T                 # mean per token
                row[f"energy_err@rho{r}"] = (errsq[bi] / fullsq) ** 0.5
            rows.append(row)
            log(f"  {name:20s} " + "  ".join(
                f"rho{r}: relerr={row[f'rel_err@rho{r}']:.4f}" for r in ratios))

        del Wu_d, Wg_d, Wd_d, col_norm_full
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "ratios": ratios}, f, indent=2)
    log(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
