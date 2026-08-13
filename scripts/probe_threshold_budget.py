#!/usr/bin/env python
"""Should the per-token channel budget be **fixed** or should it float under a
global threshold?

Every selector in this repo keeps exactly ``B`` channels for *every* token. That is
a modelling choice, not a requirement: what the hardware charges is the *mean*
number of channels loaded, so a token that is easy to approximate could keep fewer
and hand its slack to a token that is hard. Replacing "top-B per token" with
"every channel scoring above ``tau``" does exactly that, at the same mean cost —
and is *cheaper* to implement (a compare, not a per-token top-k over ``K*I``).

The same question applies to the probe's input reads: "top-``p*H`` coordinates per
token" versus "every coordinate with ``|x_i| > tau_x``".

This script measures all four combinations on the output-error metric the ladder
validated (−26.4 HellaSwag pt per unit rel_err at fixed budget):

    fixed_B   x fixed_p     the current scheme
    thresh_B  x fixed_p     float the channel budget only
    fixed_B   x thresh_p    float the input reads only
    thresh_B  x thresh_p    float both

Thresholds are calibrated per layer on the first chunk of tokens to hit the target
*mean* budget, then held fixed for the rest — i.e. an offline-calibrated constant,
exactly what a deployed kernel would carry. The realized mean is reported so the
comparison is verifiably iso-cost, not iso-nominal.

Reads the cached ``_wd`` captures. One GPU, no model load.
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

from scripts.idea_pilot_scorers import _route
from src.dynamic_active_param.sparse_probe import (
    allocate_input_reads,
    descending_abs_ranks,
    sparsify_input_topk,
    used_param_fraction,
)


def calibrate_quantile(vals_sorted_desc, target_count):
    """Threshold keeping ``target_count`` entries on average over the calibration
    tokens: the ``target_count``-th largest score, averaged across tokens."""
    idx = max(0, min(int(round(target_count)) - 1, vals_sorted_desc.shape[1] - 1))
    return float(vals_sorted_desc[:, idx].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,22,38,46")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--rho-input", "--p", dest="p", type=float, default=0.25,
                    help="input coordinates read for scoring")
    ap.add_argument("--rho-channels", "--rhos", dest="rhos",
                    default="0.10,0.125,0.15,0.20",
                    help="rho_channel grid (channels kept for compute)")
    ap.add_argument("--input-alloc", default="router",
                    choices=["uniform", "router"],
                    help="how fixed_p splits reads across the token's K experts")
    ap.add_argument("--calib-chunk", type=int, default=512,
                    help="tokens used to calibrate the thresholds")
    ap.add_argument("--capture-dir", default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--out", default=os.path.join(
        _REPO, "docs/results/idea_pilot/threshold_budget.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()

    layers = [int(v) for v in args.layers.split(",")]
    rhos = [float(v) for v in args.rhos.split(",")]
    dev = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    MODES = ["fixed_B/fixed_p", "thresh_B/fixed_p",
             "fixed_B/thresh_p", "thresh_B/thresh_p"]

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
        print(f"\n[layer {layer}] T={T} I={I} H={H} K={K} p={args.p} "
              f"alloc={args.input_alloc}", flush=True)

        n_read_target = K * max(1, int(round(args.p * H)))

        def probe_scores(x, g, hits, xs=None, nk=None, ranks=None):
            """(t,K,I) pooled probe score; reuse regime (served weights)."""
            t = x.shape[0]
            pr = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
            for e, slot, tok in hits:
                if xs is not None:
                    cs = xs[tok]
                else:
                    cs = x[tok] * (ranks[tok] < nk[tok, slot].unsqueeze(-1)).float()
                pr[tok, slot] = (F.silu(cs @ Wg_d[e].t()) * (cs @ Wu_d[e].t())).abs()
            return g.unsqueeze(-1) * pr

        # --- calibration pass: thresholds for input reads and for scores ----
        x0 = X[:args.calib_chunk].to(dev).float()
        g0 = g_all[:args.calib_chunk].to(dev)
        s0 = sel_all[:args.calib_chunk].to(dev)
        hits0 = []
        for e in torch.unique(s0):
            tok, slot = torch.where(s0 == int(e))
            hits0.append((int(e), slot, tok))
        # tau_x: keep n_read_target coordinate-reads per token on average, pooled
        # over the K slots. Under the router term the pooled ranking is g_e*|x_i|.
        ranks0, sabs0 = descending_abs_ranks(x0)
        if args.input_alloc == "router":
            pooled = (g0.unsqueeze(-1) * x0.abs().unsqueeze(1)).reshape(
                x0.shape[0], -1)
        else:
            pooled = x0.abs().repeat(1, K)
        tau_x = calibrate_quantile(pooled.sort(dim=1, descending=True).values,
                                  n_read_target)
        del pooled

        # score thresholds need the probe under each input scheme
        if args.input_alloc == "router":
            nk0 = allocate_input_reads(sabs0, g0, args.p, 1.0)
            sc_fixed = probe_scores(x0, g0, hits0, nk=nk0, ranks=ranks0)
        else:
            sc_fixed = probe_scores(x0, g0, hits0, xs=sparsify_input_topk(x0, args.p))
        # threshold-input variant: per-slot mask from tau_x
        crit0 = (g0.unsqueeze(-1) * x0.abs().unsqueeze(1) if args.input_alloc == "router"
                 else x0.abs().unsqueeze(1).expand(-1, K, -1))
        m_in0 = crit0 > tau_x                                  # (t,K,H)
        reads0 = float(m_in0.sum(dim=(1, 2)).float().mean())
        sc_thr = torch.zeros_like(sc_fixed)
        for e, slot, tok in hits0:
            cs = x0[tok] * m_in0[tok, slot].float()
            sc_thr[tok, slot] = (F.silu(cs @ Wg_d[e].t())
                                 * (cs @ Wu_d[e].t())).abs() * g0[tok, slot].unsqueeze(-1)
        tau_s = {}
        for rho in rhos:
            Bt = max(1, min(int(round(rho * K * I)), K * I))
            tau_s[("fixed_p", rho)] = calibrate_quantile(
                sc_fixed.reshape(x0.shape[0], -1).sort(dim=1, descending=True).values, Bt)
            tau_s[("thresh_p", rho)] = calibrate_quantile(
                sc_thr.reshape(x0.shape[0], -1).sort(dim=1, descending=True).values, Bt)
        print(f"  calibrated: tau_x={tau_x:.5g} (reads/token {reads0:.1f} vs target "
              f"{n_read_target}), tau_s={ {k[1]: round(v, 6) for k, v in tau_s.items() if k[0]=='fixed_p'} }",
              flush=True)
        del x0, sc_fixed, sc_thr, crit0, m_in0

        # --- measurement pass ------------------------------------------------
        acc = {(m, r): 0.0 for m in MODES for r in rhos}
        nch = {(m, r): 0.0 for m in MODES for r in rhos}
        nrd = {m: 0.0 for m in MODES}
        for st in range(0, T, args.chunk):
            x = X[st:st + args.chunk].to(dev).float()
            t = x.shape[0]
            g = g_all[st:st + args.chunk].to(dev)
            sel = sel_all[st:st + args.chunk].to(dev)
            hits = []
            for e in torch.unique(sel):
                tok, slot = torch.where(sel == int(e))
                hits.append((int(e), slot, tok))

            inter = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
            for e, slot, tok in hits:
                cur = x[tok]
                inter[tok, slot] = F.silu(cur @ Wg_d[e].t()) * (cur @ Wu_d[e].t())
            y_full = torch.zeros((t, H), dtype=torch.float32, device=dev)
            for e, slot, tok in hits:
                y_full.index_add_(0, tok, (inter[tok, slot] @ Wd_d[e].t())
                                  * g[tok, slot].unsqueeze(-1))
            fnorm = y_full.norm(dim=1).clamp_min(1e-30)

            ranks, sabs = descending_abs_ranks(x)
            crit = (g.unsqueeze(-1) * x.abs().unsqueeze(1)
                    if args.input_alloc == "router"
                    else x.abs().unsqueeze(1).expand(-1, K, -1))
            for pmode in ("fixed_p", "thresh_p"):
                if pmode == "fixed_p":
                    if args.input_alloc == "router":
                        nk = allocate_input_reads(sabs, g, args.p, 1.0)
                        score = probe_scores(x, g, hits, nk=nk, ranks=ranks)
                        nread = float(nk.sum(dim=1).float().mean())
                    else:
                        score = probe_scores(x, g, hits,
                                             xs=sparsify_input_topk(x, args.p))
                        nread = float(n_read_target)
                else:
                    m_in = crit > tau_x
                    nread = float(m_in.sum(dim=(1, 2)).float().mean())
                    score = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                    for e, slot, tok in hits:
                        cs = x[tok] * m_in[tok, slot].float()
                        score[tok, slot] = (F.silu(cs @ Wg_d[e].t())
                                            * (cs @ Wu_d[e].t())).abs()
                    score = score * g.unsqueeze(-1)

                flat = score.reshape(t, K * I)
                for Bmode in ("fixed_B", "thresh_B"):
                    mode = f"{Bmode}/{pmode}"
                    nrd[mode] += nread * t
                    for rho in rhos:
                        Bt = max(1, min(int(round(rho * K * I)), K * I))
                        if Bmode == "fixed_B":
                            ii = flat.topk(Bt, dim=1, sorted=False).indices
                            m = torch.zeros_like(flat, dtype=torch.bool).scatter_(
                                1, ii, True)
                        else:
                            m = flat > tau_s[(pmode, rho)]
                        nch[(mode, rho)] += float(m.sum())
                        dropped = inter * (~m.reshape(t, K, I))
                        y_err = torch.zeros((t, H), dtype=torch.float32, device=dev)
                        for e, slot, tok in hits:
                            y_err.index_add_(0, tok, (dropped[tok, slot] @ Wd_d[e].t())
                                             * g[tok, slot].unsqueeze(-1))
                        acc[(mode, rho)] += float((y_err.norm(dim=1) / fnorm).sum())
                del score, flat
            del inter, y_full

        for mode in MODES:
            row = {"layer": layer, "mode": mode, "p": args.p,
                   "input_alloc": args.input_alloc, "n_tokens": T,
                   "reads_per_token": nrd[mode] / T,
                   "reads_target": n_read_target, "tau_x": tau_x}
            for rho in rhos:
                B_eff = nch[(mode, rho)] / T
                rho_eff = B_eff / (K * I)
                p_eff = (nrd[mode] / T) / (K * H)
                row[f"rel_err@rho{rho}"] = acc[(mode, rho)] / T
                row[f"B_eff@rho{rho}"] = B_eff
                row[f"rho_eff@rho{rho}"] = rho_eff
                row[f"kept_eff@rho{rho}"] = used_param_fraction(p_eff, rho_eff)
            rows.append(row)
            print(f"  {mode:18s} reads/tok={row['reads_per_token']:7.1f}  " + "  ".join(
                f"rho{r}: err={row[f'rel_err@rho{r}']:.4f} "
                f"kept={row[f'kept_eff@rho{r}']:.4f}" for r in rhos), flush=True)

        del Wu_d, Wg_d, Wd_d
        torch.cuda.empty_cache()

    # ---- summary: compare at REALIZED cost, not nominal -------------------
    print("\n[summary] layer-averaged, realized cost in parentheses")
    base = None
    for mode in MODES:
        sub = [r for r in rows if r["mode"] == mode]
        if not sub:
            continue
        errs = [float(np.mean([r[f"rel_err@rho{q}"] for r in sub])) for q in rhos]
        kept = [float(np.mean([r[f"kept_eff@rho{q}"] for r in sub])) for q in rhos]
        if mode == MODES[0]:
            base = (errs, kept)
        print(f"  {mode:18s} " + "  ".join(
            f"rho{q}: {errs[i]:.4f} ({kept[i]:.4f})" for i, q in enumerate(rhos)))
    if base:
        print("\n[raw delta vs fixed_B/fixed_p] pt of HellaSwag (slope -26.4), "
              "and the cost difference that came with it")
        for mode in MODES[1:]:
            sub = [r for r in rows if r["mode"] == mode]
            if not sub:
                continue
            errs = [float(np.mean([r[f"rel_err@rho{q}"] for r in sub])) for q in rhos]
            kept = [float(np.mean([r[f"kept_eff@rho{q}"] for r in sub])) for q in rhos]
            print(f"  {mode:18s} " + "  ".join(
                f"rho{q}: {26.4*(base[0][i]-errs[i]):+.2f}pt "
                f"({100*(base[1][i]-kept[i]):+.2f}% cost)"
                for i, q in enumerate(rhos)))

        # ---- the comparison that settles it: ISO-COST ----------------------
        # Thresholding changes the realized cost as well as the error, so a raw
        # delta conflates the two. Interpolate each mode's own (kept, err) curve
        # and read every mode at the *baseline's* cost. A mode only wins if it is
        # below the baseline curve, not merely cheaper.
        print("\n[ISO-COST] each mode's err interpolated to the baseline's realized "
              "cost (lower = better; the baseline curve is the thing to beat)")
        curves = {}
        for mode in MODES:
            sub = [r for r in rows if r["mode"] == mode]
            if not sub:
                continue
            k = np.array([float(np.mean([r[f"kept_eff@rho{q}"] for r in sub]))
                          for q in rhos])
            e = np.array([float(np.mean([r[f"rel_err@rho{q}"] for r in sub]))
                          for q in rhos])
            o = np.argsort(k)
            curves[mode] = (k[o], e[o])
        bk, be = curves[MODES[0]]
        out_iso = {}
        for mode in MODES:
            if mode not in curves:
                continue
            mk, me = curves[mode]
            ei = np.interp(bk, mk, me)          # this mode's err at baseline cost
            out_iso[mode] = ei.tolist()
            tag = "  <- baseline" if mode == MODES[0] else ""
            print(f"  {mode:18s} " + "  ".join(
                f"kept{bk[i]:.4f}: {ei[i]:.4f} "
                f"({26.4*(be[i]-ei[i]):+.2f}pt)" for i in range(len(bk))) + tag)
        wins = [m for m in MODES[1:] if m in out_iso
                and np.mean(out_iso[m]) < np.mean(be) - 1e-4]
        print(f"\n[verdict] modes beating the fixed-B/fixed-p baseline at iso-cost: "
              f"{wins if wins else 'NONE'}")

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "rhos": rhos, "p": args.p,
                   "input_alloc": args.input_alloc, "modes": MODES}, f, indent=2)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
