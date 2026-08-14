#!/usr/bin/env python
"""Unstructured vs column-structured scoring at an identical read budget.

`input_sparse` spends its scoring budget on whole *coordinates*: the token's
top-``rho_input`` columns of ``up``/``gate``, read for all ``I`` channels. But the
exact contribution of entry ``(j, i)`` is ``W_ji·x_i``, so at a fixed number of
reads the greedy-optimal set is the top of the **product** ``|W_ji|·|x_i|`` — which
no column rule can express. This screen asks whether that freedom is worth
anything, at matched cost.

Everything is one family (``src/dynamic_active_param/weight_sparse.py``): an
``L``-band staircase where band ``l`` covers ``col_frac[l]`` of the token's
coordinates and reads ``row_frac[l]`` of the channels for each of them. Its two
extreme points are the schemes being compared —

    0.1125x1.0     `input_sparse` at rho_input=0.1125   (few columns, all rows)
    1.0x0.1125     a static per-column weight mask      (all columns, few rows)

— and everything between is a graded read set, all at the **same** per-branch
density 0.1125, i.e. ``used = 0.125 + 2·0.1125/3 = 0.20``: the 20%-of-dense budget
this study is aimed at, with the 12.5% channel budget the full-width oracle uses.

Metric is block-output ``rel_err`` (the ladder the repo validated against measured
accuracy at fixed budget, slope −26.4 pt/unit, R² 0.985), never index recall.

Reads the cached ``_wd`` captures from ``scripts/probe_capture.py``. One GPU, no
model load.

Variants (``--variants``, comma-separated):

    stair:<spec>[:router][:mf]   staircase; ``spec`` like ``0.0625x1.0+0.25x0.2``
    glob:<density>               ONE static global |W|·sigma_i threshold per expert
                                 (unstructured but unbalanced across columns/rows)
    taux:<density>:<ladder>       exact per-column pricing (reference for tau)
    prod:<density>               the exact per-token |W_ji·x_i| top-N set — the
                                 ceiling of the family. Expensive: it materializes
                                 an (I,H) mask per (token, expert). Use
                                 ``--prod-tokens`` to cap it.
    exact                        oracle_mag_noW reference (density 2.0)
    random                       floor
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts.idea_pilot_scorers import _route
from src.dynamic_active_param.sparse_probe import (
    allocate_input_reads,
    descending_abs_ranks,
)
from src.dynamic_active_param.weight_sparse import (
    build_layer_wsparse,
    levels_density,
    parse_levels,
    wsparse_expert_scores,
    wsparse_used_param_fraction,
)


def geo_levels(n_bands: int, density: float, col_max: float = 0.5):
    """Equal-contribution geometric staircase at an exact target density.

    Coordinate bands double in width up to ``col_max`` (so the token's strongest
    few coordinates get their own narrow band), and each band is given the same
    read budget ``c``: ``row_frac = min(1, c/col_frac)``. ``c`` is bisected until
    the total density hits the target; bands that clip at ``row_frac=1`` under-spend,
    so the surplus lands on the widest band. This is the discretization of the
    ``|W_ji·x_i|`` product rule — as ``n_bands`` grows it converges to ``prod``.
    """
    cols = [col_max / (2 ** (n_bands - 1 - l)) for l in range(n_bands)]

    def build(c):
        return [(cf, min(1.0, c / cf)) for cf in cols]

    lo, hi = 0.0, 1.0
    for _ in range(60):
        c = 0.5 * (lo + hi)
        if sum(a * b for a, b in build(c)) < density:
            lo = c
        else:
            hi = c
    lv = build(0.5 * (lo + hi))
    # nudge the widest band so the density is exact despite the row_frac=1 clips
    d = sum(a * b for a, b in lv)
    cf, rf = lv[-1]
    lv[-1] = (cf, min(1.0, rf + (density - d) / cf))
    return tuple(lv)


def _experts_shim(Wu, Wg):
    """Wrap stacked ``(E, I, H)`` capture weights as expert modules.

    ``build_layer_wsparse`` only touches ``e.<branch>.weight``, so the screen can
    drive the *production* builder and scorer instead of a re-implementation —
    which is the point: whatever this measures is what the eval will run.
    """
    return [
        SimpleNamespace(up_proj=SimpleNamespace(weight=Wu[e]),
                        gate_proj=SimpleNamespace(weight=Wg[e]))
        for e in range(Wu.shape[0])
    ]


# ---------------------------------------------------------------------------
# the two non-staircase read sets
# ---------------------------------------------------------------------------

def global_masks(Wu, Wg, sigma, density):
    """One static ``|W_ji|·sigma_i`` threshold per (expert, branch).

    The Wanda criterion, applied *globally* over the whole matrix instead of
    per column: high-variance coordinates and high-magnitude channels both attract
    entries, so neither columns nor rows are balanced. This is the natural "static
    unstructured mask" baseline, and the control for whether the per-column balance
    the staircase enforces costs anything.
    """
    out = []
    for W in (Wu, Wg):
        keys = W.abs() * sigma.view(1, 1, -1)
        E = W.shape[0]
        # stride-subsample for the quantile: torch.quantile caps input size, and a
        # 1-in-16 sample of 1.5M entries fixes the threshold to well under 1% of
        # the target density (the realized density is measured and reported anyway).
        thr = torch.stack([
            torch.quantile(keys[e].flatten()[::16].float(), 1.0 - density)
            for e in range(E)
        ])                                                     # (E,)
        out.append(keys >= thr.view(-1, 1, 1))
    return out


def prod_threshold_scores(x, Wu, Wg, WsortT_u, WsortT_g, density, iters=24):
    """Exact per-token top-``density·I·H`` entries by ``|W_ji·x_i|``, per branch.

    The count above a threshold ``tau`` is ``sum_i #{j : |W_ji| > tau/|x_i|}``,
    which is a searchsorted on each column's ascending ``|W|`` order — so the
    budget-exact ``tau`` is a cheap bisection, and only the accumulation is
    expensive (an ``(I, H)`` mask per token).

    Returns ``(T, I)`` ``|SiLU(gate_hat)·up_hat|``.
    """
    I, H = Wu.shape
    budget = density * I * H
    ax = x.abs().clamp_min(1e-20)                              # (T,H)
    outs = []
    for W, WsT in ((Wu, WsortT_u), (Wg, WsortT_g)):
        hi = (W.abs().amax(dim=0).unsqueeze(0) * ax).amax(dim=1, keepdim=True)
        lo = torch.zeros_like(hi)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            thr_col = (mid / ax).t().contiguous()               # (H,T)
            n = (I - torch.searchsorted(WsT, thr_col, right=True)).sum(dim=0)
            too_many = (n > budget).unsqueeze(-1)
            lo = torch.where(too_many, mid, lo)
            hi = torch.where(too_many, hi, mid)
        tau = 0.5 * (lo + hi)                                   # (T,1)
        acc = torch.empty((x.shape[0], I), dtype=torch.float32, device=x.device)
        for t in range(x.shape[0]):
            wx = W * x[t].unsqueeze(0)                          # (I,H)
            acc[t] = (wx * (wx.abs() >= tau[t])).sum(dim=1)
        outs.append(acc)
    return (F.silu(outs[1]) * outs[0]).abs()


def merge_and_report(paths, out):
    """Layer-average per-layer runs (one GPU each) into one summary table.

    Also prints the used-parameter fraction next to every ``rel_err``, since the
    whole comparison is iso-cost and a row is meaningless without its cost.
    """
    rows, rhos = [], None
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        rows += d["rows"]
        rhos = d["rhos"] if rhos is None else rhos
    variants = []
    for r in rows:
        if r["variant"] not in variants:
            variants.append(r["variant"])
    summ = {}
    for v in variants:
        sub = [r for r in rows if r["variant"] == v]
        summ[v] = {
            "layers": sorted(r["layer"] for r in sub),
            "density": float(np.mean([r["density_per_branch"] for r in sub])),
            "rel_err": {r_: float(np.mean([s[f"rel_err@rho{r_}"] for s in sub]))
                        for r_ in rhos},
            "used": {r_: float(np.mean([s[f"used@rho{r_}"] for s in sub]))
                     for r_ in rhos},
        }
    base = "stair:0.1125x1.0"
    print(f"layer-averaged over {len(paths)} layers "
          f"({summ[variants[0]]['layers']}); Deltapt uses the fixed-budget slope -26.4")
    head = "".join(f"{'rho' + str(r) :>10s}{'used':>8s}" for r in rhos)
    print(f"  {'variant':56s}{'dens':>8s}{head}{'  dpt@.125':>11s}")
    for v, s in summ.items():
        dpt = ""
        if base in summ and v != base and 0.125 in s["rel_err"]:
            dpt = f"{26.4 * (summ[base]['rel_err'][0.125] - s['rel_err'][0.125]):+11.2f}"
        print(f"  {v:56s}{s['density']:8.4f}"
              + "".join(f"{s['rel_err'][r]:10.4f}{s['used'][r]:8.3f}" for r in rhos)
              + dpt)
    with open(out, "w") as f:
        json.dump({"rows": rows, "summary": summ, "rhos": rhos,
                   "merged_from": list(paths)}, f, indent=2)
    print(f"\n[done] wrote {out}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,22,38,46")
    ap.add_argument("--tokens", type=int, default=8192, help="capture file token count")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--rho-channels", "--rhos", dest="rhos", default="0.10,0.125,0.15")
    ap.add_argument("--variants", default=(
        # all at per-branch density 0.1125 -> used = 0.20 at rho_channel = 0.125
        "exact,random,"
        "stair:0.1125x1.0,"                       # = input_sparse (column-only)
        "stair:0.225x0.5,"
        "stair:0.25x0.45,"
        "stair:0.5x0.225,"
        "stair:1.0x0.1125,"                       # = static per-column weight mask
        "stair:0.0625x1.0+0.25x0.2,"
        "stair:0.0625x1.0+0.125x0.2+0.375x0.0667,"
        "stair:0.03125x1.0+0.0625x0.5+0.125x0.25+0.375x0.05,"
        "stair:0.25x0.45:router,"
        "stair:0.25x0.45:mf"
    ))
    ap.add_argument("--prod-tokens", type=int, default=0,
                    help=">0 runs the exact product-threshold ceiling on this many "
                         "tokens (expensive: an (I,H) mask per token per expert)")
    ap.add_argument("--capture-dir", default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--out", default=os.path.join(
        _REPO, "docs/results/idea_pilot/wsparse_screen.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--merge", nargs="*", default=None,
                    help="merge per-layer result JSONs and print the summary "
                         "(no GPU); one file per layer from parallel runs")
    args = ap.parse_args()

    if args.merge:
        merge_and_report(args.merge, args.out)
        return

    layers = [int(v) for v in args.layers.split(",")]
    rhos = [float(v) for v in args.rhos.split(",")]
    variants = [v for v in args.variants.split(",") if v]
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
        mu = X.to(dev).float().mean(dim=0)                     # (H,) calibration mean
        sigma = X.to(dev).float().std(dim=0)                   # (H,) per-coordinate std
        print(f"\n[layer {layer}] T={T} E={E} I={I} H={H} K={K} "
              f"|mu|/rms(x)={float(mu.norm() / X.to(dev).float().norm(dim=1).mean()):.4f}",
              flush=True)

        # --- build every variant's scorer once per layer ----------------------
        shim = _experts_shim(Wu_d, Wg_d)
        built = {}
        for v in variants:
            parts = v.split(":")
            kind = parts[0]
            if kind == "stairup":
                # single-branch (up only) staircase: n_matrices=1, so the same
                # used-param budget buys twice the read density.
                spec = parts[1]
                probe = build_layer_wsparse(shim, levels=spec, use_gate=False)
                built[v] = ("stair", probe, levels_density(spec), 1)
            elif kind == "geo":
                # geo:<n_bands>:<density>[:col_max][:router|mf ...]
                nb, dens_t = int(parts[1]), float(parts[2])
                rest = parts[3:]
                cmax = 0.5
                if rest and rest[0].replace(".", "", 1).isdigit():
                    cmax, rest = float(rest[0]), rest[1:]
                spec = geo_levels(nb, dens_t, cmax)
                flags = set(rest)
                print(f"  [geo] {v} -> "
                      + "+".join(f"{a:g}x{b:.4g}" for a, b in spec)
                      + f"  density={levels_density(spec):.4f}", flush=True)
                probe = build_layer_wsparse(
                    shim, levels=spec, use_gate=True,
                    mu=mu if "mf" in flags else None,
                    input_alloc="router" if "router" in flags else "uniform",
                )
                built[v] = ("stair", probe, levels_density(spec), 2)
            elif kind in ("tau", "taux"):
                # tau:<density>:<rf|rf|...>[:router][:mf][:up]
                # "tau"  = band edges from a scalar price ladder (the eval path)
                # "taux" = exact per-column |W_ji*x_i| pricing (reference)
                dens_t = float(parts[1])
                rfs = [float(z) for z in parts[2].split("|")]
                flags = set(parts[3:])
                rb = 1
                for f in list(flags):
                    if f.startswith("b"):        # b8 = channel blocks of 8
                        rb = int(f[1:]); flags.discard(f)
                probe = build_layer_wsparse(
                    shim, levels=tuple((0.0, rf) for rf in rfs),
                    use_gate="up" not in flags,
                    mu=mu if "mf" in flags else None,
                    input_alloc="router" if "router" in flags else "uniform",
                    alloc_mode=kind, density=dens_t, count_reads=True, row_block=rb,
                    tau_iters=int(os.environ.get("WSPARSE_TAU_ITERS", 16)),
                )
                built[v] = ("stair", probe, dens_t, 1 if "up" in flags else 2)
            elif kind == "stair":
                spec = parts[1]
                flags = set(parts[2:])
                probe = build_layer_wsparse(
                    shim, levels=spec, use_gate=True,
                    mu=mu if "mf" in flags else None,
                    input_alloc="router" if "router" in flags else "uniform",
                )
                built[v] = ("stair", probe, levels_density(spec), 2)
            elif kind == "glob":
                d = float(parts[1])
                masks = global_masks(Wu_d, Wg_d, sigma, d)
                d_real = float(0.5 * (masks[0].float().mean() + masks[1].float().mean()))
                print(f"  [glob] target density {d:.4f} -> realized {d_real:.4f}",
                      flush=True)
                built[v] = ("glob", masks, d_real, 2)
            elif kind == "prod":
                d = float(parts[1])
                WsT_u = torch.sort(Wu_d.abs(), dim=1).values.transpose(1, 2).contiguous()
                WsT_g = torch.sort(Wg_d.abs(), dim=1).values.transpose(1, 2).contiguous()
                built[v] = ("prod", (WsT_u, WsT_g), d, 2)
            elif kind in ("exact", "random"):
                built[v] = (kind, None, 1.0 if kind == "exact" else 0.0, 2)
            else:
                raise ValueError(f"unknown variant {v}")

        acc = {(v, r): 0.0 for v in variants for r in rhos}
        seen = {v: 0 for v in variants}
        for s0 in range(0, T, args.chunk):
            x = X[s0:s0 + args.chunk].to(dev).float()
            t_n = x.shape[0]
            g = g_all[s0:s0 + args.chunk].to(dev)
            sel = sel_all[s0:s0 + args.chunk].to(dev)
            hits = []
            for e in torch.unique(sel):
                tok, slot = torch.where(sel == int(e))
                hits.append((int(e), slot, tok))

            # exact intermediate + true block output, shared by every variant
            inter = torch.zeros((t_n, K, I), dtype=torch.float32, device=dev)
            for e, slot, tok in hits:
                cur = x[tok]
                inter[tok, slot] = F.silu(cur @ Wg_d[e].t()) * (cur @ Wu_d[e].t())
            y_full = torch.zeros((t_n, H), dtype=torch.float32, device=dev)
            for e, slot, tok in hits:
                y_full.index_add_(0, tok, (inter[tok, slot] @ Wd_d[e].t())
                                  * g[tok, slot].unsqueeze(-1))
            fnorm = y_full.norm(dim=1).clamp_min(1e-30)

            ranks_x, sorted_abs_x = descending_abs_ranks(x)
            ranks_d, sorted_abs_d = descending_abs_ranks(x - mu)

            for v in variants:
                kind, obj, dens, _nm = built[v]
                if kind == "prod" and args.prod_tokens and s0 >= args.prod_tokens:
                    continue
                proxy = torch.zeros((t_n, K, I), dtype=torch.float32, device=dev)
                if kind == "exact":
                    proxy = inter.abs()
                elif kind == "random":
                    proxy = torch.rand((t_n, K, I), device=dev)
                elif kind == "stair":
                    probe = obj
                    rk = ranks_d if probe.mu is not None else ranks_x
                    sa = sorted_abs_d if probe.mu is not None else sorted_abs_x
                    nk = (allocate_input_reads(sa, g, probe.col_total, 1.0)
                          if probe.input_alloc == "router" else None)
                    for e, slot, tok in hits:
                        proxy[tok, slot] = wsparse_expert_scores(
                            x[tok], probe, e, ranks=rk[tok],
                            n_cols=None if nk is None else nk[tok, slot])
                elif kind == "glob":
                    mu_ = obj
                    for e, slot, tok in hits:
                        uh = x[tok] @ (Wu_d[e] * mu_[0][e]).t()
                        gh = x[tok] @ (Wg_d[e] * mu_[1][e]).t()
                        proxy[tok, slot] = (F.silu(gh) * uh).abs()
                elif kind == "prod":
                    WsT_u, WsT_g = obj
                    for e, slot, tok in hits:
                        proxy[tok, slot] = prod_threshold_scores(
                            x[tok], Wu_d[e], Wg_d[e], WsT_u[e], WsT_g[e], dens)
                seen[v] += t_n

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
                    acc[(v, rho)] += float((y_err.norm(dim=1) / fnorm).sum())
                del proxy, score, flat
            del inter, y_full
            print(f"  ..{min(s0 + args.chunk, T)}/{T}", end="\r", flush=True)

        for v in variants:
            kind, obj, dens, n_mat = built[v]
            n = max(seen[v], 1)
            # threshold mode floats the read set, so report what it actually read
            if kind == "stair" and getattr(obj, "count_reads", False) and obj.reads_n:
                dens_real = obj.reads_sum / obj.reads_n
                print(f"  [tau] {v}: realized density {dens_real:.4f} "
                      f"(target {dens:.4f})", flush=True)
                dens = dens_real
            row = {"layer": layer, "variant": v, "kind": kind,
                   "density_per_branch": dens, "n_matrices": n_mat, "n_tokens": n}
            for rho in rhos:
                row[f"rel_err@rho{rho}"] = acc[(v, rho)] / n
                # exact reads both branches in full (density 1.0 each); random
                # reads nothing. Everything else pays n_mat * density / 3.
                row[f"used@rho{rho}"] = rho + n_mat * dens / 3
            rows.append(row)
            print(f"  {v:46s} d={dens:.4f} " + "  ".join(
                f"rho{r}:{row[f'rel_err@rho{r}']:.4f}" for r in rhos), flush=True)

        del Wu_d, Wg_d, Wd_d, built, shim
        torch.cuda.empty_cache()

    # ---- layer-averaged summary -------------------------------------------
    print("\n[summary] layer-averaged rel_err (lower is better); "
          "Deltapt vs the column-only row uses the fixed-budget slope -26.4")
    base_name = "stair:0.1125x1.0"
    summ = {}
    for v in variants:
        sub = [r for r in rows if r["variant"] == v]
        if not sub:
            continue
        summ[v] = {
            "density": sub[0]["density_per_branch"],
            "rel_err": {r: float(np.mean([s[f"rel_err@rho{r}"] for s in sub]))
                        for r in rhos},
            "used": {r: sub[0][f"used@rho{r}"] for r in rhos},
        }
    hdr = "".join(f"{'rho'+str(r):>11s}" for r in rhos)
    print(f"  {'variant':46s}{'dens':>8s}{hdr}{'  d_pt@0.125':>13s}")
    for v, s in summ.items():
        dpt = ""
        if base_name in summ and v != base_name and 0.125 in s["rel_err"]:
            dpt = f"{26.4 * (summ[base_name]['rel_err'][0.125] - s['rel_err'][0.125]):+13.2f}"
        print(f"  {v:46s}{s['density']:8.4f}"
              + "".join(f"{s['rel_err'][r]:11.4f}" for r in rhos) + dpt)

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "summary": {k: v for k, v in summ.items()},
                   "rhos": rhos, "variants": variants,
                   "layers": layers, "max_tokens": args.max_tokens}, f, indent=2)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
