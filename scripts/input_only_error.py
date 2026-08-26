#!/usr/bin/env python
"""Block-output error screen for **`input_only`** — one-pass input sparsity.

``scripts/probe_output_error.py`` measures scorers: the intermediate is exact and
only the keep-*mask* varies, so it can compute the error from the dropped channels
alone (``y_err = sum_e g_e W_down (¬m ⊙ inter)``). ``input_only`` breaks that
shortcut — ``gate``/``up`` read a sparse input, so every kept channel carries a
*value* error too and the approximate output must be built directly::

    x_sp   = the token's top-rho_input coordinates by |x|   (alloc: uniform | router)
    h̃_e    = SiLU(W_gate^(e) x_sp) ⊙ (W_up^(e) x_sp)         <- this IS the compute
    keep   = per-token global top-B of g_e·|h̃_{e,j}|,  B = rho_channel·K·I
    y_appr = sum_e g_e · W_down^(e) (keep ⊙ h̃_e)

    rel_err = mean_t || y_appr(t) − y_full(t) || / || y_full(t) ||

The ``rel_err`` definition matches ``probe_output_error.py`` exactly (per-token
relative L2, averaged over tokens) so the numbers land on the same ladder the doc
fitted, **but read the caveat**: that ladder's −26.4 pt/unit slope was validated for
changes to *which channels a layer selects*. ``input_only`` changes the channel
*values*, which is outside its validated scope, so treat absolute predictions as
indicative and use the screen for what it is reliable at — **ranking iso-cost
splits against each other**.

Why the screen is needed at all: the cost is ``used = (2·rho_input + rho_channel)/3``,
so a unit of ``rho_channel`` costs **half** a unit of ``rho_input`` — the exact
inverse of the two-pass frame, where ``rho_channel`` was the expensive axis. The
standing advice "cut ``rho_input`` first" therefore does not carry over, and the
best split at a fixed budget has to be measured rather than assumed.

Runs off the ``_wd`` captures from ``probe_capture.py``; one GPU, no model load.
Reference rows (``oracle_mag_noW``, and two-pass ``input_sparse`` at the same
``used``) are measured in the same pass so the comparison is iso-instrument.
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
from src.dynamic_active_param.input_only import (
    used_param_fraction as io_used,
)
from src.dynamic_active_param.sparse_probe import (
    allocate_input_reads,
    descending_abs_ranks,
    sparsify_input_by_count,
    sparsify_input_topk,
    used_param_fraction as tp_used,
)

# The doc's solved two-pass splits, for iso-cost reference rows. The -70% entry is
# not a solved optimum (the published curve has no point there); it just puts
# rho_input at the 0.1875 the solve lands on across the deep range and takes
# whatever rho_channel closes the budget.
_TWO_PASS_REF = {
    0.30: (0.1875, 0.17500),      # used 0.3000, NOT solved (interpolated split)
    0.25: (0.1875, 0.12500),      # used 0.2500, solved (doc row 7b / bp_cut750)
    0.20: (0.1575, 0.09500),      # used 0.2000, solved (doc row 7e / bp_cut800)
}


def _sparse_inputs(x, g, rho_input, alloc):
    """Per-(token, slot) sparse inputs.

    Returns either a single ``(T,H)`` tensor (``uniform``: the coordinate set is a
    property of the token, shared by its K slots) or ``(ranks, n_keep)`` for the
    per-slot prefix lengths. Uses the *shipped* allocator from
    ``src.dynamic_active_param.sparse_probe`` so the screen and the eval cannot
    drift apart.
    """
    if alloc == "uniform":
        return sparsify_input_topk(x, rho_input), None, None
    ranks, sorted_abs = descending_abs_ranks(x)
    n_keep = allocate_input_reads(sorted_abs, g, rho_input,
                                  1.0 if alloc == "router" else 2.0)
    return None, ranks, n_keep


def _intermediate(x, Wg, Wu, hits, t, K, I, dev, x_sp=None, ranks=None, nkeep=None):
    """``(t,K,I)`` SwiGLU intermediate; sparse-input when a sparsifier is given."""
    inter = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
    for e, slot, tok in hits:
        if x_sp is not None:
            cur = x_sp[tok]
        elif ranks is not None:
            cur = sparsify_input_by_count(x[tok], ranks[tok], nkeep[tok, slot])
        else:
            cur = x[tok]
        cur = cur.float()
        inter[tok, slot] = F.silu(cur @ Wg[e].t().float()) * (cur @ Wu[e].t().float())
    return inter


def _apply_down(inter, Wd, hits, g, t, H, dev):
    """``sum_e g_e · W_down^(e) inter_e`` -> ``(t, H)``."""
    y = torch.zeros((t, H), dtype=torch.float32, device=dev)
    for e, slot, tok in hits:
        y.index_add_(0, tok, (inter[tok, slot] @ Wd[e].t().float())
                     * g[tok, slot].unsqueeze(-1))
    return y


def _topB(score, B):
    t, K, I = score.shape
    flat = score.reshape(t, K * I)
    idx = flat.topk(B, dim=1, sorted=False).indices
    return torch.zeros_like(flat, dtype=torch.bool).scatter_(1, idx, True).reshape(t, K, I)


def build_plan(budgets, rho_in_grid, allocs):
    """``{(rho_input, alloc): [(rho_channel, budget), ...]}`` — group by the
    expensive half so each ``rho_input`` pays for one gate/up pass, not one per
    ``rho_channel``. ``rho_channel = 3C − 2·rho_input``, dropped when infeasible.
    """
    plan = {}
    for alloc in allocs:
        for p in rho_in_grid:
            for C in budgets:
                r = 3.0 * C - 2.0 * p
                if r <= 1e-9 or r > 1.0:
                    continue
                plan.setdefault((round(p, 6), alloc), []).append((round(r, 6), C))
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,22,38,46")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--budgets", default="0.30,0.25,0.20",
                    help="used-parameter fractions C to sweep on iso-cost lines")
    ap.add_argument("--rho-in", default="0.10,0.15,0.1875,0.20,0.25,0.30,0.35,0.40",
                    help="rho_input grid; rho_channel = 3C - 2*rho_input")
    ap.add_argument("--allocs", default="uniform,router")
    ap.add_argument("--refs", action="store_true", default=True,
                    help="also measure oracle_mag_noW and two-pass input_sparse "
                         "at the same budgets (iso-instrument references)")
    ap.add_argument("--no-refs", dest="refs", action="store_false")
    ap.add_argument("--capture-dir", default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--out", default=os.path.join(_REPO, "docs/results/idea_pilot/input_only_error.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    budgets = [float(x) for x in args.budgets.split(",")]
    rho_in_grid = [float(x) for x in args.rho_in.split(",")]
    allocs = [a for a in args.allocs.split(",") if a]
    dev = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def log(m):
        print(m, flush=True)

    plan = build_plan(budgets, rho_in_grid, allocs)
    log(f"[plan] {len(plan)} (rho_input, alloc) groups, "
        f"{sum(len(v) for v in plan.values())} input_only points")
    # oracle_mag_noW reference at every channel budget any input_only row uses, so
    # each split can be read against "what the exact intermediate would give at
    # this same B" (its own cost is (2+r)/3, i.e. far above these budgets).
    oracle_rhos = sorted({r for v in plan.values() for r, _ in v} | set(budgets))

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
        log(f"\n[layer {layer}] T={T} E={E} I={I} H={H} K={K}")

        g, sel = _route(X, gate_w, K, norm_topk, dev)
        Wu_d, Wg_d, Wd_d = Wu.to(dev), Wg.to(dev), Wd.to(dev)
        del cap, Wu, Wg, Wd

        # accumulators keyed by row name -> summed per-token rel_err
        acc = {}

        def bump(name, meta, val, n):
            r = acc.setdefault(name, {"sum": 0.0, "n": 0, **meta})
            r["sum"] += val
            r["n"] += n

        for s0 in range(0, T, args.chunk):
            x = X[s0:s0 + args.chunk].to(dev)
            t = x.shape[0]
            gc_ = g[s0:s0 + args.chunk].to(dev)
            sc_ = sel[s0:s0 + args.chunk].to(dev)
            hits = []
            for e in torch.unique(sc_):
                tok, slot = torch.where(sc_ == int(e))   # sc_ is (t, K)
                hits.append((int(e), slot, tok))

            # exact intermediate and the reference output
            inter_x = _intermediate(x, Wg_d, Wu_d, hits, t, K, I, dev)
            y_full = _apply_down(inter_x, Wd_d, hits, gc_, t, H, dev)
            nrm = y_full.norm(dim=1).clamp_min(1e-30)

            def rel(y):
                return float(((y - y_full).norm(dim=1) / nrm).sum())

            for (p_in, alloc), rho_list in plan.items():
                x_sp, ranks, nkeep = _sparse_inputs(x, gc_, p_in, alloc)
                inter_sp = _intermediate(x, Wg_d, Wu_d, hits, t, K, I, dev,
                                         x_sp=x_sp, ranks=ranks, nkeep=nkeep)
                score = gc_.unsqueeze(-1) * inter_sp.abs()
                for r_ch, C in rho_list:
                    B = max(1, min(int(round(r_ch * K * I)), K * I))
                    m = _topB(score, B)
                    y = _apply_down(inter_sp * m, Wd_d, hits, gc_, t, H, dev)
                    bump(f"input_only|{alloc}|p{p_in:g}|r{r_ch:g}",
                         dict(method="input_only", alloc=alloc, rho_input=p_in,
                              rho_channel=r_ch, budget=C, used=io_used(p_in, r_ch),
                              B=B),
                         rel(y), t)
                del inter_sp, score

            if args.refs:
                # oracle_mag_noW: exact intermediate, exact ranking. The ceiling of
                # the family at each channel budget, at used=(2+rho_channel)/3.
                sc_ex = gc_.unsqueeze(-1) * inter_x.abs()
                for r_ch in oracle_rhos:
                    B = max(1, min(int(round(r_ch * K * I)), K * I))
                    y = _apply_down(inter_x * _topB(sc_ex, B), Wd_d, hits,
                                    gc_, t, H, dev)
                    bump(f"oracle_mag_noW|r{r_ch:g}",
                         dict(method="oracle_mag_noW", alloc="-", rho_input=1.0,
                              rho_channel=r_ch, budget=None,
                              used=(2.0 + r_ch) / 3.0, B=B),
                         rel(y), t)
                # two-pass input_sparse at the same `used`: the probe picks the
                # channels, the EXACT intermediate is then computed on them. Same
                # budget, one extra pass -- the comparison the method is for.
                for C, (p_in, r_ch) in _TWO_PASS_REF.items():
                    if C not in budgets:
                        continue
                    for alloc in allocs:
                        x_sp, ranks, nkeep = _sparse_inputs(x, gc_, p_in, alloc)
                        inter_pr = _intermediate(x, Wg_d, Wu_d, hits, t, K, I, dev,
                                                 x_sp=x_sp, ranks=ranks, nkeep=nkeep)
                        B = max(1, min(int(round(r_ch * K * I)), K * I))
                        m = _topB(gc_.unsqueeze(-1) * inter_pr.abs(), B)
                        y = _apply_down(inter_x * m, Wd_d, hits, gc_, t, H, dev)
                        bump(f"input_sparse|{alloc}|p{p_in:g}|r{r_ch:g}",
                             dict(method="input_sparse", alloc=alloc,
                                  rho_input=p_in, rho_channel=r_ch, budget=C,
                                  used=tp_used(p_in, r_ch, 2), B=B),
                             rel(y), t)
                        del inter_pr
                del sc_ex
            del inter_x, y_full, nrm

        for name, r in acc.items():
            rows.append({"layer": layer, "name": name,
                         "rel_err": r["sum"] / max(r["n"], 1),
                         "n_tokens": r["n"],
                         **{k: v for k, v in r.items() if k not in ("sum", "n")}})
        for name in sorted(acc):
            rr = [x for x in rows if x["layer"] == layer and x["name"] == name][0]
            log(f"  {name:44s} used={rr['used']:.4f} rel_err={rr['rel_err']:.4f}")

        del Wu_d, Wg_d, Wd_d
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # layer-average, then rank the splits within each budget
    names = sorted({r["name"] for r in rows})
    summary = []
    for name in names:
        rs = [r for r in rows if r["name"] == name]
        summary.append({
            "name": name, "n_layers": len(rs),
            "rel_err": float(np.mean([r["rel_err"] for r in rs])),
            **{k: rs[0][k] for k in ("method", "alloc", "rho_input", "rho_channel",
                                     "budget", "used", "B")},
        })

    log("\n=== layer-averaged (lower rel_err is better) ===")
    for C in budgets:
        log(f"\n-- budget used={C:.4f}  (cut {100 * (1 - C):.1f}%) --")
        grp = sorted([s for s in summary if s["budget"] == C],
                     key=lambda s: s["rel_err"])
        best = grp[0]["rel_err"] if grp else None
        for s in grp:
            sym = " <- symmetric" if (s["method"] == "input_only" and
                                      abs(s["rho_input"] - s["rho_channel"]) < 1e-9) else ""
            log(f"  {s['method']:13s} {s['alloc']:8s} p={s['rho_input']:.4f} "
                f"r={s['rho_channel']:.4f} used={s['used']:.4f} "
                f"rel_err={s['rel_err']:.4f}  Δpt={-26.4 * (s['rel_err'] - best):+.2f}{sym}")

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "summary": summary, "budgets": budgets,
                   "rho_in_grid": rho_in_grid, "allocs": allocs,
                   "ladder_slope_pt_per_unit": -26.4}, f, indent=2)
    log(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
