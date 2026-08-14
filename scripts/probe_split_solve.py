#!/usr/bin/env python3
"""Solve the iso-cost ``(rho_input, rho_channel)`` split from a cached rel_err surface.

Every ``input_sparse`` row in the docs hand-picked ``rho_input=0.25`` and then varied
``rho_channel``, which moves the *split* and the *budget* together. This script asks
the clean question instead: given a used-parameter budget ``C``, how should it be
divided between SCORING (``rho_input``) and COMPUTE (``rho_channel``)?

The program is one Lagrange multiplier -- the *within-layer* analogue of the
cross-layer solve in ``probe_layer_surface.py`` (which failed; see the doc)::

    minimize  rel_err(p, rho)   s.t.   rho + 2p/3 = C
      =>  (3/2) * d(rel_err)/dp  ==  d(rel_err)/d(rho)

The ``3/2`` is exactly the discount ``rho_input`` carries (2 branches -- up+gate --
spread over a 3-matrix FFN), i.e. a unit of ``rho_input`` costs two thirds of a unit
of ``rho_channel``. Unlike a per-layer schedule this keeps ONE global pair for all 48
layers, so it is a *selector* change -- the regime where the rel_err ladder is
validated (+-0.26pt on two pre-registered predictions).

No GPU: reads the cached surface written by ``probe_layer_surface.py``.

Measured outcome at ``C=0.25`` (-75.0%), router alloc, HellaSwag acc_norm:

    opt  rho_in=0.1875 rho_ch=0.12500  ->  74.08   (surface rel_err 0.4613)
    mis  rho_in=0.2500 rho_ch=0.08333  ->  73.80   (surface rel_err 0.4833)

i.e. +0.28pt for the solved split against a predicted +0.58pt. The stderr of the
difference is ~0.62pt, so this confirms the SIGN and bounds the mis-allocation
penalty, but does not establish the magnitude. Read it as "solving the split does not
hurt and probably helps a little".

Usage::

    python scripts/probe_split_solve.py
    python scripts/probe_split_solve.py --budgets 0.25 --step 0.001
    python scripts/probe_split_solve.py --surface docs/results/idea_pilot/layer_surface_8k.json
"""

import argparse
import json

# pt of HellaSwag per unit of rel_err, on the FIXED-BUDGET ruler (selector varies at
# fixed budget). Fitted in scripts/probe_relerr_linearity.py, R^2=0.985. NOT the
# budget-ladder slope (-6.9): mis-selecting at a fixed budget costs 3.8x more per unit
# rel_err than honestly having fewer channels.
LADDER_SLOPE = -26.4

# Qwen3-30B-A3B
K, I, H = 8, 768, 2048


def load_surface(path):
    """Return (ps, rhos, mean_rel_err[i_p][i_rho]) averaged over layers."""
    d = json.load(open(path))
    ps, rhos, surf = d["ps"], d["rhos"], d["surface"]
    n_layers = len(surf)
    mean = [
        [sum(surf[L][i][j] for L in range(n_layers)) / n_layers for j in range(len(rhos))]
        for i in range(len(ps))
    ]
    return ps, rhos, mean, d.get("tokens"), n_layers


def _span(v, arr):
    if v <= arr[0]:
        return 0, 0, 0.0
    if v >= arr[-1]:
        return len(arr) - 1, len(arr) - 1, 0.0
    for k in range(len(arr) - 1):
        if arr[k] <= v <= arr[k + 1]:
            return k, k + 1, (v - arr[k]) / (arr[k + 1] - arr[k])
    raise AssertionError("unreachable")


def relerr(mean, ps, rhos, p, rho):
    """Bilinear interpolation of layer-mean rel_err at an off-grid (p, rho)."""
    i0, i1, ti = _span(p, ps)
    j0, j1, tj = _span(rho, rhos)
    a = mean[i0][j0] * (1 - tj) + mean[i0][j1] * tj
    b = mean[i1][j0] * (1 - tj) + mean[i1][j1] * tj
    return a * (1 - ti) + b * ti


def used(p, rho, n_matrices=2):
    """Used-parameter fraction: scoring + compute. Mirrors sparse_probe.used_param_fraction."""
    return rho + n_matrices * p / 3.0


def solve(mean, ps, rhos, budget, step=0.0025):
    """Best (p, rho) on the budget line rho = C - 2p/3, searched on a fine p grid."""
    best = None
    p = ps[0]
    while p <= ps[-1] + 1e-12:
        rho = budget - 2.0 * p / 3.0
        if rhos[0] <= rho <= rhos[-1]:
            e = relerr(mean, ps, rhos, p, rho)
            if best is None or e < best[0]:
                best = (e, p, rho)
        p += step
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--surface", default="docs/results/idea_pilot/layer_surface_8k.json")
    ap.add_argument(
        "--budgets",
        default="0.20,0.225,0.25,0.2667,0.2917,0.3167,0.3667",
        help="used-parameter budgets (rho_channel + 2*rho_input/3) to solve at",
    )
    ap.add_argument("--step", type=float, default=0.0025, help="rho_input search step")
    ap.add_argument("--out", default=None, help="optional JSON dump")
    args = ap.parse_args()

    ps, rhos, mean, tokens, n_layers = load_surface(args.surface)
    budgets = [float(x) for x in args.budgets.split(",")]

    print(f"surface: {args.surface}  ({n_layers} layers, {tokens} tokens)")
    print(f"  rho_input grid : {ps}")
    print(f"  rho_channel    : {rhos}")
    print("\nPrinciple: min rel_err(p,rho) s.t. rho + 2p/3 = C")
    print("  => (3/2)*d(relerr)/dp == d(relerr)/d(rho)  [equal marginal rel_err per unit budget]\n")

    hdr = (
        f"{'budget':>8} {'cut':>8} | {'rho_in*':>8} {'rho_ch*':>9} {'rel_err':>8} | "
        f"{'score%':>7} {'B':>5} {'reads':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for C in budgets:
        got = solve(mean, ps, rhos, C, args.step)
        if got is None:
            print(f"{C:8.4f} {'--':>8} | (no feasible point on this grid)")
            continue
        e, p, rho = got
        print(
            f"{C:8.4f} {-100*(1-C):7.1f}% | {p:8.4f} {rho:9.5f} {e:8.4f} | "
            f"{100*(2*p/3)/C:6.1f}% {round(rho*K*I):5d} {round(p*H):6d}"
        )
        out[f"{C:.4f}"] = {
            "budget": C,
            "rho_input": p,
            "rho_channel": rho,
            "rel_err": e,
            "scoring_share": (2 * p / 3) / C,
            "B": round(rho * K * I),
            "reads_per_expert": round(p * H),
        }

    # The measured pair at C=0.25, so the script reproduces the doc's comparison.
    if any(abs(C - 0.25) < 1e-9 for C in budgets):
        print("\nMeasured arms at C=0.2500 (router alloc, HellaSwag acc_norm):")
        opt_e = relerr(mean, ps, rhos, 0.1875, 0.125)
        mis_e = relerr(mean, ps, rhos, 0.25, 1.0 / 12.0)
        print(f"  opt 0.1875/0.12500  rel_err {opt_e:.4f}  measured 74.08")
        print(f"  mis 0.2500/0.08333  rel_err {mis_e:.4f}  measured 73.80")
        print(
            f"  predicted gap {LADDER_SLOPE*(opt_e-mis_e):+.2f} pt   measured gap +0.28 pt"
            "   (stderr of difference ~0.62 pt -> sign only)"
        )
        for name, p, rho in (("opt", 0.1875, 0.125), ("mis", 0.25, 1.0 / 12.0)):
            u = used(p, rho)
            assert abs(u - 0.25) < 1e-9, f"{name} is not iso-cost: used={u}"
        print("  both arms verified iso-cost at used=0.2500")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"surface": args.surface, "solutions": out}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
