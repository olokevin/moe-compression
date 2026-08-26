#!/usr/bin/env python
"""Does an activation-eigenbasis rotation concentrate the MoE input better than |x|?

The lm_head study (`docs/exps/lm_head/results_lm_head.md` §3b) found that selecting
input coordinates in the eigenbasis of ``C = E[h hᵀ]`` beats selecting raw
coordinates by a wide margin (adaptive KL 0.287 vs 0.805 at r=256). `input_sparse`
in this package does the *raw* thing: it reads the token's top-``rho_input``
coordinates by ``|x_i|``. This script asks whether the same rotation would help
here, **before** any scorer is built, because the whole idea rests on one
measurable claim:

    a fixed orthogonal U exists in which each token's energy is carried by fewer
    coordinates than in the standard basis.

Since U is orthogonal, ``||x - U_S U_Sᵀ x||² = sum_{i not in S} z_i²`` with
``z = Uᵀ x`` — exactly the same form as the raw rule's ``sum_{i not in S} x_i²``.
So the two are directly comparable at equal ``|S|``: whichever basis captures more
energy per read is the better place to spend the scoring budget.

Four read rules are compared at each budget:

    raw-adaptive     top-r0 of |x_i|             <- what input_sparse does today
    rot-adaptive     top-r0 of |z_i|             <- the S1 screen
    raw-static       top-r0 of E[x_i²] (fixed)   <- the free static prior
    rot-static       top-r0 eigenvalues (fixed)  <- static low-rank = the dead family

Plus ``rot-adaptive`` under a **global** basis shared by all layers, which is the
variant whose rotation is free at run time (rebase the residual stream once,
QuaRot-style, and every weight reading it absorbs U offline).

Also reports the rotated column-norm spread ``CV_i ||W u_i||``: the raw-basis
column norms have CV 0.022 (which is why `colnorm` allocation was a wash), and the
``|z_i|·||W u_i||`` score of lm_head gate 0g only matters if the rotated basis
breaks that uniformity.

CPU-only, no model load. Needs the ``_wd`` captures from ``probe_capture.py``.
"""

import argparse
import json
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def energy_fracs(vals_sq: torch.Tensor, budgets, static_order=None):
    """Mean captured energy fraction per token for each budget.

    Args:
        vals_sq: ``(T, H)`` per-coordinate squared magnitudes in some basis.
        budgets: iterable of read counts.
        static_order: ``(H,)`` long, a *fixed* coordinate order; if given the same
            prefix is used for every token instead of a per-token top-k.
    """
    total = vals_sq.sum(dim=1).clamp_min(1e-30)
    out = {}
    if static_order is None:
        srt, _ = torch.sort(vals_sq, dim=1, descending=True)
        cum = torch.cumsum(srt, dim=1)
        for r in budgets:
            out[r] = float((cum[:, r - 1] / total).mean())
    else:
        cum = torch.cumsum(vals_sq[:, static_order], dim=1)
        for r in budgets:
            out[r] = float((cum[:, r - 1] / total).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,22,38,46")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--budgets", default="64,128,192,256,299,384,512,768")
    ap.add_argument("--capture-dir",
                    default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--out", default=os.path.join(
        _REPO, "docs/results/idea_pilot/rotate_diag.json"))
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 8)

    # ---- pass 1: per-layer covariance + the global (all-layer) covariance ------
    Xs, Cs = {}, {}
    C_glob = None
    for L in layers:
        p = os.path.join(args.capture_dir, f"capture_L{L}_t{args.tokens}_wd.pt")
        if not os.path.exists(p):
            print(f"[skip] no capture {p}", flush=True)
            continue
        cap = torch.load(p, map_location="cpu", weights_only=False)
        X = cap["X"].double()                                   # (T,H)
        Xs[L] = X
        # normalize each layer's contribution to the global basis: the residual
        # stream grows with depth, so an unnormalized sum would be one layer's C.
        C = X.t() @ X / X.shape[0]
        Cs[L] = C
        Cn = C / C.diagonal().sum()
        C_glob = Cn if C_glob is None else C_glob + Cn
        # column norms live with the weights; grab them here to avoid a second load
        Wu, Wg = cap["Wu"], cap["Wg"]
        Cs[(L, "cols")] = (Wu, Wg)
        del cap

    if not Xs:
        print("[abort] no captures found")
        return

    _, U_glob = torch.linalg.eigh(C_glob)
    U_glob = torch.flip(U_glob, dims=[1]).contiguous()          # desc eigenvalue

    rows = []
    for L in sorted(Xs):
        X = Xs[L]
        T, H = X.shape
        lam, U = torch.linalg.eigh(Cs[L])
        lam, U = torch.flip(lam, dims=[0]), torch.flip(U, dims=[1]).contiguous()
        Z = X @ U
        Zg = X @ U_glob

        x_sq, z_sq, zg_sq = X * X, Z * Z, Zg * Zg
        mean_x_sq = x_sq.mean(dim=0)
        raw_static_order = torch.argsort(mean_x_sq, descending=True)
        rot_static_order = torch.arange(H)                     # eigenvalues desc
        rotg_static_order = torch.argsort(zg_sq.mean(dim=0), descending=True)

        res = {
            "raw_adaptive": energy_fracs(x_sq, budgets),
            "rot_adaptive": energy_fracs(z_sq, budgets),
            "rotglob_adaptive": energy_fracs(zg_sq, budgets),
            "raw_static": energy_fracs(x_sq, budgets, raw_static_order),
            "rot_static": energy_fracs(z_sq, budgets, rot_static_order),
            "rotglob_static": energy_fracs(zg_sq, budgets, rotg_static_order),
        }

        # rotated column-norm spread, rms over experts and both branches
        Wu, Wg = Cs[(L, "cols")]
        acc = torch.zeros(H, dtype=torch.float64)
        accraw = torch.zeros(H, dtype=torch.float64)
        n = 0
        for W in (Wu, Wg):
            for e in range(W.shape[0]):
                We = W[e].double()                              # (I,H)
                acc += ((We @ U) ** 2).mean(dim=0)
                accraw += (We ** 2).mean(dim=0)
                n += 1
        rot_cn = (acc / n).sqrt()
        raw_cn = (accraw / n).sqrt()
        cv = lambda v: float(v.std() / v.mean())

        row = {"layer": L, "T": T, "H": H,
               "rot_colnorm_cv": cv(rot_cn), "raw_colnorm_cv": cv(raw_cn),
               "eig_top1_frac": float(lam[0] / lam.sum()),
               "eig_top64_frac": float(lam[:64].sum() / lam.sum()),
               **{k: {str(r): v for r, v in d.items()} for k, d in res.items()}}
        rows.append(row)

        print(f"\n[layer {L}] T={T} H={H}  "
              f"colnorm CV raw={row['raw_colnorm_cv']:.3f} "
              f"rotated={row['rot_colnorm_cv']:.3f}  "
              f"eig: top1={row['eig_top1_frac']:.3f} top64={row['eig_top64_frac']:.3f}")
        hdr = "  ".join(f"{r:>7d}" for r in budgets)
        print(f"  {'captured energy':<20s} {hdr}")
        for k in ("raw_adaptive", "rot_adaptive", "rotglob_adaptive",
                  "raw_static", "rot_static", "rotglob_static"):
            print(f"  {k:<20s} " + "  ".join(f"{res[k][r]:7.4f}" for r in budgets))
        print(f"  {'residual ratio':<20s} " + "  ".join(
            f"{(1 - res['rot_adaptive'][r]) / max(1e-12, 1 - res['raw_adaptive'][r]):7.3f}"
            for r in budgets) + "   (rot/raw dropped energy; <1 favours rotation)")
        del X, Z, Zg, x_sq, z_sq, zg_sq
        Xs[L] = None

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "budgets": budgets}, f, indent=2)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
