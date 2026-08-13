#!/usr/bin/env python
"""Probe: can a *learned* low-rank scorer beat spectral truncation at equal cost?

The recall study (``lowrank_scorer_recall.py``) shows plain SVD/BTT factors of
``W_up``/``W_gate`` recover only ~44% of the `oracle_mag_noW` top-B at ρ=0.25 for
a cheap scorer. The diagnosis offered there is that a Frobenius-optimal
truncation optimizes the *wrong objective*: it approximates ``W``, whereas what
matters is the **ranking** of ``|SiLU(W_g x)·(W_u x)|``.

This script tests that diagnosis before committing to it as a next step. Same
online form and therefore the **same cost** as ``svd_r{r}``:

    score_hat(x) = | A_2 · phi(A_1 x) |        A_1: (r, H),  A_2: (I, r)

but ``A_1, A_2`` are *trained* on calibration data against a selection-aware loss
instead of read off the SVD. Three losses, all on the same architecture:

  * ``mse``      — regress the oracle score (magnitude matching)
  * ``listmle``  — listwise softmax cross-entropy on the oracle score
                   distribution (rank matching; the cheap surrogate for top-B)
  * ``topb``     — binary cross-entropy against the oracle's top-B *mask*
                   (directly the decision the block makes)

Reports recall/mass at budget for each, next to the SVD baseline of identical
rank. One MoE layer, one held-out token split. Small: trains on the cached
capture from the recall script, so no model load is needed if that exists.

This is a *probe*, not a method: it trains per (layer, expert) on 8k tokens to
answer "is there headroom above the spectrum?", not to produce a deployable
scorer.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from src.dynamic_active_param.lowrank_scorer import (
    factorize_blocks,
    scorer_cost_fraction,
    scorer_proxy,
)


def _oracle_scores(X, Wg, Wu, chunk=2048):
    """Per-expert oracle channel score |SiLU(Wg x)·(Wu x)| for ONE expert."""
    outs = []
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk]
        outs.append((F.silu(x @ Wg.t()) * (x @ Wu.t())).abs())
    return torch.cat(outs, 0)


def _metrics(pred, target, budgets):
    """recall@B and mass@B of ``pred``'s top-B against ``target``'s top-B."""
    out = {}
    tgt_sorted_cum = target.sort(dim=1, descending=True).values.cumsum(dim=1)
    for B in budgets:
        t_idx = torch.topk(target, B, dim=1, sorted=False).indices
        tm = torch.zeros_like(target, dtype=torch.bool)
        tm.scatter_(1, t_idx, True)
        p_idx = torch.topk(pred, B, dim=1, sorted=False).indices
        recall = (tm.gather(1, p_idx).sum(1).float() / B).mean().item()
        mass = (target.gather(1, p_idx).sum(1)
                / tgt_sorted_cum[:, B - 1].clamp_min(1e-30)).mean().item()
        out[B] = (recall, mass)
    return out


def _train_probe(Xtr, Ytr, r, loss_kind, budgets, iters, lr, device, seed=0,
                 init=None, verbose=False):
    """Train ``|A2 phi(A1 x)|`` on one expert's oracle scores."""
    torch.manual_seed(seed)
    H, I = Xtr.shape[1], Ytr.shape[1]
    if init is not None:
        A1 = init[0].clone().to(device).requires_grad_(True)
        A2 = init[1].clone().to(device).requires_grad_(True)
    else:
        A1 = (torch.randn(r, H, device=device) / H**0.5).requires_grad_(True)
        A2 = (torch.randn(I, r, device=device) / r**0.5).requires_grad_(True)
    opt = torch.optim.Adam([A1, A2], lr=lr)

    # top-B mask targets for the 'topb' loss (use the tightest budget)
    B_t = min(budgets)
    with torch.no_grad():
        tm = torch.zeros_like(Ytr)
        tm.scatter_(1, torch.topk(Ytr, B_t, dim=1, sorted=False).indices, 1.0)
        y_log = torch.log(Ytr.clamp_min(1e-8))

    n = Xtr.shape[0]
    bs = min(512, n)
    for it in range(iters):
        idx = torch.randint(0, n, (bs,), device=device)
        x, y = Xtr[idx], Ytr[idx]
        h = F.silu(x @ A1.t())
        pred = (h @ A2.t()).abs()
        if loss_kind == "mse":
            loss = F.mse_loss(pred / y.mean().clamp_min(1e-8),
                              y / y.mean().clamp_min(1e-8))
        elif loss_kind == "listmle":
            # listwise CE: match the oracle's score distribution over channels
            loss = -(F.log_softmax(pred.clamp_min(1e-8).log(), dim=1)
                     * F.softmax(y_log[idx], dim=1)).sum(1).mean()
        elif loss_kind == "topb":
            # BCE on the oracle's top-B decision, scale-free in pred
            logit = pred.clamp_min(1e-8).log()
            logit = logit - logit.mean(dim=1, keepdim=True)
            loss = F.binary_cross_entropy_with_logits(logit, tm[idx])
        else:
            raise ValueError(loss_kind)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (it + 1) % max(1, iters // 4) == 0:
            print(f"      it {it+1}/{iters} loss={loss.item():.5f}", flush=True)
    return A1.detach(), A2.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=None,
                    help="capture_L<layer>_t<tokens>.pt from lowrank_scorer_recall.py")
    ap.add_argument("--layer", type=int, default=46)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--capture-dir",
                    default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--experts", type=int, default=8,
                    help="#experts to probe (averaged); each is an independent fit")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--ratios", default="0.25,0.125")
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(_REPO,
                    "docs/results/btt_dynamic/learned_probe.json"))
    args = ap.parse_args()

    cap_path = args.capture or os.path.join(
        args.capture_dir, f"capture_L{args.layer}_t{args.tokens}.pt")
    if not os.path.exists(cap_path):
        raise SystemExit(
            f"{cap_path} not found — run scripts/lowrank_scorer_recall.py first "
            "(it caches the per-layer capture this probe reuses).")
    cap = torch.load(cap_path, map_location="cpu")
    X, Wg, Wu = cap["X"], cap["Wg"], cap["Wu"]
    dev = torch.device(args.device)
    E, I, H = Wu.shape
    ratios = [float(x) for x in args.ratios.split(",")]
    # NOTE: budgets here are PER-EXPERT (the probe fits one expert at a time),
    # unlike the pooled K*I budget of the cross-expert block. Recall numbers are
    # therefore not directly comparable to the main table's; what matters is the
    # learned-vs-SVD delta at identical rank and budget.
    budgets = [max(1, int(round(r * I))) for r in ratios]
    n_tr = int(X.shape[0] * (1 - args.holdout))
    Xtr, Xte = X[:n_tr].to(dev), X[n_tr:].to(dev)
    print(f"[probe] layer={args.layer} E={E} I={I} H={H} rank={args.rank} "
          f"train={Xtr.shape[0]} test={Xte.shape[0]} per-expert budgets={budgets}",
          flush=True)
    print(f"[probe] cost of this scorer = {scorer_cost_fraction(I,H,1,1,args.rank):.4f} "
          "of one matmul (same for SVD and learned)", flush=True)

    LOSSES = ["mse", "listmle", "topb"]
    acc = {k: {B: [0.0, 0.0] for B in budgets} for k in ["svd"] + LOSSES}
    n_ex = min(args.experts, E)

    for e in range(n_ex):
        Wg_e, Wu_e = Wg[e].to(dev), Wu[e].to(dev)
        Ytr = _oracle_scores(Xtr, Wg_e, Wu_e)
        Yte = _oracle_scores(Xte, Wg_e, Wu_e)

        # --- SVD baseline of the same rank (up+gate proxy) -------------------
        scu = factorize_blocks(Wu_e.unsqueeze(0), 1, 1, args.rank)
        scg = factorize_blocks(Wg_e.unsqueeze(0), 1, 1, args.rank)
        pred_svd = (F.silu(scorer_proxy(Xte, scg.L_core[0], scg.R_core[0]))
                    * scorer_proxy(Xte, scu.L_core[0], scu.R_core[0])).abs()
        for B, (rc, ms) in _metrics(pred_svd, Yte, budgets).items():
            acc["svd"][B][0] += rc; acc["svd"][B][1] += ms

        # --- learned probes, warm-started from the SVD factors ---------------
        # (A1 = R of W_gate, A2 = L of W_up is a sensible starting point: it
        # reproduces a gate-shaped nonlinearity of the right rank.)
        init = (scg.R_core[0, 0, 0].clone(), scu.L_core[0, 0, 0].clone())
        for kind in LOSSES:
            A1, A2 = _train_probe(Xtr, Ytr, args.rank, kind, budgets,
                                  args.iters, args.lr, dev, init=init)
            pred = (F.silu(Xte @ A1.t()) @ A2.t()).abs()
            for B, (rc, ms) in _metrics(pred, Yte, budgets).items():
                acc[kind][B][0] += rc; acc[kind][B][1] += ms
        print(f"[probe] expert {e+1}/{n_ex} done", flush=True)

    print(f"\n=== layer {args.layer}, rank {args.rank}, "
          f"{n_ex} experts, held-out {Xte.shape[0]} tokens ===")
    hdr = f"{'scorer':10s}" + "".join(
        f"  recall@{r:<6.3f} mass@{r:<6.3f}" for r in ratios)
    print(hdr)
    rows = {}
    for k in ["svd"] + LOSSES:
        cells, rows[k] = "", {}
        for B, r in zip(budgets, ratios):
            rc, ms = acc[k][B][0] / n_ex, acc[k][B][1] / n_ex
            rows[k][str(r)] = {"recall": rc, "mass": ms}
            cells += f"  {rc:13.4f} {ms:12.4f}"
        print(f"{k:10s}{cells}")
    best = max(LOSSES, key=lambda k: rows[k][str(ratios[0])]["recall"])
    d_rec = rows[best][str(ratios[0])]["recall"] - rows["svd"][str(ratios[0])]["recall"]
    d_mass = rows[best][str(ratios[0])]["mass"] - rows["svd"][str(ratios[0])]["mass"]
    print(f"\nbest learned loss = {best!r}: recall {d_rec:+.4f}, mass {d_mass:+.4f} "
          f"vs SVD at rho={ratios[0]} (identical cost)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "layer": args.layer, "rank": args.rank, "n_experts": n_ex,
            "ratios": ratios, "per_expert_budgets": budgets,
            "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
            "iters": args.iters, "lr": args.lr,
            "cost_frac_of_one_matmul": scorer_cost_fraction(I, H, 1, 1, args.rank),
            "results": rows,
        }, f, indent=2)
    print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
