#!/usr/bin/env python
"""P1 — intrinsic logistic rank of the oracle mask matrix.  ★ project go/no-go

Measures how hard the set-prediction problem is *independently of the expert weights*,
by fitting logistic matrix factorization ``M ≈ σ(U Vᵀ + b)`` to the oracle mask and
sweeping the rank. Three curves land on one axis:

- ``logmf_free``   ``U`` free per token → the **intrinsic rank** of the mask matrix, and
                   an upper bound on any rank-``r`` scorer. Held-out tokens get their
                   ``U`` by solving the per-token logistic regression with ``V`` frozen,
                   so this is "best possible linear-in-embedding scorer", not a
                   memorization artifact.
- ``logmf_linear`` ``U = Qᵀh`` → the bound that is actually *achievable* by the §1.1
                   architecture (a rank-``r`` linear feature map of ``h`` plus free
                   channel embeddings). The free/linear gap is exactly the price of
                   having to compute the embedding from ``h``, and the plan's decision
                   rule is only meaningful against this curve: a low intrinsic rank with
                   no linear realization would be a false GO.
- ``svd`` / ``wsvd`` rank-``r`` truncation of the frozen expert projections, plain and
                   whitened (``Σ^{1/2}``-weighted), which is what a free-embedding router
                   has to beat to justify its parameters.

Decision rule (plan): let ``r*`` be the smallest rank with mass-recall ≥ 95%. Free
embeddings are justified if ``r*(logistic-MF) ≤ 0.5 · r(whitened SVD @ same recall)``.
GO also requires ``r* ≤ 128``.

One GPU, no model load.
"""

import argparse
import json
import os
import sys
import time

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src"))

from src.channel_router import data as D  # noqa: E402
from src.channel_router.baselines import SvdScorer  # noqa: E402
from src.channel_router.metrics import mass_recall, recall, select_topB  # noqa: E402
from src.channel_router.model import whiten_stats  # noqa: E402
from src.channel_router.prep import LayerData  # noqa: E402


class LogisticMF(torch.nn.Module):
    """``score = u_t · v_c + b_c`` with ``u`` free (``mode='free'``) or ``u = Qᵀh``."""

    def __init__(self, EI: int, r: int, *, mode: str, n_tokens: int = 0, H: int = 0,
                 device="cuda"):
        super().__init__()
        self.mode, self.r, self.EI = mode, r, EI
        self.V = torch.nn.Parameter(torch.randn(EI, r, device=device) * 0.02)
        self.b = torch.nn.Parameter(torch.zeros(EI, device=device))
        if mode == "free":
            self.U = torch.nn.Parameter(torch.randn(n_tokens, r, device=device) * 0.02)
        else:
            self.Q = torch.nn.Parameter(torch.randn(H, r, device=device) / H ** 0.5)

    def u_of(self, rows, h):
        return self.U[rows] if self.mode == "free" else h @ self.Q

    def logits(self, u):
        return u @ self.V.t() + self.b                       # (T, EI)


def fit_logmf(ld, r, ratio, *, mode, fit_tokens, eval_tokens, epochs, batch,
              lr, log, solve_steps=200, solve_lr=0.05, cache=None):
    B = ld.budget(ratio)
    dev = ld.device
    EI = ld.E * ld.I
    KI = ld.K * ld.I
    idx_fit = ld.take(ld.train_sl, fit_tokens)
    topb = cache if cache is not None else ld.cache_topb(idx_fit, ratio, log=log)
    model = LogisticMF(EI, r, mode=mode, n_tokens=len(idx_fit), H=ld.H, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    nstep = 0
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(idx_fit))
        tot = 0.0
        for s in range(0, len(idx_fit), batch):
            rows = perm[s:s + batch]
            idx = idx_fit[rows]
            x = ld.X[idx].to(dev, torch.float32)
            sel = ld.sel[idx].to(dev)
            gid = D.global_ids(sel, ld.I).reshape(x.shape[0], -1)
            y = ld.labels_from_topb(topb[rows], KI, dev)
            u = model.u_of(rows.to(dev), x)
            s_act = torch.gather(model.logits(u), 1, gid)
            loss = bce(s_act, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nstep += 1
        if ep % max(1, epochs // 4) == 0 or ep == epochs - 1:
            log(f"    [{mode} r={r}] epoch {ep} loss={tot / max(1, nstep):.4f} "
                f"({time.time() - t0:.0f}s)")
            nstep, tot = 0, 0.0

    # ---- held-out evaluation -------------------------------------------------
    idx_ev = ld.take(ld.val_sl, eval_tokens)
    recs, mrecs = [], []
    with torch.enable_grad():
        for s in range(0, len(idx_ev), batch):
            idx = idx_ev[s:s + batch]
            x, sel, g, imp = ld.batch(idx)
            gid = D.global_ids(sel, ld.I).reshape(x.shape[0], -1)
            y = select_topB(imp, B).reshape(x.shape[0], -1).float()
            if mode == "free":
                # solve the per-token logistic regression with V, b frozen — the fair
                # "best rank-r embedding for this token" bound.
                u = torch.zeros(x.shape[0], r, device=dev, requires_grad=True)
                sopt = torch.optim.Adam([u], lr=solve_lr)
                V = model.V.detach()
                bb = model.b.detach()
                for _ in range(solve_steps):
                    s_act = torch.gather(u @ V.t() + bb, 1, gid)
                    l = bce(s_act, y)
                    sopt.zero_grad(set_to_none=True)
                    l.backward()
                    sopt.step()
                u = u.detach()
                sc = torch.gather(u @ V.t() + bb, 1, gid)
            else:
                with torch.no_grad():
                    sc = torch.gather(model.logits(model.u_of(None, x)), 1, gid)
            sc = sc.reshape(imp.shape)
            pred = select_topB(sc, B)
            ref = select_topB(imp, B)
            recs.append(recall(pred, ref))
            mrecs.append(mass_recall(imp, pred, ref))
    return {"recall": sum(recs) / len(recs), "mass_recall": sum(mrecs) / len(mrecs),
            "rank": r, "mode": mode, "B": B,
            "params": int(model.V.numel() + model.b.numel()
                          + (model.Q.numel() if mode != "free" else 0))}


@torch.no_grad()
def eval_svd(ld, r, ratio, Sh, Sinv, *, whitened, bilinear, eval_tokens, batch, prior):
    B = ld.budget(ratio)
    sc = SvdScorer(ld.w, r, whitened=whitened, Sh=Sh, Sinv=Sinv, bilinear=bilinear,
                   prior=prior, device=str(ld.device))
    recs, mrecs = [], []
    idx_ev = ld.take(ld.val_sl, eval_tokens)
    for s in range(0, len(idx_ev), batch):
        x, sel, g, imp = ld.batch(idx_ev[s:s + batch])
        pred = select_topB(sc.score(x, sel, g), B)
        ref = select_topB(imp, B)
        recs.append(recall(pred, ref))
        mrecs.append(mass_recall(imp, pred, ref))
    return {"recall": sum(recs) / len(recs), "mass_recall": sum(mrecs) / len(mrecs),
            "rank": r, "mode": sc.name, "B": B, "params": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="22,46")
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "docs/results/channel_router/data"))
    ap.add_argument("--tag", default="c4")
    ap.add_argument("--tokens", type=int, default=1 << 20)
    ap.add_argument("--ratios", default="0.125")
    ap.add_argument("--ranks", default="8,16,32,64,128,256")
    ap.add_argument("--modes", default="free,linear")
    ap.add_argument("--fit-tokens", type=int, default=131072)
    ap.add_argument("--eval-tokens", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--solve-steps", type=int, default=200)
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs/results/channel_router/phase0"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True   # fp32 conventions apply to Sigma only
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)

    def log(m):
        print(f"[p1] {m}", flush=True)

    out = {"args": vars(args), "layers": {}}
    for layer in [int(v) for v in args.layers.split(",")]:
        log(f"=== layer {layer}")
        ld = LayerData(args.data_dir, layer, tag=args.tag, tokens=args.tokens,
                       device=args.device, want_down=False)
        log(f"  N={ld.N} E={ld.E} I={ld.I} K={ld.K} H={ld.H}")
        _, Sh, Sinv, _ = whiten_stats(ld.X[ld.train_sl][:262144], device=args.device)
        rows = []
        for ratio in [float(v) for v in args.ratios.split(",")]:
            # one label pass per (layer, ratio) — shared by every rank and mode
            cache = ld.cache_topb(ld.take(ld.train_sl, args.fit_tokens), ratio, log=log)
            for r in [int(v) for v in args.ranks.split(",")]:
                for whit in (False, True):
                    for bil in (False, True):
                        res = eval_svd(ld, r, ratio, Sh, Sinv, whitened=whit,
                                       bilinear=bil, eval_tokens=args.eval_tokens,
                                       batch=args.batch, prior="both")
                        res["ratio"] = ratio
                        rows.append(res)
                        log(f"  {res['mode']:22s} rho={ratio} recall={res['recall']:.4f} "
                            f"mass={res['mass_recall']:.4f}")
                for mode in args.modes.split(","):
                    res = fit_logmf(ld, r, ratio, mode=mode,
                                    fit_tokens=args.fit_tokens,
                                    eval_tokens=args.eval_tokens, epochs=args.epochs,
                                    batch=args.batch, lr=args.lr, log=log,
                                    solve_steps=args.solve_steps, cache=cache)
                    res["ratio"] = ratio
                    res["mode"] = f"logmf_{mode}"
                    rows.append(res)
                    log(f"  {res['mode']:22s} rho={ratio} r={r} "
                        f"recall={res['recall']:.4f} mass={res['mass_recall']:.4f}")
                    with open(os.path.join(args.out_dir, "p1_logistic_rank.json"), "w") as f:
                        json.dump({**out, "layers": {**out["layers"], str(layer): rows}},
                                  f, indent=2)
        out["layers"][str(layer)] = rows
        del ld
        torch.cuda.empty_cache()
    p = os.path.join(args.out_dir, "p1_logistic_rank.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {p}")


if __name__ == "__main__":
    main()
