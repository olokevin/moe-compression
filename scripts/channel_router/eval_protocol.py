#!/usr/bin/env python
"""§1.3 + §3.1 results table: every method, every budget, one protocol.

For each method a row reports (params, online FLOPs/token, recall@k, mass-recall,
block-output rel_err) at the oracle reference budget ``k`` and across the slack sweep
``s ∈ {0.9, 1.0, 1.15, 1.25}`` (predicted budget ``s·k``, oracle reference always ``k``,
so the slack columns answer "how much extra budget buys oracle-quality output").

Trainable baselines (product-key, Deja-Vu MLP) are trained here with the *same* loss and
token budget as the router, otherwise the comparison would be between a trained method
and untrained ones. Training-free baselines (static, SVD, whitened SVD, random
projection, LSH, VQ table) are built from the frozen weights or from calibration
statistics only.

One GPU; needs the ``want_down`` weights for the output-error column.
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

from src.channel_router.baselines import (  # noqa: E402
    DejaVuMLP, LshScorer, ProductKeyScorer, RandomProjScorer, StaticFreq, SvdScorer,
    VqScorer,
)
from src.channel_router.metrics import (  # noqa: E402
    mass_recall, output_rel_err, recall, router_accounting, select_topB,
)
from src.channel_router.model import whiten_stats  # noqa: E402
from src.channel_router.prep import LayerData  # noqa: E402
from src.channel_router.train_utils import load_router_artifact, ranking_loss  # noqa: E402


def fit_scorer(scorer, ld, ratio, *, steps, batch, lr, delta, w_fn, log, name):
    """Train a baseline that has parameters, with the Stage-B loss."""
    B = ld.budget(ratio)
    idx_fit = ld.take(ld.train_sl)
    ext = min(1.0, (B + delta) / (ld.K * ld.I))
    n_cache = min(len(idx_fit), steps * batch)
    idx_fit = idx_fit[:n_cache]
    topk_idx = ld.cache_topb(idx_fit, ext, batch=2048)
    opt = torch.optim.AdamW(scorer.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        rows = torch.randint(0, len(idx_fit), (batch,))
        idx = idx_fit[rows]
        x = ld.X[idx].to(ld.device, torch.float32)
        sel = ld.sel[idx].to(ld.device)
        g = ld.g[idx].to(ld.device)
        loss = ranking_loss(scorer.score(x, sel, g), topk_idx[rows].to(ld.device), B,
                            kind="margin", delta=delta, w_fn=w_fn)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 4) == 0:
            log(f"    fit[{name}] step{step}/{steps} loss={float(loss.detach()):.4f} "
                f"({time.time() - t0:.0f}s)")
    return scorer


@torch.no_grad()
def fit_vq(ld, n_centroids, ratio, *, fit_tokens, iters=15, batch=4096, log=print):
    """k-means on ``h`` + per-centroid mean log-importance profile (the lookup baseline)."""
    X = ld.X[ld.train_sl][:fit_tokens].to(ld.device, torch.float32)
    gen = torch.Generator(device="cpu").manual_seed(0)
    mu = X[torch.randperm(X.shape[0], generator=gen)[:n_centroids]].clone()
    for it in range(iters):
        assign = torch.cat([torch.cdist(X[s:s + 8192], mu).argmin(1)
                            for s in range(0, X.shape[0], 8192)])
        for c in range(n_centroids):
            m = assign == c
            if m.any():
                mu[c] = X[m].mean(0)
    del X
    table = torch.zeros((n_centroids, ld.E * ld.I), device=ld.device)
    cnt = torch.zeros((n_centroids, ld.E * ld.I), device=ld.device)
    idx_all = ld.take(ld.train_sl, fit_tokens)
    for s in range(0, len(idx_all), batch):
        x, sel, g, imp = ld.batch(idx_all[s:s + batch])
        c = torch.cdist(x, mu).argmin(1)                          # (T,)
        gid = (sel.unsqueeze(-1) * ld.I
               + torch.arange(ld.I, device=x.device).view(1, 1, ld.I)).reshape(x.shape[0], -1)
        lg = imp.reshape(x.shape[0], -1).clamp_min(1e-20).log()
        table.index_put_((c.unsqueeze(1).expand_as(gid), gid), lg, accumulate=True)
        cnt.index_put_((c.unsqueeze(1).expand_as(gid), gid),
                       torch.ones_like(lg), accumulate=True)
    table = table / cnt.clamp_min(1)
    table[cnt == 0] = -30.0
    log(f"    vq: {n_centroids} centroids fitted")
    return mu, table


@torch.no_grad()
def evaluate(scorer, ld, ratio, slacks, *, tokens, batch, want_rel_err=True):
    B = ld.budget(ratio)
    KI = ld.K * ld.I
    acc = {s: [0.0, 0.0, 0.0, 0] for s in slacks}
    idx_all = ld.take(ld.test_sl, tokens)
    for s0 in range(0, len(idx_all), batch):
        idx = idx_all[s0:s0 + batch]
        if want_rel_err:
            x, sel, g, imp, inter, _, _ = ld.batch(idx, also_parts=True)
        else:
            x, sel, g, imp = ld.batch(idx)
            inter = None
        score = scorer.score(x, sel, g)
        ref = select_topB(imp, B)
        for s in slacks:
            pred = select_topB(score, min(int(round(B * s)), KI))
            a = acc[s]
            T = x.shape[0]
            a[0] += recall(pred, ref) * T
            a[1] += mass_recall(imp, pred, ref) * T
            if want_rel_err:
                a[2] += output_rel_err(inter, pred, sel, g, ld.w.Wd) * T
            a[3] += T
    return {str(s): {"pred_budget": min(int(round(B * s)), KI), "ref_budget": B,
                     "recall": a[0] / a[3], "mass_recall": a[1] / a[3],
                     "rel_err": (a[2] / a[3]) if want_rel_err else None,
                     "tokens": a[3]}
            for s, a in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="46,22")
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "docs/results/channel_router/data"))
    ap.add_argument("--tag", default="c4")
    ap.add_argument("--tokens", type=int, default=1 << 20)
    ap.add_argument("--ratios", default="0.125,0.25")
    ap.add_argument("--slacks", default="0.9,1.0,1.15,1.25")
    ap.add_argument("--router-ckpt", default="")
    ap.add_argument("--router-label", default="router")
    ap.add_argument("--methods", default=("router,static_freq,svd,wsvd,randproj,lsh,"
                                          "product_key,vq,dejavu,oracle_up,oracle"))
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--vq-centroids", type=int, default=256)
    ap.add_argument("--dejavu-hidden", type=int, default=512)
    ap.add_argument("--fit-steps", type=int, default=1500)
    ap.add_argument("--fit-batch", type=int, default=512)
    ap.add_argument("--fit-lr", type=float, default=1e-2)
    ap.add_argument("--delta", type=int, default=256)
    ap.add_argument("--w-fn", type=float, default=3.0)
    ap.add_argument("--eval-tokens", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs/results/channel_router/eval"))
    ap.add_argument("--out-name", default="protocol")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)

    def log(m):
        print(f"[eval] {m}", flush=True)

    slacks = [float(v) for v in args.slacks.split(",")]
    methods = args.methods.split(",")
    routers = load_router_artifact(args.router_ckpt, device=args.device) \
        if args.router_ckpt else {}
    out = {"args": vars(args), "layers": {}}
    for layer in [int(v) for v in args.layers.split(",")]:
        ld = LayerData(args.data_dir, layer, tag=args.tag, tokens=args.tokens,
                       device=args.device, want_down=True)
        log(f"=== layer {layer}")
        _, Sh, Sinv, _ = whiten_stats(ld.X[ld.train_sl][:262144], device=args.device)
        rows = []
        for ratio in [float(v) for v in args.ratios.split(",")]:
            st = ld.channel_freq(ld.train_sl, ratio, n=65536, batch=2048)
            for meth in methods:
                t0 = time.time()
                acct = None
                if meth == "router":
                    if layer not in routers:
                        log(f"  no router for layer {layer}; skipping")
                        continue
                    sc = routers[layer]
                    sc.name = args.router_label
                    cfg = torch.load(args.router_ckpt, map_location="cpu")["layers"][layer]["cfg"]
                    acct = router_accounting(ld.H, ld.E, ld.I, ld.K, cfg["r"], cfg["m"],
                                             head=cfg["head"], rho=ratio)
                    params = acct["params"]
                elif meth == "static_freq":
                    sc = StaticFreq(ld.w, st["freq_active"], prior="g")
                    params = 0
                elif meth in ("svd", "wsvd"):
                    sc = SvdScorer(ld.w, args.r, whitened=(meth == "wsvd"), Sh=Sh,
                                   Sinv=Sinv, bilinear=True, prior="both",
                                   device=args.device)
                    params = 0
                elif meth == "randproj":
                    sc = RandomProjScorer(ld.w, args.r, prior="both", device=args.device)
                    params = 0
                elif meth == "lsh":
                    sc = LshScorer(ld.w, bits=64, prior="both", device=args.device)
                    params = 0
                elif meth == "product_key":
                    sc = ProductKeyScorer(ld.w, r=args.r, prior="both", device=args.device)
                    fit_scorer(sc, ld, ratio, steps=args.fit_steps, batch=args.fit_batch,
                               lr=args.fit_lr, delta=args.delta, w_fn=args.w_fn,
                               log=log, name=meth)
                    params = sc.params
                elif meth == "dejavu":
                    sc = DejaVuMLP(ld.w, hidden=args.dejavu_hidden, prior="both",
                                   device=args.device)
                    fit_scorer(sc, ld, ratio, steps=args.fit_steps, batch=args.fit_batch,
                               lr=1e-3, delta=args.delta, w_fn=args.w_fn, log=log,
                               name=meth)
                    params = sc.params
                elif meth == "vq":
                    mu, table = fit_vq(ld, args.vq_centroids, ratio,
                                       fit_tokens=131072, log=log)
                    sc = VqScorer(ld.w, mu, table, prior="none")
                    params = sc.params
                elif meth == "oracle":
                    # the exact imp — the ceiling, and the rel_err anchor that ties a
                    # layer-local number to the measured full-model ΔPPL ladder.
                    class _Or:
                        name = "oracle"

                        def score(self, x, sel, g):
                            from src.channel_router.data import oracle_scores
                            return oracle_scores(x, sel, g, ld.w)
                    sc = _Or()
                    params = 0
                elif meth == "oracle_up":
                    # reference point already measured downstream in this repo:
                    # rank by |up| (the pre-gate signal), with both free priors.
                    class _Up:
                        name = "oracle_up"

                        def score(self, x, sel, g):
                            from src.channel_router.data import oracle_scores
                            return oracle_scores(x, sel, g, ld.w, target="up")
                    sc = _Up()
                    params = 0
                else:
                    raise ValueError(meth)
                res = evaluate(sc, ld, ratio, slacks, tokens=args.eval_tokens,
                               batch=args.batch)
                row = {"method": getattr(sc, "name", meth), "ratio": ratio,
                       "params": int(params), "accounting": acct, "slack": res,
                       "secs": round(time.time() - t0, 1)}
                rows.append(row)
                r1 = res[str(1.0)]
                log(f"  rho={ratio} {row['method']:<16} params={params / 1e6:7.3f}M "
                    f"recall={r1['recall']:.4f} mass={r1['mass_recall']:.4f} "
                    f"rel_err={r1['rel_err']:.4f}")
                del sc
                torch.cuda.empty_cache()
                out["layers"][str(layer)] = rows
                with open(os.path.join(args.out_dir, f"{args.out_name}.json"), "w") as f:
                    json.dump(out, f, indent=2)
        del ld
        torch.cuda.empty_cache()
    log(f"wrote {os.path.join(args.out_dir, args.out_name + '.json')}")


if __name__ == "__main__":
    main()
