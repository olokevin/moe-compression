"""Can the lm_head's *parameter count* be cut 75% at all? A structure diagnostic.

The read-side result (``lm_head_screen_refine.py``) leaves storage at 100%. This script
asks the other question directly, and cheaply, using the identity

    E_h || (W - W_hat) h ||^2 = || (W - W_hat) C^(1/2) ||_F^2

so everything reduces to a relative Frobenius error on the activation-whitened head
``Wt = W C^(1/2)`` -- no per-state loop, no eval.

**The bar.** On this head the doc's own diagnostics pin ``KL ~= 9.5 * relerr^2``
(static low-rank r=256/384/512 give relerr^2 = .1014/.0714/.0494 against measured
KL = 1.092/.676/.422). 4-bit RTN sits at KL .0415. So a 25%-storage representation has
to reach ``relerr <= ~7%`` to be competitive with a 4-bit head, and ``<= ~10%`` to be
merely "not catastrophic". Plain low-rank r=256 reaches **31.8%**.

Candidates, each budgeted to the same 25% of ``V*D`` parameters:

``lowrank``      the doc's F2 -- one global r-dim subspace (reference point)
``clustered``    union of subspaces: k-means the rows, per-cluster PCA. This is the
                 only family that can be full-rank globally while each row stores
                 few coefficients, and it is what has to work if any storage method is
                 going to. (Adaptive-softmax is its frequency-clustered special case.)
``lr_sparse``    low-rank + the largest-magnitude residual entries kept exactly
                 (LoSparse-style). Metadata for the indices is reported, not hidden.
``freq_exact``   exact rows for the top-T frequent tokens + a low-rank tail -- the
                 storage-side analogue of B1-s, which the doc never tried.
"""

import argparse
import json
import os

import torch

from scripts.lm_head_adarank_diag import load_head, pick_device, sqrt_metric, _print


@torch.no_grad()
def rel_err(Wt, Wh, dev, chunk=16384):
    """||Wt - Wh||_F / ||Wt||_F, chunked over the vocabulary."""
    num = den = 0.0
    for s in range(0, Wt.shape[0], chunk):
        a = Wt[s:s + chunk].to(dev, torch.float32)
        b = Wh[s:s + chunk].to(dev, torch.float32)
        num += float((a - b).pow(2).sum())
        den += float(a.pow(2).sum())
    return (num / max(den, 1e-30)) ** 0.5


@torch.no_grad()
def svd_lowrank_cpu(Wt, r, dev, seed=0):
    g = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        U, S, Vv = torch.svd_lowrank(Wt.to(dev, torch.float32), q=min(r + 10, Wt.shape[1]), niter=4)
    finally:
        torch.random.set_rng_state(g)
    return (U[:, :r] * S[:r]).cpu(), Vv[:, :r].T.cpu()


@torch.no_grad()
def kmeans(X, G, iters, dev, seed=0):
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    cen = X[torch.randperm(n, generator=g)[:G]].to(dev, torch.float32).clone()
    assign = torch.zeros(n, dtype=torch.long)
    for it in range(iters):
        for s in range(0, n, 8192):
            blk = X[s:s + 8192].to(dev, torch.float32)
            assign[s:s + 8192] = torch.cdist(blk, cen).argmin(-1).cpu()
        new = torch.zeros_like(cen)
        cnt = torch.zeros(G, device=dev)
        for s in range(0, n, 8192):
            blk = X[s:s + 8192].to(dev, torch.float32)
            aa = assign[s:s + 8192].to(dev)
            new.index_add_(0, aa, blk)
            cnt.index_add_(0, aa, torch.ones_like(aa, dtype=new.dtype))
        keep = cnt > 0
        cen[keep] = new[keep] / cnt[keep].unsqueeze(1)
    return assign, cen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--calib", default="calib/lm_head_qwen3_0_6b/sigma_lm_head_c4_128x16x512.pt")
    ap.add_argument("--unigram", default="calib/lm_head_qwen3_0_6b/unigram_c4_5000000.pt")
    ap.add_argument("--out", default="results_eval/lm_head_storage_struct_0_6b.json")
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    dev = a.device or pick_device()
    W, _ = load_head(a.model)
    V, D = W.shape
    C = torch.load(a.calib, map_location="cpu")["C"].float()
    Ch, _ = sqrt_metric(C)
    Ch = Ch.float()
    Wt = torch.empty_like(W)
    for s in range(0, V, 16384):
        Wt[s:s + 16384] = (W[s:s + 16384].to(dev) @ Ch.to(dev)).cpu()
    budget = a.budget * V * D
    rows = []

    def log(name, params, err, extra=""):
        kl = 9.5 * err * err
        rows.append(dict(method=name, params=params, store_frac=params / (V * D),
                         rel_metric_err=err, kl_pred=kl, note=extra))
        _print(f"{name:<40} {params/1e6:6.1f}M ({100*params/(V*D):5.2f}%)  "
               f"rel err={100*err:6.2f}%   KL~{kl:7.4f} (PPL x{torch.tensor(kl).exp():6.3f}) {extra}")

    _print(f"\n=== storage structure, {V}x{D}, budget {100*a.budget:.0f}% "
           f"= {budget/1e6:.1f}M params; bar: rel err <= ~7% ===")

    # --- reference: one global subspace ------------------------------------- #
    r = int(budget / (V + D))
    A, B = svd_lowrank_cpu(Wt, r, dev)
    log(f"lowrank r={r}", (V + D) * r, rel_err(Wt, A @ B, dev))

    # --- union of subspaces -------------------------------------------------- #
    for G in (16, 64, 256, 1024):
        # params: V*r coefficients + G*(r*D basis + D mean)
        rg = int((budget - G * D) / (V + G * D))
        if rg < 8:
            continue
        assign, _ = kmeans(Wt, G, 12, dev)
        Wh = torch.empty_like(Wt)
        for gi in range(G):
            idx = (assign == gi).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            blk = Wt[idx].to(dev, torch.float32)
            mu = blk.mean(0, keepdim=True)
            k = min(rg, idx.numel(), D)
            Uu, Ss, Vv = torch.svd_lowrank(blk - mu, q=min(k + 10, D, idx.numel()), niter=4)
            Wh[idx] = (mu + (Uu[:, :k] * Ss[:k]) @ Vv[:, :k].T).cpu()
        log(f"clustered G={G} r={rg}", V * rg + G * (rg * D + D), rel_err(Wt, Wh, dev),
            f"(+{V*11/8/1e6:.1f}MB cluster ids)")

    # --- low-rank + entry-sparse residual ----------------------------------- #
    for frac in (0.10, 0.15):
        # params: (V+D)*r + frac*V*D kept entries (indices are metadata, flagged)
        r2 = int((budget - frac * V * D) / (V + D))
        if r2 < 8:
            continue
        A2, B2 = svd_lowrank_cpu(Wt, r2, dev)
        Wh = A2 @ B2
        Rr = Wt - Wh
        k = int(frac * D)
        for s in range(0, V, 16384):
            blk = Rr[s:s + 16384].to(dev, torch.float32)
            thr = blk.abs().topk(k, dim=-1).values[:, -1:]
            Rr[s:s + 16384] = torch.where(blk.abs() >= thr, blk, torch.zeros_like(blk)).cpu()
        log(f"lowrank r={r2} + {100*frac:.0f}% entries", int((V + D) * r2 + frac * V * D),
            rel_err(Wt, Wh + Rr, dev), f"(+{V*k*17/8/1e6:.0f}MB indices)")

    # --- exact frequent rows + low-rank tail -------------------------------- #
    if os.path.exists(a.unigram):
        order = torch.load(a.unigram, map_location="cpu")["counts"].argsort(descending=True)
        for T in (8192, 16384, 32768):
            r3 = int((budget - T * D) / (V - T + D))
            if r3 < 8:
                continue
            tail = order[T:]
            At, Bt = svd_lowrank_cpu(Wt[tail], r3, dev)
            Wh = Wt.clone()
            Wh[tail] = At @ Bt
            log(f"exact top-{T} rows + tail lowrank r={r3}",
                int(T * D + (V - T + D) * r3), rel_err(Wt, Wh, dev))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(model=a.model, V=V, D=D, budget=a.budget, rows=rows), f, indent=2)
    _print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
