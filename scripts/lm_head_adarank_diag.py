"""Diagnostic: is the lm_head's required rank *per token* much lower than on average?

Every Part-1 method in ``docs/exps/lm_head/results_lm_head.md`` compresses the head
with ONE static decision (one subspace, one row subset, one read set). This script
measures the headroom of the alternative: a **per-token** decision.

The construction that makes the question exact. With the activation second moment
``C = E[h h^T]`` and the activation-whitened head ``W C^(1/2) = P S Q^T``,

    R = S Q^T C^(-1/2)   (D x D)      z = R h        (D coefficients)
    P = W C^(1/2) Q S^-1 (V x D)      logits = P z   == W h, EXACTLY

``P`` has **orthonormal columns**, so dropping a coordinate set ``Sbar`` costs

    || logit error ||^2 = || z_Sbar ||^2                                   (*)

exactly -- no cross terms. Two consequences, and they are the whole point:

1. Selecting the ``r`` coordinates with the largest ``|z_i|`` is *provably optimal*
   for the logit MSE. There is no proxy, no scoring heuristic to tune.
2. Static activation-aware low-rank is the special case "always select the same
   ``r`` coordinates" (the largest ``E[z_i^2] = s_i^2``). So static-vs-adaptive is a
   clean apples-to-apples comparison of the SAME family, and the gap is exactly the
   gap between the average energy ordering and the per-token energy ordering.

Reads per token are ``r*V`` (r columns of P) + ``D^2`` (all of R, needed to know
which coordinates are large). Storage is ``V*D + D^2``: this is a **read/active-param**
method, not a storage method -- the honest axis, and the one the plan calls B1-a.

KL(dense || approx) on the calibration states is the instrument: the results doc's
own table has PPL_rel ~= exp(KL) to within a few percent
(KL .0119 -> 1.011, .0415 -> 1.042, .7324 -> 1.911), so a KL here is a PPL
prediction that costs seconds instead of an eval.

    python scripts/lm_head_adarank_diag.py --model Qwen/Qwen3-0.6B \
        --calib calib/lm_head_qwen3_0_6b/sigma_lm_head_c4_128x16x512.pt
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F


def _print(*a):
    print(*a, flush=True)


def pick_device(min_free_gb=5.0):
    if not torch.cuda.is_available():
        return "cpu"
    best, best_free = None, -1.0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        free_gb = free / 2**30
        if free_gb > best_free:
            best, best_free = i, free_gb
    _print(f"[dev] freest GPU is cuda:{best} with {best_free:.1f} GiB free")
    return f"cuda:{best}" if best_free >= min_free_gb else "cpu"


@torch.no_grad()
def load_head(model_id):
    """Return the lm_head weight (V, D) on CPU, float32, without a full model load."""
    from transformers import AutoConfig
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    path = snapshot_download(model_id, allow_patterns=["*.safetensors*", "*.json"])
    idx = os.path.join(path, "model.safetensors.index.json")
    files = {}
    if os.path.exists(idx):
        with open(idx) as f:
            files = json.load(f)["weight_map"]
    cands = ["lm_head.weight", "model.embed_tokens.weight"]
    for name in cands:
        fn = files.get(name)
        if fn is None and not files:
            fn = "model.safetensors"
        if fn is None:
            continue
        fp = os.path.join(path, fn)
        if not os.path.exists(fp):
            continue
        with safe_open(fp, framework="pt", device="cpu") as f:
            if name in f.keys():
                W = f.get_tensor(name).float()
                tied = name != "lm_head.weight"
                _print(f"[head] {model_id}: {name} {tuple(W.shape)}"
                       + ("  (tied -- lm_head is the input embedding)" if tied else ""))
                return W, cfg
    raise RuntimeError(f"could not find an lm_head weight in {model_id}")


@torch.no_grad()
def gram(W, dev, chunk=16384):
    """W^T W accumulated in float64 from float32 chunks."""
    D = W.shape[1]
    G = torch.zeros(D, D, dtype=torch.float64)
    for s in range(0, W.shape[0], chunk):
        blk = W[s:s + chunk].to(dev, torch.float32)
        G += (blk.T @ blk).double().cpu()
    return G


@torch.no_grad()
def sqrt_metric(C, ridge=1e-3):
    """C^(1/2) and C^(-1/2) with the repo's relative damping (float64 -> float32)."""
    Cd = C.double()
    cbar = torch.diagonal(Cd).mean().clamp_min(1e-30)
    Cd = Cd + ridge * cbar * torch.eye(Cd.shape[0], dtype=Cd.dtype)
    ev, Q = torch.linalg.eigh(Cd)
    ev = ev.clamp_min(ev.max() * 1e-12)
    return (Q * ev.sqrt()) @ Q.T, (Q * ev.rsqrt()) @ Q.T


@torch.no_grad()
def build_basis(W, C, dev, ridge=1e-3):
    """Return (P, R, sigma) with P orthonormal columns, R = S Q^T C^-1/2, P R == W."""
    Ch, Cinvh = sqrt_metric(C, ridge)
    Wt = W  # (V, D) cpu float32
    # M = C^1/2 W^T W C^1/2 = Q S^2 Q^T  -- the D x D route, so no V x D SVD is needed
    G = gram(Wt, dev)                                   # float64 (D, D)
    M = Ch @ G @ Ch                                     # float64
    ev, Q = torch.linalg.eigh(M)
    order = torch.argsort(ev, descending=True)
    ev, Q = ev[order].clamp_min(0), Q[:, order]
    sigma = ev.sqrt()                                   # singular values of W C^1/2
    inv = torch.where(sigma > sigma.max() * 1e-10, sigma.reciprocal(), torch.zeros_like(sigma))
    # P = W C^1/2 Q S^-1, built in chunks
    ChQ = (Ch @ Q).float()                              # (D, D)
    P = torch.empty_like(Wt)
    for s in range(0, Wt.shape[0], 16384):
        blk = Wt[s:s + 16384].to(dev, torch.float32)
        P[s:s + 16384] = (blk @ ChQ.to(dev) * inv.float().to(dev)).cpu()
    R = ((Q * sigma.unsqueeze(0)).T @ Cinvh).float()    # (D, D)
    return P, R, sigma.float()


@torch.no_grad()
def kl_top1(P, dense_logits, z, keep_mask, dev, chunk=512):
    """KL(dense || approx) and top-1 agreement for a per-token coordinate mask.

    ``keep_mask`` is (n, D) bool; approx logits are ``sum_{i in S} P[:,i] z_i``.
    Done as ``P @ (z * mask)`` -- same arithmetic, one GEMM.
    """
    n = z.shape[0]
    kl = 0.0
    agree = 0
    mse = 0.0
    Pd = P.to(dev, torch.float32)
    for s in range(0, n, chunk):
        zk = (z[s:s + chunk] * keep_mask[s:s + chunk]).to(dev, torch.float32)
        la = zk @ Pd.T
        ld = dense_logits[s:s + chunk].to(dev, torch.float32)
        lpd = F.log_softmax(ld, -1)
        lpa = F.log_softmax(la, -1)
        kl += float((lpd.exp() * (lpd - lpa)).sum())
        agree += int((ld.argmax(-1) == la.argmax(-1)).sum())
        mse += float((ld - la).pow(2).mean(-1).sum())
    return kl / n, agree / n, mse / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--calib", default="calib/lm_head_qwen3_0_6b/sigma_lm_head_c4_128x16x512.pt")
    ap.add_argument("--out", default="results_eval/lm_head_adarank_diag_0_6b.json")
    ap.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 256, 384, 512])
    ap.add_argument("--n-states", type=int, default=2048)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    dev = a.device or pick_device()
    t0 = time.time()
    W, cfg = load_head(a.model)
    V, D = W.shape
    pay = torch.load(a.calib, map_location="cpu")
    C, H = pay["C"].float(), pay["H"].float()[: a.n_states]
    _print(f"[calib] C {tuple(C.shape)} over {int(pay['n']):,} states, "
           f"H {tuple(H.shape)}")

    P, R, sigma = build_basis(W, C, dev)
    # sanity: P R == W, and P has orthonormal columns
    i = torch.randint(0, V, (64,))
    err = (P[i] @ R - W[i]).norm() / W[i].norm()
    orth = (P[:, :64].T.to(dev) @ P[:, :64].to(dev) - torch.eye(64, device=dev)).abs().max()
    _print(f"[basis] reconstruction rel err {err:.3e}; |P^T P - I|_max on 64 cols {orth:.3e}")
    _print(f"[basis] built in {time.time() - t0:.0f}s; sigma max/min "
           f"{sigma[0]:.3e}/{sigma[-1]:.3e}")

    z = H @ R.T                                            # (n, D) coefficients
    n = z.shape[0]
    e = z.pow(2)                                           # per-token coordinate energy
    tot = e.sum(-1, keepdim=True).clamp_min(1e-30)

    # dense logits once
    dense = torch.empty(n, V, dtype=torch.float32)
    Pd = P.to(dev, torch.float32)
    for s in range(0, n, 512):
        dense[s:s + 512] = (z[s:s + 512].to(dev) @ Pd.T).cpu()

    rows = []
    # ---- how much energy does each selection rule capture? ------------------- #
    static_share = (sigma.pow(2).cumsum(0) / sigma.pow(2).sum())
    ad_sorted = e.sort(-1, descending=True).values
    ad_share = (ad_sorted.cumsum(-1) / tot).mean(0)
    _print("\n  r   reads%   static-E%  adaptive-E%   (energy of z captured)")
    for r in a.ranks:
        _print(f"{r:5d}  {100*(r*V + D*D)/(V*D):6.2f}   {100*static_share[r-1]:8.2f}   "
               f"{100*ad_share[r-1]:9.2f}")

    def read_frac(r, extra=D * D):
        return (r * V + extra) / (V * D)

    for r in a.ranks:
        # (1) STATIC = activation-aware low-rank, the doc's F2
        m = torch.zeros(n, D, dtype=torch.bool)
        m[:, :r] = True
        kl, ag, mse = kl_top1(P, dense, z, m, dev)
        rows.append(dict(method="static_lowrank", r=r, read_frac=read_frac(r, 0),
                         store_frac=(V + D) * r / (V * D), kl=kl, top1=ag, logit_mse=mse,
                         energy_kept=float(static_share[r - 1])))
        _print(f"[static r={r:4d}] KL={kl:.4f}  top1={100*ag:5.2f}%  "
               f"pred PPL x{torch.tensor(kl).exp():.3f}")

        # (2) ADAPTIVE = per-token top-r by |z_i|, provably optimal for (*)
        idx = e.topk(r, dim=-1).indices
        m = torch.zeros(n, D, dtype=torch.bool)
        m.scatter_(1, idx, True)
        kl, ag, mse = kl_top1(P, dense, z, m, dev)
        rows.append(dict(method="adaptive_topr", r=r, read_frac=read_frac(r),
                         store_frac=1.0 + D / V, kl=kl, top1=ag, logit_mse=mse,
                         energy_kept=float(ad_share[r - 1])))
        _print(f"[adapt  r={r:4d}] KL={kl:.4f}  top1={100*ag:5.2f}%  "
               f"pred PPL x{torch.tensor(kl).exp():.3f}   reads={100*read_frac(r):.2f}%")

    # (3) DYNAMIC BUDGET: per-token energy threshold, average r matched
    _print("")
    for keep in (0.99, 0.995, 0.999, 0.9999):
        cum = ad_sorted.cumsum(-1) / tot
        r_tok = (cum < keep).sum(-1) + 1
        r_tok = r_tok.clamp(max=D)
        order = e.argsort(-1, descending=True)
        ranks_pos = torch.empty_like(order)
        ranks_pos.scatter_(1, order, torch.arange(D).expand(n, D))
        m = ranks_pos < r_tok.unsqueeze(1)
        kl, ag, mse = kl_top1(P, dense, z, m, dev)
        rbar = float(r_tok.float().mean())
        rows.append(dict(method="adaptive_energy", keep=keep, r=rbar,
                         read_frac=read_frac(rbar), store_frac=1.0 + D / V,
                         kl=kl, top1=ag, logit_mse=mse, r_p99=float(r_tok.float().quantile(0.99))))
        _print(f"[adapt  E>={keep:<7}] mean r={rbar:7.1f} (p99 {float(r_tok.float().quantile(0.99)):.0f})  "
               f"reads={100*read_frac(rbar):5.2f}%  KL={kl:.4f}  top1={100*ag:5.2f}%  "
               f"pred PPL x{torch.tensor(kl).exp():.3f}")

    # (4) PER-CLUSTER (shared read set within a context cluster) -- is the gain
    #     per-token, or would a cheap router capture it?
    _print("")
    for G in (16, 64):
        gen = torch.Generator().manual_seed(0)
        cen = H[torch.randperm(n, generator=gen)[:G]].clone()
        for _ in range(15):
            assign = torch.cdist(H, cen).argmin(-1)
            for g in range(G):
                sel = assign == g
                if sel.any():
                    cen[g] = H[sel].mean(0)
        for r in (256,):
            m = torch.zeros(n, D, dtype=torch.bool)
            for g in range(G):
                sel = assign == g
                if not sel.any():
                    continue
                top = e[sel].mean(0).topk(r).indices        # one read set per cluster
                m[sel] = False
                m[sel.nonzero(as_tuple=True)[0].unsqueeze(1), top.unsqueeze(0)] = True
            kl, ag, mse = kl_top1(P, dense, z, m, dev)
            rows.append(dict(method=f"cluster{G}_topr", r=r, read_frac=read_frac(r),
                             store_frac=1.0 + D / V, kl=kl, top1=ag, logit_mse=mse))
            _print(f"[clust G={G:3d} r={r}] KL={kl:.4f}  top1={100*ag:5.2f}%  "
                   f"pred PPL x{torch.tensor(kl).exp():.3f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(model=a.model, V=V, D=D, n_states=n,
                       sigma_energy=static_share.tolist()[:1024:8], rows=rows), f, indent=2)
    _print(f"\nwrote {a.out}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
