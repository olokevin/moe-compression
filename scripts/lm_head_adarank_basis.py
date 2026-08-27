"""Which coordinate system makes the lm_head's per-token required rank smallest?

``lm_head_adarank_diag.py`` established the headroom: selecting the ``r`` largest
coefficients *per token* beats the best static ``r``-dim subspace by ~3.4x in KL at
matched reads. That gain is basis-dependent, so this script compares candidate bases
under one rule -- factor ``W = A @ M`` (exactly), read only the ``r`` columns of ``A``
whose contribution ``|coef_i| * ||a_i||`` is largest, and measure the real KL.

Bases:

``raw``    ``A = W``, ``coef = h``. Per-token *input-channel* selection -- no rotation
           at all. This is the strongest obvious baseline and the one the repo already
           uses for expert FFNs (``dynamic_active_param/sparse_probe.py``), so the
           rotated variants have to beat it or they are not worth their ``D^2``.
``ceig``   ``A = W U``, ``coef = U^T h`` with ``U`` the eigenbasis of ``C``: decorrelates
           the coefficients but leaves ``W``'s own anisotropy in ``A``.
``wsvd``   the activation-whitened SVD basis: ``A`` has **orthonormal columns**, so the
           squared logit error is exactly the dropped coefficient energy and top-``r``
           by ``|coef|`` is provably optimal. Static selection in this basis *is*
           activation-aware low-rank, so it is also the honest static reference.

Reads per token: ``r*V`` for the columns of ``A``, plus ``D^2`` for the mixing matrix
``M`` when the basis is rotated (``raw`` needs none). Both are charged below.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from scripts.lm_head_adarank_diag import (
    build_basis,
    load_head,
    pick_device,
    sqrt_metric,
    _print,
)


@torch.no_grad()
def make_basis(kind, W, C, dev):
    """Return (A, M, needs_MD2) with ``W == A @ M`` and coefficients ``coef = h @ M.T``."""
    D = W.shape[1]
    if kind == "raw":
        return W, torch.eye(D), False
    if kind == "ceig":
        Cd = C.double()
        cbar = torch.diagonal(Cd).mean().clamp_min(1e-30)
        ev, U = torch.linalg.eigh(Cd + 1e-3 * cbar * torch.eye(D, dtype=Cd.dtype))
        U = U[:, torch.argsort(ev, descending=True)].float()
        A = torch.empty_like(W)
        for s in range(0, W.shape[0], 16384):
            A[s:s + 16384] = (W[s:s + 16384].to(dev) @ U.to(dev)).cpu()
        return A, U.T, True                      # W = A U^T, coef = U^T h
    if kind == "wsvd":
        P, R, _ = build_basis(W, C, dev)
        return P, R, True                        # W = P R, coef = R h
    raise ValueError(kind)


@torch.no_grad()
def eval_selection(A, coef, score, r, dense, dev, chunk=256):
    """Top-``r``-by-``score`` per token; return (KL, top1 agreement, logit MSE)."""
    n, D = coef.shape
    idx = score.topk(r, dim=-1).indices
    keep = torch.zeros(n, D, dtype=torch.bool)
    keep.scatter_(1, idx, True)
    Ad = A.to(dev, torch.float32)
    kl = agree = mse = 0.0
    for s in range(0, n, chunk):
        ck = (coef[s:s + chunk] * keep[s:s + chunk]).to(dev, torch.float32)
        la = ck @ Ad.T
        ld = dense[s:s + chunk].to(dev, torch.float32)
        lpd, lpa = F.log_softmax(ld, -1), F.log_softmax(la, -1)
        kl += float((lpd.exp() * (lpd - lpa)).sum())
        agree += int((ld.argmax(-1) == la.argmax(-1)).sum())
        mse += float((ld - la).pow(2).mean(-1).sum())
    return kl / n, agree / n, mse / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--calib", default="calib/lm_head_qwen3_0_6b/sigma_lm_head_c4_128x16x512.pt")
    ap.add_argument("--out", default="results_eval/lm_head_adarank_basis_0_6b.json")
    ap.add_argument("--bases", nargs="+", default=["raw", "ceig", "wsvd"])
    ap.add_argument("--ranks", type=int, nargs="+", default=[128, 256, 384])
    ap.add_argument("--n-states", type=int, default=2048)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    dev = a.device or pick_device()
    W, _ = load_head(a.model)
    V, D = W.shape
    pay = torch.load(a.calib, map_location="cpu")
    C, H = pay["C"].float(), pay["H"].float()[: a.n_states]
    n = H.shape[0]

    # dense logits, once, from W directly
    dense = torch.empty(n, V, dtype=torch.float32)
    Wd = W.to(dev, torch.float32)
    for s in range(0, n, 256):
        dense[s:s + 256] = (H[s:s + 256].to(dev) @ Wd.T).cpu()
    del Wd
    torch.cuda.empty_cache()

    rows = []
    for kind in a.bases:
        t0 = time.time()
        A, M, rotated = make_basis(kind, W, C, dev)
        coef = H @ M.T.float()
        # verify the factorization is exact before trusting any number from it
        i = torch.randint(0, n, (16,))
        rel = ((coef[i].to(dev) @ A.to(dev).T) - dense[i].to(dev)).norm() / dense[i].to(dev).norm()
        colnorm = torch.zeros(D)
        for s in range(0, V, 16384):
            colnorm += A[s:s + 16384].to(dev).pow(2).sum(0).cpu()
        colnorm = colnorm.sqrt()
        _print(f"\n[{kind}] factorization rel err {rel:.2e}; "
               f"col-norm spread p99/p50 = {colnorm.quantile(0.99) / colnorm.median():.2f}"
               f"  ({time.time() - t0:.0f}s)")
        score_ad = (coef.abs() * colnorm.unsqueeze(0))          # per-token contribution
        static_score = (coef.pow(2).mean(0).sqrt() * colnorm)    # expected contribution
        for r in a.ranks:
            extra = D * D if rotated else 0
            rf = (r * V + extra) / (V * D)
            kl_s, ag_s, _ = eval_selection(
                A, coef, static_score.unsqueeze(0).expand(n, D), r, dense, dev)
            kl_a, ag_a, _ = eval_selection(A, coef, score_ad, r, dense, dev)
            rows += [
                dict(basis=kind, sel="static", r=r, read_frac=rf, kl=kl_s, top1=ag_s),
                dict(basis=kind, sel="adaptive", r=r, read_frac=rf, kl=kl_a, top1=ag_a),
            ]
            _print(f"  r={r:4d} reads={100*rf:5.2f}%  "
                   f"static KL={kl_s:7.4f} (x{torch.tensor(kl_s).exp():6.3f}) top1={100*ag_s:5.1f}%   |   "
                   f"adaptive KL={kl_a:7.4f} (x{torch.tensor(kl_a).exp():6.3f}) top1={100*ag_a:5.1f}%")
        del A
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(model=a.model, V=V, D=D, n_states=n, rows=rows), f, indent=2)
    _print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
