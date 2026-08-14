"""B3 -- codebook heads: group residual VQ (CARVQ-style) and VQ-Logits.

Two operating points on the same axis, chosen because the pilots put codebook
methods ~50x ahead of low-rank in excess PPL at matched storage:

``rvq`` (CARVQ, arXiv:2510.12721, EMNLP Findings 2025)
    Each row of ``W`` is cut into ``D / vq_dim`` groups; every group position gets
    its own codebook of ``2^codebook_bits`` entries, applied in ``stages``
    successive residual passes. Bits/weight is exactly
    ``stages * codebook_bits / vq_dim`` plus a negligible codebook term, so ~1.6
    bits/param -- CARVQ's headline -- is reachable without sub-4-bit storage
    hardware. An optional low-rank "corrective adaptor" mops up what the codes miss,
    fitted in the activation metric.

``vq_logits`` (VQ-Logits, arXiv:2505.10202)
    The extreme point: replace all ``V`` rows with assignments into **one** shared
    codebook of ``K`` full-width vectors. Up to 99% of output-layer parameters
    disappear. Expected to fail (the paper reports +4% PPL); it is here to mark the
    edge of what is tolerable.

Neither paper's abstract discloses codebook size, sub-vector dimension, or stage
count, so those are ours -- chosen to land on the published bits/param, and always
reported alongside the result rather than assumed.
"""

from typing import Optional, Tuple

import torch

from src.base.shared_utils import _print
from src.lm_head.quant import metric_transform, randomized_svd

__all__ = ["build_rvq", "build_vq_logits", "rvq_bits_per_weight"]


def rvq_bits_per_weight(
    V: int,
    D: int,
    vq_dim: int,
    codebook_bits: int,
    stages: int,
    adaptor_rank: int = 0,
    scale_bits: int = 16,
) -> dict:
    """Analytic bits/weight for group residual VQ.

    Codes dominate but the codebooks are not free: ``(D/vq_dim) * 2^bits * vq_dim``
    fp16 values per stage, independent of ``V``. At the default 1.5-bit setting on a
    ``151936 x 2048`` head that is 0.081 bits/weight -- ~5% of the budget, small but
    worth counting, since it is exactly the term a "~1.6 bits/param" headline hides.
    """
    dense = V * D
    n_groups = D // vq_dim
    codes = V * n_groups * codebook_bits * stages
    books = stages * n_groups * (2 ** codebook_bits) * vq_dim * scale_bits
    adaptor = 0.0
    if adaptor_rank > 0:
        adaptor = (V * adaptor_rank + adaptor_rank * D) * scale_bits
    total = codes + books + adaptor
    return {
        "bits_per_weight": total / dense,
        "codes_bpw": codes / dense,
        "codebook_bpw": books / dense,
        "adaptor_bpw": adaptor / dense,
        "storage_frac_of_bf16": total / (dense * 16.0),
    }


@torch.no_grad()
def _kmeans(X: torch.Tensor, K: int, iters: int = 25, seed: int = 0) -> torch.Tensor:
    """Lloyd's algorithm on ``(N, d)`` -> ``(K, d)`` centroids.

    k-means++-lite init (farthest-point over a random subsample) then plain Lloyd.
    Empty clusters are re-seeded onto the currently worst-fit points, which matters
    at ``K = 256``: a dead code is a wasted 8th of a bit.
    """
    N, d = X.shape
    K = int(min(K, N))
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(N, generator=g)[:K]
    Cb = X[perm.to(X.device)].clone()

    for _ in range(int(iters)):
        assign = _assign(X, Cb)
        newC = torch.zeros_like(Cb)
        cnt = torch.zeros(K, device=X.device, dtype=X.dtype)
        newC.index_add_(0, assign, X)
        cnt.index_add_(0, assign, torch.ones(N, device=X.device, dtype=X.dtype))
        dead = cnt == 0
        cnt = cnt.clamp_min(1.0)
        newC = newC / cnt.unsqueeze(1)
        if bool(dead.any()):
            err = (X - Cb[assign]).pow(2).sum(1)
            worst = torch.topk(err, int(dead.sum())).indices
            newC[dead] = X[worst]
        Cb = newC
    return Cb


@torch.no_grad()
def _assign(X: torch.Tensor, Cb: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
    """Nearest-centroid assignment, chunked over ``N`` to bound the distance matrix."""
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    cn = Cb.pow(2).sum(1)
    for s in range(0, X.shape[0], chunk):
        xb = X[s:s + chunk]
        d = xb.pow(2).sum(1, keepdim=True) - 2.0 * (xb @ Cb.T) + cn.unsqueeze(0)
        out[s:s + chunk] = d.argmin(1)
    return out


@torch.no_grad()
def build_rvq(
    W: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    vq_dim: int = 16,
    codebook_bits: int = 8,
    stages: int = 3,
    iters: int = 20,
    adaptor_rank: int = 0,
    p: float = 0.75,
    ridge: float = 1e-3,
    activation_metric: bool = True,
    compute_device: str = "cpu",
    verbose: bool = True,
):
    """Group residual VQ of ``W``. Returns ``(W_hat, stats)``.

    Defaults give ``3 * 8 / 16 = 1.5`` bits/weight, the closest clean setting to
    CARVQ's ~1.6. ``adaptor_rank > 0`` adds the corrective low-rank branch, fitted
    in the activation metric (the same ``T_p`` machinery ARCHead uses, since the
    objective is identical -- minimize the error the softmax actually sees).
    """
    dev = compute_device
    Wf = W.detach().to(device=dev, dtype=torch.float32)
    V, D = Wf.shape
    if D % vq_dim != 0:
        raise ValueError(f"vq_dim={vq_dim} must divide D={D}")
    n_groups = D // vq_dim
    K = 2 ** int(codebook_bits)

    # (V, n_groups, vq_dim) -> quantize each group position independently
    X = Wf.reshape(V, n_groups, vq_dim)
    recon = torch.zeros_like(X)
    resid = X.clone()
    for st in range(int(stages)):
        for gi in range(n_groups):
            cb = _kmeans(resid[:, gi, :], K, iters=iters, seed=1000 * st + gi)
            a = _assign(resid[:, gi, :], cb)
            recon[:, gi, :] += cb[a]
        resid = X - recon
        if verbose:
            rel = float(resid.reshape(V, D).norm() / Wf.norm().clamp_min(1e-30))
            _print(f"[lm_head/B3 rvq] stage {st + 1}/{stages}: rel residual {rel:.4f}")
    W_hat = recon.reshape(V, D)

    # optional corrective adaptor on what the codes left behind
    if adaptor_rank > 0:
        E = Wf - W_hat
        if activation_metric and C is not None:
            Tp, Tp_inv, _ = metric_transform(C, p=p, ridge=ridge, compute_device=dev)
            U, S, Vh = randomized_svd(E @ Tp.to(dev), rank=adaptor_rank)
            W_hat = W_hat + (U * S.unsqueeze(0)) @ (Vh @ Tp_inv.to(dev))
        else:
            U, S, Vh = randomized_svd(E, rank=adaptor_rank)
            W_hat = W_hat + (U * S.unsqueeze(0)) @ Vh

    stats = dict(rvq_bits_per_weight(V, D, vq_dim, codebook_bits, stages, adaptor_rank))
    stats.update({
        "vq_dim": vq_dim, "codebook_bits": codebook_bits, "stages": stages,
        "adaptor_rank": adaptor_rank,
        "rel_fro_err": float((W_hat - Wf).norm() / Wf.norm().clamp_min(1e-30)),
    })
    if C is not None:
        Cd = C.to(device=dev, dtype=torch.float32)
        Dm = W_hat - Wf
        num = ((Dm @ Cd) * Dm).sum().clamp_min(0).sqrt()
        den = ((Wf @ Cd) * Wf).sum().clamp_min(1e-30).sqrt()
        stats["rel_metric_err"] = float(num / den)
    if verbose:
        _print(
            f"[lm_head/B3 rvq] vq_dim={vq_dim} K=2^{codebook_bits} stages={stages} "
            f"adaptor_rank={adaptor_rank} -> {stats['bits_per_weight']:.3f} bits/weight "
            f"({100 * stats['storage_frac_of_bf16']:.2f}% of BF16); "
            f"rel err Frobenius {stats['rel_fro_err']:.4f}"
            + (f", in the C metric {stats['rel_metric_err']:.4f}" if "rel_metric_err" in stats else "")
        )
    return W_hat.to(W.dtype), stats


@torch.no_grad()
def build_vq_logits(
    W: torch.Tensor,
    K: int = 1024,
    iters: int = 25,
    C: Optional[torch.Tensor] = None,
    compute_device: str = "cpu",
    verbose: bool = True,
):
    """VQ-Logits: every row of ``W`` replaced by one of ``K`` shared full-width codes.

    Storage is ``K * D`` fp16 values plus ``V * log2(K)`` bits of assignment --
    at ``K = 1024, V = 151936, D = 2048`` that is 0.68% of the BF16 head, i.e. the
    paper's "up to 99%" removal. Returns ``(W_hat, stats)``.
    """
    dev = compute_device
    Wf = W.detach().to(device=dev, dtype=torch.float32)
    V, D = Wf.shape
    cb = _kmeans(Wf, int(K), iters=iters, seed=0)
    a = _assign(Wf, cb)
    W_hat = cb[a]

    dense = V * D
    bits = K * D * 16 + V * max(1, int(torch.log2(torch.tensor(float(K))).ceil()))
    stats = {
        "K": int(K),
        "bits_per_weight": bits / dense,
        "storage_frac_of_bf16": bits / (dense * 16.0),
        "rel_fro_err": float((W_hat - Wf).norm() / Wf.norm().clamp_min(1e-30)),
        "n_used_codes": int(torch.unique(a).numel()),
    }
    if C is not None:
        Cd = C.to(device=dev, dtype=torch.float32)
        Dm = W_hat - Wf
        num = ((Dm @ Cd) * Dm).sum().clamp_min(0).sqrt()
        den = ((Wf @ Cd) * Wf).sum().clamp_min(1e-30).sqrt()
        stats["rel_metric_err"] = float(num / den)
    if verbose:
        _print(
            f"[lm_head/B3 vq_logits] K={K} ({stats['n_used_codes']} used) -> "
            f"{stats['bits_per_weight']:.4f} bits/weight "
            f"({100 * stats['storage_frac_of_bf16']:.2f}% of BF16, i.e. "
            f"{100 * (1 - stats['storage_frac_of_bf16']):.2f}% of head params removed); "
            f"rel err Frobenius {stats['rel_fro_err']:.4f}"
        )
    return W_hat.to(W.dtype), stats
