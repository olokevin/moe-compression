"""P5 — balanced tile construction (the DOT-MoE connection, reused offline).

DOT-MoE partitions neurons into experts with a balanced Sinkhorn/OT assignment and
trains the assignment jointly with the router. Here the same balanced-assignment
machinery runs **offline** with a *co-activation* cost instead of a weight-similarity
cost, and produces tiles — groups of channels that tend to be selected together — so a
two-level router can score ``K·n_tiles`` tiles instead of ``K·I`` channels and a kernel
can read whole tiles contiguously.

Tiles are built **inside an expert** (see ``model.py`` note 4): a tile spans one
expert's channels only, so it is contiguous in that expert's weight matrices. Four
constructions are compared, as the plan asks:

- ``coact``      balanced Sinkhorn k-means on the spectral embedding of the
                 co-activation matrix ``C_ij = P(i,j both in oracle mask)``
- ``weight``     balanced k-means on the expert's ``W_g`` rows (MoEfication's heuristic)
- ``random``     random balanced split (the null)
- ``native``     one tile per expert (``n_tiles_per_expert = 1``), i.e. today's grouping
"""

from __future__ import annotations

import torch

__all__ = ["coactivation_matrix", "spectral_embedding", "balanced_kmeans",
           "build_tiles", "tile_concentration"]


@torch.no_grad()
def coactivation_matrix(keep_per_expert: torch.Tensor) -> torch.Tensor:
    """``C_ij = P(i, j both kept)`` from a ``(n, I)`` bool matrix of per-token masks."""
    m = keep_per_expert.float()
    return (m.t() @ m) / max(m.shape[0], 1)


@torch.no_grad()
def spectral_embedding(C: torch.Tensor, dim: int = 16) -> torch.Tensor:
    """Normalized-Laplacian spectral embedding of a co-activation matrix ``(I, I)``."""
    C = C.clone()
    C.fill_diagonal_(0)
    d = C.sum(1).clamp_min(1e-9).rsqrt()
    A = d.unsqueeze(1) * C * d.unsqueeze(0)
    evals, evecs = torch.linalg.eigh(A.double())
    Z = evecs[:, -dim:].flip(-1).float()
    return Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-9)


@torch.no_grad()
def balanced_kmeans(Z: torch.Tensor, n_clusters: int, *, iters: int = 25,
                    eps: float = 0.05, sink_iter: int = 60,
                    seed: int = 0) -> torch.Tensor:
    """Sinkhorn-balanced k-means; returns ``(n,)`` labels with exactly ``n/k`` per cluster.

    The assignment step is an entropic OT with uniform cluster marginals (DOT-MoE's
    balanced assignment), and the final hard labels come from capacity-respecting
    greedy rounding of the transport plan, so the balance is exact rather than
    approximate.
    """
    n = Z.shape[0]
    if n % n_clusters:
        raise ValueError(f"n={n} not divisible by n_clusters={n_clusters}")
    cap = n // n_clusters
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=gen)[:n_clusters]
    mu = Z[perm].clone()
    logb = torch.full((n_clusters,), -torch.log(torch.tensor(float(n_clusters))).item(),
                      device=Z.device)
    for _ in range(iters):
        cost = torch.cdist(Z, mu) ** 2                       # (n, k)
        logK = -cost / eps
        v = torch.zeros(n_clusters, device=Z.device)
        for _ in range(sink_iter):
            u = -torch.logsumexp(logK + v.unsqueeze(0), dim=1)
            v = logb + torch.log(torch.tensor(float(n), device=Z.device)) \
                - torch.logsumexp(logK + u.unsqueeze(1), dim=0)
        P = torch.exp(logK + u.unsqueeze(1) + v.unsqueeze(0))  # (n, k)
        mu_new = (P.t() @ Z) / P.sum(0).clamp_min(1e-12).unsqueeze(1)
        if torch.allclose(mu_new, mu, atol=1e-6):
            mu = mu_new
            break
        mu = mu_new
    # capacity-respecting greedy rounding of the final plan
    conf, order = P.max(1)
    labels = torch.full((n,), -1, dtype=torch.long, device=Z.device)
    room = torch.full((n_clusters,), cap, device=Z.device)
    for i in conf.argsort(descending=True).tolist():
        prefs = P[i].argsort(descending=True)
        for c in prefs.tolist():
            if room[c] > 0:
                labels[i] = c
                room[c] -= 1
                break
    return labels


@torch.no_grad()
def build_tiles(*, method: str, n_tiles: int, coact: torch.Tensor | None = None,
                Wg: torch.Tensor | None = None, embed_dim: int = 16,
                seed: int = 0) -> torch.Tensor:
    """``(I,)`` tile labels for one expert's channels under the named construction."""
    if method == "native":
        I = coact.shape[0] if coact is not None else Wg.shape[0]
        return torch.zeros(I, dtype=torch.long)
    if method == "random":
        I = coact.shape[0] if coact is not None else Wg.shape[0]
        gen = torch.Generator(device="cpu").manual_seed(seed)
        lab = torch.arange(I) % n_tiles                     # exactly balanced
        return lab[torch.randperm(I, generator=gen)]
    if method == "coact":
        Z = spectral_embedding(coact, dim=embed_dim)
    elif method == "weight":
        Z = Wg / Wg.norm(dim=1, keepdim=True).clamp_min(1e-9)
    else:
        raise ValueError(method)
    return balanced_kmeans(Z, n_tiles, seed=seed)


@torch.no_grad()
def tile_concentration(keep: torch.Tensor, tile_of: torch.Tensor, n_tiles: int):
    """How concentrated is a token's oracle mask across tiles?

    Args:
        keep: ``(T, K, I)`` bool oracle mask.
        tile_of: ``(I,)`` shared within-expert labels, or ``(T, K, I)`` already gathered
            per-expert labels (when each expert has its own clustering).
        n_tiles: tiles per expert.

    Returns dict with the per-token count of *touched* tiles and the cumulative mask
    coverage of the top-n tiles (n = 1..K·n_tiles), i.e. P5's decision curve.
    """
    T, K, I = keep.shape
    lab = tile_of.view(1, 1, I).expand(T, K, I) if tile_of.dim() == 1 else tile_of
    off = (torch.arange(K, device=keep.device).view(1, K, 1) * n_tiles + lab).reshape(T, -1)
    cnt = torch.zeros((T, K * n_tiles), device=keep.device)
    cnt.scatter_add_(1, off, keep.reshape(T, -1).float())
    touched = (cnt > 0).sum(1).float()
    srt = cnt.sort(dim=1, descending=True).values
    cover = srt.cumsum(1) / srt.sum(1, keepdim=True).clamp_min(1e-9)
    return {"touched_mean": float(touched.mean()),
            "touched_p95": float(touched.kthvalue(max(1, int(0.95 * T)), dim=0).values)
            if T > 1 else float(touched.mean()),
            "coverage_curve": cover.mean(0).cpu()}
