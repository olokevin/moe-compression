"""Level-1 offline artifact: batched ridge-pivoted Cholesky over the per-expert
coupling matrix ``Theta_k = G_k ⊙ B_k``.

Realizes Phase B of ``docs/results/dynamic_active_param/plan/plan_level1.md``.
Phase A (the activation Gram ``G_k = E[phi_k phi_k^T]``) is already available:
the cached ``expert_covariances.pth`` stores exactly that uncentered second
moment of each expert's ``down_proj`` input.

Per expert we form
    B_k     = W_down_k^T H W_down_k            (weight Gram, H = I here)
    Theta_k = G_k ⊙ B_k                         (Hadamard; PSD by Schur product)
and run pivoted Cholesky with a shared ridge ``lambda_r`` to completion, yielding
a nested channel ordering ``pi_k`` (redundancy-aware) and per-channel marginal
gains ``sigma_k`` (residual diagonal at each pivot, monotone non-increasing).

The online kernel then keeps, per token, the top-``t_k`` channels in pivot order
where ``t_k`` emerges from a single global ``g_k^2 * sigma`` threshold — see
``allocate._pivchol_allocate``.
"""

import os

import torch

from src.base.shared_utils import _print
from src.base.shared_utils.safe_isinstance import (
    _get_moe_block,
    _get_experts,
    _get_moe_intermediate_size,
    _get_num_hidden_layers,
)
from src.dynamic_active_param.precompute import AllocArtifact

__all__ = ["pivoted_cholesky_batched", "build_pivchol_artifact"]


def pivoted_cholesky_batched(theta: torch.Tensor, lambda_r: float = 1.0):
    """Batched ridge-pivoted Cholesky.

    Args:
        theta: ``(E, m, m)`` symmetric PSD coupling matrices (one per expert).
        lambda_r: shared ridge added to the diagonal (kept identical across all
            experts/layers so marginal gains are on one absolute scale).

    Returns:
        perm:  ``(E, m)`` long — pivot order (physical channel index picked at
               each step; a permutation of ``0..m-1`` per expert).
        gains: ``(E, m)`` float — marginal gain (residual diagonal) at each pivot
               step, in step order (monotone non-increasing per expert).
    """
    theta = theta.to(torch.float32).clone()
    E, m, _ = theta.shape
    device = theta.device

    # ridge on the diagonal
    idx_m = torch.arange(m, device=device)
    theta[:, idx_m, idx_m] += lambda_r

    diag = theta[:, idx_m, idx_m].clone()          # (E, m) running residual diag
    perm = torch.zeros((E, m), dtype=torch.long, device=device)
    gains = torch.zeros((E, m), dtype=torch.float32, device=device)
    L = torch.zeros((E, m, m), dtype=torch.float32, device=device)  # Cholesky factor cols
    chosen = torch.zeros((E, m), dtype=torch.bool, device=device)
    ar = torch.arange(E, device=device)

    for t in range(m):
        # pick the largest residual diagonal among not-yet-chosen channels
        masked = diag.masked_fill(chosen, float("-inf"))
        piv = masked.argmax(dim=1)                 # (E,)
        perm[:, t] = piv
        g = diag[ar, piv].clamp_min(0.0)           # (E,) marginal gain
        gains[:, t] = g
        chosen[ar, piv] = True

        s = g.clamp_min(1e-12).sqrt()              # (E,)
        # residual column of theta at the pivot, minus already-built factor cols
        col = theta[ar, :, piv].clone()            # (E, m)
        if t > 0:
            # L[:, :, :t] is (E, m, t); L_piv is (E, t)
            L_piv = L[ar, piv, :t]                 # (E, t)
            col = col - torch.bmm(L[:, :, :t], L_piv.unsqueeze(-1)).squeeze(-1)
        Lt = col / s.unsqueeze(1)                   # (E, m)
        Lt[ar, piv] = s                             # exact on the pivot itself
        L[:, :, t] = Lt
        # downdate residual diagonals
        diag = (diag - Lt * Lt).clamp_min(0.0)
        diag[ar, piv] = 0.0                         # pivot fully consumed

    return perm, gains


def build_pivchol_artifact(
    model,
    scores_dir: str,
    lambda_r: float = 1.0,
    device: str = "cuda",
    save: bool = True,
    verbose: bool = True,
    compute_device: str = None,
) -> AllocArtifact:
    """Build (or load) the Level-1 pivoted-Cholesky artifact.

    Needs the loaded model (for ``down_proj`` weights) and the cached
    ``expert_covariances.pth`` in ``scores_dir`` (Phase-A activation Gram).

    Returns an ``AllocArtifact`` whose ``channel_rank`` is the pivot *rank* of
    each physical channel (position in ``pi_k``) and whose ``gains`` are the
    marginal gains in pivot-position order (so ``gains[..., r]`` is the gain of
    the channel at pivot rank ``r``, matching how ``channel_rank`` is compared).

    ``compute_device`` selects where the batched Cholesky runs. Leave it None to
    run the factorization on **CPU** — running heavy ``bmm``/linalg on a GPU that
    still holds a ``device_map='auto'`` shard of the 30B model can poison the
    CUDA context (see the cublas-crash memory). The offline warm-up path frees
    the model first and may pass a clean GPU; the in-eval path must use CPU.
    """
    cache_path = os.path.join(scores_dir, "pivchol_artifact.pth")
    if os.path.exists(cache_path):
        if verbose:
            _print(f"[PivChol] Loading cached artifact from {cache_path}")
        payload = torch.load(cache_path, map_location=device)
        return AllocArtifact(
            channel_rank=payload["channel_rank"].to(device),
            contrib=payload["contrib"].to(device),
            prefix_sums=payload["prefix_sums"].to(device),
            gains=payload["gains"].to(device),
            L=int(payload["L"]),
            E=int(payload["E"]),
            I=int(payload["I"]),
            channel_metric="pivchol",
        )

    cov_path = os.path.join(scores_dir, "expert_covariances.pth")
    if not os.path.exists(cov_path):
        raise FileNotFoundError(
            f"{cov_path} missing — pivchol needs the activation Gram (covariances). "
            "Run an eval with channel_metric=leverage once to collect them."
        )
    if verbose:
        _print(f"[PivChol] Loading covariances from {cov_path}")
    expert_covariances = torch.load(cov_path, map_location="cpu")

    # Default: factor on CPU to avoid crashing CUBLAS on a shard-resident GPU.
    comp_dev = torch.device(compute_device) if compute_device is not None else torch.device("cpu")

    m = _get_moe_intermediate_size(model)
    num_layers = _get_num_hidden_layers(model)

    pivrank_layers = []
    gains_layers = []
    mask_idx = 0
    for layer_idx in range(num_layers):
        moe_block = _get_moe_block(model, layer_idx)
        experts = _get_experts(moe_block)
        if experts is None:
            continue
        E = len(experts)

        # stacked down_proj weights (E, d, m) -> weight Gram B = Wd^T Wd (E, m, m)
        Wd = torch.stack(
            [e.down_proj.weight.detach().float().cpu() for e in experts], dim=0
        ).to(comp_dev)                                  # (E, d, m)
        B = torch.bmm(Wd.transpose(1, 2), Wd)           # (E, m, m)
        del Wd

        # stacked activation Gram G (identity fallback for missing experts)
        layer_covs = expert_covariances.get(layer_idx, {})
        G = torch.eye(m, device=comp_dev).unsqueeze(0).repeat(E, 1, 1)
        for eid, cov in layer_covs.items():
            G[eid] = cov.to(comp_dev).float()

        theta = G * B                                   # Hadamard (E, m, m)
        del G, B
        perm, gains = pivoted_cholesky_batched(theta, lambda_r=lambda_r)  # (E,m),(E,m)
        del theta

        # pivrank: physical channel -> its position in the pivot order
        pivrank = torch.argsort(perm, dim=1).to(torch.long)  # (E, m)
        # gains reindexed so gains_pos[c] is the gain of the channel at rank c;
        # perm[:, t] is the channel at rank t and gains[:, t] its gain, so the
        # rank-ordered gain vector IS just `gains` (already in step/rank order).
        pivrank_layers.append(pivrank.cpu())
        gains_layers.append(gains.cpu())
        mask_idx += 1
        if verbose and mask_idx % 8 == 0:
            _print(f"[PivChol] processed {mask_idx} MoE layers")

    channel_rank = torch.stack(pivrank_layers, dim=0)   # (L, E, m)
    gains_pos = torch.stack(gains_layers, dim=0)        # (L, E, m) rank-ordered
    L, E, I = channel_rank.shape

    # placeholders to satisfy the shared AllocArtifact schema (unused here)
    contrib = torch.zeros((L, E), dtype=torch.float32)
    prefix_sums = torch.zeros((1,), dtype=torch.float32)

    if verbose:
        _print(f"[PivChol] channel_rank {tuple(channel_rank.shape)}, gains {tuple(gains_pos.shape)}")

    if save:
        torch.save(
            {
                "channel_rank": channel_rank,
                "gains": gains_pos,
                "contrib": contrib,
                "prefix_sums": prefix_sums,
                "L": L,
                "E": E,
                "I": I,
            },
            cache_path,
        )
        if verbose:
            _print(f"[PivChol] Cached artifact to {cache_path}")

    return AllocArtifact(
        channel_rank=channel_rank.to(device),
        contrib=contrib.to(device),
        prefix_sums=prefix_sums.to(device),
        gains=gains_pos.to(device),
        L=L,
        E=E,
        I=I,
        channel_metric="pivchol",
    )
