"""Differentiable exact-budget top-k (Stage C).

Selecting ``B`` of ``D`` channels is an optimal-transport problem between the ``D``
channels (each with unit mass) and the two-point marginal ``{selected: B,
dropped: D−B}`` with cost ``−score`` for "selected" and ``0`` for "dropped". The
entropic-regularized solution is obtained by Sinkhorn iterations in the log domain and
satisfies the budget **by construction** (column marginal ``= B``), which is why the
plan uses it instead of a sigmoid-with-penalty: no Lagrange tuning, and the soft mask
sums to exactly the budget at every temperature.

``sinkhorn_topk_ste`` returns the *hard* top-B mask in the forward pass and routes
gradients through the soft plan (straight-through), so train and inference select the
same channels — the same STE pattern DOT-MoE uses for its neuron→expert assignment,
applied here at the token→channel level.

As ``eps → 0`` the plan converges to the hard top-B indicator; ``eps → ∞`` gives the
uniform ``B/D``. Both limits and the budget exactness are covered by
``tests/test_sinkhorn_topk.py``.
"""

from __future__ import annotations

import torch

__all__ = ["sinkhorn_topk", "sinkhorn_topk_ste", "hard_topk"]


def hard_topk(score: torch.Tensor, B: int) -> torch.Tensor:
    """``(T, D)`` -> ``(T, D)`` float mask with exactly ``B`` ones per row."""
    idx = score.topk(B, dim=-1).indices
    return torch.zeros_like(score).scatter_(-1, idx, 1.0)


def sinkhorn_topk(score: torch.Tensor, B: int, eps: float = 0.1,
                  n_iter: int = 400, tol: float = 1e-9) -> torch.Tensor:
    """Soft selection plan ``(T, D)`` with ``plan.sum(-1) == B`` exactly.

    The iteration ends on the *column* (budget) scaling, so the budget marginal is
    satisfied to float precision at any iteration count; the per-channel marginal
    (``plan ≤ 1``) is only met at convergence, which needs ``n_iter`` to grow as ``eps``
    shrinks. ``tol`` must stay well below ``eps``-scaled potentials — a loose tolerance
    exits early with a plan that puts several units of mass on one channel.

    Args:
        score: ``(T, D)`` float scores (higher = more likely selected).
        B: budget per row.
        eps: entropic regularization; anneal 1.0 -> 0.03 during Stage C.
        n_iter / tol: iteration budget and potential-change tolerance.
    """
    T, D = score.shape
    # fp32 is enough for training, but preserve fp64 when the caller asked for it
    # (the finite-difference gradient test needs it).
    s = score if score.dtype == torch.float64 else score.float()
    s = s - s.max(dim=-1, keepdim=True).values                # stabilize
    # log-kernel of the two "columns": selected (score/eps) and dropped (0).
    logK = torch.stack([s / eps, torch.zeros_like(s)], dim=-1)   # (T, D, 2)
    # target column marginals in log space
    logb = torch.log(torch.tensor([B, D - B], dtype=s.dtype, device=s.device)
                     .clamp_min(1e-12))
    v = torch.zeros((T, 2), dtype=s.dtype, device=s.device)
    for _ in range(n_iter):
        # rows: each channel has unit mass -> u = -logsumexp_col(logK + v)
        u = -torch.logsumexp(logK + v.unsqueeze(1), dim=-1)          # (T, D)
        v_new = logb.view(1, 2) - torch.logsumexp(logK + u.unsqueeze(-1), dim=1)
        done = bool((v_new - v).abs().max() < tol)
        v = v_new
        if done:
            break
    # (u, v) is a consistent pair: v was computed from this u, so the budget marginal
    # is exact. Do NOT refresh u afterwards — that trades the budget guarantee for the
    # per-channel one, and the budget is the property Stage C relies on.
    plan = torch.exp(logK[..., 0] + u + v[:, :1])                    # (T, D)
    return plan.clamp_min(0.0)


def marginal_error(plan: torch.Tensor, B: int):
    """Diagnostics: ``(budget_err, channel_err)`` — the two marginal violations."""
    return (float((plan.sum(-1) - B).abs().max()),
            float((plan - 1).clamp_min(0).max()))


def sinkhorn_topk_ste(score: torch.Tensor, B: int, eps: float = 0.1,
                      n_iter: int = 50) -> torch.Tensor:
    """Hard top-B forward, Sinkhorn-soft backward (straight-through)."""
    soft = sinkhorn_topk(score, B, eps=eps, n_iter=n_iter)
    hard = hard_topk(score.detach(), B)
    return hard + (soft - soft.detach())
