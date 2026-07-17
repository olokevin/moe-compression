"""Pure, vectorized per-token per-expert channel-budget allocation.

Given a token's K routing weights (and, for the contribution criterion, the
per-expert calibration contribution), split a fixed total channel budget ``B``
across the token's K experts so that more channels go to the experts that
matter more, while conserving ``sum_e k_{t,e} == B`` for every token.

The core is the largest-remainder water-filling from the plan:

    raw_{t,e}  = w_{t,e} * B
    k_{t,e}    = clip(floor(raw_{t,e}), k_min, I)
    deficit    = B - sum_e k_{t,e}

then distribute the (signed) ``deficit`` unit-by-unit to the largest-remainder
experts below cap ``I`` (or, if negative, remove from the smallest-remainder
experts above floor ``k_min``). Because remainders ``raw - floor(raw)`` are
static, "repeatedly +1 to the largest-remainder expert below cap" is equivalent
to filling experts to cap in remainder-priority order; that closed form is what
we vectorize here (O(K log K) per token, no per-unit Python loop).

This module has no torch.nn / model dependency so it is trivially unit-testable
with hand-checkable small tensors.
"""

import torch

__all__ = ["allocate_budgets"]

_VALID_CRITERIA = ("router_prob", "contribution", "uniform")


def allocate_budgets(
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    contrib: torch.Tensor,
    B: int,
    k_min: int,
    I: int,
    criterion: str = "router_prob",
) -> torch.Tensor:
    """Allocate per-token per-expert channel budgets.

    Args:
        routing_weights: ``(T, K)`` float, the norm_topk_prob-normalized softmax
            routing weights for each token's K selected experts (sums to 1 over K).
        selected_experts: ``(T, K)`` long, expert ids chosen by each token.
        contrib: ``(E,)`` float, per-expert calibration contribution for this
            layer (clamped >= 0). Only used by the ``contribution`` criterion;
            may be ``None`` for the others.
        B: total kept channels per token, across its K experts.
        k_min: per-expert floor (minimum kept channels per expert).
        I: per-expert cap (moe_intermediate_size).
        criterion: ``router_prob`` | ``contribution`` | ``uniform``.

    Returns:
        ``(T, K)`` long tensor ``k`` with ``k_min <= k <= I`` and
        ``k.sum(dim=1) == B`` for every token (given a feasible
        ``K*k_min <= B <= K*I``).
    """
    if criterion not in _VALID_CRITERIA:
        raise ValueError(f"Unknown criterion {criterion!r}; expected one of {_VALID_CRITERIA}")

    T, K = selected_experts.shape
    device = selected_experts.device
    B = int(B)
    k_min = int(k_min)
    I = int(I)

    if K * k_min > B or B > K * I:
        raise ValueError(
            f"Infeasible budget: need K*k_min ({K*k_min}) <= B ({B}) <= K*I ({K*I}). "
            "Lower k_min or raise prune_ratio."
        )

    # --- allocation weights w_{t,e} over the K selected experts -------------
    if criterion == "router_prob":
        w = routing_weights.to(torch.float32)
    elif criterion == "uniform":
        w = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device)
    else:  # contribution
        if contrib is None:
            raise ValueError("criterion='contribution' requires a contrib tensor")
        w = contrib.to(device=device, dtype=torch.float32)[selected_experts].clamp_min(0.0)

    # Normalize over K; fall back to uniform for degenerate rows (sum <= 0),
    # e.g. all-zero contribution.
    row_sum = w.sum(dim=1, keepdim=True)
    uniform = torch.full_like(w, 1.0 / K)
    w = torch.where(row_sum > 0, w / row_sum.clamp_min(torch.finfo(w.dtype).tiny), uniform)

    # --- floor + clamp, then water-fill the signed deficit ------------------
    raw = w * B
    base = torch.floor(raw)
    frac = raw - base
    k = base.clamp(min=float(k_min), max=float(I)).to(torch.long)

    deficit = B - k.sum(dim=1)  # (T,) signed
    pos_need = deficit.clamp(min=0)          # units to add
    neg_need = (-deficit).clamp(min=0)       # units to remove

    # Add path: fill largest-remainder experts (below cap I) first.
    if int(pos_need.max()) > 0:
        cap_add = (I - k).clamp(min=0)  # (T,K)
        order = torch.argsort(frac, dim=1, descending=True, stable=True)
        cap_sorted = torch.gather(cap_add, 1, order)
        cum = torch.cumsum(cap_sorted, dim=1)
        prev = cum - cap_sorted
        take = (pos_need.unsqueeze(1) - prev).clamp(min=0)
        take = torch.minimum(take, cap_sorted)
        add = torch.zeros_like(k)
        add.scatter_(1, order, take)
        k = k + add

    # Remove path: drain smallest-remainder experts (above floor k_min) first.
    if int(neg_need.max()) > 0:
        cap_rem = (k - k_min).clamp(min=0)  # (T,K)
        order = torch.argsort(frac, dim=1, descending=False, stable=True)
        cap_sorted = torch.gather(cap_rem, 1, order)
        cum = torch.cumsum(cap_sorted, dim=1)
        prev = cum - cap_sorted
        take = (neg_need.unsqueeze(1) - prev).clamp(min=0)
        take = torch.minimum(take, cap_sorted)
        rem = torch.zeros_like(k)
        rem.scatter_(1, order, take)
        k = k - rem

    return k
