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

_VALID_CRITERIA = ("router_prob", "contribution", "uniform", "coverage_alloc")


def allocate_budgets(
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    contrib: torch.Tensor,
    B: int,
    k_min: int,
    I: int,
    criterion: str = "router_prob",
    prefix_sums: torch.Tensor = None,
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
        criterion: ``router_prob`` | ``contribution`` | ``uniform`` |
            ``coverage_alloc``.
        prefix_sums: ``(E, I)`` float, per-expert cumulative sum of the
            descending-sorted channel scores (``prefix[e, n-1] = S_e(n)``).
            Required for ``coverage_alloc``; ignored otherwise.

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

    if criterion == "coverage_alloc":
        if prefix_sums is None:
            raise ValueError("criterion='coverage_alloc' requires a prefix_sums tensor")
        return _coverage_allocate(
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            prefix_sums=prefix_sums,
            B=B,
            k_min=k_min,
            I=I,
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


# Token chunk size for the coverage path — bounds the (chunk, K, I) prefix
# tensor materialized per bisection. 30B: I=768,K=8 -> ~0.25 GB fp32 at 4096.
_COVERAGE_CHUNK = 4096
_BISECT_ITERS = 30


def _coverage_allocate(
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    prefix_sums: torch.Tensor,
    B: int,
    k_min: int,
    I: int,
) -> torch.Tensor:
    """Coverage-maximized per-token allocation (paper §4.2, Algorithm 1).

    For each token, the group ``G`` is its K selected experts. Each expert ``e``
    has descending-sorted-score prefix sums ``P_e`` (``P_e[n-1] = S_e(n)``,
    ``S_tot_e = P_e[I-1]``). We initialize a coverage target from the router
    probability ``phi_e = p_{t,e}`` and a single scalar ``alpha``:

        rho_e(alpha) = min(alpha * phi_e, 1),
        N_e(alpha)   = min{ n : S_e(n) >= rho_e(alpha) * S_tot_e },

    and binary-search ``alpha`` for the largest value with
    ``sum_e N_e(alpha) <= B``. The (non-negative) residual to the exact budget
    ``B`` is then distributed by picking the globally highest *marginal* scores
    among each expert's not-yet-included tail channels — coverage-optimal, and
    lands ``sum_e k == B`` exactly.

    All ops are vectorized; tokens are processed in chunks to bound memory.
    """
    T, K = selected_experts.shape
    device = selected_experts.device
    prefix_sums = prefix_sums.to(device=device, dtype=torch.float32)  # (E, I)

    out = torch.empty((T, K), dtype=torch.long, device=device)

    for start in range(0, T, _COVERAGE_CHUNK):
        stop = min(start + _COVERAGE_CHUNK, T)
        sel = selected_experts[start:stop]                       # (t, K)
        phi = routing_weights[start:stop].to(torch.float32)      # (t, K)
        t = sel.shape[0]

        P = prefix_sums[sel]                                     # (t, K, I)
        s_tot = P[..., -1]                                      # (t, K)

        # alpha_max so min(alpha*phi, 1) can reach 1 for every expert in a
        # token (phi>0 for topk routing weights; guard tiny values anyway).
        phi_safe = phi.clamp_min(torch.finfo(phi.dtype).tiny)
        alpha_max = (1.0 / phi_safe.min(dim=1, keepdim=True).values)  # (t, 1)
        alpha_lo = torch.zeros((t, 1), dtype=torch.float32, device=device)
        alpha_hi = alpha_max.clone()

        # Best (largest-alpha) feasible counts found so far.
        n_best = torch.full((t, K), k_min, dtype=torch.long, device=device)

        for _ in range(_BISECT_ITERS):
            alpha = 0.5 * (alpha_lo + alpha_hi)                  # (t, 1)
            rho = torch.clamp(alpha * phi, max=1.0)              # (t, K)
            target = rho * s_tot                                 # (t, K)
            # smallest n (1-indexed) with P[n-1] >= target; compare against the
            # prefix curve directly — no divide by s_tot (handles s_tot==0).
            n = torch.searchsorted(P, target.unsqueeze(-1), right=False)
            n = n.squeeze(-1) + 1                                # (t, K), 1..I+1
            n = n.clamp(min=k_min, max=I)
            total = n.sum(dim=1, keepdim=True)                   # (t, 1)
            feasible = total <= B                                # (t, 1)
            # tighten bracket: feasible -> raise alpha_lo, else lower alpha_hi.
            alpha_lo = torch.where(feasible, alpha, alpha_lo)
            alpha_hi = torch.where(feasible, alpha_hi, alpha)
            n_best = torch.where(feasible, n, n_best)

        # --- exact-budget top-up (coverage-aware) ---------------------------
        # residual r >= 0 channels to add; give them to the highest marginal
        # score among each expert's tail (positions >= current n_e).
        residual = B - n_best.sum(dim=1)                         # (t,) >= 0
        if int(residual.max()) > 0:
            # marginal[e, c] = P[e,c] - P[e,c-1] = c-th descending-sorted score.
            marg = P.clone()
            marg[..., 1:] = P[..., 1:] - P[..., :-1]             # (t, K, I)
            # only tail channels (rank index >= n_e) are addable.
            pos = torch.arange(I, device=device).view(1, 1, I)   # (1,1,I)
            addable = pos >= n_best.unsqueeze(-1)                 # (t,K,I)
            marg = torch.where(addable, marg, torch.full_like(marg, float("-inf")))
            flat = marg.reshape(t, K * I)                        # (t, K*I)
            order = torch.argsort(flat, dim=1, descending=True, stable=True)
            ranks = torch.argsort(order, dim=1, stable=True)     # (t, K*I) position
            take = ranks < residual.unsqueeze(1)                 # (t, K*I) bool
            add = take.reshape(t, K, I).sum(dim=2).to(torch.long)  # (t, K)
            n_best = n_best + add

        out[start:stop] = n_best

    return out
