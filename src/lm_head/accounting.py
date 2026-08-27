"""Used-parameter / byte accounting for the lm_head baselines.

The whole point of plan section 1: on Qwen3-30B-A3B the head is **1.02% of total
but 9.28% of active** parameters, so every method here lives inside a 0 -> 9.28%
band of active-param reduction. Report the wrong denominator and the numbers read
as either trivial or magical. This module always reports both, plus the
post-expert-pruning denominator where the head's share rises to ~15.4%.

Two efficiency axes are tracked separately because they genuinely differ:

``storage``    bytes that must persist in memory for the head.
``read``       bytes touched per decoded token.

They coincide for B1-s / B2 / B3 (dense read of a compressed matrix) and diverge
for B1-a, which stores everything and reads one tier -- exactly the case where
quoting a single number would be misleading.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.base.shared_utils import _print

__all__ = ["ActiveParamContext", "count_active_params", "head_cost", "print_lm_head_accounting"]


@dataclass
class ActiveParamContext:
    """Parameter counts a head saving has to be measured against."""

    total_params: int
    active_params: int
    head_params: int
    is_moe: bool = False
    # active params with the repo's -73% expert compression applied, for the
    # composed denominator in plan section 5.
    active_params_pruned: Optional[int] = None


def count_active_params(model, expert_keep_frac: float = 0.27) -> ActiveParamContext:
    """Count total / active / head params, branching on MoE vs dense.

    "Active" = params touched for one token: all non-expert params plus only the
    top-K experts' FFN weights. ``expert_keep_frac`` is the fraction of expert-FFN
    params surviving the repo's expert compression (0.27 == the measured -73%
    ``input_sparse`` point), used only for the composed denominator.
    """
    from src.base.shared_utils.safe_isinstance import (
        _get_experts,
        _get_moe_block,
        _get_num_hidden_layers,
        _get_topk,
    )
    from src.lm_head.calib import get_lm_head

    total = sum(p.numel() for p in model.parameters())
    head = get_lm_head(model)
    head_params = head.weight.numel()

    inactive = 0
    expert_active = 0
    is_moe = False
    try:
        K = _get_topk(model)
        for li in range(_get_num_hidden_layers(model)):
            blk = _get_moe_block(model, li)
            experts = _get_experts(blk)
            if experts is None:
                continue
            is_moe = True
            per_expert = sum(p.numel() for p in experts[0].parameters())
            E = len(experts)
            inactive += (E - K) * per_expert
            expert_active += K * per_expert
    except Exception:
        is_moe = False
        inactive = 0

    active = total - inactive
    pruned = None
    if is_moe and expert_active:
        pruned = int(active - expert_active * (1.0 - float(expert_keep_frac)))
    return ActiveParamContext(
        total_params=int(total), active_params=int(active), head_params=int(head_params),
        is_moe=is_moe, active_params_pruned=pruned,
    )


def head_cost(
    V: int,
    D: int,
    storage_bits_per_weight: float,
    read_rows: Optional[int] = None,
    read_bits_per_weight: Optional[float] = None,
    stored_params: Optional[int] = None,
    read_params: Optional[int] = None,
) -> dict:
    """Cost of one head treatment on **two independent axes**.

    *Parameter count* -- how many numbers the representation holds
    (``stored_params``) and how many are touched per token
    (``read_params_per_token``), each as a fraction of ``V*D``. Quantization does
    **not** move these: an INT4 head has exactly ``V*D`` parameters. Only structural
    methods do -- low-rank factors ``(V+D)*r``, dropped rows ``T*D``, unread rows.

    *Precision / bytes* -- ``storage_bits_per_weight`` and the derived byte figures,
    which is where quantization's saving lives.

    ``read_rows=None`` means the head is read densely. Setting it to ``T`` is the
    B1-a case: storage stays at ``V`` rows but only ``T`` rows are touched per token.
    ``stored_params`` / ``read_params`` override the counts for methods whose
    representation is not a row subset (low-rank, codebooks).
    """
    dense_params = V * D
    dense_bytes = dense_params * 2.0
    store_bytes = dense_params * storage_bits_per_weight / 8.0
    rb = storage_bits_per_weight if read_bits_per_weight is None else read_bits_per_weight
    rows = V if read_rows is None else int(min(read_rows, V))
    read_bytes = rows * D * rb / 8.0

    # --- axis 1: PARAMETER COUNT (how many numbers exist / are touched) ------ #
    # Kept strictly separate from bytes, because the two axes answer different
    # questions and mixing them flatters quantization: an INT4 head stores exactly
    # as many parameters as a BF16 one, just narrower. Only structural methods
    # (low-rank factors, dropped rows, unread rows) change these counts.
    n_stored = dense_params if stored_params is None else int(stored_params)
    n_read = (rows * D) if read_params is None else int(read_params)

    return {
        "V": V, "D": D,
        "dense_params": dense_params,
        # parameter-count axis
        "stored_params": n_stored,
        "read_params_per_token": n_read,
        "stored_param_frac": n_stored / dense_params,
        "read_param_frac": n_read / dense_params,
        # byte / precision axis
        "dense_bytes": dense_bytes,
        "storage_bits_per_weight": float(storage_bits_per_weight),
        "storage_bytes": store_bytes,
        "storage_frac_of_bf16": store_bytes / dense_bytes,
        "read_rows": rows,
        "read_bytes_per_token": read_bytes,
        "read_frac_of_bf16": read_bytes / dense_bytes,
        # BF16-equivalent params: bytes re-expressed as a parameter count so a
        # precision saving can be compared against an active-parameter budget.
        # NOT a parameter count -- see stored_params / read_params_per_token.
        "used_head_params_bf16eq": read_bytes / 2.0,
        "stored_head_params_bf16eq": store_bytes / 2.0,
    }


def print_lm_head_accounting(cost: dict, ctx: ActiveParamContext, label: str = "") -> dict:
    """Print (and return) the head saving against every relevant denominator."""
    dense_eq = cost["dense_params"]
    used_eq = cost["used_head_params_bf16eq"]
    stored_eq = cost["stored_head_params_bf16eq"]
    saved_used = dense_eq - used_eq
    saved_stored = dense_eq - stored_eq

    out = dict(cost)
    out["head_share_of_total"] = ctx.head_params / max(ctx.total_params, 1)
    out["head_share_of_active"] = ctx.head_params / max(ctx.active_params, 1)
    out["delta_active_used"] = -saved_used / max(ctx.active_params, 1)
    out["delta_active_stored"] = -saved_stored / max(ctx.active_params, 1)
    out["delta_total_stored"] = -saved_stored / max(ctx.total_params, 1)
    # true parameter-count deltas (0 for any pure quantization method)
    out["delta_active_params_stored"] = (
        -(dense_eq - cost["stored_params"]) / max(ctx.active_params, 1)
    )
    out["delta_active_params_read"] = (
        -(dense_eq - cost["read_params_per_token"]) / max(ctx.active_params, 1)
    )
    if ctx.active_params_pruned:
        out["head_share_of_active_pruned"] = ctx.head_params / ctx.active_params_pruned
        out["delta_active_pruned_used"] = -saved_used / ctx.active_params_pruned

    _print(
        f"[lm_head/accounting]{' ' + label if label else ''} head {cost['V']}x{cost['D']} "
        f"= {dense_eq / 1e6:.1f}M params dense "
        f"({100 * out['head_share_of_total']:.2f}% of total, "
        f"{100 * out['head_share_of_active']:.2f}% of active)"
    )
    _print(
        f"[lm_head/accounting]   PARAMS  stored {cost['stored_params'] / 1e6:.1f}M "
        f"({100 * cost['stored_param_frac']:.2f}% of {dense_eq / 1e6:.1f}M), "
        f"read/token {cost['read_params_per_token'] / 1e6:.1f}M "
        f"({100 * cost['read_param_frac']:.2f}%)"
        + ("   <- unchanged: this is a precision method, not a structural one"
           if cost["stored_param_frac"] >= 1.0 and cost["read_param_frac"] >= 1.0 else "")
    )
    _print(
        f"[lm_head/accounting]   BYTES   {cost['storage_bits_per_weight']:.3f} bits/param "
        f"-> {cost['storage_bytes'] / 2**20:.1f} MiB "
        f"({100 * cost['storage_frac_of_bf16']:.2f}% of BF16), "
        f"{stored_eq / 1e6:.1f}M BF16-equiv params"
    )
    _print(
        f"[lm_head/accounting]   READ    {cost['read_rows']:,}/{cost['V']:,} rows "
        f"({100 * cost['read_rows'] / cost['V']:.2f}%) -> "
        f"{cost['read_bytes_per_token'] / 2**20:.1f} MiB/token "
        f"({100 * cost['read_frac_of_bf16']:.2f}% of BF16), "
        f"USED HEAD PARAMS={used_eq / 1e6:.1f}M"
    )
    line = (
        f"[lm_head/accounting]   Δ active params = {100 * out['delta_active_used']:+.2f}% "
        f"(by reads) / {100 * out['delta_active_stored']:+.2f}% (by storage)"
    )
    if ctx.active_params_pruned:
        line += (
            f"; after −73% expert pruning head share rises to "
            f"{100 * out['head_share_of_active_pruned']:.2f}% and "
            f"Δ active = {100 * out['delta_active_pruned_used']:+.2f}%"
        )
    _print(line)
    return out
