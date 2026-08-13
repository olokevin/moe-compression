"""Cheap low-rank channel *scorers* for Level-2 cross-expert selection.

`oracle_mag` reaches near-dense accuracy at a 7/8 channel cut, but its ranking
signal is the **true** SwiGLU intermediate ``inter = SiLU(gate·x) · (up·x)`` — so
`gate_proj` and `up_proj` must run at full width just to decide what to keep, and
only `down_proj` is actually narrowed. A nominal −87.5% channel cut is therefore
only a −29.2% whole-FFN cut (see the reduction-accounting tables in
``docs/exps/dynamic_active_param/q3_30b_dynamic_active.md``).

This module builds a **cheap proxy** for that signal so the keep-decision can be
made *before* any full-width matmul, which lets all three matrices be gathered to
the budget:

    ĥ_j ≈ SiLU(gate_j·x) · (up_j·x)   computed from rank-r factors of W_gate/W_up
    keep = per-token global top-B over the K active experts by g_e·|ĥ_j|
    then compute the TRUE up/gate/down on the kept channels only.

Kept fraction of the 3-matrix expert FFN becomes ``rho + n_scorer * c`` (c = one
scorer's cost, below) instead of `oracle_mag`'s ``(1+1+rho)/3`` or `oracle_up`'s
``(1+2·rho)/3``.

**The factorization (BTT, singular values in the input-side core).** Each weight
``W`` ``(I, H)`` is cut into an ``m x n`` grid of ``(a, b)`` blocks
(``a = I/m``, ``b = H/n``) and every block is truncated to rank ``r``:

    W[i,j] ≈ L[i,j] R[i,j],   L: (a, r),   R: (r, b) = S_r V_r^T

so the singular values live in the **input-side** core ``R`` (as requested: the
intermediate ``h = R x`` is then already singular-value-scaled, which is what
makes ``‖h‖`` a meaningful magnitude for the coarse group-level variant).
Online:

    h[i,j] = R[i,j] x_j            (cost m·n·r·b = r·m·H)   <- "intermediate h"
    ŵ_i    = sum_j L[i,j] h[i,j]   (cost m·n·r·a = r·n·I)   <- per-channel proxy

``m = n = 1`` degenerates to a plain **global rank-r SVD** scorer, which is the
cheapest-to-describe variant; ``m, n > 1`` is the BTT regime, where the *same*
FLOP budget buys a proxy of much higher effective rank (a block grid with
per-block rank ``r`` can express rank up to ``m·min(a, n·r)``), so BTT should
dominate SVD at equal cost. Both are reachable from one code path.
"""

from dataclasses import dataclass

import torch

from src.base.shared_utils import _print

__all__ = [
    "LowRankScorer",
    "factorize_blocks",
    "build_layer_scorer",
    "scorer_proxy",
    "scorer_cost_fraction",
    "resolve_scorer_grid",
]


@dataclass
class LowRankScorer:
    """Per-layer, expert-stacked block-low-rank cores for one weight matrix.

    ``L_core`` ``(E, m, n, a, r)`` output-side, ``R_core`` ``(E, m, n, r, b)``
    input-side (singular values merged in). Channel ``c`` of the proxy output
    corresponds to block row ``i = c // a`` and offset ``p = c % a``, matching
    ``W.reshape(m, a, n, b)`` — i.e. the natural row order of ``W``.
    """

    L_core: torch.Tensor
    R_core: torch.Tensor
    m: int
    n: int
    a: int
    b: int
    rank: int


def resolve_scorer_grid(I: int, H: int, m: int, n: int) -> tuple:
    """Validate the ``(m, n)`` block grid against ``W`` of shape ``(I, H)``."""
    m, n = int(m), int(n)
    if m <= 0 or n <= 0:
        raise ValueError(f"scorer grid must be positive, got m={m}, n={n}")
    if I % m or H % n:
        raise ValueError(
            f"scorer grid (m={m}, n={n}) must divide W shape (I={I}, H={H}); "
            f"got I%m={I % m}, H%n={H % n}"
        )
    return m, n, I // m, H // n


def factorize_blocks(
    W: torch.Tensor,
    m: int,
    n: int,
    rank: int,
    niter: int = 4,
) -> LowRankScorer:
    """Block-wise truncated SVD of expert-stacked weights.

    Args:
        W: ``(E, I, H)`` stacked weight matrices (one per expert).
        m, n: block grid over (output, input) — ``m=n=1`` gives a global SVD.
        rank: per-block rank ``r`` (clamped to ``min(a, b)``).
        niter: randomized-SVD power iterations. ``torch.svd_lowrank`` is
            matmul-only, so it is safe to run on a GPU that still holds a model
            shard (unlike a large explicit inverse / full ``linalg.svd``).

    Returns:
        ``LowRankScorer`` with cores in ``W``'s dtype and on ``W``'s device.
    """
    if W.ndim != 3:
        raise ValueError(f"expected (E, I, H) stacked weights, got {tuple(W.shape)}")
    E, I, H = W.shape
    m, n, a, b = resolve_scorer_grid(I, H, m, n)
    r = max(1, min(int(rank), min(a, b)))

    # (E, I, H) -> (E, m, n, a, b) -> flat batch of blocks
    blocks = W.reshape(E, m, a, n, b).permute(0, 1, 3, 2, 4).contiguous()
    # Factorize in at least fp32: model weights are bf16, and a bf16 SVD loses
    # far more than the rank truncation itself. float64 inputs stay float64.
    svd_dtype = W.dtype if W.dtype == torch.float64 else torch.float32
    flat = blocks.reshape(E * m * n, a, b).to(svd_dtype)

    # randomized SVD; oversample then truncate for accuracy at rank r.
    q = min(min(a, b), r + 8)
    U, S, V = torch.svd_lowrank(flat, q=q, niter=niter)
    U, S, V = U[:, :, :r], S[:, :r], V[:, :, :r]          # (N,a,r),(N,r),(N,b,r)

    L = U.reshape(E, m, n, a, r)
    # singular values merged into the INPUT-side core: R = S_r V_r^T
    R = (S.unsqueeze(-1) * V.transpose(1, 2)).reshape(E, m, n, r, b)

    return LowRankScorer(
        L_core=L.to(dtype=W.dtype).contiguous(),
        R_core=R.to(dtype=W.dtype).contiguous(),
        m=m, n=n, a=a, b=b, rank=r,
    )


def build_layer_scorer(
    experts,
    which: str,
    m: int,
    n: int,
    rank: int,
    niter: int = 4,
    compute_device=None,
) -> LowRankScorer:
    """Factorize one projection across all experts of a single MoE layer.

    Args:
        experts: the layer's expert module list.
        which: ``"up_proj"`` | ``"gate_proj"``.
        compute_device: where the (matmul-only) randomized SVD runs. ``None``
            keeps it on the weights' own device.
    """
    W = torch.stack(
        [getattr(e, which).weight.detach() for e in experts], dim=0
    )                                                    # (E, I, H)
    home = W.device
    if compute_device is not None and torch.device(compute_device) != home:
        W = W.to(compute_device)
    sc = factorize_blocks(W, m=m, n=n, rank=rank, niter=niter)
    del W
    if sc.L_core.device != home:
        sc = LowRankScorer(
            L_core=sc.L_core.to(home), R_core=sc.R_core.to(home),
            m=sc.m, n=sc.n, a=sc.a, b=sc.b, rank=sc.rank,
        )
    return sc


def scorer_proxy(
    x: torch.Tensor,
    L_core: torch.Tensor,
    R_core: torch.Tensor,
    return_h: bool = False,
):
    """Apply one expert's block-low-rank scorer to a token batch.

    Args:
        x: ``(T, H)`` hidden states for the tokens routed to this expert.
        L_core: ``(m, n, a, r)`` output-side core for this expert.
        R_core: ``(m, n, r, b)`` input-side core for this expert.
        return_h: also return the intermediate ``h`` ``(T, m, n, r)``.

    Returns:
        ``(T, I)`` proxy of the projection output (channel order matches ``W``),
        and optionally the intermediate ``h``.
    """
    m, n, a, r = L_core.shape
    b = R_core.shape[-1]
    xb = x.reshape(x.shape[0], n, b)
    h = torch.einsum("tnb,mnrb->tmnr", xb, R_core)         # cost r·m·H
    out = torch.einsum("tmnr,mnar->tma", h, L_core)        # cost r·n·I
    out = out.reshape(x.shape[0], m * a)
    return (out, h) if return_h else out


def scorer_cost_fraction(I: int, H: int, m: int, n: int, rank: int) -> float:
    """One scorer's cost as a fraction of one full ``(I, H)`` matmul.

    ``r·(m·H + n·I) / (I·H)`` — the two einsums in :func:`scorer_proxy`.
    """
    m, n, _, _ = resolve_scorer_grid(I, H, m, n)
    return float(rank) * (m * H + n * I) / float(I * H)


def report_scorer_accounting(
    I: int, H: int, m: int, n: int, rank: int, n_scorers: int, prune_ratio: float
) -> dict:
    """Whole-FFN active-parameter accounting for the proxy-scorer scheme.

    With the keep-decision made *before* any full-width matmul, all three expert
    matrices are gathered to ``rho = 1 - prune_ratio``, and the scorers are the
    only overhead:

        kept = rho + n_scorers · c / 3      (c = per-scorer cost fraction)

    (``/3`` because ``c`` is measured against **one** matrix while the FFN has
    three.) Compare `oracle_mag` ``(1+1+rho)/3`` and `oracle_up` ``(1+2·rho)/3``.
    """
    c = scorer_cost_fraction(I, H, m, n, rank)
    rho = 1.0 - float(prune_ratio)
    kept = rho + n_scorers * c / 3.0
    return {
        "scorer_cost_per_matrix": c,
        "scorer_overhead_ffn": n_scorers * c / 3.0,
        "rho": rho,
        "kept_fraction": kept,
        "whole_ffn_cut": 1.0 - kept,
        "oracle_mag_kept": (1.0 + 1.0 + rho) / 3.0,
        "oracle_up_kept": (1.0 + 2.0 * rho) / 3.0,
    }


def print_scorer_accounting(
    I: int, H: int, m: int, n: int, rank: int, n_scorers: int, prune_ratio: float
):
    acc = report_scorer_accounting(I, H, m, n, rank, n_scorers, prune_ratio)
    _print(
        f"[LowRankScorer] grid m={m} n={n} rank={rank} "
        f"({n_scorers} scorer{'s' if n_scorers > 1 else ''}): "
        f"cost/matrix={acc['scorer_cost_per_matrix']:.4f}, "
        f"FFN overhead={acc['scorer_overhead_ffn']:.4f}, "
        f"rho={acc['rho']:.4f} -> kept={acc['kept_fraction']:.4f} "
        f"(whole-FFN cut {100 * acc['whole_ffn_cut']:.1f}%); "
        f"oracle_mag would keep {acc['oracle_mag_kept']:.4f}, "
        f"oracle_up {acc['oracle_up_kept']:.4f}"
    )
    return acc
