"""S1 -- the screen-and-refine head: two nested *dynamic* sparsities.

Part 1 of ``docs/exps/lm_head/results_lm_head.md`` closes sparse reads on the strength
of B1-a: read the top-``T`` frequent rows, mask the rest. Perplexity came out infinite
and HellaSwag at chance (25.67 on the 30B). Two design choices are entangled in that
result, and separating them is what this method is:

*static* read set
    the same rows at every position, so a target the tier omits is unrecoverable, and a
    length-``L`` continuation needs *every* token in-tier -- the coverage^L decay of
    doc section 1d.
*ungraded* tail
    omitted rows get ``-inf`` (perplexity inf by construction) or one shared constant,
    so the head cannot even rank the tail approximately.

Here both are dynamic and graded:

    stage 1  SCREEN   project ``h`` onto its ``r0`` largest coordinates in a fixed
                      rotation ``U``, and score **all** ``V`` rows with the projection
    stage 2  REFINE   rescore the top-``N`` rows of that ranking with the *full* ``h``
    tail              keeps its stage-1 score -- approximate, but never ``-inf``

**Why this is a read reduction and not just arithmetic.** Writing ``S`` for the selected
coordinates, ``U_S`` for those columns of ``U`` and ``A = W U``,

    A[:, S] (U_S^T h)  ==  W U_S U_S^T h  ==  W h~ ,     h~ = U_S U_S^T h

so screening with the *projected hidden state* against the unrotated ``W`` is
**identical arithmetic** to reading ``r0`` columns of the rotated head ``A``. A
deployment stores ``A`` (plus ``U``) and touches ``r0*V + D^2`` numbers; this module
simulates it with ``W`` and ``h~``, which keeps the refine stage bit-identical to the
dense head instead of paying a second BF16 rounding through ``A``. Same convention as
the rest of ``src/lm_head``: exact numerics, cost charged analytically.

Reads/token ``= r0*V + N*(D - r0) + D^2``; storage ``= V*D + D^2``. This is a
**read / active-parameter** method -- it does not reduce the stored parameter count, and
``lm_head_storage_struct.py`` is the evidence that nothing at 25% storage can.

The rotation ``U`` is the eigenbasis of the activation second moment ``C``. It is worth
its ``D^2``: it decorrelates the coordinates, which is what makes a per-token top-``r0``
capture far more of ``||W h||`` than the standard basis does (measured 5x lower KL than
``basis="raw"``). ``basis="raw"`` (``U = I``) is the zero-calibration fallback and still
lands at KL 0.009 -- for anyone who does not want an eigendecomposition.
"""

from typing import Optional, Tuple

import torch

from src.base.shared_utils import _print

__all__ = ["build_screen_refine", "screen_refine_cost"]


@torch.no_grad()
def build_screen_refine(
    W: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    basis: str = "ceig",
    ridge: float = 1e-3,
    compute_device: str = "cpu",
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Return ``(U, col_norm, static_score, stats)`` for the screen stage.

    ``U`` is ``(D, D)`` orthogonal (identity for ``basis="raw"``) and ``col_norm[i]`` is
    ``||W u_i||``, so the per-token contribution of coordinate ``i`` to the logit vector
    is ``|coef_i| * col_norm[i]`` -- the score the screen ranks by. Without the norm
    factor the score would compare coordinates whose columns differ in scale by 1.5x
    and pick the wrong ones.
    """
    dev = compute_device
    V, D = W.shape
    if basis == "raw":
        U = torch.eye(D, dtype=torch.float32)
    elif basis == "ceig":
        if C is None:
            raise ValueError('screen_refine basis="ceig" needs the activation metric C')
        Cd = C.to(dtype=torch.float64)
        cbar = torch.diagonal(Cd).mean().clamp_min(1e-30)
        Cd = Cd + float(ridge) * cbar * torch.eye(D, dtype=Cd.dtype)
        ev, Q = torch.linalg.eigh(Cd)
        U = Q[:, torch.argsort(ev, descending=True)].float()
    else:
        raise ValueError(f"unknown screen_refine basis {basis!r} (ceig|raw)")

    # ||W u_i|| for every i, chunked so no V x D float32 copy is needed twice
    col = torch.zeros(D, dtype=torch.float64)
    Ud = U.to(dev)
    for s in range(0, V, 16384):
        blk = W[s:s + 16384].to(device=dev, dtype=torch.float32)
        col += (blk @ Ud).pow(2).sum(0).double().cpu()
    col_norm = col.sqrt().float()

    # The static-screen ablation needs the *expected* contribution of each coordinate,
    # sqrt(E[coef_i^2]) * ||W u_i||. In the C eigenbasis E[coef_i^2] is the eigenvalue,
    # so this is available without touching a single hidden state; selecting the top r0
    # of it is exactly activation-aware low-rank.
    if basis == "ceig":
        rms = torch.diagonal(U.T.double() @ C.double() @ U.double()).clamp_min(0).sqrt()
    elif C is not None:
        rms = torch.diagonal(C.double()).clamp_min(0).sqrt()
    else:
        rms = torch.ones(D, dtype=torch.float64)
    static_score = (rms.float() * col_norm)

    stats = {
        "basis": basis,
        "col_norm_p99_over_p50": float(col_norm.quantile(0.99) / col_norm.median()),
        "col_norm_max_over_min": float(col_norm.max() / col_norm.min().clamp_min(1e-30)),
    }
    if verbose:
        _print(
            f"[lm_head/S1] screen basis={basis}, ||W u_i|| spread "
            f"p99/p50={stats['col_norm_p99_over_p50']:.2f}, "
            f"max/min={stats['col_norm_max_over_min']:.1f}"
        )
    return U, col_norm, static_score, stats


def screen_refine_cost(V: int, D: int, screen_rank: int, cand_size: int) -> dict:
    """Parameter counts for one ``(r0, N)`` operating point.

    Reads charge the screen (``r0`` columns of every row), the rotation (``D^2``, needed
    in full because the selection depends on every coordinate of ``coef``), and the
    *incremental* cost of the refine stage -- the candidates' remaining ``D - r0``
    coordinates, since their first ``r0`` were already touched by the screen.
    """
    r0 = int(min(screen_rank, D))
    N = int(min(cand_size, V))
    read = r0 * V + N * max(D - r0, 0) + D * D
    stored = V * D + D * D
    return {
        "screen_rank": r0, "cand_size": N,
        "stored_params": stored, "read_params": read,
        "stored_param_frac": stored / (V * D), "read_param_frac": read / (V * D),
        # bits/param on the byte axis: nothing is quantized, so 16 bits scaled by the
        # rotation's share of the stored count.
        "bits_per_weight": 16.0 * stored / (V * D),
        "read_bits_per_weight": 16.0,
    }
