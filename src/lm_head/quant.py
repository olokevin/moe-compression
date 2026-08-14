"""Shared numerics for the lm_head baselines: group RTN, the activation metric,
and randomized truncated SVD.

Everything here is a **numerics-only simulation**: we return dequantized dense
tensors holding exactly the values a packed kernel would unpack, and charge the
bytes analytically in :mod:`src.lm_head.accounting`. That split is deliberate --
packing changes memory use, not the arithmetic under test -- and it is the same
convention ``src/dynamic_active_param/sparse_probe.py`` uses for its b-bit proxy.

The group RTN quantizer itself is imported from ``sparse_probe`` rather than
re-implemented, so the head and the expert path are charged by the same cost model
(``bits + scale_bits/group`` bits per weight).
"""

from typing import Optional, Tuple

import torch

from src.base.shared_utils import _print
from src.dynamic_active_param.sparse_probe import quantize_rtn_dequant

__all__ = [
    "quantize_rtn_dequant",
    "bits_per_weight",
    "quantize_rows_mixed",
    "quantize_int8_rowwise",
    "metric_transform",
    "randomized_svd",
]


def bits_per_weight(bits: int, group: int, scale_bits: int = 16) -> float:
    """Effective bits/weight for group-wise quantization, scales included.

    ``bits >= 16`` means "not quantized" and costs exactly 16 (BF16), with no
    scale overhead.
    """
    if bits >= 16:
        return 16.0
    if group is None or group <= 0:
        return float(bits)
    return float(bits) + float(scale_bits) / float(group)


@torch.no_grad()
def quantize_rows_mixed(
    W: torch.Tensor,
    head_mask: torch.Tensor,
    head_bits: int,
    tail_bits: int,
    group: int = 128,
) -> torch.Tensor:
    """B1-s: quantize head-tier rows at ``head_bits`` and tail rows at ``tail_bits``.

    Row-disjoint, so each tier gets its own group scales -- no group ever straddles
    the tier boundary. ``W`` is ``(V, D)``; ``head_mask`` is ``(V,) bool``.
    """
    out = W.clone()
    for mask, bits in ((head_mask, head_bits), (~head_mask, tail_bits)):
        if bits >= 16 or not bool(mask.any()):
            continue
        rows = torch.nonzero(mask, as_tuple=True)[0]
        # chunk so a 152k x 2048 fp32 intermediate never has to exist twice
        for s in range(0, rows.numel(), 16384):
            sel = rows[s:s + 16384]
            out[sel] = quantize_rtn_dequant(W[sel], bits=bits, group=group)
    return out


@torch.no_grad()
def quantize_int8_rowwise(W: torch.Tensor) -> torch.Tensor:
    """Symmetric per-row INT8 quantize+dequantize (ARCHead stores ``A_w`` this way)."""
    scale = W.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12).float() / 127.0
    scale = scale.to(torch.float16).float()
    q = torch.clamp(torch.round(W.float() / scale), -128, 127)
    return (q * scale).to(W.dtype)


@torch.no_grad()
def metric_transform(
    C: torch.Tensor,
    p: float = 0.75,
    ridge: float = 1e-3,
    compute_device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """ARCHead's damped activation-metric transform ``T_p`` and its inverse.

    With ``C_lambda = C + lambda * mean(diag C) * I = Q L Q^T`` (the damping is
    relative, per ARCHead 3.1, so ``ridge`` is scale-free),

        T_p = Q L^p Q^T,     T_p^-1 = Q L^-p Q^T.

    ``p = 1/2`` makes ``||(W - W_hat) T_p||_F^2`` exactly the damped empirical logit
    MSE ``Tr(D C D^T)``; larger ``p`` leans harder on the dominant activation
    directions. ARCHead selects ``p = 0.75`` on Qwen3-8B-Base, which is the default
    here.

    Runs on CPU by default. This is not a performance choice: a ``D x D``
    eigendecomposition issued on a GPU that is still holding a 30B model shard has
    been observed to take CUBLAS down (see the repo's Nystrom notes), and 2048^2
    ``eigh`` on CPU costs a couple of seconds.
    """
    Cd = C.to(device=compute_device, dtype=torch.float64)
    cbar = torch.diagonal(Cd).mean().clamp_min(1e-30)
    Cd = Cd + float(ridge) * cbar * torch.eye(Cd.shape[0], dtype=Cd.dtype, device=Cd.device)
    evals, Q = torch.linalg.eigh(Cd)
    # eigh can return tiny negatives on a numerically-PSD matrix; the floor keeps
    # L^-p finite. It is relative so it tracks the activation scale.
    floor = evals.max().clamp_min(1e-30) * 1e-12
    evals = evals.clamp_min(floor)
    Lp = evals.pow(float(p))
    Tp = (Q * Lp.unsqueeze(0)) @ Q.T
    Tp_inv = (Q * Lp.reciprocal().unsqueeze(0)) @ Q.T
    info = {
        "cond": float((evals.max() / evals.min()).item()),
        "eig_max": float(evals.max().item()),
        "eig_min": float(evals.min().item()),
        "p": float(p),
        "ridge": float(ridge),
    }
    return Tp.float(), Tp_inv.float(), info


@torch.no_grad()
def randomized_svd(
    A: torch.Tensor,
    rank: int,
    oversample: int = 10,
    n_iter: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomized truncated SVD of a tall ``(V, D)`` matrix. Returns ``(U, S, Vh)``.

    ``torch.svd_lowrank`` with power iterations; at the ranks these baselines use
    (6-10 for ARCHead, up to 1024 for the F2 low-rank ladder) this is both faster
    and lower-memory than a full ``linalg.svd`` of a 152k x 2048 matrix.
    """
    q = int(min(A.shape[0], A.shape[1], rank + oversample))
    U, S, V = torch.svd_lowrank(A.float(), q=q, niter=int(n_iter))
    r = int(min(rank, S.numel()))
    return U[:, :r], S[:r], V[:, :r].T
