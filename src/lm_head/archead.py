"""B2 -- ARCHead (arXiv:2608.02703), numerics-only reconstruction.

*Activation-Metric Residual Correction for LLM Output Heads.* This is the best
published storage/quality point in the survey and, decisively, it was validated on
VibeThinker-3B whose head is ``(V, D) = (151936, 2048)`` -- the **identical shape**
as Qwen3-30B-A3B's -- at 3.873x compression. That is the strongest available prior
that it transfers to our target.

Algorithm 1 of the paper, implemented verbatim:

    1.  C_lambda <- H^T H / N + lambda * mean(diag) * I
    2.  A_c B_c   <- rank-r_c SVD of W
        W_d       <- Q5(A_c) Q8(B_c) + SC4(W - A_c B_c)          (core, eq. 3)
    3.  E         <- W - W_d
    4.  Q, L      <- eigh(C_lambda);   T_p <- Q L^p Q^T
    5.  U S V^T   <- rank-r_r randomized SVD of E T_p            (eq. 4)
    6.  A_w <- U S;  B_w <- V^T T_p^-1                           (eq. 5)
    7.  A_w row-wise INT8, B_w group-wise INT8
    8.  W_hat <- W_d + A_w B_w                                   (eq. 6)

Published hyperparameters for Qwen: ``r_c = 10, r_r = 6, g = 64, p = 0.75,
ridge = 1e-3`` -- the defaults below.

**What is faithful and what is not.** The decomposition, the metric transform, the
objective, and the hyperparameters are the paper's. What we do *not* reproduce is
the packed CUDA representation: we return a dense BF16 tensor holding the values a
packed kernel would unpack, and charge storage analytically. The paper is explicit
that a dequantized prototype is not a storage claim, so our storage number is an
analytic bit count, not a measured ``state_dict`` -- reported as such. The paper's
own measured ratio on this head shape is 0.258 (153.23 / 593.5 MB); our analytic
count lands in the same 25-27% band. Accuracy transfers; throughput claims do not.
"""

from typing import Optional

import torch

from src.base.shared_utils import _print
from src.lm_head.quant import (
    bits_per_weight,
    metric_transform,
    quantize_int8_rowwise,
    quantize_rtn_dequant,
    randomized_svd,
)

__all__ = ["build_archead", "archead_bits_per_weight"]


def archead_bits_per_weight(
    V: int,
    D: int,
    rc: int = 10,
    rr: int = 6,
    group: int = 64,
    residual_bits: int = 4,
    core_left_bits: int = 5,
    core_right_bits: int = 8,
    correction_bits: int = 8,
    scale_bits: int = 16,
) -> dict:
    """Analytic bits/weight of a packed ARCHead, broken out by component.

    Everything is expressed per element of the ``V x D`` dense head so it can be
    compared directly against BF16's 16 bits/weight.
    """
    dense = V * D
    # group-wise INT4 residual over the full V x D, one fp16 scale per group of g
    resid = dense * residual_bits + (dense / max(group, 1)) * scale_bits
    # quantized low-rank core factors: A_c is (V, rc) at 5 bits, B_c is (rc, D) at 8
    core = V * rc * core_left_bits + (V * rc / max(group, 1)) * scale_bits
    core += rc * D * core_right_bits + (rc * D / max(group, 1)) * scale_bits
    # INT8 correction factors: A_w row-wise (one scale per row), B_w group-wise
    corr = V * rr * correction_bits + V * scale_bits
    corr += rr * D * correction_bits + (rr * D / max(group, 1)) * scale_bits
    total = resid + core + corr
    return {
        "bits_per_weight": total / dense,
        "residual_bpw": resid / dense,
        "core_bpw": core / dense,
        "correction_bpw": corr / dense,
        "storage_frac_of_bf16": total / (dense * 16.0),
    }


@torch.no_grad()
def build_archead(
    W: torch.Tensor,
    C: torch.Tensor,
    rc: int = 10,
    rr: int = 6,
    group: int = 64,
    p: float = 0.75,
    ridge: float = 1e-3,
    residual_bits: int = 4,
    core_left_bits: int = 5,
    core_right_bits: int = 8,
    activation_metric: bool = True,
    compute_device: str = "cpu",
    verbose: bool = True,
):
    """Build the ARCHead approximation of ``W``. Returns ``(W_hat, stats)``.

    ``W`` is ``(V, D)``; ``C`` is the ``(D, D)`` activation second moment from
    :func:`src.lm_head.calib.ensure_sigma`.

    ``activation_metric=False`` sets ``T_p = I``, i.e. fits the correction in plain
    Frobenius error. That is the ablation that isolates the paper's central claim
    (and the pilots' independent finding that the metric is what moves the number:
    whitening took a low-rank head from PPL 1477 to 42).
    """
    dev = compute_device
    Wf = W.detach().to(device=dev, dtype=torch.float32)
    V, D = Wf.shape

    # --- step 2: quantized low-rank core + group INT4 residual (eq. 3) ---------
    Uc, Sc, Vhc = randomized_svd(Wf, rank=rc)
    Ac = Uc * Sc.unsqueeze(0)                       # (V, rc)
    Bc = Vhc                                        # (rc, D)
    Ac_q = quantize_rtn_dequant(Ac, bits=core_left_bits, group=min(group, rc))
    Bc_q = quantize_rtn_dequant(Bc, bits=core_right_bits, group=group)
    core = Ac_q @ Bc_q
    resid_in = Wf - Ac @ Bc
    Wd = core + quantize_rtn_dequant(resid_in, bits=residual_bits, group=group)
    del resid_in, core, Uc, Sc, Vhc, Ac, Bc

    # --- steps 3-6: rank-rr correction fitted in the activation metric --------
    E = Wf - Wd
    if activation_metric:
        Tp, Tp_inv, minfo = metric_transform(C, p=p, ridge=ridge, compute_device=dev)
        Tp, Tp_inv = Tp.to(dev), Tp_inv.to(dev)
    else:
        Tp = torch.eye(D, device=dev, dtype=torch.float32)
        Tp_inv = Tp
        minfo = {"cond": 1.0, "p": None, "ridge": None}
    Uw, Sw, Vhw = randomized_svd(E @ Tp, rank=rr)
    Aw = Uw * Sw.unsqueeze(0)                       # (V, rr)
    Bw = Vhw @ Tp_inv                               # (rr, D)

    # --- step 7: INT8 factor quantization ------------------------------------
    Aw_q = quantize_int8_rowwise(Aw)
    Bw_q = quantize_rtn_dequant(Bw, bits=8, group=group)

    W_hat = Wd + Aw_q @ Bw_q

    cost = archead_bits_per_weight(
        V, D, rc=rc, rr=rr, group=group, residual_bits=residual_bits,
        core_left_bits=core_left_bits, core_right_bits=core_right_bits,
    )
    # Report the error in BOTH metrics: plain Frobenius is what a naive method
    # optimizes, and the C-weighted error is what actually reaches the softmax.
    Cd = C.to(device=dev, dtype=torch.float32)

    def _cerr(Dm):
        # sqrt(Tr(D C D^T)) == the RMS logit error the softmax actually sees.
        # Written as an explicit (D @ C) * D contraction: at V=151936 letting einsum
        # pick the order risks a V x V intermediate.
        return float(((Dm @ Cd) * Dm).sum().clamp_min(0).sqrt())

    denom_f = float(Wf.norm())
    stats = dict(cost)
    stats.update({
        "rc": rc, "rr": rr, "group": group, "p": p, "ridge": ridge,
        "residual_bits": residual_bits, "activation_metric": bool(activation_metric),
        "rel_fro_err": float((W_hat - Wf).norm() / max(denom_f, 1e-30)),
        "rel_fro_err_core_only": float((Wd - Wf).norm() / max(denom_f, 1e-30)),
        "rel_metric_err": _cerr(W_hat - Wf) / max(_cerr(Wf), 1e-30),
        "rel_metric_err_core_only": _cerr(Wd - Wf) / max(_cerr(Wf), 1e-30),
        "metric_cond": minfo["cond"],
    })
    if verbose:
        _print(
            f"[lm_head/B2 ARCHead] rc={rc} rr={rr} g={group} p={p} ridge={ridge} "
            f"activation_metric={activation_metric} -> "
            f"{stats['bits_per_weight']:.3f} bits/weight "
            f"({100 * stats['storage_frac_of_bf16']:.1f}% of BF16); "
            f"rel err Frobenius {stats['rel_fro_err']:.4f} "
            f"(core only {stats['rel_fro_err_core_only']:.4f}), "
            f"in the C metric {stats['rel_metric_err']:.4f} "
            f"(core only {stats['rel_metric_err_core_only']:.4f})"
        )
    return W_hat.to(W.dtype), stats
