"""§0.3 global evaluation standards: recall, importance-mass recall, output error,
and the §1.2 parameter/FLOP accounting.

Two rules from the plan are enforced here rather than left to each script:

1. Recall is *diagnostic only*; the deciding metric is what the predicted mask does
   to the model (KL / ΔPPL, measured by ``scripts/channel_router/ppl_ladder.py``).
   ``mass_recall`` is the recall variant that correlates with it, so every recall
   report carries the mass version alongside.
2. The budget is a per-token top-B, never a global threshold — floating it under one
   threshold measures router confidence rather than token compressibility (measured
   −0.16..−0.32pt at iso-cost in the sparse-probe study), so ``select_topB`` is the
   only selection entry point.
"""

from __future__ import annotations

import torch

from src.dynamic_active_param.allocate import select_global_topB as select_topB

__all__ = [
    "select_topB", "recall", "mass_recall", "recall_pair", "output_rel_err",
    "router_accounting", "degrade_oracle_keep",
]


def recall(pred_keep: torch.Tensor, ref_keep: torch.Tensor) -> float:
    """Mean |predicted ∩ oracle| / B over tokens. Both ``(T, K, I)`` bool."""
    T = pred_keep.shape[0]
    inter = (pred_keep & ref_keep).reshape(T, -1).sum(1).float()
    denom = ref_keep.reshape(T, -1).sum(1).float().clamp_min(1)
    return float((inter / denom).mean())


def mass_recall(imp: torch.Tensor, pred_keep: torch.Tensor,
                ref_keep: torch.Tensor) -> float:
    """Importance-mass recall: imp captured by the prediction / imp of the oracle top-B."""
    T = pred_keep.shape[0]
    f = imp.reshape(T, -1)
    num = (f * pred_keep.reshape(T, -1)).sum(1)
    den = (f * ref_keep.reshape(T, -1)).sum(1).clamp_min(1e-20)
    return float((num / den).mean())


def recall_pair(score: torch.Tensor, imp: torch.Tensor, B: int):
    """Convenience: select top-B by ``score``, score it against ``imp``'s own top-B."""
    ref = select_topB(imp, B)
    pred = select_topB(score, B)
    return recall(pred, ref), mass_recall(imp, pred, ref)


@torch.no_grad()
def output_rel_err(inter: torch.Tensor, keep: torch.Tensor, sel: torch.Tensor,
                   g: torch.Tensor, Wd: torch.Tensor) -> float:
    """Relative error of the MoE block output caused by dropping the non-kept channels.

    ``y = Σ_e g_e · W_d^{(e)} (m_e ⊙ inter_e)``; returns mean over tokens of
    ``‖y_full − y_kept‖ / ‖y_full‖``. This is the currency that predicts accuracy
    (a missed channel with a small ``W_d`` column is free), with a measured slope of
    −26.4 HellaSwag pt per unit rel_err for mis-selection at a fixed budget.
    """
    T, K, I = inter.shape
    dev = inter.device
    H = Wd.shape[1]
    y_full = torch.zeros((T, H), dtype=torch.float32, device=dev)
    y_keep = torch.zeros((T, H), dtype=torch.float32, device=dev)
    gg = g.to(dev).float()
    for e in sel.unique().tolist():
        tok, slot = (sel == e).nonzero(as_tuple=True)
        v = inter[tok, slot]                                   # (n, I)
        w = Wd[e].to(torch.float32)                            # (H, I)
        coef = gg[tok, slot].unsqueeze(1)
        y_full.index_add_(0, tok, coef * (v @ w.t()))
        y_keep.index_add_(0, tok, coef * ((v * keep[tok, slot]) @ w.t()))
    num = (y_full - y_keep).norm(dim=1)
    den = y_full.norm(dim=1).clamp_min(1e-20)
    return float((num / den).mean())


def degrade_oracle_keep(imp: torch.Tensor, B: int, drop_frac: float,
                        generator: torch.Generator | None = None) -> torch.Tensor:
    """§0.3 calibration curve: drop ``drop_frac`` of the oracle top-B at random and
    backfill with the next-ranked channels, keeping the budget exactly ``B``.

    Gives a controlled family of masks whose mass-recall spans (1, ~0.8) so the
    recall→ΔPPL conversion can be *fitted* instead of assumed.
    """
    T, K, I = imp.shape
    flat = imp.reshape(T, K * I)
    n_drop = int(round(drop_frac * B))
    order = flat.argsort(dim=1, descending=True)
    take = order[:, :B + n_drop]                                # candidates
    if n_drop == 0:
        sel_idx = order[:, :B]
    else:
        # drop n_drop uniformly at random from the top-B, then backfill with the
        # next n_drop ranks (so the *replacement* is the strongest available).
        r = torch.rand((T, B), device=imp.device, generator=generator)
        keep_rank = r.argsort(dim=1)[:, :B - n_drop]            # which of top-B survive
        surv = torch.gather(take[:, :B], 1, keep_rank)
        sel_idx = torch.cat([surv, take[:, B:B + n_drop]], dim=1)
    keep = torch.zeros_like(flat, dtype=torch.bool)
    keep.scatter_(1, sel_idx, True)
    return keep.reshape(T, K, I)


def router_accounting(H: int, E: int, I: int, K: int, r: int, m: int, *,
                      head: str = "linear", n_tiles: int = 0,
                      hot_size: int = 0, rho: float = 0.125) -> dict:
    """§1.2 standard: router params ≤ 2% of FFN params, online FLOPs ≤ 3% of saved.

    Both denominators are reported because they differ by 16× on an MoE: the
    *stored* FFN of the layer is ``3·E·I·H`` while the *activated* FFN per token is
    ``3·K·I·H``. Channel embeddings must be keyed by physical channel (``E·I`` of
    them) since slot ``k`` means a different expert for every token, so they are
    charged against the stored FFN; the online matmul only touches the ``K``
    selected experts' blocks and is charged against the activated FLOPs.
    """
    rp = 2 if head in ("swiglu", "bilinear") else 1           # embeddings per channel
    proj = H * r * rp
    emb = E * I * (r + m) * rp
    bias = E * I
    params = proj + emb + bias + n_tiles
    ffn_stored = 3 * E * I * H
    ffn_active = 3 * K * I * H
    # online: project h once, then score the K selected experts' channels.
    flops = 2 * H * r * rp + 2 * K * I * (r + m) * rp
    saved = 2 * ffn_active * (1.0 - rho)
    return {
        "params": params, "params_proj": proj, "params_emb": emb, "params_bias": bias,
        "params_pct_stored_ffn": 100.0 * params / ffn_stored,
        "params_pct_active_ffn": 100.0 * params / ffn_active,
        "online_flops": flops,
        "flops_pct_active_ffn": 100.0 * flops / (2 * ffn_active),
        "flops_pct_saved": 100.0 * flops / saved,
        "hot_size": hot_size,
    }
