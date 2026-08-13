"""Adapters that plug a channel selector into the *real* forward pass.

The masking simulation in ``src/dynamic_active_param/block.py`` already materializes
each token's ``(K, I)`` intermediate for the Level-2 cross-expert criteria, so any
selector that can turn ``(h, g, sel)`` into a keep-mask can be evaluated end-to-end by
adding a single criterion. ``criterion='channel_router'`` does exactly that: the block
calls the object installed at ``block._dyn_router`` and applies the mask it returns.

Three modes cover everything the plan needs from the real forward pass:

``oracle``          the exact ``imp`` top-B — the reference ceiling (identical to the
                    existing ``oracle_mag`` criterion; used as a self-check).
``oracle_degrade``  the oracle top-B with a fraction of its channels replaced by the
                    next-ranked ones — the §0.3 calibration curve's controlled
                    degradation, which converts mass-recall into ΔPPL.
``predict``         any ``ChannelRouter`` / baseline scorer, i.e. the deployed router.

Every mode optionally accumulates mass-recall and output-error statistics against the
exact oracle *inside the real forward pass*, so a single PPL job reports both the end
metric and the diagnostic that predicts it.
"""

from __future__ import annotations

import types

import torch

from src.base.shared_utils import _print
from src.base.shared_utils.safe_isinstance import (
    _get_experts,
    _get_moe_block,
    _get_moe_intermediate_size,
    _get_num_hidden_layers,
    _get_topk,
)
from src.channel_router.metrics import degrade_oracle_keep, select_topB
from src.dynamic_active_param.block import dynamic_moe_block_forward

__all__ = ["ForwardSelector", "install_channel_router", "moe_layer_indices"]


class ForwardSelector:
    """Callable installed on a MoE block; returns the ``(T,K,I)`` keep-mask.

    Args:
        mode: ``oracle`` | ``oracle_degrade`` | ``predict``.
        col_norm: ``(E, I)`` ``‖W_d[:,j]‖`` on the block's device.
        router: a ``ChannelRouter`` or baseline scorer (mode ``predict``).
        drop_frac: degradation fraction (mode ``oracle_degrade``).
        slack: budget multiplier ``s`` of §1.1 step 4 / §2.4.
        top_tiles: restrict to the top-n tiles first (0 = off).
        collect_stats: accumulate mass-recall / recall vs the exact oracle.
    """

    def __init__(self, mode: str, col_norm: torch.Tensor, *, router=None,
                 drop_frac: float = 0.0, slack: float = 1.0, top_tiles: int = 0,
                 collect_stats: bool = True, seed: int = 0, use_colnorm: bool = True):
        if mode not in ("oracle", "oracle_degrade", "predict"):
            raise ValueError(mode)
        self.mode = mode
        self.col_norm = col_norm
        self.router = router
        self.drop_frac = float(drop_frac)
        self.slack = float(slack)
        self.top_tiles = int(top_tiles)
        self.collect_stats = collect_stats
        self.use_colnorm = use_colnorm
        self.gen = torch.Generator(device=col_norm.device).manual_seed(seed)
        self.stats = {"tokens": 0, "mass_recall": 0.0, "recall": 0.0, "kept": 0.0}

    def exact_imp(self, inter_all, g, sel):
        s = inter_all.abs().float()
        if self.use_colnorm:
            s = s * self.col_norm[sel]
        return s * g.to(s.device, torch.float32).unsqueeze(-1)

    @torch.no_grad()
    def __call__(self, hidden_states, routing_weights, selected_experts, inter_all, B):
        imp = self.exact_imp(inter_all, routing_weights, selected_experts)
        budget = min(int(round(B * self.slack)), inter_all.shape[1] * inter_all.shape[2])
        if self.mode == "oracle":
            keep = select_topB(imp, budget)
        elif self.mode == "oracle_degrade":
            keep = degrade_oracle_keep(imp, budget, self.drop_frac, generator=self.gen)
        else:
            h = hidden_states.to(torch.float32)
            score = self.router.score(h, selected_experts, routing_weights)
            if hasattr(self.router, "select_from_score"):
                keep = self.router.select_from_score(
                    score, selected_experts, B, top_tiles=self.top_tiles,
                    slack=self.slack)
            else:
                keep = select_topB(score, budget)
        if self.collect_stats:
            T = keep.shape[0]
            ref = select_topB(imp, B)
            f = imp.reshape(T, -1)
            num = (f * keep.reshape(T, -1)).sum(1)
            den = (f * ref.reshape(T, -1)).sum(1).clamp_min(1e-20)
            inter = (keep & ref).reshape(T, -1).sum(1).float()
            self.stats["tokens"] += T
            self.stats["mass_recall"] += float((num / den).sum())
            self.stats["recall"] += float((inter / max(B, 1)).sum())
            self.stats["kept"] += float(keep.reshape(T, -1).sum())
        return keep

    def summary(self):
        n = max(self.stats["tokens"], 1)
        return {"tokens": self.stats["tokens"],
                "mass_recall": self.stats["mass_recall"] / n,
                "recall": self.stats["recall"] / n,
                "kept_per_token": self.stats["kept"] / n}


def moe_layer_indices(model):
    """Absolute layer indices that carry a MoE block, in order."""
    out = []
    for i in range(_get_num_hidden_layers(model)):
        blk = _get_moe_block(model, i)
        if _get_experts(blk) is not None:
            out.append(i)
    return out


def install_channel_router(model, *, prune_ratio: float, mode: str = "oracle",
                           routers: dict | None = None, layers=None,
                           drop_frac: float = 0.0, slack: float = 1.0,
                           top_tiles: int = 0, collect_stats: bool = True,
                           use_colnorm: bool = True, verbose: bool = True):
    """Bind the channel-router forward onto the requested MoE blocks.

    Unlike ``install_dynamic_alloc`` this installs on a **subset** of layers, leaving
    the rest fully dense — the plan's §2 "train on single layers first" scope control
    needs per-layer ΔPPL, which is only interpretable when nothing else is pruned.

    Args:
        prune_ratio: fraction of activated channels removed per token (B = (1−ρ)·K·I).
        mode / drop_frac / slack / top_tiles: see ``ForwardSelector``.
        routers: ``{absolute_layer_index: scorer}`` for mode ``predict``.
        layers: absolute layer indices to install on (default: all MoE layers).

    Returns:
        ``{layer_index: ForwardSelector}`` so the caller can read the stats back.
    """
    I = _get_moe_intermediate_size(model)
    K = _get_topk(model)
    B = min(int(round((1.0 - prune_ratio) * K * I)), K * I)
    all_moe = moe_layer_indices(model)
    targets = all_moe if layers is None else [l for l in layers if l in all_moe]
    out = {}
    for li in targets:
        blk = _get_moe_block(model, li)
        experts = _get_experts(blk)
        dev = next(blk.parameters()).device
        cn = torch.stack([e.down_proj.weight.detach().float().norm(dim=0)
                          for e in experts], 0).to(dev)
        router = None
        if mode == "predict":
            if routers is None or li not in routers:
                raise KeyError(f"no router for layer {li}")
            router = routers[li]
            if hasattr(router, "to"):
                router = router.to(dev)
        sel = ForwardSelector(mode, cn, router=router, drop_frac=drop_frac,
                              slack=slack, top_tiles=top_tiles,
                              collect_stats=collect_stats, use_colnorm=use_colnorm)
        blk._dyn_router = sel
        blk._dyn_col_norm = cn
        blk._dyn_B = B
        blk._dyn_k_min = 0
        blk._dyn_I = int(I)
        blk._dyn_criterion = "channel_router"
        blk.forward = types.MethodType(dynamic_moe_block_forward, blk)
        out[li] = sel
    if verbose:
        _print(f"[ChannelRouter] mode={mode} B={B}/{K * I} (rho={B / (K * I):.4f}) "
               f"on {len(out)}/{len(all_moe)} MoE layers"
               + (f", drop_frac={drop_frac}" if mode == "oracle_degrade" else "")
               + (f", slack={slack}" if slack != 1.0 else ""))
    return out
