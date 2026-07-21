"""Install the dynamic-allocation forward onto a model's MoE blocks.

Walks the model layers with the same layer -> MoE-index mapping used by
``fake_prune_wrapper`` (skip non-MoE layers, count the rest), computes the
per-token total channel budget ``B = round((1 - prune_ratio) * K * I)``, and
binds ``dynamic_moe_block_forward`` onto each ``layer.mlp`` via
``types.MethodType``. Per-layer rank/contrib tensors are moved to each block's
own device so it works under ``device_map='auto'`` sharding.
"""

import types

import torch

from src.base.shared_utils import _print
from src.base.shared_utils.safe_isinstance import (
    _get_moe_block,
    _get_experts,
    _get_moe_intermediate_size,
    _get_num_hidden_layers,
    _get_topk,
)
from src.dynamic_active_param.block import dynamic_moe_block_forward
from src.dynamic_active_param.precompute import AllocArtifact

__all__ = ["install_dynamic_alloc"]


def install_dynamic_alloc(
    model,
    artifact: AllocArtifact,
    prune_ratio: float,
    criterion: str = "router_prob",
    k_min: int = 16,
    verbose: bool = True,
):
    """Bind the dynamic MoE forward onto every MoE block of ``model``.

    Args:
        model: HF causal-LM (un-slimmed; masking simulation keeps full weights).
        artifact: AllocArtifact from ``build_alloc_artifact`` (channel_rank, contrib).
        prune_ratio: fraction of activated expert-FFN channels to remove per token.
        criterion: router_prob | contribution | uniform.
        k_min: per-expert floor on kept channels.
        verbose: print progress.

    Returns:
        The same model, with dynamic forwards installed.
    """
    I = _get_moe_intermediate_size(model)
    K = _get_topk(model)
    B = int(round((1.0 - prune_ratio) * K * I))
    B = max(K * k_min, min(B, K * I))  # keep feasible

    if verbose:
        _print(
            f"[DynamicAlloc] Installing: criterion={criterion}, metric={artifact.channel_metric}, "
            f"K={K}, I={I}, prune_ratio={prune_ratio}, B={B} (of K*I={K*I}), k_min={k_min}"
        )

    num_layers = _get_num_hidden_layers(model)

    # Same layer -> MoE-index mapping as fake_prune_wrapper: count MoE layers
    # in order; artifact.channel_rank[mask_idx] corresponds to that MoE layer.
    mask_idx = 0
    n_installed = 0
    for layer_idx in range(num_layers):
        moe_block = _get_moe_block(model, layer_idx)
        experts = _get_experts(moe_block)
        if experts is None:
            continue

        if mask_idx >= artifact.L:
            raise IndexError(
                f"More MoE layers than artifact layers ({artifact.L}); "
                "scores_dir does not match this model."
            )

        block_device = next(moe_block.parameters()).device
        moe_block._dyn_ranks = artifact.channel_rank[mask_idx].to(block_device)   # (E, I) long
        moe_block._dyn_contrib = artifact.contrib[mask_idx].to(block_device)      # (E,) float
        # prefix sums only needed by coverage_alloc; keep it off other blocks.
        if criterion == "coverage_alloc":
            moe_block._dyn_prefix = artifact.prefix_sums[mask_idx].to(block_device)  # (E, I) float
        else:
            moe_block._dyn_prefix = None
        moe_block._dyn_B = B
        moe_block._dyn_k_min = int(k_min)
        moe_block._dyn_I = int(I)
        moe_block._dyn_criterion = criterion
        moe_block.forward = types.MethodType(dynamic_moe_block_forward, moe_block)

        mask_idx += 1
        n_installed += 1

    if verbose:
        _print(f"[DynamicAlloc] ✅ Installed dynamic forward on {n_installed} MoE blocks")

    return model
