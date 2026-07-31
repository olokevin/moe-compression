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
    beta: float = 1.0,
    col_norm=None,
    pubsub_artifact=None,
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
    # Cross-expert criteria emerge per-expert quotas from a global threshold, so
    # they impose no k_min floor (a dominated expert may get 0 channels).
    cross_expert = criterion in ("oracle_mag", "pubsub")
    if not cross_expert:
        B = max(K * k_min, min(B, K * I))  # keep feasible
    else:
        B = min(B, K * I)
    # Number of MoE layers to install over (artifact.L for the router-only path;
    # pubsub carries its own L; oracle_mag derives it below).
    if pubsub_artifact is not None:
        n_moe_layers = pubsub_artifact.L
    elif artifact is not None:
        n_moe_layers = artifact.L
    else:
        n_moe_layers = None

    metric_str = artifact.channel_metric if artifact is not None else criterion
    if verbose:
        _print(
            f"[DynamicAlloc] Installing: criterion={criterion}, metric={metric_str}, "
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

        if n_moe_layers is not None and mask_idx >= n_moe_layers:
            raise IndexError(
                f"More MoE layers than artifact layers ({n_moe_layers}); "
                "scores_dir does not match this model."
            )

        block_device = next(moe_block.parameters()).device
        # router-only ranking tensors (unused by cross-expert criteria).
        if artifact is not None:
            moe_block._dyn_ranks = artifact.channel_rank[mask_idx].to(block_device)   # (E, I) long
            moe_block._dyn_contrib = artifact.contrib[mask_idx].to(block_device)      # (E,) float
        moe_block._dyn_prefix = (
            artifact.prefix_sums[mask_idx].to(block_device) if criterion == "coverage_alloc" else None
        )
        moe_block._dyn_gains = (
            artifact.gains[mask_idx].to(block_device) if criterion == "pivchol_global" else None
        )
        moe_block._dyn_beta = float(beta)

        # oracle_mag: exact per-token magnitude needs the down_proj column norms.
        if criterion == "oracle_mag":
            cn = torch.stack(
                [e.down_proj.weight.detach().float().norm(dim=0) for e in experts], dim=0
            )  # (E, I) = ||W_down[:, j]||_2 per expert/channel
            moe_block._dyn_col_norm = cn.to(block_device)

        # pubsub: private ranks/gains + public carriers.
        if criterion == "pubsub":
            pa = pubsub_artifact
            moe_block._dyn_pub_pivrank = pa.pivrank_priv[mask_idx].to(block_device)  # (E, I)
            moe_block._dyn_pub_gains = pa.gains_priv[mask_idx].to(block_device)      # (E, I)
            moe_block._dyn_pub_carrier_idx = pa.carrier_idx[mask_idx].to(block_device)  # (r, E)
            moe_block._dyn_pub_carrier_val = pa.carrier_val[mask_idx].to(block_device)  # (r, E)

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
