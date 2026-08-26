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
    _get_num_hidden_size,
    _get_topk,
)
from src.dynamic_active_param.block import dynamic_moe_block_forward
from src.dynamic_active_param.lowrank_scorer import (
    build_layer_scorer,
    print_scorer_accounting,
)
from src.dynamic_active_param.input_only import (
    InputOnlyCfg,
    print_input_only_accounting,
)
from src.dynamic_active_param.sparse_probe import (
    build_layer_probe,
    print_probe_accounting,
)
from src.dynamic_active_param.weight_sparse import (
    build_layer_wsparse,
    print_wsparse_accounting,
)
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
    scorer_kwargs=None,
):
    """Bind the dynamic MoE forward onto every MoE block of ``model``.

    Args:
        model: HF causal-LM (un-slimmed; masking simulation keeps full weights).
        artifact: AllocArtifact from ``build_alloc_artifact`` (channel_rank, contrib).
        prune_ratio: fraction of activated expert-FFN channels to remove per token.
        criterion: router_prob | contribution | uniform.
        k_min: per-expert floor on kept channels.
        verbose: print progress.
        scorer_kwargs: for ``criterion='lowrank_scorer'`` — dict with ``m``,
            ``n``, ``rank``, ``use_gate`` (bool), optional ``niter`` and
            ``compute_device``. Cores are factorized per layer at install time.

    Returns:
        The same model, with dynamic forwards installed.
    """
    I = _get_moe_intermediate_size(model)
    K = _get_topk(model)
    B = int(round((1.0 - prune_ratio) * K * I))
    # Per-layer (p, rho) schedule from scripts/probe_layer_surface.py: layers differ
    # ~2.8x in rel_err at identical cost, so a uniform schedule overspends on the
    # cheap layers. Keyed by absolute layer index; layers absent from the schedule
    # fall back to the global prune_ratio / rho_input.
    schedule = None
    if criterion == "sparse_probe" and (scorer_kwargs or {}).get("schedule_path"):
        import json as _json
        with open(scorer_kwargs["schedule_path"]) as _f:
            _sched = _json.load(_f)
        if isinstance(_sched, dict):          # a full layer_surface.json solution
            _sched = _sched.get("schedule", _sched)
        schedule = {int(e["layer"]): e for e in _sched}
        _print(
            f"[DynamicAlloc] per-layer probe schedule: {len(schedule)} layers from "
            f"{scorer_kwargs['schedule_path']} (mean kept "
            f"{sum(e.get('kept', 0.0) for e in schedule.values())/max(len(schedule),1):.4f})"
        )
    # Cross-expert criteria emerge per-expert quotas from a global threshold, so
    # they impose no k_min floor (a dominated expert may get 0 channels).
    cross_expert = criterion in (
        "oracle_mag", "oracle_mag_noW", "oracle_up", "pubsub", "lowrank_scorer",
        "sparse_probe", "weight_sparse", "input_only",
    )
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
    if criterion == "lowrank_scorer" and verbose:
        sk = scorer_kwargs or {}
        print_scorer_accounting(
            I=I, H=_get_num_hidden_size(model), m=sk["m"], n=sk["n"],
            rank=sk["rank"], n_scorers=2 if sk.get("use_gate", True) else 1,
            prune_ratio=prune_ratio,
        )
    if criterion == "sparse_probe" and verbose:
        sk = scorer_kwargs or {}
        print_probe_accounting(
            bits=sk.get("bits", 3), group=sk.get("group", 128),
            rho_input=sk.get("rho_input", 0.25),
            use_gate=sk.get("use_gate", True),
            rho_channel=1.0 - prune_ratio,
            lam=sk.get("lam", 1.0),
            input_alloc=sk.get("input_alloc", "uniform"),
        )

    if criterion == "input_only" and verbose:
        sk = scorer_kwargs or {}
        print_input_only_accounting(
            rho_input=sk.get("rho_input", 0.25),
            rho_channel=1.0 - prune_ratio,
            input_alloc=sk.get("input_alloc", "uniform"),
        )

    if criterion == "weight_sparse":
        sk = scorer_kwargs or {}
        # Per-layer calibration mean of the MoE input, for the mean-fix. One (H,)
        # vector per MoE layer, keyed by absolute layer index; see
        # scripts/collect_input_mean.py.
        wsp_mu = None
        if sk.get("mean_path"):
            _obj = torch.load(sk["mean_path"], map_location="cpu")
            _obj = _obj.get("mean", _obj) if isinstance(_obj, dict) else _obj
            wsp_mu = {int(k): v for k, v in _obj.items()}
            _print(f"[DynamicAlloc] mean-fix: {len(wsp_mu)} layer means from "
                   f"{sk['mean_path']}")
        if verbose:
            print_wsparse_accounting(
                levels=sk.get("levels", "0.25x0.45"),
                rho_channel=1.0 - prune_ratio,
                use_gate=bool(sk.get("use_gate", True)),
                I=I, H=_get_num_hidden_size(model),
                meanfix=wsp_mu is not None,
                input_alloc=str(sk.get("input_alloc", "uniform")),
                alloc_mode=str(sk.get("alloc_mode", "rank")),
                density=sk.get("density"),
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
        per_layer_B = B          # overridden below when a schedule applies

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

        # oracle_mag / oracle_up score by the exact per-token magnitude, which
        # weights each channel by its down_proj column norm ||W_down[:, j]||.
        # (oracle_mag_noW deliberately drops this factor — Q1 — so it needs none.)
        if criterion in ("oracle_mag", "oracle_up"):
            cn = torch.stack(
                [e.down_proj.weight.detach().float().norm(dim=0) for e in experts], dim=0
            )  # (E, I) = ||W_down[:, j]||_2 per expert/channel
            moe_block._dyn_col_norm = cn.to(block_device)

        # lowrank_scorer: factorize this layer's up_proj (and optionally
        # gate_proj) into block-low-rank cores used as the online ranking proxy.
        if criterion == "lowrank_scorer":
            sk = scorer_kwargs or {}
            moe_block._dyn_sc_up = build_layer_scorer(
                experts, "up_proj", m=sk["m"], n=sk["n"], rank=sk["rank"],
                niter=sk.get("niter", 4), compute_device=sk.get("compute_device"),
            )
            moe_block._dyn_sc_gate = (
                build_layer_scorer(
                    experts, "gate_proj", m=sk["m"], n=sk["n"], rank=sk["rank"],
                    niter=sk.get("niter", 4), compute_device=sk.get("compute_device"),
                )
                if sk.get("use_gate", True)
                else None
            )

        # sparse_probe: b-bit copies of this layer's up_proj (and gate_proj) used
        # as the online ranking proxy, read on the token's top-|x| coordinates.
        if criterion == "sparse_probe":
            sk = scorer_kwargs or {}
            keep_L = float(sk.get("rho_input", 0.25))
            B_L = B
            if schedule is not None and layer_idx in schedule:
                ent = schedule[layer_idx]
                keep_L = float(ent["p"])
                B_L = min(int(round(float(ent["rho"]) * K * I)), K * I)
            moe_block._dyn_probe = build_layer_probe(
                experts, bits=int(sk.get("bits", 3)),
                group=int(sk.get("group", 128)),
                use_gate=bool(sk.get("use_gate", True)),
                rho_input=keep_L,
                compute_device=sk.get("compute_device"),
                input_alloc=str(sk.get("input_alloc", "uniform")),
            )
            moe_block._dyn_probe_lam = float(sk.get("lam", 1.0))
            per_layer_B = B_L

        # input_only: nothing to build — the method reads the served gate/up
        # modules directly in the forward, so the per-layer state is just the two
        # knobs. (Contrast sparse_probe, which needs a SparseProbe holding views
        # or quantized copies.)
        if criterion == "input_only":
            sk = scorer_kwargs or {}
            moe_block._dyn_io = InputOnlyCfg(
                rho_input=float(sk.get("rho_input", 0.25)),
                input_alloc=str(sk.get("input_alloc", "uniform")),
            )

        # weight_sparse: per-column magnitude thresholds (one per staircase level)
        # over this layer's served up/gate, plus the optional mean-fix bias.
        if criterion == "weight_sparse":
            sk = scorer_kwargs or {}
            moe_block._dyn_wsparse = build_layer_wsparse(
                experts, levels=sk.get("levels", "0.25x0.45"),
                use_gate=bool(sk.get("use_gate", True)),
                mu=None if wsp_mu is None else wsp_mu[layer_idx].to(block_device),
                input_alloc=str(sk.get("input_alloc", "uniform")),
                compute_device=sk.get("compute_device"),
                alloc_mode=str(sk.get("alloc_mode", "rank")),
                density=sk.get("density"),
                tau_iters=int(sk.get("tau_iters", 16)),
                count_reads=bool(sk.get("count_reads", True)),
            )

        # pubsub: private ranks/gains + public carriers.
        if criterion == "pubsub":
            pa = pubsub_artifact
            moe_block._dyn_pub_pivrank = pa.pivrank_priv[mask_idx].to(block_device)  # (E, I)
            moe_block._dyn_pub_gains = pa.gains_priv[mask_idx].to(block_device)      # (E, I)
            moe_block._dyn_pub_carrier_idx = pa.carrier_idx[mask_idx].to(block_device)  # (r, E)
            moe_block._dyn_pub_carrier_val = pa.carrier_val[mask_idx].to(block_device)  # (r, E)

        moe_block._dyn_B = per_layer_B
        moe_block._dyn_k_min = int(k_min)
        moe_block._dyn_I = int(I)
        moe_block._dyn_criterion = criterion
        moe_block.forward = types.MethodType(dynamic_moe_block_forward, moe_block)

        mask_idx += 1
        n_installed += 1

    if verbose:
        _print(f"[DynamicAlloc] ✅ Installed dynamic forward on {n_installed} MoE blocks")
        if criterion == "sparse_probe" and schedule is not None:
            # Verify the schedule's realized budget rather than trusting the file:
            # the accounting claim is the whole point of the experiment.
            from src.dynamic_active_param.sparse_probe import used_param_fraction
            n_mat = 2 if (scorer_kwargs or {}).get("use_gate", True) else 1
            kept = []
            for layer_idx in range(num_layers):
                blk = _get_moe_block(model, layer_idx)
                if _get_experts(blk) is None:
                    continue
                rho_channel_L = blk._dyn_B / float(K * I)
                kept.append(used_param_fraction(blk._dyn_probe.rho_input,
                                                rho_channel_L, n_mat))
            mk = sum(kept) / max(len(kept), 1)
            _print(
                f"[DynamicAlloc] realized schedule cost over {len(kept)} layers: "
                f"mean kept={mk:.4f} (used-param cut {100 * (1 - mk):.1f}%), "
                f"range [{min(kept):.4f}, {max(kept):.4f}]"
            )

    return model
