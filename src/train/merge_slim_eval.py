import os
import json
import torch

# Base modules
from src.base.models import get_model
from src.base.shared_utils import log_memory_usage, eval_dispatch, test_throughput, _print
from src.base.argparser import parse_args
# Training utilities
from src.train.utils import load_adapter_with_remap

# Pruning modules
from src.prune.generate import generate_masks
from src.prune.apply.slimming import build_real_slim_model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main(args, model, tokenizer):
    
    model.eval()
    torch.set_float32_matmul_precision("high")

    total_params_initial = count_params(model)
    prune_ratio = args.prune_kwargs['prune_ratio']
    
    _print(f"[Step 0] Initial model parameters: {total_params_initial:,}")
    
    if args.resume_path:
        _print(f"\n[Step 1] Load LoRA adapter from: {args.resume_path}")
        
        model = load_adapter_with_remap(model, args.resume_path)
        
        cnt_step = "unknown"
        state_file_pt = os.path.join(args.resume_path, "training_state.pt")
        state_file_json = os.path.join(args.resume_path, "trainer_state.json")

        if os.path.exists(state_file_pt):
            state_payload = torch.load(state_file_pt, map_location=args.device)
            cnt_step = int(state_payload.get("global_step", 0))
        elif os.path.exists(state_file_json):
            try:
                with open(state_file_json, "r", encoding="utf-8") as f:
                    state_payload = json.load(f)
                if isinstance(state_payload, dict):
                    global_step = state_payload.get("global_step")
                    if global_step is None and isinstance(state_payload.get("log_history"), list):
                        for record in reversed(state_payload["log_history"]):
                            if "step" in record:
                                global_step = record["step"]
                                break
                    if global_step is not None:
                        cnt_step = int(global_step)
            except (json.JSONDecodeError, ValueError):
                pass

        _print(f"[Step 1] ✅ LoRA adapter loaded (training step: {cnt_step})")
    else:
        _print(f"\n[Step 1] Skip LoRA loading (no resume_path)")
    
    if args.resume_path:
        model = model.merge_and_unload()
        total_params_after_merge = count_params(model)
        _print(f"[Step 2] ✅ LoRA merged to base model")
        _print(f"  - Merge parameters: {total_params_after_merge:,}")
    else:
        _print(f"\n[Step 2] Skip LoRA merge (no resume_path)")
        total_params_after_merge = total_params_initial
    
    # Reduce-top-k baseline: route each token to fewer experts (e.g. K/2)
    # instead of narrowing experts. Halving top_k halves the activated
    # expert-FFN params per token — the "fewer experts" counterpart to the
    # dynamic "narrower experts" scheme, at the same active budget. No slimming;
    # eval the original weights with a smaller routing top_k.
    reduce_topk = args.prune_kwargs.get("reduce_topk", None)
    if reduce_topk:
        from src.base.shared_utils.safe_isinstance import (
            _get_moe_block,
            _get_experts,
            _get_num_hidden_layers,
            _get_topk,
        )

        orig_topk = _get_topk(model)
        new_topk = int(reduce_topk)
        _print(
            f"\n[Step 3] Reduce-top-k baseline: routing top_k {orig_topk} -> {new_topk} "
            f"(active expert-FFN params scaled by {new_topk / orig_topk:.3f})"
        )
        n_set = 0
        for layer_idx in range(_get_num_hidden_layers(model)):
            moe_block = _get_moe_block(model, layer_idx)
            if _get_experts(moe_block) is None:
                continue
            if hasattr(moe_block, "top_k"):
                moe_block.top_k = new_topk
                n_set += 1
        # keep config in sync (aux-loss / any config.num_experts_per_tok reads)
        if hasattr(model, "config") and hasattr(model.config, "num_experts_per_tok"):
            model.config.num_experts_per_tok = new_topk
        _print(f"[Step 4] ✅ Set top_k={new_topk} on {n_set} MoE blocks (no slimming)")

        _print(f"\n[Step 6] Start evaluation...")
        results = eval_dispatch(args, model, tokenizer, verbose=True)
        _print(f"[Step 6] ✅ Evaluation results: {results}")
        return

    # Dynamic per-token, per-expert active-parameter allocation (masking
    # simulation): distribute a fixed channel budget unevenly across each
    # token's top-K experts. Branches around the static mask-gen / real-slim
    # path entirely; real_slim stays false. See
    # docs/results/dynamic_active_param/plan/plan_initial.md.
    dynamic_alloc_cfg = args.prune_kwargs.get("dynamic_alloc", {}) or {}
    if prune_ratio > 0 and dynamic_alloc_cfg.get("enabled", False):
        from src.dynamic_active_param import build_alloc_artifact, install_dynamic_alloc

        criterion = dynamic_alloc_cfg.get("criterion", "router_prob")
        channel_metric = dynamic_alloc_cfg.get("channel_metric", "activation")
        k_min = dynamic_alloc_cfg.get("k_min", 16)

        # The leverage metric may be absent from an older scores_dir; collect it
        # (and covariances, which we don't use here) on-the-fly, same trigger as
        # the static Nyström path. Skipped entirely when the v2 artifact cache
        # already exists — the masking path only needs the cached leverage
        # *scores* (baked into the artifact), never the covariances, so a warmed
        # cache lets this run without expert_covariances.pth present.
        if channel_metric == "leverage":
            import os as _os
            _artifact_cache = _os.path.join(
                args.scores_dir, f"dynamic_alloc_{channel_metric}_v2.pth"
            )
            if _os.path.exists(_artifact_cache):
                _print(
                    f"[Step 2.5] Using cached dynamic-alloc artifact "
                    f"({_artifact_cache}); skipping leverage/covariance collection"
                )
            else:
                from src.calibration.channel_scoring.collect_covariance import (
                    ensure_leverage_and_covariances,
                )
                lambda_ridge = args.prune_kwargs.get("lambda_ridge", 1.0)
                ensure_leverage_and_covariances(
                    model, tokenizer, args, lambda_ridge=lambda_ridge, verbose=True
                )

        _print(
            f"\n[Step 3] Dynamic allocation "
            f"(criterion={criterion}, channel_metric={channel_metric}, k_min={k_min})"
        )
        artifact = build_alloc_artifact(
            scores_dir=args.scores_dir,
            channel_metric=channel_metric,
            device="cpu",
            verbose=True,
        )
        model = install_dynamic_alloc(
            model,
            artifact,
            prune_ratio=prune_ratio,
            criterion=criterion,
            k_min=k_min,
            verbose=True,
        )
        _print(f"[Step 4] ✅ Dynamic allocation installed (no physical slimming)")

        total_params_after_slim = count_params(model)
        _print(f"[Info] Params unchanged (masking simulation): {total_params_after_slim:,}")

        _print(f"\n[Step 6] Start evaluation...")
        results = eval_dispatch(args, model, tokenizer, verbose=True)
        _print(f"[Step 6] ✅ Evaluation results: {results}")
        return

    # Nyström reconstruction knobs (also drive the leverage metric used for ranking).
    nystrom_reconstruct = args.prune_kwargs.get("nystrom_reconstruct", False)
    lambda_ridge = args.prune_kwargs.get("lambda_ridge", 1.0)
    intra_expert_metric = args.prune_kwargs.get("mask_method_kwargs", {}).get("intra_expert_metric")
    expert_covariances = None

    # Step 2.5 — on-the-fly leverage + covariance collection. The base scoring
    # stage may predate the Nyström feature, so scores_dir can lack the
    # 'leverage' metric (needed for ranking) and expert_covariances.pth (needed
    # for reconstruction). Collect them here on the full un-slimmed model before
    # mask generation, caching to scores_dir for reuse across runs.
    if prune_ratio > 0 and (nystrom_reconstruct or intra_expert_metric == "leverage"):
        from src.calibration.channel_scoring.collect_covariance import (
            ensure_leverage_and_covariances,
        )
        expert_covariances = ensure_leverage_and_covariances(
            model, tokenizer, args, lambda_ridge=lambda_ridge, verbose=True
        )

    if prune_ratio > 0:
        mask_result = generate_masks(
            scores_dir=args.scores_dir,
            mask_dir=args.mask_dir,
            prune_kwargs=args.prune_kwargs,
            device=args.device,
            verbose=True
        )
        _print(f"[Step 3] ✅ Masks generated")
    else:
        _print(f"\n[Step 3] Skip mask generation (prune_ratio=0)")
    
    if prune_ratio > 0:
        # Fallback: if covariances weren't collected on-the-fly above, try loading
        # a previously-saved expert_covariances.pth from scores_dir.
        if nystrom_reconstruct and expert_covariances is None:
            import os as _os
            cov_path = _os.path.join(args.scores_dir, "expert_covariances.pth")
            if _os.path.exists(cov_path):
                _print(f"[Step 3.5] Loading expert covariances from {cov_path}")
                expert_covariances = torch.load(cov_path, map_location="cpu")
            else:
                _print(f"[Warning] nystrom_reconstruct=True but {cov_path} not found, falling back to plain slicing")
                nystrom_reconstruct = False

        model = build_real_slim_model(
            model,
            mask=mask_result,                 # [L, E, I] bool
            shrink_gate=args.shrink_gate,
            add_hooks=False,
            verbose=True,
            nystrom_reconstruct=nystrom_reconstruct,
            expert_covariances=expert_covariances,
            lambda_ridge=lambda_ridge,
        )
        _print(f"[Step 4] ✅ Real slim model built" + (" (Nyström reconstruct)" if nystrom_reconstruct else ""))
        
    total_params_after_slim = count_params(model)
    _print("\n" + "=" * 80)
    _print("Parameter statistics:")
    _print(f"  - Initial parameters: {total_params_initial:,}")
    if args.resume_path:
        _print(f"  - LoRA merge parameters: {total_params_after_merge:,}")
    _print(f"  - Final parameters: {total_params_after_slim:,}")
    if args.resume_path and total_params_after_merge != total_params_initial:
        _print(f"  - LoRA merge parameters change: {total_params_after_merge - total_params_initial:+,} ({100 * (total_params_after_merge - total_params_initial) / total_params_initial:+.2f}%)")
    overall_pruned = total_params_after_merge - total_params_after_slim
    _print(f"  - Pruned pruning parameters: {overall_pruned:,}")
    overall_prune_ratio = 100 * overall_pruned / total_params_after_merge
    _print("-" * 80)
    _print(f"  - Overall pruning ratio (relative to merged): {overall_prune_ratio:.2f}%")
    _print("=" * 80)
  
    if args.test_speed and (args.real_slim or prune_ratio == 0):
        _print(f"\n[Step 5] Test throughput...")
        res = test_throughput(
            model, 
            tokenizer, 
            args.device, 
            batch_size=4, 
            prompt_len=512, 
            gen_len=128
        )
        _print(f"[Step 5] ✅ Throughput = {res}")
        log_memory_usage(tag=f"real_slim: {args.test_speed}")

    _print(f"\n[Step 6] Start evaluation...")
    results = eval_dispatch(args, model, tokenizer, verbose=True)
    _print(f"[Step 6] ✅ Evaluation results: {results}")
    

if __name__ == "__main__":
    args = parse_args()

    model, tokenizer = get_model(args)
    main(args, model, tokenizer)
