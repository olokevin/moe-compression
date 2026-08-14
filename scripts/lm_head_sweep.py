"""Run many lm_head treatments against lm-eval on ONE model load.

The 30B arm is the dominant expense in the plan's budget, and ~80% of a per-config
``merge_slim_eval.py`` invocation is spent loading and sharding a 61 GB checkpoint.
This driver loads once, collects the calibration artifacts once, and then for each
variant: install -> evaluate -> restore the dense head. Same numerics as the
per-config path (which stays the canonical entry point), a fraction of the wall
clock.

    python scripts/lm_head_sweep.py --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
        --calib-dir ./calib/lm_head_qwen3_30b_a3b \
        --tasks hellaswag c4 --variants dense f3_rtn4 b1s_t4k_tail4 b2_25 b1a_t4k \
        --out ./results_eval/lm_head_sweep_30b.json
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

import torch

from src.base.shared_utils import _print
from src.lm_head import unbind_head_forward
from src.lm_head.accounting import count_active_params, head_cost
from src.lm_head.calib import ensure_sigma, ensure_unigram, get_lm_head

# Reuse the single source of truth for the variant definitions.
from scripts.gen_lm_head_configs import VARIANTS

# task -> (num_fewshot, eval_sample_limit, batch_size)
TASK_PROTOCOL = {
    "hellaswag": (0, -1, 16),
    "mmlu": (5, -1, 8),
    "c4": (0, 500, 16),
    "winogrande": (0, -1, 16),
    "arc_easy": (0, -1, 16),
}


def load_model(model_id, shard=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kw = dict(dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
    if shard:
        kw["device_map"] = "auto"
    m = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if not shard:
        m = m.to("cuda" if torch.cuda.is_available() else "cpu")
    m.eval()
    return m, tok


def make_args(model_id, task, out_dir, limit_override=None):
    fewshot, limit, bs = TASK_PROTOCOL[task]
    if limit_override is not None:
        limit = limit_override
    return SimpleNamespace(
        model_name_or_path=model_id, eval_task_names=task, num_fewshot=fewshot,
        eval_sample_limit=limit, batch_size=bs, max_seq_length=2048,
        output_dir=out_dir, use_wandb=False, device="cuda",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--calib-dir", default="./calib/lm_head_qwen3_30b_a3b")
    # nargs="*" so `--tasks` with no values runs install-only: builds every head and
    # records the cheap top-1-agreement / KL / discarded-mass diagnostics without
    # paying for an lm-eval pass.
    ap.add_argument("--tasks", nargs="*", default=["hellaswag", "c4"])
    ap.add_argument("--variants", nargs="+", default=["dense", "f3_rtn4", "b1s_t4k_tail4",
                                                     "b2_25", "b1a_t4k"])
    ap.add_argument("--out", default="./results_eval/lm_head_sweep.json")
    ap.add_argument("--sigma-batches", type=int, default=32)
    ap.add_argument("--sigma-batch-size", type=int, default=8)
    ap.add_argument("--no-shard", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="override eval_sample_limit for every task (smoke tests)")
    ap.add_argument("--bs", nargs="*", default=[], metavar="TASK=N",
                    help="override a task's batch size, e.g. --bs c4=2 hellaswag=8. "
                         "Both metrics are batch-invariant (loglikelihood is computed "
                         "per request), so this only trades speed for the peak logits "
                         "tensor -- which is what OOMs a 30B on 2 GPUs at c4's "
                         "2048-token windows.")
    a = ap.parse_args()

    for spec in a.bs:
        task, _, n = spec.partition("=")
        if task not in TASK_PROTOCOL:
            raise SystemExit(f"--bs: unknown task {task!r}")
        fs, lim, _ = TASK_PROTOCOL[task]
        TASK_PROTOCOL[task] = (fs, lim, int(n))
        _print(f"[sweep] batch size for {task} -> {int(n)}")

    os.makedirs(a.calib_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    from eval.lm_harness.eval import eval_fn

    _print(f"[sweep] loading {a.model} (shard={not a.no_shard}) ...")
    model, tok = load_model(a.model, shard=not a.no_shard)
    head = get_lm_head(model)
    V, D = head.weight.shape

    # Untie BEFORE counting. Every treatment below operates on the untied model, so
    # that is the denominator the savings must be measured against; counting first
    # would silently use the tied total (0.596B vs 0.752B on Qwen3-0.6B) and understate
    # every Delta-active by ~21%. No-op on the 30B, which ships untied.
    from src.lm_head.install import _untie_if_needed
    was_tied, untie_cost = _untie_if_needed(model, head)

    ctx = count_active_params(model)
    _print(
        f"[sweep] head {V}x{D} = {V * D / 1e6:.1f}M params; total "
        f"{ctx.total_params / 1e9:.3f}B, active {ctx.active_params / 1e9:.3f}B "
        f"-> head is {100 * ctx.head_params / ctx.active_params:.2f}% of active"
        + (f" (untied first, +{untie_cost / 1e6:.1f}M params)" if was_tied else "")
    )

    calib_cfg = {"calib_kwargs": {
        "sigma_batches": a.sigma_batches, "sigma_batch_size": a.sigma_batch_size,
    }}
    counts = ensure_unigram(tok, calib_cfg, a.calib_dir, V)
    need_sigma = any(VARIANTS[v]["method"] in ("archead", "rvq", "vq_logits")
                     for v in a.variants if v in VARIANTS)
    C = H = None
    if need_sigma:
        C, H = ensure_sigma(model, tok, calib_cfg, a.calib_dir)

    W0 = head.weight.detach().clone()
    rows = []

    def flush():
        with open(a.out, "w") as f:
            json.dump({
                "model": a.model, "V": V, "D": D, "was_tied": was_tied,
                "untie_cost_params": untie_cost,
                "total_params": ctx.total_params, "active_params": ctx.active_params,
                "active_params_pruned": ctx.active_params_pruned,
                "tasks": a.tasks, "rows": rows,
            }, f, indent=2)

    for variant in a.variants:
        _print("\n" + "=" * 78)
        _print(f"[sweep] variant = {variant}")
        _print("=" * 78)
        row = {"variant": variant}
        t0 = time.time()

        if variant == "dense":
            row["storage_frac"] = 1.0
            row["read_frac"] = 1.0
            row["used_head_params_M"] = V * D / 1e6
            row["delta_active_pct"] = 0.0
        else:
            cfg = dict(VARIANTS[variant])
            cfg.update({
                "enabled": True, "calib_dir": a.calib_dir, "compute_device": "cpu",
                "diagnostics": True, "calib_kwargs": calib_cfg["calib_kwargs"],
            })
            from src.lm_head import install_lm_head
            install_lm_head(model, cfg, tokenizer=tok, verbose=True)
            rep = model._lm_head_report
            for k in ("storage_frac_of_bf16", "read_frac_of_bf16", "bits_per_weight",
                      "top1_agreement", "kl_vs_dense", "kl_in_tier", "logit_mse",
                      "dense_mass_outside_tier", "rel_metric_err", "rel_fro_err",
                      "calib_head_mass", "calib_tail_mass", "tier_size", "n_used_codes"):
                if k in rep:
                    row[k] = rep[k]
            row["storage_frac"] = rep["storage_frac_of_bf16"]
            row["read_frac"] = rep["read_frac_of_bf16"]
            row["used_head_params_M"] = rep["used_head_params_bf16eq"] / 1e6
            row["delta_active_pct"] = 100 * rep["delta_active_used"]
            if "delta_active_pruned_used" in rep:
                row["delta_active_pruned_pct"] = 100 * rep["delta_active_pruned_used"]
        row["build_secs"] = round(time.time() - t0, 1)
        # Append BEFORE evaluating so each flush() captures the variant in progress.
        # Otherwise a job killed mid-task loses that whole variant from the JSON.
        rows.append(row)
        flush()

        for task in a.tasks:
            _print(f"\n[sweep] {variant} -> {task}")
            t1 = time.time()
            out_dir = os.path.join("./results_eval", f"sweep_{variant}")
            try:
                res = eval_fn(make_args(a.model, task, out_dir, a.limit), model, tok,
                              [task], verbose=True)
                row[task] = {k: v for k, v in res.items()
                             if k in (task, f"{task}_stderr") or k == task}
                row[f"{task}_raw"] = res.get(task, res)
            except Exception as e:  # a broken variant must not kill the sweep
                _print(f"[sweep] ⚠️  {variant}/{task} FAILED: {type(e).__name__}: {e}")
                row[f"{task}_error"] = f"{type(e).__name__}: {e}"
            row[f"{task}_secs"] = round(time.time() - t1, 1)
            flush()

        # tier hit-rate over the eval stream (only meaningful for masked variants)
        from src.lm_head import lm_head_eval_stats
        st = lm_head_eval_stats(model)
        if st:
            row["argmax_in_tier"] = st["argmax_in_tier"]
            row["eval_tokens"] = st["eval_tokens"]
            _print(f"[sweep] tier hit-rate over the eval stream: "
                   f"{100 * st['argmax_in_tier']:.2f}% of argmaxes were in-tier")
        flush()
        # restore the dense head for the next variant
        head.weight.data.copy_(W0)
        unbind_head_forward(head)
        if hasattr(model, "_lm_head_module"):
            del model._lm_head_module
        torch.cuda.empty_cache()

    _print(f"\n[sweep] wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
