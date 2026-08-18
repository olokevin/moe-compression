"""Generate the lm_head eval configs for the plan's test matrix.

Naming follows the repo convention:
``configs/eval/qwen3_30b_a3b_lmhead_<variant>_{hellaswag,mmlu,c4}.yaml``.

Writing these by hand invites drift between the 30-odd YAMLs, and the whole matrix
has to share one protocol (eval_sample_limit, batch size, few-shot) or the numbers
are not comparable against the dense references.

    python scripts/gen_lm_head_configs.py            # 30B arm
    python scripts/gen_lm_head_configs.py --slm      # Qwen3-0.6B arm (phase 4)
"""

import argparse
import os

# task -> (eval_task_names, num_fewshot, eval_sample_limit, batch_size)
TASKS = {
    "hellaswag": ("hellaswag", 0, -1, 16),
    "mmlu": ("mmlu", 5, -1, 8),
    "c4": ("c4", 0, 500, 16),
    "arc_challenge": ("arc_challenge", 0, -1, 16),
}

# variant -> the prune_kwargs.lm_head block. Phase 1 (B1), 2 (B2), 3 (B3), F3.
VARIANTS = {
    # --- Phase 1: B1, the free static prior (the floor) --------------------- #
    "b1s_t4k_tail4":  dict(method="freq_tier", tier_size=4096,  tail_bits=4),
    "b1s_t4k_tail2":  dict(method="freq_tier", tier_size=4096,  tail_bits=2),
    "b1s_t16k_tail2": dict(method="freq_tier", tier_size=16384, tail_bits=2),
    "b1p_t32k":       dict(method="freq_tier", tier_size=32768, tail_bits=0, strict=True),
    "b1p_t8k":        dict(method="freq_tier", tier_size=8192,  tail_bits=0, strict=True),
    "b1a_t4k":        dict(method="freq_tier", tier_size=4096,  tail_bits=16,
                           sparse_activate=True, strict=True),
    "b1a_t16k":       dict(method="freq_tier", tier_size=16384, tail_bits=16,
                           sparse_activate=True, strict=True),
    # the same read budget but with the tiered-softmax tail fallback, so that
    # perplexity stays finite and the cost of the missed tail is legible
    "b1a_t4k_fb":     dict(method="freq_tier", tier_size=4096,  tail_bits=16,
                           sparse_activate=True, strict=False, tail_fallback="uniform"),
    "b1a_t16k_fb":    dict(method="freq_tier", tier_size=16384, tail_bits=16,
                           sparse_activate=True, strict=False, tail_fallback="uniform"),
    # --- Phase 2: B2 ARCHead, storage-matched to B1-s ----------------------- #
    "b2_25":          dict(method="archead", rank=10, correction_rank=6, group=64,
                           residual_bits=4, metric_power=0.75, ridge=1e-3,
                           activation_metric=True),
    "b2_15":          dict(method="archead", rank=10, correction_rank=6, group=64,
                           residual_bits=2, metric_power=0.75, ridge=1e-3,
                           activation_metric=True),
    # ablation: same budget, correction fitted in plain Frobenius error
    "b2_25_nometric": dict(method="archead", rank=10, correction_rank=6, group=64,
                           residual_bits=4, activation_metric=False),
    # --- Phase 3: B3 codebook heads ---------------------------------------- #
    "b3_rvq15":       dict(method="rvq", vq_dim=16, vq_bits=8, vq_stages=3, vq_iters=15),
    "b3_rvq10":       dict(method="rvq", vq_dim=8,  vq_bits=8, vq_stages=1, vq_iters=15),
    "b3_vql":         dict(method="vq_logits", vq_codes=1024, vq_iters=15),
    # --- F3: the honest naive floor ---------------------------------------- #
    "f3_rtn4":        dict(method="rtn", bits=4, group=128),
    "f3_rtn3":        dict(method="rtn", bits=3, group=128),
    # --- F2: the low-rank ladder (the exclusion the plan wants guarded) ----- #
    # rank_frac is a fraction of D, so each name denotes the same ~25/50/75%-of-BF16
    # storage point on every model (rank = frac * D => 16*(V+D)*frac*D/(V*D) bits,
    # which is ~4*frac*16 for V >> D).
    "f2_lr25":        dict(method="lowrank", rank_frac=0.25, whiten=True),
    "f2_lr50":        dict(method="lowrank", rank_frac=0.50, whiten=True),
    "f2_lr75":        dict(method="lowrank", rank_frac=0.75, whiten=True),
    "f2_lr25_plain":  dict(method="lowrank", rank_frac=0.25, whiten=False),
    # --- S1: screen-and-refine -- dynamic reads with a graded tail ---------- #
    # The read fraction is r0/D + N*(D-r0)/(V*D) + D/V, so on a d=1024 head
    # (192, 8192) = 23.8% and on d=2048 (384, 8192) = 22.9%. screen_rank is given as
    # a FRACTION of D for the same reason f2 uses rank_frac: an absolute rank means a
    # different read budget on the two models.
    "s1_r25_n8k":     dict(method="screen_refine", screen_rank_frac=0.1875,
                           cand_size=8192, basis="ceig"),
    "s1_r25_n16k":    dict(method="screen_refine", screen_rank_frac=0.125,
                           cand_size=16384, basis="ceig"),
    "s1_r12_n8k":     dict(method="screen_refine", screen_rank_frac=0.0625,
                           cand_size=8192, basis="ceig"),
    # zero-calibration variant: no rotation, so no eigendecomposition of C
    "s1_r25_n8k_raw": dict(method="screen_refine", screen_rank_frac=0.1875,
                           cand_size=8192, basis="raw"),
    # ablation: static screen subspace (= a low-rank sketch) instead of per-token
    "s1_r25_n8k_static": dict(method="screen_refine", screen_rank_frac=0.1875,
                              cand_size=8192, basis="ceig", screen="static"),
    # ablation: the doc's B1-a candidate set (static frequency tier) with S1's graded
    # tail -- isolates "dynamic candidates" from "graded tail"
    "s1_freq_n8k":    dict(method="screen_refine", screen_rank_frac=0.1875,
                           cand_size=8192, basis="ceig", cand_source="freq"),
    # ablation: S1's candidate set, ungraded (-inf) tail -- must reproduce B1-a's inf
    "s1_r25_n8k_inf": dict(method="screen_refine", screen_rank_frac=0.1875,
                           cand_size=8192, basis="ceig", tail="inf"),
    # gate: r0 = D and N = V must reproduce the dense head exactly
    "s1_identity":    dict(method="screen_refine", screen_rank_frac=1.0,
                           cand_size=151936, basis="ceig"),
}

# Phase 5 -- the best head method composed with the repo's -73% expert config.
COMPOSED_EXPERT = {
    "prune_ratio": 0.733,
    "dynamic_alloc": {
        "enabled": True,
        "criterion": "sparse_probe",
        "probe": {"bits": 16, "rho_input": 0.25, "rho_channel": 0.5,
                  "use_gate": True, "input_alloc": "router"},
    },
}

HEADER = """\
# AUTO-GENERATED by scripts/gen_lm_head_configs.py -- edit the generator, not this file.
# {desc}
#
# lm_head baseline: {variant}
# Dense references (Qwen3-30B-A3B-Thinking-2507): HellaSwag 78.56, MMLU 80.91
# (stderr ~0.41-0.45 pt). Pre-registered success bar: HellaSwag >= 78.1 and
# MMLU >= 80.5 at <= INT4-equivalent head storage.
"""


def emit(path, model, task, variant, cfg, calib_dir, composed=False, slm=False):
    tname, fewshot, limit, bs = TASKS[task]
    lines = [HEADER.format(
        desc=f"{task} eval of the {variant} lm_head baseline",
        variant=variant,
    )]
    lines.append("test_only: true\n")
    lines.append(f'model_name_or_path: "{model}"')
    lines.append("load_in_4bit: false")
    lines.append("load_in_8bit: false")
    lines.append("trust_remote_code: true\n")
    lines.append("test_speed: false")
    lines.append("real_slim: false")
    lines.append("shrink_gate: false\n")
    lines.append('device: "cuda"')
    lines.append('dtype: "bf16"')
    lines.append("use_wandb: true")
    proj = "yequan26-a100-qwen3-0-6b-eval" if slm else "yequan26-a100-qwen3-30b-eval"
    lines.append(f'wandb_project: "{proj}"')
    lines.append(f'wandb_name: "lmhead_{variant}_{task}"\n')
    lines.append("num_workers: 0\n")
    lines.append(f"eval_sample_limit: {limit}")
    lines.append('eval_split: "test"')
    lines.append("per_device_eval_batch_size: 4")
    lines.append(f"batch_size: {bs}")
    lines.append(f"num_fewshot: {fewshot}\n")

    lines.append("prune_kwargs:")
    if composed:
        lines.append(f"  prune_ratio: {COMPOSED_EXPERT['prune_ratio']}")
        d = COMPOSED_EXPERT["dynamic_alloc"]
        lines.append("  dynamic_alloc:")
        lines.append("    enabled: true")
        lines.append(f'    criterion: "{d["criterion"]}"')
        lines.append("    probe:")
        for k, v in d["probe"].items():
            lines.append(f"      {k}: {_yaml(v)}")
    else:
        lines.append("  prune_ratio: 0.0            # head-only arm: MoE path untouched")
    lines.append("  lm_head:")
    lines.append("    enabled: true")
    for k, v in cfg.items():
        lines.append(f"    {k}: {_yaml(v)}")
    lines.append(f'    calib_dir: "{calib_dir}"')
    lines.append('    compute_device: "cpu"   # D x D eigh/SVD off the sharded GPUs')
    lines.append("    diagnostics: true")
    # Keep the 30B activation-metric collection cheap: ARCHead's own sensitivity
    # sweep uses 4k-64k calibration tokens, so 131k is already generous.
    lines.append("    calib_kwargs:")
    lines.append("      unigram_min_tokens: 5000000")
    lines.append(f"      sigma_batches: {16 if slm else 32}")
    lines.append(f"      sigma_batch_size: {16 if slm else 8}")
    lines.append("      sigma_seq_len: 512")
    lines.append("")
    tag = f"lmhead_{variant}" + ("_composed" if composed else "")
    lines.append(f'eval_task_names: "{tname}"')
    lines.append(f"output_dir: ./results_eval/{tag}_{task}")
    lines.append('scores_dir: ""')
    lines.append("")
    lines.append('attn_implementation: "sdpa"')
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _yaml(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slm", action="store_true", help="emit the Qwen3-0.6B (phase 4) arm")
    ap.add_argument("--outdir", default="configs/eval")
    a = ap.parse_args()

    if a.slm:
        model = "Qwen/Qwen3-0.6B"
        prefix = "qwen3_0_6b"
        calib = "./calib/lm_head_qwen3_0_6b"
        variants = ["b1s_t4k_tail4", "b1a_t4k", "b1a_t4k_fb", "b2_25", "f3_rtn4",
                    "b3_rvq15", "b3_vql"]
        variants += ["f2_lr25", "f2_lr50", "f2_lr75", "f2_lr25_plain"]
    else:
        model = "Qwen/Qwen3-30B-A3B-Thinking-2507"
        prefix = "qwen3_30b_a3b"
        calib = "./calib/lm_head_qwen3_30b_a3b"
        variants = list(VARIANTS)

    os.makedirs(a.outdir, exist_ok=True)
    n = 0
    for v in variants:
        for task in TASKS:
            p = os.path.join(a.outdir, f"{prefix}_lmhead_{v}_{task}.yaml")
            emit(p, model, task, v, VARIANTS[v], calib, slm=a.slm)
            n += 1
    # Phase 5: composition with the -73% expert config (30B only)
    if not a.slm:
        for v in ("b1s_t4k_tail4", "b2_25"):
            for task in ("hellaswag", "c4"):
                p = os.path.join(a.outdir, f"{prefix}_lmhead_{v}_composed_{task}.yaml")
                emit(p, model, task, v, VARIANTS[v], calib, composed=True)
                n += 1
    print(f"wrote {n} configs to {a.outdir}/")


if __name__ == "__main__":
    main()
