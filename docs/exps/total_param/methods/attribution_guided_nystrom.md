# Ridge-Leverage Ranking + Nyström Reconstruction — Qwen3-30B-A3B (one-shot)

**Method.** Rank expert-FFN intermediate channels by the **ridge leverage score**
`diag((C+λI)⁻¹C)` (`C` = per-expert `down_proj` input covariance `zᵀz/N`, `λ=1.0`),
allocate per-expert budget via the existing `attr_coverage` planner, then physically
slim each expert with **Nyström closed-form `down_proj` reconstruction**
`W_downₙₑw = (SᵀCS)⁻¹(SᵀC)W_downᵀ` (absorbs pruned-channel mass into survivors)
instead of plain column slicing.

- **Model:** `Qwen/Qwen3-30B-A3B-Thinking-2507` (bf16), **no fine-tuning** (`test_only`, one-shot).
- **Config:** `intra_expert_metric: leverage`, `intra_layer_method: attr_coverage`,
  `inter_layer_method: loss_coverage`, `nystrom_reconstruct: true`, `lambda_ridge: 1.0`,
  `shrink_gate: true`, `min_per_expert: 16`.
- **Covariance/leverage:** collected **on-the-fly at eval time** from c4 (128 batches × bs16,
  seq 512); leverage appended into `expert_scores.pth`, full covariances held in memory only.
- **Hardware:** A100-New, 4×40GB A100 per run (`FORCE_DEVICE_MAP_AUTO=1 PER_GPU_MEM=36GiB`,
  `attn_implementation: sdpa`). Run date: 2026-07-10.

## Results

| Prune ratio   | Expert-FFN pruned | Overall params | HellaSwag acc_norm | HellaSwag acc | MMLU (5-shot, full) acc   |
| ------------- | ----------------- | -------------- | ------------------ | ------------- | ------------------------- |
| **25%** | 25.0006%          | −23.74%       | **78.45%**   | 59.49%        | **76.04%** (±0.34) |
| **33%** | 33.0001%          | −31.33%       | **78.40%**   | 59.18%        | **73.00%** (±0.35) |

- HellaSwag: full validation (10 042 items), `num_fewshot=0`.
- MMLU: full 14 042 questions × 57 subtasks, `num_fewshot=5`.
- "Expert-FFN pruned" = fraction of intermediate channels removed; "Overall params" is
  relative to the merged model (attention + router kept dense).

### MMLU by category

| Category        | 25%    | 33%    |
| --------------- | ------ | ------ |
| Humanities      | 70.50% | 66.23% |
| Social sciences | 85.80% | 84.47% |
| STEM            | 69.81% | 65.87% |
| Other           | 81.11% | 79.14% |

## Comparison to baselines (25% one-shot, from prior runs)

| Method @ 25%                                         | HellaSwag acc_norm | MMLU   |
| ---------------------------------------------------- | ------------------ | ------ |
| Original (unpruned)                                  | 78.56%             | —     |
| Activation-magnitude ranking, plain slicing          | 78.23%             | 76.28% |
| **Leverage ranking + Nyström reconstruction** | **78.45%**   | 76.04% |

## Uniform-allocation baseline (33% one-shot)

To isolate the contribution of **attribution-guided allocation**, we ran an
otherwise-identical Nyström configuration (leverage ranking + closed-form
`down_proj` reconstruction) but with **uniform budget everywhere**:
`inter_layer_method: uniform` (every layer keeps the same fraction) and
`intra_layer_method: uniform` (every expert keeps the same fraction), at a flat
**33% prune ratio for all layers and all experts**. Same base model, same
covariance recipe (c4, 128 batches × bs16, seq 512), no fine-tuning.

| Method @ 33%                                                       | Expert-FFN pruned | Overall params | HellaSwag acc_norm | HellaSwag acc | MMLU (5-shot) acc         |
| ------------------------------------------------------------------ | ----------------- | -------------- | ------------------ | ------------- | ------------------------- |
| **Attribution-guided** (`loss_coverage`+`attr_coverage`) | 33.00%            | −31.33%       | **78.40%**   | 59.18%        | **73.00%** (±0.35) |
| **Uniform** (`uniform`+`uniform`)                        | 33.00%            | −31.28%       | 65.10% (±0.48)    | 48.56%        | 27.40% (±0.38)           |

- Configs: `configs/eval/qwen3_30b_a3b_nystrom_uniform33_{hellaswag,mmlu}.yaml`.
  Run date: 2026-07-14. Raw JSONs:
  `run_results/A100-New/.../qwen3_nystrom_uniform33_{hellaswag,mmlu}_*/lm_harness/`.
- MMLU by category (uniform 33%): humanities 25.87%, social sciences 28.66%,
  STEM 27.18%, other 28.71%.

**The uniform baseline collapses.** At the same 33% storage cut and the same
leverage+Nyström machinery, uniform allocation loses **13.3 pts** on HellaSwag
(78.40% → 65.10%) and falls to **near-random on MMLU** (73.00% → 27.40%; MMLU
chance ≈ 25%). Attribution-guided planning — which concentrates budget in the
loss-sensitive layers and high-coverage experts and strips the least-routed
capacity — is what makes 33% one-shot pruning viable; pruning every expert and
every layer by the same fraction destroys the model.

## Takeaways

- **HellaSwag:** leverage + Nyström (78.45%) **beats the activation baseline** (78.23%)
  and nearly matches the unpruned model (78.56%) — and is remarkably robust to heavier
  pruning: **33% barely degrades it (78.40%)**.
- **MMLU @ 25%:** essentially tied with the activation baseline (76.04% vs 76.28%,
  within ~1 stderr). 5-shot MMLU leans on attention/reasoning context that expert-FFN
  pruning touches less, so both methods hold up.
- **MMLU @ 33%:** the accuracy/compression tradeoff appears on the harder 5-shot task
  (73.00%, ~3pt drop) while HellaSwag stays flat — the clean story is that leverage +
  reconstruction lets you push to 33% prune with almost no HellaSwag loss at a modest
  MMLU cost, **with zero fine-tuning**.

## Reproduce

```bash
# 25%
python src/train/merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_nystrom_25p_hellaswag.yaml
python src/train/merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_nystrom_25p_mmlu.yaml
# 33%
python src/train/merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_nystrom_33p_hellaswag.yaml
python src/train/merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_nystrom_33p_mmlu.yaml
# Uniform 33% baseline (all layers / all experts pruned by the same fraction)
python src/train/merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_nystrom_uniform33_hellaswag.yaml
python src/train/merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_nystrom_uniform33_mmlu.yaml
```

The base `scores_dir` may predate the Nyström feature (no `leverage` metric, no
`expert_covariances.pth`). `merge_slim_eval.py` now collects both on-the-fly
before mask generation (a single hooked c4 forward sweep, ~17 min on 4×40GB
A100) and caches them into `scores_dir` for reuse
(`src/calibration/channel_scoring/collect_covariance.py`).

Prefix with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_DEVICE_MAP_AUTO=1 PER_GPU_MEM=36GiB`
on 40GB A100s (4 GPUs — the on-the-fly covariance pass runs on the full un-slimmed model).

Raw result JSONs: `run_results/A100-New/results_eval/qwen3_nystrom_{25p,33p}_{hellaswag,mmlu}_*/lm_harness/`.

## Active-parameter analysis (25% prune, C4 routing)

**Total params drop 25%, but *active* params barely move.** Attribution-guided
pruning is non-uniform per expert (`attr_coverage` budget + `loss_coverage`
inter-layer), and MoE routes each token to only `top_k=8` of 128 experts — so the
active-param fraction is data-dependent, not the flat 0.75 of storage. We captured
the router's real top-8 choices on C4 and weighted by each expert's kept-channel
count from the mask.

| Scope                       | Avg              | Min    | Max    | Std    |
| --------------------------- | ---------------- | ------ | ------ | ------ |
| **Expert-FFN only**   | **0.9735** | 0.7122 | 1.0000 | 0.0296 |
| **Full-model active** | **0.9857** | 0.8444 | 1.0000 | 0.0160 |

- Ratio = active(compressed) / active(uncompressed), per token. **21 494 tokens**
  over 64 C4 samples (seq ≤ 1024).
- Reference: uniform total expert kept-frac = 0.75 (25.0% of expert params removed).
  Fixed non-expert active params (attention + router + embed + norms + lm_head) =
  **1.541 B**; uncompressed active total = **3.353 B**.
- Per-layer budgets range 0.67–0.85 (`layerwise_keep_plan`, 48 layers).

**Takeaways**

- **~25% storage cut → only ~1.4% average active-compute cut** (full-model avg
  0.986). Attribution-guided pruning strips the *least-contributing* channels, which
  live in the *least-routed* experts — so the removed capacity rarely enters a
  token's active top-8. Surviving channels concentrate in the heavily-used experts
  that dominate active FLOPs, hence active retention (0.97) ≫ storage retention (0.75).
- **Strongly data-dependent:** per-token expert-FFN active ratio spans 0.71–1.00. A
  token routing to 8 heavily-pruned experts sees ~29% fewer active FFN params; one
  routing to unpruned experts sees the full model.
- **Practical read:** this checkpoint delivers a real 25% memory/storage reduction
  but should **not** be expected to yield a proportional inference-FLOPs speedup —
  the trade favors accuracy retention (see benchmarks above) over active-compute savings.

*Provenance:* measured from the 25% attribution-guided mask
`outputs/qwen3_30b_a3b_25p_4bit_20260709_102531/checkpoint-80/masks.pth` on the
4-bit base model. Active-param *counts* depend only on per-expert kept budgets
(`attr_coverage` + `loss_coverage`), which are shared with the leverage/Nyström
runs, so this distribution is representative of the 25% attribution-guided family.
Full-model counts use analytic module dimensions (4-bit weight packing makes
`.numel()` unreliable).

```bash
python scripts/active_param_stats.py \
  --mask outputs/qwen3_30b_a3b_25p_4bit_20260709_102531/checkpoint-80/masks.pth \
  --n_samples 64 --max_length 1024
```

Raw log: `run_results/A100-New/run_logs/actparam_0713-230345.log`.
