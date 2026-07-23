# Plan: Dynamic Per-Token, Per-Expert Active-Parameter Allocation

## Context

The existing pipeline prunes each expert's intermediate dimension **statically** — every
expert (or every token routed to it) keeps the *same* fixed set of channels, chosen offline
and physically removed at eval time (`build_real_slim_model`). This request adds a
**dynamic, token-adaptive** scheme: for every token, distribute a fixed channel budget
*unevenly across its top-K experts*, giving more channels to the experts that matter more
for that token, while keeping the total activated expert-FFN params at a preset budget
(reduce 33% ⇒ keep ρ = 0.67 of channels).

Two orthogonal decisions, exactly as framed:

1. **How much** budget each expert gets for a token — either by **router probability**
   (the softmax routing weight) or by **router contribution**
   (`expert_out_token_contrib`, precomputed by the attribution-guided scorer).
2. **Which** channels to keep once an expert's budget is known — rank channels by either
   the **attribution channel score** (`activation`) or the **Nyström ridge-leverage score**
   (`leverage`). For leverage we only use the *score* to pick columns (up/gate) / rows
   (down); we **do not** apply the Nyström down_proj weight correction.

All ranking statistics are already precomputed and saved by the scoring stage — nothing new
must be collected. This is realized as **masking simulation** (fake pruning: zero the
intermediate channels beyond a token's budget), consistent with the repo's convention, so it
measures accuracy at the target budget without needing variable-width matmuls.

Decisions locked with the user:
- Budget denominator = **expert-FFN channels only** (keep 67% of `K·I` per token).
- Realization = **masking simulation** (no real speedup; exact accuracy at budget).
- Non-leverage channel ranking metric = **`activation`** (repo default).

## The math

Per MoE layer ℓ: `I` = `moe_intermediate_size`, `E` experts, `K` = `num_experts_per_tok`.

**Precomputed offline (from `scores_dir/expert_scores.pth`):**
- Channel rank `r_{ℓ,e,c} ∈ {0..I-1}`: rank of channel `c` in expert `e` sorted by
  *descending* channel score `s` (`s = activation` or `s = leverage`, both shape `(E,I)`
  per layer). Rank 0 = most important.
- Expert contribution `a_{ℓ,e} = expert_out_token_contrib[ℓ][e]` (shape `(E,)`), clamped
  non-negative.

**Per-token budget.** Baseline active channels per token = `K·I`. Keep fraction ρ (=0.67):
```
B = round(ρ · K · I)               # total kept channels per token, across its K experts
```

**Allocation weight** `w_{t,e}` for the K experts selected by token t (`e ∈ E_t`):
- Router-probability criterion: `w_{t,e} = p_{t,e}`, the (norm_topk_prob-normalized)
  softmax routing weight — already sums to 1 over `E_t`.
- Contribution criterion: `w_{t,e} = a_{ℓ,e} / Σ_{e'∈E_t} a_{ℓ,e'}`.
- Uniform baseline: `w_{t,e} = 1/K`.

**Per-expert channel budget** with floor `k_min` and cap `I`, conserving total = `B`:
```
raw_{t,e}  = w_{t,e} · B
k_{t,e}    = clip(floor(raw_{t,e}), k_min, I)
deficit    = B − Σ_e k_{t,e}
```
Distribute the remaining `deficit` by **largest-remainder water-filling**: repeatedly +1 to
the expert with the largest fractional remainder `raw − floor` among experts below cap `I`
(or, if `deficit < 0`, −1 from the smallest-remainder expert above `k_min`) until
`Σ_e k_{t,e} = B`. This guarantees the budget is met exactly for every token.

**Keep mask** for token t, expert e, channel c:
```
m_{t,e,c} = 1[ r_{ℓ,e,c} < k_{t,e} ]      # keep the top k_{t,e} channels by score
```

**Expert output** (SwiGLU, masking form):
```
y_{t,e} = down_proj( (act(gate_proj(x_t)) ⊙ up_proj(x_t)) ⊙ m_{t,e} )
```
Note: because this is masking, `down_proj` is applied to the full-width (zeroed) intermediate
with the *original* weights — i.e. no Nyström correction — which is exactly the requested
"skip the transformation on down_proj". Leverage vs activation only changes which channels
`m` keeps.

**Param-budget check:** kept channels per token = `Σ_e k_{t,e} = B = 0.67·K·I`, and expert
FFN params scale linearly in kept channels ⇒ activated expert-FFN params reduced by exactly
33% per token.

## Implementation — `src/dynamic_active_param/`

New self-contained package (test-driven). Reuses existing helpers:
`dict_to_tensor` (`src/base/shared_utils/dict_to_tensor.py`), the MoE getters in
`src/base/shared_utils/safe_isinstance.py` (`_get_moe_block`, `_get_experts`,
`_get_moe_intermediate_size`, `_get_num_hidden_layers`, `_get_num_experts`, `_get_topk`),
and the layer→MoE-index mapping pattern from
`src/prune/apply/masking/expert/fake_prune_wrapper.py:127-134`.

- **`precompute.py`** — `build_alloc_artifact(scores_dir, channel_metric, device)`:
  loads `expert_scores.pth`, stacks the chosen metric via `dict_to_tensor` → `(L,E,I)`,
  computes `channel_rank` per `(ℓ,e)` = `argsort(argsort(score, descending))` → int `(L,E,I)`;
  stacks `expert_out_token_contrib` → `(L,E)` (clamped ≥0). Returns a small dataclass holding
  both tensors + `L,E,I` and the metric name. Optionally `torch.save`s to
  `scores_dir/dynamic_alloc_<metric>.pth` for reuse.

- **`allocate.py`** — `allocate_budgets(routing_weights, selected_experts, contrib, B, k_min,
  criterion) -> LongTensor[T,K]`: the pure, fully-unit-tested core implementing the math
  above (largest-remainder water-filling, vectorized over tokens). No model/torch-module
  dependency — trivially testable with hand-checkable small tensors.

- **`block.py`** — `dynamic_moe_block_forward(self, hidden_states)`: a drop-in replacement for
  `Qwen3MoeSparseMoeBlock.forward` / `Qwen2MoeSparseMoeBlock.forward`. Same routing/top-k as
  upstream, but per hit expert it builds a `[n_e, I]` keep-mask
  (`rank_row[None,:] < k_col[:,None]`) and applies it to the SwiGLU intermediate before
  `down_proj`. Reads `self._dyn_ranks[e]`, `self._dyn_contrib`, `self._dyn_B`,
  `self._dyn_k_min`, `self._dyn_criterion` attached at install. Returns
  `(final_hidden_states, router_logits)` exactly like upstream (preserves aux-loss path).

- **`install.py`** — `install_dynamic_alloc(model, artifact, prune_ratio, criterion, k_min)`:
  walks layers (same MoE-index mapping as the wrapper), computes `K` via `_get_topk`,
  `B = round((1-prune_ratio)·K·I)`, and binds `dynamic_moe_block_forward` onto each
  `layer.mlp` via `types.MethodType`, attaching per-layer rank/contrib tensors moved to the
  block's device (handles `device_map='auto'` sharding).

### Tests — `src/dynamic_active_param/tests/` (write first)

`pytest`-style (repo has no runner, so tests are runnable via `python -m pytest`):
- `test_allocate.py`: budget conservation `Σk == B` across random weights; bounds `k_min ≤ k ≤ I`;
  monotonicity (larger weight ⇒ ≥ budget); uniform criterion ⇒ near-even split; ρ=1.0 ⇒ all
  channels; degenerate contrib (all zero) ⇒ uniform fallback.
- `test_precompute.py`: rank is a valid permutation per `(ℓ,e)`; top-ranked channel = argmax score.
- `test_block.py`: build a tiny synthetic module mimicking `Qwen3MoeSparseMoeBlock`
  (small `H,I,E,K`); assert (a) ρ=1.0 dynamic forward ≈ original forward (allclose), and
  (b) at ρ<1 the number of nonzero intermediate channels per token equals its allocated `B`.

## Wiring into eval — `src/train/merge_slim_eval.py`

Add a `dynamic_alloc` sub-dict under `prune_kwargs`. When
`prune_kwargs.dynamic_alloc.enabled` is true, **branch before** the `generate_masks` /
`build_real_slim_model` block (`merge_slim_eval.py:91-126`): build the artifact from
`scores_dir` + `channel_metric`, call `install_dynamic_alloc(...)`, skip static
mask-gen/slimming, then fall through to `eval_dispatch` unchanged. `real_slim` stays false.
No change to `E2EArguments` needed — `prune_kwargs` is a free-form `Dict`.

New config block:
```yaml
prune_kwargs:
  prune_ratio: 0.33
  dynamic_alloc:
    enabled: true
    criterion: "router_prob"      # router_prob | contribution | uniform
    channel_metric: "activation"  # activation | leverage
    k_min: 16
```

### New configs — `configs/eval/`
Copy `qwen3_30b_a3b_oneshot_hellaswag.yaml`, set `real_slim: false`, `prune_ratio: 0.33`, and
the `dynamic_alloc` block. One file per combination:
- `qwen3_30b_a3b_dynamic_prob_act_hellaswag.yaml`  (router_prob × activation)
- `qwen3_30b_a3b_dynamic_prob_lev_hellaswag.yaml`  (router_prob × leverage)
- `qwen3_30b_a3b_dynamic_contrib_act_hellaswag.yaml` (contribution × activation)
- `qwen3_30b_a3b_dynamic_contrib_lev_hellaswag.yaml` (contribution × leverage)
- `qwen3_30b_a3b_dynamic_uniform_act_hellaswag.yaml` (uniform × activation — dynamic-path baseline)

All point `scores_dir` at the existing
`.../Qwen_Qwen3-30B-A3B-Thinking-2507/c4/scores`.

## Docs / results
- `docs/results/dynamic_active_param/q3_30b.md` — fill the HellaSwag results table after the
  A100 runs: rows = {dense baseline, static oneshot-33 activation, static nystrom-33 leverage,
  5 dynamic configs}, columns = criterion / channel_metric / acc / acc_norm.

## Verification / running

1. **Unit tests (local, fast):** `python -m pytest src/dynamic_active_param/tests/ -q`.
   Must pass before any GPU run — this is the correctness gate for the allocation math and the
   ρ=1.0-equals-baseline invariant.
2. **A100 HellaSwag runs** (via the `launch-on-a100` skill): rsync code up, then for each
   config `CONFIG=configs/eval/qwen3_30b_a3b_dynamic_*.yaml bash scripts/eval.sh`. Also run the
   dense baseline (`qwen3_30b_a3b_baseline_hellaswag.yaml`) and the two static 33% baselines
   for comparison. Results land in each config's `output_dir/lm_harness/hellaswag-fs0-results.json`.
3. Pull results back, tabulate `acc`/`acc_norm` into
   `docs/results/dynamic_active_param/q3_30b.md`, and report.

## Open notes
- `expert_out_token_contrib` is a per-expert *scalar* (calibration-averaged), so the
  contribution criterion gives a fixed per-expert weight independent of the token; the
  per-token variation comes only through *which* experts a token selects. Router-probability
  is the truly per-token criterion. Both are covered; the results table will show which wins.
- If `scores_dir` lacks the `leverage` metric, reuse
  `src/calibration/channel_scoring/collect_covariance.ensure_leverage_and_covariances` (same
  trigger already used at `merge_slim_eval.py:83`) to populate it before building the artifact.
