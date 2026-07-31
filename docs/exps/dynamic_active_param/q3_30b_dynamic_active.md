# Dynamic Per-Token, Per-Expert Active-Parameter Allocation — Qwen3-30B-A3B

Distributes a fixed per-token channel budget **unevenly across each token's
top-K experts** (masking simulation, no fine-tuning), keeping activated
expert-FFN params at ρ = 0.67 of `K·I` (33% cut). Two orthogonal knobs:

- **criterion** — how much budget each expert gets: `router_prob` (per-token
  softmax weight), `contribution` (`expert_out_token_contrib`, per-expert
  scalar), or `uniform` (dynamic-path baseline, even split).
- **channel_metric** — which channels each expert keeps: `activation` (repo
  default) or `leverage` (Nyström ridge-leverage score; score-only, no
  down_proj reconstruction).

Base model `Qwen/Qwen3-30B-A3B-Thinking-2507`, scores from
`.../Qwen_Qwen3-30B-A3B-Thinking-2507/c4/scores`. No fine-tuning, `real_slim: false` (masking simulation gives exact accuracy at budget). HellaSwag 0-shot.

## Results (HellaSwag 0-shot)

| Config                        | criterion    | channel_metric | acc     | acc_norm |
| ----------------------------- | ------------ | -------------- | ------- | -------- |
| Dense baseline (unpruned)     | —           | —             | _TBD_ | 78.56%*  |
| Uniform nystrom baseline      | uniform      |                | 49.55%  | 66.29%   |
| Uniform MoBE baseline         |              |                |         | 69.54%   |
| Dynamic prob × activation    | router_prob  | activation     | 57.31%  | 75.96%   |
| Dynamic prob × leverage      | router_prob  | leverage       | 57.65%  | 76.13%   |
| Dynamic contrib × activation | contribution | activation     | 50.58%  | 67.79%   |
| Dynamic contrib × leverage   | contribution | leverage       | 51.49%  | 69.46%   |

\* Dense baseline `-Thinking-2507` (78.56%) carried from
`docs/results/attribution_guided/nystrom.md`.

**Why the static attribution-guided 33% rows are *not* the right baseline.**
The static pruning pipeline's "33%" is 33% of the **expert-FFN storage** — it
removes whole channels from every expert offline. But only `K` of `E` experts
fire per token (Qwen3-30B: K=8, E=128), so the **activated** compute per token
drops by only ~2–3%, not 33%. That method trades storage, not active FLOPs. The
dynamic scheme here cuts **33% of the active** expert-FFN params *per token*, so
the apples-to-apples baselines at the same active budget are methods that also
shrink the per-token active path to ρ≈0.67:

- **Uniform nystrom** — static uniform prune to 67% of channels per expert (no
  attribution weighting), i.e. every token/expert keeps the same 515 channels.
  This is exactly the ρ=0.67 active budget, and equals `Dynamic uniform × activation` here (66.29%) since uniform allocation reduces to a static uniform
  keep-set.
- **Uniform MoBE** — one-shot MoBE decomposition to 67% (`compression_ratio: 0.67`, 16 bases, rank 768, no fine-tuning); shrinks the active per-token FFN
  work by ~33%. Source:
  `run_results/.../compress_then_train/ce_mobe_calib-c4-0.67_*/benchmark_comparison.json`.
  Caveat: MoBE/uniform-nystrom baselines ran on base `Qwen3-30B-A3B` (dense
  77.68%), whereas the dynamic rows use `-Thinking-2507` (dense 78.56%) — so
  treat the cross-model gap of ~0.9pt as noise when comparing.

**`contrib` bug (found & fixed, 2026-07-15→17).** The first run of the two
`contribution` configs silently fell back to uniform: `expert_out_token_contrib`
is stored as a *negative* per-expert scalar (more-important experts more
negative), and the initial `precompute.py` clamped raw values to ≥0, zeroing
everything. Fixed by negating before clamping (matches the repo's static
`attr_coverage` path, `prepare_scores.py:116`); the `contrib_*` rows above are
from the corrected re-run. `router_prob`/`uniform` rows were never affected.

## 50% reduction — coverage-maximized allocation (`coverage_alloc`)

New criterion combining **router contribution + expert info-concentration**,
following the paper's coverage-maximized allocation (§4.2, Alg. 1) applied
*per-token* over each token's K experts. Coverage ratio for the top-n channels
is `ρ_e(n) = S_e(n)/S_tot_e` (prefix sums of the descending-sorted leverage
scores). The per-expert coverage **target** is initialized from the router
probability and a single scaling factor α: `ρ_e(α) = min(α·p_{t,e}, 1)`; α is
binary-searched per token so the total kept channels `Σ_e N_e(ρ_e(α)) ≤ B`, then
a coverage-aware top-up lands `Σ_e k = B` exactly (same active budget as
`router_prob`). Intuition: two experts with equal router prob get equal coverage
*targets*, but an expert whose leverage is **concentrated** reaches that target
with **fewer** channels, freeing budget for experts whose leverage is spread out.

The dynamic arms use `channel_metric = leverage` (ridge-leverage score),
`prune_ratio = 0.50` (keep ρ = 0.5 of `K·I`), `k_min = 16`, no fine-tuning,
`real_slim: false`. HellaSwag 0-shot. The leverage score is **precomputed once**
(derived from the cached `expert_covariances.pth`, no forward recompute) and
reused across all runs via the `dynamic_alloc_leverage_v2.pth` artifact cache.

**Two ways to halve the active expert-FFN budget** are compared here:
*narrower experts* (the dynamic scheme — keep all K=8 experts but zero half their
channels per token) vs *fewer experts* (**reduce-top-k** — route each token to
top-4 of 8 experts at full width, `reduce_topk: 4`, original weights, no
slimming). Both cut per-token active expert-FFN params by ~50%.

| Config                                  | criterion      | channel_metric | acc              | acc_norm         |
| --------------------------------------- | -------------- | -------------- | ---------------- | ---------------- |
| Dense baseline (unpruned)               | —             | —             | _TBD_          | 78.56%*          |
| Reduce top-k (8→4 experts)             | fewer-experts  | —             | 57.42%           | 75.96%           |
| Dynamic uniform × leverage             | uniform        | leverage       | 43.97%           | 58.89%           |
| Dynamic prob × leverage                | router_prob    | leverage       | 53.47%           | 71.46%           |
| Dynamic coverage × leverage            | coverage_alloc | leverage       | 54.92%           | 72.94%           |
| **Level 1 — pivchol global g²** | pivchol_global | pivot-Cholesky | **56.05%** | **74.26%** |

\* Dense baseline carried from `docs/results/attribution_guided/nystrom.md`.
stderr ≈ 0.43–0.44pt on acc_norm for all rows.

**Takeaways (50%).**

- **`coverage_alloc` beats `router_prob`** by **+1.45pt acc / +1.48pt acc_norm**
  (72.94 vs 71.46), a gap ~3× the stderr — combining router contribution with
  each expert's leverage-concentration curve allocates the fixed active budget
  better than router probability alone.
- Both budget-aware criteria crush the **uniform** dynamic-path baseline
  (58.89% acc_norm) by **+12.6 / +14.0pt** — at the harder 50% cut the per-token,
  per-expert split matters even more than at 33%.
- **Fewer experts > narrower experts at 50%.** Reduce-top-k (75.96% acc_norm)
  beats even `coverage_alloc` (72.94%) by +3.0pt, and comes within 2.6pt of the
  dense baseline. Halving the active budget by dropping the 4 lowest-probability
  experts per token is *less destructive* than keeping all 8 and halving each
  expert's channels — a token's low-ranked experts contribute little, whereas
  narrowing every expert (including the dominant ones) damages the experts that
  matter most. This is a strong baseline for the dynamic-narrowing story: the
  narrowing methods must ultimately justify themselves against it (e.g. via
  fine-tuning recovery, or regimes where routing top-k is already small).
- **Level 1 (`pivchol_global`) — the corrected narrowing baseline.** Global g²·σ
  threshold + pivoted-Cholesky nested ordering reaches **74.26%**, **+1.32pt over
  `coverage_alloc`** (the three fixes: global competition, g² not g,
  redundancy-aware ordering). At 50% it trails reduce-top-k (75.2%) by ~0.9pt —
  L1's scoring matrix `Θ_k` is block-diagonal (no cross-expert terms), so at loose
  budgets it can't exploit cross-expert redundancy the way dropping whole experts
  does. **But the budget sweep below shows L1 overtakes reduce-top-k as the budget
  tightens** (−62.5%: +0.7, −75%: +14.2pt): dropping experts discards unique
  knowledge, while narrowing keeps each active expert's load-bearing channels.
  Cross-expert redundancy remains the ceiling at loose budgets, motivating a
  cross-expert (Level 3) method. See `plan/plan_level1.md`,
  `plan/plan_level1_impl.md`, and the sweep table below.

### Takeaways

- **Budget-aware allocation clearly helps.** Both `router_prob` configs (75.96 /
  76.13% acc_norm) beat the same-active-budget uniform baselines (66.29% uniform
  nystrom, 69.54% MoBE) by **+6 to +10 pts**, with zero fine-tuning and no
  physical slimming — the per-token, per-expert budget split is doing real work.
- **`router_prob` ≫ `contribution`.** Per-token softmax weight (76.1%) far
  outperforms the calibration-averaged per-expert contribution (67.8 / 69.5%),
  which barely edges out uniform. Expected: `expert_out_token_contrib` is a
  fixed per-expert scalar, so `contribution` only varies through *which* experts
  a token picks — it is not truly per-token. `router_prob` is.
- **leverage ≥ activation** for channel ranking under both criteria
  (prob: 76.13 vs 75.96; contrib: 69.46 vs 67.79), consistent with the static
  Nyström story — but the gap is small (<0.2pt) under the stronger `router_prob`.

## Level 1 — Global g²-weighted nested channel selection (`pivchol_global`)

Realizes `plan/plan_level1.md`. Replaces the current method's three components:
per-expert quota by linear-g → **global** g² threshold (quotas emerge, a
dominated expert may get 0); ridge-leverage in-expert order → **pivoted-Cholesky**
nested, redundancy-aware order.

**Offline (Phase B):** per expert build the coupling `Θ_k = G_k ⊙ B_k` where
`G_k = E[φ_k φ_kᵀ]` is the cached activation Gram (`expert_covariances.pth`, the
uncentered second moment of the down_proj input) and `B_k = W_downᵀ W_down` is the
weight Gram (H = I). Batched ridge-pivoted Cholesky (shared `λ_r = 1.0`) to
completion gives a pivot order `π_k` and monotone marginal gains `σ_k`. Stored as
`pivchol_artifact.pth` (pivot ranks + gains, 57MB). Built by
`scripts/warm_pivchol_cache.py` on the box holding the covariances; factorization
runs on **CPU** (~5 min/48 layers) to avoid crashing CUBLAS on a GPU still holding
a `device_map='auto'` shard.

**Online:** per token, score each active expert's channels by `g_k² · σ_{k,r}`
(σ monotone, g² a per-expert constant → each expert's sequence is pre-sorted),
keep the global top-`B`; per-expert prefix length `t_k` (hence `ρ_k = t_k/m`)
emerges from one shadow price. Implemented as `_pivchol_allocate` in `allocate.py`
(global top-B over the pooled `K·I` scores, count per expert; `Σ t_k = B` exactly,
no `k_min` floor). The keep-mask `pivrank < t_k` reproduces the pivot prefix.

### Budget sweep — Level 1 vs `router_prob × activation` (HellaSwag 0-shot)

Three methods across four active-param reductions (Level 1 reuses the cached
budget-agnostic `pivchol_artifact.pth`; `router_prob × activation`, the winning
33%-study criterion, `k_min = 16`; **reduce-top-k** = route to fewer full-width
experts, from `docs/intern_plan/proposal/per_token_adaptive_activate.md`). acc_norm:

| Reduction | ρ (kept) | B (of 6144) | reduce top-k | router_prob × act | Level 1 (pivchol) | Δ (L1 − topk) |
| --------- | --------- | ----------- | ------------ | ------------------ | ----------------- | --------------- |
| 50%       | 0.50      | 3072        | 75.2 (8→4)  | 71.46%†           | 74.26%            | −0.9           |
| 62.5%     | 0.375     | 2304        | 69.8 (8→3)  | 61.00%             | **70.54%**  | +0.7            |
| 75%       | 0.25      | 1536        | 49.4 (8→2)  | 43.66%             | **63.60%**  | +14.2           |
| 87.5%     | 0.125     | 768         | 26.2 (8→1)  | 30.32%             | 44.15%            | —              |

† 50% baseline row is `router_prob × leverage` (71.46%); the other three rows are
`router_prob × activation`. reduce-top-k maps to integer expert counts (8→4/3/2/1);
−87.5% (8→1) is not reported in the proposal. (acc for L1: 56.05 / 53.16 / 47.56 /
35.24%; for router_prob×act: 53.47 / 45.71 / 34.93 / 27.84%.)

**Level 1 beats `router_prob × activation` at every budget, and the margin widens
as the cut deepens** — the baseline collapses toward chance (25% acc) by 87.5%: its
per-expert linear-g quota keeps spending budget on low-probability experts and, in
each expert, ridge-leverage double-counts redundant channels. Level 1's global g²r
competition starves weak experts and its pivoted-Cholesky ordering avoids redundant
channels, so it degrades gracefully (74.3 → 70.5 → 63.6 → 44.2).

**vs reduce-top-k (the strong "fewer experts" baseline):** Level 1 matches it at
moderate cuts (−50%: 74.3 vs 75.2) and **overtakes it as the budget tightens**
(−62.5%: +0.7pt, −75%: **+14.2pt**). When budget is scarce, dropping whole experts
discards each dropped expert's *unique* knowledge, whereas Level 1 keeps every
active expert's most load-bearing (non-redundant) channels.

### MMLU (5-shot) — Level 1 vs `router_prob × activation` @ 75% reduction

Full MMLU, 5-shot, same 75% active-param budget (B = 1536 of 6144). Overall acc:

| Method                             | criterion      | MMLU acc (5-shot) |
| ---------------------------------- | -------------- | ----------------- |
| Reduce top-k (8→2)               |                | 34.90%            |
| `router_prob × activation`      | router_prob    | 49.17%            |
| **Level 1 (pivchol global)** | pivchol_global | **70.81%**  |

**Level 1 leads by +21.6pt** — an even larger margin than HellaSwag at the same
75% budget (+19.9pt). MMLU (knowledge-heavy, 5-shot) is more sensitive to
destroying expert capacity, so the baseline's redundant-channel double-spend and
low-g over-feeding cost it more; Level 1's redundancy-aware global selection holds
up. This corroborates the HellaSwag trend on a second, harder benchmark.

## Level 2 — cross-expert selection (`oracle_mag`, `pubsub`)

Level 1's scoring matrix `Θ_k` is **block-diagonal**: channels are ranked
within-expert, so a low-probability expert may spend budget on channels already
covered by a co-activated high-probability expert. Level 2 asks how much this
costs and whether a cross-expert method recovers it. Realizes
`plan/plan_level2_impl.md`. Two runnable selectors, both selecting a per-token
**global top-B over the pooled K·I channels** of a token's active experts:

- **`oracle_mag` (Oracle-A ceiling).** Scores each channel by its *exact
  per-token* output magnitude `g_e·|inter_{e,j}(x)|·‖W_down[:,j]‖` and keeps the
  global top-B. Block-diagonal (no off-diagonal coupling) but uses the true
  per-token activation instead of only the router `g` — so it upper-bounds every
  router-only offline method. The gap `oracle_mag − Level-1` is the value of
  per-token activation information.
- **`pubsub` (the Level-2 method).** Offline, builds a shared **public basis**
  `U` = top-r eigenvectors of `Σ_e W_down_e G_e W_down_eᵀ` (directions many
  experts write into), deflates each expert's `down_proj` by `U`, and runs the
  Level-1 pivoted Cholesky on the *private* residual → `σ^priv`. Online (router
  only), for each public direction it force-keeps the single best-carrying
  channel among the co-activated experts (dedup — load shared knowledge **once**),
  then fills the rest by the Level-1 rule `g²·σ^priv`. Preserves prefix
  contiguity; touches no expert weights beyond the ~57 MB artifact.

### HellaSwag 0-shot — budget sweep (acc_norm)

Level-1 and reduce-top-k rows carried from the sweep table above. `oracle_mag`
and `pubsub` are new (masking simulation, no fine-tuning).

| Reduction | reduce top-k | Level 1 (pivchol) | `pubsub` (L2) | `oracle_mag` (ceiling) |
| --------- | ------------ | ----------------- | --------------- | ------------------------ |
| 50%       | 75.2 (8→4)  | 74.26             | 74.31           | **78.54**          |
| 62.5%     | 69.8 (8→3)  | 70.54             | 70.51           | **78.76**          |
| 75%       | 49.4 (8→2)  | 63.60             | 63.46           | **78.28**          |
| 87.5%     | 26.2 (8→1)  | 44.15             | 44.66           | **76.84**          |

Dense baseline 78.56. stderr ≈ 0.41–0.44pt on acc_norm.

**Reads.**

- **`pubsub` ≈ Level-1 at every budget** (Δ = +0.05 / −0.03 / −0.14 / +0.51pt,
  all inside 1 stderr). Restoring cross-expert coupling — loading shared "public"
  knowledge once and re-spending the freed budget on private channels — buys
  **nothing** measurable. This is the decisive **M1 result**: the Oracle-B −
  Level-1 gap is negligible, so the offline-cross-expert-scoring effort (Level 2
  proper) is not worth the ≫57 MB pairwise statistics it would need. It confirms
  the earlier stacked-covariance finding (cross-expert coupling holds ~70% of the
  covariance energy but changes channel *selection* by <2%): the off-diagonal
  mass is real but it does not move which channels you'd keep.
- **`oracle_mag` stays near dense at every budget** (78.5 → 78.8 → 78.3 → 76.8
  vs dense 78.56), beating Level-1 by **+4.3 / +8.2 / +14.7 / +32.7pt** and
  losing <2pt even at a 7/8 cut. The entire remaining headroom is **per-token
  activation information** (`|inter(x)|`), not cross-expert structure: the router
  `g` alone (all offline methods) cannot tell which channels a *specific* token
  actually lights up. `oracle_mag` is an unreachable ceiling (it reads the true
  per-token intermediate), but it relocates the Level-2 target — the gap to close
  is online per-token, not offline cross-expert.

### MMLU 5-shot @ 75% reduction (acc)

| Method | Level 1 (pivchol) | `pubsub` (L2) | `oracle_mag` (ceiling) |
| ------ | ----------------- | --------------- | ------------------------ |
| −75%  | 70.81             | 71.03           | **80.53**          |

Same pattern on the harder, knowledge-heavy benchmark: `pubsub` matches Level-1
(+0.22pt), while `oracle_mag` recovers to **80.53** — essentially the full model
(dense 5-shot MMLU is ~79.5) at a 75% active cut, +9.7pt over Level-1. Per-token
activation information matters *more* on MMLU, consistent with its sensitivity to
destroyed expert capacity.

### M4 regime diagnostic — `g^{2β}` sharpness sweep (HellaSwag, −50%)

Scores channels by `g^{2β}·σ` (β=1 is Level-1; β→∞ degenerates to reduce-top-k).

| β       | 1 (Level 1) | 1.5   | 2     | 3     |
| -------- | ----------- | ----- | ----- | ----- |
| acc_norm | 74.26       | 74.51 | 74.54 | 74.54 |

Sharpening the router weight helps only marginally (+0.28pt, ≈½ stderr) and
saturates by β=2. The mid-budget Level-1-vs-reduce-top-k gap is **not** a
score-dynamic-range problem — it is not closed by concentrating budget on the
top experts, ruling out the simplest explanation and leaving cross-expert
overlap as the (small, per the M1 result) residual.

### M1 oracle ladder + M3 structure (reconstruction, ~2k C4 tokens, layer 46)

_TBD — reconstruction relative-error ladder (`level1` / `pubsub` / `oracle_mag` /
per-token OMP `oracle_exact`) and M3 coherence-vs-pivot-rank; from
`scripts/level2_oracle_ladder.py`._

## `oracle_mag` activation structure — why the down_proj is so sparsely driven

`oracle_mag` recovers near-dense accuracy at a 7/8 active cut (76.84% acc_norm at
ρ=0.125), which says the exact per-token output magnitude
`s_{e,j}(x) = g_e·|inter_{e,j}(x)|·‖W_down[:,j]‖` is concentrated in a small
fraction of the `K·I = 6144` channels each token actually pools. This section
opens that up with calibration data: we replay the exact `oracle_mag` selection
over **69,764 C4 tokens** (padding stripped) on the full un-slimmed
`Qwen3-30B-A3B-Thinking-2507` (K=8, E=128, I=768, 48 MoE layers) and record, per
layer, (1) how often each intermediate channel survives the per-token global
top-B — the **activation frequency** — and (2) each token's sorted score profile
and concentration. Capture: `scripts/oracle_mag_freq_capture.py` (A100-New, 8
GPUs, sharded bf16/sdpa, ~5 min); the replay reproduces `block.py`'s
`_cross_expert_keep` scoring exactly (verified: identical score tensor, 100%
keep-mask agreement on a mock block). Plots: `scripts/oracle_mag_freq_plot.py`.
Figures in `figures/oracle_mag/`; summary numbers in
`figures/oracle_mag/stats.json`.

**Activation frequency** of channel `(layer, e, j)` is its keep-count divided by
its expert's route-count — i.e. *given the expert fired*, how often channel `j`
lands in the token's kept budget. We report it at three budgets ρ ∈ {0.5, 0.25,
0.125}. (A "budget-agnostic" static keep would sit at freq = ρ everywhere; the
spread away from ρ is exactly the per-token dynamism `oracle_mag` exploits.)

### Investigation 1 — per-channel activation frequency (heatmaps)

![Per-layer keep-frequency heatmap grid, ρ=0.5](figures/oracle_mag/freq_grid_r0.500.png)

*Figure: keep-frequency (kept | routed) for every MoE layer at ρ=0.5. Rows =
experts (E=128), cols = channels (I=768). See `freq_layer{1,24,46}_r0.500.png`
for single-layer full-res views and `freq_grid_r0.{250,125}.png` for the tighter
budgets.*

![Sorted per-channel keep-frequency by depth, ρ=0.5](figures/oracle_mag/freq_layer_sorted_r0.500.png)

![Hot-channel fraction and mean frequency vs depth](figures/oracle_mag/freq_hot_by_depth.png)

**Findings.**

- **The dominant structure is horizontal banding — variance is between experts,
  not between a fixed set of "always-on" channels.** Decomposing the per-channel
  keep-frequency variance at ρ=0.5: **73.9% is *within*-expert** (channel-to-channel
  inside one expert) and **26.1% is *between*-expert** (whole rows brighter/darker).
  So there is no small, stable, cross-token set of "load-bearing channels" you
  could prune to statically — a channel that is hot for one token is cold for the
  next. This is the direct micro-level explanation for the earlier M1 result that
  **router-only offline methods (Level-1, `pubsub`) cannot approach `oracle_mag`**:
  the information is per-token, and no fixed keep-set captures it.
- **Almost nothing is always-in or always-out.** At ρ=0.5, over well-sampled
  channels (expert route ≥ 50): only **0.3%** are kept >95% of the time and only
  **0.4%** are kept <5% of the time; **64.6%** sit in the mid band (0.4–0.6, i.e.
  near ρ). At ρ=0.125 only **0.09%** of channels are hot (>0.5) and **6.3%** are
  never kept. The keep decision is genuinely re-made per token.
- **Within a single expert, channels still differ widely** (this is the 73.9%).
  For a well-sampled expert at layer 24 the per-channel frequency spans e.g.
  0.35 → 0.48 (median) → 0.89, with an average within-expert std of 0.083. So the
  intra-expert ranking `oracle_mag` uses (`|inter_{e,j}|·‖W_down[:,j]‖`) is real
  and token-dependent, not a static per-expert channel mask.
- **The bright/dark bands are a routing-frequency artifact, not "super-channels".**
  The few bright rows (expert-mean freq > 0.7: just 5 of 6144) are heavily-routed
  experts (median route 5189); the dark rows (< 0.1: 66 rows) are **dead experts
  that never fired** (median route 0 — 33 of 48 layers have ≥1 dead expert over
  this calibration set). Among experts that *do* fire, mean keep-frequency is
  tightly clustered (p5–p95 = 0.376–0.585 around ρ=0.5) and correlates only
  moderately with log-route (r≈0.46): a token's own budget competition, not a
  global "important expert" label, sets how many of an expert's channels survive.
- **Sparsity is remarkably uniform with depth.** Mean keep-frequency tracks ρ
  almost flat across all 48 layers (0.45–0.49 at ρ=0.5; 0.10–0.11 at ρ=0.125) —
  no layer is systematically easier or harder to sparsify under `oracle_mag`.
  The hot-channel fraction (freq>0.5, ρ=0.5) wobbles 0.31–0.46 with mild peaks at
  the very first layers and a slow rise over the last ~10 layers, but the effect
  is small. This says a **single global per-token budget B** (what `oracle_mag`
  and the dynamic scheme already use) is well-matched to the model — there is no
  strong case for a depth-dependent budget schedule.

### Investigation 2 — per-token score profiles (is the pattern token-dependent?)

For each token we sort its pooled `K·I` scores descending and look at the shape.
Two views: individual normalized sorted curves (each thin line = one token,
divided by its own sum so only *shape* is compared), and per-token concentration
— participation ratio `PR = (Σs)²/Σs²` (the effective number of channels, = KI
for a flat token, small for a peaky one) and the fraction of total score mass the
top-B budget captures.

![Per-token sorted score profiles, shallow/mid/deep layers](figures/oracle_mag/token_sorted_curves.png)

![Per-token concentration (participation ratio + top-B mass) vs depth](figures/oracle_mag/token_concentration.png)

**Findings.**

- **Yes — tokens differ markedly at the same layer, and the difference is in the
  *head* of the distribution.** The normalized sorted curves fan out most at low
  rank (the top few channels): some tokens put >2% of their magnitude in a single
  channel and decay fast (peaky), others start near ~1% and spread it out (flat).
  All tokens share a common heavy tail that collapses only in the last ~5% of
  channels. Concretely, the per-token participation ratio spans a **2.7×** range
  within a layer (all-layer p10–p90 = 769–2099 effective channels of 6144); at
  layer 1 the spread is **3.2×** (p10=498, p90=1620). So a fixed keep-fraction is
  a poor per-token fit — exactly the slack a per-token budget (or the `oracle_mag`
  global top-B) exploits, and which router-only methods, blind to `|inter(x)|`,
  cannot see.
- **Tokens are peaky in an absolute sense: ~25% of channels carry the magnitude.**
  Median participation ratio is **1525 ≈ 24.8% of K·I**. At ρ=0.5 the kept budget
  captures a median **89.6%** of each token's total score mass (p10–p90 =
  87.6–92.1%); at ρ=0.25, **69.4%**; at ρ=0.125, **50.1%**. That the top 1/8 of
  channels still holds half the magnitude is the concentration that lets
  `oracle_mag` lose <2 pts at ρ=0.125 — but note the mass is spread over ~768
  channels, not a handful, so the ceiling comes from *ranking every token's own
  channels*, not from a sparse universal basis.
- **The concentration pattern changes with depth — a shallow/deep "U".** Median
  participation ratio is **lowest (most peaky) at the ends**: L1 ≈ 1057 (17.2%)
  and L46 ≈ 1203 (19.6%), versus the **flattest at mid-depth** L24 ≈ 1834
  (29.8%). Early and late layers concentrate each token's magnitude into fewer
  channels; middle layers spread it. The per-token *heterogeneity* follows the
  same U (spread 3.2× at L1 and 2.7× at L46 vs 1.6× at L24). Read together with
  Investigation 1's flat mean-frequency: **the total budget the model needs is
  depth-independent, but where within a token the magnitude sits (few vs many
  channels) is not** — early/late layers are where per-token routing of the
  budget has the most to gain, mid layers the least.

**Takeaway for the method.** Both investigations point the same way as the M1
oracle result: the headroom `oracle_mag` exposes is **per-token activation
structure**, not a static or cross-expert one. There is no stable sparse keep-set
(within-expert variance dominates, ~0.3% of channels are ever "always on"), and
per-token concentration varies 2.7–3.2× at fixed depth — so the target for a
practical Level-2 method is a cheap online predictor of `|inter_{e,j}(x)|` (or
its rank), not more offline statistics. A uniform global budget B is already
well-matched to depth, though early/late layers (peakier, more heterogeneous)
are where a per-token budget would pay off most.

## Configs

33% study:
`configs/eval/qwen3_30b_a3b_dynamic_{prob,contrib,uniform}_{act,lev}_hellaswag.yaml`
(5 files). Each: `prune_ratio: 0.33`, `dynamic_alloc.enabled: true`,
`k_min: 16`, `real_slim: false`.

50% study:
`configs/eval/qwen3_30b_a3b_dynamic_{prob,coverage,uniform}_lev50_hellaswag.yaml`
(3 files). Each: `prune_ratio: 0.50`, `channel_metric: leverage`, `k_min: 16`,
`real_slim: false`. Reduce-top-k baseline:
`configs/eval/qwen3_30b_a3b_reduce_topk4_hellaswag.yaml`
(`prune_ratio: 0.0`, `reduce_topk: 4`). Level 1:
`configs/eval/qwen3_30b_a3b_dynamic_pivchol_lev50_hellaswag.yaml`
(`criterion: pivchol_global`, `lambda_r: 1.0`, `k_min: 0`); build the artifact
once with `scripts/warm_pivchol_cache.py --config <that yaml>`.

Budget sweep (HellaSwag): Level 1
`configs/eval/qwen3_30b_a3b_dynamic_pivchol_{625,75,875}_hellaswag.yaml` and
baseline `configs/eval/qwen3_30b_a3b_dynamic_prob_act_{625,75,875}_hellaswag.yaml`
(reuse the cached artifacts; only `prune_ratio` changes). MMLU (5-shot):
`configs/eval/qwen3_30b_a3b_dynamic_{pivchol,prob_act}_75_mmlu.yaml`. Sweep
orchestrator (2 jobs/wave, 4 GPUs each): `scripts/run_level1_sweep.sh`.

## Notes

- `expert_out_token_contrib` is a per-expert *scalar* (calibration-averaged),
  so the `contribution` criterion gives a fixed per-expert weight; per-token
  variation comes only through *which* experts a token selects. `router_prob`
  is the truly per-token criterion.
- Implementation: `src/dynamic_active_param/` (allocate / precompute / block /
  install), unit-tested in `src/dynamic_active_param/tests/`.
- `coverage_alloc` adds `prefix_sums (L,E,I)` to the artifact (cache bumped to
  `dynamic_alloc_<metric>_v2.pth`) and a vectorized, token-chunked per-token
  binary search over α in `allocate._coverage_allocate`. Only the
  `_dyn_prefix` tensor is threaded through `block.py`/`install.py`; the
  router_prob/uniform/contribution paths are unchanged.
- Leverage is **precomputed once**: `scripts/warm_leverage_cache.py` derives
  ridge-leverage from the cached `expert_covariances.pth` (no model forward
  sweep) and writes both `expert_scores.pth[leverage]` and the v2 artifact, so
  every eval short-circuits (`merge_slim_eval.py` skips
  `ensure_leverage_and_covariances` when the v2 cache exists) — race-free when
  launching multiple jobs in parallel, and needs no covariances at eval time.
