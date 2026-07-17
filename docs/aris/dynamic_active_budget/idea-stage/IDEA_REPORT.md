# Idea Discovery Report — Per-Token Dynamic Active-Parameter Allocation for MoE

**Direction**: Reduce 33% of active parameters in an MoE LLM by allocating a *different* per-token budget to each expert — more parameters to experts with higher routing probability / higher attribution contribution — then choosing which channels to load via (a) attribution channel score or (b) Nyström ridge-leverage score (select-only, no down_proj reconstruction).
**Date**: 2026-07-16
**Pipeline**: research-lit → (idea-creator) → novelty-check → research-review → research-refine
**Status**: Phase 1–3 complete (survey + verified novelty). Awaiting user pick before refine.

---

## Executive Summary

The user's *first* idea (per-token, router-weighted, uneven expert **width**) is **substantially pre-empted** — MoSE (ICML 2026), TENP, AnyExperts, Alloc-MoE, FlexMoE all do token- or importance-adaptive budgeting. Worse, the repo's own pilot shows the naive `router_prob × activation` version **underperforms** uniform pruning (75.96 vs 78.23 acc_norm, masking-sim). **But** two sub-ideas survive as genuinely novel:

1. 🏆 **Global cross-expert per-token channel ranking** — drop the per-expert budget step entirely; pool all K×I channels of a token's routed experts, keep the global top-B. Per-expert width *emerges*. **NOVEL** (moderate-high confidence): every prior method, incl. MoNE and FlexMoE, selects within each expert.
2. **Ridge-leverage / Nyström score as a pure channel *selector*** (which up/gate columns + down rows to keep) **without** the Nyström weight reconstruction. **NOVEL** in MoE/LLM-FFN pruning: CUR/leverage work never decouples selection from reconstruction; Wanda selects-without-reconstruct but on activation magnitude only.

**Honesty anchor (user-supplied, critical):** the numbers above are *masking simulation* (no real speedup). A **real** 33% active-param cut (uniform Nyström, physically slimmed) is only **~65% HellaSwag**. The paper's story must therefore be "close the 65→~75 gap while achieving a *real*, deployable per-token reduction," not "beat 78 with a mask." Static pruning is out of scope per user; the contribution must be token-dynamic.

---

## Literature Landscape (Phase 1, verified)

**All arXiv IDs below were fetched and confirmed real unless marked.**

### The per-token uneven-*width* idea is crowded

| Paper | arXiv | Venue | Does per-token? | Uneven across experts? | Importance-weighted? | Width granularity? | Relation |
|---|---|---|---|---|---|---|---|
| **MoSE** | 2602.06154 | ICML 2026 (code) | ✅ | ✅ | ✅ router conf | ✅ slimmable | **Near pre-emption.** Maps router confidence→per-expert width under fixed budget. *Requires multi-width slimmable pretraining.* |
| **TENP** | 2606.09885 | 2026 | ❌ static | ✅ | ✅ output mag | ✅ neuron prune | Uneven width but static per token |
| **FlexMoE** | 2606.27866 | 2026 | ❌ static/budget | ✅ | ✅ ranked | ✅ nested channels | Ranks expert FFN channels, per-expert nested masks + recovery FT |
| **AnyExperts** | 2511.18314 | 2025 | ✅ | ✅ | ✅ | ❌ slot count | Per-token variable expert *count*, multimodal |
| **Alloc-MoE** | 2604.08133 | 2026 | ✅ | ✅ | ✅ routing scores | ❌ activation count | Budget-aware expert *activation* count via DP |
| **MoNE (vision)** | 2407.19985 | ECCV 2024 | ✅ | per-token nested | ✅ | ✅ nested width | Routes token to *one* nested-width expert, not uneven split over top-K |
| **MoNE (neuron experts)** | 2510.05781 | 2025 | — | within-expert | ✅ | ✅ neuron | top-k **within each expert** — sharpest contrast for idea #1 |

### The two surviving novelty pockets

- **Global cross-expert channel ranking**: no work pools K×I channels across a token's experts and keeps the global top-B. Closest (MoNE, FlexMoE) still select within-expert; closest per-token-dynamic (MC#, Ban&Pick, AnyExperts) prune whole experts. Global-vs-local sparsity allocation exists in *dense* pruning (OWL, SparseGPT/Wanda) and in MoE only at *layer/expert-budget* level (GRAPE, DiEP, EvoESAP) — never cross-expert per-token channels.
- **Leverage-select-without-reconstruct**: CUR/leverage (L-DEIM, Boutsidis-Woodruff) always builds a low-rank C·U·R; LLM low-rank (ASVD, SVD-LLM, NSVD) always reconstructs. Using the ridge-leverage score *only* to pick a keep-mask on original weights is unclaimed.

### MoE channel/expert importance scoring (background)
MoE-Pruner (2410.12013, |W|·‖X‖·router-prob), Wanda (2306.11695), SparseGPT (2301.00774), MoE-I² (2411.01016, low-rank), EASY-EP (output-aware token contribution — mirrors repo's `expert_out_token_contrib`), SHAPE (Shapley over experts).

---

## Diagnosis of the repo's negative pilot

Repo `docs/results/dynamic_active_param/q3_30b_dynamic_active.md`: `router_prob × activation` = **75.96 acc_norm** vs static uniform **78.23** (both masking-sim, 33%). Two root causes, both fixable and both idea leads:

1. **Routing-weight double-counting** (`block.py:78`): the expert output is *already* scaled by `routing_weights`. Allocating *width* by routing weight too means low-prob experts are penalized twice → their channels are starved even when individually important. A global ranking (idea #1) or a decorrelated criterion (idea #2) avoids this.
2. **`contribution` criterion isn't per-token**: `expert_out_token_contrib` is a calibration-averaged per-expert scalar; per-token variation only comes from *which* experts fire. Only `router_prob` is truly per-token. → motivates a genuinely per-token importance signal (idea #3).

---

## Ranked Ideas

### 🏆 Idea 1 — GATE: Global Allocation of Token-adaptive Experts (per-token global cross-expert channel ranking) — RECOMMENDED

**Mechanism.** For each token, concatenate the FFN intermediate channels of all K routed experts into one pool of K×I candidates. Score each channel `s_{e,c}` (its importance metric, optionally modulated by expert routing weight/contribution), and keep the **global top-B** across the pool, B = round(0.67·K·I). Per-expert width `k_{t,e}` is **emergent**: important experts keep more channels, marginal ones fewer, possibly zero. No per-expert budget step — this is the exact move no prior work makes.

**Why it beats the naive baseline.** Removes the artificial "every selected expert gets ≥ its floor" constraint; lets the model spend the token's budget where it helps most globally. Naturally sidesteps routing-weight double-counting if the score is chosen well.

**Novelty**: NOVEL (verified, moderate-high confidence). Differentiation: drops per-expert/per-layer budget allocation; allocation emerges from one global ranking; zero-channel experts allowed.

**Pilot** (cheap, masking-sim, code mostly exists): replace `allocate_budgets` + per-expert keep-mask in `block.py` with a single global top-B over stacked `(K, I)` scores. Compare vs static-uniform and vs the per-expert dynamic baselines on HellaSwag. ~1 GPU-run per config.

**Risk**: masking-sim only shows an *oracle* upper bound; needs the real-reduction realization (idea 4) to be a deployable claim.

### Idea 2 — Leverage-select channel scoring for the global pool — RECOMMENDED companion

**Mechanism.** Use the Nyström ridge-leverage score `diag((C+λI)⁻¹C)` as the channel importance metric feeding idea 1's global ranking — selection only, original weights kept, **no** down_proj reconstruction. Compare against `activation` metric.

**Novelty**: NOVEL as a standalone framing (select-without-reconstruct with a leverage score). Naturally composes with idea 1 as the "which channels" axis while idea 1 owns the "how many per expert" axis — together they replace *both* of the user's original two-knob decisions with one unified ranking.

**Risk**: reviewers ask "why not do the Nyström reconstruction you already have?" — answer must be a controlled ablation showing select-only ≈ or > reconstruct at far lower cost, *and* that leverage-select > activation-select.

### Idea 3 — Marginal-loss (not routing-prob) allocation signal

**Mechanism.** Replace the routing-weight allocation criterion with a per-(expert,channel) **loss-sensitivity** score (Taylor / gradient·activation on calibration), which is *not* already baked into the output scaling. Feeds idea 1's pool. Directly targets the double-counting diagnosis.

**Novelty**: medium — importance scoring is crowded, but *decoupling allocation signal from the routing weight already applied downstream* is a clean, defensible insight and a strong ablation axis rather than a standalone paper.

### Idea 4 — Real (not masked) per-token reduction via gathered variable-width experts — the systems contribution

**Mechanism.** Realize idea 1 as an actual FLOP/param reduction: per token, gather only the kept channels' rows/cols of gate/up/down and run a narrowed matmul (grouped/ragged), instead of masking full-width. Report *real* HellaSwag at a *real* 33% active-param cut vs the ~65% uniform-Nyström-slim baseline.

**Novelty**: the honest, deployable version of the story; distinguishes from MoSE (which needs slimmable pretraining) by being **post-hoc on a frozen pretrained model + LoRA recovery**.

**Risk**: engineering-heavy (ragged matmul on GPU); may only show speedup in a microbenchmark. Could be scoped as "we report real accuracy at real budget; kernel efficiency is future work" with masking-sim as the accuracy oracle.

### Combined recommended paper
**Idea 1 (global ranking) + Idea 2 (leverage-select scoring) as the method; Idea 3 as the key ablation; Idea 4 as the real-reduction validation vs the ~65% baseline.** One coherent story: *"A post-hoc, per-token global cross-expert channel selector that turns a frozen pretrained MoE into a token-adaptive-width model, closing most of the real 33%-reduction accuracy gap without slimmable pretraining."*

---

## Eliminated / de-prioritized
- **Naive per-token uneven width by routing prob** (the original framing): pre-empted (MoSE et al.) AND underperforms static in the repo pilot. Keep only as the baseline to beat.
- **Static leverage pruning**: user explicitly out of scope (token-dynamic only).

## User decisions (locked)
- **Idea set**: Global ranking + leverage (combined) → refined as **GATE**.
- **Realization**: masking-sim now as oracle; real physically-slimmed reduction as the goal. Honest floor = real uniform-Nyström ≈ **65% HellaSwag**. Static pruning out of scope; token-dynamic only.

## External Critical Review (Phase 4, adversarial AC) — score 3/10 as originally framed

Brutal but fair. Decisive corrections (all adopted into the refined method):
1. **Never zero a routed expert** — enforce a ≥1-block per-expert floor. Emergent-zero-width is strictly worse than static (drops a router-selected expert, orphans its softmax mass) — likely why the naive pilot lost. "Allow zero" kept only as an ablation.
2. **Routing weight `g_e` is the correct cross-expert scale, not a double-count.** Channel (e,i)'s marginal output contribution is `g_e·SiLU(gate)·up·down` — so the global score should be **`g_e`-weighted leverage**; the earlier double-counting instinct is likely backwards. Now a derived score + central ablation.
3. **Honesty on "compression".** Masking-sim = oracle upper bound (labeled as such); must carry the winner to a **real block-granular gathered-GEMM** with wall-clock, or reframe as active-FLOP@iso-accuracy. Per-token *arbitrary* width has no GPU speedup path — **block granularity (16/128)** is mandatory for a real claim.
- **Verified-against-code caveat**: the leverage score uses an *aggregate* covariance → it is a **static** per-channel scalar. "Per-token" applies to selection/budget, NOT the score. Must be stated plainly.

**Minimum bar to accept**: (1) a real deployable GATE number beating the 65% floor, recovering to within 2–3 pt of dense; (2) the isolating ablation global-pool > per-expert at matched-everything by >1 pt on ≥3 tasks; (3) ordering (GATE > static > naive) survives real slimming, not just sim; (4) a real latency number or defensible FLOP reframe; (5) multi-task + ≥2 model families; (6) random-selection control.

## Refined deliverables
- Method: `idea-stage/refine-logs/FINAL_PROPOSAL.md`
- Experiments: `idea-stage/refine-logs/EXPERIMENT_PLAN.md` (gated on the isolating ablation B2; run A2/A1/A0' pilots first)

## Next steps
- [ ] Implement `global` criterion + block-granularity + per-expert floor in `src/dynamic_active_param/block.py`; extend unit tests
- [ ] Run Block A masking-sim pilots (A2 g_e·leverage global vs A1 naive vs A0' static) → check the >1pt gate
- [ ] If gate passes → Block B2 isolating ablation → Block C real reduction + LoRA → Block D generality
- [ ] If gate fails → write up as negative result + leverage-select analysis
