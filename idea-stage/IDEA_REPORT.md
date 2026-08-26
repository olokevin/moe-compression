# Idea Discovery Report — Intra-Expert Activation Sparsity for MoE

**Direction**: Dense-MLP activation sparsity ("keep only channels with large intermediate
activation") applied to MoE. Two named categories: (1) cheap **proxy** of gate/up (FLOE family),
(2) **Deja-Vu-style trained predictor**.
**Date**: 2026-08-17
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline
**Status**: Phase 1 (literature) complete; Phase 2 (idea gen) pending user steer at checkpoint.

---

## Executive Summary

This repo has **already built working instances of both categories** the direction names:
`input_sparse` (efficient proxy) and `channel_router` (Deja-Vu predictor), both measured
end-to-end on Qwen3-30B-A3B. The literature survey shows `input_sparse` **already fills the
exact gap the strongest prior-art paper (Berkeley, arXiv:2605.08575) lists as future work** —
a cheap proxy that predicts high-magnitude expert channels *before* the full gate matmul, with
cross-expert budget pooling. So the research question is no longer "can this be done" but
**"what is the highest-value, genuinely-novel next contribution given a very mature base and a
large map of closed negative results."** Idea generation (Phase 2) must clear a double bar:
novel vs the literature **and** not already refuted in this repo.

---

## Framing Clarification (load-bearing)

| what you called it | what it actually is | arXiv | role |
|---|---|---|---|
| "FLOE" | **FloE** — On-the-Fly MoE Inference on **memory-constrained GPU** (offloading + 9.3× expert param compression + expert-prefetch prediction). ICML 2025. | 2505.05950 | *Not* per-token channel activation sparsity. Adjacent (offloading). |
| the intra-expert-sparsity paper | **"Uncovering Intra-expert Activation Sparsity for Efficient MoE Model Execution"** (Park, Kim, Gu, **Stoica, Cheung**; UC Berkeley). | 2605.08575 | **THE direct prior art.** Must-cite. |
| "Prox" | **Prox** — Training-Free FFN Activation Sparsity via Approximate Intermediate-Channel Salience. Dense LLMs, 0 MoE. | 2607.27591 | Dense proxy precedent. ID confirmed correct. |

---

## Literature Landscape (all arXiv IDs verified against the arXiv API)

### A. Efficient-proxy / intra-expert channel sparsity (the core slice)

- **2605.08575 (Berkeley, May 2026) — direct prior art.** Training-free; finds up to 90%
  intra-expert activation sparsity. **Computes the full dense gate**, thresholds SwiGLU by a
  **fixed per-model** value, skips low-mag neurons' up/down; fused Triton kernel. **Caps at ~3×**
  (gate stays dense). **Studied cross-expert neuron budgeting → only ≤2pp, abandoned it.**
  2.5× MoE-layer / 1.2× e2e vs dense vLLM. Explicitly leaves to future work: *"a lightweight
  predictor that estimates sparse activation patterns before computing the full gate projection."*
- **Prox (2607.27591)** — cheap proxy (input sparsity + quantized proxy weights) → rank →
  exact compute. **Dense only.** 1.99× decode at 70% FFN sparsity.
- **FlexMoE (2606.27866)** — "One-for-All Nested Intra-Expert Pruning": ranks expert FFN
  channels, learns per-expert discrete prune actions, **one recovery fine-tune** at 40%; ~99.8%
  of Qwen2-57B-A14B at 50% expert-param prune. **Static structural** (not per-token), *with*
  recovery fine-tuning — the closest structural analogue that fine-tunes.
- **Turbo Sparse / dReLU (2406.05955)** — the one dense-sparsity method that reaches MoE
  (TurboSparse-Mixtral), but via **retraining** (dReLU + continued pretraining). 2–5× decode.
- **BlockFFN (2507.08771)** — trains an MoE arch so the *union* of activated params over a chunk of
  consecutive tokens stays small; measured **vanilla MoE has ~70% union over 8 tokens** (contrary
  prior for Idea 3's small-union hypothesis).

**Concurrent adaptive-budget / joint-conditional-computation work (surfaced by Phase-3 novelty —
these preempt pieces of Ideas 1–2):**
- **GeMoE (2606.26287)** — per-token **expert count from gating entropy** (MDL framing). Preempts
  "adaptive budget from a router signal, not a threshold" *for expert count*.
- **CARE (2607.26052)** — nucleus expert admission + a **"budget thermostat"** calibrating the
  per-token threshold from router confidence to a target average. Same idea, expert-count axis.
- **TriRoute (2607.06601)** — one controller jointly emits (attention mode, sparse expert set,
  KV-cache bits) — joint multi-axis per-token conditional compute, but **no intra-expert channel
  axis**.
- **DynaMoE (2502.12325)** — token-difficulty-driven MoEfication (hard tokens → larger experts).
- Also: DTop-p (2512.13996), LExI (2509.02753), Matryoshka-MoE (2509.26520) — all expert-count
  adaptivity only.

### B. Deja-Vu-style trained predictors

- **Deja Vu (2310.17157, ICML'23)** — small 2-layer MLP per block predicts active heads/neurons
  from `h`; 93–99% val accuracy; >2× on OPT-175B. **Dense.**
- **PowerInfer / PowerInfer-2 (2312.12456 / 2406.06282)** — hot/cold neuron split + adaptive MLP
  predictor; 11.69× on a 4090. **Dense, ReLU-family.**
- **LTE (2402.06126, NeurIPS'24)** — trains the model itself toward fewer active neurons (the
  only *jointly-learned-toward-sparsity* method). Does **not** fix a per-token per-expert budget.
- **MoE predictors are all EXPERT-level, for offload/prefetch:** Pre-gated MoE (2308.12066),
  SiDA (2310.18859, >99% hash-hits), MoE-Infinity (2401.14361), Lina (2210.17223), ProMoE,
  ExpertFlow, AdapMoE. **None predict intra-expert channels; none report wall-clock speedup from
  intra-expert channel skipping in a SwiGLU MoE.**

### C. Dense-MLP activation-sparsity foundations (the source idea)

Lazy-Neuron (2210.06313), **CATS (2404.08763)** (SiLU-gate threshold — needs the gate matmul),
**TEAL (2408.14690)** (magnitude threshold on hidden-state *inputs* — decides cheaply, no
matmul), **R-Sparse (2504.19449)** (rank-aware, predictor-free), ReLU-Strikes-Back (2310.04564),
ProSparse (2402.13516), ReLU² (2402.03804), Q-Sparse (2407.10969, top-k), **GRIFFIN (2404.01365)**
(sequence-level "flocking": fix a neuron set from the prompt, reuse during decode). All
**dense-only** except where noted.

### D. Surrounding MoE landscape (positioning)

- **Expert pruning/merging (static, whole-expert):** NAEE (2402.14800), MC-SMoE (2310.01334),
  EEP (2407.00945), He et al. trim-vs-slim study (2406.02500).
- **Dynamic-k / expert skipping (per-token, whole-expert):** **Ada-K (2410.10456)** (PPO
  allocator sets per-token k; ~25% FLOPs, ~1.2–1.4× *real* speedup), AdaMoE (2406.13233),
  DynMoE (2405.14297).
- **Static intra-expert neuron pruning:** MoE-I² (2411.01016), DERN (2509.10377), TENP
  (2606.09885, 2026 preprint — verify authorship).
- **Quantization (orthogonal):** QMoE (2310.16795), MoQE (2310.02410), MC-MoE (2410.06270).

---

## Where This Repo Already Sits (the double bar for new ideas)

**Category 1 — efficient proxy → `input_sparse` (`sparse_probe`).** Reads served up/gate on the
token's top-`ρ_input` coords by `|x|`, scores `g_e·|SiLU(gate)⊙up|`, keeps global top-B pooled
across the token's K experts. **Zero extra storage** (probe = served weight view). Frontier:
**−73.3% used-params → 74.64 HS / 77.67 MMLU**, smooth to **−80% → 72.55/76.11** (dense
78.56/80.91). Entry-level `weight_sparse` reaches **−80% → 74.87 HS**.
→ *Already beats 2605.08575's design on the axis they capped* (cheap proxy avoids the full gate;
cross-expert pooling *works* here via top-B where their per-expert threshold gave ≤2pp).

**Category 2 — Deja-Vu predictor → `channel_router`.** Learned `h→channel` router, built
end-to-end. **Goal NOT met:** near-lossless needs mass-recall ≥ 0.99; cheap weight-free predictor
caps at ~0.63; mis-selection costs ~600% PPL / unit mass. Beat the Deja-Vu MLP baseline at 1/5
params but the family can't reach the bar.

**Closed negatives (idea-gen must NOT re-propose):** cross-layer budget allocation (3 solves, all
lose to uniform); quantizing the probe; low-rank / activation-aware scorers (below a free static
prior); static masks / hot-sets; floating the per-token *channel* budget under a global τ
(worse — score is already g_e-scaled); per-token temporal reuse (adjacent-mask IoU 0.11–0.17);
`colnorm` / `router2`; channel-block metadata; relaxed-candidate cascade (no cheaper than a better
probe).

---

## The Gap Map — genuinely open frontiers (intersection of lit-gap ∧ repo-open ∧ not-refuted)

| # | Frontier | Lit gap | Repo status | Notes |
|---|---|---|---|---|
| G1 | **Two-level dynamic: dynamic-k (drop experts) × per-token channel sparsity, jointly budgeted** | **No published system** combines whole-expert *dynamic* sparsity with per-token intra-expert channel selection (survey D). Static combos only (MoE-I²/DERN/TENP). | Flagged open ("stack with reduce-top-k"; top-4×input_sparse >−85%); both levels' machinery exist. | Router `g_e` already says which experts to thin/drop. Needs a *principled joint per-token allocation*, else it's just stacking. |
| G2 | **LoRA recovery of a per-token dynamic mask** | 2605.08575 & Prox are **training-free, no recovery**; FlexMoE fine-tunes but for a **static** mask. No per-token-dynamic-mask recovery for MoE exists. | Repo's own **"largest single win," untouched.** LoRA pipeline exists. | Real question: can one *static* adapter absorb a *distribution* of per-token masks? |
| G3 | **Real wall-clock kernel + honest used-param+latency ledger** | Field-wide: predictor/scorer cost usually excluded; 2605.08575 gets 2.5× but leaves gate dense; per-token gather often *slower* than dense. | Everything is masking-sim / active-param; **no speedup measured**; index_select < dense; fused upper bound 1.24–1.30×. | input_sparse's proxy can beat 2605.08575's 3× cap — *if* the kernel realizes it. High eng cost. |
| G4 | **Adaptive per-token / per-layer budget by a non-score signal** | 2605.08575 fixed per-model threshold; dense lit favors float (CATS/TEAL) but MoE float-vs-fixed "genuinely open" (survey C). | **Caution:** floating the channel budget under a g_e-scaled τ is a *closed negative* here. But a budget set by token *compressibility* (not a score threshold) is different. | Adjacent to a closed result — must dodge the g_e confounder. |
| G5 | **Sequence-level channel sets (prefill→decode), GRIFFIN-for-MoE** | GRIFFIN (dense) fixes a neuron set from the prompt; **no MoE version**. | Per-token temporal reuse is dead (IoU 0.11–0.17) — but sequence-*union* sets are a different object. | Batch/decode amortization also open. |
| G6 | **Weight-aware learned predictor / nominate-then-verify hybrid** | **No predictor reads the weights** (all use `h` only); **no "cheap-nominate → exact-verify" hybrid** exists (survey B). | channel_router caps at 0.63 recall; cascade λ=1.5 was "no cheaper than a better probe." | Partly weakened by repo evidence; needs a genuinely new angle to beat input_sparse. |

---

## ⚑ Validation synthesis (Phase 3+4) — the max-novelty ∧ accuracy-only target is nearly empty

Three independent agents (2 novelty, 1 adversarial ICML reviewer) converge on a hard conclusion:
**the two "obvious" new mechanisms are each killed by a *different* reviewer, from opposite sides.**

- **Idea 2 (two-level drop-experts):** adversarial **REJECT** — global top-B is the exact optimum
  of the per-token channel knapsack, so the joint/expert-structure is *weakly worse* at fixed
  used-param; the win is latency-only. Novelty also thin.
- **Idea 1 (adaptive budget from router signal):** **PREEMPTED** — GeMoE (2606.26287, per-token
  expert count from *gating entropy*) and CARE (2607.26052, per-token budget "thermostat" from
  router confidence) already publish "adaptive per-token budget from a router signal, not a score
  threshold." Berkeley 2605.08575 already publishes per-token channel sparsity. **Neither headline
  is new alone.** The *only* unclaimed slice is (i) a **joint** (expert-count × channels) single
  budget — which the adversarial review says can't help accuracy — and (ii) driving the **channel**
  budget (not expert count) from a *compressibility* signal orthogonal to `g_e` (e.g. score-decay).
- **Idea 3 (sequence-level MoE):** 5/10, occupies a *named-future-work* gap, but a strong contrary
  empirical prior (BlockFFN's ~70% 8-token union) puts the core hypothesis at real risk.

**Consequence:** under *max-novelty ∧ accuracy-only*, the surviving research-grade candidates are
the two **riskier** ones — **Idea 4 (output-coverage / non-scalar selection**, the *only* idea that
attacks the "top-B is optimal" premise — not yet novelty-checked) and **Idea 3** (needs a surprising
empirical result). The safe, high-EV path (**LoRA recovery**, unpreempted) was de-scoped by the
direction choice, and the genuinely-novel *joint* framing gives **latency**, which was de-scoped by
the accuracy-only choice. This tension is a real finding and is the subject of the Phase-2 checkpoint.

## Ranked Ideas

**Scope filter applied:** max-novelty *new mechanism*, ICML-style accuracy accounting (accuracy vs
used-active-params in masking simulation; latency is secondary). This is a demanding filter here,
because the repo has *already closed* every representation/geometry/static-structure route. What
remains open for an **accuracy-frontier** move at fixed used-param is essentially **budget
allocation via a signal orthogonal to the activation score**, and **selection-objective** changes
— everything else re-derives top-B-by-magnitude, which is provably near-optimal here.

> Honest north star: the existing global top-B `input_sparse` frontier is the baseline every idea
> must *beat on accuracy at equal used-param*, not merely match at lower latency.

### 🏆 Idea 1 (headline): Difficulty-adaptive per-token budget via a score-orthogonal signal
*(G4 done to dodge the closed negative; fuses with the two-level structure below)*

**Mechanism.** Keep `input_sparse`'s cheap proxy + global top-B *selection*, but make the **total
per-token channel budget `B_tok` variable** at a fixed *mean* budget, set from a **token-difficulty
signal that does not contain the router-confidence confounder**: candidates are (i) the **decay
rate of the token's sorted proxy-score profile** (a fast-decaying profile ⇒ compressible ⇒ small
`B_tok`), (ii) **router entropy / top-1 margin**, (iii) hidden-state norm concentration. Easy
tokens release budget; hard tokens spend it.

**Why this is not the closed negative.** The repo already refuted *floating the channel budget
under a global threshold on the g_e-scaled score* (−0.16…−0.32pt at iso-cost) — because that
threshold mostly measures *how confident the router was*, so high-g tokens hoard channels. This
idea sets `B_tok` from a **per-token-normalized compressibility signal that is independent of the
score's magnitude**, which is a different quantity and the failure mechanism does not apply.
(Note the repo *did* find floating the **input reads** by `|x|` is a small win — consistent with
"float by a non-confounded signal.")

**Accuracy hypothesis.** Reallocating budget from easy→hard tokens moves the accuracy-vs-used-param
frontier up at fixed mean cost. This is the cleanest *accuracy* lever still open.

**Novelty.** 2605.08575 uses a **fixed per-model** threshold; dense CATS/TEAL float per token but
have no expert structure and no orthogonal-signal budget. No MoE work sets a per-token channel
budget from difficulty rather than a score threshold. *(novelty agent running)*

**⚠ Adversarial review caveat (W2):** entropy/margin are functions of the router logits already
baked into the score, so a budget driven by *those* repeats the closed negative. **The defensible
thread:** the prior float lost because it fed **confident = easy** tokens *more* (backwards); and
**sorted-score-decay is NOT a pure router-logit function** — it depends on the activation pattern
`|SiLU⊙up|`, so it measures token compressibility beyond `g_e` and allocates in the **correct**
direction (flat decay = hard = more). Idea 1 survives *only* if it (a) uses a signal provably
orthogonal to router confidence and (b) clears **≥0.5pt iso-used-param across ≥2 models, ≥3 tasks**
(above ~0.35–0.40pt/task noise). Cheap offline sign-test first.

**Cheap pilot (offline, ~minutes/GPU).** Extend `scripts/probe_threshold_budget.py` (already does
iso-cost interpolation) to allocate `B_tok` by sorted-score-decay / entropy instead of a score
threshold; measure block-output `rel_err` at iso-mean-cost vs fixed-B on L6/22/38/46 × ~3–4k C4
tokens; convert via the validated −24.5 HS pt/unit slope. **Kill criterion:** rel_err not below
fixed-B ⇒ dead (same as the closed float-by-score result).

### Idea 2 (chosen direction G1): Joint two-level allocation — dynamic-k × channels, with gate renorm

**Mechanism.** Per token, select over **(expert, channel) pairs** under one budget, but treat
**dropping a whole expert** as a distinct action: when an expert's best channels don't clear a
per-token bar, drop it entirely and **renormalize the surviving gates** (top-k-style). The
per-token expert count `k_tok ≤ K` *emerges* from the channel-value distribution (unlike Ada-K,
which learns `k` alone with no channel view).

**The honest accuracy caveat (this is the crux).** Under *pure used-param accounting*, the global
top-B **already** starves low-g experts of channels, so dropping them adds little unless you also
count the per-expert fixed cost (routing/gather/down_proj epilogue) — which is a **latency**
quantity the chosen scope de-prioritizes. The **only accuracy content** is the **gate
renormalization**: does a smaller `k_tok` with renormalized gates + more channels each beat a
larger `k` with thin experts, at equal `B`? That is a real, cheap, binary test.

**Why it's still worth featuring.** (a) It's the user's chosen direction and the cleanest confirmed
*literature* gap (survey D: no published system does per-token dynamic-k × intra-expert channels;
2605.08575 abandoned cross-expert budgeting). (b) It reframes the repo's own negative
"expert-redundancy-is-NOT-expert-level (R²=0.03)" into a **positive per-token** statement:
redundancy is per-token, and dynamic-k is how you collect it. (c) It sets up the two-level story
for a later systems/latency paper without needing it now.

**Cheap pilot.** Modify `select_global_topB` → knapsack with expert-drop + gate renorm at fixed
`B`; offline rel_err vs global top-B on cached captures. **Kill criterion:** renorm is
accuracy-neutral vs global top-B ⇒ demote to a latency-only result (out of this scope).

**⚠ Adversarial review verdict (senior-ICML agent): REJECT under the chosen accuracy-only scope.**
W3 (the deepest): global top-B is **the exact optimum of the per-token unconstrained channel
knapsack** — imposing per-expert grouping + integer expert on/off is a *strictly tighter* feasible
set → weakly **worse** at fixed used-param, by construction. W1: expert-dropping is invisible on
the used-param axis (low-g experts already get ≈0 channels); its only saving is the fixed
per-expert overhead — a **latency** quantity the scope de-prioritizes. So **Idea 2 is a latency
idea mislabeled as accuracy** unless the gate-renorm residual (predicted <0.1pt) surprises. It
re-introduces structure on exactly the axes (expert, budget) the repo already showed buy no
accuracy (expert-redundancy R²=0.03; cross-layer schedule closed-negative). *Consequence: Idea 2
demoted; it is the natural headline of a separate **systems/latency** paper, not this one.*

### Idea 3: Sequence-level intra-expert channel sets for MoE (GRIFFIN-for-MoE, prefill→decode)

**Mechanism.** During **prefill**, accumulate per-expert the **union / frequency histogram** of
high-activation channels over the prompt tokens; during **decode**, restrict each expert to its
prompt-derived hot set (fixed per sequence). Amortizes selection over the whole generation; static
per-sequence gather pattern.

**Novelty.** GRIFFIN (2404.01365) does exactly this "flocking" for *dense* LLMs; **no MoE version
exists** (survey C/B). *(novelty agent running.)*

**Accuracy hypothesis + tension.** Works iff the per-sequence *union* of hot channels is small.
**Caution:** the repo measured *per-token* temporal reuse is dead (adjacent-mask IoU 0.11–0.17) —
but a per-sequence *union* is a different object (a set, not adjacent overlap). The pilot resolves
this directly.

**Cheap pilot.** From order-preserving captures (the repo notes these must be collected), measure
(a) union-set size as a fraction of `I` per expert per sequence, and (b) rel_err of decode tokens
restricted to it. **Kill criterion:** union > ~40% of `I` at usable accuracy ⇒ no amortization win.

**Novelty verdict (agent): PARTIALLY NOVEL, 5/10.** No GRIFFIN-style sequence-level channel-set
method exists *for MoE* — GRIFFIN (2404.01365) is dense; 2605.08575 does it *per-token* and
**explicitly lists "sequence-level mask reuse" as its own future work** (dangerous closest cite).
But the mechanism itself is GRIFFIN verbatim, so defensibility rests on a *surprising empirical
result*, not conceptual originality. **Load-bearing risk (strong contrary prior):** **BlockFFN
(2507.08771)** measured that vanilla MoE has ~**70% union over just 8 consecutive tokens** (union
grows fast; they needed CLS-aware *training* to shrink it); combined with the repo's adjacent-token
IoU 0.11–0.17, the "small union" hypothesis is at real risk. **Mitigator:** per-*expert* unions see
only that expert's routed tokens, so may be smaller/more topical — but **routing drift** during
decode is the failure mode. Run the union-size kill-shot before building anything. Mandatory
baselines: 2605.08575 (must beat, per-token), GRIFFIN, BlockFFN.

### Idea 4 (stretch): Output-coverage-aware per-token selection
*(novel selection objective; higher risk)*

**Mechanism.** Replace top-B-by-scalar-magnitude with a set that best **reconstructs the expert
output**: channel `j`'s contribution is the vector `SiLU(gate_j)·up_j·W_down[:,j]`, so two
high-magnitude but near-collinear channels are partly redundant. A greedy orthogonal-residual /
facility-location selection could keep a more *informative* set at the same `B`.

**Risk (why it's a stretch).** The repo found `‖W_down‖` weighting a wash (CV 0.055) and
output-side SVD ≡ activation-aware SVD (dead). Those close *norm/rank* geometry — but **not
per-token collinearity of selected channels**, which is untested. Cost is the catch: a per-token
output-Gram is expensive and fights the cheap-proxy thesis; needs a cheap surrogate (e.g. sign
patterns / precomputed W_down column clusters).

**Cheap pilot.** On cached captures, compare rel_err of greedy-coverage vs top-B at equal `B` on
one layer. **Kill criterion:** ≤0.1 rel_err improvement ⇒ dead (geometry wash confirmed for
selection too).

### Composable capstone (not the headline, but the highest-EV finisher): LoRA recovery (G2)

The user chose *new mechanism* over the capstone, but note: **LoRA recovery of the per-token
dynamic mask** is the repo's own "largest single win," is a literature gap (2605.08575 & Prox are
training-free; FlexMoE recovers only a *static* mask), and **composes with whichever of Ideas 1–4
wins** — it closes the ~2.5–3.9pt gap to dense at −73.3%. Carry it as the recovery stage of the
final proposal.

## Eliminated at generation (already refuted in-repo — did NOT propose)

- **Better scoring via low-rank / activation-aware / weight-geometry** — below a free static prior;
  output-side SVD ≡ activation-aware SVD (analytic). Dead.
- **Quantized proxy** — dominated on storage *and* accuracy vs serving-precision reads.
- **Static hot-sets / masks (incl. Wanda)** — a fixed mask encodes only the mean token.
- **Cross-layer budget schedule** — 3 solves, all lose to uniform (wrong *form*, not weighting).
- **Learned h→channel predictor (Deja-Vu style)** — `channel_router` capped at 0.63 mass-recall vs
  the ≥0.99 needed; weight-free predictors provably can't reach it here.
- **Nominate-then-verify cascade (λ=1.5)** — "no cheaper than a better probe at the same budget."
- **Floating the channel budget by a score threshold** — worse (g_e confounder). *(Idea 1 is the
  version that dodges this.)*
- **Real-kernel / wall-clock as the headline** — out of the chosen ICML/accuracy scope (kept as
  future systems paper; input_sparse can beat 2605.08575's ~3× dense-gate cap *if* built).

## Refined Proposal

*Pending Phase 4.5.*
