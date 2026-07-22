# Heterogeneous Budget Allocation for MoE Compression: Intern Plan

## 1. Introduction

### The Mission

The Macro team's goal is to run large MoE models on Cor3 hardware — P1 (cloud) and Tehama (edge) — through structural parameter reduction (layers, width, MLP dimensions, experts). The target is 2x reduction for Tehama (e.g., Qwen3-30B-A3B) within 1pp accuracy degradation; current manual recipes give ~4pp drop at 1.5x. On Cor3, decode is memory-bandwidth bound, so cutting the weights that must be loaded translates almost directly into speedup.

### The Goal: Heterogeneous Budget Across Experts

A MoE layer holds dozens of experts that are *not* equally important or equally compressible. A rarely-used expert may still be information-dense; a frequently-used one may be highly redundant. Compressing every expert by the same ratio therefore leaves value on the table. The unifying goal of this plan is to **allocate a per-expert budget** — give each expert exactly as much capacity as it is worth. This applies along two axes:

- **Total parameters (static).** Allocate a different compression ratio per expert. This shrinks the model footprint in DRAM. Token-independent — decided once, offline.
- **Active parameters (dynamic).** Load a different subset of each expert's parameters *per token*. This cuts the weights loaded on each forward pass — the quantity that bounds decode latency on Cor3 — without shrinking the stored model.

Both reduce to the same question — *how much of each expert do we keep, and which part* — asked at two different granularities.

---

## 2. Background: How Prior Work Allocates Budget Across Experts

Recent MoE-compression papers converge on the same recipe: **estimate each expert's importance and compressibility, then allocate a matched budget.** They differ in the metric and the mechanism. Three are representative.

### 2.1 Learn the Mask — *Elastic Training*

The prior intern plan (`legacy/intern_plan_v2.md`). Rather than score experts by a fixed heuristic, define an importance ordering per axis so that any submodel is a *prefix* ("keep the top-$k$"), then train a small **router** that maps a target budget to a per-layer, per-expert config. Gumbel-Softmax makes the discrete choice differentiable, and the router is trained jointly with the model under a distillation + budget loss — so the allocation is *learned* end-to-end rather than read off a hand-designed score.

### 2.2 Leverage / Coverage — *Attribution-Guided Pruning*

Structured, token-independent row/column pruning. Two ingredients: a per-expert **contribution prior** $\phi_g$ (loss-based, how much the group matters) and a per-channel score $s_i$ (**activation magnitude**). Channels are ranked by a **coverage ratio** $\rho_g(n)$ — the fraction of a group's total score captured by its top-$n$ channels (an effective-rank measure that makes groups comparable). The budget is then split so each group's target coverage is $\propto \phi_g$, keeping the highest-scoring channels to maximize covered score under a global budget.

### 2.3 Contribution & Compressibility — *RFID-MoE*

Keeps all experts but gives each a low-rank SVD budget matched to its importance. Importance fuses **routing frequency** (how often the expert fires) with **information density** via the **effective rank** of its singular spectrum (a rarely-used expert can still be high-rank). Budget is allocated proportionally — important experts keep more rank, none goes to zero — and each expert is compressed via a shared-basis truncated SVD (building on MoBE), with the low-energy residual cheaply reconstructed.

### Common Structure and Its Gaps

All three share a **two-level** design — first a per-expert budget, then per-channel selection within each expert — and all target **total-parameter** compression with a **forward-only, static** metric. This leaves three gaps:

1. **The channel metric ignores trainability.** Activation magnitude and forward effective rank measure inference reconstruction. But in the Macro pipeline the compressed model *heals* afterward; a channel with low activation but large gradient is exactly what recovery needs, and forward-only scores discard it.
2. **The two-level split is artificial.** Allocating a budget per expert and *then* selecting channels is strictly less flexible than ranking every expert's channels together against one budget — the split forces an importance prior we would rather not hand-design.
3. **Everything is static.** None of these methods exploit *per-token* parameter loading to cut **active** parameters — the axis that most directly moves decode latency on Cor3.

These three gaps define three levels of increasingly ambitious ideas.

---

## 3. Three Levels of Ideas

| | Idea | What changes | Axis | Effort |
| - | ---- | ------------ | ---- | ------ |
| **A** | Training-aware channel score | Better *metric* inside the existing two-level framework | Total params | Low — drop-in |
| **B** | Unified cross-expert selection | Remove the two-level split; one ranking across all experts | Total params | Medium |
| **C** | Per-token active parameters | New axis: dynamic, per-token parameter loading | Active params | High — novel |

Idea A improves the metric; Idea B improves the allocation mechanism; Idea C opens a new axis. They stack: A's score and B's unified kernel both feed directly into C.

---

## 4. Idea A: Training-Aware Channel Scores

**Under the Attribution-Guided budget framework, replace the forward-only channel score with a trainability-aware one.**

Attribution-Guided ranks channels by activation magnitude and coverage — both forward-only. We instead score each channel by the **joint forward+backward kernel** developed in our prior training-aware compression work (`ref/training_aware_compression.md`). For a gated MLP the hidden-channel kernel is

$$
K_{\text{joint}} = \bar C_f^{1/2}\, \bar C_b\, \bar C_f^{1/2} + \lambda I,
\qquad
C_f = Z^\top Z,
\qquad
C_b = B_u^\top B_u + B_g^\top B_g,
$$

where $C_f$ is the forward hidden-activation covariance and $C_b$ is the covariance of the gradient flowing *into* the up/gate matrices. The per-channel Nyström score

$$
\text{score}_i = \operatorname{diag}\!\big((K_{\text{joint}}+\lambda I)^{-1} K_{\text{joint}}\big)
$$

replaces activation magnitude, and the joint spectrum replaces the forward-only effective rank inside the coverage ratio. Channels are then kept to maximize *trainability-aware* coverage under the same budget solver.

**Why it helps.** The score now favors channels that carry both forward signal and backward gradient — the subspaces continued training actually updates. When gradients are isotropic ($C_b \propto I$) it reduces exactly to the forward-only baseline, so it can only help. Optionally, apply a BTT/TT structured factorization to the retained channels for further compression. This is the cheapest of the three ideas — a metric swap inside an existing allocator — and a clean first result.

---

## 5. Idea B: Unified Cross-Expert Selection

**Rank the channels of *all* experts in a layer together, rather than allocating a budget per expert and selecting within each.**

Treat a MoE layer's experts as columns/rows of one large concatenated expert, and run a **single Nyström selection** over the combined pool at the layer's total budget. The per-expert allocation then *falls out* of one ranking — the selection naturally keeps more channels from important experts and fewer from redundant ones, with no separate budget-allocation step and no hand-designed importance prior. Importance and compressibility are read jointly off one unified spectrum.

Two requirements make this work:

- **Router-aware statistics.** The unified kernel must reflect actual usage, so each token contributes to an expert's covariance only when the router activates that expert, weighted by its gate probability. Each token thus contributes a *sparse*, router-weighted covariance — compression is decided over experts as they are really used.
- **Nyström reconstruction across the pool.** Information in truncated columns is merged in closed form into the retained columns (as in the joint-kernel down-projection formula), now over the whole layer rather than one expert at a time.

**Why it helps.** The two-level split is a heuristic that ranks channels only *within* an expert and never compares a strong channel in a "weak" expert against a weak channel in a "strong" one. Unified selection removes that artificial barrier — strictly more flexible, and it dissolves the expert-importance prior that Attribution-Guided and RFID-MoE must design by hand.

---

## 6. Idea C: Per-Token Dynamic Active Parameters

**For each token, load a different subset of each selected expert's parameters — a heterogeneous budget applied dynamically instead of statically.**

Standard top-$k$ routing loads $k$ *full* experts per token. Instead, once the router selects the experts for a token, we keep only a subset of each selected expert's rows/columns, chosen by a **global ranking** across all the selected experts' channels. Targeting, say, 50% active-parameter reduction, the budget is spread across the selected experts by their score — a high-score expert (high router probability × high importance) activates more of its channels than a low-score one.

**Objective.** Minimize truncation error / final loss for a given active-parameter budget. This needs two ingredients:

- **Truncation error per channel** — the loss cost of dropping each rank/column/row, precomputed *offline* per expert (Nyström col/row ranking gives this). It decomposes into per-expert error, then a ranking over channels.
- **Router probability per token** — available *online* at routing time.

With both, at each token we rank all candidate channels across the selected experts by (marginal error × router weight) and keep the top ones up to budget. The challenge is that each expert's error-vs-index spectrum differs, so a fixed per-expert cutoff is wrong — the global ranking is what adapts the cut to each token.

**Efficiency.** As in Idea B, the selected experts' rows/columns are concatenated and computed together (grouped GEMM), so per-token selection adds no compute overhead — and because Cor3 decode is bandwidth-bound, loading fewer channels is a near-proportional speedup.

This is the most ambitious and most novel idea. Existing active-parameter methods (top-$p$ routing, LExI, PreMoE) act only at *whole-expert* granularity — they choose how many experts to fire. Idea C goes finer, choosing how much of each expert to load, and is the axis that most directly hits the Cor3 latency target.

---

## 7. Timeline

| Week | Task |
| ---- | ---- |
| 1 | Finalize plan. Reproduce Attribution-Guided and RFID-MoE baselines on a small MoE (OLMoE-7B-1B). Set up calibration + covariance-collection pipeline (reuse $K_{\text{joint}}$ tooling). |
| 2 | **Idea A:** swap $K_{\text{joint}}$ score into the Attribution-Guided allocator. Ablate training-aware vs. forward-only channel scores after healing. |
| 3 | **Idea B:** implement router-weighted unified Nyström selection over a whole MoE layer. Compare per-expert heterogeneous allocation against the two-level baselines. |
| 4 | **Idea C:** prototype per-token dynamic channel selection with offline error tables + online router weights. Initial active-parameter vs. accuracy curve. Identify next steps. |

---

## 8. Deliverables

1. **Training-aware Attribution-Guided (Idea A)** on Qwen3-30B-A3B at c.f.=1.5 / 2.0 — quantify the pp gain over forward-only scoring after recovery training.
2. **Unified cross-expert selection (Idea B)** — show heterogeneous per-expert allocation emerges from one ranking, and benchmark against the two-level baselines at matched budget.
3. **Per-token active-parameter reduction (Idea C)** — accuracy vs. active-parameter (memory-bandwidth) trade-off curve at ~50% active reduction, with projected Cor3 decode speedup.

---

## 9. Connection to Broader Goals

- **Macro Workstream 2 (Compression Composition).** Ideas A and B are drop-in improvements to the per-expert budget step of any composed recipe, replacing hand-picked expert ratios with a principled, training-aware allocation.
- **Macro Workstream 3 (Variable Experts per Token).** Idea C generalizes variable-expert-per-token routing from *whole experts* down to *partial experts*: instead of only choosing how many experts to fire, choose how much of each — a finer-grained lever on active parameters.
- **The joint kernel as a general tool.** The $K_{\text{joint}}$ metric underlying Idea A improves every "which parameters to keep" decision across Macro — width pruning, expert pruning, and the selections inside Ideas B and C.

---

## 10. Summary

| Idea | Contribution | Axis | Success metric |
| ---- | ------------ | ---- | -------------- |
| A | Training-aware channel score in an existing allocator | Total params | Beats forward-only scoring after healing |
| B | One unified ranking replaces two-level per-expert allocation | Total params | Matches/beats two-level baselines, no importance prior |
| C | Per-token dynamic parameter loading | Active params | Accuracy held at ~50% active reduction; Cor3 speedup |

**The core bet:** experts differ in worth and in compressibility, so budget should be allocated heterogeneously — with a metric that reflects continued training (A), a mechanism that ranks all experts together (B), and, ultimately, a budget that adapts to every token (C).
