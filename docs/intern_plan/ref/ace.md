### The Opportunity

We have built, and continue to build, a powerful compression toolkit in Percipio. Across Hologram, Macro, and Micro, we deliver methods for parameter compression, precision reduction, and distillation. Our answer to *"what methods exist?"* grows richer every quarter.

But a growing toolkit without a selection mechanism becomes harder to use, not easier. Today, when a new compression goal arrives, we face the same questions every time:

* Which subset of our methods applies to *this* model on *this* hardware target?
* In what order should they be composed?
* Where are the limits, and at what point does a method stop helping and start hurting?

We currently answer these questions through trial and error, relying on the intuition of the people who built the methods. That worked when the toolkit was small and goals were few. It does not scale.

### The Problem ACE Would Solve

**Every method we develop is only as valuable as our ability to deploy it at the right moment.** A compression technique that sits unused because no one knew it applied is wasted R&D. A technique applied in the wrong regime, because we didn't understand its limits, burns experimentation time and erodes trust in the platform.

Concrete example: methods we developed for the 70B parameter compression goal did not transfer to Alexa+ requirements. The model architecture was different, the inference constraints were different, and the composition order mattered. We discovered this *after* weeks of experimentation, not before.

This wasn't a failure of our methods. It was a failure of  **method selection and composition** , a capability we don't yet have in Percipio. ACE would fill this gap.

### What ACE Would Do

ACE would be a decision layer that sits on top of our compression streams and answers: *"Given this source model and this target budget, what is the best path through our toolkit?"*

Three components:

1. **Inference Aware Feasibility Profiling.** ACE would profile the actual inference engine to determine which final architectures meet the latency and throughput target, rather than relying on FLOP estimates alone.
2. **Compression Pathway Search.** ACE would explore combinations of our methods (pruning, quantization, distillation, in various orders and intensities) rather than requiring us to evaluate each in isolation.
3. **Composed Scaling Laws.** ACE would establish predictive relationships for how our methods interact, so we can estimate quality *before* committing to full training. This is what would turn weeks of trial and error into hours of informed search.

### Why ACE Would Make Percipio More Valuable

| Without ACE                                        | With ACE                                   |
| -------------------------------------------------- | ------------------------------------------ |
| Each new goal triggers ad hoc exploration          | Each new goal becomes a query to ACE       |
| Method developers must also be method selectors    | Methods are automatically matched to goals |
| Compositions discovered by accident                | Compositions discovered by search          |
| New methods add complexity                         | New methods add searchable capability      |
| Knowledge of "when to use" lives in people's heads | Knowledge is encoded and reusable          |

As we continue to ship new methods across our streams, ACE would ensure every new addition is *immediately* available for selection and composition. Without it, a larger toolkit paradoxically becomes harder to wield. With it, every new method we build multiplies Percipio's reach.

### Why Now

1. **Our toolkit is large enough that intuition doesn't scale.** Three streams, multiple methods each, composable in various orders. The combinatorial space exceeds what any of us can navigate by feel.
2. **New model architectures keep arriving, and they differ in fundamental ways.** Qwen3 235B is deep (94 layers) while Qwen3.5 397B is wide and shallow (60 layers). Gemma4 introduces a different balance of architecture choices. Attention mechanisms vary across models: grouped query attention in Llama, multi-latent attention in DeepSeek. A compression strategy that works for one architecture does not automatically transfer to another. Each new model resets our intuition. A systematic framework would not reset.
3. **Product timelines are compressing.** Alexa+ and future product goals impose hard deadlines. We cannot afford multi week exploration cycles for every new target.

### What Success Would Look Like

* A new compression goal arrives. ACE returns a recommended pathway through our toolkit within days, not weeks, with predicted quality.
* A new method lands from any of our streams. ACE characterizes it and makes it available for composition without requiring everyone to manually update their mental model.
* Percipio evolves from a *collection of compression methods* into a  *compression platform with built in navigation* . ACE is the component that would make that leap possible.` `

```
-----
```

## Executive Summary

Our compression toolkit has grown significantly — from Nyström and width pruning to MoFication, layer pruning, and expert pruning this year — and will likely expand further as product goals (e.g., Alexa+) impose strict latency and throughput constraints. Today, we evaluate each compression strategy in isolation and only consider combinations after the fact, leading to two problems:

1. **Delays** — sequential trial-and-error across techniques wastes weeks of experimentation before arriving at a viable configuration.
2. **False generalization** — a technique that works well on one architecture may fail on another. For example, layer pruning was effective for Qwen3-235B-22A because that model is deep (94 layers), but assuming the same approach transfers to Qwen3.5-397B-A17B is incorrect — Qwen3.5 is a wider, shallower architecture (60 layers) where depth removal is far more destructive.

This proposal introduces a systematic framework that, given any source model and a target inference budget, identifies the optimal compressed architecture by:

* Profiling the inference engine to determine which final architectures are feasible (rather than relying on FLOP estimates)
* Efficiently searching over both architecture shapes and compression pathways
* Establishing scaling laws for composed compression techniques to predict quality without exhaustive training

The goal is to replace ad-hoc, technique-first exploration with a principled, architecture-first pipeline that generalizes across models, techniques, and hardware targets.

## Problem Statement

Given a source model **M** (dense or MoE), an inference engine **E** (e.g., vLLM, Core3 etc), and a set of compression techniques  **C** , we want to find the compressed model that maximizes quality subject to hard inference constraints (i.e., throughput ≥ R, end-to-end latency ≤ T).

The challenge is that multiple compressed architectures satisfy the same inference constraint, and each architecture can be reached via multiple compression pathways — yielding a combinatorial search space where evaluating any single candidate requires expensive recovery training.

**This proposal investigates how to efficiently prune this (architecture × pathway) search space to identify the best-performing compression configuration without exhaustive training.**

## Motivation

| Dimension                 | Current Practice                   | Proposed Approach                                     |
| ------------------------- | ---------------------------------- | ----------------------------------------------------- |
| Technique selection       | Heuristic (FLOPs, parameter count) | Empirical profiling on target inference engine        |
| What is optimized         | Parameter count / FLOPs as proxy   | Actual throughput and E2E latency on target hardware  |
| Search space handling     | Human intuition narrows candidates | Systematic elimination via profiling + cheap proxies  |
| Composition of techniques | Studied in isolation               | Studied as sequential pipelines with ordering effects |

## Key Insight

The search space has two orthogonal dimensions:

1. **Architecture dimension** — What final model shapes (layers, width, dense vs MoE, expert count, top-k) meet the inference constraints on engine E?
2. **Pathway dimension** — Given a feasible final shape, which sequence of compression transformations from M preserves the most quality after recovery training?

|

```
Source Model M ──→ [Transform₁ → Transform₂ → ... → Transformₖ] ──→ Architecture Aᵢ ──→ Engine E
                   └─────────── Pathway Pⱼ ──────────────────┘        └── Must meet R, T ─┘

Candidates = { (Aᵢ, Pⱼ) | Aᵢ ∈ Feasible(E, R, T) and Pⱼ ∈ Reachable(M → Aᵢ) }
```

|  |
| - |

The same final architecture can be reached via different paths with vastly different quality outcomes. For example, targeting a 48-layer MoE with 32 experts:

* **Path A** : M(dense, 64L) → layer prune(48L) → MoEfication (32 experts)
* **Path B** : M(dense, 64L) → MoEfication (64 experts) → layer prune(48L) → expert prune(32 experts)
* **Path C** : M(dense, 64L) → Width prune + layer prune(50L) → MoEfication (32 experts)

All three produce the same inference profile, but their quality ceilings after recovery training may differ dramatically.

## Current Compression Techniques Under Consideration

| Technique               | Transformation                     | Inference Gain Mechanism                       |
| ----------------------- | ---------------------------------- | ---------------------------------------------- |
| ShotGPT (Layer Pruning) | Remove transformer blocks          | Fewer sequential ops → lower latency          |
| Nyström                | Reduce FFN dimensions              | Smaller matrices → less compute per FFN layer |
| Width pruning           | Reduce hidden dimensions           | Smaller matrices → less compute per layer     |
| DOT-MoE (MoEfication)   | Convert dense layers to sparse MoE | Sparse activation → higher capacity per FLOP  |
| REAP (Expert pruning    | Remove experts from MoE            | Less memory bandwidth, smaller working set     |

## Research Questions

### RQ1: Can we predict quality rank-order without full training?

Given N feasible (architecture, pathway) candidates, does a cheap proxy correctly rank their post-recovery quality (measured after days of training)?

Candidate proxies:

* Zero-shot calibration perplexity
* Information-theoretic loss (KL divergence from source)
* Gradient-based importance / Fisher information
* Short micro-training trajectory + extrapolation

### RQ2: Is the search space decomposable?

Can we score architectures and pathways independently, or does the interaction dominate?

### RQ3: Do dominance relationships exist across technique families?

* Does MoE always dominate Dense at the same throughput target?
* Do such rules structurally eliminate large portions of the space?

### RQ4: Do compression scaling laws extend to sequential compositions?

Prior work (Frantar et al., 2025; Panferov et al., 2025) shows compression acts as a multiplicative modifier on effective parameters — but only for individual techniques. Does this hold when chaining layer_prune → MoEfication → expert_prune? Or do compositions exhibit super-additive quality degradation?
