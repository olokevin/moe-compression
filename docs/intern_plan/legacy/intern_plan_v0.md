# Elastic Training for MoE Compression: Intern Plan

## 1. Introduction

### The Problem

Compressing a large MoE model requires choosing how much to compress along each axis: MLP intermediate dimension, depth (layers), width (embedding dimension), and number of experts. Today this is done by hand — pick a config, train, evaluate, repeat. The space is combinatorial and each trial costs days of GPU time

### What We Want

An **automatic, learning-based** engine that:

1. Discovers the best compression composition config (how to allocate compression budget across axes and layers)
2. Incorporates training-aware compression (keep parameters the model actually needs for continued learning)
3. Incorporates compression-aware training (frozen backbone + efficient adaptation)

### Research Target

> *Can we train one large model and have it directly contain optimally-performing smaller models at continuous sizes — eliminating the need for separate compression + retraining per target?*

### Context

This work feeds into the Macro team's Phase 1 goal of achieving 1.5–2x parameter reduction for MoE models (Qwen3-30B-A3B for Tehama, GPT-OSS-120B/Qwen3-235B for P1) with <1pp accuracy degradation. See the Macro Plan for full hardware targets and constraints.

---

## 2. Preliminaries

### 2.1 Training-Aware Compression

**Core idea:** Standard pruning picks parameters important for the *forward pass* (inference). But if you're going to keep training after compression, you should also preserve parameters important for *gradient flow* — otherwise the compressed model can't learn effectively.

**Method:** Construct a joint kernel that weights each hidden neuron by both its forward activation magnitude and its backward gradient magnitude:

$$
K_{\text{joint}} = \bar{C}_f^{1/2}\, \bar{C}_b\, \bar{C}_f^{1/2}
$$

where:

- $C_f = Z^\top Z$ — forward hidden activation covariance (which neurons fire)
- $C_b = \delta z^\top \delta z$ — backward gradient covariance (which neurons receive gradient)
- Both are trace-normalized for scale balance

**Effect:** Neurons are ranked by $\text{diag}(K_{\text{joint}}^{-1} K_{\text{joint}})$ — those important to both forward and backward survive. Selection uses Nyström approximation on $K_{\text{joint}}$ instead of the forward-only kernel $C_f$.

**Why it matters:** At aggressive compression (c.f. > 1.3), training-aware selection retains 1–2pp more accuracy after recovery training vs. forward-only selection, because the compressed model can still learn.

### 2.2 Compression-Aware Training

**Core idea:** After compression, freeze the compressed backbone and train only lightweight adapters. This is favored because:

- The backbone already encodes good representations — don't corrupt them with full fine-tuning on limited data
- LoRA/adapters provide regularization against catastrophic forgetting
- Faster iteration: fewer trainable parameters = faster steps

**Extension:** Increase adapter capacity (higher rank, more adapter locations) to provide additional degrees of freedom without unfreezing the backbone.

### 2.3 FuRA (Full-Rank Adaptation)

**Core idea:** Standard LoRA decomposes the weight update as $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$ with $r \ll d$. This caps the update rank at $r$.

FuRA instead decomposes the *pretrained weight itself* via SVD: $W = U \Sigma V^\top$, then freezes $U, V$ and trains only the diagonal $\Sigma$. This gives:

- **Full-rank** updates (every singular direction can be adjusted)
- **$d$ trainable parameters** per matrix (vs. $2rd$ for LoRA) — extremely parameter-efficient
- A natural **importance ordering**: singular values are sorted, so the first $k$ entries correspond to the most important subspace

**Why it matters for elastic training:** FuRA's SVD decomposition provides a built-in importance ranking. If you truncate the diagonal at position $k$, you get the best rank-$k$ approximation. This is exactly the mechanism needed for nesting smaller models inside larger ones (Stage 2).

---

## 3. Stage 1: Elastic Training Engine

### 3.1 Key Ideas from Nemotron/Star Elastic

**The elastic model concept:** Train one large model that contains many smaller models as proper subsets. At any inference budget, extract the corresponding submodel — no separate compression run needed.

**How it works:**

1. **Importance ordering per axis:** For each compressible axis (MLP dim, attention heads, layers, experts), establish an importance ranking so that "smaller model = keep the top-k most important units."
2. **Elastic configurations:** A config $c$ specifies, for each layer, how much of each axis to keep. The nested constraint ensures config $c_{\text{small}} \subset c_{\text{large}}$ — the small model's parameters are always a subset of the large model's.
3. **Joint training objective:**

   $$
   \mathcal{L} = \mathcal{L}_{\text{full}} + \sum_{c \sim \mathcal{C}} \mathcal{L}_c + \lambda \cdot \text{BudgetLoss}(c)
   $$

   - $\mathcal{L}_{\text{full}}$: language modeling loss on the full model
   - $\mathcal{L}_c$: loss on sampled sub-configs (the nested smaller models)
   - BudgetLoss: penalizes configs that exceed a target parameter/latency budget
4. **Config sampling:** Each training step samples a random elastic config. Over many steps, all sizes get optimized. The full model acts as a regularizer for smaller ones (shared parameters).

**Why elastic > compress-then-train:**

- Shared parameters between sizes means the large model's capacity helps smaller models learn
- One training run amortizes cost across all target sizes
- No error accumulation from sequential compress → train → compress → train

### 3.2 Our Framework

**Architecture:** Start from a pretrained MoE model. Define elastic axes:

- **MLP intermediate dimension** (per-layer, per-expert): keep top-$k$ neurons by importance
- **Depth**: keep top-$k$ layers by importance
- **Experts**: keep top-$k$ experts per layer by importance

**Dynamic config:** At each step, sample a compression factor $\text{c.f.} \in [1.0, 2.0]$. The config allocator distributes this budget across axes and layers (learned, not uniform).

**Budget loss:** Start with parameter count as proxy:

$$
\text{BudgetLoss}(c) = \left(\frac{\text{Params}(c)}{\text{TargetParams}} - 1\right)^2
$$

Later replace with actual Cor3 latency predictor (from Macro workstream D1.2) for hardware-aware optimization.

### 3.3 Improvements Over Vanilla Elastic

1. **Training-aware importance ranking:** Replace magnitude/activation-based neuron ranking with $K_{\text{joint}}$-based ranking. The elastic ordering reflects what matters for *continued training*, not just current inference.
2. **Compression-aware training (FuRA):** During elastic training, freeze the backbone weights. Each elastic config gets a FuRA-style diagonal adapter. The importance ordering of the diagonal naturally aligns with the elastic nesting — truncating the diagonal gives the smaller model's adapter.

   **Key insight:** FuRA's sorted singular values + elastic nesting are the same idea. The elastic model's "keep top-k" operation in each layer is exactly FuRA's "keep the top-k singular values." This unifies the adaptation and compression mechanisms.

### 3.4 Method Overview (Full Flow)

```
Pretrained MoE (e.g., Qwen3-30B-A3B)
    │
    ▼
[1] Compute K_joint per layer/expert
    (forward + backward covariance on calibration data)
    │
    ▼
[2] Establish importance ordering per axis
    (neurons, layers, experts ranked by K_joint eigenvalues)
    │
    ▼
[3] Decompose weights via SVD (FuRA-style)
    Freeze U, V; train Σ (diagonal)
    │
    ▼
[4] Elastic training loop:
    For each batch:
      - Sample target c.f. ~ Uniform[1.0, 2.0]
      - Config allocator distributes budget across axes
      - Extract submodel (top-k per axis per layer)
      - Forward pass on submodel
      - Compute LM loss + budget loss
      - Update Σ diagonals + config allocator
    │
    ▼
[5] Result: one model containing nested submodels at any c.f.
    Extract at c.f.=1.5 or c.f.=2.0 by reading off the config
```

### 3.5 Expected Results

Based on Nemotron Elastic and Star Elastic results on dense models:

- At c.f.=1.5: expect <1pp degradation on reasoning benchmarks (vs. 4pp with current manual compression)
- At c.f.=2.0: expect 1–2pp degradation
- Single training run produces all intermediate sizes

The improvement over manual config search comes from: (a) joint optimization avoids error accumulation, (b) training-aware ranking preserves trainability, (c) FuRA adapters add capacity without unfreezing.

### 3.6 Timeline

| Week | Task                                                                                          |
| ---- | --------------------------------------------------------------------------------------------- |
| 1    | Finalize plan, set up codebase and compute                                                    |
| 2    | Prepare training pipeline on small MoE (OLMoE-7B-1B); implement elastic config sampling       |
| 3    | Implement training-aware compression (K_joint ranking) + FuRA decomposition; launch baselines |
| 4    | Launch elastic training; get initial results; identify improvement directions                 |

### 3.7 Deliverables

- Elastic training for Qwen3-30B-A3B at c.f.=1.5 and 2.0: comparison against manually-picked compression config baseline
- Ablation quantifying the improvement from training-aware compression vs. forward-only ranking
- Ablation quantifying compression-aware training (FuRA) vs. full fine-tuning during elastic training

---

## 4. Stage 2: All-in-One Pretraining (Paper Target)

### 4.1 The Research Question

> *Can we pretrain a large model such that it directly contains optimal smaller models at continuous sizes — without any post-hoc compression or elastic fine-tuning?*

### 4.2 Comparison of Approaches

|                   | Standard (Qwen series)                  | Elastic (Stage 1)                        | All-in-One (Proposed)              |
| ----------------- | --------------------------------------- | ---------------------------------------- | ---------------------------------- |
| Training          | Separate pretrain + post-train per size | Pretrain largest, then elastic fine-tune | Single pretrain produces all sizes |
| Parameter sharing | None (independent models)               | Nested (smaller ⊂ larger)               | Nested (smaller ⊂ larger)         |
| Model sizes       | Discrete (8B, 4B, 1B)                   | Discrete (user-specified targets)        | Continuous (any c.f.)              |
| Cost              | N × full training                      | 1 × full training + 1 × elastic FT     | 1 × full training                 |
| Quality ceiling   | Best (each model fully optimized)       | Good (shared params help smaller)        | ? (the research question)          |

**The gap:** Elastic training (Stage 1) still requires a pretrained model + expensive fine-tuning phase. If we could build the elastic structure *into pretraining itself*, we'd get all sizes for free at the end of one training run.

### 4.3 Core Idea

**Insight:** If training naturally pushes the most important information into the most important parameters (highest singular values / most-used neurons), then a well-trained model already has an approximate elastic structure. The question is whether we can *strengthen* this tendency during pretraining without sacrificing the full model's quality.

**Mechanism:**

1. **FuRA decomposition as pretraining parameterization:** Instead of training standard weight matrices $W$, train in the SVD basis: $W = U \Sigma V^\top$ with $U, V$ on the Stiefel manifold (orthogonal) and $\Sigma$ diagonal. The sorted singular values create a natural importance ordering.
2. **Spectral preconditioning:** Bias the training dynamics so that:

   - High-importance information (frequent patterns, core knowledge) concentrates in the top singular components
   - Low-importance information (rare patterns, edge cases) goes to the tail

   This can be achieved by applying a decaying learning rate schedule *per singular component* — top components learn faster and stabilize first, tail components continue adapting.
3. **Elastic regularizer during pretraining:**

   $$
   \mathcal{L} = \mathcal{L}_{\text{LM}} + \alpha \sum_{c \sim \mathcal{C}} \mathcal{L}_{\text{LM}}^{(c)}
   $$

   where $\mathcal{L}_{\text{LM}}^{(c)}$ is the LM loss evaluated on the truncated model at config $c$. This explicitly optimizes submodel quality during pretraining.

### 4.4 Why This Might Work

The pretrained model's spectrum is already approximately the right importance ordering (the Eckart-Young theorem guarantees SVD gives the best rank-$k$ approximation). What we add is:

- **Explicit optimization of submodels** (not just hoping truncation works)
- **Spectral preconditioning** (actively pushing information into the right singular components)
- **Continuous rather than discrete** sizes (any truncation point yields a valid model)

### 4.5 Risks and Open Questions

1. **Cross-layer importance ranking:** Within a layer, SVD gives a clear ordering. Across layers, how do we decide "keep 90% of layer 5 but only 70% of layer 20"? Need a global budget allocation mechanism (possibly learned, possibly based on layer-wise Fisher information).
2. **Pretraining overhead:** Evaluating submodel loss at each step adds compute. Mitigation: sample subconfigs stochastically, evaluate on a subset of the batch.
3. **Capacity interference:** Forcing nesting might reduce the full model's quality (the large model sacrifices some capacity to make submodels work). Need to quantify: is the full model degradation acceptable (< 0.5pp)?
4. **Comparison to independent training:** Even if all-in-one works, does it match independently-trained models at each size? Or is there an inherent capacity tax for nesting?

### 4.6 Experimental Plan

1. **Proof-of-concept on small scale:** Train OLMoE-7B-1B from scratch with all-in-one pretraining. Compare submodel quality at 4B, 2B, 1B against independently-trained baselines.
2. **Ablations:**

   - Spectral preconditioning on vs. off
   - Elastic regularizer strength ($\alpha$)
   - Number of sampled subconfigs per step
3. **Scaling:** If PoC works, apply to Qwen3-30B-A3B scale.

---

## 5. Connections to Broader Macro Goals

This work directly addresses Macro workstream 2 (Compression Composition):

- Stage 1 replaces manual composition search with learned elastic optimization
- Stage 2 eliminates the compression step entirely by building it into training

The training-aware kernel $K_{\text{joint}}$ developed here also improves standalone compression tools (Nyström, width pruning) used across all Macro workstreams.

The long-term vision aligns with ACE (Automated Compression Engine): instead of searching over discrete compression pathways, train a model that continuously interpolates between sizes. ACE becomes a lookup into the elastic model rather than an optimization problem.

---

## 6. Summary

| Stage | Goal                                   | Method                                           | Timeline    | Deliverable                                     |
| ----- | -------------------------------------- | ------------------------------------------------ | ----------- | ----------------------------------------------- |
| 1     | Automatic compression config discovery | Elastic training + training-aware ranking + FuRA | Weeks 1–4  | Elastic Qwen3-30B-A3B at c.f.=1.5/2.0           |
| 2     | One model contains all sizes           | All-in-one pretraining with spectral structure   | Weeks 5–12 | Paper: capacity limits of nested elastic models |
