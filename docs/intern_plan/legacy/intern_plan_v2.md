# Elastic Training for MoE Compression: Intern Plan

## 1. Introduction

### Macro MoE Compression: The Mission

The Macro team's goal is to enable large MoE models to run on Cor3 hardware — both P1 (cloud) and Tehama (edge) — through structural parameter reduction (pruning layers, width, MLP dimensions, experts). The target is 2x for Tehama (e.g., Qwen3-30B-A3B), all within 1pp accuracy degradation. Current best manual recipes give ~4pp drop at 1.5x on Qwen3-30B-A3B. Closing this gap requires solving three critical problems.

### Problem 1: Compression Configuration

Compressing a large MoE model means choosing how much to reduce along each axis:

- **MLP intermediate dimension** (per-layer, per-expert)
- **Depth** (number of layers)
- **Width** (embedding/hidden dimension)

The space is combinatorial: a 48-layer MoE with 64 experts per layer and three granularity levels per axis yields thousands of valid configurations. Today this is navigated by hand — pick a config, compress, train, evaluate, repeat. Each trial costs days of GPU time. We need automatic, learning-based config discovery.

### Problem 2: Training-Aware Compression (What to Preserve)

Standard pruning preserves inference-time importance — it keeps parameters that minimize output error on calibration data. But in the Macro pipeline, the compressed model will keep training. Minimizing forward error does not guarantee the compressed model can *learn* — it may discard subspaces that carry little activation but large gradient, leaving the model unable to recover during fine-tuning. We need compression that preserves what matters for continued training: the subspaces that gradients flow through.

### Problem 3: Compression-Aware Training (How to Recover)

After compression, the model needs recovery training to regain lost accuracy. Full fine-tuning risks corrupting the retained backbone — the compressed model already holds good representations, and unrestricted updates on limited recovery data invite catastrophic forgetting. We need training strategies that recover performance without destabilizing the compressed backbone.

### Goal

The intern plan contains two phases:

Phase 1

1. An automatic, learning-based engine to discover the best compression config — replacing manual search with joint optimization (addresses Problem 1).
2. Incorporate training-aware compression and compression-aware training as improvements over vanilla elastic methods (addresses Problems 2 and 3).

Phase 2 (target publication)

Standing upon the baseline built in phase 1, target on a broader, more challenging problem setup:

*Can we train one large model and have it directly contain optimally-performing models at all scales?*

---

## 2. Preliminaries

### 2.1 Training-Aware Compression

**The limitation of prior compression.** Existing methods (SVD truncation, Nyström neuron selection) minimize forward truncation error — they find $\hat{W}$ closest to $W$ in the sense of reconstructing the layer's output on calibration data. This is the right objective if the compressed model is used as-is for inference. But in our pipeline the compressed model will keep training. Minimizing forward error does not guarantee the compressed model can *learn* — it may discard subspaces that carry little activation but large gradient, leaving the model unable to recover during fine-tuning.

**Two-sided whitening for linear layers.** The idea: weight the compression error by both input statistics and output gradient statistics. For a linear layer $y = xW$, the task-loss change from replacing $W$ with $\hat{W}$ is (second-order Taylor):

$$
\Delta \mathcal{L} \approx \| C_g^{1/2} (W - \hat{W}) C_x^{1/2} \|_F^2
$$

where $C_x = \mathbb{E}[xx^\top]$ (input covariance) and $C_g = \mathbb{E}[\delta y\, \delta y^\top]$ (output gradient covariance). This is a K-FAC approximation to the Hessian: $H_W \approx C_x \otimes C_g$.

**What it preserves:** minimizing this two-sided objective discards the subspace with lowest *loss curvature* — directions the loss landscape is flat along. These are precisely the directions that training does not actively update. The surviving high-curvature directions are those that both carry forward signal ($C_x$) and receive backward gradient ($C_g$).

**Extension to Nyström MLP compression.** For a gated MLP ($z = u \odot \phi(g)$, $y = zW_d$), the compressed object is not a single weight matrix but the hidden-neuron transport core. The two-sided whitening idea applies to this core:

- **Forward covariance** $C_f = Z^\top Z$ — which hidden neurons activate (analogous to $C_x$)
- **Backward covariance** $C_b = \delta z^\top \delta z$ — which hidden neurons receive gradient flow into $W_u, W_g$ (analogous to $C_g$), where $\delta z = \delta y\, W_d^\top$

The joint kernel that captures both:

$$
K_{\text{joint}} = \bar{C}_f^{1/2}\, \bar{C}_b\, \bar{C}_f^{1/2} + \lambda I
$$

(trace-normalized: $\bar{C}_f = C_f / \text{tr}(C_f)$, $\bar{C}_b = C_b / \text{tr}(C_b)$). This is the PSD analogue of $C_g^{1/2} W C_x^{1/2}$ — its eigenvalues are the squared singular values of the two-sided-whitened hidden core $C_b^{1/2} C_f^{1/2}$.

**Application to Nyström selection.** Replace the forward-only kernel $C_f$ with $K_{\text{joint}}$ everywhere in the standard Nyström pipeline:

- **Neuron scoring:** $\text{score}_i = \text{diag}\big((K_{\text{joint}}+\lambda I)^{-1} K_{\text{joint}}\big)$ — keep the top-$k$ neurons
- **Down-projection reconstruction:** $\hat{W}_D = (S^\top K_{\text{joint}} S)^{+}\, S^\top K_{\text{joint}}\, W_D$

When $C_b \propto I$ (isotropic gradients), $K_{\text{joint}} \propto C_f$ and this reduces exactly to standard forward-only Nyström. The training-aware kernel only departs from vanilla when gradient structure is non-uniform — exactly the regime where forward-only selection fails.

**Impact:** At aggressive compression (c.f. > 1.3), training-aware selection retains 1–2pp more accuracy after recovery training vs. forward-only, because the compressed model preserves its ability to learn.

### 2.2 Compression-Aware Training

**Developed by Evgeny.** After compression, freeze the compressed backbone and train only lightweight adapters (LoRA) outperforms directly train the backbone.

- The lost (pruned) parameters is a low-rank residual -> LoRA applies inductive bias towards recovering it
- Avoid drifting off the pretrained manifold of the backbone.
- LoRA confines update to stable subspace

**My addition:**

---

## 3. Stage 1: Elastic Training Engine

### 3.1 The Elastic Model Concept

The key insight from Nemotron Elastic and Star Elastic: instead of compressing *after* training, train one model that *contains* many smaller models as nested subsets.

**Why elastic training beats compress-then-train:**

| Compress-then-train                                     | Elastic training                                               |
| ------------------------------------------------------- | -------------------------------------------------------------- |
| Compress once → commit to one size                     | One model → extract any size                                  |
| Errors accumulate through the pipeline                  | Joint optimization avoids error buildup                        |
| Each size trained independently — no knowledge sharing | Shared parameters: large model's capacity helps smaller models |
| $N$ compression runs for $N$ sizes                  | 1 training run for all sizes                                   |

**The mechanism is simple:** establish an importance ordering over parameters per layer, define "smaller model = keep the top-$k$ most important." Train the full model and simultaneously optimize the submodels by sampling random configs at each step.


**Why Elastic Training Outperforms Select-then-Train**

To find the best performing config, one solution is select-then-train: first propose multiple candidates, then use a short training for all candidates, and continue full training on the best performing candidate. Other automatic selection criteria could also be developed.

Elastic training has two core advantages over it:

1. **No wasted compute.** In select-then-train, you train $N$ candidate configs independently, pick the winner, and discard the rest — all compute spent on losers is wasted. The candidate space is combinatorial (per-layer × per-axis × per-expert), so you can only afford to sample a tiny fraction. Elastic training sidesteps this: because all configs share parameters (smaller ⊂ larger), every training token improves *all* configs simultaneously. No tokens are wasted on a config that won't be selected.
2. **Configs improve each other.** The router samples a different config each step, but the weight updates accumulate in the shared pool. Tokens spent training config A update the same top-$k$ weights that config B will later use. The final selected config benefits from *all* prior training — not just the steps where it happened to be sampled. This is impossible when candidates are trained in isolation.

### 3.2 How Elastic Training Works (Nemotron Elastic Flow)

![How elastic training works](fig/elastic_training_concept.png)

The full model sits at the bottom, shown forwarding in one sampled config: gray marks masked parameters/layers along the three compressible axes — **depth** (skipped layers use a residual bypass), **width** (hidden dim), and the **MoE MLP intermediate dim** (per-expert). On top, the per-axis **router** turns a budget target into that config (the kept size on each axis). On the right, the student's logits and the teacher (full model, no masks) feed the **loss** = KD(teacher ‖ student) + budget loss. The dashed red paths show the loss flowing back to update *both* the model weights and the router jointly.

```
Pretrained LLM
    │
    ▼
[1] IMPORTANCE ORDERING
    Per axis (MLP neurons, attention heads, layers, experts):
    rank units by activation-based importance.
    This defines the nesting order — any submodel is a prefix.
    (e.g., neurons ranked [#2047, #15, #3891, ...]; keeping 2048 = take top-2048)
    │
    ▼
[2] LEARNABLE ROUTER → CONFIG
    One tiny 2-layer MLP per axis. Input: one-hot budget target.
    Output: how much to keep per axis (per-layer in heterogeneous mode).
    Gumbel-Softmax makes the discrete selection differentiable.
    (e.g., target c.f.=1.5 → router outputs: depth=42/48, FFN=3072/4096, experts=6/8)
    │
    ▼
[3] MASKED SUBMODEL FORWARD
    Apply binary masks to activations following the importance ordering.
    Masked-out neurons/layers/experts contribute zero.
    Skipped layers use residual bypass: y_{l+1} = y_l.
    (e.g., at c.f.=1.5: run 42 layers, each FFN uses 3072 of 4096 neurons, 6 of 8 experts)
    │
    ▼
[4] JOINT TRAINING
    For each batch:
      • Sample budget target → router produces config
      • Forward full model (teacher) + masked submodel (student)
      • L = KD(teacher‖student) + α·L_CE(teacher) + ‖Cost(c) − Target‖
      • Backprop updates model weights + router parameters jointly
    │
    ▼
[5] ZERO-SHOT EXTRACTION
    Query router with desired budget → get config → slice checkpoint.
    Standalone smaller model, no fine-tuning needed.
```

**Key detail: the router.** The config is not hand-specified — it is *learned*. Each router is a 2-layer MLP ($\sim$2% extra parameters) that takes a budget target and outputs per-layer dimension choices via Gumbel-Softmax. The router is updated end-to-end: it receives gradient from both the distillation loss (what config gives best accuracy?) and the budget loss (does the config hit the target size?). This is what enables heterogeneous per-layer allocation — e.g., the router may discover that later layers tolerate more FFN compression while early layers need width.

**Budget loss:** $\mathcal{L}_{\text{router}} = \| \text{Cost}(c) - \hat{\mathcal{C}} \|$, where Cost can be parameter count or actual hardware latency. S

### 3.3 detail version

**Step 1: Importance Ordering (one-time, before training)**

For each compressible axis, rank all units by a forward-pass importance metric:

- **FFN neurons:** $\text{score}(i) = \sum_{B,L} |X W_1^{(i)}|$ (activation magnitude of the $i$-th intermediate neuron)
- **Attention heads:** aggregate head output norms
- **Layers:** normalized MSE contribution — how much each layer's removal increases output error
- **Experts (MoE):** router load or activation magnitude per expert

This ranking is fixed and defines the nesting order: neuron #1 is always kept before neuron #2, etc. Any submodel is a *prefix* of this ordering.

**Step 2: Learnable Router (the config generator)**

The elastic config is *not* hand-specified — it is the output of a small learned router network. There is one router per axis (depth, FFN width, attention heads, embedding width). Each router is a tiny 2-layer MLP:

$$
h = \text{LeakyReLU}(W_1 \cdot \mathbf{e}_{\text{budget}} + b_1), \quad z = W_2 \cdot h + b_2
$$

**Input:** a one-hot vector $\mathbf{e}_{\text{budget}}$ indicating the target budget (e.g., "6B" or "9B").

**Output:** logits over the set of possible dimension choices for that axis. In heterogeneous mode, the output is per-layer: the FFN router outputs $N_{\text{layers}} \times |\mathcal{F}|$ logits, one choice per layer.

The logits are passed through **Gumbel-Softmax** (temperature $\tau$) to produce differentiable discrete selections — enabling gradient flow back to the router during training.

![Elastic router for the MoE MLP dimension](fig/elastic_router_moe_mlp.png)

The figure traces the mechanism for a single axis (MoE MLP intermediate dim). The router is a 2-layer MLP whose 3 logits pass through a Gumbel-Softmax; each output dim maps to one discrete dim *level* (2048 / 2560 / 3072) and the Gumbel-Softmax emits a probability per level — e.g. {2048: 0.2, 2560: 0.3, 3072: 0.5}. Here the 3072 level is selected with prob $z=0.5$. In the LLM forward (a stack of decoder layers; the selected layer is expanded to show its Self-Attn and MoE sub-blocks, and the MoE block is zoomed to the right), each expert keeps only its top-3072 intermediate neurons — but **each expert keeps a different set** (different rows of $W_{\text{up}}$ / columns of $W_{\text{down}}$) because each has its own importance ranking. The MoE output is scaled by $z$, so $z$ sits on the backprop path: the final loss — KD$(\text{teacher}\,\|\,z\cdot\text{student}) + $ budget loss — updates *both* the LLM weights and the router parameters jointly (the router gradient flows through $z$).

*Example config output (target c.f.=1.5 on a 30B MoE with 48 layers):*

| Axis               | Full size    | Router's selected size             | Effective c.f. on this axis |

| ------------------ | ------------ | ---------------------------------- | --------------------------- |

| Depth              | 48 layers    | 42 layers (skip 6 least-important) | 1.14×                      |

| FFN (layer 0–20)  | 4096 neurons | 3072 neurons                       | 1.33×                      |

| FFN (layer 21–47) | 4096 neurons | 2560 neurons                       | 1.60×                      |

| Experts            | 8 per layer  | 6 per layer                        | 1.33×                      |

The router discovers that later layers tolerate more FFN compression while early layers need width — this kind of heterogeneous allocation is what manual search cannot find efficiently.

**Step 3: Masked Submodel Forward Pass**

Given the router's config, the submodel forward pass works by **masking out** the least-important components at each layer. No architecture change — just binary masks applied to activations:

For FFN at layer $\ell$ with router selecting $k$ neurons:

$$
\text{mask}_\ell = [\underbrace{1, 1, \ldots, 1}_{k}, \underbrace{0, 0, \ldots, 0}_{d_{\text{int}} - k}] \quad \text{(ordered by importance ranking)}
$$

$$
h_\ell = (W_1 \cdot x) \odot \text{mask}_\ell, \quad y_\ell = W_2 \cdot \sigma(h_\ell)
$$

For depth, skipped layers use residual bypass:

$$
y^{(\ell+1)} = \begin{cases} y^{(\ell)} + \mathcal{L}_\ell(y^{(\ell)}) & \text{if } \gamma_\ell = 1 \text{ (layer active)} \\ y^{(\ell)} & \text{if } \gamma_\ell = 0 \text{ (layer skipped)} \end{cases}
$$

*Example forward pass at c.f.=1.5:* The 48-layer model runs only 42 layers. At each active layer, the FFN computes with a masked intermediate dimension (e.g., 3072 of 4096 neurons active). Only 6 of 8 experts are evaluated per token. The masked-out neurons/layers/experts contribute zero — their weights exist in the checkpoint but don't participate in this submodel's computation.

**Step 4: Joint Optimization**

Each training step:

1. Sample a target budget (e.g., "6B" or "9B") — uniform sampling in Stage 1, curriculum-weighted in Stage 2
2. Router produces config for that budget
3. Forward the full model (teacher) + forward the masked submodel (student) on the same batch
4. Compute losses:

$$
\mathcal{L} = \underbrace{D_{\text{KL}}(p_{\text{teacher}} \| p_{\text{student}})}_{\text{distillation: submodel matches teacher}} + \underbrace{\alpha \cdot \mathcal{L}_{\text{CE}}(\text{teacher})}_{\text{keep teacher sharp}} + \underbrace{\| \text{Cost}(c) - \text{Target} \|}_{\text{router loss: hit the budget}}
$$

5. Backprop through everything: model weights get gradient from the distillation loss; router parameters get gradient from both distillation loss (via Gumbel-Softmax) and budget loss

The router learns which config gives the best accuracy *at* the target budget. The model weights adapt to work well under multiple configs simultaneously.

**Step 5: Zero-Shot Extraction**

After training, extract any target-budget submodel by:

1. Query the router with the desired budget → get the config (which layers, how many neurons per layer, etc.)
2. Slice the full checkpoint: permanently remove masked-out weights
3. The result is a standalone smaller model — no further fine-tuning needed

**Budget loss** (Star Elastic): shapes the router toward feasible configs:

$$
\mathcal{L}_{\text{router}} = \| \text{Cost}(c) - \hat{\mathcal{C}} \|
$$

where Cost can be parameter count (portable) or actual latency on target hardware. Star Elastic shows hardware-aware cost gives 5–10% better configs than parameter-count proxy.

### 3.4 Results from the Literature

Nemotron Elastic results on Qwen2.5-32B (reasoning benchmarks, post-elastic training):

| Method                    | Budget (active params) | AIME'24 | MATH-500 | LiveCodeBench |
| ------------------------- | ---------------------- | ------- | -------- | ------------- |
| Full model                | 32B                    | 72.7    | 95.4     | 52.4          |
| Uniform pruning + retrain | 16B                    | 51.3    | 88.2     | 38.1          |
| Nemotron Elastic          | 16B                    | 63.3    | 92.8     | 45.2          |

Key takeaway: elastic training at 50% compression (c.f.=2.0) loses only 6–7pp on hard reasoning tasks, vs. 20+pp for naive uniform pruning. At c.f.=1.5, degradation is typically <2pp.

### 3.5 Our Improvements Over Vanilla Elastic

**Improvement 1: Training-aware importance ranking.**

Nemotron/Star Elastic use activation magnitude or gradient magnitude separately to rank neurons. We replace this with the joint kernel $K_{\text{joint}}$ ranking in Nystrom, which captures which neurons are important for *both* forward and backward. The elastic ordering reflects what matters for continued training — not just current inference accuracy.

**Improvement 2: Compression-aware training (frozen backbone + LoRA).**

During elastic training, freeze the compressed backbone and train lightweight LoRA adapters for recovery. This prevents drifting off the pretrained manifold while still recovering lost accuracy.

### 3.6 Full Method Flow

```
Pretrained MoE (e.g., Qwen3-30B-A3B)
    │
    ▼
[1] Compute K_joint per layer/expert
    (calibration pass: collect C_f and C_b)
    │
    ▼
[2] Elastic training loop:
    For each batch:
      • Sample target c.f. ~ Uniform[1.0, 2.0]
      • Per-layer budget allocation (learned allocator)
      • Extract submodel: mask top-k(layer) neurons per importance ordering
      • Forward full model (teacher) + masked submodel (student)
      • L = KD(teacher‖student) + α·L_CE(teacher) + L_budget
      • Update model weights + allocator parameters
    │
    ▼
[3] Extraction:
    Given target c.f., read optimal config from allocator,
    slice checkpoint to keep only top-k neurons/layers/experts.
```

### 3.7 Timeline

| Week | Task                                                                                                                       |
| ---- | -------------------------------------------------------------------------------------------------------------------------- |
| 1    | Finalize plan. Set up codebase, compute allocation, data pipeline.                                                         |
| 2    | Implement elastic config sampling + training loop on small model (OLMoE-7B-1B). Standalone experiments outside Percipio 2. |
| 3    | Integrate training-aware compression (K_joint). Launch baseline runs (vanilla elastic vs. training-aware).                 |
| 4    | Full elastic training on target model. Initial results → identify gaps and next improvement steps.                        |

### 3.8 Deliverables

1. **Elastic Qwen3-30B-A3B** at c.f.=1.5 and 2.0: benchmark against current manual compression baseline (which shows ~4pp drop at 1.5x).
2. **Ablation: training-aware vs. forward-only ranking** — quantify pp improvement from $K_{\text{joint}}$ initialization.

---

## 4. Stage 2: All-in-One Pretraining (Paper Target)

### 4.1 The Research Question

> *Can we pretrain a large model once such that it directly contains optimal smaller models at continuous sizes — no post-hoc compression, no elastic fine-tuning?*

Stage 1 still requires: pretrain model → run elastic fine-tuning. Stage 2 asks: what if we bake the elastic structure into pretraining itself?

### 4.2 Comparison of Paradigms

|                      | Standard (Qwen series)         | Elastic (Stage 1)                             | All-in-One (Proposed)        |
| -------------------- | ------------------------------ | --------------------------------------------- | ---------------------------- |
| **Training**   | Pretrain + post-train per size | Pretrain largest → elastic FT                | Single pretrain produces all |
| **Parameters** | Independent per size           | Nested (small ⊂ large)                       | Nested (small ⊂ large)      |
| **Sizes**      | Discrete (8B, 4B, 1B)          | Discrete (user-specified)                     | Continuous (any c.f.)        |
| **Cost**       | $N \times$ full training     | $1\times$ pretrain + $1\times$ elastic FT | $1\times$ pretrain         |
| **Quality**    | Best (fully optimized each)    | Good (shared params help)                     | ? (the research question)    |

**Training cost comparison** (approximate GPU-hours for a Qwen3-30B-A3B class model, targeting 3 sizes: 8B, 6B, 4B):

![Paradigm cost comparison](fig/paradigm_cost_comparison.png)

The Standard paradigm requires independent pretrain + post-train for each model size (3×50k pretrain + 3×30k post-train = 240k GPU-hrs). Elastic training pretrains only the largest model, then runs a single elastic fine-tuning pass to produce all sizes (50k + 30k + 10k = 90k). All-in-One bakes the elastic structure into pretraining itself, eliminating the separate elastic FT step (55k + 30k = 85k).

### 4.3 Core Idea

**Parameterization:** Decompose each weight as $W = U \Sigma V^\top$ and train *all three factors* ($U, \Sigma, V$) end-to-end.

**Why this produces elastic structure:** $\Sigma$ sits on the backpropagation path. Since $\sigma_i$ directly scales the $i$-th component's contribution to both the forward output and the backward gradient, SGD naturally routes more important information into higher-$\sigma$ components — the loss is more sensitive to them, so they attract larger updates. The spectral basis acts as an importance-aware preconditioner *during training*, not just at extraction time.

**Elastic extraction:** Truncate at position $k$ → keep the top-$k$ singular components → get a smaller model. Because training dynamics already concentrate important information into the top components, truncation preserves quality.

**Elastic regularizer:** Explicitly optimize submodel quality alongside the full model:

$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \alpha \sum_{c \sim \mathcal{C}} \mathcal{L}_{\text{task}}^{(c)}
$$

This strengthens the natural spectral concentration — ensures each prefix $(\sigma_1, \ldots, \sigma_k)$ is a good model on its own.

### 4.4 Training Scope

**Start with post-training (SFT/RL on a pretrained checkpoint).** Pretraining from scratch is expensive and risks conflating the elastic structure question with pre-training recipe issues. Post-training is sufficient to demonstrate the core claim (spectral concentration during training)

**Later:** lightweight pretraining (small model, short schedule) to show the idea extends to training from scratch. Full-scale pretraining only if post-training results are strong.

### 4.5 Risks and Challenges

1. **Cross-layer budget allocation.** Within a layer, SVD gives a clear ordering. Across layers: how to decide "keep 90% of layer 5 vs. 70% of layer 20"? Use the same learned router from Stage 1.
2. **Capacity interference.** Forcing nesting may tax the full model. Acceptable if < 0.5pp degradation.
3. **Overhead.** Submodel loss evaluation adds compute. Mitigation: sample 1 subconfig per step on a batch subset.

### 4.6 Experimental Plan

1. **Post-training PoC (Weeks 5–8):** Take pretrained OLMoE-7B-1B, run SFT with spectral parameterization ($U, \Sigma, V$ all trainable) + elastic regularizer. Extract submodels at various c.f. Compare against: (a) standard SFT + post-hoc compression, (b) Stage 1 elastic fine-tuning.
2. **Ablations (Weeks 7–9):**
   - With vs. without elastic regularizer
   - $\alpha$ sweep (regularizer strength)
   - Freeze $U,V$ (Stage 1 style) vs. train all (Stage 2 style)
3. **Lightweight pretraining (Weeks 9–12):** If post-training works, train a small model from scratch to show the idea generalizes beyond post-training.

---

## 5. Connection to Broader Goals

### Macro Workstream 2 (Compression Composition)

Stage 1 directly addresses the composition search problem: instead of manually exploring which axes to compress and by how much, elastic training *learns* the optimal allocation. This replaces the need for evolutionary search or exhaustive grid search over (depth × width × MLP × expert) configurations.

### ACE (Automated Compression Engine)

The long-term vision is ACE — a system that takes (source model, target hardware, budget) and returns the optimal compression pathway. Elastic training changes ACE from an NP-hard search into a lookup:

- **Without elastic:** ACE must evaluate many (architecture, pathway) candidates
- **With elastic:** ACE queries the trained elastic model for the best config at the target budget — already optimized, no search needed

### Training-Aware Kernel as a General Tool

The $K_{\text{joint}}$ kernel developed here improves all Macro compression tools (Nyström, width pruning, expert pruning) across all workstreams — not just elastic training. Any compression step that selects "which parameters to keep" benefits from training-aware selection.

---

## 6. Summary

| Stage | Goal                                  | Key Method                                     | Timeline    | Success Metric                                          |
| ----- | ------------------------------------- | ---------------------------------------------- | ----------- | ------------------------------------------------------- |
| 1     | Automatic compression config learning | Elastic training +$K_{\text{joint}}$ ranking | Weeks 1–4  | Qwen3-30B-A3B at c.f.=1.5: <2pp drop (vs. 4pp baseline) |
| 2     | One pretrain → all sizes             | Spectral pretraining + elastic regularizer     | Weeks 5–12 | Nested submodels within 1pp of independently trained    |

**The core bet:** compression config search is a training problem, not a search problem. Let gradient descent find the optimal allocation — it's better at high-dimensional optimization than we are.
