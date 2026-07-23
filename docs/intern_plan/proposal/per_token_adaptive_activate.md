# Proposal: Adaptive Active Parameter Allocation across Experts at Inference Time

---

## Motivations

On Cor3 hardware and GPUs, decode is memory-bandwidth bound: latency tracks the number of weights **loaded** per token.

To guarantee an active-parameter reduction, a common solution is to reduce the number of experts a token activates under standard top-$k$ routing or further use top-$p$ routing.

However,

* Some token have only 1-2 high probability tokens → cutting activated experts suffice, while some token have a flatter distribution across multiple experts
* The experts with low router probability may still contain critical knowledge. It can be more valuable to activate the *critical, unique* channels of a low-probability expert than the *tail* channels of a high-probability expert.

Here we reformulate the question one level finer:

> Given a a target active ratio $\rho\in[0,1]$, for each token, **how much of each expert, or which channels (neurons) is activated**?

This becomes a more general budget allocation framework to maximize the performance. Reducing $K$ is a special case where each expert is all-or-nothing ($\rho_e \in \{0, 1\}$).

**Ultimate goal**: spend that budget only on the **unique, non-overlapping knowledge** each expert holds — and to avoid re-loading redundant knowledge that has already covered elsewhere.

### Scope

This framework target budget allocation across activated experts

* The budget of active parameter is the **same** for each token.
* The total parameters are **NOT** reduced. Only active parameters is reduced.

While it still opens to combine with orthogonal techniques towards a more efficient MOE model.

* Combine with top-p routing to achieve dynamic budget for different tokens.
* Work on an already compressed and healed MOE model checkpoint.

### Hardware concern

Selecting which channels (neurons) to activate falls in the most general framework of sparse computing. Yet, brute-forcely adding router to each channel is inefficient: the router would be as large as the moe parameter itself, and the memory access will be much slower due to in-conitiguous loading.

the feasible formulation should be:

* **offline**: on the pretrained model, score the channel within each expert
* **online**: reuse the expert router output to reweight the score of each channel

With pre-sorted channel importance, we can organize the storage of expert beforehand, so the loading during inference still sees contiguous memory. This also fit in modern MoE inference in which multiple experts are loaded into a large unified GEMM.

### Common formulation

We score every channel of every active expert with a single per-token quantity that **factorizes** into an *online* router term and an *offline* channel-importance term:

$$
\text{score}_{e,j}(x)\;=\;\underbrace{\,r\!\left(g_e(x)\right)\,}_{\text{online: router reweight (free)}}\;\cdot\;\underbrace{\,s_{e,j}\,}_{\text{offline: pre-computed channel score}} .
$$

- **Online — router reweight $r(g_e(x))$.** $g_e(x)\in[0,1]$ is the router probability of expert $e$ at token $x$. It is emitted by the gate anyway, so it costs no extra weight reads and is the only genuinely per-token signal. $r(\cdot)$ is a fixed monotone map (e.g. $r(g)=g^2$) shared by all channels of an expert, so it acts as one per-expert, per-token scalar.
- **Offline — per-channel importance $s_{e,j}$.** A single static score per channel $(e,j)$, calibrated once on a small dataset and stored. It answers "how much does the layer output / loss degrade if channel $j$ of expert $e$ is removed," and is *token-independent*. Every channel metric used later — activation-energy, ridge leverage, pivoted-Cholesky marginal gain — is just a concrete choice of $s_{e,j}$.

Given a target active ratio $\rho$, the task is: per token, keep the channels with the largest $\text{score}_{e,j}(x)$ subject to the global active-parameter budget $B=(1-\rho)\,K\,m$.

The central design question of this proposal is how far this can be pushed:

> How much of the per-token channel score can be moved **offline** into $s_{e,j}$, so the online decision touches no expert weights and reduces to reweighting a pre-computed score by $r(g_e(x))$?

### Early investigation and Hypothesis.

- (Some) Experts are highly redundant: activating ~50% of a routed expert's parameters likely already covers most of its unique knowledge.
- It can be more valuable to activate the *critical, unique* channels of a low-probability expert than the *tail* channels of a high-probability expert — precisely the trade static reduction cannot make.

![Ridge leverage scores in descending order for a channel subset](../docs/results/stats/figures/ridge_leverage_descending_subset.png)

Reads of the figure: Each panel is one MoE layer; every curve is one expert's per-channel **ridge leverage score** sorted descending, where a high score means the channel is *unique* (not reconstructible from the others) and a low score means it is redundant. In early/middle layers a few channels spike while the long tail sits near zero — experts are highly redundant, so a small keep-set covers most of their knowledge — whereas the last layer's leverage is nearly uniform and resists narrowing. Redundancy is thus strongly layer-dependent, motivating a per-token, per-channel budget over a static uniform cut.

### Different levels

We consider Reduced top-$K$ routing (fewer activated experts per token) as our baseline, and consider the following 3 granularities, in increasing generality:

1. allocate per-expert, score per-expert. First decide the activation budget $\rho_e$ of each active expert, then decide *which* channels to activate within each expert.
2. allocate cross-experts, score per-expert. Put all channels in activate experts together and have a joint ranking. Active the top channels.
3. allocate cross-experts, score cross-experts. On top of 2, score each channel considering the interation / overlapping with channels in (co-activated) experts.

### Preliminary Results (TL; DR)

Granularity 1 <u>underperforms reducing k</u>. Different experts share overlapping knowledge, especially in their principal subspaces.

Granularity 2 **outperforms reducing k** in high reducing regime (reduce >50% of active parameters).

Continue to work out identifying feature covariance / overlapping across experts, to eventually achieve granularity 3 that loads **unique, non-overlapping knowledge** each expert holds

---

## Allocate per-expert, score per-expert

**Idea.** Keep a fixed per-token channel budget $B=\rho\,K\,m$, but distribute it **unevenly across each token's top-$K$ experts** — more channels to the experts that matter more *for this token*, fewer to the rest, conserving the budget exactly.

- **Offline**: rank channels within each expert by importance $s_{e,(1)}\ge s_{e,(2)}\ge\cdots$. Store once.
- **Online**: use the router probabilities to decide each expert's activation ratio $\rho_e$; *which* channels to keep is deterministic (top-$\lfloor\rho_e m\rfloor$ in the pre-computed order).

Two orthogonal knobs: (1) **expert ratio allocation** — how the budget $B$ is split across a token's $K$ experts (`router_prob`, `coverage_alloc`, `contribution`, `uniform`); (2) **channel metric** — which score ranks channels within an expert (`activation`, `ridge leverage`).

### Results (50% active cut, HellaSwag 0-shot, no training)

| Config                               | expert ratio allocation | channel selection |    acc_norm    | Active MoE param ↓ |
| ------------------------------------ | ----------------------- | ----------------- | :-------------: | :-----------------: |
| Qwen 30B A3B baseline                |                         |                   |      78.56      |          0          |
| static Nyström baseline             | uniform                 | ridge leverage    |      58.89      |        −50%        |
| Reduce top-k (8→4 experts) baseline |                         |                   | **75.96** |        −50%        |
| Dynamic                              | router_prob             | activation        |      69.45      |        −50%        |
| Dynamic                              | router_prob             | ridge leverage    |      71.46      |        −50%        |
| Dynamic                              | contribution            | ridge leverage    |      65.23      |        −50%        |
| Dynamic                              | coverage_alloc          | ridge leverage    |    *72.94*    |        −50%        |

**Reads.** Per-token heterogeneity is decisive (uniform collapses to 58.89); `ridge leverage` edges `activation` (+2 pt); `coverage_alloc` beats `router_prob` at 50% (72.94 vs 71.46).

**Limitation.** The best config (72.94) still trails reduce-top-k (75.96) by ~3 pts because per-expert budgeting cannot see **cross-expert overlap** — a low-probability expert wastes budget on channels already covered by a co-activated high-probability expert.

### Detailed method

**Expert ratio allocation (`criterion`).** All options feed a common **largest-remainder water-filling** step: set per-expert weight $w_e$, take $\mathrm{raw}_e=w_e B$, floor and clamp to $[k_{\min},m]$, redistribute remainder so $\sum_e k_e=B$ exactly.

- `router_prob` — $w_e = g_e(x)$, linear in routing weight. Empirically strongest single criterion.
- `coverage_alloc` — give expert $e$ enough channels to cover fraction $\rho_e(\alpha)=\min(\alpha\,g_e,1)$ of its total score $S_e$; binary-search $\alpha$ so $\sum k_e\le B$. Experts with concentrated scores need fewer channels, freeing budget for spread-out experts.
- `contribution` — static per-expert attribution scalar. Token-independent; barely above uniform.
- `uniform` — even $1/K$ split (ablation baseline).

**Channel metric (`channel_metric`).** Precomputed offline into descending rank per $(\text{layer},\text{expert})$; online keeps top-$k_e$ as a contiguous prefix.

- `activation` — $h_{e,j}^2\lVert w_j\rVert^2$.
- `leverage` — ridge-leverage order (redundancy-aware; downweights channels reconstructible from others). Required by `coverage_alloc`.

**Implementation.** For measurement: realize selection as a mask (zero channels beyond budget). For deployment: permute expert weights by channel order offline so each kept set is a contiguous prefix, enabling regular GEMM slices.

---

## Allocate cross-experts, score per-expert

**Idea.** Instead of pre-splitting the budget across experts, rank *all* $K\cdot m$ channels of a token's active experts on **one scale** and keep the global top-$B$. Per-expert budgets $\rho_e$ *emerge* from a single threshold — a dominated expert may receive 0 channels, and no $k_{\min}$ floor is needed.

Two improvements over the per-expert scheme:

1. **Global $g^2$ threshold** replaces per-expert quota — all channels compete on one scale.
2. **Pivoted-Cholesky order** replaces ridge leverage — redundancy-aware *within* each expert (a duplicate channel's residual gain collapses after its twin is selected).

### Results (HellaSwag 0-shot, no training)

acc_norm across the active-param budget. Reduce-top-k maps each reduction to an
integer expert count (`8→k`); `router_prob × act` is the strongest per-expert
scheme (largest-remainder water-filling, `k_min=16`).

| Active-param reduction |  Reduce top-k (8→k)  | `MoSE (router_prob × act)` |      Ours      |
| :--------------------: | :-------------------: | :---------------------------: | :-------------: |
|       Baseline        |      77.8 (k=8)      |                              |                |
|        −37.5%        | **77.1** (8→5) |             75.96             |      76.30      |
|         −50%         | **75.2** (8→4) |             69.45             |      74.26      |
|        −62.5%        |      69.8 (8→3)      |             61.00             | **70.54** |
|         −75%         |      49.4 (8→2)      |             43.66             | **63.60** |
|        −87.5%        |      26.2 (8→1)      |             30.32             | **44.15** |

`pivchol_global` beats `router_prob × act` at **every** budget (the gap widens as
the cut deepens: +4.8 → +9.5 → +19.9 → +13.8 pt), and vs reduce-top-k it matches at
moderate cuts (−37.5/−50%) and **outperforms** at high cuts (−62.5%: +0.7, −75%:
**+14.2 pt**). When the budget is tight, dropping whole experts discards unique
knowledge, whereas narrowing retains each expert's most load-bearing channels.
(reduce-top-k −87.5% = 8→1 not run; `router_prob × act` −37.5% not run.)

**MMLU (5-shot, −75% active).** The trend holds on a harder, knowledge-heavy
benchmark:

| Method (−75% active) | Reduce top-k (8→k) | `MoSE (router_prob × act)` | Ours  |
| :-------------------- | :-----------------: | ----------------------------- | ----- |
| Baseline              |     79.5 (k=8)     |                               |       |
| -75%                  |        34.90        | 49.17                         | 70.81 |

`pivchol_global` leads by **+21.6 pt** over `router_prob × act` (and +35.9 over
reduce-top-k) — an even larger margin than HellaSwag at the same budget, since
knowledge-heavy MMLU is more sensitive to destroying expert capacity.

**Remaining limitation.** The scoring matrix $\Theta_e$ is still **block-diagonal** — channels are ranked within-expert only, so cross-expert redundancy is not exploited.

### Detailed method

**Offline — pivoted Cholesky.** Per expert, build the coupling matrix $\Theta_e = G_e \odot B_e$ (activation covariance $\odot$ weight Gram, PSD). Run pivoted Cholesky: at each step, select the channel with the largest **residual diagonal** (conditional marginal gain $\sigma_{e,r}$, monotone non-increasing). This yields a nested pivot order $\pi_e$ and per-step gains $\sigma_{e,r}$, stored once as `pivchol_artifact.pth` (~57 MB for Qwen3-30B, budget-agnostic).

*Why pivoted Cholesky over ridge leverage:* ridge leverage is a static per-column score — two near-duplicate channels split the leverage, so a top-$k$ cut either keeps both (double-spend) or drops both (lose information). Pivoted Cholesky is *conditional*: once one duplicate is selected, the other's residual collapses to $\approx 0$.

**Online.** Per token, score each active expert's channels by $g_e^2 \cdot \sigma_{e,r}$ (each expert's sequence is pre-sorted since $g^2$ is a per-expert constant and $\sigma$ is monotone). Keep the global top-$B$ — touches **no expert weights**, only the offline gains and free router probabilities. Overhead: $\approx 0.016\%$ of the expert-FFN MACs.

---

## Allocate cross-experts, score cross-experts

**Goal.** Spend the budget only on the **unique, non-overlapping knowledge** each expert holds. This requires the **cross-expert covariance** — measuring how much of one expert's channels can already be reconstructed from channels of co-activated experts — which the block-diagonal scoring above cannot see.

### Experimental plan

We first quantify how much accuracy the block-diagonal approximation forfeits, and in what structural form the redundancy appears. Three measurements:

1. **Oracle ladder (M1)** — compare: (A) oracle with exact per-token activations + full off-diagonal $\Theta$, (B) oracle with router only + offline cross-expert $\Theta$, (C) current block-diagonal Level 1. Gap (B−C) is the value of Level-2; negligible gap terminates the effort.
2. **Redundancy structure (M2)** — plot cross-expert coherence vs. pivot rank. Monotone decay ⇒ overlap is concentrated in leading ("public") channels, cheap prefix correction suffices. Flat ⇒ full global selection required.
3. **Regime diagnostic (M3)** — rule out simpler causes of the ~1 pt mid-budget gap via $g^{2\beta}$ sharpness sweep (contains both Level 1 and reduce-top-k as limits) and per-token accuracy bucketed by router entropy.

### Detailed method

#### M1 — Oracle ladder

Three selectors on ~1–2k tokens, ignoring efficiency:

|                    | Online information            | Coupling matrix                            | Status                                  |
| ------------------ | ----------------------------- | ------------------------------------------ | --------------------------------------- |
| **Oracle-A** | exact per-token$h_{e,j}(x)$ | full$\Theta(x)$, off-diagonal included   | absolute upper bound (unreachable)      |
| **Oracle-B** | router$g(x)$ only           | offline$\Theta$ with cross-expert blocks | **the ceiling Level-2 can reach** |
| **Level 1**  | router$g(x)$ only           | block-diagonal$\Theta_e$                 | current                                 |

Run at every budget point ($-37.5\%$ to $-75\%$). **Decision:** negligible (B − Level 1) at all budgets terminates the effort.

#### M2 — Where the redundancy lives

- **Coherence vs. pivot rank.** Bucket channels by pivoted-Cholesky rank, plot cross-expert coherence $\mu_{(e,j),(f,l)} = |\Theta_{(e,j),(f,l)}| / \sqrt{\Theta_{(e,j),(e,j)}\,\Theta_{(f,l),(f,l)}}$ against rank.
- **Subspace geometry.** Principal angles / Grassmann distance between leading eigen-subspaces of $\Theta_e$ for frequently co-activated expert pairs.

**Decision:** monotone decay → low-rank prefix correction; flat profile → full global selection.

#### M3 — Regime diagnostic

- **Sharpness sweep.** $s_{e,j}=g_e^{2\beta}\,\sigma_{e,j}$ for $\beta\in\{1,1.5,2,3\}$. Contains Level 1 ($\beta=1$) and degenerates to reduce-top-k as $\beta\to\infty$.
- **Entropy bucketing.** Per-token accuracy delta bucketed by router entropy.

**Decision:** if the sweep closes the mid-budget gap, Level-2's target regime is redefined before collecting cross-expert statistics.
