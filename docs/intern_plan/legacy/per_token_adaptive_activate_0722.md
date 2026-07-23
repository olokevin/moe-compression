# Proposal: Adaptive Budget Allocation across Experts / Channels at Inference Time

---

## Motivations

On Cor3 hardware, decode is memory-bandwidth bound: latency tracks the number of weights **loaded** per token. 

To *guarantee* an active-parameter reduction, a common solution is to reduce the number of experts a token activates under standard top-$k$ routing or further use top-$p$ routing.

However, the experts with low router probability may still contain critical knowledge. Some preliminary studies showed that experts could be highky redundant, and it can be more valuable to activate the *critical, unique* channels of a low-probability expert than the *tail* channels of a high-probability expert.



Here we reformulate the question one level finer:

> Given a a target active ratio $\rho\in[0,1]$, **how much of each expert is activated** — down to the channel?

This becomes a more general budget allocation framework. Reducing $K$ is a special case where each expert is all-or-nothing ($\rho_e \in \{0, 1\}$).



we propose to

- **fix a target active ratio** $\rho\in[0,1]$ (target active-parameter ratio; e.g. $\rho=0.5$), and
- **decide the budget allocation for each token, on the fly** during inference.

The ultimate goal is to spend that budget only on the **unique, non-overlapping knowledge** each expert holds — and to avoid re-loading redundant knowledge that has already covered elsewhere.

### A general active parameter budget allocation framework

The common way to reduce active parameters is to reduce $K$ — the number of experts a token activates under standard top-$k$ routing or use top-$p$ routing.

However, router probability distribution across tokens is quite different:

- some token have only 1-2 high probability tokens → cutting activated experts suffice
- some token have a flatter distribution across multiple experts


### Early investigation and Hypothesis.

- (Some) Experts are highly redundant: activating ~50% of a routed expert's parameters likely already covers most of its unique knowledge.
- It can be more valuable to activate the *critical, unique* channels of a low-probability expert than the *tail* channels of a high-probability expert — precisely the trade static reduction cannot make.

![Ridge leverage scores in descending order for a channel subset](../docs/results/stats/figures/ridge_leverage_descending_subset.png)

### Common formulation

Deleting channel $j$ of expert $e$ is exact in output space — it changes the layer output by exactly $-\,g_e(x)\,h_{e,j}(x)\,w_j$. To second order, the loss impact **factorizes** into a router part (online, cheap) and a channel-importance part:

$$
\delta\mathcal{L}_{e,j}(x)\;\propto\;\underbrace{g_e(x)^2}_{\text{router (online)}}\;\cdot\;\underbrace{h_{e,j}(x)^2\,\lVert w_j\rVert_H^2}_{s_{e,j}(x)\;:\;\text{channel importance}}.
$$

where $g_e(x)\in[0,1]$ is the router probability for expert $e$ at token $x$, $h_{e,j}(x)$ is the SwiGLU intermediate activation of channel $j$, $w_j$ its `down_proj` row (with $\lVert\cdot\rVert_H$ the loss-curvature norm), and $s_{e,j}(x)$ the resulting per-channel importance.

So the problem input is:

- the **router probability output** $g_e(x)$ (free, per token), and
- the **pre-calibrated error of removing each channel / expert** ($s_{e,j}$, offline),

and the task is to select the set of experts/channels that **minimizes the error for the current token** subject to the global active-parameter budget.

**The binding constraint (bandwidth trap).** The channel term $s_{e,j}(x)$ needs $h_{e,j}(x)$, which requires reading *every* expert's gate + up projections.The central design question of this proposal is therefore:

> How much of the per-token channel score can be moved **offline**, so the online decision touches no expert weights?

### Different levels

We consider three granularities, in increasing generality:

1. **Reduced top-$K$ routing** — fewer activated experts per token
2. **(Proposed) Expert–channel two-level** — first decide the activation budget $\rho_e$ of each active expert, then decide *which* channels within it. The within-expert order is frozen offline.
3. **(Proposed) Unified channel-selection framework** — rank *all* channels in a MoE layer on one scale and select globally, crossing expert boundaries.

The two proposed levels are the same water-filling objective under different approximations of the coupling between channels.

### Results (TL; DR)

Under expert-channel two-level allocation: <u>underperforms reducing k</u>. Different experts share overlapping knowledge, especially in their principal subspaces.

In a channel-selection framework: **outperforms reducing k** in high reducing regime (reduce >50% of active parameters).

Continue to work out identifying feature covariance / overlapping across experts, to eventually achieve loading **unique, non-overlapping knowledge** each expert holds

---

## Expert–Channel Two Level

**Idea.** Keep a fixed per-token channel budget $B=\rho\,K\,m$, but distribute it **unevenly across each token's top-$K$ experts** — more channels to the experts that matter more *for this token*, fewer to the rest, conserving the budget exactly. This two-level "expert score × within-expert channels" structure treats channels as independent contributions to the loss.

### Flow

- **Offline**: collect and save the channel metrics once (per-expert sorted channel importance $s_{e,(1)}\ge s_{e,(2)}\ge\cdots$, ). Nothing new is collected beyond ranking statistics the scoring stage already saves.
- **Online**: only the expert activation ratio $\rho_e$ is dynamic. Given a $\rho_e$, *which* channels within the expert are activated is **deterministic** — load the top-$\lfloor\rho_e m\rfloor$ channels in the pre-computed channel order.

### The two design knobs

Selection factorizes into two orthogonal choices, exposed in the implementation as `criterion` (which sets each expert's budget $k_e$) and `channel_metric` (which ranks channels inside an expert). The total budget is $B=\lfloor(1-\rho)\,K\,m\rceil$ channels per token, conserved exactly across the top-$K$; each expert is held to $k_{\min}\le k_e\le m$ (with $k_{\min}=16$ in the runs below).

- **Knob 1 — Expert ratio allocation** (`criterion`): how the budget $B$ is split across a token's $K$ experts. All four options feed a common **largest-remainder water-filling** step — set a per-expert weight $w_e$, take $\mathrm{raw}_e=w_e B$, floor and clamp to $[k_{\min},m]$, then hand the signed budget deficit to the largest-remainder experts below cap (or reclaim from the smallest-remainder experts above floor) so $\sum_e k_e=B$ exactly. The options differ only in $w_e$:
  - `router_prob` — the per-token normalized softmax weight $g_e(x)$ itself, so budget is allocated **linearly** in the routing weight ($\mathrm{raw}_e=g_e B$). Empirically the strongest single criterion, and the truly per-token signal.
  - `coverage_alloc` — replaces the linear split with a coverage *target*: give expert $e$ enough channels to cover a fraction $\rho_e(\alpha)=\min(\alpha\,g_e,\,1)$ of its total channel score $S_e$, where the single per-token scalar $\alpha$ is binary-searched so $\sum_e k_e\le B$, then a coverage-aware top-up lands $\sum_e k_e=B$ exactly. Two experts with equal $g_e$ get equal coverage *targets*, but an expert whose score is **concentrated** reaches that target with fewer channels, freeing budget for experts whose score is spread out (uses the precomputed descending-score prefix sums $S_e(n)$).
  - `contribution` — the static per-expert attribution scalar `expert_out_token_contrib` (stored negative; negated and clamped $\ge 0$). Token-independent per expert, so it varies only through *which* experts a token selects — not truly per-token, and empirically barely above `uniform`.
  - `uniform` — even $1/K$ split across the top-$K$ (ablation baseline; reduces to a static uniform keep-set).
- **Knob 2 — Channel selection** (`channel_metric`): which channels an expert keeps at its budget $k_e$. Both metrics are precomputed offline into a descending-score rank per $(\text{layer},\text{expert})$; online, the expert deterministically keeps the top-$k_e$ ranks (a contiguous prefix), zeroing the rest of the SwiGLU intermediate before `down_proj`.
  - `activation` — activation-magnitude order (repo default), $h_{e,j}^2\lVert w_j\rVert^2$.
  - `leverage` — Nyström ridge-leverage order (redundancy-aware; downweights channels reconstructible from others), score-only, no `down_proj` reconstruction. `coverage_alloc` requires this metric (its coverage curve is the prefix sum of the leverage scores).

### Implementation

- **Fast experiment (measurement).** Realize selection as a **mask** — zero the channels beyond each token's budget. This measures exact accuracy at the target active budget without variable-width matmuls, and reuses ranking statistics already saved by the scoring stage.
- **Actual saving (deployment).** The MoE forward merges the top-$K$ experts into a single large grouped GEMM. To realize the *bandwidth* saving, the memory-access strategy must load only the fractional (top-$\rho_e m$) slice of each expert into that GEMM. Permuting each expert's weights by its channel order offline makes each kept set a **contiguous prefix**, so reads are slices rather than scattered gathers — the property that keeps the GEMM regular.

### Preliminary results

**50% active cut (c.f. = 2.0):**

| Config                               | expert ratio allocation | channel selection |    acc_norm    | Active MoE param ↓ |
| ------------------------------------ | ----------------------- | ----------------- | :-------------: | :-----------------: |
| Qwen 30B A3B baseline                |                         |                   |      78.56      |          0          |
| static Nyström baseline             | uniform                 | ridge leverage    |      58.89      |        −50%        |
| Reduce top-k (8→4 experts) baseline |                         |                   | **75.96** |        −50%        |
| Dynamic                              | router_prob             | activation        |      69.45      |        −50%        |
| Dynamic                              | router_prob             | ridge leverage    |      71.46      |        −50%        |
| Dynamic                              | contribution            | ridge leverage    |      65.23      |        −50%        |
| Dynamic                              | coverage_alloc          | ridge leverage    |    *72.94*    |        −50%        |

**Reads.**

- **Per-token heterogeneity is decisive.** The uniform even-split baseline collapses (66.29 at 33%, 58.89 at 50%); routing the budget by `router_prob` recovers to 75.96–76.13 (33%) and 69.45–71.46 (50%).
- **`ridge leverage` edges `activation`** as the channel-selection metric (33%: 76.13 vs 75.96; 50%: 71.46 vs 69.45), even without the Nyström `down_proj` correction.
- **`coverage_alloc` beats `router_prob`** at 50% (72.94 vs 71.46) — combining router contribution with each expert's leverage-concentration curve allocates the fixed budget better than router probability alone.

**Baseline comparison and limitation.**

- **Static-prune ceiling not yet matched.** A per-token active budget costs ~2–2.5 pts vs. the static 33% storage prune — but that static cut barely reduces active compute, whereas these configs deliver a true ~33–50% active cut.
- **Below the reduced-experts baseline at 50%, and why.** The best two-level config (72.94) still trails reduce-top-k (75.96) by ~3 pts. The cause is a structural limitation of the per-expert budget: **different experts share overlapping knowledge, especially in their principal subspaces.** Because the two-level scheme never compares a channel in expert $A$ against a channel in expert $B$, a low-probability expert is forced to spend its budget on channels that merely re-load information a co-activated high-probability expert already carries. Exploiting this cross-expert overlap requires ranking channels *across* experts on one scale — the Unified Framework below.

---

## Unified Framework: Channel Selection Across Experts

**Motivation.** The two-level scheme assumes channels are independent within an expert (diagonal coupling) and that experts don't overlap. Neither holds cleanly:

- Different experts **share overlapping knowledge**, especially in their principal subspaces.
- We want to activate the *unique* knowledge of low-probability experts — which a per-expert budget cannot see, because it never compares a channel in expert $A$ against a channel in expert $B$.

**Overall goal.** Spend the budget only on the **unique, non-overlapping knowledge** each expert holds. This requires looking at the **covariance / coupling of features across co-activated experts** — measuring how much of one expert's channels can already be reconstructed from the channels of the other experts a token fires together with, and declining to re-load what is redundant. A per-expert budget cannot see this, because it never compares a channel in expert $A$ against a channel in expert $B$.

The framework ranks *all* $K\cdot m$ channels of a token's active experts on **one scale** and selects the global top-$B$: per-expert budgets $\rho_e$ *emerge* from a single shadow price rather than being pre-assigned. Level 1 below is the simplest realization — a global selection whose scoring matrix is still block-diagonal — and Direction 1 adds the off-diagonal cross-expert coupling.

### Level 1 (current): global $g^2$-weighted, redundancy-aware nested selection (`pivchol_global`)

The current best zero-training method. It pools all channels of a token's active experts into one global competition, replacing three components of the earlier per-expert scheme — each identified as a defect:

1. **per-expert quota → global $g^2$ threshold.** Rather than splitting the budget across experts by a per-token weight, all $K\cdot m$ channels of a token's active experts compete on **one scale**; per-expert prefix lengths $t_e$ (hence $\rho_e$) *emerge* from a single shadow price, and a dominated expert may receive 0 channels (no $k_{\min}$ floor).
2. **ridge-leverage in-expert order → pivoted-Cholesky nested order.** A redundancy-aware pivot order that de-prioritizes channels reconstructible from those already selected within the expert.
   1. Ridge leverage is a static, per-column score designed for *randomized sampling with reweighting*. Under a *deterministic top-$k$ cut* it fails on redundancy: two near-duplicate channels split the leverage in half, so a threshold either keeps both (double-spend on one piece of information) or drops both (lose information that should be kept once).
   2. Pivoted-Cholesky marginal gain is *conditional*: once one duplicate is selected, the other's residual diagonal collapses to $\approx 0$ and it is naturally excluded. This is literal "no budget on redundant knowledge" — **within** an expert.

**Offline.** Per expert build the coupling $\Theta_e = G_e \odot B_e$, where $G_e = \mathbb{E}[\phi_e \phi_e^\top]$ is the cached activation covariance matrix and $B_e = W_{\text{down}}^\top W_{\text{down}}$ the weight Gram ($H = I$). $\Theta_e$ is PSD (Schur product of two PSD matrices), and its diagonal $\Theta_e[j,j]$ is exactly the single-channel importance $s_{e,j}$ — deleting channel $j$ alone costs $\Theta_e[j,j]$, while its off-diagonals encode how much any two channels overlap.

*What pivoted Cholesky is.* It is Cholesky factorization with **greedy symmetric (diagonal) pivoting** — equivalently a greedy Nyström column-subset selection that builds a low-rank factor $L$ one channel at a time. At each step it selects the channel with the largest **residual diagonal**: the portion of that channel's importance *not yet explained* by the channels already picked. That residual is the channel's **conditional** (marginal) contribution, so a channel whose direction is already covered by earlier pivots has its residual collapse toward $0$ and is deferred — this is what makes the order redundancy-aware. Run to completion it infers (i) a nested pivot order $\pi_e$ (every prefix is a near-optimal subset reconstructing $\Theta_e$) and (ii) per-step marginal gains $\sigma_{e,r}$ = the residual diagonal at the moment channel $\pi_e(r)$ is chosen, **monotone non-increasing** by construction.

*How it is computed* (batched over the $E$ experts of a layer): add a shared ridge $\lambda_r=1.0$ to the diagonal, then keep a running residual diagonal $d\leftarrow\operatorname{diag}(\Theta_e)$. For each step $t=0\dots m{-}1$: (1) pivot $= \arg\max_j d_j$ over not-yet-chosen channels; (2) record the gain $\sigma_{e,t}=d_{\text{pivot}}$; (3) form the new factor column $\ell = \bigl(\Theta_e[:,\text{pivot}] - L_{:,:t}\,L_{\text{pivot},:t}^\top\bigr)/\sqrt{\sigma_{e,t}}$ (the pivot column with the span of the $t$ already-selected columns projected out); (4) downdate $d \leftarrow (d - \ell^{\odot 2})_{+}$ and zero the pivot. The result is stored once as `pivchol_artifact.pth` — the pivot *rank* of each physical channel plus the rank-ordered gains $\sigma_e$; it is **budget-agnostic** (independent of $\rho$).

**Online.** Per token, score each active expert's channels by $g_e^2 \cdot \sigma_{e,r}$ ($\sigma$ monotone and $g^2$ a per-expert constant → each expert's sequence is pre-sorted), keep the global top-$B$ with $B = \operatorname{round}((1-\rho)\,K\,m)$ **The online decision touches no expert weights** — only the offline gains and the free router probabilities. The table below summarizes what is stored offline and computed online (numbers for Qwen3-30B-A3B: $L=48$ MoE layers, $E=128$ experts, $m=768$, $K=8$).

| Phase                                                                | Item                                                                                     | Cost                                             |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **Stored offline** (`pivchol_artifact.pth`, budget-agnostic) | pivot ranks`channel_rank` $(L,E,m)$ int64                                            | 37.7 MB                                          |
|                                                                      | marginal gains`gains` $(L,E,m)$ fp32                                                 | 18.9 MB                                          |
|                                                                      | **total** (one factorization covers every $\rho$; only $B$ changes at install) | **≈ 57 MB**                               |
| **Online / token / MoE layer**                                 | gather$\sigma$ for the $K$ active experts, scale by $g_e^2$                        | $K\!\cdot\!m = 6144$ mults                     |
|                                                                      | global top-$B$ over the pooled $K\!\cdot\!m$ scores                                  | $O(K\!\cdot\!m\,\log B)$                       |
|                                                                      | count$\to t_e$ (scatter-add) + keep-mask $\pi\text{-rank}<t_e$                       | $K\!\cdot\!m$                                  |
|                                                                      | expert weights read for the scoring decision                                             | **none**                                   |
|                                                                      | overhead vs. the expert-FFN MACs it gates ($3Kmd$)                                     | **≈ 0.016 %.Results (HellaSwag 0-shot).** |

| Active-param reduction |  Reduce top-k  | Level 1 (`pivchol_global`) |
| :--------------------: | :------------: | :--------------------------: |
|        −37.5%        | **77.1** |            76.30            |
|         −50%         | **75.2** |            74.26            |
|        −62.5%        |      69.8      |       **70.54**       |
|         −75%         |      49.4      |       **63.60**       |

- **Our allocation matches reduce-top-k at low-to-moderate reductions.** At −37.5% (76.30 vs 77.1) and −50% (74.26 vs 75.2) Level 1 is within ~1 pt of reduce-top-k — well inside a couple of stderr — so per-token narrowing is on par with dropping the lowest-probability experts.
- **Our allocation outperforms reduce-top-k in the high-reduction regime.** At −62.5% Level 1 leads (70.54 vs 69.8), and at −75% the gap widens sharply (**63.60 vs 49.4, +14.2 pt**). When the budget is tight, dropping whole experts discards their unique knowledge, whereas narrowing every expert retains each expert's most load-bearing channels — precisely where per-token, per-channel allocation pays off.

**What it establishes — and its remaining limitation.** Level 1 is a *correct* narrowing ceiling: it competes channels globally, but its scoring matrix $\Theta_e$ is **block-diagonal** (no cross-expert terms), so it **still cannot exploit cross-expert redundancy**. It isolates the variable and motivates adding the off-diagonal coupling, i.e. the full cross-expert direction below.

### Level-2: cross-expert feature redundancy / overlapping measurement

We first quantify **how much accuracy the block-diagonal approximation actually forfeits**, and **in what structural form** the cross-expert redundancy appears.

- **Headroom.** An oracle ladder at every budget point: full-covariance selection with exact per-token activations (absolute ceiling) vs. full-covariance selection driven by offline statistics and the router only (what Level-2 can actually reach) vs. current Level 1. The middle-minus-bottom gap is the entire value of Level-2; if it is negligible, the effort stops here.
- **Attribution of the current gap.** Before attributing the ~1 pt shortfall at $-37.5\%/-50\%$ to cross-expert coupling, rule out the simpler cause — insufficient allocation sharpness — via a $g^{2\beta}$ sweep (which contains both Level 1 and reduce-top-k as limits) and a per-token breakdown by router entropy.
- **Structure of the redundancy.** Whether cross-expert overlap is concentrated in the leading ("public") channels and decays along the pivot order, or is spread uniformly. This single profile decides whether a cheap prefix-preserving correction suffices or full global selection is required.

#### M1 — Oracle ladder: is there headroom at all?

Three selectors on a small eval subset (~1–2k tokens), ignoring all efficiency constraints:

|                    | Online information            | Coupling matrix                            | Status                                           |
| ------------------ | ----------------------------- | ------------------------------------------ | ------------------------------------------------ |
| **Oracle-A** | exact per-token$h_{e,j}(x)$ | full$\Theta(x)$, off-diagonal included   | absolute upper bound (unreachable)               |
| **Oracle-B** | router$g(x)$ only           | offline$\Theta$ with cross-expert blocks | **the ceiling Level-2 can actually reach** |
| **Level 1**  | router$g(x)$ only           | block-diagonal$\Theta_e$                 | current                                          |

- $(\text{B}-\text{Level 1})$ — the value of restoring the off-diagonal blocks. This is the Level-2 target.
- $(\text{A}-\text{B})$ — the price of the "online decision touches no expert weights" constraint, isolated as its own quantity.

Run at every budget point ($-37.5\%$ to $-75\%$); the gap is expected to be non-monotone, peaking in the mid-compression regime where the budget is tight enough for redundant re-loading to hurt but loose enough that experts still overlap.

**Decision:** a negligible $(\text{B}-\text{Level 1})$ at all budgets terminates the Level-2 engineering effort.

#### M2 — Where the redundancy lives

Two views of the same question — is overlap concentrated in the principal ("public") channels, leaving the tail private?

- **Coherence vs. pivot rank.** Bucket channels by their pivoted-Cholesky rank and plot the cross-expert coherence

$$
\mu_{(e,j),(f,l)} \;=\; \frac{\bigl|\Theta_{(e,j),(f,l)}\bigr|}{\sqrt{\Theta_{(e,j),(e,j)}\,\Theta_{(f,l),(f,l)}}}
$$

  against rank. A monotone decay is the "head-public, tail-private" signature.

- **Subspace geometry.** Principal angles / Grassmann distance between the leading eigen-subspaces of $\Theta_e$ for frequently co-activated expert pairs.

**Decision:** monotone decay licenses a low-rank ("publicness") correction that preserves prefix-contiguity; a flat profile forces full global selection. The measured $\mu_\ell$ also instantiates the coherence bound in the theory section.

#### M3 — Regime diagnostic: is the residual gap even a coupling problem?

Level 1 trails reduce-top-k by ~1 pt at $-37.5\%/-50\%$ but wins decisively at $-62.5\%/-75\%$. This pattern is not necessarily caused by cross-expert coupling, and must be ruled out first.

- **Induced allocation.** Compare the distribution of emergent prefix lengths $t_e$ against reduce-top-k's hard $0/1$ allocation. An over-flat $t_e$ profile indicates a dynamic-range problem in the score, not a structural one.
- **Sharpness sweep.** $s_{e,j}=g_e^{2\beta}\,\sigma_{e,j}$ for $\beta\in\{1,1.5,2,3\}$. This family contains Level 1 ($\beta=1$) and degenerates to reduce-top-k as $\beta\to\infty$, so it strictly contains both baselines and should not lose to either.
- **Entropy bucketing.** Per-token (Level 1 − reduce-top-k) accuracy delta bucketed by router entropy. Expected: reduce-top-k is near-lossless on low-entropy tokens (probability mass concentrated, the dropped experts contributed little), and we win on high-entropy tokens.

**Decision:** if the sweep closes the mid-budget gap, Level-2's target regime is redefined before any statistics are collected.
