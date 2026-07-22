# Proposal: Adaptive Budget Allocation across Experts / Channels at Inference Time

---

## Motivations

On Cor3 hardware, decode is memory-bandwidth bound: latency tracks the number of weights **loaded** per token. To *guarantee* an active-parameter reduction, we propose to

- **fix a target active ratio** $\rho\in[0,1]$ (target active-parameter ratio; e.g. $\rho=0.5$), and
- **decide the budget allocation for each token, on the fly** during inference.

The ultimate goal is to spend that budget only on the **unique, non-overlapping knowledge** each expert holds — and to avoid re-loading redundant knowledge that has already covered elsewhere.

**Different axes.** The common way to reduce active parameters is to reduce $K$ — the number of experts a token activates under standard top-$k$ (Matryoshka / adaptive-$K$ routing, top-$p$ routing); for the target architecture $K=8$ out of $N=128$ experts per layer. Here we reformulate the question one level finer:

> Given a global budget, **how much of each expert is activated** — down to the channel?

Reducing $K$ is a special case where each expert is all-or-nothing ($\rho_e \in \{0, 1\}$).

**Early investigation and Hypothesis.**

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

**33% active cut (c.f. = 1.5):**

| Config                        | expert ratio allocation | channel selection |    acc_norm    | Active MoE param ↓ |
| ----------------------------- | ----------------------- | ----------------- | :-------------: | :-----------------: |
| Qwen 30B A3B baseline        |                         |                   |      78.56      |          0          |
| static Nyström baseline      | uniform                 | —                |      66.29      |        −33%        |
| Dynamic prob × activation    | router_prob             | activation        |      75.96      |        −33%        |
| Dynamic prob × leverage      | router_prob             | ridge leverage    | **76.13** |        −33%        |
| Dynamic contrib × activation | contribution            | activation        |      67.79      |        −33%        |
| Dynamic contrib × leverage   | contribution            | ridge leverage    |      69.46      |        −33%        |

**50% active cut (c.f. = 2.0):**

| Config                               | expert ratio allocation | channel selection |    acc_norm    | Active MoE param ↓ |
| ------------------------------------ | ----------------------- | ----------------- | :-------------: | :-----------------: |
| Qwen 30B A3B baseline               |                         |                   |      78.56      |          0          |
| static Nyström baseline             | uniform                 | —                |      58.89      |        −50%        |
| Reduce top-k (8→4 experts) baseline |                         |                   | **75.96** |        −50%        |
| Dynamic prob × leverage             | router_prob             | ridge leverage    |      71.46      |        −50%        |
| Dynamic coverage × leverage         | coverage_alloc          | ridge leverage    |      72.94      |        −50%        |

**Reads.**

- **Per-token heterogeneity is decisive.** The uniform even-split baseline collapses (66.29); routing the budget by `router_prob` recovers to 75.96–76.13
- **`ridge leverage` edges `activation`** as the channel-selection metric (76.13 vs 75.96), even without the Nyström `down_proj` correction.

**Baseline comparison:**

- **Static-prune ceiling not yet matched.** A per-token active budget costs ~2–2.5 pts vs. the static 33% storage prune — but that static cut barely reduces active compute, whereas these configs deliver a true ~33–50% active cut.
- **Underperformed to reduced experts baseline.** The principal subspaces / features of low probability experts actually contains **overlapped** information with high probabiliy experts,

---

## Unified Framework: Channel Selection Across Experts

**Motivation.** The two-level scheme assumes channels are independent within an expert (diagonal coupling) and that experts don't overlap. Neither holds cleanly:

- Different experts **share overlapping knowledge**, especially in their principal subspaces.
- We want to activate the *unique* knowledge of low-probability experts — which a per-expert budget cannot see, because it never compares a channel in expert $A$ against a channel in expert $B$.

**Overall goal.** Spend the budget only on the **unique, non-overlapping knowledge** each expert holds. This requires looking at the **covariance / coupling of features across co-activated experts** — measuring how much of one expert's channels can already be reconstructed from the channels of the other experts a token fires together with, and declining to re-load what is redundant. A per-expert budget cannot see this, because it never compares a channel in expert $A$ against a channel in expert $B$.

Below are some possible directions toward that goal, at a high level.

### Direction 1 Full cross-expert Nyström

The right per-channel score for the shared goal is the **ridge leverage score** (as used in Nyström approximation): higher score → more unique / load-bearing; lower score → redundant, reconstructible from other channels. Ranking by leverage naturally deprioritizes channels whose direction is already covered — the exact behavior we want for cross-expert redundancy.

This points to the most direct realization of the goal: run **one Nyström selection across all co-activated experts jointly**, on the combined feature covariance rather than per-expert blocks. Such a joint kernel captures the coupling *across* experts and measures directly how representable each expert's channels are by the channels of the other co-activated experts — selecting a globally non-redundant set.

**Challenge.** The joint covariance spans all $N\cdot m$ channels of the layer, so computing ridge leverage scores over such a huge matrix (and doing so per token, online) is the central difficulty. Making this tractable — via block structure, low-rank sketching, restriction to the router's top-$K$, or offline precomputation of the reusable pieces — is the open problem for this direction.
