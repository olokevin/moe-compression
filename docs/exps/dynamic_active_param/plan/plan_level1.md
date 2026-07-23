# Level 1 — Global $g^2$-Weighted Nested Channel Selection

## High-Level Analysis

### What this replaces

The current method has three components, each of which is subtly wrong for the goal:

|                              | Current method                                                  | Level 1                                                   |
| ---------------------------- | --------------------------------------------------------------- | --------------------------------------------------------- |
| **Per-expert budget**  | router prob$g_k$ directly sets each expert's quota $\rho_k$ | quotas**emerge** from a single global threshold     |
| **Gate power**         | linear$g_k$                                                   | $g_k^2$ (correct squared-error weight)                  |
| **In-expert ordering** | ridge leverage score (static, per-column)                       | pivoted-Cholesky marginal gain (nested, redundancy-aware) |

### Why each change matters

**1. Global competition instead of pre-assigned quotas.** The current two-level structure (assign a quota per expert by $g_k$, then select within it) implicitly assumes channels across experts are incomparable — it only ever ranks channels *inside* an expert. But they are comparable: a mid-ranked channel of a high-$g$ expert usually contributes more to the output than the top channel of a low-$g$ expert. Under a pre-assigned quota, the low-$g$ expert is *forced* to spend its budget even on channels weaker than the ones being cut from a strong expert. This is exactly the mechanism behind the observed failure ("low-probability experts spend budget on redundant information"). Level 1 pools all channels of the 8 active experts and selects the global best under one shadow price; a weak expert that loses the competition simply gets nothing, and its budget flows to strong experts. $\rho_k$ becomes an **output** of the competition, not an input.

**2. $g^2$, not $g$.** Under squared-error, truncating a channel of expert $k$ costs $\propto g_k^2$. With `norm_topk_prob=True`, top-8 gates are steep (top-1 $\approx 0.3$–$0.5$, 8th $\approx 0.02$–$0.05$); squaring drops the 8th expert's weight from $\sim 1/10$ to $\sim 1/100$ of the top. Linear-$g$ allocation systematically over-feeds low-probability experts by an order of magnitude. Notably, the $g^2$ weighting *itself predicts expert-dropping should win*: it concentrates budget on the top 1–2 experts, whose extreme is dropping the low-$g$ experts entirely.

**3. Pivoted-Cholesky marginal gain, not ridge leverage — this is the redundancy fix.** Ridge leverage is a static, per-column score designed for *randomized sampling with reweighting*. Under a *deterministic top-$k$ cut* it fails on redundancy: two near-duplicate channels split the leverage in half, so a threshold either keeps both (double-spend on one piece of information) or drops both (lose information that should be kept once). Pivoted-Cholesky marginal gain is *conditional*: once one duplicate is selected, the other's residual diagonal collapses to $\approx 0$ and it is naturally excluded. This is literal "no budget on redundant knowledge" — **within** an expert.

### Relation to Nyström / MoDeGPT

Pivoted-Cholesky is not the opposite of Nyström — it **is** the greedy, deterministic version of Nyström landmark selection. MoDeGPT selects MLP landmarks by ridge-leverage sampling because it does *static, single-point* compression, where sampling + reweighting is near-optimal. Our target is different: **per-token, budget-agnostic, deterministic** selection. That target requires (a) a *nested* ordering (any prefix is good, since the online budget $t_k(x)$ varies per token), (b) *deterministic hard truncation with no reweighting* (online cannot reweight), and (c) *cross-expert comparability* on one absolute scale (for a single global threshold). Ridge leverage satisfies none of these under hard truncation; pivoted-Cholesky satisfies all three. This is a structural fit, not a marginal tuning choice.

### Honest ceiling

The scoring matrix $\Theta_k$ is **block-diagonal** — it has no cross-expert terms, so it structurally discards the cross-expert overlap that caused the original failure. Level 1 is expected to **match** expert-dropping and cleanly establish the block-diagonal upper bound; it is not expected to beat it. Its value is twofold: (i) it yields a *correct* zero-training baseline (the current method's baseline is skewed by the three errors above), and (ii) it isolates the variable — if L1 matches but does not beat expert-dropping, cross-expert redundancy is confirmed as the true bottleneck, motivating Level 3.

---

## Implementation

Two phases: an **offline** calibration pass that produces two small per-expert tables, and an **online** kernel that consumes only the router output.

### Notation

- $d = 2048$ (model dim), $m = 768$ (`moe_intermediate_size`), $E = 128$ experts, top-$k = 8$, $L = 48$ layers.
- $\phi_k(x) = \sigma(W_{gate}^{(k)} x) \odot (W_{up}^{(k)} x) \in \mathbb{R}^{m}$ — the SwiGLU intermediate activation.
- $d_{k,j}$ = column $j$ of $W_{down}^{(k)} \in \mathbb{R}^{d \times m}$ — channel $j$'s output direction.
- $H$ = shared output metric (start with $H = I$; optionally a per-layer output covariance).
- $\lambda_r$ = single shared ridge, fixed across all experts and layers.

### Offline — Phase A: accumulate the activation Gram

For each layer, for each expert $k$, accumulate a running $m \times m$ matrix over calibration tokens **routed to $k$**:

$$
M_k \mathrel{+}= \phi_k(x)\,\phi_k(x)^\top, \qquad n_k \mathrel{+}= 1.
$$

- Do **not** store the per-token activations — only the $m \times m$ accumulator and the count $n_k$.
- Use the **uncentered** second moment (do not subtract the mean): we score total energy contribution to the output, and the mean activation is part of that contribution.
- Accumulate in fp32; inputs can be bf16.
- Calibration set: a few hundred to a few thousand sequences of general text is typically sufficient; verify each expert receives enough tokens ($n_k$ not too small) or its statistics will be noisy.
- Peak transient memory is $E \cdot m^2$ per layer ($\approx 75$M floats $\approx 300$MB fp32) if all experts are accumulated in one pass. If tight, shard experts across passes.

At the end: $G_k = M_k / n_k$ (the activation Gram $\mathbb{E}[\phi_k \phi_k^\top]$).

### Offline — Phase B: build $\Theta_k$ and factor it

Per expert (loop; never hold all 128 simultaneously):

1. **Weight Gram** (data-free): $B_k = D_k^\top H D_k \in \mathbb{R}^{m \times m}$, where $D_k = W_{down}^{(k)}$. Entry $(i,j) = \langle d_{k,i}, d_{k,j}\rangle_H$.
2. **Coupling matrix** (Hadamard product): $\Theta_k = G_k \odot B_k$.
   - $\Theta_k[i,j] = \mathbb{E}[a_{k,i} a_{k,j}] \cdot \langle d_{k,i}, d_{k,j}\rangle_H$.
3. **Ridge-pivoted Cholesky** on $\Theta_k$ with shared $\lambda_r$:
   - Standard pivoted Cholesky: at each step $t$, pick the channel with the largest current residual diagonal, record it as $\pi_k(t)$, record its residual diagonal as the marginal gain $\sigma_{k,\pi_k(t)}$, then downdate the remaining diagonals (subtract the selected direction's contribution).
   - The ridge $\lambda_r$ regularizes the pivot inverse / stabilizes small residuals; keep it identical across all experts so gains are on one absolute scale.
   - Run to completion ($m$ steps) so every channel gets a rank and a gain.
4. **Store two length-$m$ vectors** and discard everything $m \times m$:
   - $\pi_k$ — the pivot order (permutation of the original physical channel indices; $m$ ints).
   - $\sigma_{k,\cdot}$ — the marginal gain per channel (monotonically non-increasing along $\pi_k$; $m$ floats).

**Total persistent artifact:** $L \times E \times m \times (\text{1 int} + \text{1 float}) \approx 47$M values, tens of MB. Nothing $m \times m$ ships. No weight is modified.

**Key properties to assert in tests:**

- $\sigma_{k,\pi_k(t)}$ is monotonically non-increasing in $t$ (required for the online threshold to cut a prefix).
- $\pi_k$ indexes original physical columns of $W_{down}$ (pivoted Cholesky reorders channels; it does **not** rotate them into new directions, unlike SVD). Runtime gathers real columns directly.

### Optional — Phase C: layout for gather efficiency

Pre-arrange each expert's $W_{down}^{(k)}$ columns (and, if you couple in/out later, the corresponding gate/up rows) in pivot order $\pi_k$, so that "select the top $t_k$" is a **contiguous prefix** — a cheap, gather-friendly slice at runtime.

### Online kernel (consumes only $g(x)$)

Per token, per layer:

1. Router gives the active set $\mathcal{A}$ (8 experts) and gates $g_k(x)$. *(standard MoE, unchanged.)*
2. Compute per-channel score $g_k(x)^2 \cdot \sigma_{k,j}$. Each expert's sequence over its pivot order is already sorted descending (since $\sigma$ is monotone and $g_k^2$ is a per-expert constant).
3. **8-way heap merge** over the 8 pre-sorted sequences; pop until the global budget $B$ is filled. This yields a prefix length $t_k(x)$ per expert. Cost $O(B \log 8)$. **No weights touched in this step.**
4. Gather the first $t_k$ columns of $W_{down}^{(k)}$ (contiguous, per Phase C), multiply by the corresponding $\phi_{k,j}(x)$, sum into the output.

Equivalently: a single global threshold $\tau$ (the $B$-th largest score) defines $t_k = \max\{t : g_k^2 \sigma_{k,\pi_k(t)} \ge \tau\}$, and $\rho_k = t_k / m$ emerges from that one shadow price.

**The online path never builds a matrix, never computes a covariance, never does SVD.** It multiplies stored scalars by live $g_k^2$ and does a bounded heap merge.

---

## Validation plan

Three-way comparison at matched active-parameter budget (50%):

1. **Expert-dropping baseline** ($k = 8 \to 4$).
2. **Current method** (linear-$g$ per-expert quota + ridge-leverage in-expert ordering).
3. **Level 1** (global $g^2$ threshold + pivoted-Cholesky nested ordering).

Expected outcome: L1 $\geq$ current method, and L1 $\approx$ expert-dropping. Interpretation:

- L1 matches expert-dropping but does not beat it → cross-expert redundancy is the true bottleneck → proceed to Level 3.
- L1 fails to match expert-dropping → likely an implementation bug (check $g^2$, monotonicity of $\sigma$, shared $\lambda_r$/$H$).

**Correctness checklist:**

- [ ] Activation Gram is uncentered (mean not subtracted).
- [ ] $\lambda_r$ and $H$ are identical across all experts and layers (cross-expert comparability).
- [ ] $\sigma_{k,\pi_k(t)}$ verified monotonically non-increasing.
- [ ] Score uses $g_k^2$, not $g_k$.
- [ ] $\pi_k$ maps to original physical columns; runtime gathers real weight columns, no reconstruction.
