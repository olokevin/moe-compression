# Per-Token Dynamic Parameter Activation for MoE

**Setting.** Keep all expert parameters resident in memory; per token, activate only a sparse atom-level subset that (1) minimizes final-loss impact and (2) fits a global active-parameter budget. Target architecture: Qwen3-30B-A3B (48 layers, $d=2048$, $E=128$ routed experts, top-$K=8$, $m=\texttt{moe\_intermediate\_size}=768$, SwiGLU, `norm_topk_prob=True`, no shared experts).

The two-level "expert score $\times$ within-expert ratio" structure is **not** assumed here — it falls out of the objective. Knowing which approximation produces it tells you exactly which statistics are needed and where the formulation breaks.

---

## 1. Setup and notation

Fix layer $\ell$ (dropped from notation). Token hidden state $x \in \mathbb{R}^d$. Router weights $\alpha_k(x) \ge 0$, $k \in [E]$.

SwiGLU expert:

$$
E_k(x) \;=\; W_D^{(k)}\Big(\sigma\big(W_G^{(k)}x\big)\odot\big(W_U^{(k)}x\big)\Big),
\qquad W_G^{(k)},W_U^{(k)}\in\mathbb{R}^{m\times d},\; W_D^{(k)}\in\mathbb{R}^{d\times m}.
$$

**The atomic unit.** Define the neuron activation and the down-projection column

$$
h_{k,c}(x) \;=\; \sigma\!\big(w_{G,c}^{(k)\top}x\big)\cdot\big(w_{U,c}^{(k)\top}x\big) \;\in\mathbb{R},
\qquad
d_{k,c} \;=\; W_D^{(k)}[:,c] \;\in\mathbb{R}^{d}.
$$

The layer output decomposes into $E\cdot m = 98{,}304$ rank-one **atoms**:

$$
y \;=\; \sum_{k=1}^{E}\alpha_k(x)\,E_k(x)
\;=\; \sum_{k=1}^{E}\sum_{c=1}^{m}\underbrace{\alpha_k(x)\,h_{k,c}(x)}_{=:~a_{k,c}(x)}\;d_{k,c}.
$$

Channel $c$ of expert $k$ owns $\{w_{G,c}^{(k)}, w_{U,c}^{(k)}, d_{k,c}\}$: exactly $3d$ parameters, **uniform across all atoms**. This makes the budget a cardinality constraint, not a knapsack.

**Decision variable.** Per token, a mask $M \in \{0,1\}^{E\times m}$, with $M_{k,c}=1$ meaning "activate". Write $u = \mathbf{1}-M$ (the drop indicator). $n_k = \sum_c M_{k,c}$, $\rho_k = n_k/m \in [0,1]$. Approximate output:

$$
\hat y(M) \;=\; \sum_{k,c} M_{k,c}\, a_{k,c}\, d_{k,c},
\qquad
\Delta \;=\; \hat y - y \;=\; -\sum_{k,c} u_{k,c}\, a_{k,c}\, d_{k,c}.
$$

**Budget.** The full model's active parameter count per layer is $K\cdot m\cdot 3d$. A target ratio $\mathrm{P}$ gives

$$
\boxed{\;\sum_{k=1}^{E} n_k \;\le\; B \;:=\; \mathrm{P}\cdot K\cdot m\;}
\qquad\Longleftrightarrow\qquad
\sum_{k=1}^{E}\rho_k \;\le\; \mathrm{P}\cdot K.
$$

> **Note on the constraint.** A common slip is to write $\sum_k \rho_k \le \mathrm{P}$; this is off by a factor of $K$. With $\mathrm{P}=0.25$: $B = 0.25\cdot 8\cdot 768 = 1536$ channels spread across **128** experts (mean $\bar\rho = 1536/(128\cdot768) = 1.56\%$). The regime is extremely sparse per expert — "keep all experts alive" only makes sense because most experts receive $n_k = 0$ or a handful of channels.

---

## 2. The exact per-token objective

Final-loss impact of the perturbation $\Delta$ at layer $\ell$, to second order (Gauss–Newton):

$$
\delta \mathcal{L} \;\approx\; g^\top \Delta + \tfrac12 \Delta^\top H \Delta,
\qquad H \;=\; \mathbb{E}\big[J_\ell^\top \nabla_y^2\mathcal{L}\, J_\ell\big] \succeq 0,
$$

where $J_\ell$ is the Jacobian of the network output w.r.t. $y$. If the mask is calibrated so $\mathbb{E}[\Delta]\approx 0$ (bias correction, §6), the linear term vanishes in expectation and

$$
\boxed{\;\delta\mathcal{L}(M) \;\approx\; \tfrac12\,\|\Delta\|_H^2 \;=\; \tfrac12\, u^\top \mathcal{G}(x)\, u\;}
$$

with the **per-token atom Gram matrix** $\mathcal{G}(x) \in \mathbb{R}^{Em\times Em}$:

$$
\big[\mathcal{G}(x)\big]_{(k,c),(k',c')}
\;=\; a_{k,c}(x)\,a_{k',c'}(x)\;\underbrace{\langle d_{k,c},\, d_{k',c'}\rangle_H}_{=:~\Theta_{(k,c),(k',c')}}.
$$

Compactly, with $a = a(x) \in \mathbb{R}^{Em}$ and $\Theta = D^\top H D$ ($D = [\,d_{k,c}\,]$ the $d \times Em$ dictionary):

$$
\mathcal{G}(x) \;=\; \big(a a^\top\big) \odot \Theta \;=\; \mathrm{diag}(a)\,\Theta\,\mathrm{diag}(a).
$$

This is exact — no expectation, no calibration averaging. Everything downstream approximates *this object*.

**Bridge to the bilinear framework.** Dropping column set $\bar S$ of $W_D^{(k)}$ is $\Delta W_D = W_D \Pi_{\bar S}$, and

$$
\big\|H^{1/2}\,\Delta W_D\, G_k^{1/2}\big\|_F^2
\;=\; \mathrm{tr}\big(\Pi_{\bar S}\Theta_k \Pi_{\bar S} G_k\big)
\;=\; \sum_{c,c'\in\bar S}\Theta_{k,cc'}\, G_{k,cc'}
\;=\; u_k^\top\big(\Theta_k \odot G_k\big) u_k,
$$

where $G_k = \mathbb{E}[\alpha_k^2 h_k h_k^\top]$. So the **calibration-averaged** version of $\mathcal{G}$ is precisely $\Theta \odot G$ — the same $\|S^{1/2}\Delta W H^{1/2}\|_F^2$ form, with $S \leftarrow H$ (output metric) and $H \leftarrow G$ (input second moment of `down_proj`). Per-token dynamic allocation is what you get by *not* taking the expectation over $a a^\top$.

This gives the covariance argument teeth: $\mathbb{E}_x[\mathcal{G}(x)] = \Theta \odot \mathbb{E}[aa^\top]$, and

$$
\mathbb{E}[a_{k,c}^2] = \mathbb{E}[\alpha_k^2 h_{k,c}^2] \neq \mathbb{E}[\alpha_k^2]\,\mathbb{E}[h_{k,c}^2].
$$

Static allocation optimizes against a fixed $\mathcal{G}$; per-token allocation sees the realization.

---

## 3. Separable relaxation ⟹ the two-level score

The exact problem $\min_{u\in\{0,1\}^{Em}} u^\top\mathcal{G}(x)u$ s.t. $\mathbf{1}^\top u \ge Em - B$ is a binary quadratic program (NP-hard in general). Two structured relaxations follow.

### 3(a) Diagonal approximation (atom orthogonality)

Assume $\Theta_{(k,c),(k',c')} \approx 0$ for $(k,c)\neq(k',c')$. Then

$$
\delta\mathcal{L}(M) \;\approx\; \tfrac12\sum_{k=1}^{E}\alpha_k(x)^2\sum_{c=1}^{m} (1-M_{k,c})\;
\underbrace{h_{k,c}(x)^2\,\|d_{k,c}\|_H^2}_{=:~s_{k,c}(x)}.
$$

**This is the source of the two-level structure, and it is exact under the diagonal assumption:**

$$
w_{k,c}(x) \;=\; \underbrace{\alpha_k(x)^2}_{\text{router}}\;\cdot\;\underbrace{\mathcal{E}_k(x)}_{\text{expert energy}}\;\cdot\;\underbrace{\hat s_{k,c}(x)}_{\text{within-expert ratio}},
\qquad
\mathcal{E}_k := \sum_c s_{k,c},\quad \hat s_{k,c} := s_{k,c}/\mathcal{E}_k,\;\; \textstyle\sum_c \hat s_{k,c}=1.
$$

Note $\alpha_k^{\mathbf{2}}$, **not** $\alpha_k$. The correct exponent depends on the error functional:

- squared / second-order error ($\|\Delta\|_H^2$) $\Rightarrow$ weight $\alpha_k^2$;
- first-order attribution ($g^\top\Delta$, as in the attribution-guided paper's Eq. 3–4) $\Rightarrow$ weight $\alpha_k$.

These give genuinely different allocations (the $\alpha^2$ version is far more selective across experts). For "impact on final loss" with a well-defined $H$, use $\alpha_k^2$ and be explicit that the first-order term is zeroed by mean correction.

### 3(b) Ridge-leverage relaxation (redundancy-aware)

The diagonal approximation over-counts: if $d_{k,c}$ and $d_{k,c'}$ are nearly collinear under $H$, dropping one is nearly free because the other absorbs it. The principled scalar per-atom score accounting for this is the **ridge leverage score** of $\mathcal{G}$:

$$
\tau_{k,c}(\lambda_r) \;=\; \Big[\mathcal{G}\big(\mathcal{G}+\lambda_r I\big)^{-1}\Big]_{(k,c),(k,c)} \;\in\;(0,1].
$$

Three facts make this the right object:

1. **It interpolates.** As $\lambda_r \to \infty$, $\tau_{k,c}\to \mathcal{G}_{(k,c),(k,c)}/\lambda_r = w_{k,c}/\lambda_r$ — the magnitude score of 3(a), up to global scale. As $\lambda_r\to 0$, $\tau \to$ exact leverage (pure subspace geometry, magnitude-blind). $\lambda_r$ is the redundancy-vs-magnitude dial.

2. **Effective dimension.** $\sum_{k,c}\tau_{k,c} = d_{\mathrm{eff}}(\lambda_r) = \mathrm{tr}\big(\mathcal{G}(\mathcal{G}+\lambda_r I)^{-1}\big)$. Selecting $n \gtrsim d_{\mathrm{eff}}(\lambda_r)\log(d_{\mathrm{eff}}/\delta)$ atoms by RLS sampling gives, w.h.p., a Nyström approximation $\tilde{\mathcal{G}}$ with $\|\mathcal{G}-\tilde{\mathcal{G}}\|_2 \le \lambda_r$ (Alaoui–Mahoney / Musco–Musco). $\lambda_r$ directly controls the residual.

3. **$\lambda_r$ is the Lagrange multiplier** (see §4.3).

**Per-expert (block-diagonal) form.** Assume cross-expert blocks of $\Theta$ vanish, i.e. $\langle d_{k,c},d_{k',c'}\rangle_H\approx 0$ for $k\neq k'$. Then $\mathcal{G} = \mathrm{blkdiag}(\mathcal{G}_1,\dots,\mathcal{G}_E)$ with

$$
\mathcal{G}_k(x) \;=\; \alpha_k(x)^2 \cdot \mathrm{diag}(h_k(x))\,\Theta_k\,\mathrm{diag}(h_k(x)),
\qquad \Theta_k = W_D^{(k)\top} H\, W_D^{(k)} \in \mathbb{R}^{m\times m}.
$$

The router probability enters the leverage score as a **relative shrinkage**: $\tau_{k,c}$ is computed from $\alpha_k^2\,\mathrm{diag}(h_k)\Theta_k\mathrm{diag}(h_k)$ against a **shared** $\lambda_r$. Experts with small $\alpha_k$ have their whole spectrum shrunk toward zero, so their leverage scores collapse — router weighting and budget allocation happen through the *same* mechanism. No manual blending of "expert importance" and "channel importance".

---

## 4. Solving the allocation: global threshold = water-filling = shared ridge

### 4.1 Exact solution under separability

Define, per expert, the sorted score sequence $s_{k,(1)}\ge s_{k,(2)}\ge\cdots\ge s_{k,(m)}$ and

$$
\epsilon_k(n_k) \;:=\; \min_{|S_k|=n_k}\;\sum_{c\notin S_k} s_{k,c} \;=\; \sum_{j>n_k} s_{k,(j)} \qquad\text{(sorted tail sum).}
$$

**Proposition 1 (convexity).** $\epsilon_k$ is nonincreasing and *convex* on $\{0,\dots,m\}$: its backward difference $\epsilon_k(n)-\epsilon_k(n-1) = -s_{k,(n)}$ is nondecreasing in $n$ since $s_{k,(n)}$ is nonincreasing. $\blacksquare$

Therefore

$$
\min_{n\in\mathbb{Z}_{\ge0}^E}\;\sum_{k}\alpha_k^2\,\epsilon_k(n_k)
\quad\text{s.t.}\quad \sum_k n_k \le B
$$

is a **separable convex resource allocation problem**. Greedy (award the next channel to the expert with largest marginal gain) is exactly optimal, and KKT gives a **single global threshold** $\lambda \ge 0$:

$$
\boxed{\;\alpha_k^2\, s_{k,(n_k)} \;\ge\; \lambda \;\ge\; \alpha_k^2\, s_{k,(n_k+1)} \quad \forall k\;}
$$

i.e. activate channel $(k,c)$ **iff** $\alpha_k^2 s_{k,c}(x) \ge \lambda$, with $\lambda$ set so $\sum_k n_k = B$.

**Corollary.** Because channel cost is uniform ($3d$ params per atom), the optimal mask is simply the **global top-$B$ of $w_{k,c} = \alpha_k^2 s_{k,c}$ over all $Em$ atoms**. The per-expert $\rho_k$ never need explicit computation; they *emerge* as $n_k = |\{c: w_{k,c}\ge\lambda\}|$.

> **Improvement over the attribution-guided paper's CBA.** Their Algorithm 1 bisects a scalar $\alpha$ on a coverage-target vector $\rho_g(\alpha)=\min(\alpha\phi_g,1)$, requiring a hand-designed importance prior $\phi_g$ plus square-root smoothing to compress dynamic range. That is a heuristic surrogate for a problem that, once you fix one scoring function $s$ and one error functional, has the exact threshold solution above. Coverage-maximization is what you *must* do when expert score $\phi$ and channel score $s$ live in incommensurable units. Under $\|\Delta\|_H^2$ they are commensurable — both contributions to the same scalar error — so the units problem, and its smoothing hyperparameters, disappear.

### 4.2 Closed-form water-filling under spectral decay

Suppose sorted within-expert scores follow a power law $s_{k,(j)} \approx C_k\, j^{-\gamma_k}$, $\gamma_k > 1$ (the concentration documented in the attribution-guided paper's Fig. 4a / Fig. 16 for Qwen3-30B-A3B). Then $\epsilon_k(n) \approx \tfrac{C_k}{\gamma_k-1}n^{-(\gamma_k-1)}$ and $\alpha_k^2 C_k n_k^{-\gamma_k} = \lambda$ gives

$$
\boxed{\;n_k(\lambda) \;=\; \Big(\frac{\alpha_k^2\,C_k}{\lambda}\Big)^{1/\gamma_k}\;}
\qquad \text{clipped to } [0,m],\qquad \lambda \text{ s.t. } \sum_k n_k(\lambda)=B.
$$

$\sum_k n_k(\lambda)$ is strictly decreasing in $\lambda$, so bisection costs $O(E\log(1/\varepsilon))$ — no sorting of $Em$ items. Reading it:

- $\alpha_k^2$ **scales the error curve vertically**; $\gamma_k$ (spectral decay) **sets the slope**. Since $\partial\log n_k/\partial\log\alpha_k^2 = 1/\gamma_k$, a highly concentrated expert (large $\gamma_k$) responds *weakly* to being highly routed — it recovers its contribution from few channels. This is the attribution-guided paper's Takeaway 3, now derived rather than observed.
- Two experts with equal $\alpha$ but different $\gamma$ get different budgets, automatically.

### 4.3 The ridge parameter is the multiplier

If $\mathcal{G}_k$ has eigenvalues $\sigma_{k,1}\ge\sigma_{k,2}\ge\cdots$, the best rank-$n$ residual is $\epsilon_k(n)=\sum_{j>n}\sigma_{k,j}$, so $-\epsilon_k'(n)=\sigma_{k,n}$. The effective dimension at ridge $\lambda_r$ is $d^k_{\mathrm{eff}}(\lambda_r)=\sum_j \sigma_{k,j}/(\sigma_{k,j}+\lambda_r)$, softly counting directions with $\sigma_{k,j}\gtrsim\lambda_r$. Setting $n_k = d^k_{\mathrm{eff}}(\lambda_r)$ enforces $\sigma_{k,n_k}\approx\lambda_r$, i.e.

$$
-\,\epsilon_k'(n_k) \;\approx\; \lambda_r \quad \forall k,
$$

which *is* the KKT stationarity condition of §4.1 with $\lambda \equiv \lambda_r$ (the $\alpha_k^2$ is already inside $\mathcal{G}_k$). Therefore:

> Compute global ridge leverage scores over all $(k,c)$ with a single shared $\lambda_r$, and pick $\lambda_r$ so that $d_{\mathrm{eff}}(\lambda_r)=B$. The resulting counts $n_k = \sum_c \tau_{k,c}$ are the water-filling allocation. One scalar, no blending, no coverage heuristic.

---

## 5. Statistics required

### Offline (per expert, once)

| Object | Shape | How | Cost / storage (per layer) |
|---|---|---|---|
| $H$ — output metric | $d\times d$, or $\mathrm{diag}$ | Empirical Fisher $\mathbb{E}[\nabla_y\mathcal{L}\nabla_y\mathcal{L}^\top]$, one backward pass | $d^2 = 4\text{M}$; diagonal $\Rightarrow 2048$ |
| $\Theta_k = W_D^{(k)\top}HW_D^{(k)}$ | $m\times m$ ×128 | dense GEMM | $128\cdot768^2 = 75\text{M}$; **diagonal only** $\|d_{k,c}\|_H^2 \Rightarrow 98\text{K}$ |
| $G_k = \mathbb{E}\big[\alpha_k^2 h_k h_k^\top\big]$ | $m\times m$ ×128 | forward hooks, EMA, weighted by $\alpha_k^2$ | same; diagonal $\Rightarrow 98\text{K}$ |
| $\mathcal{G}_k^{\text{static}} = \Theta_k \odot G_k$ | $m\times m$ | Hadamard | — |
| Pivoted-Cholesky order $\pi_k$ + prefix sums of $s_{k,(j)}$ | $m$ ×128 | greedy pivots on $\mathcal{G}_k^{\text{static}}$ | 98K + 98K; **nested / budget-agnostic** |
| Power-law fit $(C_k,\gamma_k)$ | 2×128 | log-log regression on $s_{k,(j)}$ | trivial; enables §4.2 |
| Mean-correction prefix sums $\mu_k(n) = \sum_{j>n}\mathbb{E}[h_{k,\pi_k(j)}]\,d_{k,\pi_k(j)}$ | $d\times m$ ×128 | cumulative | $d\cdot m\cdot E$ — expensive; store rank-truncated |

Note $\big[\mathcal{G}_k^{\text{static}}\big]_{cc} = \mathbb{E}[\alpha_k^2 h_{k,c}^2]\cdot\|d_{k,c}\|_H^2$ — the calibration-averaged atom energy, exactly the static score. The static ranking is a one-line consequence of the same object.

### Online (per token, per layer)

- $\alpha_k(x)$ — free, the router already computes it.
- $h_{k,c}(x)$ for every *candidate* atom. **This is the bottleneck** and determines deployability.

---

## 6. The systems obstruction, and three variants

Computing $s_{k,c}(x) = h_{k,c}(x)^2\|d_{k,c}\|_H^2$ for all 128 experts requires every expert's `gate` and `up` projections — $2\cdot E\cdot m\cdot d = 2\cdot128\cdot768\cdot2048 \approx 402$ MFLOP/token, versus $2\cdot 8\cdot 768\cdot 2048\approx 25$ MFLOP for the dense top-8 baseline. **16× worse than the thing being compressed.** Naïve full per-token scoring is a non-starter. Three ways out, in increasing accuracy and cost:

**Variant A — static ranking, dynamic budget.**
$s_{k,c}\leftarrow \big[\mathcal{G}_k^{\text{static}}\big]_{cc}$ (or static $\tau_{k,c}$). Only $\alpha_k(x)$ is per-token. Then $n_k$ follows from §4.2 with $O(E\log(1/\varepsilon))$ scalar work, and the kept set is a **prefix of the fixed pivot order $\pi_k$** — contiguous, nested, Matryoshka-friendly, zero gather cost, cache-coherent. Scoring overhead $\approx 0$. Captures $\mathrm{Cov}_x(\alpha_k^2, \cdot)$ across experts; cannot capture within-expert token-dependence of which channels matter.

**Variant B — candidate pre-filter + exact scoring.**
Score only experts surviving a bound. With $\bar{\mathcal{E}}_k := \max_x \mathcal{E}_k$ precomputed, expert $k$ contributes **no** atom above threshold $\lambda$ if

$$
\alpha_k(x)^2 \cdot \bar s_k \;<\; \lambda,
\qquad \bar s_k := \max_c \big(\sup_x h_{k,c}^2\big)\|d_{k,c}\|_H^2.
$$

A *safe* filter (no atom wrongly discarded) but needs $\lambda$ — bootstrap with $\lambda_0$ from Variant A, then verify. In practice: keep top-$K'$ experts by $\alpha_k^2\bar s_k$, $K'\in\{16,32\}$, then exact-score those. Cost $K'/E$ of the naïve, so $2$–$4\times$ the baseline gate+up.

**Variant C — sketched predictor.**
Precompute rank-$r$ factorizations $W_G^{(k)}\approx \tilde{W}_G^{(k)}\Pi$, $W_U^{(k)}\approx\tilde W_U^{(k)}\Pi$ with a **shared** sketch $\Pi\in\mathbb{R}^{r\times d}$, $r\ll d$ (e.g. $r=64$). Compute $\tilde x = \Pi x$ **once** ($rd$ FLOPs, shared across all 128 experts), then $\tilde h_{k,c}=\sigma(\tilde w_{G,c}^\top\tilde x)(\tilde w_{U,c}^\top\tilde x)$ at $2Emr = 2\cdot128\cdot768\cdot64 \approx 12.6$ MFLOP — **half** the baseline gate+up. Use $\tilde h$ only for *ranking*; recompute $h$ exactly on the selected $B$ atoms. Ranking error is second-order, bounded by $2\max_{k,c}|s_{k,c}-\tilde s_{k,c}|\cdot B$.

**Memory bandwidth caveat (the real cost).** Even with free scoring, gathering $B=1536$ *scattered* channels across 128 experts touches 128 weight tiles instead of 8, destroying the GEMM. Mitigation: coarsen the atom to a **channel block** of size $b\in\{16,32,64\}$, score blocks by $\sum_{c\in\text{block}}w_{k,c}$ (still separable, still uniform cost $\Rightarrow$ still exact top-$B/b$). The attribution-guided paper's AAR does this alignment post-hoc; doing it *inside* the objective rather than as a repair step is cleaner. Their Table 16 shows $m=128$ minimum channels near-optimal — consistent with block granularity being cheap in accuracy.

**Bias correction.** After masking, $\mathbb{E}[\Delta]\neq 0$ because $\mathbb{E}[h_{k,c}]\neq0$ under SwiGLU. Add

$$
\hat y \;\leftarrow\; \hat y \;+\; \sum_k \alpha_k(x)\,\mu_k(n_k),
\qquad \mu_k(n)=\sum_{j>n}\mathbb{E}[h_{k,\pi_k(j)}]\,d_{k,\pi_k(j)},
$$

a prefix-sum lookup given the nested pivot order. This is what licenses dropping the first-order term $g^\top\Delta$ in §2.

---

## 7. Loose ends worth deciding now

1. **Which $\alpha$?** With all experts alive, `norm_topk_prob=True` loses its obvious meaning. Using raw softmax over 128 gives $\sum_k\alpha_k=1$; low-$\alpha$ experts are water-filled to $n_k=0$, so the top-$B$ rule *reconstructs* a routing decision rather than presupposing one — the clean headline: **expert routing is a degenerate special case of atom selection.** Renormalizing over surviving experts post-selection changes the objective and breaks the KKT derivation (would need a fixed-point iteration).

2. **Asymmetry across $W_G, W_U, W_D$.** The atom formulation ties all three together — a channel is in or out. This is *stronger* than leaving `down_proj` uncompressed (as in MoBE); the atom framing has no room to spare $W_D$, making that gap structurally unavailable rather than something to argue about.

3. **The discarded off-diagonal term.** The gap between exact BQP and separable relaxation is $u^\top(\mathcal{G}-\mathrm{diag}\,\mathcal{G})u$. Ridge leverage recovers part of it in the shrinkage sense but not exactly. Honest characterization: RLS is exact when $\mathcal{G}$'s off-diagonal energy is well-modeled by an isotropic $\lambda_r I$ residual. Measuring $\|\mathcal{G}_k - \mathrm{diag}\,\mathcal{G}_k\|_F/\|\mathcal{G}_k\|_F$ per layer (within-expert and across-expert) is the single cheapest experiment that tells you whether the diagonal story holds — and it settles the block-diagonality question.

4. **Optimality claim scope.** Global top-$B$ is exactly optimal for the *separable* surrogate with *uniform* cost, not for the true loss. State this plainly; the value is a closed-form optimum where prior work needed a bisection heuristic over hand-tuned coverage targets.

---

## Compact statement of the method

$$
\boxed{\;M_{k,c}(x) \;=\; \mathbb{1}\Big[\;\alpha_k(x)^2\,h_{k,c}(x)^2\,\|d_{k,c}\|_H^2 \;\ge\; \lambda(x)\;\Big],
\qquad \lambda(x):\;\textstyle\sum_{k,c}M_{k,c}=B=\mathrm{P}\,K\,m\;}
$$

with $\lambda$ obtained either by quickselect on the atom scores ($O(Em)$) or by bisection on the power-law water-filling identity ($O(E\log\tfrac1\varepsilon)$), and with the magnitude score $h^2\|d\|_H^2$ optionally replaced by the ridge leverage score $\tau(\lambda_r)$, in which case $\lambda_r = \lambda$ — the threshold and the regularizer are the same number.
