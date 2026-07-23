
### Level-2: What to Measure

Level-2 requires either per-pair statistics or a global coupling sketch — an order of magnitude more offline cost than the current 57 MB artifact, and it may sacrifice the prefix-contiguity that keeps the grouped GEMM regular. Before paying that, we quantify **how much accuracy the block-diagonal approximation actually forfeits**, and **in what structural form** the cross-expert redundancy appears.

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

#### M3 — Where the redundancy lives

Two views of the same question — is overlap concentrated in the principal ("public") channels, leaving the tail private?

- **Coherence vs. pivot rank.** Bucket channels by their pivoted-Cholesky rank and plot the cross-expert coherence

$$
\mu_{(e,j),(f,l)} \;=\; \frac{\bigl|\Theta_{(e,j),(f,l)}\bigr|}{\sqrt{\Theta_{(e,j),(e,j)}\,\Theta_{(f,l),(f,l)}}}
$$

  against rank. A monotone decay is the "head-public, tail-private" signature.

- **Subspace geometry.** Principal angles / Grassmann distance between the leading eigen-subspaces of $\Theta_e$ for frequently co-activated expert pairs.

**Decision:** monotone decay licenses a low-rank ("publicness") correction that preserves prefix-contiguity; a flat profile forces full global selection. The measured $\mu_\ell$ also instantiates the coherence bound in the theory section.

#### M4 — Regime diagnostic: is the residual gap even a coupling problem?

Level 1 trails reduce-top-k by ~1 pt at $-37.5\%/-50\%$ but wins decisively at $-62.5\%/-75\%$. This pattern is not necessarily caused by cross-expert coupling, and must be ruled out first.

- **Induced allocation.** Compare the distribution of emergent prefix lengths $t_e$ against reduce-top-k's hard $0/1$ allocation. An over-flat $t_e$ profile indicates a dynamic-range problem in the score, not a structural one.
- **Sharpness sweep.** $s_{e,j}=g_e^{2\beta}\,\sigma_{e,j}$ for $\beta\in\{1,1.5,2,3\}$. This family contains Level 1 ($\beta=1$) and degenerates to reduce-top-k as $\beta\to\infty$, so it strictly contains both baselines and should not lose to either.
- **Entropy bucketing.** Per-token (Level 1 − reduce-top-k) accuracy delta bucketed by router entropy. Expected: reduce-top-k is near-lossless on low-entropy tokens (probability mass concentrated, the dropped experts contributed little), and we win on high-entropy tokens.

**Decision:** if the sweep closes the mid-budget gap, Level-2's target regime is redefined before any statistics are collected.
