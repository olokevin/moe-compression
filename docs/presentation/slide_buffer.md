# Slide Buffer — reworked Slides 6–8 (Motivation)

Drop-in replacement for the current Slides 6–8 in `midpoint_slides.md`. Everything
here builds one thesis and nothing else:

> **A token doesn't need 8 whole experts — it needs a sparse, token-specific
> subset of channels across those experts.**

Slide 6 states the claim and unpacks the two halves ("sparse subset" +
"token-specific"). Slide 7 proves *sparse subset suffices* (the activation
magnitude pattern). Slide 8 proves *it's token-specific* (a fixed set fails).

All numbers are measured on `Qwen/Qwen3-30B-A3B-Thinking-2507`, WikiText-2 test,
8,000 tokens routed to a single expert (layer 0, expert 0), replaying the exact
`oracle_mag` channel score `s_j(x) = SiLU(gate_j·x)·(up_j·x)` (`block.py`).
Capture: `scripts/expert_activation_capture.py`; plots:
`scripts/expert_activation_plot.py`; figures in `figs/`, numbers in
`figs/stats_activation.json`.

---

## Slide 6: The Right Granularity Is the Channel

**A token doesn't need 8 whole experts — it needs a sparse, token-specific
subset of *channels* across those experts.**

Recall the SwiGLU expert: `y = Σⱼ SiLU(gate_j·x)·(up_j·x)·W_down[:,j]`. The block
output is a **sum of K·I = 6144 rank-1 channel terms** — each channel
`(gate_row, up_row, down_column)` is a self-contained micro-expert that either
fires or stays silent for a token. MoE is *already* a sparse-computing framework;
the finest unit of that sparsity is the channel, not the expert.

The thesis has two testable halves:

1. **Sparse subset suffices** — per token, almost all of an expert's output
   magnitude sits in a small fraction of its channels (**slide 7**).
2. **Token-specific** — *which* channels those are is re-decided for every token,
   so no fixed subset can capture it (**slide 8**).

If both hold, the deployable move is: per token, keep only the top-B channels by
activation magnitude — a training-free, exact-at-budget sparsification at the
granularity the model actually uses.

---

## Slide 7: A Sparse Subset of Channels Suffices

**Per token, the activation magnitude concentrates on a handful of channels**

![One expert's SwiGLU activations are long-tailed and carried by a few uneven neurons](figs/fig_sparse_suffices.png)

Profiling a single expert (layer 0, expert 0) over 8,000 WikiText-2 tokens —
histogram of the SwiGLU output `hⱼ = SiLU(gate_j·x)·(up_j·x)`, and the per-neuron
survival count after masking the bottom 95% by |h| (note the log y-axis in (a)):

- **(a) The activation output is long-tailed.** Count falls exponentially as
  magnitude grows, with a huge spike at zero: **43% of all activations are
  ≈0** (|h| < 0.003). So even at **95% sparsity** the threshold sits at only
  |h| ≈ 0.04 — the large-magnitude tail (orange) stays on; only the small stuff
  (blue) is dropped. The output the token actually needs lives in a thin tail.
- **(b) …and that tail is carried by a few, very uneven neurons.** After masking,
  survival counts are wildly skewed: mean **403** activations/neuron, but **8
  neurons fire >5× the mean**. The magnitude is not spread evenly — a small
  subset of channels does the work.

**Takeaway:** keeping the top ~12.5% of channels by |h| already captures a median
**50%** of a token's total output magnitude (top 50% → 90%). A sparse per-token
subset is enough.

---

## Slide 8: The Subset Is Token-Specific

**Which channels matter is re-decided every token → no fixed subset works**

![The kept channel set changes token to token; a static ranking leaves magnitude on the table](figs/fig_token_specific.png)

Same expert, now looking at *which* channels each token keeps at budget ρ=0.125:

- **(a)** The per-token top-B keep mask (rows = tokens, cols = channels) has its
  lit columns moving from row to row: consecutive tokens routed to this expert
  share only **13%** of their kept channels.
- **(b)** Almost no channel is stable: **0%** are kept >95% of the time and only
  12% are kept <5% of the time — the mass sits near the budget ρ, not at 0 or 1.
  There is no small "always-important" set to prune to.
- **(c)** The cost of ignoring this: a **static** top-B (rank channels by their
  *mean* score) captures only **27%** of a token's magnitude at ρ=0.125, while the
  **per-token** top-B captures **51%** — a **+24-point** gap that is pure
  token-specific information, unreachable by any fixed ranking or offline prune.

**Conclusion — the thesis holds.** A token needs a sparse subset of channels
(slide 7), and that subset is token-specific (slide 8). This is why the method
must select **online, per token, at channel granularity** — and why the offline /
static-ranking baselines (slides 16–17) top out far below it.

---

### Notes for wiring into `midpoint_slides.md`

- These replace the current Slides 6–8. Slides 9+ ("Per-Token Channel Activation",
  design challenges, framework) follow unchanged — slide 8's conclusion hands off
  directly into slide 9's figure.
- Figures used: `figs/fig_sparse_suffices.pdf`, `figs/fig_token_specific.pdf`
  (both new). The old `fig_expert_overlap`, `fig_leverage_spectrum`,
  `fig_channel_granularity`, `fig_load_balance` are no longer referenced by 6–8
  (kept on disk; `fig_fixed_fails` / `fig_union_budgets` still used by slide 17).
- If you want the depth story too, the capture also has layer 24 / 46 experts
  (34 / 55); re-run `expert_activation_plot.py` pointing `_primary_target` at them
  for a supplementary slide.
