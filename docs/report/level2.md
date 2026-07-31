# Dynamic Channel Selection per Token

**TL;DR.** (1) Keeping `gate_proj`/`up_proj` at full intermediate width while activating only a *very sparse* per-token subset of `down_proj`'s input channels is enough to stay at dense accuracy — even a 7/8 cut loses < 2 pt. (2) Ranking channels *across* the active experts (cross-expert) does not beat ranking them *within* each expert — the gap is inside one stderr at every budget.

## Setup

Model `Qwen/Qwen3-30B-A3B-Thinking-2507` (E = 128 experts, K = 8 active per token, per-expert intermediate width I = 768, so K·I = 6144 active channels per token). We fix a per-token channel budget B = ρ·K·I (ρ = kept fraction) and zero the un-kept SwiGLU intermediate channels **before** `down_proj`, which still runs at full width with original weights — masking simulation, no fine-tuning, no reconstruction correction, so the reported accuracy is exact at that active budget. HellaSwag 0-shot / MMLU 5-shot; dense = 78.56 (HS) / ≈ 79.5 (MMLU); stderr ≈ 0.4 pt. Source: `docs/exps/dynamic_active_param/q3_30b_dynamic_active.md`; code in `src/dynamic_active_param/{block,install,allocate}.py`.

An FFN expert computes `y_e(x) = W_down^(e) · h_e(x)` with intermediate `h_e(x) = SiLU(gate·x) ⊙ (up·x)`, and the block output is `Σ_e g_e·y_e(x)` over the K routed experts (`g_e` = router weight). The question in both experiments below is *which* B of the K·I intermediate channels to keep per token — they differ in what information the ranking is allowed to use.

## Experiment 1 — Per-token activation vs. an offline surrogate

**Motivation.** Any offline/static selector ranks channels from calibration averages, so it can only see the router weight `g` — it cannot tell which channels a *specific* token actually lights up. But `down_proj` is linear, so `y_e(x) = Σ_j h_{e,j}(x)·W_down^(e)[:,j]` and the intermediate dimension is not globally redundant: every channel matters for *some* token, yet for any *given* token only a few carry the output. This experiment measures how much accuracy that per-token information is worth by ranking on the **exact contribution magnitude** each channel makes to the token output. Zeroing channel `j` of expert `e` removes exactly `g_e·h_{e,j}(x)·W_down^(e)[:,j]`, whose L2 magnitude is

$$
s_{e,j}(x) \;=\; g_e \cdot \big|h_{e,j}(x)\big| \cdot \big\lVert W_{\text{down}}^{(e)}[:,j]\big\rVert_2 .
$$

The factors `g_e` and `‖W_down[:,j]‖₂` put all experts' channels in the same physical units, so all K·I channels of a token are pooled and the global top-B is kept (per-expert keep-counts emerge unevenly, no floor). This is an **unreachable ceiling** — it reads the true `|h(x)|`, which no offline method can — and it upper-bounds every offline selector.

**Compute / save.** No offline artifact. Online it materializes each token's `(K, I)` intermediate (work the model already does), scores, and takes the global top-B; the only precomputed state is `col_norm[e,j] = ‖W_down[:,j]‖₂`, read straight off the weights at install — no calibration pass, nothing to persist.

| Reduction (ρ kept)         | 50% (0.50)      | 62.5% (0.375)   | 75% (0.25)      | 87.5% (0.125)   | MMLU 75%        |
| --------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| Best offline (within-expert)| 74.26           | 70.54           | 63.60           | 44.15           | 70.81           |
| **Per-token activation**    | **78.54**       | **78.76**       | **78.28**       | **76.84**       | **80.53**       |

Per-token selection stays essentially at dense at every budget — losing < 2 pt even at a 7/8 cut — beating the best offline method by +4.3 / +8.2 / +14.7 / +32.7 pt on HellaSwag and +9.7 pt on MMLU. The entire headroom above offline methods is per-token activation information.

**Takeaway — full-width `gate`/`up`, sparsely-activated `down_proj`.** We must produce the full-width `h(x)` to know which channels matter per token, but then only a small per-token-selected fraction of `down_proj`'s input channels needs to fire and accuracy holds at dense. This makes the efficient factorization of an expert asymmetric and composable: **compress `gate`/`up` by rank** (MoBE/Nyström-style shared bases in `src/compress/`; ranking needs only the *values* of `h(x)`, not full-rank matrices), and **compress `down_proj` by per-token input sparsity** (only B of I columns touched per token, changing token to token). The open problem is a *deployable* online estimate of `|h(x)|` — the ceiling shows the accuracy is there to be had.

## Experiment 2 — Within-expert vs. cross-expert ranking (both offline)

**Motivation.** The best offline selector ranks channels *within each expert* — its scoring matrix is **block-diagonal**, with no term coupling distinct experts. So when a low-probability expert co-fires with a high-probability one, it may spend budget on channels the other already covers. This experiment asks whether adding cross-expert coupling to the offline ranking recovers that waste. The cross-expert variant builds a shared **public basis** offline (the top directions many experts' `down_proj` write into), force-keeps the single best-carrying channel per public direction across the co-activated experts (load shared knowledge once), and re-spends the freed budget on each expert's private channels — same router-only online cost, but the ranking now sees off-diagonal structure.

| Reduction (ρ kept)         | 50%   | 62.5% | 75%   | 87.5% | MMLU 75% |
| --------------------------- | ----- | ----- | ----- | ----- | -------- |
| Within-expert (block-diag)  | 74.26 | 70.54 | 63.60 | 44.15 | 70.81    |
| Cross-expert (+ coupling)   | 74.31 | 70.51 | 63.46 | 44.66 | 71.03    |
| Δ (cross − within)          | +0.05 | −0.03 | −0.14 | +0.51 | +0.22    |

**Takeaway — cross-expert coupling buys nothing.** Every delta is inside one stderr and the sign is not even consistent. Restoring off-diagonal coupling does not change *which* channels are kept. This corroborates the stacked-covariance finding — cross-expert coupling holds ~70% of the covariance *energy* yet shifts channel *selection* by < 2% — so the block-diagonal within-expert selector already captures all the offline-recoverable signal. Combined with Experiment 1, the value is in **online per-token activation, not offline cross-expert structure**: a heavier cross-expert method (needing ≫ 57 MB of pairwise expert statistics) is not worth building; the headroom is closed by estimating `|h(x)|` online instead.
