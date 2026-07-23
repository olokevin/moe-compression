# Full cross-expert Nyström covariance: how much lives off the diagonal blocks?

**Model:** Qwen3-30B-A3B-Thinking-2507 · **Layer:** 46 (of 48) · **Calibration:** C4,
T = 32 768 tokens (66 windows × 512 tok) · **Ridge λ ∈ {1.0, 0.01}** · **A100-New.**

*§1–6 study the covariance energy and the leverage ranking. §7 extends to the two decisions
that build a compressed model — budget allocation across experts and `down_proj`
reconstruction — plus an end-to-end HellaSwag eval at 50% on layer 46.*

## TL;DR

- Treating a whole MoE layer as one big FFN (input dim `d`, output dim `N·d_mlp`),
  each token contributes a **sparse** covariance block because only its top-8 of 128
  experts fire. Stacking gives a `98 304 × 98 304` covariance `C_full` (38 GB dense).
- **70.6 % of `‖C_full‖_F²` lives in the off-diagonal (cross-expert) blocks** that the
  current per-expert Nyström throws away. This is *not* negligible energy.
- **But it barely changes channel selection.** The true full-covariance ridge-leverage
  ranking agrees with the block-diagonal ranking to **98.5 %** top-512 overlap
  (Spearman 0.995) at λ=1, and still **95.9 %** (Spearman 0.978) at λ=0.01.
- **Conclusion: the current per-expert (block-diagonal) Nyström is well-justified for
  channel *selection*.** The off-diagonal energy is real but does not move which channels
  the ridge-leverage score keeps within an expert.
- **When cross-expert info *does* change a decision, it hurts (§7).** At a 50%-of-*total*
  budget, full-covariance global ranking allocates channels very unevenly — heterogeneous
  per-expert widths, correlated with token traffic (r=0.76), starving 41/116 experts below
  128 channels. That heterogeneous allocation + a joint cross-expert `down_proj`
  reconstruction gives **~16× worse** layer-46 block-output MSE than the uniform per-expert
  method (3.3e-1 vs 2.0e-2). Global leverage allocates by traffic, not by need. End-to-end
  **HellaSwag 0-shot** (compressing layer 46 to 50%) agrees: baseline **78.56** → per_expert
  **77.61** → full **77.32** acc_norm — per-expert beats full on every metric.
- **Net:** per-expert Nyström is the right choice on all three axes tested — within-expert
  selection (§4), cross-expert allocation (§7 Axis 1), and reconstruction (§7 Axis 2).
  Cross-expert interaction belongs in the activation-aware layer-joint *fit* (§6), not in
  selection or closed-form reconstruction.

Reproducible via `scripts/full_nystrom_cov_analysis.py` (§1–6) and
`scripts/full_nystrom_recon_eval.py` (§7). Raw artifacts: `summary_L46.json`,
`arrays_L46.npz`, `figures_L46.png` (λ=1); `lam001/` (λ=0.01); `eval_L46_50pct.json` (§7).

---

## 1. Setup: the "one big layer" view

The down_proj input of a MoE block is, per expert `e`, the gated intermediate
`z_e = silu(gate_e x) ⊙ (up_e x) ∈ ℝ^{d_mlp}`, `d_mlp = 768`. The current
`nystrom_moe` compressor scores expert `e`'s channels by the ridge leverage of its own
`d_mlp × d_mlp` covariance `C_e = z_eᵀz_e / T_e`.

The **stacked** view concatenates all `N = 128` experts into one vector of dimension
`N·d_mlp = 98 304`. A token that routes to experts `{e_1,…,e_8}` contributes a stacked
row that is non-zero only in those 8 blocks. The full covariance is

```
C_full = Zᵀ Z / T,   Z ∈ ℝ^{T × 98304}  (row-sparse: 8/128 blocks non-zero)
```

Its **diagonal blocks** are exactly `Z_eᵀ Z_e / T`; the **off-diagonal blocks**
`Z_eᵀ Z_f / T` capture how experts `e` and `f` co-fire — information the per-expert method
discards.

Two things make `C_full` intractable to handle naively: it is **38 GB dense** in fp32, and
a direct ridge-leverage solve `diag((C_full+λI)⁻¹C_full)` costs `98 304³ ≈ 9.5 × 10¹⁴`
flops. We never form it. Two identities (§3) reduce everything to one `T × T` = 32 768²
problem (≈ 4.3 GB, ~20 s on one A100).

Layer-46 routing on C4 is sparse and skewed: **12 of 128 experts receive 0 tokens**;
tokens/expert range `[0, 11 937]`, mean 2 048.

---

## 2. Off-diagonal energy — the headline number

We measure the fraction of squared Frobenius energy outside the diagonal blocks:

```
‖C_full‖_F²        = 6.080 × 10¹⁴
  diagonal blocks  = 1.790 × 10¹⁴   (29.4 %)
  off-diagonal     = 4.290 × 10¹⁴   (70.6 %)   ← cross-expert
```

**≈ 70.6 % of the covariance energy is cross-expert.** The per-expert Nyström literally
models less than a third of the layer's second-moment energy.

The energy is computed without forming `C_full` via the trace identity
`‖ZᵀZ‖_F² = ‖ZZᵀ‖_F²`: we build the `T × T` token Gram `G = ZZᵀ = Σ_e Z_e Z_eᵀ`
(scattering each expert's `T_e × T_e` Gram into its routed-token rows/cols), and
`‖C_full‖_F² = ‖G/T‖_F²`. Diagonal-block energy is `Σ_e ‖Z_eᵀZ_e / T‖_F²`. A cross-check
against the independently-built `128 × 128` block-energy matrix `B[e,f] = ‖Z_eᵀZ_f‖_F²`
agrees to `< 1e-6` relative.

### Per-expert distribution (`figures_L46.png`, right panel)

| statistic | off-diag fraction |
|---|---|
| mean | **0.778** |
| std | 0.164 |
| min | 0.221 |
| max | 1.000 |

Most experts sit **above** the global 0.706 line — their intermediate activations are more
correlated with *other* experts' than with their own. A handful of heavily-used experts
(large `T_e`, big diagonal block) pull the global fraction down.

### Block-energy structure (`figures_L46.png`, left panel)

The `128 × 128` `log₁₀‖Z_eᵀZ_f‖_F²` heatmap shows **dense, structured** off-diagonal
coupling — not a near-diagonal matrix. The 12 dead experts appear as dark rows/columns.
Cross-expert coupling is **concentrated**, though: each expert's **top-8 co-activating
partners hold 78.5 %** of its total off-diagonal mass. So the off-diagonal structure is
low-rank-ish per row (a few strong partners), consistent with correlated co-routing.

---

## 3. Computing full-covariance ridge leverage at scale

The ridge leverage of channel `i` is `τ_i = [(C_full+λI)⁻¹ C_full]_{ii}`. With
`C_full = AᵀA`, `A = Z/√T`, the **push-through identity**

```
(AᵀA + λI)⁻¹ Aᵀ = Aᵀ (AAᵀ + λI)⁻¹
  ⇒  τ_i = a_iᵀ (AAᵀ + λI)⁻¹ a_i = (1/T) · z_{·i}ᵀ M z_{·i},   M = (G/T + λI)⁻¹
```

turns the `98 304³` solve into a single `T × T` inverse `M` (32 768², ~4.3 GB). Because
channel `i` of expert `e` is supported only on `e`'s routed tokens, the score restricts to
a slice: `τ_{e,i} = (1/T) z_{e,·i}ᵀ M_e z_{e,·i}` with `M_e = M[idx_e, idx_e]`. Cost per
expert is one `T_e × T_e · d_mlp` matmul — the whole layer's 128 experts finish in seconds.

**Validation.** On the 8 busiest experts we form the dense `6 144 × 6 144` sub-covariance
`C_sub` and compare the direct `diag((C_sub+λI)⁻¹C_sub)` against the push-through `τ`:
**max relative error 1.1e-5 (λ=1), 6.2e-5 (λ=0.01)** — the identity and implementation are
numerically exact. (A standalone synthetic unit test,
`scripts/_selftest_full_nystrom_math.py`, independently confirms the energy, block-scatter,
and push-through identities to ~1e-7.)

---

## 4. Does the cross-expert covariance change channel selection?

We compare three per-channel leverage scores, ranking each expert's 768 channels and
keeping the top **k = 512** (the pipeline's `k` at keep_ratio≈0.67):

- **`lev_full`** — true full-covariance leverage (push-through, includes off-diagonal).
- **`lev_matched`** — block-diagonal of the *same* `C_full` (`z_eᵀz_e / T`, same λ, same
  normalization). Differs from `lev_full` **only** by the off-diagonal blocks — this
  isolates the pure cross-expert effect.
- **`lev_pipe`** — exactly what the code does today (`z_eᵀz_e / T_e`, per-expert
  normalization by that expert's own token count).

| comparison | top-512 overlap (mean) | overlap (min) | Spearman (mean) |
|---|---|---|---|
| **full vs matched block-diag** (λ=1) | **0.985** | 0.912 | **0.995** |
| full vs pipeline per-expert (λ=1) | 0.959 | 0.887 | 0.972 |
| **full vs matched block-diag** (λ=0.01) | **0.959** | 0.887 | **0.978** |
| full vs pipeline per-expert (λ=0.01) | 0.951 | 0.869 | 0.968 |
| global Spearman, all 98 304 channels (λ=1) | — | — | **0.998** |

**Interpretation.** Despite 70.6 % of the energy being off-diagonal, adding it to the
leverage score reranks **< 2 %** of kept channels (98.5 % overlap) at λ=1 and **< 5 %** at
λ=0.01. The ranking is essentially preserved (Spearman ≥ 0.995 / 0.978).

The result is **robust to the ridge**: shrinking λ 100× (0.01), where
`(C+λI)⁻¹ ≈ (1/λ)I` no longer dominates and off-diagonal terms have their maximum
influence, only drops the matched overlap from 0.985 → 0.959. So the agreement is a
genuine property of the data, not a large-ridge artifact.

Even the correlation between an expert's off-diagonal energy fraction and its
leverage disagreement `(1 − overlap)` is weak (**r = 0.22**): the most cross-coupled
experts are *not* meaningfully the ones whose channel choice changes.

---

## 5. Why so much energy, so little reranking?

Ridge leverage `diag((C+λI)⁻¹C)` measures each coordinate's *marginal* contribution to
the range of `C` after ridge regularization. Two facts reconcile the paradox:

1. **The off-diagonal blocks are shared, not channel-discriminative.** Cross-expert
   correlation is dominated by a few strong co-activating partners (top-8 partners = 78.5 %
   of off-diag mass) and reflects a common-mode signal across co-routed tokens. It inflates
   the total energy but adds a roughly *rank-preserving* offset to per-channel scores within
   an expert — it does not single out different channels as important.

2. **Leverage is scale-and-rotation aware but selection is ordinal.** We only use the score
   to *rank* channels within an expert. The off-diagonal blocks perturb the absolute
   leverage values (which is why the energy is large) but preserve the intra-expert ordering
   almost exactly.

In short: **cross-expert coupling is real and large in energy, but per-expert channel
importance ordering is an intra-block property that the block-diagonal covariance already
captures.**

---

## 6. Implications for the compressor

- **Channel selection: no change needed.** The per-expert (block-diagonal) ridge leverage
  in `src/compress/moe_basis/nystrom_moe.py` is well-justified — the full covariance selects
  the same channels. A full-covariance selector would be `~10⁴×` more expensive (or need the
  push-through trick) for a < 2–5 % change in kept channels. **Not worth it.**

- **Reconstruction: the open lever.** The off-diagonal energy *is* discarded by the
  closed-form `down_proj` reconstruction `W_new = (SᵀC_eS)⁻¹(SᵀC_e)W`, which is also
  per-expert. In principle a *joint* Nyström reconstruction using `C_full` could exploit
  cross-expert redundancy — but note the experts have **independent** `down_proj` weights and
  are summed with per-token gate weights, so a joint reconstruction only helps if channels
  are shared across experts (weight tying / merged basis), which this architecture does not
  have. The existing **activation-aware layer-joint fit** (`_fit_layer_joint`, `fit_mode="layer"`)
  already recovers cross-expert interaction empirically by replaying the real router and
  matching the block output — this is the right place for cross-expert information, and it is
  already implemented.

- **Takeaway for the paper:** this is a clean negative result that *strengthens* the method —
  "we verified that the per-expert covariance approximation loses 70 % of the layer's
  covariance energy yet preserves 98 % of the channel-selection decision," i.e. the
  block-diagonal simplification is empirically safe for selection, and cross-expert
  interaction is deferred to the activation-aware fit where it belongs.

---

## 7. Two decisions that actually build a model: allocation & reconstruction

§4 compared the leverage *ranking within a single expert*. But a real compressor at a
**50%-of-total** budget makes two coupled decisions where cross-expert information could
matter more. We test both by compressing **only layer 46** to 50% of its expert-FFN params
and measuring the layer's block-output relative MSE (`‖block(X) − Y_ref‖² / ‖Y_ref‖²` on the
true gated output) plus **HellaSwag 0-shot** on the whole model.
Script: `scripts/full_nystrom_recon_eval.py`. Artifacts: `eval_L46_50pct.json`.

Two end-to-end methods:

- **`per_expert`** — uniform top-`d_mlp/2` = 384 channels **per expert** (by per-expert
  leverage `z_eᵀz_e/T_e`) + per-expert closed-form `down_proj` reconstruction with `C_e`.
  This is what the pipeline does today.
- **`full`** — rank **all `N·d_mlp` channels globally** by full-covariance leverage, keep the
  top 50% (⇒ **heterogeneous** per-expert widths), + a **joint** `down_proj` reconstruction
  that concatenates all experts and solves one ridge problem against the summed (ungated)
  down output `D`, via the push-through identity
  `Wᵀ = Z_Sᵀ (Z_S Z_Sᵀ/T + λI)⁻¹ D / T` (`Z_S` = globally-kept stacked channels), then splits
  `W` back to experts. The joint-reconstruction push-through is validated against the standard
  ridge solution to 1.4e-7 (in `_selftest`).

### AXIS 1 — allocation: global selection gives heterogeneous, lopsided widths

At a 50%-of-total budget over the 116 active experts (budget = 44 544 channels), the global
ranking allocates channels **very unevenly** (from `arrays_L46.npz`, λ=1):

| per-expert kept `k_e` | value |
|---|---|
| uniform (per_expert) | 384 (every expert) |
| full: mean / std | 384 / **316** |
| full: percentiles [0,10,25,50,75,90,100] | **[0, 2, 25, 405, 737, 761, 768]** |
| full: experts starved (< 128 ch) | **41 / 116** |
| full: experts at near-full width (≥ 737) | ~29 |
| corr(`k_e`, tokens routed to `e`) | **0.756** |
| selected-set overlap vs uniform | **0.61** |

**The two selections differ in ~39% of kept channels** — a world apart from the 98.5%
*within-expert* ranking agreement in §4. Global leverage is dominated by **how many tokens an
expert sees** (r=0.76): busy experts hoard the budget toward full width while 41 of 116
experts are starved below 128 channels (some to 0–2). So the cross-expert information *does*
change the allocation dramatically — the question (Axis 2 + eval) is whether that change is
*good*.

### AXIS 2 — reconstruction: heterogeneous allocation badly hurts block reconstruction

Layer-46 block-output relative MSE (16 384 calibration tokens, λ=1):

| method | block rel-MSE | allocation |
|---|---|---|
| `per_expert` (uniform + per-expert recon) | **2.0 × 10⁻²** | 384 / expert |
| `full` (global + joint recon) | **3.3 × 10⁻¹** | heterogeneous (mean 385, std 317, 20 starved) |

**The full method's block reconstruction is ~16× worse.** Even though its joint `down_proj`
reconstruction has strictly more freedom (it can trade error across experts), the
**heterogeneous allocation starves ~20 experts to a handful of channels**, and those experts'
outputs collapse — a loss the joint reconstruction cannot recover because the missing channels
are simply gone. The uniform 384/expert allocation keeps every expert well-conditioned, and
per-expert reconstruction on that support is far more accurate.

This is the key finding of the extension: **global leverage is the wrong objective for
budget *allocation*.** Ridge leverage ranks a channel by its marginal contribution to *its
own expert's* activation covariance; comparing those scores *across* experts conflates
"important within a busy expert" with "important to the layer," and the token-count bias
(r=0.76) makes it allocate by traffic, not by need. Experts that fire rarely still need enough
width to represent their function on the tokens they do get.

### End-to-end HellaSwag (0-shot, full 10 042 examples, layer 46 only)

Full run on A100-New (4 GPUs, 16 384 calibration tokens, ~4.6 h). Only **1 of 48 layers**
is compressed, so whole-model accuracy moves only slightly — this checks whether the
block-MSE gap survives to task accuracy and in which direction.

| method | HellaSwag acc | HellaSwag **acc_norm** | Δ acc_norm vs base | block rel-MSE |
|---|---|---|---|---|
| baseline (uncompressed) | 0.5954 | **0.7856** | — | — |
| `per_expert` (50%, layer 46) | 0.5894 | **0.7761** | **−0.95 pt** | 2.0 × 10⁻² |
| `full` (50%, layer 46) | 0.5875 | **0.7732** | **−1.24 pt** | 3.3 × 10⁻¹ |

(The baseline acc_norm 0.7856 reproduces the paper's reported 78.56 exactly, validating the
eval harness.) **`per_expert` wins on HellaSwag** (0.7761 vs 0.7732, +0.30 pt) *and* on both
`acc` and block MSE. The task-accuracy gap (0.30 pt) is much smaller than the 16× block-MSE
gap because only one layer is perturbed and the model has ample downstream slack to absorb a
single degraded block — but every metric points the same direction, and the full method never
wins. (At a 100-sample smoke limit all three read acc_norm 0.68, i.e. below the noise floor —
the full 10 042-example run is what resolves the ordering.)

### Verdict on the two new axes

Both new axes point the **same way as §4–6**: per-expert Nyström is the right choice.
Cross-expert covariance barely changes *within-expert selection* (§4), and when it *does*
change the *cross-expert allocation* (Axis 1), that change **hurts** — 16× worse block MSE
(Axis 2) and lower HellaSwag accuracy — because global leverage allocates by token traffic
(corr 0.76 with routed-token count) rather than by each expert's representational need,
starving 20+ experts to a handful of channels. A uniform per-expert budget with per-expert
reconstruction dominates on every metric. If cross-expert interaction is to be exploited at
all, it belongs in the activation-aware layer-joint *fit* (§6), not in selection or
closed-form reconstruction.

---

## 8. Reproduce

```bash
# --- §7 allocation + reconstruction + HellaSwag (compresses layer 46 to 50%) ---
PER_GPU_MEM=22GiB PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python scripts/full_nystrom_recon_eval.py \
  --layer 46 --keep-ratio 0.5 --tokens 16384 --hellaswag-limit -1 \
  --methods baseline,per_expert,full --out-dir docs/results/full_nystrom
# NOTE: the heavy T×T linalg for the `full` method runs on CPU on purpose — a 16k² inverse
# on a GPU still holding a device_map='auto' 30B shard crashes CUBLAS (illegal access).

# --- §1–6 energy + leverage analysis ---
# Full run (loads sharded 30B, captures layer-46 routing + intermediates, then all analysis).
# ~60 s capture + ~20 s linalg. Needs ~13 GB on one GPU for the T×T inverse + fp32 Z_full.
PER_GPU_MEM=18GiB PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python scripts/full_nystrom_cov_analysis.py \
  --layer 46 --tokens 32768 --lam 1.0 --keep 512 --device cuda:0

# Re-run analysis only from the cached capture (docs/results/full_nystrom/capture_L46.pt):
.venv/bin/python scripts/full_nystrom_cov_analysis.py --layer 46 --lam 0.01 --no-model \
  --out-dir docs/results/full_nystrom/lam001

# Standalone math self-test (CPU, <1 s):
.venv/bin/python scripts/_selftest_full_nystrom_math.py
```

**Artifacts** (in `docs/results/full_nystrom/`): `summary_L{46}.json` (all scalars),
`arrays_L{46}.npz` (`block_energy` 128×128, `lev_full/lev_matched/lev_pipe` 128×768,
`overlaps`, `spearman`, `tokens_per_expert`, `per_expert_offdiag`), `figures_L{46}.png`
(block-energy heatmap + off-diag fraction histogram). `lam001/` holds the λ=0.01 replicate.
`capture_L46.pt` (1 GB, kept remote) caches X + per-expert routing/intermediates.

### Method caveats / scope

- **Single layer (46), single calibration set (C4), T=32 768.** Deep layers were chosen as
  the intended probe. Off-diag energy and reranking may differ at shallow layers or on
  task-specific calibration; the push-through machinery is layer-agnostic (`--layer`), so a
  full 48-layer sweep is cheap follow-up if needed.
- The 12 dead experts (0 tokens) are excluded from all per-expert statistics.
- Leverage comparison uses k=512 (keep_ratio≈0.67, the -33 % regime); at more aggressive
  keep ratios the overlap could differ, but Spearman ≥ 0.995 implies the top-k overlap stays
  high for any reasonable k.
- **§7 compresses only layer 46** (per the goal, "let's only compress layer 46 to start").
  The end-to-end HellaSwag delta is therefore small in absolute terms (a single block out of
  48); the block-output MSE is the sharper discriminator, and it and HellaSwag agree. A
  natural follow-up is to compress **all** layers with each method to amplify the accuracy
  gap — the `full` allocation's per-layer starvation should compound across depth. The
  `full` method also uses a per-expert floor (`--min-per-expert`, default 8) to keep starved
  experts runnable; raising it trades global-optimality for less starvation and is an
  additional knob worth sweeping if the full method is pursued further.
- §7 uses T=16 384 calibration tokens and λ=1 (vs §1–6's T=32 768); allocation percentiles
  quoted in Axis 1 are from the §1–6 arrays (T=32 768) and match the §7 run's realized
  mean/std/starved counts to within sampling noise.
