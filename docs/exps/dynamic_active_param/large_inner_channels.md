# Level 3 — global channel selection across **all** experts

*Branch `global_channels`. Model: Qwen3-30B-A3B-Thinking-2507 (E=128 experts,
K=8 routed, I=768 expert-FFN intermediate, 48 MoE layers). Masking simulation,
no fine-tuning. HellaSwag 0-shot.*

## The question

The Level-2 result (`q3_30b_dynamic_active.md`) is that ranking a token's channels
by their **exact per-token contribution magnitude** and keeping only the global
top-`B` across its *routed* experts loses almost nothing: at `B = 768`, i.e.
**12.5 % of the 8×768 = 6144 activated channels**, `oracle_mag` scores **76.84**
HellaSwag acc_norm against a dense **78.56**.

That selection is still fenced in by the router: only the 8 experts the router
picked are eligible. This experiment removes the fence. Keep the *same* per-token
budget `B = 768`, but draw it from **all `E·I = 128 × 768 = 98304` channels of all
128 experts**:

```
Level 2 :  S(x) = top-B of { s_{e,j}(x) : e ∈ top-K(x),  j ∈ [I] }      6144 candidates
Level 3 :  S(x) = top-B of { s_{e,j}(x) : e ∈ [E],       j ∈ [I] }     98304 candidates
```

Per-expert width *and* the set of participating experts both emerge from one
global per-token threshold. The router stops being a gate and becomes (at most) a
scalar prior on the score.

**This is an oracle / headroom study, not a speedup.** Scoring the full candidate
set requires `gate_proj` and `up_proj` for all 128 experts — 16× the activated
FLOPs. The question it answers is whether the *formulation* holds more accuracy
per channel than Level 2 does, i.e. whether per-token channel selection should be
allowed to cross the routing boundary at all. If it does, the follow-up problem is
predicting the winning channels cheaply (the same problem
[`channel_router.md`](channel_router.md) and [`efficient_scorer.md`](efficient_scorer.md)
attack one level down).

## What the router probability should do

Once experts are no longer selected, it is not obvious what `g` is *for*. Level 2
used the top-K renormalized routing weight `g_e` both as part of the score and as
the output weight, and that was well-defined because `Σ_{e∈top-K} g_e = 1`. In
Level 3 neither is forced. Three settings, plus one candidate fix:

| Setting | `router_mode` | `norm_match` | Score | Output weight per kept channel |
| --- | --- | --- | --- | --- |
| **1** | `none` | ✗ | `\|h_{e,j}(x)\|·‖W_down[e][:,j]‖` | `1` |
| **2** | `none` | ✓ | same as 1 | `1`, then the block output is rescaled per token to `‖y_orig‖` |
| **3** | `prob` | ✗ | `p_e·\|h_{e,j}(x)\|·‖W_down[e][:,j]‖` | `p_e` |
| **3n** | `prob` | ✓ | same as 3 | `p_e`, then rescaled to `‖y_orig‖` |
| **4** | `prob_renorm` | ✗ | same as 3 | `p_e / Σ_{e ∈ used(x)} p_e` |

where `h = SiLU(gate_proj(x)) · up_proj(x)` is the SwiGLU intermediate and
`p = softmax(router_logits)` is the **raw** softmax over all 128 experts — *not*
the top-K truncated-and-renormalized `g`.

Three things to note about this table before looking at any number.

- **Setting 1 breaks a conservation law.** The original block forms a *convex*
  combination (`Σ_e g_e = 1`), so its output is an average of expert outputs.
  Setting 1 sums 768 unit-weight channel contributions chosen for being the
  largest — it is a sum, not an average, and there is no reason for its scale to
  match. Setting 2 exists precisely to separate *direction* (did global selection
  pick a better channel subspace?) from *scale* (did dropping `g` blow up the
  norm?). This matters because a systematic per-layer gain error compounds
  multiplicatively over 48 layers.
- **Setting 3 breaks it in the other direction.** `Σ_{e∈top-K} p_e < 1` always —
  typically the top-8 of 128 hold well under half the softmax mass — so weighting
  by raw `p_e` systematically *shrinks* the block output. Setting 4 is the
  principled repair: renormalize `p` over the experts that actually ended up
  owning kept channels, which is exactly what `norm_topk_prob` does for the
  top-K set, generalized to the emergent set.
- **`‖W_down[e][:,j]‖` is load-bearing here in a way it was not at Level 2.** The
  Level-2 ablation (`oracle_mag` vs `oracle_mag_noW`) found the column-norm factor
  worth ~0.1–0.3 pt, i.e. nothing: within one expert it only breaks ties. At
  Level 3 channels are compared *across experts whose `down_proj` columns have
  different scales*, so the column norm is the cross-expert unit conversion, not a
  tie-break. All rows here use it (`use_col_norm: true`).

## The candidate-width sweep

Level 2 and Level 3 are the two endpoints of one axis: how many experts are
*eligible*. `candidate_experts: M` interpolates — the candidate pool is each
token's top-`M` experts by router probability, so

```
M = K   = 8    the Level-2 candidate set (same channels oracle_mag ranks over)
M = 128 = E    the full Level-3 set
```

`M = 8` is the control the three settings above lack: it holds the score, the
budget and the output weighting fixed and changes *only* the candidate width, so
`acc(M)` separates "global selection is a bad idea" from "the way setting 1/2/3
weights the output is a bad idea". It also matters practically — the all-expert
forward costs `M/E` of the full one, so if the curve peaks at small `M` the
formulation has a realizable version and if it is monotonically decreasing it
does not.

Because the top-`M` and top-`K` sets come from the *same* softmax ordering,
`top-K ⊆ top-M` for every `M ≥ K`; the reference forward is therefore always
computable and `M` never removes a routed expert.

## Pilot

A 200-doc pilot of setting 2 (`gc_probe`, `router_mode="none"`, `norm_match=True`,
`M = 128`) came back at **33.0 % acc_norm** — chance on HellaSwag is 25 % — with
`n_experts = 122.3`, `frac_top_k = 0.087`, `norm_ratio = 19.0` and `cos = 0.227`,
measured over 35 232 tokens × 48 layers. That is what motivated adding the
candidate-width sweep before spending the full budget: the `M = 128` endpoint was
already known to be destroyed, so the interesting question moved to *where between
8 and 128 it breaks*.

## Implementation

`src/dynamic_active_param/global_channels.py`, criterion `global_channels`, wired
into `src/train/merge_slim_eval.py` (its own branch, ahead of the Level-2 branch).
Per MoE block, per token chunk:

1. compute `h_{e,·}(x)` for **all** `E` experts → `(n, E, I)`;
2. score, optionally multiply by `p_e`, and take the per-token global top-`B` over
   the flattened `E·I` axis (reuses the tested `allocate.select_global_topB`);
3. `down_proj` only the `(token, expert)` pairs that own ≥1 kept channel, with the
   setting's output weight;
4. in the same pass, recompute the **unmodified** block output from the already
   materialized `h` (top-K experts, renormalized `g`) — this costs 8 extra
   `down_proj` per token and is what `norm_match` and every diagnostic below are
   measured against. Only the modified output carries to the next layer.

`token_chunk = 512` bounds the `(n, E, I)` fp32 score tensor at ~200 MB.

Correctness anchors in `src/dynamic_active_param/tests/test_global_channels.py`
(11 tests): with `E == K` and `B == E·I`, `prob_renorm` must reproduce the upstream
`Qwen3MoeSparseMoeBlock.forward` **exactly** (the renormalize-over-used-experts
weight then *is* the top-K renormalized routing weight), `none` must reproduce the
unit-weighted sum and `prob` the raw-`p`-weighted sum; the keep-mask must equal an
independent brute-force global top-`B`; `norm_match` must equalize the output norm
without rotating it; and chunking must be invariant.

### Diagnostics

Accumulated per MoE layer over the whole eval stream and printed at the end
(`summarize_global_channels`):

| Metric | Meaning |
| --- | --- |
| `n_experts` | distinct experts owning ≥1 kept channel, per token (vs `K = 8`) |
| `frac_top_k` | share of the `B` kept channels that lie inside the router's top-8 |
| `frac_top_1` | share inside the router's argmax expert |
| `p_mass` | raw softmax mass `Σ_{e ∈ used} p_e` of the experts actually used |
| `norm_ratio` | `‖y_mod‖ / ‖y_orig‖` **before** any norm matching |
| `cos` | `cos(y_mod, y_orig)` — scale-invariant, so this is the *direction* quality |
| `rel_err` | `‖y_carried − y_orig‖ / ‖y_orig‖` of what actually propagates |
| `oracle_overlap` | `\|S_global ∩ S_oracle_mag\| / B` at the same budget |

Two of these are worth reading together. `cos` is the ceiling on what *any* output
rescaling can achieve: the best possible relative error after an optimal per-token
scalar is `sqrt(1 − cos²)`. So `cos` says whether global selection found a better
*subspace*, and `norm_ratio` says how much of the damage is a fixable gain error.

There is also an analytic identity worth using as a live check: for
`router_mode ∈ {prob, prob_renorm}`, `oracle_overlap == frac_top_k` exactly.
Within a token, `g_e = p_e / Σ_{top-K} p_e` is a monotone rescaling of `p_e` that
is common to all its experts, so the global score and the `oracle_mag` score
induce the *same ordering* on the routed channels; whatever the global rule keeps
inside the routed set is automatically among `oracle_mag`'s top-`B`. The two
numbers therefore only differ under `router_mode="none"`, where dropping `p`
genuinely reorders the routed channels — and that difference is itself the measure
of how much the router prior matters *within* the routed set.

## Protocol

- `prune_ratio: 0.875` → `B = round(0.125 · K · I) = 768` channels per token per
  layer, identical to the Level-2 `−87.5 %` row.
- `k_min = 0` — no per-expert floor. Whether an expert participates at all is the
  emergent quantity under study.
- Masking simulation, no fine-tuning, `real_slim: false`, no LoRA.
- **HellaSwag 0-shot on the first 3000 of 10042 docs** (12000 loglikelihood
  requests). Full-set runs cost ~7 h each at ~1.6 it/s because of the all-expert
  forward; 3000 docs gives stderr ≈ 0.76 pt, enough to order settings that we
  expect to differ by several points. The **dense** and **`oracle_mag`** reference
  rows were re-run at the *same* limit (`ref_dense`, `ref_oracle_mag`), so every
  comparison in the results table is on identical documents. Full-set values for
  those two (78.56 and 76.84) are in `q3_30b_dynamic_active.md`.
- Configs: `configs/eval/qwen3_30b_a3b_global_channels_*_hellaswag.yaml` — five
  router/norm settings at `M = 128` (`s1_plain`, `s2_normmatch`, `s3_prob`,
  `s3n_prob_normmatch`, `s4_probrenorm`), five candidate-width points
  (`m8_normmatch`, `m16_normmatch`, `m32_normmatch`, `m16_probrenorm`,
  `m32_probrenorm`) and two references (`ref_dense`, `ref_oracle_mag`).
- Driver: `scripts/run_global_channels_sweep.sh <GPU_CSV> <PER_GPU_MEM> [tags...]`
  (sequential — the boxes are shared and the all-expert forward saturates them).
  Scraper: `scripts/collect_global_channels.py <log dirs...>`.

## Results

All 12 runs completed 2026-08-13/14. HellaSwag 0-shot **acc_norm** on the matched
first-3000-doc slice; stderr 0.83–0.91 pt on every row. Diagnostics are means over
48 MoE layers × 35 232 tokens.

> **Read this table as deltas, not levels.** `limit: 3000` takes the *first* 3000
> validation docs, which is a systematically harder slice than the full set — dense
> reads **67.23** here versus **78.56** full-set. What licenses the table is that
> the effect being measured survives the slice: dense → `oracle_mag` is **−1.16 pt**
> here against **−1.72 pt** full-set. Absolute values are not comparable to
> `q3_30b_dynamic_active.md`; differences within the table are.

| Row | `M` | `router_mode` | `norm_match` | acc_norm | Δ vs `oracle_mag` | `n_experts` | `frac_top_k` | `frac_top_1` | `p_mass` | `norm_ratio` | `cos` | `rel_err` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense (ref) | — | — | — | **67.23** | +1.16 | — | — | — | — | 1.000 | 1.000 | 0.000 |
| `oracle_mag` L2 (ref) | 8 | `g` (top-K renorm) | ✗ | **66.07** | 0.00 | ≤8 | 1.000 | — | — | — | — | — |
| `m8_normmatch` | 8 | `none` | ✓ | 64.60 | −1.47 | 7.99 | 1.000 | 0.140 | 0.407 | 6.57 | 0.852 | 0.525 |
| `m16_normmatch` | 16 | `none` | ✓ | 57.77 | −8.30 | 15.99 | 0.551 | 0.080 | 0.556 | 8.19 | 0.658 | 0.809 |
| `m32_normmatch` | 32 | `none` | ✓ | 46.07 | −20.00 | 31.97 | 0.300 | 0.045 | 0.702 | 10.39 | 0.477 | 1.008 |
| `m16_probrenorm` | 16 | `prob_renorm` | ✗ | 64.93 | −1.14 | 15.15 | 0.919 | 0.316 | 0.554 | 0.715 | 0.921 | 0.434 |
| `m32_probrenorm` | 32 | `prob_renorm` | ✗ | 63.53 | −2.54 | 23.47 | 0.888 | 0.313 | 0.644 | 0.627 | 0.913 | 0.496 |
| **1** `s1_plain` | 128 | `none` | ✗ | **28.90** | −37.17 | 119.78 | 0.071 | 0.010 | 0.926 | 20.98 | 0.305 | 20.70 |
| **2** `s2_normmatch` | 128 | `none` | ✓ | **33.13** | −32.94 | 122.34 | 0.087 | 0.012 | 0.961 | 18.97 | 0.224 | 1.236 |
| **3** `s3_prob` | 128 | `prob` | ✗ | **55.73** | −10.34 | 32.52 | 0.862 | 0.319 | 0.684 | 0.406 | 0.911 | 0.659 |
| **3n** `s3n_prob_normmatch` | 128 | `prob` | ✓ | **66.80** | **+0.73** | 30.30 | 0.871 | 0.295 | 0.678 | 0.397 | 0.895 | 0.427 |
| **4** `s4_probrenorm` | 128 | `prob_renorm` | ✗ | 62.80 | −3.27 | 30.70 | 0.873 | 0.311 | 0.680 | 0.601 | 0.908 | 0.523 |

Chance on HellaSwag acc_norm is 25.0. Raw `acc` (not length-normalized) orders the
rows identically: 51.90 dense, 51.83 `oracle_mag`, 51.27 `s3n`, 48.80 `s4`,
44.37 `s3`, 30.50 `s2`, 27.77 `s1`.

### The three settings

**Setting 1 — 28.90, i.e. essentially destroyed** (chance 25.0). The conservation
argument is confirmed numerically: `norm_ratio = 21.0`, so the block emits an
output 21× too large, and because each layer feeds the next the error compounds to
`rel_err = 20.7` — the modified output is twenty times further from the true one
than the true one is from zero. Predicting **zero** would be 20× better than
setting 1.

**Setting 2 — 33.13, +4.2 pt over setting 1 and still near chance.** So the
21× gain error accounts for only about a tenth of setting 1's damage. The rest is
not a scale problem at all, and `cos = 0.224` says so decisively: since the best
relative error any per-token rescale can reach is `sqrt(1 − cos²)`, setting 2's
floor is **0.975** no matter how it is normalized. The global top-`B` set simply
does not span the direction the block was supposed to output.

**Setting 3 — 55.73, a 22.6 pt jump over setting 2**, and with the scale fixed
(**3n — 66.80**) it *matches* `oracle_mag` (+0.73 pt, inside one stderr) and lands
0.43 pt under dense. Multiplying the score by `p_e` is worth more than everything
else in this experiment combined.

But look at *why* it works, in `frac_top_k`: **0.871**. With the router prior in
the score, 87 % of the budget goes right back inside the router's top-8, and
`n_experts` falls from 122 to 30. Setting 3n is not really global selection — it
is Level-2 selection with a 13 % cross-boundary leak. The `p_e` factor is not
adding information to the ranking; it is **re-imposing the routing decision** that
pure magnitude threw away. The formulation only becomes viable at the point where
it stops being the formulation.

**Setting 4 — 62.80.** Renormalizing `p` over the used expert set is worth +7.1 pt
over raw `p` but is **4.0 pt worse than simply norm-matching**, and it does not
even fix the scale it was designed to fix (`norm_ratio = 0.601`, still 40 % short).
The reason is visible in `n_experts = 30.7` and `p_mass = 0.680`: the denominator
sums `p` over *all* ~31 experts that own at least one kept channel, but the ~23
tail experts among them contribute a handful of channels each and almost no output.
They eat normalization mass without supplying output, so the experts that matter
get shrunk. The correct normalizer is not "how much router mass did I touch" but
"how much of each expert's output survived the budget" — which is exactly what the
empirical norm match measures, and why it wins.

### The winning branch, exactly — `s3n_prob_normmatch`

The only configuration that reached `oracle_mag` (66.80 vs 66.07). Config:
`configs/eval/qwen3_30b_a3b_global_channels_s3n_prob_normmatch_hellaswag.yaml`.

```yaml
prune_kwargs:
  prune_ratio: 0.875           # -> B = round(0.125 * K * I) = 768 channels/token/layer
  dynamic_alloc:
    enabled: true
    criterion: "global_channels"
    global_channels:
      router_mode: "prob"      # raw softmax p_e in BOTH score and output weight
      norm_match: true         # rescale block output to ||y_orig|| per token
      use_col_norm: true       # score includes ||W_down[e][:,j]||
      candidate_experts: 0     # 0 = all E = 128 experts eligible
      token_chunk: 512
```

Per MoE block (all 48 identical), per token `x`:

1. **Router, untruncated.** `p = softmax(gate(x))` over **all `E = 128`** experts.
   No top-k slice, no renormalization — `p` sums to 1 over 128, so each routed
   expert holds ≈0.05–0.15 and `Σ_{e ∈ top-8} p_e ≈ 0.4–0.5`.
2. **Candidates: everything.** `h_{e,·}(x) = SiLU(gate_proj_e(x)) ⊙ up_proj_e(x)`
   is computed at full width for **all 128 experts** → 98304 candidate channels.
3. **Score** `s_{e,j}(x) = p_e · |h_{e,j}(x)| · ‖W_down[e][:, j]‖₂`.
4. **Select** `S(x) =` per-token global top-`B` over the flattened 98304, `B = 768`.
   `k_min = 0`: no per-expert floor, no guarantee a routed expert survives.
5. **Modified output**, raw `p_e` as the gain, kept channels only:
   `y_mod = Σ_{e} p_e · W_down[e] · (h_{e,·}(x) ⊙ 1[(e,·) ∈ S(x)])`
   — only the ~30 experts owning ≥1 kept channel contribute.
6. **Reference output**, from the same already-materialized `h` (8 extra
   `down_proj`, no recompute): `y_orig = Σ_{e ∈ top-8} g_e · W_down[e] · h_{e,·}(x)`
   with `g = p_{top-8} / Σ p_{top-8}` — bit-for-bit the upstream forward.
7. **Norm match, then carry:**
   `y = y_mod · ‖y_orig‖₂ / max(‖y_mod‖₂, 10⁻⁶)`. Only `y` propagates; `y_orig` is
   discarded. Router logits pass through unchanged.

Masking simulation throughout — no weight is slimmed, no LoRA, no fine-tuning.

#### Why it works

**The router plays two separate roles, and this is the only row that gets both
right.**

*In the score*, `p_e` is the cross-expert unit conversion. Without it, magnitude
ranking is router-blind (`frac_top_k` 0.087 ≈ the 0.0625 you get by chance) and
`cos` collapses to 0.224; with it, `frac_top_k = 0.871` and `cos = 0.895`. That
fixes the **direction** of the output.

*In the output*, `p_e` is the wrong gain, for two compounding reasons: the top-8
hold under half the softmax mass, and only 768 of the 6144 routed channels survive,
so each surviving expert's own contribution is truncated too. Result:
`norm_ratio = 0.397`, an output 2.5× too small. That is a pure **magnitude** error.

The two errors live on orthogonal axes, and that is the whole reason the
composition works. `cos` is scale-invariant, so `p_e`'s contribution to direction
is untouched by its gain being wrong; and a per-token scalar rescale changes only
the gain, never the direction. So norm matching removes exactly the error `p_e`
leaves behind and nothing else — worth **+11.07 pt** (55.73 → 66.80), which
recovers the observable ceiling: at `cos = 0.895` the best achievable relative
error is `sqrt(1 − cos²) = 0.446`, and the measured post-rescale `rel_err` is
**0.427**, i.e. the rescale is essentially optimal.

**Why it beats the principled fix** (setting 4, `prob_renorm`, 62.80). Dividing by
`Σ_{e ∈ used} p_e = 0.680` supplies a 1.47× correction where 2.52× is needed, so it
under-corrects by 40 % — it only models the *router-truncation* half of the deficit
and is blind to the *channel-truncation* half. Worse, its denominator sums over all
~31 used experts, including ~23 tail experts that own a handful of channels and
contribute almost no output: they inflate the denominator without supplying
numerator. The empirical norm measures both halves at once and cannot be fooled
this way.

#### The honest caveat, and the deployable version

**As implemented, norm matching is an oracle**, not a mechanism: computing
`‖y_orig‖` requires the full-width unmodified forward, so the "compressed" path
needs the uncompressed one. Its value here is diagnostic — it proves the residual
damage in setting 3 was *entirely* a scalar gain error.

A deployable version has to predict that scalar. The data suggests it is easy:
`norm_ratio` for this row is remarkably flat across depth — **0.332, 0.411, 0.403,
0.362, 0.455, 0.447, 0.519** at layers 0/8/16/24/32/40/47 (mean 0.397, ~1.6× spread
over 48 layers). A single **per-layer constant** calibrated once on calibration data
would plausibly capture most of the +11 pt, at zero inference cost. What this run
cannot tell us is the *per-token* variance of `norm_ratio` around its layer mean
(only layer means were accumulated) — that is the one measurement a follow-up needs
before claiming a constant suffices.

Note this is worth testing on `oracle_mag` **itself**, independent of anything
global: `oracle_mag` also drops 87.5 % of its channels and must therefore lose
output norm too. `m8_prob_normmatch` (in flight) is that test.

### Why cross-expert magnitude fails (the mechanism)

`frac_top_k` is the whole story. A selector that ignored the router entirely would
put `K/E = 8/128 = 6.25 %` of its budget in the routed experts by chance. Pure
magnitude at `M = 128` puts **8.7 %** there — an enrichment of only **1.39×**.

**The magnitude of the SwiGLU intermediate is very nearly independent of the
router's choice.** Every expert has channels that fire hard on any given token,
because `|SiLU(gate(x))·up(x)|` is large whenever `x` has a large projection onto
that channel's rows, and the router is not selecting experts on that basis. So
`|h|·‖W_down‖` ranking spreads the 768-channel budget over **122 of 128 experts**,
~6 channels each, and computes a different function than the model.

This is the same conclusion as
[`expert-redundancy-is-not-expert-level`](q3_30b_dynamic_active.md): unrouted
experts are not silent, they are *wrong*, and no activation statistic recovers
which ones the model wanted. The router's probability is not redundant with
activation magnitude — it is the missing cross-expert scale, and the only thing
that supplies it.

### The candidate-width sweep — monotone, no sweet spot

| `M` | 8 | 16 | 32 | 128 |
| --- | --- | --- | --- | --- |
| pure magnitude (`none` + norm match) | 64.60 | 57.77 | 46.07 | 33.13 |
| `frac_top_k` | 1.000 | 0.551 | 0.300 | 0.087 |
| `cos` | 0.852 | 0.658 | 0.477 | 0.224 |
| `prob_renorm` | — | 64.93 | 63.53 | 62.80 |

Both families **decrease monotonically in `M`**. Widening the candidate pool never
helps, at any width, under either scoring rule. There is no small-`M` peak, so
there is no cheap realizable version of this idea to chase — which is the practical
verdict, since cost scales as `M/E`.

The `m8` control does the job it was added for. It decomposes setting 2's 32.94 pt
collapse into its two causes:

| Change | Rows compared | Cost |
| --- | --- | --- |
| output weighting: `g_e` → unit + norm match | `oracle_mag` → `m8_normmatch` | **−1.47 pt** |
| candidate pool: 8 experts → 128 | `m8_normmatch` → `s2_normmatch` | **−31.47 pt** |

So ~96 % of the damage is the candidate widening, not the weighting. Widening is
the thing that does not work.

One more reading of `m8_normmatch`: its `oracle_overlap` is **0.695**, so even
*inside* the routed set, dropping `g` from the score changes 30 % of the chosen
channels — and that costs only ~1.5 pt. Channel choice *within* an expert is
robust to the router term; channel choice *across* experts is not. The router
weight is doing cross-expert work exclusively.

### Depth structure

Every failing configuration fails **in the middle of the network**, not uniformly
(`cos` by layer):

| Row | L0 | L8 | L16 | L24 | L32 | L40 | L47 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `m8_normmatch` | 0.968 | 0.831 | 0.812 | 0.810 | 0.818 | 0.890 | 0.989 |
| `m16_normmatch` | 0.922 | 0.625 | 0.604 | 0.586 | 0.594 | 0.726 | 0.949 |
| `m32_normmatch` | 0.858 | 0.443 | 0.424 | 0.413 | 0.397 | 0.517 | 0.895 |
| `s2_normmatch` | 0.587 | 0.090 | 0.100 | 0.238 | 0.191 | 0.276 | 0.685 |
| `s3n_prob_normmatch` | 0.983 | 0.892 | 0.893 | 0.882 | 0.889 | 0.927 | 0.980 |

The first and last MoE layers tolerate cross-expert selection; layers 8–40 do not.
The norm blow-up also grows with depth (`s2`: `norm_ratio` 12.4 at L0 → 50.9 at
L47), so the two failure modes reinforce each other going deeper.

`s3_prob`/`s3n` show the mirror image at L47: `n_experts` jumps to 47–80 and
`frac_top_k` falls to 0.59–0.76, i.e. the last layer's router is flat enough that
even the `p`-weighted score reaches far outside the routed set — and that layer is
precisely where doing so is harmless.

### Correctness check that fired

The predicted analytic identity held exactly in every `prob`-family run:
`oracle_overlap == frac_top_k` to 4 decimals (`s3_prob` 0.8624/0.8624, `s3n`
0.8705/0.8705, `s4` 0.8726/0.8726, `m16_probrenorm` 0.9185/0.9185), and broke only
in the `none` family (`m8_normmatch`: 1.000 vs 0.695) — exactly as derived, since
`g_e` is a per-token monotone rescale of `p_e` and therefore cannot reorder a
token's routed channels. Two independently computed diagnostics agreeing to 4
decimals where theory says they must, and differing where theory says they may, is
a strong signal the selection and scoring paths are implemented as intended.

## Verdict

**The formulation does not improve on Level 2, and cannot be made to.** The best
global variant (`s3n`, 66.80) ties `oracle_mag` (66.07) within one stderr while
requiring `gate_proj`+`up_proj` for all 128 experts — **16× the activated FLOPs** —
to pick its channels. And it only gets there by putting the router probability back
into the score, at which point 87 % of its selections are inside the routed top-8
and it *is* Level 2 with a leak. Every rule that genuinely ranks channels on one
global scale, independent of routing, lands between 28.9 and 46.1.

So the 12.5 %-of-activated-channels result is a **within-routed-expert**
phenomenon. The eligible-expert set is not slack to be exploited; it is
information, and per-token magnitude does not contain it.

Two things worth keeping from the negative result:

1. **Per-token output-norm matching is a cheap, real win** — worth +4.2 pt on
   setting 1 and **+11.07 pt** on setting 3, and it beats the principled
   renormalization (setting 4) by 4.0 pt. Any budget-pruned MoE block loses output
   norm when it drops most of its channels, so this should be tried on `oracle_mag`
   *itself*: `m8_prob_normmatch` (in flight) is exactly that test — `oracle_mag`'s
   channel set with raw-`p` weights plus a norm match. If it clears 66.07, norm
   matching improves the current best method rather than merely rescuing a broken
   one.
2. **`frac_top_k` is a cheap go/no-go instrument.** Any future cross-expert scoring
   idea can be screened by measuring how much of its budget lands in the routed set
   *before* running an eval: at 6.25 % it is router-blind and will collapse; only
   values near 1 survive. That is ~2 orders of magnitude cheaper than an accuracy
   sweep.

### In flight

| Run | Purpose | Status |
| --- | --- | --- |
| `s3n_full` | full-set (10042 docs) confirmation of the one row that matched `oracle_mag`; comparator is the published 76.84 | running (~2.8 h) |
| `m8_prob_normmatch` | `oracle_mag`'s channel set + raw `p` + norm match — does norm matching improve Level 2? | running |
| `m16_prob_normmatch` | candidate width 16 under the winning treatment | queued |
| `m32_prob_normmatch` | candidate width 32 under the winning treatment | queued |

The three `*_prob_normmatch` rows complete the width curve under `prob` + norm
match, the only treatment that reached `oracle_mag`; the `none` and `prob_renorm`
families are already known to be monotone.
