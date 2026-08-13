# Cheap low-rank channel scorers — making the *realized* reduction match the *channel* reduction

> **Correction (2026-08-12).** The accounting below (and in `sparse_probe.md`) charges
> the scorer in *matmul-FLOPs / bytes of a separate proxy* and folds that into a
> "whole-FFN cut". That is the wrong frame for the target deployment, which is a
> **quantized served model**. There the right metric is **used parameters**, and the
> scorer should **reuse the served weights** rather than be a separate (lower-precision)
> copy. Re-accounted honestly, the low-rank family stays dead, the `sparse_probe`
> headline shrinks (−84.2% → −72.9% at the same operating point), and the surviving
> lever is input-sparse reads of the served weights. See the section
> **["Correction: count used parameters on a quantized model"](#correction-count-used-parameters-on-a-quantized-model)**
> at the end — start there.
>
> **Update (2026-08-12): the low-rank family is now closed, with a reason.**
> Investigation C brought the activation statistics in (SVD-LLM-style activation-aware
> SVD, the provably optimal shared basis, output whitening, quantized factors) and
> measured the one baseline nobody had: **a scorer that reads no `x` at all**, ranking
> channels by their long-run average score, gets recall 0.363 — while a rank-32
> activation-aware basis gets **0.353**. The reason is that the top direction of the
> score-error metric *is the mean token* (`cos² ≥ 0.998`), so rank buys the average and
> the target is the deviation from it. Read
> **["Investigation C"](#investigation-c--we-brought-the-data-in-and-low-rank-is-still-dead)**
> for the setup and numbers.

## The problem

`oracle_mag` ranks channels by the **true** SwiGLU intermediate
`s_{e,j}(x) = g_e·|SiLU(W_g x) ⊙ (W_u x)|_j`, so `gate_proj` and `up_proj` must run
at full width *just to decide what to keep*. Only `down_proj` gets narrowed:

| selector | up | gate | down | whole-FFN kept | nominal −75% is really |
| --- | --- | --- | --- | --- | --- |
| `oracle_mag` / `oracle_mag_noW` | full | full | ρ | (1+1+ρ)/3 | **−25.0%** |
| `oracle_up` | full | ρ | ρ | (1+2ρ)/3 | **−50.0%** |
| **`lowrank_scorer`** (this doc) | ρ | ρ | ρ | ρ + overhead | **−67 to −74%** |

**Idea.** Rank channels with a *cheap low-rank proxy* of that intermediate, built
from factored `W_up`/`W_gate`. The decision then precedes every full-width matmul,
so **all three** matrices can be gathered to budget and the scorer is the only
overhead.

## Results (HellaSwag 0-shot, −75% nominal channel cut)

Masking simulation, no fine-tuning, `k_min=0`. stderr 0.47–0.50pt. `cost` = compute
spent purely to rank, in units of one full `(I,H)` matmul; `recall` = agreement with
the `oracle_mag_noW` top-B at ρ=0.25 (random = 0.25).

| scorer | cost | whole-FFN cut | recall | acc | **acc_norm** |
| --- | --- | --- | --- | --- | --- |
| Dense | — | — | 1.000 | — | 78.56 |
| `oracle_mag_noW` (ref) | 2.000 | −25.0% | 1.000 | 59.77 | 78.36 |
| `oracle_up` (ref) | 1.000 | −50.0% | 0.685 | 57.81 | 75.31 |
| Level-1 `pivchol` (ref) | 0 | −25.0% | — | — | 63.60 |
| reduce-top-k 8→2 (ref) | 0 | −75.0% | — | — | 49.4 |
| `router_prob×activation` (ref) | 0 | −25.0% | — | — | 43.66 |
| `svd_r16` up+gate | 0.057 | −73.1% | 0.440 | 41.02 | 54.95 |
| `svd_r32` up+gate | 0.115 | −71.2% | 0.468 | 47.71 | 63.80 |
| **`btt_m2n2_r32` up+gate** | 0.229 | −67.4% | 0.479 | 50.30 | **66.83** |
| `svd_r16` up-only | 0.029 | **−74.0%** | 0.425 | 46.06 | 60.71 |
| `svd_r32` up-only | 0.057 | −73.1% | 0.444 | 48.60 | 63.94 |
| **`btt_m2n2_r32` up-only** | 0.115 | −71.2% | 0.456 | 50.53 | **65.97** |

### The four results that matter

**1. The accounting goal is achieved — ~3× more real reduction.** A rank-16 up-only
scorer costs 2.9% of one matmul, turning a nominal −75% channel cut into a **−74.0%
whole-FFN cut** (vs `oracle_mag`'s −25.0%). The overhead is ~1pp of the FFN.

**2. The frontier point: 65.97 acc_norm at −71.2% whole-FFN** (66.83 at −67.4%).
That is **+2.4pt over Level-1 `pivchol`** — the best *offline* selector, which
realizes only −25% — and **+16.6pt over reduce-top-k** at comparable realized cut.
But it stays **−11.5pt below `oracle_mag_noW`**: this buys real compression at an
accuracy cost, it does not match the oracle.

**3. At equal cost, drop the gate from the scorer and spend the budget on rank.**

| cost | up+gate | up-only | winner |
| --- | --- | --- | --- |
| 0.057 | 54.95 (`svd_r16`) | **63.94** (`svd_r32`) | up-only by **9.0pt** |
| 0.115 | 63.80 (`svd_r32`) | **65.97** (`btt_r32`) | up-only by **2.2pt** |

Predicted by the iso-cost recall analysis (up-only at 2× rank ties up+gate on
recall, Δ≤0.008), but the accuracy margin is far larger than recall suggested.
Comparing at equal *nominal rank* makes up+gate look better; that is the wrong axis.

**4. BTT wins on accuracy while *losing* on recall — so recall mis-ranks families.**
The block grid is consistently −0.02 to −0.03 recall against a plain SVD at equal
cost (5/5 cost points), yet produces the two best accuracy numbers in the study.
Over the six eval rows pearson(recall, acc_norm) = **0.65**, pearson(mass, acc_norm)
= 0.52.

![measured accuracy vs predicted recall/mass](figures/btt_dynamic/recall_vs_accuracy.png)

The BTT points (red) sit ~3pt above the SVD points (blue) at essentially the same
recall. Within a family recall orders correctly; *across* families it does not.
Plausible mechanism: BTT's errors are confined within channel blocks, so wrongly
dropped channels spread evenly over the output, whereas an SVD's errors concentrate
in the truncated directions — and losing a whole output direction hurts more than
losing scattered channels of equal mass. **Untested**: the equal-cost `svd_r64`
controls were launched and then stopped by request, so this remains the one open
question that would change the design.

**Methodological note.** Recall was the metric used to gate which configs got
evaluated, and it would have discarded the best one. Recall/mass are useful cheap
filters *within* a family; any ranking *between* families needs an accuracy probe.

## Supporting investigation A — recall vs cost (why the cheap end is hopeless)

`scripts/lowrank_scorer_recall.py` replays the exact per-token cross-expert
selection over **4 MoE layers × 8192 C4 tokens** (layers 6/22/38/46) and scores each
proxy against the `oracle_mag_noW` top-B. Full numbers:
`docs/results/btt_dynamic/recall{,_hirank}.json`.

![recall vs scorer cost](figures/btt_dynamic/recall_vs_cost.png)

| scorer | cost | whole-FFN cut | recall@ρ.25 | mass@ρ.25 |
| --- | --- | --- | --- | --- |
| `svd_r16` | 0.057 | −73.1% | 0.440 | 0.588 |
| `svd_r32` | 0.115 | −71.2% | 0.468 | 0.629 |
| `svd_r64` | 0.229 | −67.4% | 0.507 | 0.682 |
| `svd_r128` | 0.458 | −59.7% | 0.564 | 0.749 |
| `svd_r256` | 0.917 | −44.4% | 0.657 | 0.839 |
| `btt_m2n2_r256` | 1.833 | −13.9% | 0.802 | 0.941 |
| `oracle_up` (ref) | 1.000 | −41.7% | 0.685 | 0.816 |

**Recall grows only logarithmically with cost, so the family can never reach the
oracle at a cost that preserves the accounting win.** Matching `oracle_up`'s recall
needs `svd_r256` (cost 0.917 → −44.4% whole-FFN, *worse* than `oracle_up`'s
−41.7%). Reaching 0.80 recall needs `btt_m2n2_r256` at cost 1.83 — more expensive
than just computing the true gate and up (cost 2.0). Also: **deeper layers are
easier to proxy** (`svd_r32` recall 0.42 at layer 6 → 0.55 at layer 46) while
`oracle_up`'s recall *falls* with depth, which suggests a per-depth cost schedule.

## Supporting investigation B — the objective is the bottleneck, not the rank

Is low-rank *structure* the limit, or is spectral truncation just the wrong
objective? An SVD is Frobenius-optimal for approximating `W`, but the block needs a
good *ranking*. `scripts/lowrank_scorer_learned_probe.py` trains the identical
online form (hence **identical cost**) — `|A_2·SiLU(A_1 x)|` — on calibration
tokens instead of reading the factors off the SVD. Layer 46, 4 experts, 5734 train /
2458 held-out tokens; budgets are **per-expert**, so absolute recall is not
comparable to the table above — only the learned-vs-SVD delta is.

| rank (cost) | scorer | recall@ρ.25 | mass@ρ.25 |
| --- | --- | --- | --- |
| 16 (0.029) | SVD | 0.398 | 0.553 |
| 16 (0.029) | learned, listwise loss | **0.469** | **0.685** |
| 64 (0.115) | SVD | 0.442 | 0.603 |
| 64 (0.115) | learned, top-B BCE | **0.471** | **0.687** |

**+7.1pt recall / +13.2pt mass for free at rank 16**, and a *trained* r=16 beats an
SVD r=64 (4× the cost). Ranking losses beat magnitude regression. But learned
scorers **saturate at ~0.47 recall regardless of rank** — so there are two distinct
gaps and only the first is closed:

| gap | size | closed by |
| --- | --- | --- |
| spectrum → best low-rank ranking | 0.40 → 0.47 recall | training on a ranking loss (done) |
| best low-rank ranking → oracle | 0.47 → 1.00 recall | **closed negative** — investigation C: no low-rank form gets there, because rank buys the *average* token and the target is the deviation from it |

## Method and code

Each weight `W (I,H)` is cut into an `m×n` grid of `(a,b)` blocks, every block
truncated to rank `r`, with **singular values merged into the input-side core**
(so `h = R x` is already singular-value-scaled):

    W[i,j] ≈ L[i,j] R[i,j],   L: (a,r),   R: (r,b) = S_r V_rᵀ
    h[i,j] = R[i,j] x_j              cost r·m·H    ← the intermediate h
    ŵ_i    = Σ_j L[i,j] h[i,j]       cost r·n·I    ← per-channel proxy
    c      = r·(m·H + n·I)/(I·H)     →  whole-FFN kept = ρ + n_scorers·c/3

`m=n=1` is a plain global rank-r SVD; `m,n>1` is the BTT regime. On the brief's
first question — `h` itself is a per-block latent of dim `m·n·r`, not per-channel,
so it must be pushed through `L` to score the `I` channels; that `ŵ` form is what
is implemented and it is what lets all three matrices be cut.

`src/dynamic_active_param/lowrank_scorer.py` (factorization + online kernel + cost
accounting); criterion `lowrank_scorer` in `allocate._CROSS_EXPERT_CRITERIA`, scored
in `block._cross_expert_keep`, cores built per layer in `install_dynamic_alloc`,
config in `merge_slim_eval.py`. 16 unit tests
(`src/dynamic_active_param/tests/test_lowrank_scorer.py`), including the two that
pin the semantics: **at full rank the proxy reproduces `oracle_mag_noW` exactly**,
and the singular values provably live in the input-side core.

## Investigation C — we brought the data in, and low rank is still dead

### Why we did this

Everything above builds the scorer from `W` alone. An SVD of `W` is the best way to
approximate `W`, but that is not the job: the scorer only has to get `W x` right for
the `x` that actually show up. So the obvious objection to investigation A's verdict
is "you truncated the wrong thing" — use the *activation statistics* (SVD-LLM style)
and low rank might come back to life.

It does help. It is still dead. And we can now say **why**, which the earlier
saturation result could not.

### Exactly what was run

One shared setup for every row below, so the numbers are comparable:

* Qwen3-30B-A3B-Thinking-2507, MoE layers **6 / 22 / 38 / 46**, C4, seq-len 512.
  `E=128` experts, `I=768` channels/expert, `H=2048`, top-`K=8`.
* `Σ = E[x xᵀ]` (input covariance) and `M = Σ_e W_eᵀW_e` are fit on tokens
  **0–4095** and every scorer is evaluated on tokens **4096–5119** (1024 held-out
  tokens). No statistic ever sees the tokens it is scored on.
* Budget `ρ = 0.125`: keep 768 of the token's 6144 pooled (`K·I`) channels.
* **recall** = fraction of `oracle_mag_noW`'s top-B channels the scorer also picks
  (random = 0.125). **rel_err** = `‖y_full − y_kept‖/‖y_full‖` on the MoE block
  output — the quantity that predicts accuracy (`sparse_probe.md` ladder).
* No downstream eval was run for this investigation; see *Why no eval* below.

The methods, stated plainly:

| name | what the scorer reads | what it stores beyond the model |
| --- | --- | --- |
| `static_prior` | **nothing** — ranks channels by their average oracle score over the 4096 fit tokens, times `g_e` | one float per (expert, channel) |
| `insp_rN` | the true gate+up, but only the token's **top-`N` coordinates of `x`** by `|x_i|` (so `p = N/H`) | nothing — reads the served weights |
| `pcabasis_rN` | `A_e(Px)` with **one shared** `P` = top-`N` eigenvectors of `Σ` | `P` plus an `(I,N)` factor per expert |
| `actbasis_rN` | same, but `P` = top-`N` eigenvectors of `Σ^{1/2}MΣ^{1/2}` — **provably the best possible shared `P`** (derivation in `actaware_scorer.md`) | same |
| `awsvd_rN` | **per-expert** rank-`N` truncation of `W Σ^{1/2}` — this is SVD-LLM / "activation-aware SVD" | `(I,N)` + `(N,H)` per expert |
| `outwhiten_rN` | per-expert rank-`N` truncation of `D⁻¹W Σ^{1/2}`, `D = diag(rms of each channel's output)` — equalizes **relative** error across channels | same |
| `svd_rN` | per-expert plain SVD of `W` (investigation A's scorer, as the control) | same |
| `adapt_rN` | per-token largest `N` of `4N` shared-basis coefficients | `P` at rank `4N` + factors |
| `mix` | shared rank-`N/2` basis **plus** the top `N/2` coordinates of the leftover `x − PᵀPx` | both |
| `qbasis` / `qawsvd` | the above factors **quantized to 3 or 4 bits**, so a given rank costs ~5× less | quantized factors |
| `probe_q3_k25` | the `sparse_probe` incumbent: 3-bit copy of gate+up, read on the top-25% coordinates | a 3-bit second copy of gate+up |

### Result 1 — a scorer that ignores the token entirely does about as well

This is the finding. `static_prior` reads **no `x` at all** — it just knows each
channel's long-run average importance. Recall@ρ=0.125, and each scorer's *gain over
that free baseline*:

| scorer | L6 | L22 | L38 | L46 | mean | gain over free |
| --- | --- | --- | --- | --- | --- | --- |
| `static_prior` (reads nothing) | 0.322 | 0.324 | 0.400 | 0.406 | **0.363** | — |
| `actbasis_r32` | 0.288 | 0.300 | 0.398 | 0.428 | 0.353 | **−0.010** |
| `awsvd_r32` | 0.330 | 0.364 | 0.471 | 0.512 | 0.419 | +0.056 |
| `actbasis_r128` | 0.336 | 0.370 | 0.470 | 0.495 | 0.418 | +0.054 |
| `awsvd_r128` | 0.446 | 0.500 | 0.588 | 0.605 | 0.535 | +0.172 |
| `probe_q3_k25` | 0.587 | 0.593 | 0.657 | 0.678 | 0.629 | **+0.266** |

A rank-32 activation-aware basis is **worse than not looking at the token** (−0.010
on average, −0.035 and −0.025 at layers 6 and 22). Even at rank 128 the whole
low-rank family buys only about a fifth of what the quantized probe buys.

That reframes investigation A's recall numbers: a large part of the 0.44–0.47 those
scorers earned was never evidence they understood the token. It was the static
channel profile, which costs nothing. The same check on the *variation* alone —
Spearman after subtracting each channel's mean across tokens — says the same thing:

| scorer | L6 | L22 | L38 | L46 |
| --- | --- | --- | --- | --- |
| `actbasis_r32` | 0.159 | 0.183 | 0.273 | 0.413 |
| `awsvd_r128` | 0.365 | 0.431 | 0.508 | 0.598 |
| `probe_q3_k25` | **0.560** | **0.570** | **0.622** | **0.693** |

### Result 2 — why: the best low-rank direction *is* the average token

Here is the mechanism, and it is literal rather than hand-wavy.

If a scorer sees `x̃` instead of `x`, every channel's score is off by
`⟨w_j, x−x̃⟩`. Averaged over channels that error is `dxᵀM dx / I`, so the matrix
that says *how much a given mistake in `x` hurts the ranking* is
`C = Σ^{1/2} M Σ^{1/2}`. Two things about `C`:

| layer | eff. rank `tr(C)/λ₁` | `λ₁/tr` | share of that energy from `E[x]` alone | `cos²(E[x], top eigenvector)` | `‖E[x]‖²/E‖x‖²` |
| --- | --- | --- | --- | --- | --- |
| 6 | 3.11 | 0.322 | 0.314 | 0.998 | 0.122 |
| 22 | 3.50 | 0.286 | 0.284 | 0.999 | 0.122 |
| 38 | 2.43 | 0.412 | 0.411 | 1.000 | 0.174 |
| 46 | 1.79 | 0.558 | 0.549 | 0.999 | 0.191 |

**First**, of 2048 available directions, the score-damaging energy is spread over the
equivalent of only **1.8–3.5** of them. **Second — the key column — the top direction
*is the mean token*** (`cos² ≥ 0.998`), and the mean alone accounts for `λ₁` to within
0.01.

So a rank-1 data-optimal sketch is, to three digits, "replace every token by the
average token." That produces a score profile identical for all tokens, i.e. exactly
`static_prior`. A rank-`r` sketch spends its **largest and best-conditioned
component** rebuilding something that carries no per-token information at all.

Worse, activation weighting makes this *stronger*, not weaker: that one direction
holds **29–56% of the score-damaging energy** while the mean is only **12–19% of the
raw input energy**. The "optimal" objective concentrates the budget onto the
common-mode part of `x` — the one part a per-token ranking cannot use. Low rank is an
averaging operator, and the top-B set is the deviation from the average.

This is consistent with two earlier results: the static prefilter is weak
(`probe_prefilter_diag.py`: banning the bottom 25% of channels by keep-frequency
already loses 12–18% of top-B mass) and redundancy is not expert-level
(`expert-redundancy-is-not-expert-level`) — all the slack is per-token.

### Result 3 — activation-awareness works, and it is not enough

Credit where due: the SVD-LLM idea does exactly what its theory says. At **identical**
cost and storage, replacing plain `svd` with `awsvd` gains:

| rank | `svd` | `awsvd` | gain |
| --- | --- | --- | --- |
| 32 | 0.382 | 0.419 | **+0.037** |
| 88 | 0.461 | 0.496 | **+0.035** |
| 128 | 0.503 | 0.535 | **+0.031** |

A real, consistent, free improvement — which lands nowhere near the probe.

### Result 4 — the output-side question is answered on paper, not by a run

The brief asked to also try the output side (SVD of `yyᵀ`, or output whitening). Half
of that turns out to be the same operator we already ran. Because
`Σ_y = W Σ_x Wᵀ`, the top-`r` eigenvectors of `YYᵀ` are the left singular vectors of
`W Σ_x^{1/2}`, so

    U_r U_rᵀ W = U_r U_rᵀ (W Σ^{1/2}) Σ^{-1/2} = U_r S_r V_rᵀ Σ^{-1/2}

which *is* activation-aware input SVD. Checked in fp64 with an exact SVD:
`‖awsvd − output-side PCA‖/‖·‖` = **3.8e-13 / 9.5e-13 / 4.2e-13 / 4.9e-13** at layers
46/6/22/38. Not approximately equal — the same thing. So that branch needed no GPU
time.

What *is* genuinely different is output **whitening** (equalize relative error per
channel, which is arguably what a ranking wants). Tested as `outwhiten`, it is a dead
heat at every rank: **0.420 / 0.497 / 0.535** vs `awsvd`'s **0.419 / 0.496 / 0.535**
at r=32/88/128. Channel output norms are too uniform for reweighting to re-order
anything — the same wash the `‖W_down‖` and norm-weighted-coordinate ablations found.

### Result 5 — at matched used-parameters, just reading `x`'s big coordinates wins

Accounted the honest way (the used-parameter frame from the correction section
below — a separate stored factor adds `read/3`, while reading the *served* weights on
`p` coordinates costs `ρ + 2p(1−ρ)/3` because scored columns and kept rows overlap):

| scorer | per-token read | extra storage | used-param cut | recall | rel_err |
| --- | --- | --- | --- | --- | --- |
| `oracle_mag_noW` (full width) | 2.000 | none | −29.2% | 1.000 | 0.327 |
| **`insp_r128`** (top-6.25% coords, served weights) | 0.125 | **none** | **−83.9%** | **0.448** | **0.600** |
| `awsvd_r32` | 0.115 | +3.8% | −83.7% | 0.419 | 0.645 |
| `actbasis_r128` (shared) | 0.146 | +4.9% | −82.6% | 0.418 | 0.666 |
| `awsvd_r128` | 0.458 | +15.3% | −72.2% | 0.535 | 0.500 |
| `insp_r88` | 0.086 | none | −85.0% | 0.410 | 0.646 |
| `actbasis_r32` (shared) | 0.037 | +1.2% | −86.3% | 0.353 | 0.744 |

At essentially the same used-parameter cut (−83.9% vs −83.7%), **just reading the
token's largest coordinates beats the best low-rank scorer on both recall (0.448 vs
0.419) and output error (0.600 vs 0.645) — and stores nothing extra.** An *adaptive*
choice of an arbitrary basis beats a *fixed* choice of the provably optimal one. The
low-rank branch is not merely behind, it is behind while also demanding storage.

Note this also fixes an error in the first pass of this investigation: the screen
booked cost as `ρ + read/3` for *every* scorer, including the oracle, which
understates the oracle's cut (−20.8% rather than the correct −29.2%) and gives
`insp` no credit for the row/column overlap. The table above uses the correct frame.

### Result 6 — the decisive control: buy rank cheaply, still lose

If low rank were merely *underfunded*, quantizing the factors would rescue it: at 3
bits a rank-448 basis costs what fp16 rank-88 does. Measured (L46):

| scorer | per-token read | extra storage | recall |
| --- | --- | --- | --- |
| `qbasis_b3_r448` | 0.100 | +3.3% | 0.457 |
| **`probe_q3_k25`** (same read budget) | **0.098** | +13.0% | **0.678** |
| `qawsvd_b4_r768` | 0.709 | +23.6% | 0.750 |
| 4-bit probe on dense `x` (from `sparse_probe.md`) | 0.516 | +26% | **0.917** |

At the same read budget, rank loses to precision by **0.22 recall**. And at the high
end, a rank-768 4-bit factorization (0.750) is beaten by a plain 4-bit copy with *no
truncation at all* (0.917) at less cost. **Spending a byte on rank is strictly worse
than spending it on precision.** Rank is never the right purchase.

### Why no downstream eval

Deliberate. Every variant here is dominated by an operating point whose accuracy is
already **measured** (`sparse_probe` 3-bit/keep-25% = 74.56 HellaSwag), usually at
lower cost, so an eval could only confirm a loss. The rel_err ladder predicted the two
measured probe points to within 0.1pt, which is what justifies screening on rel_err.

### What this closes

Investigation B's open row — "best low-rank ranking → oracle, needs a richer
predictor class" — is **closed negative**, and next-steps 2 and 3 with it. The
ceiling is not the fit and not the rank; it is the functional form. Any operator that
*averages* (low rank, shared bases, PQ codebooks, static priors) rebuilds the free
part of the ranking and misses the part that matters. The two axes that do buy
per-token signal are **precision** and **which coordinates you read** — which is what
`sparse_probe.md` spends its budget on, and it holds the live frontier.

Code: `scripts/actaware_scorer_screen.py` (all families on one cost axis) and
`scripts/actaware_diag_static.py` (the static floor, centered Spearman, and the
mean-direction analysis of `C`). **Run the static-floor diagnostic — ~4 minutes, one
GPU — before proposing any new scorer**; it separates "buys per-token signal" from
"re-derives the free prior" and would have killed this whole family on day one.
Results in `docs/results/actaware/`; full write-up in `actaware_scorer.md`.

One implementation note that matters if these are rerun: use an **exact** batched SVD,
not `svd_lowrank`. At r=32, `svd_lowrank(q=r+16, niter=4)` — what
`lowrank_scorer.py` uses — is **13.8%** off the exact rank-`r` factors of
`W Σ^{1/2}`, because activation weighting concentrates the spectrum so hard. That
error is as large as the truncation being studied.

## Next steps

**Closed by investigation C — do not spend GPU time on these.** Steps 1–3 as
originally written are dead: the low-rank family cannot reach the oracle at any rank,
any objective (including the provably optimal data-aware one), or any factor
precision, and the reason is structural. Specifically:

* ~~Settle BTT-vs-SVD at equal cost~~ — still formally unmeasured on *accuracy*
  (the `svd_r64` pair was stopped mid-run), but it is now a question about the
  ordering of two variants that both sit ~0.2 recall below a cheaper, storage-free
  alternative. Not worth the runs.
* ~~Train the scorer~~ / ~~break the 0.47 ceiling with a richer form~~ — investigation
  C shows the ceiling is the *functional class*, not the fit. One sub-idea survives in
  altered form: 3(b) "predict a residual to a static prior" is now known to be the
  **only** part of a low-rank scorer that was ever working — the static prior *is*
  most of its recall. A learned residual is therefore still open, but it must be
  evaluated against `static_prior` as the baseline, not against random.
* ~~Per-depth cost schedule~~ — the depth trend is real (recall rises with depth) but
  it modulates a dead family.

Still open and worth doing:

1. **Sequence-level carry-forward.** Every scorer in this doc treats each token
   independently. Consecutive tokens share hot channels, so predicting the keep-set
   from an EMA of recently-realized activations would be a *zero-weight-read* scorer.
   Needs an order-preserving capture (current ones flatten across sequences). See the
   out-of-scope section below.
2. **Stack with reduce-top-k.** `reduce_topk` composes with `dynamic_alloc`; top-4 ×
   `sparse_probe` reaches beyond −85%, and the smaller `K·I` pool is an easier ranking
   problem.
3. **Allocation instead of a hard global top-B.** Every row here uses one global
   threshold; feeding the probe score into `coverage_alloc` is one run.

## Caveats

* Masking simulation (`real_slim: false`): arithmetic is exact at budget, but the
  realized speedup needs gather-based kernels. "whole-FFN cut" is an active-param
  accounting statement, as elsewhere in this series.
* Investigation A covers 4 of 48 MoE layers; per-layer recall varies ~0.1, so
  layer-averaged figures are good to ~0.02.
* Investigation B is a *probe*: one layer, 4 experts, per-expert (not pooled over
  `K`), train/test split by token within the same C4 slice. It shows headroom above
  the spectrum exists at equal cost; it does not show a single shared, generalizing
  head reaches those numbers.
* Investigation C covers the same 4 of 48 layers, 1024 scored tokens each (per-layer
  recall varies ~0.1, so layer averages are good to ~0.02 — well inside every gap it
  calls). Its `static_prior` baseline is fit per (expert, channel) on 4096 held-out
  tokens and still carries the router weight `g_e`, so it is not literally
  token-independent — `g_e` is genuinely per-token. That makes it the right baseline
  for *channel* selection, which is what these scorers claim to do, and it is if
  anything a slightly *strong* baseline. Its identification of the investigation-A/B
  saturation as the same phenomenon is a strong explanation, not a demonstration:
  those runs used per-expert BTT/SVD on a different budget axis and were not re-run
  under the static-floor diagnostic.
* `oracle_mag_noW`'s cost is booked as 2.0 (gate + up at full width); it needs no
  offline artifacts, so in latency terms it is cheaper than that number implies.

## Reproducing

```bash
# A — recall/mass vs cost (loads the 30B once, caches per-layer captures)
python scripts/lowrank_scorer_recall.py --layers 6,22,38,46 --tokens 8192 \
    --chunk 1024 --ranks 4,8,16,32 --grids 1x1,2x2,4x2 --out-dir docs/results/btt_dynamic
python scripts/lowrank_scorer_recall.py --layers 6,22,38,46 --tokens 8192 \
    --chunk 1024 --ranks 64,128,256 --grids 1x1,2x2 --out-dir docs/results/btt_dynamic_hirank
python scripts/lowrank_scorer_plot.py                    # -> figures/btt_dynamic/

# B — learned probe (reuses the cached capture; any single GPU)
python scripts/lowrank_scorer_learned_probe.py --layer 46 --rank 16 --experts 4
python scripts/lowrank_scorer_learned_probe.py --layer 46 --rank 64 --experts 4 \
    --out docs/results/btt_dynamic/learned_probe_r64.json

# Evals (A100, 8 GPUs) — configs:
#   configs/eval/qwen3_30b_a3b_dynamic_lowrank_{svd_r16,svd_r32,btt_m2n2_r32}_{upgate,uponly}_75_hellaswag.yaml
bash scripts/run_lowrank_scorer_sweep.sh   # svd_r32 pair, 4 GPUs each
bash scripts/run_lowrank_scorer_tail.sh    # remaining 4, 2 GPUs each, concurrent
bash scripts/run_lowrank_r64_pair.sh       # equal-cost BTT controls (not yet run to completion)

python -m pytest src/dynamic_active_param/tests/test_lowrank_scorer.py -q   # 16 tests

# C — activation-aware families + the static floor (1 GPU each, no model load,
#     reuses the _wd captures from scripts/probe_capture.py)
python scripts/actaware_scorer_screen.py --layers 46 --fit-tokens 4096 \
    --score-tokens 1024 --ranks 8,16,32,64,88,128,256 \
    --variants ref,pcabasis,actbasis,awsvd,outwhiten,svd,insp,adapt,mix \
    --gate-modes upgate,uponly --out docs/results/actaware/screen_L46.json
python scripts/actaware_scorer_screen.py --layers 6,22,38 --ranks 32,88,128 \
    --variants ref,actbasis,awsvd,outwhiten,svd,insp --gate-modes upgate \
    --out docs/results/actaware/screen_L6_22_38.json
# the decisive control: quantized factors (rank at ~5x off)
python scripts/actaware_scorer_screen.py --layers 46 --variants ref,qbasis,qawsvd \
    --ranks 32 --qbits 3,4 --qranks 128,256,448,768 --gate-modes upgate \
    --verify-equiv 0 --out docs/results/actaware/screen_qbasis_L46.json
# the static floor -- run this BEFORE proposing any new scorer (~4 min)
python scripts/actaware_diag_static.py --layers 6,22,38,46 --ranks 32,128 \
    --out docs/results/actaware/static_diag.json
```

Additional figures: `figures/btt_dynamic/{mass_vs_cost,svd_vs_btt,by_depth}.png`.

---

# Correction: count used parameters on a quantized model

The whole series so far measures the scorer in **units of one full `(I,H)` matmul**
and books a "whole-FFN cut" `= ρ + scorer_cost/3`. That answers "how many FLOPs
did scoring add", which is the wrong question. We will deploy on a **quantized
model**, where what costs money is the number of **parameters actually loaded** to
serve a token. Two consequences the old frame hides:

1. **A separate quantized proxy is not free — it is extra storage.** The
   `sparse_probe` result books a 3-bit gate+up proxy at `2·(3+16/128)/16·keep` of a
   matrix and calls the rest a −84% cut. But on a model *served at 4 bits* that proxy
   is a **second copy** of gate+up at 3 bits — `+50%` of the served expert weights —
   and its per-token read is *additional* traffic, not a substitute for anything.
   "Just quantize the proxy" is the move to avoid.
2. **The scorer should reuse the served weights.** If the model is served at `b`
   bits, reading a subset of those same weights to score costs **zero extra storage**;
   the only overhead is the *extra columns* you fetch for scoring that you would not
   already fetch to compute the kept channels.

**The honest used-parameter accounting.** Serve at `b` bits; score by reading the
top-`p` fraction of `x`'s coordinates (per token) from the *served* gate/up, take the
global top-B, then gather the `ρ = 1−prune_ratio` kept rows of gate/up/down for
compute. The distinct served entries touched, per matrix, are the **union** of
{kept rows} and {scored columns}:

    up, gate:  |{ρ·I rows} ∪ {p·H cols}| / (I·H) = ρ + p − ρ·p
    down:      ρ
    FFN kept   = [2(ρ + p − ρp) + ρ] / 3 = ρ + 2·p·(1−ρ)/3

`p=1` (read full gate+up) recovers `oracle_mag`'s `(1+1+ρ)/3`; there is no
separate-proxy term because there is no separate proxy. This is the number to
report.

## The honest frontier (measured)

Served at 4 bits; scorer = 4-bit (= serving precision, so it *is* the served
weight, no extra storage) read on the top-`p` coordinates. Block-output
`rel_err = ‖y_full − y_kept‖/‖y_full‖`, layer-averaged over L6/22/38/46, 8192 C4
tokens each (`scripts/probe_output_error.py --serve-bits 4`,
`docs/results/btt_dynamic/reuse_frontier.json`). Predicted HellaSwag uses the
`sparse_probe.md` ladder (−24.3 pt / unit rel_err, anchored at
`oracle_mag_noW` = 77.11); it is a bracket, not a measurement — 4-bit serving
shaves a little off the absolute, but the *shape* is what matters.

| selector (ρ=0.125)              | input `p` | rel_err | **FFN kept** | **used-param cut** | pred HS |
| --- | --- | --- | --- | --- | --- |
| `oracle_mag_noW` (full gate+up) | 1.00  | 0.326 | 0.708 | **−29.2%** | 77.1 |
| reuse `p=0.5`                   | 0.50  | 0.345 | 0.417 | **−58.3%** | 76.6 |
| reuse `p=0.375`                 | 0.375 | 0.367 | 0.344 | −65.6% | 76.1 |
| reuse `p=0.25`                  | 0.25  | 0.409 | 0.271 | **−72.9%** | 75.1 |
| reuse `p=0.1875`                | 0.1875| 0.445 | 0.234 | −76.6% | 74.2 |
| reuse `p=0.125`                 | 0.125 | 0.501 | 0.198 | **−80.2%** | 72.9 |
| `oracle_up` (full up, cut gate) | —     | 0.540 | 0.417 | −58.3% | 71.9 |
| separate 3-bit proxy, `p=0.25`  | 0.25  | 0.433 | 0.271 + storage | −72.9% *+50% store* | 74.5 |

**Three findings.**

1. **The `sparse_probe` headline was an accounting artifact.** Its "−84.2% at
   3-bit/keep-25%" is, honestly re-accounted, **−72.9%** (reuse, `p=0.25`), and its
   "parity at −70.3%" (4-bit / dense `x`) is just `oracle_mag` reading the served
   weights at full width — **−29.2%**, not −70%. Reading full-width 4-bit gate+up *is*
   computing the oracle. Roughly 11 points of "cut" at the goal point, and the entire
   parity claim, came from charging the scorer as a cheap separate object.

2. **"Don't just quantize the proxy" is right, and measurable.** A separate 3-bit
   proxy on a 4-bit base is both *worse* (rel_err 0.433 vs 0.409 for the 4-bit reuse
   probe at the same `p=0.25`) and *costs +50% storage*. Below-serving precision loses
   on both axes; the only precision worth using is the serving precision, reused.

3. **The one real, deployable win is input-sparse reuse — and it is genuinely good
   at equal cost.** At `p=0.5` it lands at exactly `oracle_up`'s used-parameter cost
   (−58.3%) with rel_err 0.345 vs 0.540 → **≈ +4.7 pred-pt over `oracle_up` for free**.
   At `p=0.25` it buys −72.9% at ~75.1. This is the honest version of the whole
   program: not a −84% miracle, but a strict Pareto improvement over the previous best
   *deployable* selector.

**Where the "< 10% of a matrix" scorer goal lands.** Under reuse the scorer adds
**zero storage**, so a storage budget is trivially met; the binding constraint is
scoring *traffic*. The marginal scoring read per matrix is `p·(1−ρ)`; `< 0.10`
means `p ≲ 0.11`. That operating point (`p≈0.125`) is **−80.2% used-params at
≈72.9 HS**, i.e. **≈ −4.2 pt below `oracle_mag_noW`**. So the literal goal — oracle
parity at ρ=0.125 with a sub-10% scorer — is **still not met**, same verdict as
`sparse_probe.md`, but now stated in the accounting we will actually be charged.

## Brainstorm, with verdicts

The ask was for decompositions/structures that expose the output-activation ranking
with a very light proxy, including unstructured sparsity. Measured on L46, recall vs
the `oracle_mag_noW` top-B at ρ=0.125, `p=0.25`-equivalent budgets:

| structure | idea | recall | verdict |
| --- | --- | --- | --- |
| per-token top-\|x\| reuse | read served gate/up on the token's largest coords | **0.716** | **the frontier** |
| per-row weight sparsity | keep each channel's top-`t` weights (unstructured), dense `x` | 0.622 | **dead** — worse *and* needs a separate indexed structure + dense `x` |
| static input subset | one fixed high-energy column set for all tokens | 0.558 | **dead** — per-token top-coords overlap the global set only 0.375 |
| discriminability-weighted input | pick coords by `x_h²·Var_j(W_{j,h})` | ≈ top-\|x\| | **no-op** — `CV(Var_j W)=0.042`; its top-25% is 93.8% identical to plain top-\|x\| |
| low-rank / BTT / SVD of weights | (this doc, above) | 0.44 | **dead** |
| activation-aware SVD (SVD-LLM) | truncate `W Σ^{1/2}` instead of `W` | 0.512 | **dead** — +0.037 over plain SVD but still below `insp` at equal cost, and it needs storage (investigation C) |
| output-side SVD of `yyᵀ` | rank-truncate in output space | — | **identical operator** to activation-aware SVD (proved, 3.8e-13); output *whitening* is a dead heat |
| optimal shared cross-expert basis | best possible single `P` for all experts, from `Σ^{1/2}MΣ^{1/2}` | 0.428 | **dead** — at rank 32 it is *below* a scorer that reads no `x` |
| separate quantized proxy | 3-bit copy of gate/up | (0.675) | **wrong frame** — extra storage on a quantized base |

The unifying reason, re-confirmed three independent ways here (row-clusterability ≈
random, `Var_j(W)` flat, static column subset weak): **the expert weight rows carry
near-maximal information per weight and vary per token only through `x`.** So no
static or low-rank *structure* of the weights beats simply reading the served weights
on the coordinates `x` says matter — and reusing them is what makes that free of
storage.

Investigation C adds the sharper version of that reason: any operator that
**averages** — low rank, shared bases, PQ codebooks, static priors — rebuilds the part
of the ranking that is *free* (the mean channel profile) and misses the part that
matters (the per-token deviation). Measured: the top direction of the score-error
metric is the mean token itself (`cos² ≥ 0.998`), and a rank-32 optimal basis scores
below a scorer that reads no `x` at all.

## What is actually still open (out-of-scope directions)

The per-token, per-token-independent regime is exhausted. Real headroom is elsewhere:

1. **Batch / prefill amortization of the scoring read (regime, not free lunch).** The
   `2p(1−ρ)/3` overhead is *per token* only in batch-1 decode. In a batch, tokens
   routed to the same expert share the served-weight fetch: the union of their
   top-`p=0.25` coordinate sets is 0.60 of `H` at batch 4, **0.94 at batch 16**, so
   amortized scoring cost per token falls to `0.059` (b=16), `0.016` (b=64). *But* the
   same batching inflates the compute gather (different tokens keep different channels)
   toward dense, so this scheme is fundamentally a **small-batch/decode** optimization
   — the honest frontier above is the decode frontier, and it is the right one to
   quote. (`docs/results/btt_dynamic/reuse_frontier.json` context;
   union-growth measured inline.)

2. **Sequence-level channel carry-forward (untested, most promising).** Everything
   here scores each token independently. Consecutive tokens in a sequence route to
   overlapping experts and hot channels; predicting the keep-set from an EMA of
   recently-realized activations would be a **zero-weight-read** scorer. Not testable
   on the current captures (they flatten across sequence boundaries) — needs an
   order-preserving capture, which is the next instrument to build.

3. **Learned residual to a static per-channel prior.** The static prior alone is weak
   (forbidding the bottom-25%-by-frequency loses 12–18% of top-B mass), but a *small*
   correction to it may be far cheaper to learn than the full ranking. Investigation B
   (above) shows learned heads help at equal cost but saturate at ~0.47 recall in the
   low-rank form; a prior-plus-correction head is a different functional class and is
   the one learned direction not yet closed.

## Reproducing

```bash
# honest reuse frontier (served at 4 bits; probe = served weights, input-sparse)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python scripts/probe_output_error.py \
    --serve-bits 4 --layers 6,22,38,46 --ratios 0.25,0.125 \
    --probes "4:1.0,4:0.5,4:0.375,4:0.25,4:0.1875,4:0.125,3:0.25" \
    --out docs/results/btt_dynamic/reuse_frontier.json
# re-account with FFN kept = rho + 2*p*(1-rho)/3 (union of kept rows + scored cols)
```
