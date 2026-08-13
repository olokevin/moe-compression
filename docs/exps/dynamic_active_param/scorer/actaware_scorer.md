# Activation-aware low-rank scorers — why bringing the data in does not save low rank

> **Accounting note.** The `cB` / "whole-FFN cut" columns in this doc use the older
> `kept = ρ + cB/3` convention, which charges a scorer as a separate object in units of
> one full matmul. `btt_dynamic.md`'s correction section supersedes it: on a quantized
> served model the right metric is **used parameters**, where reading the served weights
> on `p` coordinates costs `ρ + 2p(1−ρ)/3` (kept rows and scored columns overlap) and a
> separately-stored factor also carries storage. Re-accounted that way the *ordering*
> here is unchanged but the margin against low rank widens, because `insp` stores
> nothing while every low-rank row stores factors — see
> **Investigation C, Result 5** in `btt_dynamic.md` for the corrected table. Recall,
> mass, rel_err and every conclusion below are unaffected; only the cost column's frame
> changed.

## The question

`btt_dynamic.md` measured the plain low-rank scorer family dead: a truncated SVD of
`W_up`/`W_gate` saturates at recall ~0.47 against the `oracle_mag_noW` top-B no
matter the rank. Every scorer in that study was built from `W` **alone**, which
leaves an obvious objection — a Frobenius-optimal truncation of `W` is the wrong
truncation, because the scorer does not need to approximate `W`, only `W x` for `x`
drawn from the data. So: bring the input/output activation statistics in.

Target, unchanged: a scorer costing **< 10% of one expert `(I,H)` matrix** that
emits indices, so `up`/`gate`/`down` are all gathered to budget. At ρ=0.125 that is
a **−84.2% whole-FFN cut**; the bar is `oracle_mag_noW`'s 77.11 HellaSwag.

## Answer

**Activation-awareness is a real and large win *within* the low-rank family — and
the family is still dead.** Layer-averaged (6/22/38/46, ρ=0.125), activation-aware
SVD beats plain SVD at identical cost by **+0.037 / +0.035 / +0.031 recall** at
r=32 / 88 / 128. The mechanism works exactly as advertised. It just does not matter,
because:

| scorer | cB | whole-FFN cut | recall | mass | rel_err |
| --- | --- | --- | --- | --- | --- |
| `oracle_mag_noW` (target) | 2.000 | −20.8% | 1.000 | 1.000 | 0.327 |
| **`sparse_probe` q3/keep-25%** (incumbent) | **0.098** | **−84.2%** | **0.629** | **0.835** | **0.431** |
| `awsvd_r128` (best activation-aware ≤0.46) | 0.458 | −72.2% | 0.535 | 0.741 | 0.500 |
| `outwhiten_r128` | 0.458 | −72.2% | 0.535 | 0.743 | 0.503 |
| `svd_r128` (weight-only reference) | 0.458 | −72.2% | 0.503 | 0.716 | 0.514 |
| `insp_r128` (top-|x| coords, no rank) | 0.125 | −83.3% | 0.448 | 0.640 | 0.600 |
| `awsvd_r32` | 0.115 | −83.7% | 0.419 | 0.600 | 0.645 |
| `actbasis_r32` (shared basis, cheapest) | 0.037 | −86.3% | 0.353 | 0.507 | 0.744 |

**The incumbent 3-bit probe beats every activation-aware variant at 4.7× less cost.**
At the goal budget (cB ≈ 0.10) the best activation-aware scorer reaches recall 0.42
against the probe's 0.63. Via the fitted ladder (−24.3 HellaSwag pt per unit rel_err)
the rel_err gap of 0.21 is ≈5pt — so this family is not merely behind, it is behind
by more than the entire margin the probe gives up to the oracle.

## Why — the finding that generalizes

The reason is not that rank is too small. It is that **a low-rank sketch of `x`
mostly reproduces a ranking that costs nothing to obtain.**

`scripts/actaware_diag_static.py` measures the free baseline: rank channels by their
**mean** oracle score on held-out tokens, ignoring `x` entirely. Recall@ρ0.125 of
that static prior, and each scorer's *excess* over it:

| scorer | L6 | L22 | L38 | L46 | mean | mean excess |
| --- | --- | --- | --- | --- | --- | --- |
| `static_prior` (never reads `x`) | 0.322 | 0.324 | 0.400 | 0.406 | **0.363** | +0.000 |
| `actbasis_r32` | 0.288 | 0.300 | 0.398 | 0.428 | 0.353 | **−0.010** |
| `awsvd_r32` | 0.330 | 0.364 | 0.471 | 0.512 | 0.419 | +0.056 |
| `actbasis_r128` | 0.336 | 0.370 | 0.470 | 0.495 | 0.418 | +0.054 |
| `awsvd_r128` | 0.446 | 0.500 | 0.588 | 0.605 | 0.535 | +0.172 |
| `sparse_probe` q3/k25 | 0.587 | 0.593 | 0.657 | 0.678 | 0.629 | **+0.266** |

**A rank-32 shared activation-aware basis is worse than not reading the token at
all** (−0.010 mean, and −0.035/−0.025 at layers 6/22). Recall in this regime is
mostly the static channel profile; the low-rank scorer buys almost none of the
per-token signal. Confirmed directly by Spearman computed after removing each
channel's per-token mean — agreement on the *variation* alone:

| scorer | L6 | L22 | L38 | L46 |
| --- | --- | --- | --- | --- |
| `static_prior` (by construction ~0 signal, nonzero via `g_e`) | 0.133 | 0.170 | 0.230 | 0.340 |
| `actbasis_r32` | 0.159 | 0.183 | 0.273 | 0.413 |
| `awsvd_r128` | 0.365 | 0.431 | 0.508 | 0.598 |
| `sparse_probe` q3/k25 | **0.560** | **0.570** | **0.622** | **0.693** |

The mechanism is visible in the spectrum, and it is literal. The metric that governs
how much a perturbation of `x` moves channel *scores* is `C = Σ^{1/2} M Σ^{1/2}`
with `M = Σ_e W_eᵀW_e`, because `E_j <w_j, dx>² = dxᵀ M dx / I`. Its **effective rank
`tr(C)/λ₁` is 1.8–3.5** across the four layers — i.e. of the 2048 available
directions, the score-moving energy is spread as if over ~2–3.5 of them. And the top
direction is not merely "concentrated", it is **the mean of `x` itself**:

| layer | eff. rank `tr/λ₁` | `λ₁/tr` | score-energy from `E[x]` alone | `cos²(E[x], top-1 eigvec)` | `‖E[x]‖²/E‖x‖²` |
| --- | --- | --- | --- | --- | --- |
| 6 | 3.11 | 0.322 | 0.314 | 0.998 | 0.122 |
| 22 | 3.50 | 0.286 | 0.284 | 0.999 | 0.122 |
| 38 | 2.43 | 0.412 | 0.411 | 1.000 | 0.174 |
| 46 | 1.79 | 0.558 | 0.549 | 0.999 | 0.191 |

Two things to read off. The top eigenvector of `C` **is** the mean direction
(`cos² ≥ 0.998`), and the score-energy attributable to the mean alone (0.284–0.549)
accounts for essentially all of `λ₁` (0.286–0.558) — they agree to within 0.01. So a
rank-1 data-optimal sketch is, to three digits, "replace every token by the average
token". That single direction carries **29–56% of all score-moving energy** while the
mean carries only **12–19% of the raw input energy** — activation-weighting
*concentrates* the budget onto the common mode rather than away from it, which is the
precise sense in which the optimal objective works against the task here. A rank-r
sketch therefore spends its largest and best-conditioned component reconstructing a
per-token-constant score profile, which is exactly the static prior. This closes the
loop with two prior results in this series: `probe_prefilter_diag.py` (a static
per-channel prefilter loses 12–18% of top-B mass) and
`expert-redundancy-is-not-expert-level` (all exploitable slack is per-token). **The
top-B set is a per-token object, and low rank is precisely the wrong prior for it —
low rank is an averaging operator and the signal is the deviation from the average.**

This also explains the rank saturation in `btt_dynamic.md` (stuck at ~0.47) that the
earlier study observed but could not diagnose, and it is why the learned probe there
saturated at the same place: the ceiling is the functional form, not the fit.

## Two questions from the brief, answered

**1. Output-side SVD is not a separate mechanism — it is the same operator.** Since
`Σ_y = W Σ_x Wᵀ`, the top-r eigenvectors of `YYᵀ` are the left singular vectors `U_r`
of `W Σ_x^{1/2}`, so

    U_r U_rᵀ W = U_r U_rᵀ (W Σ^{1/2}) Σ^{-1/2} = U_r S_r V_rᵀ Σ^{-1/2}

which *is* the activation-aware input SVD. Verified numerically in fp64 with an exact
SVD: `‖awsvd − outside_pca‖/‖·‖` = **3.8e-13 / 9.5e-13 / 4.2e-13 / 4.9e-13** at
layers 46/6/22/38. So "first SVD on `yyᵀ`, derive the equivalent" and
"activation-aware SVD on the weight" cannot differ, and only one of them needed
running.

The genuinely distinct output-side lever is *whitening* — equalize **relative** error
across the `I` channels instead of absolute, which is the objective a *ranking* wants.
Tested as `outwhiten`: layer-averaged recall **0.535 vs 0.535** for `awsvd_r128`,
0.497 vs 0.496 at r=88, 0.420 vs 0.419 at r=32. A dead heat at every rank. The
per-channel output norms are too uniform for reweighting to re-rank anything — the
same wash the `‖W_down‖` and norm-weighted-coordinate ablations found.

**2. Sparsity (top-5% of input positions) is the *cheap* axis but not competitive
alone.** `insp_r*` reads the top-|x| coordinates at cost-matched rank (`keep = r/H`),
which is the apples-to-apples comparison with a rank-r sketch. At equal cost it
*beats* the data-optimal shared basis — layer-averaged `insp_r128` 0.448 at cB 0.125
vs `actbasis_r128` 0.418 at cB 0.146 (L46 alone: 0.531 vs 0.495) — despite retaining
far less raw input energy (0.55 vs 0.82 of score-energy at r=128, L46). **An adaptive
choice of an arbitrary
basis beats a fixed choice of the optimal basis**, which is the same lesson as the
static-floor table from a different angle. This is why `sparse_probe` composes
sparsity with *precision* rather than with rank.

## What else was tried, all dominated

| mechanism | best result | why it fails |
| --- | --- | --- |
| `actbasis` — **shared** input factor, provably optimal for one (derivation below). MoE-native: `Px` is computed once per token and reused by all K=8 experts and both matrices, so cB = `r/(K·I) + n·r/H` — **3.1× the rank per byte** of a per-expert factor | 0.353 @ cB 0.037; 0.418 @ 0.146 | Sharing works and is cheap; what it buys is rank, and rank is not the binding constraint. Below the free static floor at r=32. |
| `pcabasis` — top-r eigenvectors of `Σ_x`, weight-agnostic | 0.422 vs `actbasis` 0.428 @ r=32 (L46) | Weight-aware `M` adds ~+0.003 score-energy over plain PCA: `M` is nearly isotropic relative to `Σ`, so the optimal basis ≈ PCA. The derivation is right and the correction is negligible. |
| `adapt` — per-token top-r of `r'=4r` shared-basis coefficients (adaptive subspace from a data-optimal dictionary) | 0.460 @ cB 0.052 (L46) vs `actbasis_r64` 0.460 @ 0.073 | The best of the low-rank variants per byte, and still ~0.17 short of the probe. Adaptivity helps (consistent with `insp`), but a 4r dictionary is still a rank object. |
| `mix` — shared rank-r basis **plus** top-q coordinates of the residual `x − PᵀPx` | 0.537 @ cB 0.135 (L46) | The two mechanisms are additive but neither is strong enough; at every budget, spending the whole budget on the sparse-quantized branch wins. |
| `qbasis` / `qawsvd` — **quantize the factors**, so rank is 5.3× cheaper (3-bit rank-448 costs what fp16 rank-88 does) | `qawsvd_b4_r768` 0.750 @ cB 0.709 (L46) | The decisive control: if low rank were merely *underfunded*, this is where it would win. Instead, at cB 0.516 a 4-bit probe on dense `x` gets **0.917**. Same bits, no rank truncation, better by 0.17 — **spending bytes on rank is strictly worse than spending them on precision.** |

Composing the two axes fails in the informative direction: at cB≈0.10, `qbasis_b3_r448`
gets 0.457 while the plain 3-bit probe gets 0.678 (L46). Rank never pays.

## Verdict on the goal

**Not met by this family, and the negative result is now structural rather than
empirical.** `sparse_probe` (3-bit RTN, top-25% input) remains the frontier at
cB 0.098 / −84.2% / 74.56 HellaSwag; nothing here comes within 5pt of it at that
budget. Combined with the ten mechanisms in `sparse_probe.md`, the search space of
"cheap approximations of `W`" is now well mapped and the conclusion is uniform:

> Expert weight rows carry near-maximal information per weight, and the top-B set is
> a per-token object. Any operator that *averages* — low rank, shared bases, PQ
> codebooks, static priors — reproduces the part of the ranking that is free and
> misses the part that matters. The only axes that buy per-token signal are
> **precision** (bits per weight) and **adaptive support** (which coordinates), which
> is exactly what the probe already spends its budget on.

What this does add to the paper: a *principled* negative result. The low-rank family
is not dead because of a bad choice of objective (we used the provably optimal one,
in both the input and the whitened-output metric, and the data-aware version beats
the weight-only one as theory predicts) — it is dead because low rank is the wrong
inductive bias for per-token selection, and the free static baseline proves it.

## Method and code

`scripts/actaware_scorer_screen.py` — all sketch families on one cost axis, reporting
recall/mass (comparable to `btt_dynamic.md`, `probe_frontier.py`) and block-output
`rel_err` (the currency that predicts accuracy). `Σ` and `M` are fit on a **held-out**
4096-token slice and scored on the next 1024, so no statistic sees its own tokens.
Per-expert factors use an **exact** batched SVD, not `svd_lowrank`: at r=32 the
randomized form (`q=r+16, niter=4`, what `lowrank_scorer.py` uses) is **13.8%** off
the exact factors of `W Σ^{1/2}`, because activation weighting concentrates the
spectrum hard — an error comparable to the truncation under study.

`scripts/actaware_diag_static.py` — the static floor and the centered-Spearman
decomposition. This is the instrument worth keeping: it answers "is this scorer
buying per-token signal, or re-deriving the free prior?" in ~4 minutes, and it should
gate any future scorer proposal before an eval is spent on it.

Derivation of the optimal shared factor. Requiring one `P` for all experts,

    min_P Σ_e E_x ‖(W_e − A_e P)x‖²  =  min_G tr((I−Π_G) Σ^{1/2} M Σ^{1/2} (I−Π_G))

with `G = P Σ^{1/2}`, `M = Σ_e W_eᵀW_e`; the optimum is the top-r eigenvectors `Q` of
`Σ^{1/2} M Σ^{1/2}`, giving `P = Qᵀ Σ^{-1/2}` and `A_e = W_e Σ^{1/2} Q`. `M = I`
recovers PCA; `Σ = I` recovers the cross-expert weight basis measured dead in
`idea_pilot_scorers.py`. `Σ^{±1/2}` uses a relative eigenvalue floor (`--eps 1e-6`),
the standard SVD-LLM conditioning guard.

Results: `docs/results/actaware/{screen_L46,screen_L6_22_38,screen_qbasis_L46,static_diag_L46,static_diag_L6_22_38}.json`.

```bash
# main screen — 4 layers, all families (1 GPU each, ~10 min/layer)
python scripts/actaware_scorer_screen.py --layers 46 --fit-tokens 4096 \
    --score-tokens 1024 --ranks 8,16,32,64,88,128,256 \
    --variants ref,pcabasis,actbasis,awsvd,outwhiten,svd,insp,adapt,mix \
    --gate-modes upgate,uponly --out docs/results/actaware/screen_L46.json
python scripts/actaware_scorer_screen.py --layers 6,22,38 --ranks 32,88,128 \
    --variants ref,actbasis,awsvd,outwhiten,svd,insp --gate-modes upgate \
    --out docs/results/actaware/screen_L6_22_38.json

# the decisive control: quantized factors (rank at 5.3x off)
python scripts/actaware_scorer_screen.py --layers 46 --variants ref,qbasis,qawsvd \
    --ranks 32 --qbits 3,4 --qranks 128,256,448,768 --gate-modes upgate \
    --verify-equiv 0 --out docs/results/actaware/screen_qbasis_L46.json

# the static floor — run this before proposing any new scorer
python scripts/actaware_diag_static.py --layers 6,22,38,46 --ranks 32,128 \
    --out docs/results/actaware/static_diag.json
```

## Caveats

* Recall/mass/rel_err only; **no downstream eval was spent** here, deliberately —
  every variant is dominated by an already-*measured* point (`sparse_probe` q3/k25 at
  74.56), so an eval could only confirm a loss. The rel_err ladder predicted the two
  measured probe points to within 0.1pt, which is the basis for that call.
* 4 of 48 MoE layers, 1024 scored tokens per layer. Per-layer recall varies ~0.1, so
  layer-averaged numbers are good to ~0.02 — well inside the gaps being called.
* `static_prior` is fit per (expert, channel) on 4096 held-out tokens and still
  carries the router weight `g_e`, so its centered-Spearman is nonzero (`g_e` is
  genuinely per-token). It is the right baseline for *channel* selection, which is
  what these scorers claim to do.
* The shared-basis cost model assumes `Px` is reused across the K co-activated
  experts within a token — true in batch-1 decode, which is the regime this series
  targets, and the same accounting convention used throughout.
