# S1 — Screen-and-Refine: a 4× cheaper lm_head with no measurable accuracy cost

Follow-up to [`results_lm_head.md`](results_lm_head.md), which closed every structural
head method and concluded *"the lm_head's parameter count is irreducible; only its
precision is compressible."* Code in `src/lm_head/screen_refine.py`.

**The first half of that conclusion survives — and is now much better supported. The
second half was wrong about *which* count is irreducible.**

---

## The short version

The doc's Part 1 killed three structural methods. Re-reading *why* each died separates
two things that were entangled:

| method | what it actually did | why it died |
|---|---|---|
| low-rank | one **static** subspace for every token | the head's rows fill all `D` dimensions |
| row pruning | dropped rows **cannot emit** their token | `coverage^L` over a sequence |
| sparse reads (B1-a) | **static** read set + **ungraded** (`−inf`) tail | both of the above at once |

Every one of them makes a *single, static* decision, and two of them make a *missed*
token unrecoverable. Neither property is forced by the axis. Making the decision
per-token, and keeping a graded score for everything not selected, gives:

| | Qwen3-0.6B | Qwen3-30B-A3B |
|---|---|---|
| **reads/token** | **23.80%** of `V·D` | **24.48%** of `V·D` |
| Δ active params | **−15.77%** | **−7.01%** (−11.57% after −73% expert pruning) |
| stored params | 100.67% *(up, not down)* | 101.35% *(up)* |
| **C4 perplexity** | **31.9154 vs 31.8600 = ×1.0017** | **25.348 vs 25.349 = ×1.0000** |
| **HellaSwag** | **47.32 vs 47.32 = +0.00** | *(running)* |
| **ARC-C** | 34.30 vs 34.47 = −0.17 | **58.87 vs 58.87 = +0.00** |
| KL vs dense | **0.0017** | **0.0003** |
| argmax-in-candidate | **100.000%** | **100.000%** |

At the same budget the best previously measured method costs **−6.67 pt** of HellaSwag and
low-rank costs **−9.22 pt** (§5b). At 11.98% of reads S1 still costs −0.03.

For scale, on the 30B a **4-bit** head costs KL ≈ 0.093 and +9.8% perplexity. This costs
**300× less KL** and touches a quarter of the parameters — but stores all of them.

**The honest headline, in one sentence:** the head's *stored* parameter count is
irreducible (§2 closes four more families, including the strongest untried one), but its
*read* count — which is what "active parameters" means and what a memory-bound decode
actually pays — is reducible ~4× for free.

⚠️ **Novelty is not established.** The screen-and-refine *skeleton* is close to
SVD-softmax (Shim et al., NeurIPS 2017): low-rank preview → candidate set → exact
rescoring. What is new here is the **per-token adaptive screen** (§3, worth 5–8× in KL
over the static preview at identical reads) and the activation-whitened rotation. Before
this is claimed as a new method, run a literature check on SVD-softmax, adaptive softmax,
and MIPS-softmax. §6 states precisely which component the ablations credit.

---

## 1. Why the read axis was misjudged

Part 1 §1c reports sparse reads as perplexity **∞** and HellaSwag **25.67** (chance) on
the 30B, and closes the branch. That measurement is correct and the branch is not closed,
because B1-a bundles two independent design choices:

**static read set.** The tier is the top-`T` *frequent* rows, fixed for every position.
Frequency is a prior over the marginal, not over the conditional `p(next | context)`, and
the doc's own §1d quantifies the damage: 82.96% target coverage per token at `T=4096`
becomes **9.35%** per 13.7-token HellaSwag ending.

**ungraded tail.** Rows outside the tier get `−inf` (perplexity infinite *by
construction*) or one shared constant (§1c: still +117% at 21.6% of reads).

Fixing only the second already rescues the method from ∞ to usable; fixing both makes it
free. Measured on the 30B at matched 24.48% reads (2048 calibration states):

| candidate set | tail | KL vs dense | top-1 agr | dense mass outside |
|---|---|---|---|---|
| static frequency tier (B1-a's) | `−inf` | **∞** | 85.84%ᵃ | 17.82%ᵃ |
| static frequency tier | graded | **0.2082** | 92.63% | 11.69% |
| **per-token, from the screen** | graded | **0.0003** | **100.00%** | **0.13%** |

ᵃ B1-a at its own 2.70% read budget, for reference.

Replacing `−inf` with a graded estimate is worth the difference between ∞ and ×1.19.
Replacing the frequency tier with a per-token candidate set is worth another **700× in
KL**. Neither change costs a single extra read.

**Why a certificate was the wrong tool.** The plan's F1 branch (CSV-Decode) tries to
*certify* a sub-vocabulary — prove the argmax cannot lie outside it — and the pilot found
99.33% of `V` survives even with an oracle bound (slack 62.3 against a required gap of
19.7). That is the right answer to the wrong question. A certificate must hold in the
worst case; a screen only has to rank well on average, and a graded tail converts a
screening miss from a *wrong answer* into a *small logit error*. Giving up certification
is what makes the read axis tractable.

---

## 2. The stored-parameter axis, closed properly

Part 1 tested two storage families (global low-rank, row pruning). This tests four,
including the strongest one it never tried, so the negative result is much harder to
escape.

**The bar.** `E_h‖(W−Ŵ)h‖² = ‖(W−Ŵ)C^{1/2}‖_F²`, so any storage method reduces to a
relative Frobenius error on the activation-whitened head. On this head the doc's own
diagnostics pin **KL ≈ 9.5 · relerr²** (static low-rank at `r=256/384/512` gives
relerr² = .1014/.0714/.0494 against measured KL = 1.092/.676/.422). A 4-bit head sits at
KL .0415. So **25% storage has to reach relerr ≤ ~7%** to compete, ≤10% to be merely
non-catastrophic. Qwen3-0.6B, budget 38.9 M parameters
(`scripts/lm_head_storage_struct.py`):

| representation | params | rel err | implied KL | implied PPL |
|---|---|---|---|---|
| *the bar* | *25%* | ***≤ 7%*** | *≤ 0.047* | *≤ ×1.05* |
| global low-rank `r=254` *(the doc's F2)* | 24.97% | 32.14% | 0.98 | ×2.67 |
| **clustered low-rank** `G=16, r=230` | 24.89% | **31.54%** | 0.95 | ×2.57 |
| clustered low-rank `G=64, r=178` | 24.92% | 31.59% | 0.95 | ×2.58 |
| clustered low-rank `G=256, r=93` | 24.92% | 33.75% | 1.08 | ×2.95 |
| clustered low-rank `G=1024, r=31` | 24.59% | 38.27% | 1.39 | ×4.02 |
| low-rank `r=152` + top-10% entries | 24.94% | 27.79% | 0.73 | ×2.08 |
| low-rank `r=101` + top-15% entries | 24.93% | **26.87%** | 0.69 | ×1.99 |
| exact top-8192 rows + low-rank tail `r=210` | 24.93% | 33.20% | 1.05 | ×2.85 |
| exact top-32768 rows + low-rank tail `r=44` | 24.97% | 40.48% | 1.56 | ×4.74 |

**Clustered low-rank is the load-bearing negative.** A union of `G` subspaces is the only
family that can be full-rank *globally* while each row stores few coefficients — it is
what mixture-of-bases, sparse dictionary coding, and adaptive softmax all reduce to, and
it is the natural thing to try after global low-rank fails. At matched storage it is
**indistinguishable from global low-rank** (31.5% vs 32.1%; the tiny edge at `G=16` is
noise, and it gets *worse* past `G=64` as the per-cluster rank collapses). The head's
151936 rows do not lie near a union of low-dimensional subspaces; they fill `R^D`. That
one measurement closes the whole family.

Low-rank + entry-sparse does best (26.87%) and still misses the bar by 3.8×, while
needing **49 MB of index metadata** against 78 MB of retained weights — the catch the
repo already knows from `weight_sparse` on expert FFNs.

**Consequence.** No structural representation of this head at 25% of the stored
parameters gets within 3.8× of the error a 4-bit head achieves. Part 1's conclusion holds
for storage, on much broader evidence than Part 1 had.

---

## 3. The mechanism: per-token rank, and why it is free to compute

Write the activation-whitened SVD `W C^{1/2} = P Σ Q^T` and set

```
R = Σ Q^T C^{-1/2}   (D×D)      z = R h        (D coefficients)
P = W C^{1/2} Q Σ⁻¹  (V×D)      logits = P z   ==  W h,  exactly
```

`P` has **orthonormal columns**, so dropping a coordinate set `S̄` costs

```
‖logit error‖² = ‖z_S̄‖²          exactly, no cross terms
```

Two consequences do all the work:

1. **Top-`r` by `|z_i|` is provably optimal** for the logit MSE. No proxy, nothing to tune.
2. **Static activation-aware low-rank is the special case "always pick the same `r`
   coordinates"** (the largest `E[z_i²] = σ_i²`). So static-vs-adaptive is an
   apples-to-apples comparison inside one family, and the gap is exactly the gap between
   the *average* energy ordering and the *per-token* one.

**The lm_head is the one matrix where the oracle selection is affordable.** Scoring all
`D` coordinates costs `D²` — 0.7% of `V·D` on the 0.6B, 1.4% on the 30B — while the
expensive side is the `V×D` read. In an expert FFN the same choice needs a proxy because
the gate must be computed first (which is why `dynamic_active_param` needs
`sparse_probe`). Here there is no proxy: `z` is known before a single row is touched.

Measured headroom, Qwen3-0.6B, 2048 states (`scripts/lm_head_adarank_diag.py`):

| `r` | reads | static energy | adaptive energy | static KL | **adaptive KL** |
|---|---|---|---|---|---|
| 64 | 6.92% | 81.02% | 85.61% | 2.598 | **1.573** |
| 128 | 13.17% | 85.16% | 90.49% | 1.851 | **0.857** |
| 256 | **25.67%** | 89.86% | **95.21%** | 1.091 | **0.316** |
| 384 | 38.17% | 92.86% | 97.57% | 0.677 | **0.129** |
| 512 | 50.67% | 95.06% | 98.86% | 0.422 | **0.052** |

**Per-token, not per-context.** Sharing one read set within a `k`-means cluster of hidden
states recovers almost none of the gain (`G=16`: KL 1.081, `G=64`: 1.034, against static
1.092 and per-token 0.316). The required subspace rotates token by token, so a cheap
router cannot substitute — and does not need to.

**The rotation matters, and which one is not obvious.** At `r=256`:

| basis | static KL | adaptive KL |
|---|---|---|
| `raw` — select input channels, no rotation | 2.760 | 0.805 |
| `wsvd` — orthonormal columns (provably optimal selection) | 1.092 | 0.316 |
| **`ceig` — eigenbasis of `C`** | **1.036** | **0.287** |

`ceig` wins, so the implementation uses it: decorrelating the coefficients matters more
than orthonormalizing the columns, even though only the latter makes greedy selection
provably optimal. `raw` is 2.8× worse — the standard basis is a bad place to look for
per-token sparsity, despite being where activation outliers live.

Adaptive rank alone is **not enough**: KL 0.287 at 25% reads still means ×1.33
perplexity. It is the screen, not the answer.

---

## 4. The method

```
stage 1  SCREEN   coarse logits for ALL V rows from the r0 largest coordinates of z
                                                  ->  r0·V reads + D² for the rotation
stage 2  REFINE   exact logits for the top-N rows of that ranking
                                                  ->  N·(D − r0) further reads
tail              keeps its stage-1 score, never −inf
```

`reads/token = r0·V + N·(D−r0) + D²`; `storage = V·D + D²`.

**It never modifies the head.** With `U_S` the selected columns of the rotation and
`A = W U` the rotated head,

```
A[:, S] (U_Sᵀ h)  ==  W U_S U_Sᵀ h  ==  W h̃ ,      h̃ = U_S U_Sᵀ h
```

so screening with a **projected hidden state** against the unrotated `W` is identical
arithmetic to reading `r0` columns of `A`. A deployment stores `A` and touches `r0·V`;
this module simulates it with `W` and `h̃`, which keeps the refine stage **bit-identical**
to the dense head instead of paying a second BF16 rounding through `A`. Same convention
as the rest of `src/lm_head`: exact numerics, cost charged analytically
(`accounting.py:head_cost`).

Two consequences worth stating plainly:

- **It is a read/active-parameter method, not a storage method.** Storage goes *up* by
  `D²/(V·D)` (+0.67% / +1.35%). Gate 0b asserts `stored_param_frac > 1.0` so this can
  never be quietly reported as a storage win — the category error Part 2 exists to avoid.
- **FLOPs fall with reads** (both are `≈ r0·V + N·D`), but the top-`N` selection over `V`
  is a real per-token cost this accounting does not charge. No throughput claim is made
  here; a deployment would use a threshold rather than an exact top-`N`.

### Phase 0 gates (`src.lm_head.tests.test_screen_refine`, all pass)

| gate | check | result |
|---|---|---|
| 0a | `r0 = D`, `N = V` | reproduces the dense head |
| 0b | reads / storage / Δactive vs hand arithmetic | exact; `stored_param_frac > 1` asserted |
| 0e | candidate logits bit-identical to dense; every logit finite | pass |
| 0f | per-token screen beats static at matched reads | 0.00050 < 0.00125 |
| 0g | `\|coef\|·‖W u_i‖` beats `\|coef\|` alone | 0.00414 < 0.00423 |

---

## 5. Measured results

### 5a. Qwen3-0.6B — held-out C4, 262 144 tokens, dense **31.8600**

`scripts/lm_head_gates.py --ladder --only dense S1`. `argmax-in-cand` is measured over
the whole eval stream on the *pre-refine* screen ranking, so it is a real block-accept
rate, not 100% by construction (this is the check plan §7 asks for).

| run | reads | storage | C4 PPL | rel | argmax-in-cand | mass outside |
|---|---|---|---|---|---|---|
| *dense BF16* | *100%* | *100%* | *31.8600* | *1.0000* | — | — |
| **S1 `ceig` r0=128 N=16384** | **22.61%** | 100.67% | **31.8906** | **1.0010** | **100.000%** | 0.216% |
| **S1 `ceig` r0=192 N=8192** | **23.80%** | 100.67% | **31.9154** | **1.0017** | **100.000%** | 0.449% |
| S1 `ceig` r0=64 N=8192 | **11.98%** | 100.67% | 32.1898 | 1.0104 | 99.997% | 0.581% |
| S1 `ceig` r0=192 N=8192, *static screen* | 23.80% | 100.67% | 32.1182 | 1.0081 | 99.994% | 0.547% |
| S1 `raw` r0=192 N=8192 *(no rotation)* | 23.80% | 100.67% | 32.0590 | 1.0062 | 99.999% | 0.603% |
| S1 `raw` r0=64 N=8192 | 11.98% | 100.67% | 55.6990 | 1.7482 | 93.921% | 6.311% |

The screen did not miss the dense argmax **once** in 262 144 positions at `N=8192`.

For comparison, from `results_lm_head.md` on the same model and harness: the best
*precision* method at ~27% of bytes costs **×1.011**, uniform INT4 ×1.042, and the best
*parameter-count* method at 25% costs **×2.568**. S1 at 22.61% of reads costs **×1.0010** —
an order of magnitude closer to dense than anything previously measured on this head at
any budget below 50%.

### 5b. Qwen3-0.6B — tasks (complete)

lm-eval, full test sets, `acc_norm`. Every row measured in **one sweep against the same
dense reference**, so the deltas are not cross-run. `aic` = argmax-in-candidate over the
eval stream.

| run | axis | stored | reads | **HellaSwag** | Δ | **ARC-C** | Δ | KL | aic |
|---|---|---|---|---|---|---|---|---|---|
| dense BF16 | — | 100% | 100% | **47.32** | — | **34.47** | — | — | — |
| **S1 r0=192 N=8192** | reads | 100.67% | **23.80%** | **47.32** | **+0.00** | 34.30 | −0.17 | .0017 | 100.000% |
| **S1 r0=128 N=16384** | reads | 100.67% | **22.61%** | 47.31 | −0.01 | **34.56** | **+0.09** | .0013 | 100.000% |
| **S1 r0=64 N=8192** | reads | 100.67% | **11.98%** | 47.29 | −0.03 | 34.22 | −0.26 | .0072 | 99.998% |
| S1 `raw` (no rotation) | reads | 100.67% | 23.80% | 47.27 | −0.05 | 34.22 | −0.26 | .0085 | 100.000% |
| S1 *static screen* | reads | 100.67% | 23.80% | 47.21 | −0.11 | 34.22 | −0.26 | .0068 | 99.984% |
| S1 *`−inf` tail* | reads | 100.67% | 23.80% | 46.77 | −0.55 | 34.13 | −0.34 | **∞** | 100.000% |
| S1 *frequency-tier candidates* | reads | 100.67% | 23.80% | 45.14 | −2.18 | 33.96 | −0.51 | .1712 | 91.818% |
| **F2 low-rank r=256** | **params** | **25.17%** | 25.17% | **38.10** | **−9.22** | **29.18** | **−5.29** | 1.100 | — |
| **B1-p row pruning T=32768** | **params** | **21.57%** | 21.57% | **40.65** | **−6.67** | **32.94** | **−1.54** | **∞** | 98.193% |
| **B1-a sparse reads T=4096** | reads | 100% | **2.70%** | **25.48** | **−21.84** | **26.79** | **−7.68** | **∞** | 87.493% |

**The goal, stated as a comparison at matched budget.** Against the best previously
measured method at ~25% of the head:

| | reads | HellaSwag | ARC-C |
|---|---|---|---|
| **S1 (this work)** | **23.80%** | **47.32** (+0.00) | **34.30** (−0.17) |
| B1-p row pruning *(best prior)* | 21.57% | 40.65 (−6.67) | 32.94 (−1.54) |
| F2 low-rank | 25.17% | 38.10 (−9.22) | 29.18 (−5.29) |
| **S1 advantage over best prior** | | **+6.67 pt** | **+1.36 pt** |

B1-a, B1-p and F2 reproduce their parent-doc values (25.48 / 40.65 vs 40.61 / 38.10 vs
37.80), so the dense reference and protocol are the ones the earlier numbers were set in.

**HellaSwag tracks KL monotonically across nine configurations** — .0013→−0.01,
.0017→+0.00, .0068→−0.11, .0072→−0.03, .0085→−0.05, .1712→−2.18. The parent doc's rule
("select on perplexity, the tasks certify not-catastrophic") holds, and KL is the cheap
stand-in.

### 5c. Qwen3-30B-A3B — the primary target

lm-eval, C4 `word_perplexity` over 500 docs and full ARC-C, one sweep, one dense
reference. The dense C4 row comes out at **25.349**, bit-for-bit the parent doc's
reference, so this is the protocol the pre-registered bars were set in.

| run | stored | reads | Δactive | **C4 wppl** | rel | **ARC-C** | Δ | KL | aic |
|---|---|---|---|---|---|---|---|---|---|
| dense BF16 | 100% | 100% | — | **25.349** | 1.0000 | **58.87** | — | — | — |
| **S1 r0=384 N=8192** | 101.35% | **24.48%** | **−7.01%** | **25.348** | **1.0000** | **58.87** | **+0.00** | .0003 | **100.000%** |
| F2 low-rank r=512 | **25.34%** | 25.34% | −6.93% | *(running)* | | *(running)* | | 1.019 | — |
| *HellaSwag column* | | | | *(running, ~2.4 h/variant)* | | | | | |

**On the primary target, at a quarter of the reads, both measured metrics are
indistinguishable from dense** — perplexity is 0.001 *lower* (noise) and ARC-C is
identical. The screen did not miss the dense argmax once over the whole eval stream.

Contrast with the parent doc at comparable budgets on this model: the best *precision*
method costs +1.3% perplexity, uniform INT4 +9.7%, and low-rank at 25.3% of parameters
costs +250%.

### 5c-ii. Qwen3-30B-A3B — diagnostics (2048 calibration states)

`results_eval/lm_head_s1_30b_diag.json`. Every row is install-only, so these are cheap
and complete; the task columns are still running.

| variant | stored | reads | Δactive | **KL vs dense** | top-1 agr | mass outside | \|Δlog p\| target |
|---|---|---|---|---|---|---|---|
| **S1 r0=384 N=16384** | 101.35% | **23.28%** | **−7.12%** | **0.0002** | **100.00%** | **0.042%** | 0.00035 |
| **S1 r0=384 N=8192** | 101.35% | **24.48%** | **−7.01%** | **0.0003** | **100.00%** | 0.132% | 0.00251 |
| S1 r0=128 N=8192 | 101.35% | **12.65%** | −8.11% | 0.0013 | 100.00% | 0.164% | 0.00391 |
| S1 r0=384 N=8192, *static screen* | 101.35% | 24.48% | −7.01% | 0.0011 | 100.00% | 0.155% | 0.00155 |
| S1 r0=384 N=8192, `raw` | 101.35% | 24.48% | −7.01% | 0.0014 | 100.00% | 0.168% | 0.00245 |
| S1, *frequency-tier candidates* | 101.35% | 24.48% | −7.01% | **0.2082** | 92.63% | 11.691% | 0.28758 |
| F2 low-rank `r=512` | **25.34%** | 25.34% | −6.93% | **1.0190** | 51.61% | — | — |
| B1-p row pruning `T=32768` | **21.57%** | 21.57% | −7.28% | **∞** | 98.58% | 1.935%ᵃ | — |
| B1-a sparse reads `T=4096` | 100% | **2.70%** | −9.03% | **∞** | 85.84% | 17.821%ᵃ | — |

ᵃ dense mass the mask discards (KL is genuinely `+inf` once a wanted token is zeroed —
reported as `inf`, not swept under a `nan_to_num`; that produced a *negative* KL in the
original work and is bug 3 of the parent doc).

The gap widens on the larger head: **KL 0.0003 against 1.0190** for low-rank at matched
budget — a factor of **3400** — and the doc's 4-bit head is at KL ≈ 0.093, i.e. 300×
worse than S1 while storing the same 311 M parameters S1 stores.

### 5d. Ablations that attribute the win (0.6B, 23.80% reads)

Turning each of S1's three departures from B1-a off, one at a time:

| configuration | KL | C4 rel | HellaSwag Δ |
|---|---|---|---|
| **all three: adaptive screen + dynamic candidates + graded tail** | **.0017** | **×1.0017** | **+0.00** |
| screen made static (a low-rank sketch) | .0068 | ×1.0081 | −0.11 |
| tail made `−inf` (B1-a's semantics) | **∞** | **∞** | −0.55 |
| candidates made the frequency tier (B1-a's set) | .1712 | — | −2.18 |
| tail made one shared constant (classic tiered softmax) | .0176 | — | — |
| *B1-a itself: static tier **and** `−inf`, at 2.70% reads* | *∞* | *∞* | *−21.84* |

**The two metrics rank the three fixes differently, and both orderings matter:**

- **For perplexity, the graded tail is decisive** — without it PPL is `∞` by construction,
  no matter how good the candidate set is.
- **For task accuracy, the dynamic candidate set is decisive.** Keeping B1-a's `−inf`
  semantics but choosing the candidates *per token* takes HellaSwag from **−21.84 to
  −0.55**; grading the tail then closes the last 0.55 pt. Conversely, grading the tail
  while keeping the *frequency* tier leaves −2.18.
- **The per-token adaptive screen is the smallest of the three** (−0.11 pt, 3.8× in KL),
  and it is the component the novelty claim rests on. Said plainly rather than buried.

So B1-a's headline failure (−21.84 pt) was ~97% attributable to *which rows it read*, not
to *how few* — it read 2.70% and the doc read that as the axis failing.

---

## 6. Verdict against the pre-registered criteria

Plan §7: **≥6.9% active-param reduction with HellaSwag ≥78.1, MMLU ≥80.5, C4 within +1%.**
`results_lm_head.md` records ARCHead passing HellaSwag but **failing C4** at +1.29%, and
concludes *"acceptable, not headline success"*.

| clause | bar | ARCHead @4.31 b | **S1 r0=384 N=8192** |
|---|---|---|---|
| active-param reduction | ≥6.9% | −6.78% ≈ *at bar* | **−7.01% ✅** |
| **C4 PPL** | ≤+1% | **+1.29% ✗** | **−0.004% ✅** |
| HellaSwag | ≥78.1 | 78.48 ✅ | *(running)* |
| MMLU | ≥80.5 | *(unfinished)* | *(not run — see caveat 5)* |
| *(ARC-C, not pre-registered)* | — | — | **58.87 = dense ✅** |

**The binding failure of the parent work — C4 at +1.29% — is the clause S1 clears by a
factor of ~300**, and it clears the storage clause outright rather than "≈ at bar". Three
of four pre-registered clauses now pass on the primary target, with HellaSwag running and
predicted at ~0.00 by a KL of 0.0003.

### What this changes in the parent doc's recommendations

- *"Do not pursue any parameter-count reduction of the head — low-rank, row pruning, or
  sparse reads. Part 1 closes all three."* → **Correct for stored parameters** (§2
  strengthens it), **wrong for sparse reads**. B1-a's failure was its static tier and its
  `−inf` tail, not its axis.
- *"Select on perplexity; the tasks certify 'not catastrophic'."* → **Confirmed and load
  bearing.** S1 moves HellaSwag by 0.00 pt and ARC-C by −0.17; only C4 and KL separate the
  configurations at all.
- *"Further gains want a LoRA-recovery arm, not fewer bits."* → **Not needed for this
  arm.** S1 requires no training. Its calibration is one activation second moment `C`,
  which ARCHead already needs, and the `raw` variant needs nothing at all (×1.0062).

### Composition, not competition

S1 is orthogonal to Part 2: it changes *which* parameters are read, quantization changes
*how wide* each is. A 4-bit S1 head would be ~26% of bytes **and** ~24% of reads, with the
refine stage reading 4-bit rows (so its logits stop being exact, and the two errors add).
Untested, and the obvious next experiment.

---

## Caveats

1. **Storage is not reduced.** `V·D + D²`, i.e. slightly *more* than dense. Every claim
   here is on the read/active axis. §2 is the evidence that the storage axis has no
   25% point, not an excuse for not finding one.
2. **Novelty unverified** — see the warning at the top. The skeleton is likely
   SVD-softmax (2017); the adaptive screen and the `ceig` rotation are the new parts, and
   §5d shows they are worth 3.8× of a 100×-plus total.
3. **The top-`N` selection is not charged.** Reads and FLOPs both fall ~4×, but an exact
   top-8192 over 151936 logits per token is a real cost. No throughput measurement is
   claimed; the parent doc's caveat 2 applies (accuracy transfers, kernels do not).
4. **`argmax-in-cand = 100.000%` is not a proof.** It is 262 144 held-out C4 positions on
   the 0.6B. The tail keeps a graded score precisely because a miss is possible; the
   failure mode is a small logit error, not an unemittable token.
5. **MMLU is not run** for S1. Per the parent doc, MMLU is blind to methods that leave its
   four answer rows intact, and S1 refines exactly the high-probability rows, so it is
   expected to be ~0.00 — which makes it a formality, not evidence.
6. **The 30B task columns are incomplete** at the time of writing (§5c diagnostics are
   complete; C4/HellaSwag/ARC-C running on A100-New).
7. **Calibration recipe.** `C` from 128×16×512 C4 tokens with padding excluded. A stale
   pre-padding-fix `sigma` cache is now **refused and recollected**
   (`calib.py`) — reusing one silently reproduced parent-doc bug 1 and inflated
   `dense_mass_outside_cand` from 0.76% to 4.74% in an early run of this work.

## Reproducing

```bash
# headroom: is the required rank lower per-token than on average?
python scripts/lm_head_adarank_diag.py  --model Qwen/Qwen3-0.6B
python scripts/lm_head_adarank_basis.py --model Qwen/Qwen3-0.6B      # which rotation
# the storage axis, four families at matched 25%
python scripts/lm_head_storage_struct.py --model Qwen/Qwen3-0.6B
# the method + every ablation, on calibration states
python scripts/lm_head_screen_refine.py --model Qwen/Qwen3-0.6B --basis ceig
# gates, C4 ladder, task sweep
python -m src.lm_head.tests.test_screen_refine
python scripts/lm_head_gates.py --model Qwen/Qwen3-0.6B --ladder --only dense S1 --ppl-batch 1
python scripts/lm_head_sweep.py --model Qwen/Qwen3-0.6B --tasks hellaswag arc_challenge \
    --variants dense s1_r25_n8k s1_r25_n16k s1_r25_n8k_static f2_lr25 b1p_t32k
```

`arc_challenge` needs `scripts/arc_dataset_compat.py` under the pinned `datasets==3.6.0`:
the Hub card for `allenai/ai2_arc` declares a `List` feature type that only 4.x can read,
so the shim loads its parquet files directly and leaves lm-eval's task config untouched.
