# Efficient Channel Scoring — Consolidated Report

> Making the *realized* whole-FFN reduction match the *channel* reduction.

## The problem

`oracle_mag` (rank by `g_e·|SiLU(gate)⊙up|`) is the best selector, but it runs
gate+up at full width just to decide. Its whole-FFN kept fraction is
`(1+1+ρ)/3 ≥ 2/3`, so **−33.3% is its hard floor** regardless of channel budget.
Goal: score channels with something costing **≪ full gate+up**, emit indices, gather
only the kept channels of all three matrices. At ρ=0.125 the target is oracle
parity (~77.1 HellaSwag acc_norm).

---

## The method: `input_sparse`

One method, two sparsities. **`input_sparse`** scores channels from the *served*
`up`/`gate` weights read on only the top-`ρ_input` fraction of the token's input
coordinates by `|x_i|`, pools the `K·I` scores of the token's experts on one scale
via `g_e·|SiLU(g̃ate)⊙ũp|`, keeps the global top-`B`, and gathers all three
matrices to those channels. Nothing is quantized and nothing extra is stored — the
probe *is* the served weight (implemented as views onto the expert tensors, so
measured extra allocation is **0.00 MB**).

**Two sparsities**, and they are the same kind of quantity — both *keep* fractions:

| symbol | config key | what it keeps | paid to |
|---|---|---|---|
| **`ρ_input`** | `probe.rho_input` | fraction of the token's input coordinates read | **scoring** |
| **`ρ_channel`** | `probe.rho_channel` (= `1 − prune_ratio`) | fraction of the pooled `K·I` channels kept | **compute** |

An optional third knob, `input_alloc`, decides how the pooled coordinate-read budget
is split across a token's K experts.

> Registered as criterion `sparse_probe` in the code (historical name), with
> `bits: 16`. Quantized-probe variants (`bits < 16`) are **excluded from this
> report** — they cost +13% extra storage and buy nothing; see
> [Why no quantization](#why-no-quantization).

### Used-parameter accounting

**Used params = params read for scoring + params read for compute**, in units of one
expert `(I,H)` matrix (a dense expert FFN is 3):

```
scoring : n · ρ_input     (n = 2 branches, up+gate; all I rows, ρ_input of the H columns)
compute : 3 · ρ_channel   (up, gate, down gathered to the ρ_channel kept channels)
used    = (n·ρ_input + 3·ρ_channel)/3 = ρ_channel + n·ρ_input/3
```

This is deliberately **conservative**: the two passes overlap on the kept rows, and
this bills that overlap twice rather than discounting it. Consequences worth stating
— at `ρ_input→1` it lands *above* a single-pass exact scorer (`oracle_mag`'s
`(1+1+ρ_channel)/3`) by exactly `2·ρ_channel/3`, correctly, since a two-pass scheme
that reads everything really does read the kept rows twice; at `ρ_input→0` it gives
bare `ρ_channel`. It also matches `lowrank_scorer`'s `ρ + n·c/3`, so **every scorer
row below is on one scale**.

---

## Leaderboard

Qwen3-30B-A3B, HellaSwag 0-shot acc_norm (dense = 78.56) and full MMLU 5-shot
(dense = 80.91). stderr ≈0.42pt / ≈0.34pt. Masking simulation, no fine-tuning.
All `input_sparse` rows read the served weights: **zero extra storage**.

| #  | method                                   | `ρ_input` | `ρ_channel` | used-param cut | HellaSwag | MMLU 5-shot | status |
| -- | ---------------------------------------- | ------ | ------ | -------------- | --------- | ----------- | ------ |
| 1  | `oracle_mag_noW` (full gate+up, single pass) | —  | 0.125  | −29.2%        | **77.11** | **79.44** | measured |
| 2  | `input_sparse`                           | 1.00   | 0.125  | −20.8%        | **76.95** | —          | measured ¶ |
| 3  | `input_sparse`                           | 0.50   | 0.125  | −54.2%        | ~76.6     | —          | predicted (ladder) |
| 4  | `input_sparse`                           | 0.25   | 0.20   | −63.3%        | **76.72** | *running*  | measured |
| 5  | `input_sparse`                           | 0.25   | 0.15   | **−68.3%** | **76.47** | **78.63** | measured |
| 6  | `input_sparse`                           | 0.25   | 0.10   | **−73.3%** | **74.06** | **77.20** | measured (ladder 74.2) |
| 7  | **`input_sparse` + `router` alloc** | 0.25   | 0.10   | **−73.3%** | **74.64** | **77.67** | measured (ladder 74.9) |
| 8  | `input_sparse` + per-layer schedule (unweighted) | mixed | mixed | −73.3%   | **73.90** | **77.85** | measured (ladder 74.9 ✗) |
| 8b | `input_sparse` + per-layer schedule (slope-weighted) | mixed | mixed | −73.0% | **70.77** | —      | measured (**fails**) |
| 9  | `oracle_up` (full up, cut gate)          | —     | 0.125  | −58.3%        | **71.30** | **76.43** | measured |
| 10 | `lowrank_scorer` BTT m2n2 r32 (up-only)  | —     | 0.25   | −71.2%        | **65.97** | —          | measured |
| 11 | `lowrank_scorer` SVD r32 (up-only)       | —     | 0.25   | −73.1%        | **63.94** | —          | measured |
| 12 | Level-1 `pivchol` (offline, no scoring pass) | — | 0.125  | −87.5%        | **44.15** | **45.51** | measured |

¶ Row 2 was run with a 4-bit probe on dense `x`. At `p=1` a 4-bit probe and the
served-weight probe select **identically** on a 4-bit served model (RTN is
idempotent on its own output), and on this bf16 model the 4-bit selection is within
0.006 rel_err of exact — so its *accuracy* transfers to `input_sparse` at `p=1`,
which is why it is kept. Its cut is quoted in the `input_sparse` frame.

**The pre-registered ladder predictions were 74.2 / 74.9 / 74.9 for rows 6–8.**
Rows 6 and 7 came in at **74.06 and 74.64** — within **0.14 / 0.26pt**, well inside
one stderr, so the instrument holds for anything that changes the *selector*. Row 8
(the per-layer schedule) came in **1.0pt below** its HellaSwag prediction and lost to
row 6 there — while *gaining* +0.65pt on MMLU. The ladder does **not** transfer to a
schedule that redistributes budget *across layers*; see "The schedule failure" below.

**Key takeaways.**

- **The headline: −73.3% used parameters at 74.64 HellaSwag / 77.67 MMLU** (row 7),
  with zero extra storage. Against the only prior selectors reaching comparable
  depth: **+30.5pt over Level-1 `pivchol`** (−87.5%, 44.15) and **+3.34pt over
  `oracle_up`** while cutting **15.0 more points** of the FFN. Costs 3.92pt of
  HellaSwag and 3.24pt of MMLU against dense.
- **It beats the low-rank family at matched cost by ~9–11pt.** Rows 10–11 sit at
  −71.2% / −73.1% on the *same* accounting and score 65.97 / 63.94; row 7 is
  −73.3% at 74.64. That is the cleanest iso-cost comparison in this table, since
  `lowrank_scorer` already used the scoring-plus-compute frame.
- **`router` input allocation is confirmed: +0.58pt HellaSwag, +0.47pt MMLU**
  (74.06→74.64, 77.20→77.67) at *identical* cost and reads/token. Predicted +0.66
  from the offline screen. Free — one extra bisection over a sort the layer already
  computes.
- **The two sparsities are not interchangeable.** Row 2 (`ρ_input=1`) spends two
  thirds of its budget on *scoring* and only reaches −20.8%; row 5
  (`ρ_input=0.25, ρ_channel=0.15`) spends a sixth on scoring, reaches −68.3%, and
  gives up just **0.48pt** of HellaSwag against it. **Cutting `ρ_input` is far
  cheaper than cutting `ρ_channel`** — scoring is discounted 3× per branch pair,
  compute is not.
- **MMLU degrades more gracefully than HellaSwag**, contrary to the prior
  expectation in this doc. At −73.3% the method loses **3.24pt** on MMLU
  (80.91→77.67) versus **3.92pt** on HellaSwag, and gives up only **1.77pt** against
  the full-width `oracle_mag_noW` (79.44) while cutting 44.1 more points. The
  hypothesis that per-token activation matters *more* on MMLU is **not supported** —
  the gap narrows rather than widens.

---

## Why no quantization

An earlier round built the probe as a *separate* low-precision (3–4 bit) copy of
`up`/`gate`. That is now excluded from this report, because at matched budget it is
dominated on both axes at once:

| | separate 3-bit proxy | `input_sparse` (served weights) |
|---|---|---|
| extra storage | **+13% of expert weights** | **0** (measured 0.00 MB alloc) |
| error sources | input sparsity **+ quantization** | input sparsity **only** |
| ranking quality at matched `ρ_input` | strictly worse | better |

The measured comparison at `ρ_input=0.25`: the 3-bit probe scores 74.56 HellaSwag /
77.65 MMLU, while `input_sparse` at `ρ_input=0.25, ρ_channel=0.15` (row 5) scores
**76.47 / 78.63** —
and *pays no storage*. So quantizing the probe buys nothing: bits are the expensive
axis and input sparsity is the cheap one. The implementation keeps the `bits` knob
(`< 16` still works, and the offline screens below characterize it) but no headline
result uses it.

The probe is a **view**, not a copy: `build_layer_probe` returns the expert tensors
themselves (`data_ptr` equality, pinned by
`test_reuse_probe_aliases_served_weights_without_copying`). That is load-bearing —
a stacked fp16 copy of up+gate is ~39 GB on this model.

One special case worth noting: on a model *already served* at 4 bits, a 4-bit probe
**is** `input_sparse` (RTN is idempotent on its own output), so its selection is
identical to the served oracle's — measured excess rel_err **0.0000**. The rule is
"score at serving precision", not "score at fp16".

---

## The `ρ_input` / `ρ_channel` trade

Only the sum `ρ_channel + 2·ρ_input/3` is paid for, so the two sparsities compete
for one budget. At `ρ_input=0.25`:

| `ρ_channel` | scoring | compute | used  | used-param cut |
| ----------------- | ------- | ------- | ----- | -------------- |
| 0.100             | 0.1667  | 0.100   | 0.2667 | **−73.3%** |
| 0.125             | 0.1667  | 0.125   | 0.2917 | −70.8%        |
| 0.150             | 0.1667  | 0.150   | 0.3167 | −68.3%        |
| 0.200             | 0.1667  | 0.200   | 0.3667 | −63.3%        |

And the mirror image — `ρ_channel=0.125` fixed, varying `ρ_input`:

| `ρ_input` | scoring | compute | used  | used-param cut |
| --------- | ------- | ------- | ----- | -------------- |
| 0.125     | 0.0833  | 0.125   | 0.2083 | **−79.2%** |
| 0.1875    | 0.1250  | 0.125   | 0.2500 | −75.0%        |
| 0.25      | 0.1667  | 0.125   | 0.2917 | −70.8%        |
| 0.50      | 0.3333  | 0.125   | 0.4583 | −54.2%        |
| 1.00      | 0.6667  | 0.125   | 0.7917 | −20.8%        |

**Reading the two tables together.** The scoring term `2·ρ_input/3` carries a 3×
discount (two branches spread over a three-matrix FFN) that the compute term
`3·ρ_channel/3 = ρ_channel` does not, so a unit of `ρ_input` costs **two thirds**
of a unit of `ρ_channel`. That is why `ρ_input` is the axis to cut first — and why
`ρ_input=1` is hopeless regardless of `ρ_channel`: scoring alone (0.667) already
exceeds what the whole method is trying to get under.

**Measured budget sweep** (`input_sparse`, `ρ_input=0.25`, `uniform` allocation,
masking sim, no finetune; dense = 78.56 / 80.91):

| `ρ_channel`      | used-param cut    | HellaSwag acc_norm | Δ vs dense      | MMLU 5-shot     | Δ vs dense      |
| ----------------- | ----------------- | ------------------ | ---------------- | --------------- | ---------------- |
| 0.200             | −63.3%           | **76.72**  | −1.84           | *running*     | —               |
| 0.150             | −68.3%           | **76.47**    | −2.09           | **78.63** | −2.28           |
| 0.100             | **−73.3%** | **74.06**    | −4.50           | **77.20** | −3.71           |
| 0.100 +`router` | **−73.3%** | **74.64**    | **−3.92** | **77.67** | **−3.24** |

stderr ≈0.43pt (HellaSwag) / ≈0.33pt (MMLU). Both benchmarks degrade smoothly here
— no cliff of the kind Level-1 hits at −87.5% — and **MMLU consistently loses less
than HellaSwag** at the same cut.

---

## Which input coordinates, and how many per expert?

The pooled score is `g_e·|SiLU(g̃ate)⊙ũp|`, so a coordinate read spent on a
dominated expert moves the ranking less than the same read on the top-routed one.
That makes "same budget for every routed expert" a *choice*, not a given. Four
terms, all at an **identical** pooled budget of `K·round(p·H)` reads/token:

| term        | what it ranks                                     | per-expert budget                  |
| ----------- | ------------------------------------------------- | ---------------------------------- |
| `uniform` | `\|x_i\|`, same set for all K experts             | equal (what every earlier row did) |
| `router`  | `g_e·\|x_i\|`, pooled top-N across (slot, coord) | emergent                           |
| `router2` | `g_e²·\|x_i\|` (Level-1's `g²σ` motivation) | emergent, sharper                  |
| `colnorm` | `\|x_i\|·rms_j(W[:,i])`                          | equal                              |

The `router` terms are a *single threshold*, not a per-expert sort: because the
score factors as `g_e^β` times the shared `|x_i|`, each slot keeps a **prefix** of
the token's one shared descending-`|x|` order, so the allocation is one bisection
on `τ` plus a small top-up (`allocate_input_reads`, verified against an explicit
pooled top-N in `test_alloc_matches_bruteforce_pooled_topk`).

**Measured** — block-output `rel_err`, 4 layers × 4096 C4 tokens, `ρ_input=0.25`,
`input_sparse` (served weights).  Lower is better; Δpt uses the fixed-budget slope −26.4 (derived below):

| term                 | ρ_ch=0.10       | ρ_ch=0.125      | ρ_ch=0.15       | ρ_ch=0.20       | Δpt vs uniform          |
| -------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ------------------------ |
| `uniform`          | 0.4434           | 0.4099           | 0.3820           | 0.3368           | —                       |
| **`router`** | **0.4183** | **0.3860** | **0.3592** | **0.3169** | **+0.53 … +0.66** |
| `router2`          | 0.4272           | 0.3977           | 0.3736           | 0.3364           | +0.01 … +0.43           |
| `colnorm`          | 0.4429           | 0.4095           | 0.3814           | 0.3362           | +0.01                    |

**`router` wins in all 16 layer×budget cells.** So the answer to the design
question is: *multiply the router probability into the input magnitude and rank
across experts* — do not fix a budget per routed expert. The gain is free (same
reads, one extra bisection over a sort the layer already needs).

Two reads worth keeping:

- **`router2` overshoots.** Weighting by `g²` starves the tail experts past the
  point of return — the optimum is `β=1`, not Level-1's `β=2`. Sharper is not
  better here because the *coordinate* budget and the *channel* budget respond
  differently to `g_e`.
- **`colnorm` is a wash on output error too** (+0.01pt), confirming the earlier
  recall-based verdict from a second, accuracy-anchored metric. Column-norm CV is
  0.022 — there is nothing to weight.

---

## Is `rel_err` really a linear predictor of accuracy?

The ladder was fitted on layer-*averaged* rel_err. Before using it to trade error
between layers, it has to be checked per layer — and the two ways rel_err arises
have to be separated (`scripts/probe_relerr_linearity.py`, 8 measured
selector×budget points, no GPU):

| family                                                     | how rel_err varies      | slope                    | R²             |
| ---------------------------------------------------------- | ----------------------- | ------------------------ | --------------- |
| `budget_ladder` (`oracle_mag`, ρ=0.5→0.125)          | honestly fewer channels | −6.9 pt/unit            | 0.867           |
| **`fixed_rho0.125`** (selector varies at fixed ρ) | **mis-selection** | **−26.4 pt/unit** | **0.985** |

**Mis-selecting at a fixed budget costs 3.8× more accuracy per unit rel_err than
an honestly smaller budget.** The two are different rulers, and the fixed-budget
one is the correct ruler for comparing scorers or trading error between layers.
Within it the relationship is close to a straight line (R² = 0.985 layer-averaged;
per layer 0.60 / 0.87 / 0.99 / 0.96 for L6 / L22 / L38 / L46). Per-layer slopes
vary (−36.1 → −15.1, CV 0.29): deeper layers are *less* sensitive per unit of
their own rel_err. Figure: `figures/btt_dynamic/relerr_linearity.png`.

This validates rel_err as a per-layer objective, which is what makes the next
section a well-posed optimization rather than a heuristic.

---

## A principled budget allocation — across layers, not just within one

Every probe row so far applies one `(ρ_input, ρ_channel)` to all 48 MoE layers, because that is
what a hand-written config can say. But layers are not interchangeable:
measured over **all 48** layers (`scripts/probe_layer_surface.py`, 8192 C4 tokens,
each layer hooked to measure its own error while **returning the unperturbed
output**, so layer L's error never contaminates L+1):

> at `ρ_input=0.25, ρ_channel=0.125` the per-layer rel_err spans **0.186 → 0.543, a 2.92× spread**
> — cheapest are the edges (L47, L0, L1, L45), most expensive the middle band L17–L33.

**The program.** With rel_err validated as a per-layer objective and `used = ρ + 2p/3`
as the cost:

```
minimize  Σ_L rel_err_L(p_L, ρ_L)
s.t.      mean_L [ ρ_L + 2 p_L (1−ρ_L)/3 ] ≤ target
```

Separable, so one Lagrange multiplier is exactly optimal on the grid: each layer
independently minimizes `rel_err_L + λ·kept_L`, and `λ` is bisected until the mean
cost hits the target (`allocate()`; verified feasible, argmin-consistent and
monotone in the target).

**Solved at three targets** (grid `ρ_input ∈ {0.125…1.0}`, `ρ_channel ∈ {0.0625…0.25}`,
8192-token surface, `used = ρ_channel + 2·ρ_input/3`):

| target cut        | best uniform                  | per-layer optimum | gain               | ≈Δpt (predicted) |
| ----------------- | ----------------------------- | ----------------- | ------------------ | ------------------ |
| **−73.3%** | ρ_in=0.1875, ρ_ch=0.125 → 0.4613  | **0.4423**  | **−0.0190** | **+0.50**    |
| −70.8% | ρ_in=0.25, ρ_ch=0.125 → 0.4248    | 0.4168            | −0.0080           | +0.21              |
| −68.3% | ρ_in=0.1875, ρ_ch=0.1875 → 0.4032 | 0.3935            | −0.0097           | +0.26              |

Figure: `figures/btt_dynamic/layer_surface.png` (drawn on the original 2048-token
surface; the ranking it shows is unchanged).

**Robustness of the surface.** Re-measured at 8192 tokens (4× the original):
layer-rank correlation **0.970** at `ρ_input=0.25, ρ_channel=0.125`, mean |Δrel_err| 0.021, and
the solved gain reproduces (+0.019 vs +0.020). 29/48 layers pick an identical grid
point. So the *measurement* is sound — which makes the eval result below a statement
about the **objective**, not about noise.

### The schedule failure — and what it teaches

**Measured: 73.90 HellaSwag (−0.16pt vs the matched uniform row's 74.06) but 77.85
MMLU (+0.65pt vs 77.20)**, against a predicted +0.50 on both. The realized budget
was verified correct in-run and is iso-cost with the uniform row under the new
accounting (mean used 0.2674 vs 0.2667, both −73.3%), so this is not an accounting
slip.

The split verdict is itself informative: the schedule is not simply broken, it
**trades HellaSwag for MMLU**. It still fails as stated — the prediction was a gain
on both, and against the `router` row (74.64 / 77.67) it is −0.74 / +0.18, i.e.
dominated on HellaSwag and a tie on MMLU. But "reallocating budget across layers
shifts *which* capability you keep" is a different and more interesting claim than
"it doesn't work", and it is not something the single-number rel_err objective can
express at all.

The cause is visible in this doc's own linearity data, which I under-used when
setting up the solve. Per-layer accuracy sensitivity is **not** uniform — it falls
off almost perfectly linearly with depth:

| layer                                  | 6                 | 22      | 38      | 46                |
| -------------------------------------- | ----------------- | ------- | ------- | ----------------- |
| pt per unit of*that layer's* rel_err | **−36.07** | −29.77 | −24.99 | **−15.06** |

(+0.478 pt/unit per layer, r = 0.957 — **early layers are ~2.4× more sensitive than
late ones**.) Minimizing the *unweighted* `Σ_L rel_err_L` is therefore the wrong
program: it is free to dump error wherever error is cheapest to reduce, which is
not where error is cheapest to *pay*. And that is exactly what it did — by depth
band, the mean kept fraction it chose was:

| band                 | L0–5            | L6–15 | L16–35 | L36–47 |
| -------------------- | ---------------- | ------ | ------- | ------- |
| mean used            | **0.2309** | 0.2802 | 0.2818  | 0.2465  |
| sensitivity (fitted) | **−38.7** | −34.8 | −27.7  | −20.0  |

**The most sensitive band got the smallest budget of all four.** The solver traded
a large error increase on L0–5 for a smaller decrease in the middle, which is a win
on the unweighted objective and a loss in the model.

The fix is a one-line objective change — minimize `Σ_L w_L · rel_err_L` with `w_L`
the fitted per-layer slope (`layer_slope_weights()`, `--no-slope-weight` reproduces
the old behaviour). Re-solved, it **inverts** the allocation, giving the early layers
more budget and pushing L35–47 down to `ρ_input=0.125`. Under the accuracy-relevant
objective the ranking becomes (8192-token surface, `used = ρ + 2p/3`):

| config                            | used   | mean rel_err     | **slope-weighted** rel_err |
| --------------------------------- | ------ | ---------------- | -------------------------------- |
| schedule, slope-weighted          | 0.2663 | 0.4495           | **0.4382**                 |
| schedule, unweighted              | 0.2663 | **0.4423** | 0.4458                           |
| best uniform (ρ_in=0.1875, ρ_ch=0.125) | 0.2500 | 0.4613           | 0.4653                           |

Note the reversal: the unweighted schedule wins on unweighted error and loses on
weighted error, which is the diagnosis restated.

**And it was wrong too — measured 70.77 HellaSwag, i.e. −3.29pt versus the uniform
row (74.06) and −3.13pt versus the unweighted schedule it was meant to fix**, at a
realized used-param cut of −73.0% (i.e. very slightly *more* budget than the
baseline's −73.3%). So the fix made things **much worse**, not better.

**Cross-layer budget allocation is therefore closed as a negative result.** Three
solves — unweighted, slope-weighted, and every uniform grid point — and the *uniform*
schedule wins on HellaSwag. Two readings, both worth carrying:

1. **The objective was not the (only) problem.** Slope-weighting fixed the diagnosed
   defect and made the outcome worse, so "minimize a weighted sum of per-layer
   rel_err" is not merely mis-weighted — it is the wrong *form*. The likely reason:
   the surface measures each layer against an **unperturbed** input (deliberately, so
   the layers are separable), but a deployed schedule perturbs every layer at once.
   Errors compound across depth in a way no separable objective can see, and the
   slope-weighted solution — which pushes `ρ_input` down to 0.125 on twelve
   consecutive late layers — is exactly the kind of correlated, contiguous damage
   that assumption hides.
2. **Uniform is a strong baseline, and that is the finding.** Layers differ 2.92× in
   rel_err at equal cost, which *looks* like large headroom, and three independent
   attempts to collect it all failed. The per-token selector is where the slack is;
   the per-layer budget is not.

**The transferable lesson.** The rel_err ladder is validated for changes that alter
*which channels a layer selects* (rows 6–7 landed within 0.26pt of prediction). It
is **not** validated for changes that move budget *between* layers, because it
implicitly assumes every layer's error is worth the same, and this model's layers
differ 2.4× in that respect. Any future cross-layer allocation must weight by
per-layer sensitivity — and should be spot-checked with a real eval before being
believed.

**Expressed as thresholds instead of 48 integer pairs.** The −73.3% slope-weighted
solution induces a mean `|x|` cut of **0.6676** and a mean pooled-score cut of
**0.03016**. That
matters for implementability: rather than shipping a 48-entry table, a kernel can
carry two global magnitude thresholds and let the per-layer counts emerge — the
same "one shadow price" structure Level-1 already uses online. (The next section
tests this and finds only the `|x|` threshold survives.)

**Caveat on the surface.** Measured at 8192 C4 tokens, cross-checked against an
independent 2048-token run: layer-rank correlation 0.970, mean |Δrel_err| 0.021, and
29/48 layers pick the same grid point. So the *ranking* is solid but the specific
grid point a near-tied layer lands on is not. Read the schedule as "the middle band
needs more budget, the edges less" rather than as 48 independently meaningful
choices.

---

## Methods tested

### 1. `input_sparse` — served weights read on a sparse input

**Setting.** Read the served gate/up on the token's top-`ρ_input` coordinates by `|x|`,
compute `g_e·|SiLU(gate)⊙up|`, take the global top-`B` across the token's K experts,
gather all three matrices to those channels.

**Formulation.** `used = ρ + 2p/3` (scoring + compute; see
[Used-parameter accounting](#used-parameter-accounting)). Zero extra storage.

**Results.** The measured `ρ_input`/`ρ_channel` sweep is in the
[Leaderboard](#leaderboard) and in §"The `ρ_input` / `ρ_channel` trade".

**Why it works.** Expert weight rows carry near-maximal information per weight; no
compression of the weight itself matches just reading fewer served entries on the
coordinates where `x` has energy. Input sparsity is the cheap axis — dropping 75% of
the coordinates costs 0.48pt of HellaSwag (row 2 → row 5) while freeing 47.5pp of
budget.

**The excluded axis (bits).** Quantizing the probe was the earlier design and is
dominated; the offline characterization is kept because it is the evidence for
"precision is the expensive axis": at matched bytes, 2-bit on dense `x` reaches
recall 0.506 while 4-bit on 50% of `x` reaches 0.836. See
[Why no quantization](#why-no-quantization).

**Cite:** Prox (arXiv:2607.27591) — same mechanism on 10 dense LLMs, zero MoE.
The contribution is the MoE instantiation (cross-expert `g_e` pooling + the
negative-result map).

---

### 2. Low-rank scorer — factored proxy (SVD, BTT)

**Setting.** Factor `W_up` (and optionally `W_gate`) into `L·R` at rank r per
`m×n` block grid. Score = `|L·(R·x)|`. The proxy runs *before* any full-width
matmul so all three matrices get gathered.

**Formulation.** BTT with block grid `m×n`, rank r:

```
W[i,j] ≈ L[i,j] R[i,j],   L: (a,r),  R: (r,b)
cost c  = r·(m·H + n·I) / (I·H)
```

`m=n=1` is plain SVD. At r=32, `svd` costs 0.057, `btt_m2n2` costs 0.115.

**Results (nominal −75% channel cut).**

| scorer               | cost  | whole-FFN cut | recall | acc_norm |
| -------------------- | ----- | ------------- | ------ | -------- |
| SVD r32 up-only      | 0.057 | −73.1%       | 0.444  | 63.94    |
| BTT m2n2 r32 up-only | 0.115 | −71.2%       | 0.456  | 65.97    |
| SVD r32 up+gate      | 0.115 | −71.2%       | 0.468  | 63.80    |
| BTT m2n2 r32 up+gate | 0.229 | −67.4%       | 0.479  | 66.83    |

**Analysis.**

- **Drop the gate, buy rank instead.** At equal cost, up-only at 2× rank ties
  up+gate on recall and *wins* on accuracy by 2–9pt.
- **BTT wins accuracy while losing recall** — recall ≠ accuracy across families.
  BTT errors are confined within blocks (spread evenly), while SVD errors
  concentrate in truncated directions (losing a whole output direction hurts more).
- **Recall grows only logarithmically with cost**, so matching `oracle_up` needs
  rank-256 (cost 0.917 → worse used-param cut than `oracle_up` itself). The family
  can never economically reach the oracle.

**Why it fails (structurally).** Investigation C proved it: low-rank is an
averaging operator, and the top-B set is the per-token deviation from the average.
A rank-32 activation-aware basis scores *below* a free static prior that reads no
`x` at all (0.353 vs 0.363). Low-rank's first component reconstructs the mean
token, which produces a per-token-constant score profile — exactly the static
prior. More rank buys diminishing returns on the residual.

---

### 3. Activation-aware low-rank (SVD-LLM style)

**Setting.** Replace plain SVD of `W` with SVD of `W Σ^{1/2}` (Σ = input
covariance). Also tested: shared cross-expert basis from `Σ^{1/2}MΣ^{1/2}`
(provably optimal), output whitening, quantized factors.

**Formulation.** Optimal shared basis: minimize `Σ_e E_x ‖(W_e − A_e P)x‖²` →
top-r eigenvectors of `Σ^{1/2} M Σ^{1/2}`, with `M = Σ_e W_eᵀW_e`.

**Results (recall@ρ=0.125, layer average).**

| scorer                              | cB    | recall | gain over static prior |
| ----------------------------------- | ----- | ------ | ---------------------- |
| `static_prior` (reads nothing)    | 0     | 0.363  | —                     |
| `actbasis_r32` (shared, cheapest) | 0.037 | 0.353  | **−0.010**      |
| `awsvd_r32` (per-expert)          | 0.115 | 0.419  | +0.056                 |
| `awsvd_r128`                      | 0.458 | 0.535  | +0.172                 |
| `sparse_probe` q3/k25             | 0.098 | 0.629  | **+0.266**       |

**Analysis.** Activation-awareness is a real gain over plain SVD (+0.035 recall at
any rank, free). But the family remains dead:

- At the goal budget (cB ≈ 0.10), best activation-aware gets recall 0.42 vs
  probe's 0.63 — a 5pt accuracy gap via the ladder.
- Quantized factors (buying rank cheaply) don't help: 3-bit rank-448 @ cB 0.10 →
  recall 0.457 vs probe 0.678. Bytes on rank are strictly worse than bytes on
  precision.
- The decisive control: `qawsvd_b4_r768` (recall 0.750) vs a plain 4-bit probe on
  dense x (recall **0.917**) at less cost. Rank never pays.

**Why.** The effective rank of the score-error metric `C = Σ^{1/2}MΣ^{1/2}` is only
**1.8–3.5** directions. Its top eigenvector *is* the mean token (`cos² ≥ 0.998`),
carrying 29–56% of score-damaging energy while the mean carries only 12–19% of raw
input energy. Activation weighting *concentrates* the budget onto the common mode —
the one part a per-token ranking cannot use.

**Analytic bonus (no GPU needed):**

- Output-side SVD ≡ activation-aware input SVD (proved, verified 3.8e-13 in fp64).
- Output whitening is a dead heat (per-channel norms too uniform).

---

### 4. Other mechanisms tested (all dominated)

Marked **[q]** = on the now-excluded quantization axis; kept because together they
are the evidence that compressing the *weight* never pays, which is what motivates
`input_sparse`. Unmarked rows are about the input/score axes and remain live.

| mechanism | result | why it fails |
| --------- | ------ | ------------ |
| **[q]** Product quantization (0.25–1.0 bits/w) | recall 0.42–0.60 vs RTN 0.71 at same bytes | Rows near-orthogonal (cos-to-centroid 0.078 vs 0.022 random) |
| **[q]** Hadamard rotation | 3-bit 0.818 vs 0.813 unrotated | No weight outliers; destroys input sparsity |
| **[q]** 1-bit sign + per-group scale | recall 0.546 at cB 0.126 | Below ~3 bits per-channel ordering is gone |
| **[q]** Asymmetric gate/up precision | 0.611 vs 0.675 symmetric | Score is a product; errors add in quadrature |
| **[q]** Router-adaptive precision | mass capped at 0.839 < 0.864 uniform | Redundancy is not expert-level |
| **[q]** Closed-form debiasing of the proxy | rel_err 0.437 vs 0.369 baseline — *worse* | Per-channel noise uniform; clamping destroys ordering |
| Norm-weighted coordinate selection (`\|x_i\|·rms_i(W)`) | ≈ plain top-`\|x\|` | Column-norm CV = 0.022 — nothing to weight |
| Relaxed-candidate cascade (λ=1.5) | recall 0.787, but no cheaper than a better probe at the same budget | The extra exact reads buy less than spending them on `ρ_input` |
| `‖W_down[:,j]‖` weighting of the score | rel_err 0.370 vs 0.369 | Factor CV = 0.055 |
| `\|gate\|` instead of `SiLU(gate)` | rel_err 0.647 vs 0.369 | The SiLU gate is load-bearing |
| Static pre-filter (ban low-freq channels) | loses 12–18% of top-B mass at 25% ban | Keep-frequency near-uniform |
| Per-row weight sparsity | recall 0.622 | Below top-`\|x\|` at serving precision (0.716) and needs extra indexed structure |
| Static input subset (one fixed coordinate set) | recall 0.558 | Overlap of top-`\|x\|` with a global set is only 0.375 |
| Discriminability-weighted input | ≈ top-`\|x\|` | `CV(Var_j W) = 0.042`; 93.8% identical selection |
| Floating the per-token **channel** budget under a global τ | −0.16…−0.32pt at iso-cost | Score is already `g_e`-scaled; τ tracks router confidence |

---

## Unifying negative result

> Expert weight rows carry near-maximal information per weight. Any operator that
> **averages** — low rank, shared bases, PQ codebooks, static priors — reproduces
> the part of the ranking that is *free* (the mean channel profile) and misses the
> part that matters (the per-token deviation from it). The only two axes that buy
> per-token signal are **precision** (bits per weight) and **adaptive support**
> (which input coordinates of `x` are read).

And of those two, **only adaptive support is economical**: precision at serving
precision is free (the weights are already there), while dropping *below* it costs
storage and accuracy at once. That is the whole content of `input_sparse` — spend
nothing on representing the weight, spend everything on choosing which of its
columns to read.

---

## Tools developed

| tool                                                                         | what it does                                                                                                                                                                                                                                                                                                                | when to use                                                                                                                                                       |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Output-error ladder** (`scripts/probe_output_error.py`)            | Measures layer-averaged block-output`rel_err = ‖y_full − y_kept‖/‖y_full‖`; fitted slope −24.3 HellaSwag pt/unit rel_err predicted two measured points to **within 0.1pt**                                                                                                                                    | Screening any new scorer design — 4 min, 1 GPU, replaces 150 GPU-hours of eval                                                                                   |
| **Linearity check** (`scripts/probe_relerr_linearity.py`)            | Regresses rel_err → measured accuracy per layer and per*family*; found the two families are different rulers (−26.4 vs −6.9 pt/unit)                                                                                                                                                                                   | **Before trusting any rel_err → pt conversion**; use the −26.4 fixed-budget slope for scorer comparisons                                                  |
| **All-layer surface + allocator** (`scripts/probe_layer_surface.py`) | `rel_err(ρ_input, ρ_channel)` over all 48 MoE layers (each measured against unperturbed input), then a **slope-weighted** Lagrangian solve for the per-layer budget split; emits a schedule JSON. `--reuse-surface` re-solves from cache with no GPU; `--no-slope-weight` reproduces the measured-worse unweighted objective | Choosing budgets across layers instead of hand-picking one global `(ρ_input, ρ_channel)` — but note the unweighted version **lost** an eval, so verify before trusting |
| **Input-allocation screen** (`scripts/probe_input_alloc_screen.py`)  | Compares`uniform`/`router`/`router2`/`colnorm` coordinate allocation at an identical pooled read budget                                                                                                                                                                                                             | Deciding how to split the probe's input reads across a token's K experts                                                                                          |
| **Threshold-vs-top-B** (`scripts/probe_threshold_budget.py`)         | Floats the per-token channel budget and/or input reads under an offline-calibrated global threshold; reports every mode**interpolated to the baseline's realized cost**                                                                                                                                               | Any "let the budget float per token" proposal — the iso-cost interpolation is what keeps a merely-cheaper variant from looking like a win                        |
| **Recall screen** (`scripts/probe_frontier.py`)                      | Index recall + captured score mass, fast rerun from cached captures                                                                                                                                                                                                                                                         | Quick within-family ranking (do NOT use to rank across families — BTT/SVD mismatch)                                                                              |
| **Static-floor diagnostic** (`scripts/actaware_diag_static.py`)      | Reports recall excess over a free static prior + centered Spearman (mean removed)                                                                                                                                                                                                                                           | **Run before proposing ANY new scorer** (~4 min, 1 GPU) — separates "buys per-token signal" from "re-derives the free prior"                               |
| **Activation-aware screen** (`scripts/actaware_scorer_screen.py`)    | All sketch families on one cost axis; recall/mass/rel_err                                                                                                                                                                                                                                                                   | Evaluating a new low-rank or basis-based family                                                                                                                   |
| **Learned probe** (`scripts/lowrank_scorer_learned_probe.py`)        | Trains an online-form scorer on calibration tokens, measures headroom above the SVD                                                                                                                                                                                                                                         | Showing whether objective or rank is the bottleneck                                                                                                               |
| **Probe capture** (`scripts/probe_capture.py`)                       | Caches per-layer`gate/up/down` activations + `x`, `g_e` for offline analysis                                                                                                                                                                                                                                          | Prerequisite for`probe_output_error.py` (needs the `_wd` captures)                                                                                            |
| **Low-rank recall** (`scripts/lowrank_scorer_recall.py`)             | Replays exact per-token cross-expert selection over 4 layers × 8192 tokens                                                                                                                                                                                                                                                 | Scanning the SVD/BTT cost–recall curve                                                                                                                           |

**Methodological lesson:** recall orders correctly *within* a scorer family but
**not across families** (pearson 0.65 with accuracy). The output-error ladder is
the correct screening instrument for cross-family comparison.

---

## Honest accounting summary

Three frames have appeared across these docs. **Only the last is used here**, and
the older two are recorded so old numbers can be re-read:

| frame                                | formula                    | why it is not used                                     |
| ------------------------------------ | -------------------------- | ------------------------------------------------------ |
| Proxy cost (oldest, `sparse_probe.md`) | cut = 1 − ρ − cB/3       | Prices a *separate* quantized scorer object, which is excluded — and it double-counts nothing but also credits nothing to storage |
| Overlap-discounted traffic           | kept = ρ + 2·p·(1−ρ)/3   | Defensible for memory traffic under a perfect cache, but it assumes the kernel keeps the scoring reads resident, and it flatters the method |
| **Used parameters (this doc)** | **kept = ρ + 2·p/3** | **scoring params + compute params, overlap billed twice — conservative, kernel-independent, and identical to `lowrank_scorer`'s frame** |

Reading old numbers: the switch from the discounted frame to this one costs
1.7–4.2pt of quoted cut (e.g. `ρ_input=0.25, ρ_channel=0.10` was −75.0%, now
**−73.3%**). Nothing
about accuracy changes — only what the budget is called.

Consequences:

- The scorer reads the served weights → **zero extra storage**, so no row needs a
  storage column.
- "Probe on dense `x`" (`ρ_input=1`) is not a deep cut at all: −20.8% at
  `ρ_channel=0.125`. Any
  headline claiming otherwise was an artifact of the proxy frame.
- **Never build the probe below serving precision.** It is worse on rel_err *and*
  costs +13% of expert weights.

---

## Tested and refuted: floating the per-token budget under a global threshold

The obvious next move — since the redundancy is per *token*, let the budget float
per token at a fixed mean, replacing "top-B" with "everything above `τ`" — was
tested before recommending it (`scripts/probe_threshold_budget.py`, 4 layers ×
3072 tokens, thresholds calibrated offline on the first 512 tokens to hit the
target mean, then frozen). Because thresholding changes cost *and* error, each mode
is read at the **baseline's realized cost** by interpolating its own curve:

| mode                            | used 0.267               | used 0.292               | used 0.317      | used 0.367      |
| ------------------------------- | ------------------------ | ------------------------ | --------------- | --------------- |
| `fixed_B / fixed_p` (current) | 0.4170                   | 0.3848                   | 0.3581          | 0.3158          |
| `thresh_B / fixed_p`          | 0.4244 **(−0.20pt)** | 0.3911 (−0.17)          | 0.3641 (−0.16) | 0.3281 (−0.32) |
| **`fixed_B / thresh_p`** | **0.4087 (+0.22pt)** | **0.3796 (+0.14)** | 0.3560 (+0.06)  | 0.3243 (−0.22) |
| `thresh_B / thresh_p`         | 0.4145 (+0.07)           | 0.3846 (+0.01)           | 0.3612 (−0.08) | 0.3402 (−0.64) |

**Floating the channel budget makes things worse** (−0.16 to −0.32pt at iso-cost),
consistently at every budget and in every one of the 4 layers. Floating only the
*input reads* is a small win that grows as the budget tightens (+0.22pt at
used=0.267, vanishing by used=0.317). The verdict is frame-invariant: re-derived
under `used = ρ + 2p/3` it is the same to within 0.03pt.

**Why the asymmetry.** The channel score `g_e·|SiLU⊙up|` is already scaled by
`g_e`, whose per-token variance is large, so a single global `τ_s` mostly measures
*how confident the router was*, not how compressible the token is: high-`g` tokens
blow past the threshold and hoard channels they do not need. The input criterion
`|x_i|` has no such confounder, so `τ_x` genuinely tracks how concentrated the
token's hidden state is. The lesson is narrower than "make it dynamic": **float
the budget only where the criterion is not already router-scaled.**

This also revises the reading of the induced thresholds from the schedule solve —
the `|x|` cut (0.6676) is worth deploying as a threshold; the score cut (0.03016)
is not, and the per-token top-B should stay a top-B.

---

## Proposed strategy — where the remaining headroom actually is

Ranked by expected value per unit of work, given everything above.

**1. Spend the `router` gain on the budget, not on accuracy.** `router` is the one
allocation win that is *measured* (+0.58pt HellaSwag / +0.47pt MMLU at fixed cost).
Via the −26.4 slope that is worth **~1.5–2pt of extra cut at fixed accuracy**,
pushing the frontier from −73.3% toward roughly −75%. Since `router` is free and now
confirmed end-to-end, the cheapest next result is simply re-running it at a deeper
ρ and reading off where it crosses the −73.3% accuracy.

**2. Threshold the input reads only** (+0.25pt at the tightest budget, free, and it
*removes* a sort from the kernel). The measured asymmetry above says to apply this
to `τ_x` and leave the channel selection a per-token top-B. Cheap to build: the
threshold is one offline-calibrated scalar per layer, already computed.

**3. LoRA recovery at the goal point (unchanged, still the largest single win).**
The gap to the full-width oracle at −73.3% is 2.47pt on HellaSwag and 1.77pt on MMLU
— exactly the size stage-2 LoRA closes elsewhere in this repo. The pipeline
already exists; nothing new needs designing. This is the most likely route to
"parity at −73.3%".

**4. Do not spend more on cross-layer allocation.** Settled negatively: unweighted
lost 0.16pt, slope-weighted lost 3.29pt, and the best uniform grid point beats both.
The separable objective is the wrong *form*, not just mis-weighted (see "The schedule
failure"). Anything further here needs an objective that models error compounding
across depth, which costs far more to fit than the ≤0.5pt it was chasing. **Uniform
`(ρ_input, ρ_channel)` is the recommended default.**

**5. Stack with reduce-top-k.** Orthogonal axis, untouched here: top-4 ×
`input_sparse` reaches beyond −85%, and a smaller `K·I` pool is an easier ranking
problem for the probe (fewer cross-expert comparisons at the same `B`).

**Do not bother with:** anything that quantizes the probe (dominated at serving precision on
both storage *and* accuracy), `colnorm`-style coordinate reweighting (wash on two
independent metrics), or `g²`-sharpened coordinate allocation (`router2`
overshoots `router` at every budget).

---

## What is still open

| direction                                 | status                                                  | expected value                                                                                                                     |
| ----------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **LoRA recovery at the goal point** | not started                                             | The 3.92pt HellaSwag / 3.24pt MMLU gap at −73.3% is the size stage-2 LoRA closes elsewhere;**now the largest single win**     |
| **`router` at a deeper ρ**       | not started                                             | The +0.58pt is confirmed, so spend it: ~−77% at the current accuracy. Cheapest remaining result                                   |
| **Why the schedule trades HS for MMLU** | observed, unexplained | The unweighted schedule is −0.16 HS / **+0.65 MMLU** at iso-cost. If the trade is controllable it is a knob, not a bug |
| **Non-separable cross-layer objective** | not attempted | Every separable solve failed; an objective that sees compounding across depth is the only remaining route, and is much more expensive to fit |
| **Threshold the input reads only**  | measured offline (+0.22pt at used=0.267), not evaluated | Free, removes a sort; channel budget must stay a top-B (measured)                                                                  |
| **Sequence-level carry-forward**    | untested                                                | Consecutive tokens share hot channels → zero-weight-read scorer from an EMA of recent activations; needs order-preserving capture |
| **Stack with reduce-top-k**         | not started                                             | top-4 ×`input_sparse` reaches beyond −85%; smaller K·I pool is an easier ranking problem                                      |
| **Wall-clock microbenchmark**       | not done                                                | Everything is an active-parameter claim; gathered-expert latency needs measuring                                                   |

**Closed by this round:**

- **−73.3% with zero extra storage is real**: 74.64 HellaSwag / 77.67 MMLU (row 7).
- **`router` input allocation works end-to-end** (+0.58 / +0.47pt at fixed cost),
  confirming the offline screen's +0.66 prediction.
- **MMLU at the goal point** — measured; the gap to the full-width oracle *narrows*
  rather than widens, refuting this doc's prior expectation.
- **A separate quantized proxy is never worth it** — serving precision dominates on storage
  *and* accuracy.
- **Per-expert-equal input budgets are wrong** — `router` wins in all 16 cells.
- **The per-token *channel* budget must stay a top-B** — floating it under a global
  threshold costs 0.16–0.32pt at iso-cost, because the score is already `g_e`-scaled.
- **The rel_err ladder's scope is now bounded**: excellent for selector changes
  (±0.26pt on two pre-registered predictions), **invalid** for cross-layer budget
  moves (missed by 1.0pt) unless weighted by per-layer sensitivity.
- **Per-layer budget allocation does not work on this model** — three solves
  (unweighted −0.16pt, slope-weighted −3.29pt, and every uniform grid point) and
  **uniform wins**. A separable `Σ_L w_L·rel_err_L` objective is the wrong form, not
  merely mis-weighted: it cannot see error compounding across depth. The 2.92×
  per-layer spread is real but not collectable this way.
- **The same schedule is −0.16pt on HellaSwag and +0.65pt on MMLU**, so a scalar
  objective cannot represent what reallocation even does.

---

## Reproducing

### Offline screens (shared)

```bash
# Recall screen (1 GPU, cached captures, minutes)
python scripts/probe_frontier.py

# Output-error ladder (1 GPU, needs _wd captures). "16:p" = input_sparse at that p.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python scripts/probe_output_error.py \
    --layers 6,22,38,46 --ratios 0.25,0.125 \
    --probes "16:1.0,16:0.5,16:0.375,16:0.25,16:0.1875,16:0.125"

# The served-at-4-bit special case: there a 4-bit probe IS input_sparse, so this
# reproduces the "excess rel_err 0.0000 at p=1" identity.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python scripts/probe_output_error.py \
    --serve-bits 4 --layers 6,22,38,46 --ratios 0.25,0.125 \
    --probes "4:1.0,4:0.5,4:0.375,4:0.25,4:0.1875,4:0.125" \
    --out docs/results/btt_dynamic/reuse_frontier.json
```

### `input_sparse`: input allocation, per-layer schedule

```bash
# 1. Is rel_err a linear predictor? (no GPU, seconds)
python scripts/probe_relerr_linearity.py
python scripts/probe_relerr_linearity_plot.py

# 2. Which input-allocation term? (1 GPU, ~10 min, cached _wd captures)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python scripts/probe_input_alloc_screen.py \
    --layers 6,22,38,46 --max-tokens 4096 --chunk 256 \
    --ps 0.25 --rhos 0.10,0.125,0.15,0.20

# 3. All-layer surface + the allocation solve (4 GPUs, ~40 min, needs the model)
python scripts/probe_layer_surface.py --tokens 8192 --batch-size 4 \
    --out docs/results/idea_pilot/layer_surface_8k.json
python scripts/probe_layer_surface_plot.py
# re-solve a cached surface under a different objective (no GPU, seconds)
python scripts/probe_layer_surface.py --targets 0.25 \
    --reuse-surface docs/results/idea_pilot/layer_surface_8k.json \
    --out docs/results/idea_pilot/layer_surface_8k_weighted.json
# schedules: schedule_cut75.json (unweighted, MEASURED WORSE than uniform),
#            schedule_cut75_weighted.json (slope-weighted, under eval)

# 3b. Does a floating per-token budget help? (1 GPU, ~10 min) -- channel: NO, input: small yes
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python scripts/probe_threshold_budget.py \
    --layers 6,22,38,46 --max-tokens 3072 --chunk 256 \
    --rhos 0.10,0.125,0.15,0.20 --input-alloc router

# 4. Evals (A100, 8 GPUs, 2 waves)
bash scripts/run_probe_reuse_sweep.sh    # rho_input=0.25 x rho_channel {0.10,0.15,0.20}
bash scripts/run_probe_alloc_sweep.sh    # router alloc / per-layer schedule / both
# single config, e.g. the slope-weighted schedule:
python src/train/merge_slim_eval.py \
    --config configs/eval/qwen3_30b_a3b_probe_schedw_cut75_hellaswag.yaml

# Unit tests (30; anchors: view-not-copy, used-param limits, pooled-top-N)
python -m pytest src/dynamic_active_param/tests/test_sparse_probe.py -q
```

### Low-rank scorer

```bash
# Recall vs cost (4 layers × 8192 tokens)
python scripts/lowrank_scorer_recall.py --layers 6,22,38,46 --tokens 8192 \
    --chunk 1024 --ranks 4,8,16,32 --grids 1x1,2x2,4x2 --out-dir docs/results/btt_dynamic
python scripts/lowrank_scorer_recall.py --layers 6,22,38,46 --tokens 8192 \
    --chunk 1024 --ranks 64,128,256 --grids 1x1,2x2 --out-dir docs/results/btt_dynamic_hirank

# Evals (A100, 8 GPUs)
bash scripts/run_lowrank_scorer_sweep.sh    # svd_r32 pair
bash scripts/run_lowrank_scorer_tail.sh     # remaining configs

# Learned probe (investigation B)
python scripts/lowrank_scorer_learned_probe.py --layer 46 --rank 16 --experts 4

# Unit tests
python -m pytest src/dynamic_active_param/tests/test_lowrank_scorer.py -q  # 16 tests
```

### Activation-aware / static floor

```bash
# Static-floor diagnostic (RUN FIRST before any new scorer)
python scripts/actaware_diag_static.py --layers 6,22,38,46 --ranks 32,128 \
    --out docs/results/actaware/static_diag.json

# Full activation-aware screen
python scripts/actaware_scorer_screen.py --layers 46 --fit-tokens 4096 \
    --score-tokens 1024 --ranks 8,16,32,64,88,128,256 \
    --variants ref,pcabasis,actbasis,awsvd,outwhiten,svd,insp,adapt,mix \
    --gate-modes upgate,uponly --out docs/results/actaware/screen_L46.json

# Quantized factors (the decisive rank-vs-precision control)
python scripts/actaware_scorer_screen.py --layers 46 --variants ref,qbasis,qawsvd \
    --ranks 32 --qbits 3,4 --qranks 128,256,448,768 --gate-modes upgate \
    --out docs/results/actaware/screen_qbasis_L46.json
```

### Implementation pointers

| component                                                 | path                                                                                                                           |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `input_sparse` criterion (+ input allocation) | `src/dynamic_active_param/sparse_probe.py`                                                                                   |
| Sparse probe tests (30)                                   | `src/dynamic_active_param/tests/test_sparse_probe.py`                                                                        |
| Per-layer schedule loading                                | `src/dynamic_active_param/install.py` (`schedule_path`)                                                                    |
| Linearity study                                           | `scripts/probe_relerr_linearity.py`, `..._plot.py`                                                                         |
| All-layer surface + allocator                             | `scripts/probe_layer_surface.py`, `..._plot.py`                                                                            |
| Input-allocation screen                                   | `scripts/probe_input_alloc_screen.py`                                                                                        |
| Threshold-vs-top-B study                                  | `scripts/probe_threshold_budget.py`                                                                                          |
| Solved −73.3% schedules                                    | `docs/results/idea_pilot/schedule_cut75{,_8k,_weighted}.json`                                                                |
| Eval configs (this round)                                 | `configs/eval/qwen3_30b_a3b_probe_{reuse_k25_r{10,15,20},router_k25_r10,sched_cut75,sched_router_cut75,schedw_cut75}_*.yaml` |
| Low-rank scorer (SVD/BTT)                                 | `src/dynamic_active_param/lowrank_scorer.py`                                                                                 |
| Low-rank tests (16)                                       | `src/dynamic_active_param/tests/test_lowrank_scorer.py`                                                                      |
| Probe output error                                        | `scripts/probe_output_error.py`                                                                                              |
| Probe recall                                              | `scripts/probe_frontier.py`                                                                                                  |
| Probe capture (prerequisite)                              | `scripts/probe_capture.py`                                                                                                   |
| Static floor diagnostic                                   | `scripts/actaware_diag_static.py`                                                                                            |
| Activation-aware screen                                   | `scripts/actaware_scorer_screen.py`                                                                                          |
| Learned probe experiment                                  | `scripts/lowrank_scorer_learned_probe.py`                                                                                    |
| Results (JSON)                                            | `docs/results/btt_dynamic/`, `docs/results/actaware/`                                                                      |
| Figures                                                   | `docs/exps/dynamic_active_param/figures/btt_dynamic/`                                                                        |
| Eval configs                                              | `configs/eval/qwen3_30b_a3b_probe_*.yaml`, `configs/eval/qwen3_30b_a3b_dynamic_lowrank_*.yaml`                             |

---

## Prior art

- **Prox (arXiv:2607.27591):** quantized proxy + input sparsity → rank → exact
  compute for SwiGLU FFNs. Same mechanism, 10 dense LLMs, zero MoE. Must cite.
- **SVD-LLM:** activation-aware SVD; the theory is confirmed (consistent +0.035
  gain), the application to channel *selection* is novel and negative.
