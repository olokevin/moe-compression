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

| symbol                   | config key                                     | what it keeps                                  | paid to           |
| ------------------------ | ---------------------------------------------- | ---------------------------------------------- | ----------------- |
| **`ρ_input`**   | `probe.rho_input`                            | fraction of the token's input coordinates read | **scoring** |
| **`ρ_channel`** | `probe.rho_channel` (= `1 − prune_ratio`) | fraction of the pooled`K·I` channels kept   | **compute** |

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

**Qwen3-30B-A3B**, **HellaSwag 0-shot acc_norm (dense = 78.56) and full MMLU 5-shot
(dense = 80.91)**. stderr ≈0.42pt / ≈0.34pt. Masking simulation, no fine-tuning.
All `input_sparse` rows read the served weights: **zero extra storage**.

| #  | method                                                      | `ρ_input` | `ρ_channel` | used-param cut    | HellaSwag       | MMLU 5-shot     | status                                  |
| -- | ----------------------------------------------------------- | ------------ | -------------- | ----------------- | --------------- | --------------- | --------------------------------------- |
| 0a | **`weight_sparse` `tau`** (unstructured, uniform) | 0.1125 †    | 0.125          | **−80.0%** | **74.87** | *rerunning*   | HS measured (ladder 74.7 ✓); MMLU OOM'd 08-14, relaunched 08-19 |
| 0b | `weight_sparse` rectangle `0.25x0.45` (1 level)         | 0.1125 †    | 0.125          | **−80.0%** | **74.47** | **77.62** | measured (ladder 74.1 ✓)               |
| 0c | `weight_sparse` `tau` + `router`                      | 0.1125 †    | 0.125          | **−80.0%** | **74.41** | **77.82** | measured (ladder 75.0, −0.6)           |
| 0d | `input_sparse` + `router` at the same reads (control)   | 0.1125       | 0.125          | **−80.0%** | **71.41** | **75.82** | measured (ladder 73.4,**−1.95**) |
| 0e | **`input_only` + `router`** (one-pass, symmetric)  | 0.300 ‡     | 0.300 ‡        | **−70.0%** | **73.18** | **77.60** | measured (2-pass −2.21 better)          |
| 0f | **`input_only` + `router`** (one-pass, symmetric)  | 0.250 ‡     | 0.250 ‡        | **−75.0%** | **71.35** | **76.06** | measured (2-pass −2.73 better)          |
| 0g | **`input_only` + `router`** (one-pass, symmetric)  | 0.200 ‡     | 0.200 ‡        | **−80.0%** | **67.38** | **73.97** | measured (2-pass −5.17 better)          |
| 0h | `input_only` + `uniform` (one-pass, symmetric)           | 0.300 ‡     | 0.300 ‡        | **−70.0%** | **64.33** | **70.24** | measured (`router` control, −8.85)    |
| 0i | `input_only` + `uniform` (one-pass, symmetric)           | 0.250 ‡     | 0.250 ‡        | **−75.0%** | **56.95** | **62.82** | measured (`router` control, −14.40)   |
| 0j | `input_only` + `uniform` (one-pass, symmetric)           | 0.200 ‡     | 0.200 ‡        | **−80.0%** | **45.58** | **49.52** | measured (`router` control, −21.80)   |
| 1  | `oracle_mag_noW` (full gate+up, single pass)              | —           | 0.125          | −29.2%           | **77.11** | **79.44** | measured                                |
| 2  | `input_sparse`                                            | 1.00         | 0.125          | −20.8%           | **76.95** | —              | measured ¶                             |
| 3  | `input_sparse`                                            | 0.50         | 0.125          | −54.2%           | ~76.6           | —              | predicted (ladder)                      |
| 4  | `input_sparse`                                            | 0.25         | 0.20           | −63.3%           | **76.72** | **79.10** | measured                                |
| 5  | `input_sparse`                                            | 0.25         | 0.15           | **−68.3%** | **76.47** | **78.63** | measured                                |
| 6  | `input_sparse`                                            | 0.25         | 0.10           | **−73.3%** | **74.06** | **77.20** | measured (ladder 74.2)                  |
| 7  | **`input_sparse` + `router` alloc**               | 0.25         | 0.10           | **−73.3%** | **74.64** | **77.67** | measured (ladder 74.9)                  |
| 7a | **`input_sparse` + `router`, solved split**       | 0.1875       | 0.14170        | **−73.3%** | **74.63** | **77.94** | measured (ties row 7)                   |
| 7b | **`input_sparse` + `router`, solved split**       | 0.1875       | 0.125          | **−75.0%** | **74.08** | **77.77** | measured                                |
| 7c | `input_sparse` + `router`, mis-allocated split          | 0.25         | 0.0833         | −75.0%           | **73.80** | **77.33** | measured (iso-cost control)             |
| 7d | **`input_sparse` + `router`, solved split**       | 0.1875       | 0.10000        | **−77.5%** | **73.33** | **76.81** | measured                                |
| 7e | **`input_sparse` + `router`, solved split**       | 0.1575       | 0.09500        | **−80.0%** | **72.55** | **76.11** | measured                                |
| 8  | `input_sparse` + per-layer schedule (unweighted)          | mixed        | mixed          | −73.3%           | **73.90** | **77.85** | measured (ladder 74.9 ✗)               |
| 8b | `input_sparse` + per-layer schedule (slope-weighted)      | mixed        | mixed          | −73.0%           | **70.77** | —              | measured (**fails**)              |
| 9  | `oracle_up` (full up, cut gate)                           | —           | 0.125          | −58.3%           | **71.30** | **76.43** | measured                                |
| 10 | `lowrank_scorer` BTT m2n2 r32 (up-only)                   | —           | 0.25           | −71.2%           | **65.97** | —              | measured                                |
| 11 | `lowrank_scorer` SVD r32 (up-only)                        | —           | 0.25           | −73.1%           | **63.94** | —              | measured                                |
| 12 | Level-1`pivchol` (offline, no scoring pass)               | —           | 0.125          | −87.5%           | **44.15** | **45.51** | measured                                |

† Rows 0a–0d are on a different sparsity *pattern*, not a different budget: the
0.1125 is a **read density** over `(channel, coordinate)` entries rather than whole
columns, so all four read exactly what `input_sparse` at `ρ_input=0.1125` reads (row 0d
*is* `input_sparse` at that `ρ_input`, and is the control). See
[Unstructured sparsity](#unstructured-sparsity-buy-entries-not-coordinates). Rows 0a–0c
pay 4–17% of extra *storage* for positional metadata, which rows 1–12 do not.
Realized read density was verified in-run over the eval stream: 0.1118 and 0.1123 of a
branch against the 0.1125 target, i.e. `used` 0.1995 / 0.1999.

‡ Rows 0e–0j are the **one-pass** method
[`input_only`](#input_only-stop-scoring-just-compute-on-the-sparse-input), whose cost
is `(2ρ_input + ρ_channel)/3` rather than `ρ_channel + 2ρ_input/3` — the sparse read
*is* the compute, so nothing is billed twice and the two sparsities are not
comparable row-for-row with rows 1–12. At the symmetric split `ρ_input = ρ_channel`
the used-parameter cut equals `1 −` that sparsity exactly. **The method is dominated:
at identical `used` the two-pass rows beat it by 2.21 / 2.73 / 5.17pt HellaSwag, and
the gap widens with depth** — so the discarded exact pass is worth paying for. Rows
0h–0j are the `uniform` controls, and they carry the section's real finding:
`router` allocation is worth **+8.85 / +14.40 / +21.80pt** here, the largest single
effect in this document.

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

- **The headline is now −80.0% used parameters at 74.87 HellaSwag** (row 0a),
  which is 6.7 points deeper than the old −73.3% row *and* 0.23pt above it. What buys
  it is the sparsity **pattern**, not the budget: at identical reads, spending them on
  `(channel, coordinate)` entries beats spending them on whole coordinates by
  **+3.46pt** (74.87 vs the 71.41 control, rows 0a vs 0d — 5.5σ). The price is 4–17%
  of extra *storage* for positional metadata; see
  [Unstructured sparsity](#unstructured-sparsity-buy-entries-not-coordinates).
- **The zero-extra-storage frontier is now a full curve on both metrics** (rows 7a–7e
  plus 6/7), six budgets measured end-to-end:
  **−63.3% → 76.61 / 79.45**, **−73.3% → 74.63 / 77.94**, **−77.5% → 73.33 / 76.81**,
  **−80.0% → 72.55 / 76.11** (dense 78.56 / 80.91). It reaches **−80.0% while staying
  within 6.01pt HellaSwag / 4.80pt MMLU of dense**, descending smoothly at ≈0.24pt
  HellaSwag per pp with **no cliff**. Against the prior selectors at comparable depth:
  **+32.4pt over Level-1 `pivchol`** (which is at 44.15 by −87.5%) and **+3.33pt over
  `oracle_up`** at −73.3% while cutting 15.0 more points of the FFN. Figures:
  `docs/presentation/figs/fig_probe_curve_{hellaswag,mmlu}.pdf`.
- **MMLU degrades more gracefully at every budget, and the gap widens with depth** —
  0.49pt less loss than HellaSwag at −63.3%, 1.21pt less at −80.0%. Mechanism: the
  fitted rel_err→accuracy slope is **−20.3 for MMLU vs −24.5 for HellaSwag**, so the
  same block-output error buys less MMLU damage.
- **It beats the low-rank family at matched cost by ~9–11pt.** Rows 10–11 sit at
  −71.2% / −73.1% on the *same* accounting and score 65.97 / 63.94; row 7 is
  −73.3% at 74.64. That is the cleanest iso-cost comparison in this table, since
  `lowrank_scorer` already used the scoring-plus-compute frame.
- **`router` input allocation is confirmed but budget-dependent: +0.58pt HellaSwag,
  +0.47pt MMLU at `ρ_channel=0.10`** (74.06→74.64, 77.20→77.67) at *identical* cost and
  reads/token, predicted +0.66 — but only **−0.11 / +0.35** by `ρ_channel=0.20`, where
  the offline screen predicted a near-identical +0.53. Free either way (one extra
  bisection over a sort the layer already computes), and worth the most exactly where
  this method operates. And it is worth **vastly** more when the sparse reads carry the *values*
  rather than a ranking: **+8.9 to +21.8pt** for one-pass
  [`input_only`](#input_only-stop-scoring-just-compute-on-the-sparse-input) (rows
  0e–0j), the largest single effect in this document.
- **The two-pass structure is worth paying for, now measured.** Reusing the sparse
  read as the computation instead of re-reading exactly (`input_only`, rows 0e–0g)
  loses **2.21 / 2.73 / 5.17pt** HellaSwag at identical used-parameters. So the third
  of the budget `input_sparse` spends on a decision it discards buys back more than it
  costs — the first direct test of that design choice.
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

|                                        | separate 3-bit proxy             | `input_sparse` (served weights)    |
| -------------------------------------- | -------------------------------- | ------------------------------------ |
| extra storage                          | **+13% of expert weights** | **0** (measured 0.00 MB alloc) |
| error sources                          | input sparsity**+ quantization** | input sparsity**only**         |
| ranking quality at matched`ρ_input` | strictly worse                   | better                               |

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

| `ρ_channel` | scoring | compute | used   | used-param cut    |
| -------------- | ------- | ------- | ------ | ----------------- |
| 0.100          | 0.1667  | 0.100   | 0.2667 | **−73.3%** |
| 0.125          | 0.1667  | 0.125   | 0.2917 | −70.8%           |
| 0.150          | 0.1667  | 0.150   | 0.3167 | −68.3%           |
| 0.200          | 0.1667  | 0.200   | 0.3667 | −63.3%           |

And the mirror image — `ρ_channel=0.125` fixed, varying `ρ_input`:

| `ρ_input` | scoring | compute | used   | used-param cut    |
| ------------ | ------- | ------- | ------ | ----------------- |
| 0.125        | 0.0833  | 0.125   | 0.2083 | **−79.2%** |
| 0.1875       | 0.1250  | 0.125   | 0.2500 | −75.0%           |
| 0.25         | 0.1667  | 0.125   | 0.2917 | −70.8%           |
| 0.50         | 0.3333  | 0.125   | 0.4583 | −54.2%           |
| 1.00         | 0.6667  | 0.125   | 0.7917 | −20.8%           |

**Reading the two tables together.** The scoring term `2·ρ_input/3` carries a 3×
discount (two branches spread over a three-matrix FFN) that the compute term
`3·ρ_channel/3 = ρ_channel` does not, so a unit of `ρ_input` costs **two thirds**
of a unit of `ρ_channel`. That is why `ρ_input` is the axis to cut first — and why
`ρ_input=1` is hopeless regardless of `ρ_channel`: scoring alone (0.667) already
exceeds what the whole method is trying to get under.

**Measured budget sweep** (`input_sparse`, `ρ_input=0.25`, `uniform` allocation,
masking sim, no finetune; dense = 78.56 / 80.91):

| `ρ_channel`    | used-param cut    | HellaSwag acc_norm | Δ vs dense      | MMLU 5-shot     | Δ vs dense      |
| ----------------- | ----------------- | ------------------ | ---------------- | --------------- | ---------------- |
| 0.200             | −63.3%           | **76.72**    | −1.84           | **79.10** | −1.81           |
| 0.150             | −68.3%           | **76.47**    | −2.09           | **78.63** | −2.28           |
| 0.100             | **−73.3%** | **74.06**    | −4.50           | **77.20** | −3.71           |
| 0.100 +`router` | **−73.3%** | **74.64**    | **−3.92** | **77.67** | **−3.24** |

stderr ≈0.43pt (HellaSwag) / ≈0.33pt (MMLU). Both benchmarks degrade smoothly here
— no cliff of the kind Level-1 hits at −87.5% — and **MMLU consistently loses less
than HellaSwag** at the same cut.

### How to split a *fixed* budget between the two sparsities

Every row above picked `ρ_input=0.25` by hand and then varied `ρ_channel`, which
moves the split **and** the budget together. So the claim "cutting `ρ_input` is
cheaper" was inferred from points at *different* costs and had never been tested
at iso-cost. The clean question: given a budget `C`, what is the best split?

**The principle** is one Lagrange multiplier — the *within-layer* analogue of the
cross-layer solve that failed:

```
minimize  rel_err(ρ_input, ρ_channel)   s.t.   ρ_channel + 2·ρ_input/3 = C
  =>  (3/2) · ∂rel_err/∂ρ_input  ==  ∂rel_err/∂ρ_channel
```

The `3/2` is exactly the discount `ρ_input` carries. Unlike the per-layer schedule,
this is a **selector** change (one global pair for all 48 layers), which is the
regime where the ladder is validated. Solved on the cached 8192-token surface
(`layer_surface_8k.json`, bilinear in `(p, ρ)`) — **no GPU needed, the data was
already there**:

| budget`C`      | cut               | **`ρ_input`\*** | **`ρ_channel`\*** | rel_err          | scoring share   |
| ---------------- | ----------------- | ------------------------ | -------------------------- | ---------------- | --------------- |
| 0.2000           | −80.0%           | 0.1575                   | 0.0950                     | 0.5234           | 52.5%           |
| 0.2250           | −77.5%           | 0.1875                   | 0.1000                     | 0.4923           | 55.6%           |
| **0.2500** | **−75.0%** | **0.1875**         | **0.1250**           | **0.4613** | **50.0%** |
| 0.2667           | −73.3%           | 0.1875                   | 0.1417                     | 0.4445           | 46.9%           |
| 0.2917           | −70.8%           | 0.2025                   | 0.1567                     | 0.4204           | 46.3%           |
| 0.3667           | −63.3%           | 0.2500                   | 0.2000                     | 0.3559           | 45.5%           |

**`ρ_input`\* sits at 0.1875 across the whole deep-budget range** — the hand-picked
0.25 was too much scoring at every budget at or below −73.3%, and the optimum keeps
the split near **50/50**, not the 67/33 that `ρ_input=0.25` forces at `C=0.25`.

**Measured at `C=0.2500` (−75.0%), varying only the split** (`router` alloc, masking
sim, no finetune; two arms, both verified in-run at `USED PARAMS=0.2500`):

| arm               | `ρ_input`     | `ρ_channel`  | scoring/compute | `B` | reads | surface rel_err  | HellaSwag       | predicted |
| ----------------- | ---------------- | --------------- | --------------- | ----- | ----- | ---------------- | --------------- | --------- |
| **`opt`** | **0.1875** | **0.125** | 50 / 50         | 768   | 384   | **0.4613** | **74.08** | 74.53     |
| `mis`           | 0.25             | 0.0833          | 67 / 33         | 512   | 512   | 0.4833           | **73.80** | 73.95     |

**Verdict after three iso-cost tests: solving the split is a wash.** `opt` beats
`mis` here by **+0.28pt HS / +0.44pt MMLU** against a pre-registered **+0.58pt**,
with a ≈0.62pt stderr on the difference — inside one sigma of both the prediction
and of zero. The full sweep then added a *cleaner* test at −73.3%, where the
hand-picked and solved splits can be compared at **identical** `router` allocation:

| budget  | hand-picked /`mis`           | solved                           | Δ HellaSwag     | Δ MMLU         | ladder predicted |
| ------- | ------------------------------ | -------------------------------- | ---------------- | --------------- | ---------------- |
| −73.3% | 0.25 / 0.10 → 74.64 / 77.67   | 0.1875 / 0.1417 → 74.63 / 77.94 | **−0.01** | **+0.27** | +0.34            |
| −75.0% | 0.25 / 0.0833 → 73.80 / 77.33 | 0.1875 / 0.125 → 74.08 / 77.77  | **+0.28**  | **+0.44** | +0.58            |

Four iso-cost measurements: **−0.01 / +0.27 / +0.28 / +0.44**, mean ≈ **+0.25pt**,
every one at or inside its ≈0.5–0.6pt stderr and every one at or below its prediction.
Two readings that survive the full sweep:

- **On HellaSwag it is a wash.** The −73.3% pair — the cleanest test, since only the
  split differs — is a **dead tie** (−0.01), and −75.0% is +0.28.
- **On MMLU it is consistently, weakly positive** (+0.27 and +0.44, both budgets, same
  sign). Suggestive of a real small effect rather than noise, but neither point clears
  its own stderr, so it stays "probably slightly helps".

**But at −80.0% the split matters a lot.** Row 0d is an unintended fifth iso-cost test:
`input_sparse` + `router` at `ρ_input=0.1125, ρ_channel=0.125` (37.5% of the budget on
scoring), against row 7e's solved `0.1575 / 0.095` (52.5%, the optimum) at the *same*
`used = 0.2000`:

| −80.0%, both`input_sparse`+`router` | `ρ_input`     | `ρ_channel`  | scoring share   | HellaSwag       |
| ---------------------------------------- | ---------------- | --------------- | --------------- | --------------- |
| row 0d (under-scoring)                   | 0.1125           | 0.125           | 37.5%           | **71.41** |
| **row 7e (solved)**                | **0.1575** | **0.095** | **52.5%** | **72.55** |

**+1.14pt for the solved split** — well outside stderr, and ~4× the effect at −75.0%.

So the correct statement is **depth-dependent**: the split is a flat direction at
moderate budgets (−0.01 to +0.44pt over three tests at −73.3%/−75.0%) and a **real
lever once the budget gets tight** (+1.14pt at −80.0%). The intuition is that at
`used=0.20` there is no slack anywhere, so mis-spending 15pp of the budget on the wrong
axis can no longer be absorbed. Practical rule: **hand-picking is harmless down to
about −75%, and worth solving below that.** The seconds-long offline solve is cheap
enough to just always run.

One caveat on the ladder here: `ρ_input=0.1125` is *below* the surface's grid minimum
(0.125), so `probe_split_solve.py` clamps rather than interpolates and its rel_err for
row 0d is not trustworthy — indeed it predicts the wrong sign (−0.21pt vs the measured
+1.14). The **measurement** is the evidence at this budget, not the ladder.

This is a **negative result for the split solve specifically**, not for the surface:
see the sweep-wide ladder check below, where the same surface predicts the *whole*
accuracy curve at R² = 0.997. rel_err tracks accuracy well **along** the budget axis
and poorly **across** iso-cost splits — a narrower version of the same lesson the
per-layer schedule taught.

**What it does buy is depth.** Placed against the two −73.3% rows:

| row                                  | `ρ_input` | `ρ_channel` | cut               | HellaSwag       |
| ------------------------------------ | ------------ | -------------- | ----------------- | --------------- |
| 6 (uniform)                          | 0.25         | 0.10           | −73.3%           | 74.06           |
| 7 (`router`)                       | 0.25         | 0.10           | −73.3%           | **74.64** |
| **7a (`router`, opt split)** | 0.1875       | 0.125          | **−75.0%** | **74.08** |

Row 7a **ties row 6** (74.08 vs 74.06) while cutting **1.7pp more** — so relative to
the uniform baseline the extra cut is free. But it does **not** dominate row 7:
against the actual best it is −0.56pt for +1.7pp of cut, i.e. a local trade rate of
roughly **0.33pt per pp** in this region. Neither point dominates the other; 7a
**extends** the frontier to −75.0% rather than replacing the −73.3% headline.

Two caveats. The surface was measured under `uniform` allocation, so `router`'s
+0.58pt is *stacked* from its own end-to-end delta rather than read off the surface
— fine for ranking the two arms against each other, weaker as an absolute
prediction, and the −0.45pt miss on `opt` is where that shows. And `ρ_input`\* is
read off a grid whose nearest neighbours are 0.125 and 0.25, so 0.1875 is "the best
grid point", not a solved continuum optimum.

### The full best-practice sweep — six budgets

All six budget points at best-practice settings (`bits=16`, `use_gate=true`,
`input_alloc=router`, `lam=1.0`, `k_min=0`, no schedule) with the split solved per
budget. **HellaSwag complete; MMLU in flight** (only −75.0% measured so far).
Figures: `docs/presentation/figs/fig_probe_curve_{hellaswag,mmlu}.pdf`.

| cut     | `ρ_input` | `ρ_channel` | `B` | reads | surface rel_err | HellaSwag       | Δ dense | MMLU            | Δ dense |
| ------- | ------------ | -------------- | ----- | ----- | --------------- | --------------- | -------- | --------------- | -------- |
| −63.3% | 0.2500       | 0.20003        | 1229  | 512   | 0.3557          | **76.61** | −1.95   | **79.45** | −1.46   |
| −68.3% | 0.2400       | 0.15670        | 963   | 492   | 0.3984          | **75.78** | −2.78   | **78.98** | −1.93   |
| −73.3% | 0.1875       | 0.14170        | 871   | 384   | 0.4442          | **74.63** | −3.93   | **77.94** | −2.97   |
| −75.0% | 0.1875       | 0.12500        | 768   | 384   | 0.4613          | **74.08** | −4.48   | **77.77** | −3.14   |
| −77.5% | 0.1875       | 0.10000        | 614   | 384   | 0.4923          | **73.33** | −5.23   | **76.81** | −4.10   |
| −80.0% | 0.1575       | 0.09500        | 584   | 323   | 0.5246          | **72.55** | −6.01   | **76.11** | −4.80   |

**Complete: 12/12 evals.** Both metrics monotonic across the whole range.

**No cliff.** The curve is monotonic and gently sloped across the whole range —
1.95pt down at −63.3%, 6.01pt at −80.0%, i.e. roughly **0.24pt per pp of extra cut**.
For contrast, the only offline method that reaches comparable depth (Level-1
`pivchol`) is at **44.15** by −87.5%, and activate-fewer-experts is at **26.2**. At
−80.0% this method is still within 6pt of dense while using a fifth of the FFN
parameters per token.

**The ladder's best validation yet.** Regressing the six measured accuracies on their
surface rel_err:

| metric    | slope (pt per unit rel_err)              | R²             |
| --------- | ---------------------------------------- | --------------- |
| HellaSwag | **−24.5** (pre-registered −26.4) | **0.997** |
| MMLU      | **−20.3**                         | **0.979** |

So the 4-minute offline instrument predicts *two* entire 6-point accuracy sweeps,
spanning 17pp of budget, to within ~7% of the pre-registered HellaSwag slope. The MMLU
slope is **shallower (−20.3)**, which is the quantitative form of the "MMLU degrades
more gracefully" observation below — one unit of block-output error buys less MMLU
damage than HellaSwag damage. A single rel_err surface therefore serves both metrics,
with a per-metric slope.

That is also what makes the split-solve non-result above interpretable rather than
confusing: rel_err is an excellent predictor **along the budget axis** and an
unreliable one **between iso-cost splits**, where the true surface is flat and the
measured differences are noise.

**MMLU degrades more gracefully at every single budget**, and the gap *widens* with
depth — MMLU gives up 0.49pt less than HellaSwag at −63.3%, rising to 1.21–1.34pt less
from −75.0% on. At −80.0% the method is **−4.80 on MMLU versus −6.01 on HellaSwag**.
This confirms across six budgets what the doc previously noted at one: the hypothesis
that per-token activation matters *more* on knowledge-heavy tasks is **not supported**;
the reverse holds, consistently and with a mechanism (the shallower rel_err slope).

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

### Measured end-to-end — and the offline gain does *not* hold across budgets

The table above is offline. HellaSwag + full MMLU at `ρ_input=0.25`, everything else
at best practice (`bits=16`, `use_gate=true`, `lam=1.0`, `k_min=0`), so `input_alloc`
is the only thing that moves within a row:

| `ρ_channel` | cut | `uniform` HS / MMLU | **`router`** HS / MMLU | Δ HS | Δ MMLU | offline predicted Δ |
|---|---|---|---|---|---|---|
| 0.10 | −73.3% | 74.06 / 77.20 | **74.64 / 77.67** | **+0.58** | **+0.47** | +0.66 |
| 0.15 | −68.3% | 76.47 / 78.63 | *not measured* ¤ | | | +0.60 |
| 0.20 | −63.3% | 76.72 / 79.10 | **76.61 / 79.45** | **−0.11** | **+0.35** | +0.53 |

¤ The `ρ_channel=0.15` `router` arm was launched and then cancelled, so the trend rests
on its **two endpoints only**. Whether +0.58 → −0.11 decays smoothly or has a knee
somewhere in between is therefore open. Configs are ready if it matters:
`configs/eval/qwen3_30b_a3b_probe_alloc_rtr_k25_r15_*.yaml`.

The −63.3% pair is exact-iso: `reuse_k25_r20` (uniform) and `bp_cut633` (router) both
run `ρ_input=0.25`, 512 reads/expert and `B=1229` — only the allocation differs.

**The correction: `router`'s gain is budget-dependent, and on HellaSwag it is gone by
−63.3%.** The offline screen predicted a nearly flat +0.53…+0.66pt across the whole
`ρ_channel` range; measured, HellaSwag goes **+0.58 → −0.11** across the two budgets
where both arms exist.
MMLU stays positive at both ends (+0.47, +0.35) but also shrinks. So the honest reading
is:

- **`router` is worth ~+0.5pt where it matters (tight budgets) and nothing where it
  doesn't.** It is still the right default — free, never measurably harmful, and the
  regime this method targets *is* the tight end — but "+0.53…+0.66pt" should not be
  quoted as a budget-independent constant. The doc's headline +0.58pt is specifically
  the −73.3% number.
- **Mechanism, and it is the same one as everywhere else in this doc:** the value of
  reallocating reads scales with how binding the read budget is. At `ρ_channel=0.20`
  there is enough channel budget that a slightly worse ranking costs little; at 0.10
  every mis-ranked channel is one that had to displace a good one. The extreme of this
  trend is [`input_only`](#input_only-stop-scoring-just-compute-on-the-sparse-input),
  where the reads carry the *values* and the same allocation is worth **+8.9 to
  +21.8pt**, also growing as the budget tightens.
- **Another offline-screen miss, and the mildest so far**: the screen got the sign right
  at the tight end and the magnitude wrong at the loose end. Note how *well* it does at
  the budget it was anchored on — `router`'s residual at `ρ_channel=0.10` is
  **−0.08pt HS / −0.04pt MMLU**, essentially exact. Compare its outright method
  mis-ranking for `input_only` and its −0.46pt miss on `tau`+`router`. Pattern: the
  ladder is trustworthy for "which of these selectors is better *at the budget I care
  about*" and unreliable for extrapolating a delta *across* budgets — the same
  along-vs-across distinction that made the
  [split solve](#how-to-split-a-fixed-budget-between-the-two-sparsities) a wash.

**`router2` and `colnorm` remain offline-only, by decision.** Evals were started and
then cancelled: both are predicted to land *below* `router`, neither is a candidate for
best practice, and ~26 GPU-hours to confirm a term nobody would ship is not worth it
when the ladder is well-calibrated at exactly this budget (`router`'s residual here is
−0.08pt). Their verdicts therefore rest on the offline screen alone — noted as a known
soft spot rather than a settled result:

| term | rel_err | offline-predicted HS / MMLU | status |
|---|---|---|---|
| `uniform` (anchor) | 0.4434 | — | **74.06 / 77.20** measured |
| **`router`** | 0.4183 | 74.7 / 77.7 | **74.64 / 77.67** measured (resid −0.08 / −0.04) |
| `router2` | 0.4272 | 74.5 / 77.5 | offline only — **not evaluated** |
| `colnorm` | 0.4429 | 74.1 / 77.2 | offline only — **not evaluated** |

If the question is ever revisited, `router2` is the one worth the GPU time: the offline
screen puts it between `uniform` and `router`, but sharper `g_e` weighting is the
mechanism Level-1 found **better** for the *channel* budget (`g²·σ`), so a surprise
there would sharpen the "coordinate and channel budgets respond differently to `g_e`"
claim rather than merely confirm it. The configs are generated and ready —
`configs/eval/qwen3_30b_a3b_probe_alloc_{router2,colnorm}_k25_r10_*.yaml` from
`scripts/gen_probe_alloc_configs.py`, runner `scripts/run_probe_alloc_study.sh`.

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

| target cut        | best uniform                         | per-layer optimum | gain               | ≈Δpt (predicted) |
| ----------------- | ------------------------------------ | ----------------- | ------------------ | ------------------ |
| **−73.3%** | ρ_in=0.1875, ρ_ch=0.125 → 0.4613  | **0.4423**  | **−0.0190** | **+0.50**    |
| −70.8%           | ρ_in=0.25, ρ_ch=0.125 → 0.4248    | 0.4168            | −0.0080           | +0.21              |
| −68.3%           | ρ_in=0.1875, ρ_ch=0.1875 → 0.4032 | 0.3935            | −0.0097           | +0.26              |

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

| config                                   | used   | mean rel_err     | **slope-weighted** rel_err |
| ---------------------------------------- | ------ | ---------------- | -------------------------------- |
| schedule, slope-weighted                 | 0.2663 | 0.4495           | **0.4382**                 |
| schedule, unweighted                     | 0.2663 | **0.4423** | 0.4458                           |
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
needs more budget, the edges less" rather than as 48 independently meaningfu choices.

## Current best practice for input_sparse

Everything above, reduced to the settings to actually deploy. This is the
configuration to carry into evaluations at other levels/budgets — copy the block,
change **only** the two sparsities.

### The config block

```yaml
prune_kwargs:
  prune_ratio: 0.875            # = 1 - rho_channel  (MUST match rho_channel below)
  dynamic_alloc:
    enabled: true
    criterion: "sparse_probe"   # the input_sparse method (historical code name)
    k_min: 0
    probe:
      bits: 16                  # serving precision -> probe IS the served weight
      group: 128                # ignored at bits>=16; harmless to leave
      rho_input: 0.1875         # SCORING budget
      rho_channel: 0.125        # COMPUTE budget; = 1 - prune_ratio
      use_gate: true            # score BOTH up and gate
      lam: 1.0                  # no candidate cascade
      input_alloc: "router"     # split coord reads across experts by g_e*|x_i|
```

### Why each knob is set that way

| knob                       | set to               | why — and what it costs to get wrong                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bits`                   | **16**         | Probe aliases the served weight →**zero extra storage** (measured 0.00 MB). A separate 3-bit copy is dominated on *both* axes: +13% storage and worse rel_err. **The code default is 3, not 16** — omitting `bits` silently gives you the dominated variant.                                                                                                                             |
| `use_gate`               | **true**       | Scores`g_e·\|SiLU(g̃ate)⊙ũp\|`, i.e. both branches. Dropping to up-only is `oracle_up`: **−3.34pt** at comparable depth. Replacing `SiLU(gate)` with `\|gate\|` → rel_err 0.647 vs 0.369. The gate is load-bearing here (the *opposite* of the low-rank family's verdict).                                                                                                               |
| `input_alloc`            | **`router`** | Pools coordinate reads across the token's K experts by`g_e·\|x_i\|` instead of an equal set per expert. **+0.58pt HellaSwag / +0.47pt MMLU at `ρ_channel=0.10`**, identical cost and reads/token; won all 16 layer×budget cells offline. **Budget-dependent — measured −0.11pt HS / +0.35pt MMLU by `ρ_channel=0.20`**, so do not treat +0.58 as a constant ([why](#measured-end-to-end--and-the-offline-gain-does-not-hold-across-budgets)). Still the right default: free, never measurably harmful, and worth the most at the tight budgets this method targets. Do **not** use `router2` (`g²` overshoots) or `colnorm` (a wash, column-norm CV = 0.022) — both **offline-only verdicts**, never evaluated end-to-end. |
| `lam`                    | **1.0**        | No relaxed-candidate cascade: the extra exact reads buy less than spending them on`ρ_input`.                                                                                                                                                                                                                                                                                                            |
| `k_min`                  | **0**          | No per-expert floor; the pooled top-`B` is global by design. (Note the code default is 16.)                                                                                                                                                                                                                                                                                                              |
| `prune_ratio`            | `1 − rho_channel` | Redundant with`rho_channel` and **not** cross-checked at load — a mismatch silently changes `B`. Verify against the in-run `USED PARAMS=` line.                                                                                                                                                                                                                                               |
| schedule / per-layer table | **do not use** | Cross-layer budget reallocation is a closed negative result: unweighted −0.16pt, slope-weighted −3.29pt, and the best*uniform* grid point beats both. One global pair for all 48 layers.                                                                                                                                                                                                               |

### Choosing the two sparsities at a new budget

Cost is `used = ρ_channel + 2·ρ_input/3`, so a budget `C` admits many splits. Solve
it — don't hand-pick `ρ_input` (the hand-picked 0.25 was never optimal at deep
budgets). No GPU, seconds:

```bash
python scripts/probe_split_solve.py --budgets <C>
```

Solved optima on the cached 8192-token surface:

| target cut        | budget`C` | `ρ_input`\*   | `ρ_channel`\* | `B` | reads | HellaSwag       | MMLU            |
| ----------------- | ----------- | ---------------- | ---------------- | ----- | ----- | --------------- | --------------- |
| −63.3%           | 0.3667      | 0.2500           | 0.2000           | 1229  | 512   | **76.61** | **79.45** |
| −68.3%           | 0.3167      | 0.2400           | 0.1567           | 963   | 492   | **75.78** | **78.98** |
| −73.3%           | 0.2667      | 0.1875           | 0.1417           | 871   | 384   | **74.63** | **77.94** |
| **−75.0%** | 0.2500      | **0.1875** | **0.1250** | 768   | 384   | **74.08** | **77.77** |
| −77.5%           | 0.2250      | 0.1875           | 0.1000           | 614   | 384   | **73.33** | **76.81** |
| −80.0%           | 0.2000      | 0.1575           | 0.0950           | 584   | 323   | **72.55** | **76.11** |

**All six are measured at their own solved optimum, on both metrics** (12/12 evals).
Reproduce: `scripts/run_probe_curve_sweep.sh` → `scripts/probe_curve_collect.sh` →
`scripts/probe_curve_plot.py`.

**`ρ_input`\* = 0.1875 at every budget ≤ −73.3%** — a roughly 50/50 scoring/compute
split. How much *solving* it is worth is **depth-dependent**, from five iso-cost tests:

| budget  | Δ for the solved split                    | verdict         |
| ------- | ------------------------------------------ | --------------- |
| −73.3% | **−0.01** HS / **+0.27** MMLU | wash            |
| −75.0% | **+0.28** HS / **+0.44** MMLU  | wash / marginal |
| −80.0% | **+1.14** HS (row 7e vs row 0d)      | **real**  |

Hand-picking `ρ_input=0.25` is therefore harmless down to about −75%, and costs over a
point by −80%, where there is no slack left to absorb budget spent on the wrong axis.
The solve takes seconds offline, so just always run it. Caveat: 0.1875 is the best
*grid point* (neighbours 0.125 / 0.25), not a continuum optimum.

### The operating points

The curve above is the frontier; these are the points worth quoting:

|                                    | `ρ_input` | `ρ_channel` | cut               | HellaSwag       | MMLU            |
| ---------------------------------- | ------------ | -------------- | ----------------- | --------------- | --------------- |
| **best accuracy**            | 0.25         | 0.10           | −73.3%           | **74.64** | 77.67           |
| **best MMLU at that budget** | 0.1875       | 0.1417         | −73.3%           | 74.63           | **77.94** |
| **deepest within ~6pt**      | 0.1575       | 0.095          | **−80.0%** | 72.55           | 76.11           |

The two −73.3% rows are statistically tied on HellaSwag; prefer the solved split, which
is +0.27 on MMLU at the same cost. Going from −73.3% to −80.0% costs 2.08pt HellaSwag /
1.83pt MMLU for 6.7pp of extra cut (≈0.31 / 0.27pt per pp).

### Running an eval — the operational parts

```bash
# 4 GPUs per job. MMLU 5-shot OOMs on 2x40GB (a 30B shard + 5-shot contexts);
# it died once at 99% after ~10h, so do not economize here.
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python src/train/merge_slim_eval.py \
    --config configs/eval/qwen3_30b_a3b_probe_router_split_opt_hellaswag.yaml
```

**Always confirm the realized budget from the log** rather than trusting the YAML —
the line to look for, and what a correct −75.0% run prints:

```
[input_sparse] rho_input=0.1875 rho_channel=0.125 (scoring 0.1250 + compute 0.1250)
  -> USED PARAMS=0.2500, cut 75.0%; ... extra proxy storage = 0.0% of expert weights
[DynamicAlloc] Installing: ... prune_ratio=0.875, B=768 (of K*I=6144)
[DynamicAlloc] ✅ Installed dynamic forward on 48 MoE blocks
```

Check `USED PARAMS`, that `extra proxy storage` is **0.0%**, and that `B` is what you
intended. Reference configs: `configs/eval/qwen3_30b_a3b_probe_router_split_opt_*.yaml`
(−75.0%) and `..._probe_router_k25_r10_*.yaml` (−73.3%).

### Screening a change before spending an eval

Use the **fixed-budget** ladder slope **−26.4 pt per unit rel_err** (`R²`=0.985), not
the budget-ladder −6.9. It is validated to ±0.26pt for changes that alter *which
channels a layer selects*, and **invalid** for moves that shift budget *between*
layers (missed by 1.0pt). Run `scripts/probe_output_error.py` first — 4 min on 1 GPU
versus ~10h for a real eval.

---

## Unstructured sparsity: buy *entries*, not coordinates

> The question: at a **total** budget of 20% of what the dense model activates —
> scorer reads *plus* selected-channel compute — how close can a scorer get to what
> full-width `gate+up` achieves when selecting the same 12.5% of channels?

Everything above spends the scoring budget on whole **coordinates**: the token's
top-`ρ_input` columns of `up`/`gate`, read for all `I` channels. That is
*column-structured* sparsity. But the exact contribution of entry `(j, i)` to
channel `j`'s pre-activation is `W_ji·x_i`, so at a fixed number of reads the
greedy-optimal set is the top of the **product** `|W_ji|·|x_i|`. A column rule sees
only the `|x_i|` factor; a static weight mask sees only the `|W_ji|` factor. Neither
is the argmax of the product. Lifting the structure constraint is what this section
measures.

**The budget.** With `ρ_channel = 0.125` (the channel budget `oracle_mag_noW` uses)
and the same used-parameter frame as everywhere else,

```
used = ρ_channel + 2·density/3 = 0.125 + 2·0.1125/3 = 0.200      → −80.0%
```

so the scorer may read **11.25% of the entries** of `up` and of `gate` per token.
Note this is *deeper* than any previously measured row: the old headline was −73.3%.
The reference to beat is `oracle_mag_noW` — 77.11 HellaSwag / 79.44 MMLU at
`used = 0.792`.

### One family, two familiar endpoints

`src/dynamic_active_param/weight_sparse.py` implements an `L`-band **staircase**:
per token, coordinates are ranked by `|x_i|` and split into bands; band `l` spans a
fraction `col_frac[l]` of them and is read for the `row_frac[l]` fraction of
channels whose `|W_ji|` is largest **in that column**. Read density per branch is
`Σ_l col_frac[l]·row_frac[l]`. Both schemes being compared are points of this one
family, which is what makes the comparison a within-family sweep rather than an
apples-to-oranges one:

| spec           | what it is                                                                     |
| -------------- | ------------------------------------------------------------------------------ |
| `0.1125x1.0` | **`input_sparse` at `ρ_input=0.1125`** — few columns, all channels |
| `1.0x0.1125` | a**static per-column weight mask** — all columns, few channels          |
| `0.25x0.45`  | a rectangle in between: 25% of coordinates × 45% of channels                  |
| `tau[…]`    | the product rule itself (below)                                                |

A unit test pins the first identity exactly (`weight_sparse` with `((p,1.0),)`
reproduces `sparse_probe` at `rho_input=p` bit-for-bit), and another pins density 1
to `oracle_mag_noW`.

**No mask is stored.** The kept channels of column `i` at level `l` are those above a
per-column magnitude threshold `θ_i[l]`, computed offline from the served weights —
`L·H` floats per expert per branch (0.1–0.7% of the FFN). The mask is *derived* from
the weights, not stored. What unstructured sparsity does cost is positional
metadata, and that is priced honestly [below](#what-unstructured-sparsity-actually-costs).

### The threshold rule (`tau`) — the method that wins

Fixed band widths are still a guess. The budget-exact version of the product rule
reads column `i` down to

```
level(i) = min { l : θ_i[l]·|x_i| ≥ τ }        (unread if no level qualifies)
```

with **one `τ` per token**, bisected until the realized read count meets that token's
budget. The cost is then exact while the *shape* floats: a token whose `|x|` is
concentrated reads deep into a few coordinates, a flat token reads shallowly into
many. This is the same asymmetry the doc already measured — floating the **input**
criterion helps, floating the **channel** budget does not — and it is worth far more
here (+0.53pt) than in the column-only setting (+0.22pt).

Two implementation facts, both load-bearing:

- **The `τ` search is nearly free.** Pricing every column exactly needs a
  `(T·K, L, H)` tensor — 840 MB per branch per layer at eval shapes, 16 passes over
  it, measured **54× the cost of a full-width `gate+up`**. But `θ_i[l]` is a quantile
  of column `i`'s weight magnitudes and this model's per-column weight scales barely
  vary (column-norm CV **0.022**), so the price ordering collapses onto `|x_i|` — the
  order that is *already sorted once per token*. The level of a coordinate becomes a
  function of its rank, the search becomes `L` searchsorteds on that shared order,
  and the pass drops to **0.8×** the reference. Only *which* level a column lands in
  is approximated; the read **count** stays exact. Measured cost of the
  approximation: **0.002 rel_err (≈0.05pt)** — `alloc_mode="taux"` keeps the exact
  pricing for this comparison.
- **The level loop must be one `bmm`.** A python loop over levels is 128 experts × 2
  branches × `L` × ~6 kernels ≈ 12k launches per block; stacking the masked weights
  into `(L,I,H)` and the banded inputs into `(L,T,H)` for a single `bmm` cut the
  8-level scorer from **359 ms → 140 ms** per block (`scripts/wsparse_bench.py`,
  E=128/I=768/H=2048/T=1600, one L4).

### Measured: iso-cost comparison

Block-output `rel_err` at `ρ_channel = 0.125` (`used = 0.200`), layer-averaged over
L6/L22/L38/L46 × 4096 C4 tokens. Every scorer row reads the **same** 0.1125 of each
branch. Δpt uses the validated −26.4 pt/unit fixed-budget slope.

The instrument is the doc's own: run on the existing `input_sparse` rows it
reproduces the published table to four decimals (`ρ_input=0.25`: uniform
0.4434/0.4099, router 0.4183/0.3860 at `ρ_channel` 0.10/0.125), so the two measured
HellaSwag anchors at those points transfer.

| #  | read set                                                          | density | `rel_err`      | Δpt vs column-only |
| -- | ----------------------------------------------------------------- | ------- | ---------------- | ------------------- |
| — | `oracle_mag_noW` — full `gate+up` (`used` **0.792**) | 1.0 ×2 | **0.3272** | +5.01               |
| 1  | **`tau` + `router`, exact pricing (`taux`)**          | 0.1122  | **0.4027** | **+3.02**     |
| 2  | **`tau` + `router`, shipped (band edges)**              | 0.1121  | **0.4044** | **+2.98**     |
| 3  | exact product rule`prod` (ceiling, 1024 tok)                    | 0.1125  | 0.4132           | +2.74               |
| 4  | `tau`, exact pricing, uniform                                   | 0.1124  | 0.4147           | +2.70               |
| 5  | `tau`, shipped, uniform                                         | 0.1123  | 0.4168           | +2.65               |
| 6  | 6-band rank staircase (`geo:6`)                                 | 0.1125  | 0.4369           | +2.12               |
| 7  | rectangle`0.25x0.45`                                            | 0.1125  | 0.4380           | +2.09               |
| 8  | rectangle`0.225x0.5`                                            | 0.1125  | 0.4425           | +1.97               |
| 9  | rectangle`0.5x0.225`                                            | 0.1125  | 0.4585           | +1.55               |
| 10 | **`input_sparse` + `router` (the fair control)**        | 0.1125  | **0.4668** | +1.33               |
| 11 | static global Wanda mask `                                        | W_ji    | ·σ_i`          | 0.1141              |
| 12 | **`input_sparse`, uniform (baseline row)**                | 0.1125  | **0.5171** | —                  |
| 13 | static per-column mask`1.0x0.1125`                              | 0.1125  | 0.5376           | −0.54              |

**Note row 1 against the previous best.** `input_sparse` + `router` at `used=0.2667`
measures **0.4183** and scored 74.64 HellaSwag. Row 1 is **0.4027 at `used=0.200`** —
a *lower* error at **6.7 points less budget**. Unstructured sparsity does not merely
soften the tightening; at this depth it moves the frontier.

**Reading the table.**

- **Unstructured beats column-structured by ~1.7pt at matched reads**, and by
  +3.0pt against the uniform baseline. The whole gain comes from spending reads on
  entries rather than coordinates.
- **The interior of the family wins; both endpoints lose.** Columns-only (row 12) and
  the static per-column mask (row 13) are the two worst scorers in the table — and
  the static mask is *worse than* columns-only. "Unstructured" is not the win by
  itself; **per-token adaptivity on both axes** is.
- **Static weight masks are dead**, restated twice: the per-column-balanced mask
  (row 13, −0.54pt) and the global Wanda mask (row 11, +0.33pt) both sit within
  0.9pt of columns-only despite having full freedom over which entries to keep. This
  is the doc's [unifying negative result](#unifying-negative-result) again — a mask
  fixed offline can only encode the *average* token.
- **The discretized ladder has converged onto the exact rule.** `prod` (row 3,
  0.4132) is the true per-token top-`N` of `|W_ji·x_i|` and is the ceiling of the
  family *without* cross-expert allocation; the 8-level ladder at the same setting
  (row 4, 0.4147) is within **0.0015** of it. Row 1 goes *past* `prod` only because
  it adds the `router` split, which `prod` does not have. Nothing is left on the
  discretization axis.
- **The cheap approximation costs 0.05pt** (rows 1→2, 4→5) for a **68× cheaper**
  threshold search. This is the version in the eval configs.
- **`router` allocation composes**: +0.0124 rel_err on top of `tau`
  (0.4168 → 0.4044, **+0.33pt**) and +0.0503 on top of columns-only
  (0.5171 → 0.4668, **+1.33pt**). The doc measured +0.53–0.66pt at `ρ_input=0.25`;
  its value grows as the budget tightens, as predicted but not previously measured
  this deep.

### Where to put the 20%

The same total budget can be split differently between scoring and compute. All rows
`used = 0.200`, `tau` + `router`:

| `ρ_channel` (compute) | density (scoring) | `rel_err`      |
| ------------------------ | ----------------- | ---------------- |
| 0.100                    | 0.150             | 0.4137           |
| **0.125**          | **0.1125**  | **0.4027** |
| 0.150                    | 0.075             | 0.4117           |

**`ρ_channel = 0.125` with an 11.25% read density is the optimum** — the point the
question named, confirmed rather than assumed. The optimum is *method-dependent*:
column-only prefers `ρ_channel = 0.100` (0.5077 vs 0.5171), because its scoring is
inefficient enough that buying more scoring is a worse deal than buying channels.
Making the scorer better moves budget *back* into the scorer.

### Negative results from this round

| tried                                                                                                                   | result                                                                                                     | why                                                                                                                                                                                                                                                                                                             |
| ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Mean-fix**: precompute `W·μ` exactly (free, `I` floats/expert/branch), score the centered input `x−μ` | **0.4165 vs 0.4147** — no gain, marginally worse. On column-only, 0.5272 vs 0.5171 — clearly worse | `μ` is a *rank-1 common mode*; removing it also removes the part of `                                                                                                                                                                                                                                      |
| **Up-only** proxy at 2× density (`use_gate=false`, density 0.225, same `used=0.200`)                         | **0.5564 vs 0.4147** — much worse                                                                   | The`SiLU(gate)` factor is load-bearing (as `oracle_up`'s 71.30 already said). Contrast the low-rank family, where dropping the gate to buy rank *won*: with reads-not-rank as the axis, that trade reverses                                                                                               |
| **Ladder resolution** — 8 levels ratio 0.7 vs 6 and 4 levels ratio 0.5                                           | 0.4168 / 0.4194 / 0.4201                                                                                   | Once band*edges* float, resolution barely matters: **4 levels is within 0.09pt of 8** at half the wall-clock. (Under *fixed* edges it mattered 4× more — on L46 alone, a 4-level ratio-0.25 ladder lost 1.2pt to the 8-level ratio-0.7 one, and ratio 0.8 × 12 levels was the best of five shapes) |
| **Dropping the full-column level** (max `row_frac` 0.7 instead of 1.0)                                          | 0.4023 vs 0.4027 — a wash                                                                                 | Nothing needs a whole column read; the deepest band can be capped                                                                                                                                                                                                                                               |
| **`router` allocation on top of `tau`**                                                                       | offline +0.33pt,**measured −0.46 ± 0.62pt** (74.41 vs 74.87)                                       | The one offline screen that did*not* transfer. Floating the read set per token already does what reallocating reads across experts was doing                                                                                                                                                                  |
| **Channel blocks** of 4 / 8 / 16 / 32 (to cheapen the metadata)                                                   | 0.4486 / 0.4615 / 0.4671 / 0.4691 vs 0.4044                                                                | Half the gain is per-*channel* freedom. See [what it costs](#what-unstructured-sparsity-actually-costs)                                                                                                                                                                                                        |

### What unstructured sparsity actually costs

The used-parameter frame counts *reads*, which is the axis the question named. But
unlike `input_sparse` — whose read set is one contiguous column list and needs **zero**
metadata — an entry-level read set has to be locatable without reading the weights it
skips. Priced as a fraction of the expert FFN (3 matrices):

| scheme                                     | positional metadata    | thresholds | total           | Δpt vs columns-only   |
| ------------------------------------------ | ---------------------- | ---------- | --------------- | ---------------------- |
| `input_sparse` (columns)                 | **0**            | 0          | **0**     | +1.33 (with`router`) |
| rectangle`0.25x0.45` (1 level, 2 states) | 1 bit/weight → 4.17%  | 0.09%      | **4.3%**  | +2.09                  |
| `tau`, 3-level ladder (4 states)         | 2 bit/weight → 8.33%  | 0.26%      | **8.6%**  | ~+2.6                  |
| `tau`, 4-level ladder (5 states)         | 3 bit/weight → 12.50% | 0.35%      | **12.9%** | +2.56                  |
| `tau`, 8-level ladder (9 states)         | 4 bit/weight → 16.67% | 0.69%      | **17.4%** | **+2.98**        |

The obvious way out is to make the read set **semi**-structured: select channels in
blocks of `r` consecutive ones, so the level map costs `4/r` bits per weight. That is
now measured, and it does not work — `tau`+`router`, density 0.1125,
`ρ_channel=0.125`, blocks scored by `max |W|` within the block:

| channel block`r`               | metadata | `rel_err`      | Δpt vs columns-only |
| -------------------------------- | -------- | ---------------- | -------------------- |
| **1** (fully unstructured) | 17.4%    | **0.4044** | **+2.98**      |
| 4                                | 5.1%     | 0.4486           | +1.81                |
| 8                                | 2.8%     | 0.4615           | +1.47                |
| 16                               | 1.7%     | 0.4671           | +1.32                |
| 32                               | 1.1%     | 0.4691           | +1.27                |

**Half the gain is in per-*channel* freedom, and it is gone by `r=8`**: blocks of 8
land at 0.4615 against the *zero-metadata* column control's 0.4668 — **+0.14pt for
2.8% of extra storage**, which is not a deal. By `r=16` the family has converged back
onto column sparsity (0.467 ≈ 0.4668), which makes sense: coarse blocks reinstate
"read the column or don't".

So the honest headline is two-sided, and the block sweep closes the escape hatch:

- Under the metric this study was set to optimize — **activated parameters**, scorer
  reads included — unstructured sparsity wins by **+1.7pt over the iso-read column
  control** and +3.0pt over the uniform baseline.
- It buys that with **4–17% of extra storage** for positional metadata, and that cost
  is **not compressible**: every coarsening that would cheapen it also removes the
  gain.
- If storage is charged, the frontier is: 4.3% for +2.09pt (the 1-bit rectangle), or
  17.4% for +2.98pt. `input_sparse` remains the only zero-overhead option.

### Measured: HellaSwag at −80.0%

Predictions were pre-registered from the −26.4 slope off the measured anchor
(`input_sparse`+`router`, `rel_err` 0.4183 → 74.64). All four rows read the same
0.1125 of each branch, `ρ_channel = 0.125`, `used = 0.200`; stderr ≈0.44pt.

| config              | read set                                | `rel_err` | predicted | **measured** | resid            |
| ------------------- | --------------------------------------- | ----------- | --------- | ------------------ | ---------------- |
| `wsp_tau`         | `tau` 8-level, uniform                | 0.4168      | 74.7      | **74.87**    | +0.19            |
| `wsp_rect`        | rectangle`0.25x0.45`                  | 0.4380      | 74.1      | **74.47**    | +0.35            |
| `wsp_tau_router`  | `tau` 8-level + `router`            | 0.4044      | 75.0      | **74.41**    | −0.60           |
| `probe_col_r1125` | `input_sparse` + `router` (control) | 0.4668      | 73.4      | **71.41**    | **−1.95** |

**The result: +3.46pt for unstructured reads at identical cost** (74.87 vs 71.41,
5.5σ on the difference). That is *twice* the +1.7pt the ladder predicted, because the
ladder under-estimated how badly a column-only scorer degrades once it can afford
only 11.25% of the coordinates. Against the whole leaderboard: **74.87 at −80.0%**
beats the previous best of 74.64 at −73.3%, so this is 6.7 points of extra cut for
free — the first row in this doc to move the frontier rather than trade along it.

Three things the offline screen got wrong or could not see:

- **`router` allocation does not compose with the threshold mode.** Predicted +0.33pt,
  measured **−0.46 ± 0.62pt** (74.41 vs 74.87) — no gain, and possibly a small loss.
  The same allocation *did* transfer for column-only scoring (+0.58 measured against
  +0.66 predicted, rows 6→7). The reading: `router` reallocates *coordinate* reads
  across a token's experts, which helps when the read set is otherwise rigid; once
  `tau` is already floating the read set per token on both axes, there is nothing left
  for it to fix, and forcing the per-slot budgets apart costs a little. **Use `tau`
  with uniform allocation.**
- **The one-level rectangle is statistically tied with the full ladder** — 74.47 vs
  74.87, Δ = 0.40 ± 0.61pt. It is the better engineering choice by a wide margin:
  **1 bit/weight of metadata instead of 4** (4.3% vs 17.4% of the expert FFN) and a
  scoring pass of 6.4× a full `gate+up` instead of 14.5×. The elaborate part of the
  method — per-token thresholds, 8 levels, the bisection — buys ≤0.4pt over "read the
  top 25% of coordinates for the top 45% of channels".
- **The ladder's scope needs one more caveat.** It landed within 0.6pt on all three
  *unstructured* rows but missed the deep column-only row by 1.95pt. Its validated
  scope was "changes to which channels a layer selects"; a change to the read
  *pattern* is still fine, but a large move along `ρ_input` at fixed pattern is not —
  and the error is in the safe direction here (it under-sold the winner's margin).

**MMLU at −80.0%, measured:** `wsp_rect` **77.62**, `wsp_tau_router` **77.82**, and the
iso-cost column-only control `probe_col_r1125` **75.82** — so the unstructured gain is
**+1.80 to +2.00pt on MMLU** at identical reads, against +3.46pt on HellaSwag. Smaller
than on HellaSwag, consistent with MMLU's shallower rel_err slope (−20.3 vs −24.5)
everywhere else in this doc. `wsp_tau`'s own MMLU (the best HellaSwag row) **OOM'd on
2026-08-14** when it was co-scheduled onto a GPU already holding 17.5 GiB and has been
relaunched on 4 dedicated GPUs; until it lands, `wsp_tau_router` (77.82) is the best
measured MMLU here even though it is the *worse* HellaSwag arm.

**The cost claim is verified on the model, not just asserted.** The threshold mode
floats its read set per token, so `weight_sparse` counts what it actually reads and
reports it at the end of the eval. On Qwen3-30B-A3B over a HellaSwag stream (48 MoE
layers, 4 GPUs): **realized 0.1114 of each branch per token against a 0.1125 target →
used params 0.1993, a −80.1% cut** — i.e. the budget is met with a hair to spare,
never exceeded (the ladder snaps conservatively). Any future row here should quote
the realized number, not the configured one.

Wall-clock, for planning: the scoring pass costs 1.6× a full-width `gate+up` for
`input_sparse`, 6.4× for the 1-level rectangle and 14.5× for the 8-level `tau`
(one L4, `scripts/wsparse_bench.py`) — so a HellaSwag eval goes from ~3.3 h to
~5 h and MMLU from ~9.5 h to ~14 h. The 4-level ladder halves the overhead for
0.09pt.

---

## `input_only`: stop scoring, just compute on the sparse input

> Every method above spends input sparsity on a **decision** and then re-reads the
> served weights at full width to **compute**. What if the sparse read *is* the
> computation?

**Verdict up front: it does not work, and the reason is quantified.** At identical
used-parameters the one-pass method loses **2.21 / 2.73 / 5.17pt** HellaSwag to
two-pass `input_sparse` at −70 / −75 / −80%, so the exact pass it deletes is worth far
more than the third of the budget it consumed. Two findings survive and are worth more
than the method: **`router` read allocation is worth +8.9 to +21.8pt here** — the
largest single effect in this document — and **the rel_err ladder is badly broken for
any method that perturbs channel *values*** rather than the keep-mask (it missed by up
to 22.9pt and does not even rank the two arms correctly).

### The method

Delete the second pass. Per token:

1. rank the token's input coordinates by `|x_i|`, keep the top `ρ_input`
   (allocated across the token's K experts by `uniform` or `router`, the same
   allocator `input_sparse` uses);
2. run `gate`/`up` **on those coordinates only**, and treat the result as the FFN's
   actual intermediate — `h̃_e = SiLU(W_gate^(e) x_sp) ⊙ (W_up^(e) x_sp)`;
3. keep the pooled top-`B = ρ_channel·K·I` of `g_e·|h̃_{e,j}|`;
4. run `down` on the kept channels only.

There is no proxy and no scorer: **the reads that decide are the reads that
compute.** Step 3 is `oracle_mag_noW` — the best selector in this doc — applied to
the intermediate that is actually being used, so **the selection is free**. The
price is that every kept channel now carries a *value* error, not just the
selection error. That trade is the experiment.

Read literally it is **one rule applied uniformly to all three matrices**: *read
only the largest-magnitude coordinates of your own input*. For `gate`/`up` that
input is `x`; for `down` it is the intermediate, whose top-magnitude coordinates
are exactly the pooled top-B. Implementation: criterion `input_only`
(`src/dynamic_active_param/input_only.py`, forward in `block.py`).

### Accounting: two things disappear at once

Units of one expert `(I,H)` matrix, a dense FFN being 3:

```
gate : all I rows, ρ_input of the H columns   ->  ρ_input
up   : same                                  ->  ρ_input
down : only the kept channels' columns       ->  ρ_channel

used = (2·ρ_input + ρ_channel) / 3
```

| | `oracle_mag_noW` | `input_sparse` (two-pass) | **`input_only`** |
|---|---|---|---|
| cost | `(1+1+ρ_ch)/3` | `ρ_ch + 2ρ_in/3` | **`(2ρ_in + ρ_ch)/3`** |
| floor | **2/3** (gate+up full width) | none | none |
| kept rows billed | once | **twice** (scoring + compute) | once |
| cheap axis | — | `ρ_input` (2/3 vs 1) | **`ρ_channel` (1/3 vs 2/3)** |

Two consequences worth stating plainly:

- **It is a plain read count, not a conservative frame.** `input_sparse`'s
  `used = ρ_ch + 2ρ_in/3` deliberately bills the overlap between its two passes
  twice ([Used-parameter accounting](#used-parameter-accounting)). One-pass has no
  overlap to bill, so nothing here is conservative *or* flattering.
- **The cheap axis inverts.** `∂used/∂ρ_input = 2/3` but `∂used/∂ρ_channel = 1/3`,
  so a unit of `ρ_channel` now costs **half** a unit of `ρ_input`. The standing
  advice "cut `ρ_input` first" — correct for `input_sparse`, where `ρ_channel` cost
  1 — **does not carry over**. Pinned in `test_rho_channel_is_the_cheap_axis`.

Two anchors, both unit-tested: at `ρ_input = 1` the method reproduces
`oracle_mag_noW` bit-for-bit **and** its cost `(2+ρ_ch)/3` equals
`oracle_mag_noW`'s `(1+1+ρ_ch)/3` exactly — the frames agree precisely where the
methods coincide. At `ρ_channel = 1` it is pure input sparsity with no channel
selection, `used = (2ρ_in+1)/3`.

### The split: symmetric is the right answer

Since `used = (2ρ_in + ρ_ch)/3`, setting **`ρ_input = ρ_channel = C`** puts the
used-parameter cut exactly at `1 − C` — a −75% cut simply *is* "read a quarter of
every input". That is both the literal reading of the method and, as measured, the
operating point to use. Solved over an iso-cost grid
(`scripts/input_only_error.py`, 4 layers × 8192 C4 tokens, `router`):

| budget | cut | best split `(ρ_in, ρ_ch)` | rel_err | symmetric | gain for solving |
|---|---|---|---|---|---|
| 0.30 | −70.0% | 0.3500 / 0.2000 | 0.3520 | 0.3545 | **+0.07pt** |
| 0.25 | −75.0% | 0.2750 / 0.2000 | 0.4044 | 0.4108 | **+0.17pt** |
| 0.20 | −80.0% | 0.2250 / 0.1500 | 0.4707 | 0.4764 | **+0.15pt** |

The optimum leans consistently toward *more* scoring than symmetric (fidelity of the
values beats channel count, all three budgets, same sign) — but only by 0.07–0.17pt,
inside this instrument's noise and inside the range where this repo has already
measured iso-cost split changes to be [a wash](#how-to-split-a-fixed-budget-between-the-two-sparsities).
Every eval row below uses the symmetric split.

> ⚠️ **This table is the only claim in the section that still rests on the offline
> instrument, and that instrument is [discredited for this
> method](#the-ladder-broke-here-and-now-we-know-by-how-much) — it under-predicted by
> up to 22.9pt and mis-ranked the two arms.** The split was never measured
> end-to-end. Since the optimum leans toward more `ρ_input` and the instrument
> systematically *under*-penalizes value error, the true optimum plausibly leans
> further that way than 0.07–0.17pt suggests. Read this as "symmetric is a reasonable
> default", not as a solved split. It is moot in practice: the method is dominated at
> every budget, so the split is not worth measuring.

### `router` allocation is worth 15–37× more here — the section's real finding

Measured end-to-end, symmetric split, `uniform` vs `router` at identical cost and
identical reads/token:

| cut | HellaSwag `uniform` | **`router`** | Δ | MMLU `uniform` | **`router`** | Δ | offline predicted |
|---|---|---|---|---|---|---|---|
| −70.0% | 64.33 | **73.18** | **+8.85** | 70.24 | **77.60** | **+7.36** | +3.0 |
| −75.0% | 56.95 | **71.35** | **+14.40** | 62.82 | **76.06** | **+13.24** | +3.5 |
| −80.0% | 45.58 | **67.38** | **+21.80** | 49.52 | **73.97** | **+24.45** | +3.9 |

**This is the largest single effect measured anywhere in this document** — an order of
magnitude past the +0.58pt the same allocation buys the two-pass scorer, and 3–6×
past its own offline prediction. It also *rescues* the method: without it, `uniform`
at −80% is at **45.58**, i.e. no better than Level-1 `pivchol` at −87.5% (44.15).

The mechanism follows from the method. Under `input_sparse` a mis-allocated read only
degrades a *ranking* that the exact pass then recomputes anyway, so the damage is
bounded by one wrong channel choice. Here it degrades the expert's **values**, which
nothing downstream repairs — and those value errors then compound through 48 layers.
Weighting reads by `g_e·|x_i|` spends fidelity where the router already said the
output mass is, and the tokens it starves are the ones whose contribution is scaled
down by a small `g_e` anyway. It remains free: one bisection over a sort the layer
computes regardless.

Two readings worth carrying:

- **`router` allocation composes with rigid read sets and not with adaptive ones.**
  +0.58pt for column scoring, **−0.46pt** for the `tau` threshold ladder (which
  already floats the read set per token), **+8.9 to +21.8pt** here — where the read
  set is rigid *and* the reads carry the values. The value of reallocating reads
  scales with how much the read set determines the *output* rather than the ranking.
- **Its value grows sharply as the budget tightens**, from +8.85 to +21.80pt over
  10pp of cut. The doc previously observed this trend at +0.53→0.66pt; here the same
  trend has ~30× the slope, because at `ρ=0.2` a uniformly-starved expert has almost
  no signal left to contribute.

### Measured

Qwen3-30B-A3B, masking simulation for `down_proj` only (the `gate`/`up` sparsity is
real, not simulated), no fine-tuning. Dense = **78.56** / **80.91**. Symmetric split,
`k_min=0`, all 12 runs rc 0 with the realized `USED PARAMS` and `B` verified in-run.
Configs `configs/eval/qwen3_30b_a3b_inputonly_{uni,rtr}_cut{700,750,800}_*.yaml`,
runner `scripts/run_input_only_sweep.sh`, harvest `scripts/input_only_collect.sh`.

| cut | `ρ_in` = `ρ_ch` | `B` | reads | arm | HellaSwag | Δ dense | MMLU 5-shot | Δ dense |
|---|---|---|---|---|---|---|---|---|
| −70.0% | 0.300 | 1843 | 614 | `uniform` | 64.33 | −14.23 | 70.24 | −10.67 |
| −70.0% | 0.300 | 1843 | 614 | **`router`** | **73.18** | −5.38 | **77.60** | −3.31 |
| −75.0% | 0.250 | 1536 | 512 | `uniform` | 56.95 | −21.61 | 62.82 | −18.09 |
| −75.0% | 0.250 | 1536 | 512 | **`router`** | **71.35** | −7.21 | **76.06** | −4.85 |
| −80.0% | 0.200 | 1229 | 410 | `uniform` | 45.58 | −32.98 | 49.52 | −31.39 |
| −80.0% | 0.200 | 1229 | 410 | **`router`** | **67.38** | −11.18 | **73.97** | −6.94 |

**12/12 complete.** Both metrics monotonic in the budget within each arm. MMLU
degrades more gracefully at every point, as everywhere else in this doc (−3.31 vs
−5.38 at −70%, widening to −6.94 vs −11.18 at −80%).

### Verdict: dominated by the two-pass scorer, and the gap widens

The comparison at identical `used`, against the measured best-practice `input_sparse`
curve:

| cut | **`input_only` + `router`** | `input_sparse` + `router` | Δ HellaSwag | Δ MMLU |
|---|---|---|---|---|
| −70.0% | 73.18 / 77.60 | 75.39 / 78.63 † | **−2.21** | **−1.03** |
| −75.0% | 71.35 / 76.06 | **74.08 / 77.77** | **−2.73** | **−1.71** |
| −80.0% | 67.38 / 73.97 | **72.55 / 76.11** | **−5.17** | **−2.14** |

† interpolated between the measured −68.3% (75.78 / 78.98) and −73.3% (74.63 / 77.94)
points; the other two rows are exact measured matches (doc rows 7b and 7e).

**Deleting the exact pass is a losing trade, and it gets worse with depth.** Read as
budget rather than accuracy: two-pass reaches one-pass's −70% score (73.18) at about
**−78.0%**, so keeping the second pass buys **8pp more cut at equal accuracy** — and
one-pass's −75% and −80% scores fall off the bottom of the measured two-pass curve
entirely. The intuition the offline screen missed: spending a third of the budget on a
decision you then discard *looks* wasteful, but what you buy with it is exact values
on every kept channel, and value errors are the ones that compound across 48 layers.

So `input_only` is a **negative result as a method** — but a clean one, and the two
things it establishes are reusable: the `router` allocation result above, and the
ladder-scope result below.

### The ladder broke here, and now we know by how much

It failed in **two separate ways**, and the first is the more dangerous.

**1. It mis-ranked the two methods.** On the same instrument, in the same pass,
`input_only` measured *lower* block-output error than two-pass `input_sparse` at two of
three budgets:

| cut | `input_only` + `router` | `input_sparse` + `router` | offline said | **evals said** |
|---|---|---|---|---|
| −70.0% | 0.3545 `(0.30/0.30)` | 0.3658 `(0.1875/0.1750)` | one-pass **+0.30pt** | one-pass **−2.21pt** |
| −75.0% | 0.4108 `(0.25/0.25)` | 0.4121 `(0.1875/0.1250)` | one-pass **+0.03pt** | one-pass **−2.73pt** |
| −80.0% | 0.4764 `(0.20/0.20)` | 0.4655 `(0.1575/0.0950)` | one-pass −0.29pt | one-pass **−5.17pt** |

The screen's verdict was "an iso-cost tie that crosses over"; the truth is "dominated
at every budget, gap widening". A screen that gets the *sign* wrong on a cross-method
comparison is worse than no screen, and the instrument reproduces the doc's published
two-pass anchors to four decimals — so this is not a measurement bug, it is the
objective being wrong.

**2. It mis-predicted the levels.** Every prediction missed low, and the misses are
not small:

| arm | cut | rel_err | predicted HS | measured HS | resid |
|---|---|---|---|---|---|
| `router` | −70.0% | 0.3545 | 75.6 | 73.18 | **−2.42** |
| `router` | −75.0% | 0.4108 | 74.1 | 71.35 | **−2.76** |
| `router` | −80.0% | 0.4764 | 72.4 | 67.38 | **−5.00** |
| `uniform` | −70.0% | 0.4698 | 72.6 | 64.33 | **−8.23** |
| `uniform` | −75.0% | 0.5430 | 70.6 | 56.95 | **−13.67** |
| `uniform` | −80.0% | 0.6236 | 68.5 | 45.58 | **−22.92** |

Fitting the *effective* slope per arm against the doc's validated −26.4 pt/unit for
selector changes:

| arm | HellaSwag slope | MMLU slope |
|---|---|---|
| `router` | **−47.9** | −29.8 |
| `uniform` | **−122.2** | −135.2 |
| (validated, selector changes) | −26.4 | −20.3 |

**A unit of single-layer rel_err costs 1.8–4.6× more accuracy when it comes from
corrupted values than when it comes from a mis-selected channel.** And worse, the
instrument does not even *order* the two arms — `uniform` at rel_err **0.4698** scores
**64.33** while `router` at a *higher* **0.4764** scores **67.38**:

| rel_err | arm | cut | HellaSwag |
|---|---|---|---|
| 0.3545 | `router` | −70% | 73.18 |
| 0.4108 | `router` | −75% | 71.35 |
| **0.4698** | `uniform` | −70% | **64.33** |
| **0.4764** | `router` | −80% | **67.38** |
| 0.5430 | `uniform` | −75% | 56.95 |
| 0.6236 | `uniform` | −80% | 45.58 |

**The cause is the same one that closed the cross-layer schedule.** The surface
measures each layer against an **unperturbed** input, deliberately, so layers stay
separable. That is sound when the perturbation is a keep-mask: dropping a channel is a
bounded, self-limiting error that the next layer sees as a slightly-wrong input. It is
unsound when the perturbation is to the *values*, because then every one of the 48
layers feeds the next a corrupted input and the errors compound multiplicatively —
which is exactly why the residuals grow with rel_err (−2.4 at 0.35, −22.9 at 0.62)
rather than staying constant.

**So the ladder's scope, restated** — the third and sharpest revision it has taken in
this doc:

| change being screened | ladder error | status |
|---|---|---|
| which channels a layer selects (fixed budget) | ±0.26pt | **validated** |
| the read *pattern* at fixed budget (`weight_sparse`) | ≤0.6pt | validated |
| a large move along `ρ_input` at fixed pattern | 1.95pt | unreliable |
| budget moved *between* layers | 1.0pt, wrong sign | **broken** |
| **the channel values themselves (`input_only`)** | **2.4–22.9pt, wrong sign across methods and not rank-preserving across arms** | **badly broken** |

Any future method where the cheap read feeds the computation needs its own end-to-end
anchor. A single-layer error surface will systematically **flatter** it, and worst
exactly where the method is worst — the residual grew from −2.4pt at rel_err 0.35 to
−22.9pt at 0.62. The general rule this suggests: a separable, single-layer objective is
sound for perturbations that a downstream layer sees as *noise on its input*, and
unsound for perturbations that *change what the layer computes*, because only the
latter compound.

### What it does establish

- **The exact second pass is worth 2.2–5.2pt HellaSwag**, i.e. far more than the
  third of the budget it consumes. This is the first direct measurement of what
  `input_sparse`'s two-pass structure actually buys, and it justifies the structure.
- **Read allocation is the dominant lever once reads carry values** (+8.9 to
  +21.8pt). Free, and it grows with depth.
- **A single-layer rel_err surface cannot screen a value-perturbing method.** Cost:
  one 12-eval sweep to learn, which is cheap for a scope boundary this sharp.
- The engineering advantages are real but cannot pay for the accuracy: one pass
  instead of two (no scoring overhead, no gather between passes), zero extra storage,
  no probe object, no calibration, and the selector question disappears entirely
  (`oracle_mag_noW`'s choice is exact by construction). At −2.2 to −5.2pt they are not
  enough — **spend the reads on `input_sparse` instead.**

---

---

## Rotated read sets: porting lm_head's `S1` screen

> Every method above picks the token's read set in the **standard basis** — top
> `ρ_input` coordinates by `|x_i|`. `docs/exps/lm_head/results_lm_head.md` §3b found
> that on the lm_head this is the *wrong* basis by a factor of 2.8× (adaptive
> KL 0.805 `raw` vs 0.287 `ceig`): rank the coordinates of `z = Uᵀx` instead, with
> `U` the eigenvectors of `C = E[x xᵀ]`, and the same read budget captures far more.
> Does that transfer to expert FFNs?

**Verdict: no. Every form that pays for its own rotation loses by 0.9–4.4pt, and
the one form whose rotation is free gains +0.32pt — inside the noise floor.** The
mechanism is real but small once the basis has to be *estimated*, and the part of it
that looked large is a covariance-overfitting artifact.

### Why the rotation is not free here, and the three forms that follow

The screen is the same arithmetic as lm_head's: projecting the hidden state and
multiplying by the unrotated `W` is identical to reading `r0` columns of the rotated
`W U`, since `W U_S U_Sᵀ x == W x̃`. What differs is **who pays for `z`**. On the
lm_head the rotation costs `D²` against a `V×D` read — 1.35%, negligible. In an MoE
layer the comparable ratio is `H²` against `K·3·I·H`, and that is **not** negligible:

```
scoring   2·r0/(3H)          r0 columns of both branches, all I rows
compute   ρ_channel          all three matrices gathered to the kept channels
rotation  r'/(3·K·I)         H·r' reads of U_{r'}, shared by the token's K experts
                             = 0.1111 for the full basis (r'=H) on this model
```

At the −75% operating point the whole budget is `used = 0.25`, so a full per-layer
rotation eats **44% of it** before a single weight column is read. Three ways out,
with three different bills:

| form                        | basis            | rotation cost      | extra storage | why                                                                                         |
| --------------------------- | ---------------- | ------------------ | ------------- | ------------------------------------------------------------------------------------------- |
| **`rot_full`**        | per-layer`U`   | `H/(3KI)` = .1111 | **0**   | rebase the served weights (`W ← WU`); the compute pass then needs all of `z`             |
| **`rot_lr`**          | per-layer`U_r'` | `r'/(3KI)`       | `2r'/(3H)`  | store a separate rank-`r'` probe `W·U_{r'}`; compute stays in the original basis         |
| **`rotglob`** (free)  | one global`U`  | **0**        | **0**   | rotate the residual stream *once*, QuaRot-style, folding `U` into every weight that reads or writes it — then `z` **is** the residual state |

`rotglob` is the interesting one: RMSNorm commutes with an orthogonal rotation
(`‖Ux‖ = ‖x‖`) once `γ` is folded forward, so the whole model can be rebased offline
and the rotation genuinely costs nothing at run time. The price is that **one basis
must serve all 48 layers**. `rot_full` keeps each layer's own optimum but must
materialize `z = U_ℓᵀx`, an `H²` matvec per layer per token.

This taxonomy is also what makes "count `U` or not" a real question rather than a
bookkeeping choice: **`rotglob` is the only honest way to not count it.**

### The basis is the whole experiment, and it overfits

`C` is `2048×2048`. A capture holds 8192 tokens, so `T/H = 4` and the sample
eigenbasis fits noise. Measured (`scripts/probe_rotate_diag.py`), captured input
energy at `r0 = 384` reads:

| basis for the top-`r0` `|z|` rule | L6              | L22             | L38             | L46              |
| --------------------------------------- | --------------- | --------------- | --------------- | ---------------- |
| raw`|x_i|` (what `input_sparse` does) | 0.691           | 0.684           | 0.744           | 0.765            |
| eigenbasis,**in-sample**          | **0.856** | **0.874** | — | **0.909** |
| eigenbasis,**held out**           | 0.709           | 0.707           | —              | 0.761            |

**The +0.17 in-sample gain collapses to +0.02 held out, and at L46 it goes
negative.** So the first read of "the rotation halves the dropped energy" was an
artifact. A learning curve over the fit-set size (512 → 6144 tokens) climbs ~+0.010
energy per doubling, which is why the real experiment needed a properly estimated
basis, not a capture-sized one: `scripts/probe_cov_collect.py` accumulates `C` for
all 48 MoE layers over **158 853 non-pad C4 tokens** (`T/H = 77`), storing only the
`H×H` matrix per layer so more tokens cost no more memory. Padding is excluded —
44% of the batch grid is padding here, the same trap as lm_head's bug 1.

Every rotated arm below is therefore run against four bases: `heldout` (capture,
`T/H=2`), `calib` (the 158k-token collection), `oracle` (fit on the very tokens
being scored — **unachievable**, an upper bound on any calibration effort), and the
pooled-over-all-48-layers versions for the free form.

### Measured: iso-cost at −75%

Layer-averaged block-output `rel_err`, 4 layers × 4096 C4 tokens, `router`
allocation, `ρ_channel = 0.125`. The instrument reproduces this doc's published
anchors **to four decimals** — `oracle_mag_noW` 0.3272, `input_sparse` uniform
0.4099 / router 0.3860 at `ρ_input=0.25` — so these numbers are on the same scale as
everything above. Δpt uses the validated −26.4 (HellaSwag) / −20.3 (MMLU) slopes.
The baseline is the doc's own −75% operating point: `input_sparse` + `router` at
`ρ_input=0.1875` (`r0=384`), measured end-to-end at **74.08 HS / 77.77 MMLU**.

**Setting A — the rotation is billed as active parameters.** Every row is at
`used = 0.2500`; `r0` is whatever the budget allows *after* the rotation is paid for:

| arm                                   | basis            | `r'` | `r0` | rot cost      | extra storage   | `rel_err`      | ΔHS pred          | ΔMMLU pred |
| ------------------------------------- | ---------------- | ------ | ------ | ------------- | --------------- | ---------------- | ------------------ | ---------- |
| **`input_sparse` (baseline)**   | —               | —     | 384    | 0             | **0**     | **0.4134** | —                 | —         |
| `rot_lr`                            | `calib`        | 768    | 256    | 0.0417        | 25.0%           | 0.4477           | **−0.91**   | −0.70     |
| `rot_lr`                            | `calib`        | 1152   | 192    | 0.0625        | 37.5%           | 0.4501           | **−0.97**   | −0.75     |
| `rot_lr`                            | `calib`        | 510    | 299    | 0.0277        | 16.6%           | 0.4758           | −1.65             | −1.27     |
| `rot_full`                          | `calib`        | 2048   | **43** | 0.1111        | 0               | 0.5800           | **−4.40**   | −3.38     |
| *`rot_lr`, oracle basis*            | *`oracle`*   | 1152   | 192    | 0.0625        | 37.5%           | *0.3960*       | *+0.46*         | *+0.35*  |
| *`rot_full`, oracle basis*          | *`oracle`*   | 2048   | 43     | 0.1111        | 0               | *0.5114*       | *−2.59*        | *−1.99* |

**Setting B — the rotation is not billed.** All rows read `r0 = 384` columns at
`ρ_channel = 0.125`, i.e. exactly the baseline's cost:

| arm                                    | basis                       | rotation really free? | `rel_err`      | ΔHS pred          | ΔMMLU pred |
| -------------------------------------- | --------------------------- | --------------------- | ---------------- | ------------------ | ---------- |
| **`input_sparse` (baseline)**    | —                          | —                    | **0.4134** | —                 | —         |
| **`rotglob`**                    | `globcalib` (48 layers)   | **yes**         | **0.4013** | **+0.32**   | **+0.25** |
| `rot_full`, unbilled                 | `calib` per-layer         | no                    | 0.3863           | +0.72             | +0.55     |
| `rot_full`, unbilled                 | `heldout` (capture)       | no                    | 0.4096           | +0.10             | +0.08     |
| *`rot_full`, unbilled*               | *`oracle`* (upper bound) | no                    | *0.3575*       | *+1.47*         | *+1.13* |
| *`rotglob`, unbilled*                | *`globoracle`*            | yes                   | *0.3786*       | *+0.92*         | *+0.71* |
| static rank-384 subspace               | `calib`                   | (0.0208 billed)       | 0.5055           | **−2.43**   | −1.87     |

### What the two tables say

- **Counting `U` closes the idea outright.** The cheapest achievable arm is −0.91pt
  *and* asks for 25% extra storage; the zero-storage arm (`rot_full`, which must
  rebase the served weights and therefore pay the full `H²`) is −4.40pt, because
  after the rotation it can only afford **43** of 2048 coordinates. Even the
  **oracle** basis — which cannot be built — tops out at +0.46pt for 37.5% of extra
  storage. There is no version of this trade that works.
- **Not counting `U` is only legitimate for `rotglob`, and it buys +0.32pt.** That is
  free in both reads and storage, and it is a real deployable object (one offline
  rebase of the residual stream). But 0.32pt sits inside the eval stderr (≈0.42pt
  HellaSwag) and inside the ladder's own demonstrated residual band (−0.08 to −1.95pt
  across this doc). Compare `router` allocation: predicted +0.66, measured +0.58 at
  this budget — and *−0.11* one budget looser. A +0.32pt prediction is not a
  shippable claim.
- **Per-layer bases are worth ~2× the global one** (+0.72 vs +0.32 unbilled) —
  exactly the quantity you cannot have for free, since a shared basis is what makes
  the rotation cost nothing. The mechanism and its affordability are anti-correlated.
- **The basis has not converged, and it does not matter.** `calib` (158k tokens) is
  still 0.029 rel_err short of `oracle`, so more calibration would keep improving
  Setting B. It cannot rescue Setting A, because `oracle` *is* the ceiling there and
  `oracle` already loses at every `(r', r0)` except one that costs 37.5% storage for
  +0.46pt.
- **The static-subspace control is dead again**, on a third axis: a fixed rank-384
  activation eigen-subspace (no per-token choice) scores 0.5055 against 0.4134 for
  the per-token raw rule — **−2.43pt**. This is the same
  [unifying negative result](#unifying-negative-result) as ever: an operator fixed
  offline reproduces only the average token. Note this arm *is* activation-aware
  input-side low-rank, i.e. the family §3 already killed by recall, now killed by
  output error too.
- **The column-norm factor is a wash**, as it is in the raw basis. lm_head gate 0g
  needs `|z_i|·‖W u_i‖`; here it moves `rel_err` by 0.0005–0.0007 (`rotglob`
  0.4008 vs 0.4013), i.e. +0.02pt. The rotated column norms are more spread than the
  raw ones (CV 0.040–0.048 vs 0.022–0.036) and still far too uniform to matter.

### Why S1 transfers to the lm_head and not to this

Three differences, all structural rather than incidental — worth recording because
they predict *which* future lm_head ideas are worth porting:

1. **The rotation's cost is amortized over the wrong thing.** `D²` against a `V×D`
   read is 1.35%; `H²` against `K·3·I·H` is 11.1%, because an MoE layer reads only
   `K` of its `E` experts. The lm_head is one enormous matrix, which is exactly the
   shape that makes a `D×D` preprocessing step free.
2. **The screen has somewhere cheaper to go.** S1's alternative to a rotated screen
   was a *static frequency tier* (a 700× worse KL). `input_sparse`'s alternative is
   an already-adaptive per-token `|x_i|` rule, and this model's input energy is
   **axis-concentrated** — massive-activation coordinates — so the standard basis is
   already close to the right one. There was a 2.8× gap to collect on the lm_head and
   roughly a 1.05× gap here.
3. **A shared basis is worth less than a per-layer one.** The lm_head needs exactly
   one basis, at one point in the network, so per-layer-vs-global never arises.

The transferable rule: **port an lm_head read-axis method only when the rotation's
`D²` analogue is small against the read it saves.** The `rot_lr` variant — one basis
shared across a layer's 128 experts, coefficients per expert — was the design built
specifically to beat that arithmetic, and it does beat plain `lowrank_scorer` by 16×
on cost per read; it still loses, because the *basis quality* term rather than the
cost term is binding.

### Reproducing

```bash
# 1. mechanism gate: does the eigenbasis concentrate x better than |x|? (CPU, ~5 min)
python scripts/probe_rotate_diag.py --layers 6,22,38,46
# 2. a basis worth screening: all 48 layers' C over ~160k non-pad C4 tokens
#    (4 GPUs, ~20 min; do NOT set HF_HUB_OFFLINE=1 -- the c4 shard config must resolve)
CUDA_VISIBLE_DEVICES=0,1,2,3 PER_GPU_MEM=36GiB \
  python scripts/probe_cov_collect.py --tokens 262144 --layers all
# 3. the iso-cost screen, both accounting settings (1 GPU, ~25 min)
CUDA_VISIBLE_DEVICES=0 python scripts/probe_rotate_screen.py
CUDA_VISIBLE_DEVICES=0 python scripts/probe_rotate_screen.py --colnorm --ratios 0.125 \
  --arms "raw:-:384:router,globcalib:2048:384:router,calib:2048:384:router"
```

**No end-to-end eval was spent on this**, by decision: the screen's anchors are
exact, Setting A loses by more than the ladder's worst-ever residual, and Setting B's
+0.32pt is below the resolution an eval would give. If it is ever revisited, the only
arm worth the ~13 GPU-hours is `rotglob` at `r0=384` — and it needs `install.py`
support for a residual-basis rebase, which does not exist.

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

| mechanism                                                        | result                                                              | why it fails                                                                     |
| ---------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **[q]** Product quantization (0.25–1.0 bits/w)            | recall 0.42–0.60 vs RTN 0.71 at same bytes                         | Rows near-orthogonal (cos-to-centroid 0.078 vs 0.022 random)                     |
| **[q]** Hadamard rotation                                  | 3-bit 0.818 vs 0.813 unrotated                                      | No weight outliers; destroys input sparsity                                      |
| **[q]** 1-bit sign + per-group scale                       | recall 0.546 at cB 0.126                                            | Below ~3 bits per-channel ordering is gone                                       |
| **[q]** Asymmetric gate/up precision                       | 0.611 vs 0.675 symmetric                                            | Score is a product; errors add in quadrature                                     |
| **[q]** Router-adaptive precision                          | mass capped at 0.839 < 0.864 uniform                                | Redundancy is not expert-level                                                   |
| **[q]** Closed-form debiasing of the proxy                 | rel_err 0.437 vs 0.369 baseline —*worse*                         | Per-channel noise uniform; clamping destroys ordering                            |
| Norm-weighted coordinate selection (`\|x_i\|·rms_i(W)`)         | ≈ plain top-`\|x\|`                                                | Column-norm CV = 0.022 — nothing to weight                                      |
| Relaxed-candidate cascade (λ=1.5)                               | recall 0.787, but no cheaper than a better probe at the same budget | The extra exact reads buy less than spending them on`ρ_input`                 |
| `‖W_down[:,j]‖` weighting of the score                       | rel_err 0.370 vs 0.369                                              | Factor CV = 0.055                                                                |
| `\|gate\|` instead of `SiLU(gate)`                             | rel_err 0.647 vs 0.369                                              | The SiLU gate is load-bearing                                                    |
| Static pre-filter (ban low-freq channels)                        | loses 12–18% of top-B mass at 25% ban                              | Keep-frequency near-uniform                                                      |
| Per-row weight sparsity                                          | recall 0.622                                                        | Below top-`\|x\|` at serving precision (0.716) and needs extra indexed structure |
| Static input subset (one fixed coordinate set)                   | recall 0.558                                                        | Overlap of top-`\|x\|` with a global set is only 0.375                           |
| Activation-eigenbasis rotation of the read set (lm_head`S1`)   | free/global form **+0.32pt** predicted; every self-paying form **−0.9…−4.4pt** | `H²/(3KI)` = 11.1% of the entire −75% budget, and the only *free* form needs one basis for all 48 layers, which captures half the concentration. [Full section](#rotated-read-sets-porting-lm_heads-s1-screen) |
| Discriminability-weighted input                                  | ≈ top-`\|x\|`                                                      | `CV(Var_j W) = 0.042`; 93.8% identical selection                               |
| Floating the per-token**channel** budget under a global τ | −0.16…−0.32pt at iso-cost                                        | Score is already`g_e`-scaled; τ tracks router confidence                      |

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

| tool                                                                         | what it does                                                                                                                                                                                                                                                                                                                                           | when to use                                                                                                                                                                      |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Unstructured read-set screen** (`scripts/wsparse_screen.py`)       | One`rel_err` table over the whole column-vs-entry family at an identical read budget: staircase rectangles, graded ladders, the budget-exact `                                                                                                                                                                                                       | W·x                                                                                                                                                                             |
| **One-pass error screen** (`scripts/input_only_error.py`)            | `rel_err` for `input_only`, where the mask lands on the **approximate** intermediate so the error cannot be computed from the dropped channels alone (`probe_output_error.py`'s shortcut). Sweeps `(ρ_input, ρ_channel)` along iso-cost lines and measures `oracle_mag_noW` + two-pass `input_sparse` in the same pass                                                                                                                                                                                                                                          | **⚠️ Do not trust its verdicts.** It called `input_only` an iso-cost tie with `input_sparse`; the evals showed it dominated by 2.2–5.2pt, and its level predictions missed by up to 22.9pt. Kept because it is the correct *mechanics* for a value-perturbing method and because that failure is [the scope lesson](#the-ladder-broke-here-and-now-we-know-by-how-much) — but any such method needs a real eval |
| **Scoring-pass benchmark** (`scripts/wsparse_bench.py`)              | Wall-clock of each scorer per MoE block at real dimensions, relative to one full-width`gate+up`, with the `tau` bisection timed separately                                                                                                                                                                                                         | Before committing to an eval: this is how the 54×→0.8× and 359→140 ms fixes were found                                                                                       |
| **Output-error ladder** (`scripts/probe_output_error.py`)            | Measures layer-averaged block-output`rel_err = ‖y_full − y_kept‖/‖y_full‖`; fitted slope −24.3 HellaSwag pt/unit rel_err predicted two measured points to **within 0.1pt**                                                                                                                                                               | Screening any new scorer design — 4 min, 1 GPU, replaces 150 GPU-hours of eval                                                                                                  |
| **Linearity check** (`scripts/probe_relerr_linearity.py`)            | Regresses rel_err → measured accuracy per layer and per*family*; found the two families are different rulers (−26.4 vs −6.9 pt/unit)                                                                                                                                                                                                              | **Before trusting any rel_err → pt conversion**; use the −26.4 fixed-budget slope for scorer comparisons                                                                 |
| **All-layer surface + allocator** (`scripts/probe_layer_surface.py`) | `rel_err(ρ_input, ρ_channel)` over all 48 MoE layers (each measured against unperturbed input), then a **slope-weighted** Lagrangian solve for the per-layer budget split; emits a schedule JSON. `--reuse-surface` re-solves from cache with no GPU; `--no-slope-weight` reproduces the measured-worse unweighted objective             | Choosing budgets across layers instead of hand-picking one global`(ρ_input, ρ_channel)` — but note the unweighted version **lost** an eval, so verify before trusting |
| **Iso-cost split solver** (`scripts/probe_split_solve.py`)           | Solves`min rel_err(ρ_input, ρ_channel)` s.t. `ρ_channel + 2ρ_input/3 = C` from the **cached** surface — the *within-layer* analogue of the per-layer solve, and unlike it a **selector** change (one global pair), so the ladder applies. Reports `ρ_input*`, `B`, reads/token and asserts both measured arms are iso-cost | Picking the split at a target budget instead of hand-fixing`ρ_input=0.25`. No GPU, seconds — run before spending an eval on a new `(ρ_input, ρ_channel)` pair            |
| **Input-allocation screen** (`scripts/probe_input_alloc_screen.py`)  | Compares`uniform`/`router`/`router2`/`colnorm` coordinate allocation at an identical pooled read budget                                                                                                                                                                                                                                        | Deciding how to split the probe's input reads across a token's K experts                                                                                                         |
| **Threshold-vs-top-B** (`scripts/probe_threshold_budget.py`)         | Floats the per-token channel budget and/or input reads under an offline-calibrated global threshold; reports every mode**interpolated to the baseline's realized cost**                                                                                                                                                                          | Any "let the budget float per token" proposal — the iso-cost interpolation is what keeps a merely-cheaper variant from looking like a win                                       |
| **Recall screen** (`scripts/probe_frontier.py`)                      | Index recall + captured score mass, fast rerun from cached captures                                                                                                                                                                                                                                                                                    | Quick within-family ranking (do NOT use to rank across families — BTT/SVD mismatch)                                                                                             |
| **Static-floor diagnostic** (`scripts/actaware_diag_static.py`)      | Reports recall excess over a free static prior + centered Spearman (mean removed)                                                                                                                                                                                                                                                                      | **Run before proposing ANY new scorer** (~4 min, 1 GPU) — separates "buys per-token signal" from "re-derives the free prior"                                              |
| **Activation-aware screen** (`scripts/actaware_scorer_screen.py`)    | All sketch families on one cost axis; recall/mass/rel_err                                                                                                                                                                                                                                                                                              | Evaluating a new low-rank or basis-based family                                                                                                                                  |
| **Learned probe** (`scripts/lowrank_scorer_learned_probe.py`)        | Trains an online-form scorer on calibration tokens, measures headroom above the SVD                                                                                                                                                                                                                                                                    | Showing whether objective or rank is the bottleneck                                                                                                                              |
| **Probe capture** (`scripts/probe_capture.py`)                       | Caches per-layer`gate/up/down` activations + `x`, `g_e` for offline analysis                                                                                                                                                                                                                                                                     | Prerequisite for`probe_output_error.py` (needs the `_wd` captures)                                                                                                           |
| **Low-rank recall** (`scripts/lowrank_scorer_recall.py`)             | Replays exact per-token cross-expert selection over 4 layers × 8192 tokens                                                                                                                                                                                                                                                                            | Scanning the SVD/BTT cost–recall curve                                                                                                                                          |
| **Basis-concentration gate** (`scripts/probe_rotate_diag.py`)        | Captured input energy per read for four rules (raw/rotated × adaptive/static), **in-sample and held out**, plus a fit-set learning curve and the rotated column-norm CV                                                                                                                                                                        | **Before any change of read *basis*** (~5 min, CPU). It is what showed the eigenbasis' +0.17 energy gain is only +0.02 held out — an in-sample screen would have shipped a fiction |
| **Input covariance collector** (`scripts/probe_cov_collect.py`)      | Accumulates`C = E[x xᵀ]` and `μ` for all 48 MoE layers over ~160k non-pad C4 tokens, storing only the `H×H` matrix per layer, so 100× more tokens costs no more memory. Excludes padding — 44% of the batch grid                                                                                                                          | Any method needing an activation second moment. A capture-sized`C` (`T/H=4`) is **not** enough for an eigenbasis; this reaches `T/H=77`                                    |
| **Rotated read-set screen** (`scripts/probe_rotate_screen.py`)       | One`rel_err` table over the rotated-basis family (`rot_full`/`rot_lr`/`rotglob` × four basis qualities, including an unachievable **oracle**), reporting `used` under **both** cost models — rotation billed and not — plus each arm's extra storage                                                                                | Comparing any scorer whose cost depends on a*shared* object. The oracle-basis column is what makes its negative verdict final rather than "needs more calibration"              |

**Methodological lesson:** recall orders correctly *within* a scorer family but
**not across families** (pearson 0.65 with accuracy). The output-error ladder is
the correct screening instrument for cross-family comparison.

---

## Honest accounting summary

Three frames have appeared across these docs. **Only the last is used here**, and
the older two are recorded so old numbers can be re-read:

| frame                                   | formula                      | why it is not used                                                                                                                               |
| --------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Proxy cost (oldest,`sparse_probe.md`) | cut = 1 − ρ − cB/3        | Prices a*separate* quantized scorer object, which is excluded — and it double-counts nothing but also credits nothing to storage              |
| Overlap-discounted traffic              | kept = ρ + 2·p·(1−ρ)/3  | Defensible for memory traffic under a perfect cache, but it assumes the kernel keeps the scoring reads resident, and it flatters the method      |
| **Used parameters (this doc)**    | **kept = ρ + 2·p/3** | **scoring params + compute params, overlap billed twice — conservative, kernel-independent, and identical to `lowrank_scorer`'s frame** |
| One-pass (`input_only` only)          | kept = (2·p + ρ)/3      | Not an alternative frame for the rows above — it prices a *different method*, one with no second pass and therefore no overlap to bill. Reduces to `oracle_mag_noW`'s `(1+1+ρ)/3` at `p=1`, which is the cross-check that the two frames agree where the methods do |

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

| mode                             | used 0.267                 | used 0.292               | used 0.317      | used 0.367      |
| -------------------------------- | -------------------------- | ------------------------ | --------------- | --------------- |
| `fixed_B / fixed_p` (current)  | 0.4170                     | 0.3848                   | 0.3581          | 0.3158          |
| `thresh_B / fixed_p`           | 0.4244**(−0.20pt)**       | 0.3911 (−0.17)          | 0.3641 (−0.16) | 0.3281 (−0.32) |
| **`fixed_B / thresh_p`** | **0.4087 (+0.22pt)** | **0.3796 (+0.14)** | 0.3560 (+0.06)  | 0.3243 (−0.22) |
| `thresh_B / thresh_p`          | 0.4145 (+0.07)             | 0.3846 (+0.01)           | 0.3612 (−0.08) | 0.3402 (−0.64) |

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

**0. Make the read set unstructured — measured, and now the biggest lever.** At the
same reads it is worth **+3.46pt** over the column control and moves the achievable
cut from −73.3% to −80.0% *above* the old accuracy. Take the **one-level rectangle**
(`levels: "0.25x0.45"`, `alloc_mode: rank`, uniform): it ties the 8-level threshold
ladder within noise at a quarter of the metadata (4.3% of the FFN) and less than half
the scoring latency. Do **not** stack `router` allocation on it — that composed for
column scoring and does not compose here.

**0b. Do NOT reuse the sparse read as the computation — closed negative.**
[`input_only`](#input_only-stop-scoring-just-compute-on-the-sparse-input) deletes
`input_sparse`'s exact second pass and loses 2.2–5.2pt HellaSwag at iso-cost, widening
with depth. Keep the two-pass structure. The transferable parts are that **`router`
allocation is worth +8.9 to +21.8pt once reads carry values** (so any value-perturbing
scheme must allocate by `g_e`) and that the **rel_err ladder cannot screen such a
scheme at all** — it mis-ranked the two methods outright.

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

| direction                                     | status                                                                    | expected value                                                                                                                                                                                                                 |
| --------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`weight_sparse` MMLU at −80.0%**   | 4 HellaSwag**done** (74.87 / 74.47 / 74.41 / 71.41), 4 MMLU running | HellaSwag confirmed the frontier move. MMLU decides whether the +3.46pt holds on the benchmark that degrades most gracefully elsewhere                                                                                         |
| **Why `router` stopped paying**       | observed, unexplained                                                     | It transferred for column scoring (+0.58) and not for`tau` (−0.46). If the mechanism is "both fix read rigidity", that predicts `router` is also dead for the rectangle — cheap to test and it would simplify the recipe |
| **Channel-block level maps**            | **closed, negative**                                                | Blocks of 8 cut metadata to 2.8% but give back 1.5 of the 3.0pt, landing +0.14pt from the zero-metadata column control;`r=16` converges back onto column sparsity entirely                                                   |
| **Pooled cross-expert `τ`**          | not attempted                                                             | `router` approximates the pooled `g_e·                                                                                                                                                                                      |
| **LoRA recovery at the goal point**     | not started                                                               | The 3.92pt HellaSwag / 3.24pt MMLU gap at −73.3% is the size stage-2 LoRA closes elsewhere;**now the largest single win**                                                                                               |
| **Iso-cost split, MMLU**                | **closed**                                                          | Measured:**+0.27pt at −73.3%, +0.44pt at −75.0%** — consistently positive on MMLU, both inside stderr. Superseded by the depth-dependent verdict below                                                                |
| **Iso-cost split at resolution**        | **closed by depth, not by samples**                                 | Resolved by a deeper budget instead of more samples: a wash at −73.3%/−75.0% but**+1.14pt HS at −80.0%** (row 7e vs 0d). Solve the split when the budget is tight; hand-picking is fine above −75%                   |
| **`router` at a deeper ρ**           | **closed**                                                          | Spent on depth and measured to −80.0%: the full curve is 76.61 → 72.55 HS / 79.45 → 76.11 MMLU across −63.3% → −80.0%, ≈0.24pt HS per pp, no cliff                                                                      |
| **Why the schedule trades HS for MMLU** | observed, unexplained                                                     | The unweighted schedule is −0.16 HS /**+0.65 MMLU** at iso-cost. If the trade is controllable it is a knob, not a bug                                                                                                   |
| **Non-separable cross-layer objective** | not attempted                                                             | Every separable solve failed; an objective that sees compounding across depth is the only remaining route, and is much more expensive to fit                                                                                   |
| **Threshold the input reads only**      | measured offline (+0.22pt at used=0.267), not evaluated                   | Free, removes a sort; channel budget must stay a top-B (measured)                                                                                                                                                              |
| **Sequence-level carry-forward**        | untested                                                                  | Consecutive tokens share hot channels → zero-weight-read scorer from an EMA of recent activations; needs order-preserving capture                                                                                             |
| **Stack with reduce-top-k**             | not started                                                               | top-4 ×`input_sparse` reaches beyond −85%; smaller K·I pool is an easier ranking problem                                                                                                                                  |
| **Wall-clock microbenchmark**           | not done                                                                  | Everything is an active-parameter claim; gathered-expert latency needs measuring                                                                                                                                               |

**Closed by this round:**

- **The zero-extra-storage frontier is a measured curve, not a point**: six budgets ×
  two benchmarks (12/12 evals), −63.3% → −80.0%, monotonic with no cliff, reaching
  **−80.0% at 72.55 HellaSwag / 76.11 MMLU**.
- **The rel_err ladder generalizes across the whole budget axis** — HellaSwag slope
  −24.5 (R²=0.997) vs the pre-registered −26.4, and MMLU −20.3 (R²=0.979). One surface,
  two metrics, per-metric slope.
- **MMLU degrades more gracefully at every budget and the gap widens with depth**
  (0.49pt → 1.21pt), with the shallower MMLU rel_err slope as the mechanism.
- **Solving the iso-cost `ρ_input`/`ρ_channel` split is depth-dependent**: a wash at
  −73.3%/−75.0% (−0.01 … +0.44pt over four tests) but **+1.14pt at −80.0%**. The
  hand-picked `ρ_input=0.25` was never costing anything above −75%.
- **Column-structured sparsity is the wrong shape for the scoring budget** — measured
  end-to-end, not just offline. At identical reads, spending them on
  `(channel, coordinate)` **entries** scores **74.87 vs 71.41** HellaSwag (**+3.46pt**,
  5.5σ) at `used = 0.200` / −80.0%, which also beats the old −73.3% headline by 0.23pt
  while cutting 6.7 points more.
- **The cheap version is as good as the clever one.** A fixed rectangle (25% of
  coordinates × 45% of channels, one masked GEMM, 1 bit/weight of metadata) scores
  74.47 — within noise of the 8-level per-token threshold ladder's 74.87. Prefer it.
- **`router` allocation does not compose with the threshold mode**: −0.46 ± 0.62pt,
  against +0.33 predicted, even though the same allocation transferred cleanly for
  column-only scoring. Per-token floating and cross-expert reallocation fix the same
  deficiency.
- **Static weight masks remain dead** — a per-column-balanced mask is *worse* than
  columns-only, and a global Wanda mask is +0.33pt. Freedom over *which* entries only
  pays when the choice is made **per token**.
- **The entry-level gain is not compressible into cheap metadata**: channel blocks of
  8 keep 2.8% storage but give back half the gain, landing +0.14pt from the
  zero-metadata column control.
- **The mean-fix does not work** — exactly precomputing `W·μ` and scoring the centered
  input is a wash for the unstructured scorer and a loss for the column one, despite
  `μ` carrying 12–19% of input energy.
- **Up-only is the wrong trade here**: one branch at 2× density loses 3.7pt to two
  branches at 1×, the opposite of the low-rank family's verdict.
- **−73.3% with zero extra storage is real**: 74.64 HellaSwag / 77.67 MMLU (row 7).
- **`router` input allocation works end-to-end** (+0.58 / +0.47pt at fixed cost),
  confirming the offline screen's +0.66 prediction.
- **−75.0% is reachable at the *uniform* −73.3% accuracy** (74.08 vs 74.06) by
  solving the `ρ_input`/`ρ_channel` split instead of hand-picking `ρ_input=0.25`.
  It does not beat `router` at −73.3% (−0.56pt), so it extends the frontier rather
  than moving it.
- **The hand-picked `ρ_input=0.25` was never optimal at deep budgets** — the solved
  `ρ_input`\* is 0.1875 at every budget ≤ −73.3%, i.e. a ~50/50 scoring/compute
  split rather than 67/33.
- **The iso-cost split is confirmed in sign only** — +0.28pt for the solved split
  against a predicted +0.58, with a 0.62pt stderr on the difference. Not resolvable
  from one pair; the practical read is "solving the split does not hurt and may help
  a little", which is weaker than this round's other allocation claims.
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

### Unstructured read sets (`weight_sparse`)

```bash
# 1. The iso-cost read-pattern screen: one layer per GPU, then merged (~5 min/layer)
bash scripts/run_wsparse_screen.sh                 # TOKENS=4096 by default

# 2. Any single comparison, e.g. cheap band edges vs exact per-column pricing vs the
#    exact top-N ceiling, all at density 0.1125 (= used 0.200 at rho_channel 0.125)
LAD="1.0|0.7|0.49|0.343|0.24|0.168|0.118|0.082"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python scripts/wsparse_screen.py \
    --layers 46 --max-tokens 4096 --rhos 0.10,0.125 --prod-tokens 1024 \
    --variants "stair:0.1125x1.0,tau:0.1125:$LAD,taux:0.1125:$LAD,prod:0.1125"

# 3. Scoring-pass wall clock at real MoE dimensions (1 GPU, no model)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python scripts/wsparse_bench.py --tokens 1600

# 4. Evals (4 GPUs each; the queue waits for idle GPUs before starting)
NEED=4 MINFREE=30000 bash scripts/run_wsparse_evals.sh \
    qwen3_30b_a3b_wsp_tau_router_hellaswag.yaml qwen3_30b_a3b_wsp_tau_router_mmlu.yaml

# Unit tests (30; anchors: single-level == input_sparse, density 1 == oracle_mag_noW,
# budget never exceeded, tau tracks the exact product rule)
python -m pytest src/dynamic_active_param/tests/test_weight_sparse.py -q
```

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

# 3c. ISO-COST SPLIT: solve the rho_input/rho_channel split at a fixed budget from
# the CACHED surface -- no GPU, seconds. Equalizes marginal rel_err per unit budget,
# (3/2)*d(relerr)/d(rho_input) == d(relerr)/d(rho_channel). rho_input* = 0.1875 at
# every budget <= -73.3%, i.e. the hand-picked 0.25 over-spends on scoring.
python scripts/probe_split_solve.py --budgets 0.20,0.225,0.25,0.2667,0.2917 \
    --surface docs/results/idea_pilot/layer_surface_8k.json

# 4. Evals (A100, 8 GPUs, 2 waves)
bash scripts/run_probe_reuse_sweep.sh    # rho_input=0.25 x rho_channel {0.10,0.15,0.20}
bash scripts/run_probe_alloc_sweep.sh    # router alloc / per-layer schedule / both
# iso-cost split, both arms at used=0.2500 (-75.0%); MMLU needs 4 GPUs/job (2 OOMs)
bash scripts/run_probe_split_sweep.sh        # opt 0.1875/0.125 vs mis 0.25/0.0833
GPUS=0,1,2,3 bash scripts/run_probe_split_mmlu_retry.sh   # MMLU arms, 4 GPUs, serial
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

| component                                            | path                                                                                                                           |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `weight_sparse` criterion (unstructured read sets) | `src/dynamic_active_param/weight_sparse.py`                                                                                  |
| `weight_sparse` tests (30)                         | `src/dynamic_active_param/tests/test_weight_sparse.py`                                                                       |
| Unstructured read-set screen                         | `scripts/wsparse_screen.py`, `scripts/run_wsparse_screen.sh`                                                               |
| Scoring-pass benchmark                               | `scripts/wsparse_bench.py`                                                                                                   |
| Eval queue (waits for idle GPUs)                     | `scripts/run_wsparse_evals.sh`                                                                                               |
| Eval configs (this round)                            | `configs/eval/qwen3_30b_a3b_{wsp_tau_router,wsp_tau,wsp_rect,probe_col_r1125}_{hellaswag,mmlu}.yaml`                         |
| Screen results (JSON)                                | `docs/results/idea_pilot/wsparse_{screen,splits,validation,tau_vs_exact}.json`                                               |
| `input_sparse` criterion (+ input allocation)      | `src/dynamic_active_param/sparse_probe.py`                                                                                   |
| Sparse probe tests (30)                              | `src/dynamic_active_param/tests/test_sparse_probe.py`                                                                        |
| Per-layer schedule loading                           | `src/dynamic_active_param/install.py` (`schedule_path`)                                                                    |
| Linearity study                                      | `scripts/probe_relerr_linearity.py`, `..._plot.py`                                                                         |
| All-layer surface + allocator                        | `scripts/probe_layer_surface.py`, `..._plot.py`                                                                            |
| Iso-cost split solver                                | `scripts/probe_split_solve.py`                                                                                               |
| Input-allocation screen                              | `scripts/probe_input_alloc_screen.py`                                                                                        |
| Threshold-vs-top-B study                             | `scripts/probe_threshold_budget.py`                                                                                          |
| Solved −73.3% schedules                             | `docs/results/idea_pilot/schedule_cut75{,_8k,_weighted}.json`                                                                |
| Eval configs (this round)                            | `configs/eval/qwen3_30b_a3b_probe_{reuse_k25_r{10,15,20},router_k25_r10,sched_cut75,sched_router_cut75,schedw_cut75}_*.yaml` |
| Low-rank scorer (SVD/BTT)                            | `src/dynamic_active_param/lowrank_scorer.py`                                                                                 |
| Low-rank tests (16)                                  | `src/dynamic_active_param/tests/test_lowrank_scorer.py`                                                                      |
| Probe output error                                   | `scripts/probe_output_error.py`                                                                                              |
| Probe recall                                         | `scripts/probe_frontier.py`                                                                                                  |
| Probe capture (prerequisite)                         | `scripts/probe_capture.py`                                                                                                   |
| Static floor diagnostic                              | `scripts/actaware_diag_static.py`                                                                                            |
| Activation-aware screen                              | `scripts/actaware_scorer_screen.py`                                                                                          |
| Learned probe experiment                             | `scripts/lowrank_scorer_learned_probe.py`                                                                                    |
| Results (JSON)                                       | `docs/results/btt_dynamic/`, `docs/results/actaware/`                                                                      |
| Figures                                              | `docs/exps/dynamic_active_param/figures/btt_dynamic/`                                                                        |
| Eval configs                                         | `configs/eval/qwen3_30b_a3b_probe_*.yaml`, `configs/eval/qwen3_30b_a3b_dynamic_lowrank_*.yaml`                             |

---

## Prior art

- **Prox (arXiv:2607.27591):** quantized proxy + input sparsity → rank → exact
  compute for SwiGLU FFNs. Same mechanism, 10 dense LLMs, zero MoE. Must cite.
- **SVD-LLM:** activation-aware SVD; the theory is confirmed (consistent +0.035
  gain), the application to channel *selection* is novel and negative.
