# Lightweight channel scoring — how cheap can the `oracle_mag` ranking get?

## The question

`oracle_mag` (and its tied variant `oracle_mag_noW`) is the best selector this repo
has: rank a token's pooled `K·I` expert channels by `g_e·|SiLU(gate_e)⊙up_e|` and
keep the global top-B, and at ρ=0.125 HellaSwag acc_norm is **77.11** against a
dense 78.56 — a 7/8 channel cut for ~1.5pt. But the ranking signal *is* the exact
intermediate, so `gate_proj` and `up_proj` must run at full width just to decide.
Its whole-FFN active cut is therefore only **−29.2%**, and no amount of extra
channel pruning can push it past −33.3%: its kept fraction is `(1+1+ρ)/3 ≥ 2/3`.

So the target for this stage:

> Score channels with something costing **< 10% of one expert `(I,H)` matrix**,
> emit indices, and load only those channels of `up`/`gate`/`down`. At ρ=0.125
> that makes the whole-FFN kept fraction `0.125 + 0.10/3 = 0.158`, a **−84.2%**
> cut. Does it still score like `oracle_mag`?

**Cost model** (units of one expert `(I,H)` fp16 matrix; a dense expert FFN is 3).
Group-wise RTN at `b` bits with group `g` costs `(b + 16/g)/16` per matrix (payload
plus the fp16 group scale); reading only the top `p` fraction of `x`'s coordinates
multiplies that by `p`, because only the matching *columns* are fetched. Both
branches → `c_probe = 2·(b+16/g)/16·p`. The budget is `c_probe ≤ 0.10`.

## Two instruments

**Recall screen** — `scripts/probe_frontier.py`, 1 GPU, minutes, no model load,
reusing the cached captures. Reports index recall and captured score *mass* of the
oracle's top-B, plus the relaxed-candidate cascade. Reproduces the published L46
numbers exactly (`quant_w4` 0.917, `insp0.25_q4` 0.707, `oracle_up` 0.421).

**Output-error ladder** — `scripts/probe_output_error.py` (needs the `_wd`
captures from `scripts/probe_capture.py`, which add `down_proj`). Recall is the
wrong currency: missing a channel whose `W_down` column barely moves the output is
free. What the model feels is `‖y_full − y_kept‖/‖y_full‖` on the MoE block output.
This matters because the repo already has downstream accuracy for `oracle_mag` at
four budgets, so `rel_err → accuracy` can be **fitted** instead of extrapolated:

Averaged over layers 6/22/38/46 (2048 C4 tokens each), which is the right basis
since accuracy is a whole-model quantity:

| selector           | ρ    | rel_err | HellaSwag acc_norm |
| ------------------ | ----- | ------- | ------------------ |
| dense              | 1.0   | 0       | 78.56              |
| `oracle_mag`     | 0.5   | 0.078   | 78.54              |
| `oracle_mag`     | 0.25  | 0.199   | 78.28              |
| `oracle_mag`     | 0.125 | 0.325   | 76.84              |
| `oracle_mag_noW` | 0.125 | 0.326   | 77.11              |
| `oracle_up`      | 0.125 | 0.540   | 71.30              |
| random             | 0.125 | 0.919   | —                 |

Accuracy is flat out to rel_err ≈ 0.20 and then falls. Two slopes are visible and
they differ, which is itself the finding: along the *oracle's own budget ladder*
(ρ=0.25→0.125) accuracy falls at **≈11.5 pt per unit rel_err**, while the step from
`oracle_mag_noW` to `oracle_up` at fixed ρ costs **≈27 pt per unit rel_err**. So
rel_err is not a sufficient statistic — a *mis-selection* at fixed budget hurts
more than the same rel_err produced by an honestly smaller budget. Predictions for
a mis-selecting probe should therefore use the steeper slope, and are what the
in-flight evals are checking.

Even so the instrument does its job: it turns a 4-minute run into an accuracy
bracket, which is why the design search below cost ~1 GPU-hour instead of ~150.

## What wins

Layer 46, 2048 C4 tokens, ρ=0.125. `cB` is the probe cost per the model above;
"FFN cut" is the realized whole-FFN active cut including the probe.

| variant                       | cB              | recall          | mass            | FFN cut           |
| ----------------------------- | --------------- | --------------- | --------------- | ----------------- |
| `oracle_up` (exact up only) | 1.000           | 0.425           | 0.542           | −54.2%           |
| 4-bit, dense x                | 0.516           | 0.917           | 0.990           | −70.3%           |
| 3-bit, dense x                | 0.391           | 0.813           | 0.953           | −74.5%           |
| 4-bit, keep 50%               | 0.258           | 0.836           | 0.962           | −78.9%           |
| 3-bit, keep 50%               | 0.195           | 0.771           | 0.930           | −81.0%           |
| 4-bit, keep 25%               | 0.129           | 0.707           | 0.886           | −83.2%           |
| **3-bit, keep 25%**     | **0.098** | **0.675** | **0.864** | **−84.2%** |
| 3-bit, keep 20%               | 0.078           | 0.645           | 0.838           | −84.9%           |
| 3-bit, keep 10%               | 0.039           | 0.563           | 0.754           | −86.2%           |

**Input sparsity is the cheap axis and bits are the expensive one.** At iso-bytes
0.25, keeping 50% of `x` at 4 bits gives recall 0.836 while 2-bit on dense `x`
gives 0.506. The optimum sits at **3 bits** — 2-bit collapses (0.536 even on dense
`x`), 4-bit spends more than the extra accuracy is worth.

At the budget the goal sets, the winner is **3-bit RTN read on the token's top-25%
coordinates**: `cB = 0.098`. Layer-averaged rel_err at ρ=0.125 for the probe family:

| probe                          | cB              | FFN cut           | rel_err @ρ0.125 |
| ------------------------------ | --------------- | ----------------- | ---------------- |
| `oracle_mag_noW` (reference) | 2.000           | −29.2%           | 0.326            |
| 4-bit, dense x                 | 0.516           | −70.3%           | 0.332            |
| 3-bit, dense x                 | 0.391           | −74.5%           | 0.356            |
| 3-bit, keep 50%                | 0.195           | −81.0%           | 0.373            |
| 4-bit, keep 25%                | 0.129           | −83.2%           | 0.413            |
| **3-bit, keep 25%**      | **0.098** | **−84.2%** | **0.431**  |
| 3-bit, keep 20%                | 0.078           | −84.9%           | 0.456            |
| 2-bit, dense x                 | 0.266           | −78.6%           | 0.498            |
| `oracle_up`                  | 1.000           | −54.2%           | 0.540            |

Two things stand out. **A 4-bit probe on dense `x` is already within 0.006 rel_err
of the exact oracle** at a −70.3% cut — i.e. the ranking problem is fully solved
once you can afford 0.5 of a matrix. And **the goal budget costs 0.105 rel_err**
over the oracle, which the steeper slope puts at ≈2.8pt.

## Measured — HellaSwag 0-shot, acc_norm (stderr ≈ 0.42–0.43)

| config                             | probe cB        | whole-FFN cut     | rel_err | **measured** | ladder predicted |
| ---------------------------------- | --------------- | ----------------- | ------- | ------------------ | ---------------- |
| dense                              | —              | 0%                | 0       | 78.56              | —               |
| `oracle_mag_noW` ρ=0.125        | 2.000           | −29.2%           | 0.326   | 77.11              | —               |
| `probe` 4-bit dense x            | 0.516           | **−70.3%** | 0.332   | **76.95**    | 76.9             |
| `probe` 3-bit keep 50%           | 0.195           | −81.0%           | 0.373   | **76.37**    | 75.97            |
| **`probe` 3-bit keep 25%** | **0.098** | **−84.2%** | 0.431   | **74.56**    | 74.5             |
| `oracle_up` ρ=0.125             | 1.000           | −58.3%           | 0.540   | 71.30              | —               |
| Level-1`pivchol` ρ=0.125        | 0               | −87.5%           | —      | 44.15              | —               |

**The ladder predicted all three measured points to within 0.4pt** (`q3_k50`
predicted 75.97, measured **76.37**), which validates it as a screening instrument:
the probe family's slope is **−24.3 pt per unit rel_err**, fitted consistently from
either measured point (−24.2 from `q4_k100`→`q3_k25`, −24.4 from the
oracle→`q3_k25`). Refitting over all measured fixed-budget points gives **−26.4
pt/unit at R² 0.985** (`scripts/probe_relerr_linearity.py`), which is the slope to
use going forward. Design decisions can therefore be made on a 4-minute run.

**Verdict on the goal.** At a probe costing 9.77% of one matrix, ρ=0.125 gives
**74.56** — **2.55pt below** `oracle_mag_noW`'s 77.11, about 6 stderr, so this is a
real gap, not noise. Parity *is* reachable but needs a ~0.5-matrix probe: the 4-bit
dense-`x` probe scores **76.95 vs 77.11, a −0.16pt difference inside 1 stderr**, at
a **−70.3%** whole-FFN cut against the oracle's −29.2%. Inverting the fitted slope,
staying within 1 stderr of the oracle requires rel_err ≤ 0.344, i.e. `cB ≳ 0.45`,
i.e. a cut no deeper than ≈−71%.

So the frontier is: **−70% cut at oracle parity, −84% cut at −2.6pt.**

**Against the repo's previously measured whole-FFN frontier** (the top-4 × narrowed
stacking table in `q3_30b_dynamic_active.md`, every row of which is an *oracle*
selector):

| whole-FFN cut     | config                                                 | HellaSwag acc_norm |
| ----------------- | ------------------------------------------------------ | ------------------ |
| −66.7%           | top-4 ×`oracle_up` −50% (oracle)                   | 74.02              |
| **−70.3%** | **`sparse_probe` 4-bit/dense x (deployable)**  | **76.95**    |
| −75.0%           | top-4 ×`oracle_up` −75% (oracle)                   | 69.99              |
| **−84.2%** | **`sparse_probe` 3-bit/keep 25% (deployable)** | **74.56**    |

Both probe rows dominate it. At −84.2% the probe matches the old frontier's −66.7%
oracle row, i.e. **+17.5pt of extra cut at equal accuracy**, while being the only
entry that does not read the true per-token intermediate.

## Is ρ=0.125 with a 0.10 probe the best way to spend that budget?

The goal fixes two numbers separately (channels at ρ=0.125, probe ≤ 0.10), but only
their sum `(c_probe + 3ρ)/3` is paid for. Sweeping the split at a fixed total of
0.158 (layer-averaged rel_err, 4 layers × 2048 tokens):

| ρ              | 3-bit input keep | c_probe         | rel_err                  |
| --------------- | ---------------- | --------------- | ------------------------ |
| 0.150           | 0.06             | 0.023           | 0.5866                   |
| 0.140           | 0.14             | 0.055           | 0.4842                   |
| **0.125** | **0.25**   | **0.098** | **0.4305**         |
| 0.115           | 0.33             | 0.129           | 0.4159                   |
| **0.100** | **0.445**  | **0.174** | **0.4148** ← best |
| 0.090           | 0.52             | 0.203           | 0.4223                   |
| 0.075           | 0.64             | 0.250           | 0.4421                   |

The optimum is a broad plateau at ρ ≈ 0.10–0.115 — **spend more on the probe and
keep fewer channels** than the goal's split. The gain is small (0.4305 → 0.4148,
worth ≈0.2–0.4pt), so the requested operating point is nearly optimal, which is
worth knowing. The curve is strongly asymmetric: under-funding the probe is far
more costly (ρ=0.15 → 0.587) than over-funding it, so when in doubt, buy probe.

Note the best split violates the *scorer* budget (0.174 > 0.10) while respecting
the total. Which constraint is real depends on whether the 10% is a memory-traffic
budget (then only the total matters) or a latency/ordering budget for the scoring
pass itself.

## Why the comparison that matters is at matched *total* bytes

`oracle_mag` cannot be run at a deep whole-FFN cut at all: its kept fraction is
`(1+1+ρ)/3 ≥ 2/3`, so **−33.3% is its hard floor** no matter how few channels it
keeps. At the depths this stage targets, the only previously available selectors are
`oracle_up` (`(1+2ρ)/3`) and the router-only Level-1 family, whose decision is
entirely offline (`g_k²·σ_{k,r}`) and so shrinks all three matrices to ρ:

| selector                             | whole-FFN cut at ρ=0.125 | HellaSwag acc_norm |
| ------------------------------------ | ------------------------- | ------------------ |
| `oracle_mag_noW`                   | −29.2%                   | 77.11              |
| `oracle_up`                        | −58.3%                   | 71.30              |
| **`sparse_probe` q3/keep25** | **−84.2%**         | **74.56**    |
| Level-1`pivchol`                   | −87.5%                   | 44.15              |

That is the frame in which the result is strong. At ρ=0.125 the probe is
**+30.4pt over Level-1**, the only prior method that reaches a comparable cut, and
**+3.26pt over `oracle_up`** while scoring at **1/10** of `oracle_up`'s cost and
cutting 26pt more of the FFN. What it does not do is match `oracle_mag` — and
`oracle_mag` is not an available option at these depths in any case.

## What does not work — nine mechanisms, all dominated

Everything here was measured at matched bytes against the row above.

| mechanism                                                                                                                                  | result                                                                                                                                                           | why                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Product quantization** (0.25–1.0 bits/weight)                                                                                     | rel. recon. error**0.78** at 0.5 bits/w, 0.58 at 1.0; recall 0.421–0.600 vs 0.707 for RTN at the same bytes                                               | The weights have no exploitable structure at*any* granularity. Row clusterability: mean cosine to a k-means centroid is 0.078 (k=8) → 0.127 (k=64) against a random baseline of 0.022, i.e. rows are near-mutually-orthogonal and 8-dim subvectors are effectively i.i.d. Gaussian, where PQ cannot beat scalar quantization.                                                               |
| **Hadamard rotation** (± PQ)                                                                                                        | 3-bit 0.818 vs 0.813 unrotated; 2-bit 0.539 vs 0.536                                                                                                             | Rotation fixes*activation* outliers in PTQ; there are none in the weights, and it destroys input sparsity (it spreads `x`'s energy), so it cannot even reach the budget.                                                                                                                                                                                                                   |
| **1-bit sign + per-group scale**                                                                                                     | cB 0.126 → recall 0.546, dominated by 4-bit/keep-25% at cB 0.129 → 0.707                                                                                       | Below ~3 bits the per-channel ordering is gone.                                                                                                                                                                                                                                                                                                                                                |
| **Asymmetric gate/up precision**                                                                                                     | `u2g4` at cB 0.098 → 0.611 (vs 0.675 symmetric); `u4g3`/`u3g4` land on the same frontier                                                                  | The score is a product, so the two factors' relative errors add in quadrature; there is nothing to reallocate.                                                                                                                                                                                                                                                                                 |
| **Norm-weighted coordinate selection** (`                                                                                            | x_i                                                                                                                                                              | ·rms_i(W)`)                                                                                                                                                                                                                                                                                                                                                                                   |
| **Router-adaptive precision** (bytes by router rank, incl. skipping tail experts)                                                    | top-4 experts at 3-bit/keep-50%, cB 0.098 → mass 0.849 vs 0.864 for uniform                                                                                     | The router-rank*mass* shares are `[.34 .23 .16 .11 .07 .04 .03 .02]`, so restricting to the top-4 experts caps mass at **0.839** — below what uniform cheap scoring already achieves. Consistent with the earlier finding that MoE redundancy is not expert-level.                                                                                                                  |
| **Relaxed-candidate cascade** (exact re-rank of λB nominees)                                                                        | λ=1.5 on the 0.098 probe → recall 0.787 at −80.1%; but plain 3-bit/keep-50% gets 0.771 at −81.0%                                                             | The extra`2(λ−1)ρ` of exact reads buys less than spending the same bytes on a better probe.                                                                                                                                                                                                                                                                                               |
| **Closed-form debiasing** of the proxy score                                                                                         | rel_err**0.437** vs 0.369 baseline — much worse                                                                                                           | Per-channel noise is near-uniform (row-norm CV 0.06), and with uniform σ no shrinkage can re-rank; the unbiased estimator`√(û²−σ²)` additionally clamps a large fraction of channels to exactly 0, destroying their ordering outright.                                                                                                                                                |
| **`‖W_down[:,j]‖` weighting** of the probe score                                                                                 | rel_err 0.370 vs 0.369                                                                                                                                           | Same wash as the Q1 ablation found for the exact oracle; the factor's CV is only 0.055.                                                                                                                                                                                                                                                                                                        |
| **`\|gate\|` instead of `SiLU(gate)`**                                                                                             | rel_err 0.647 vs 0.369                                                                                                                                           | The SiLU gate is load-bearing, not a detail.                                                                                                                                                                                                                                                                                                                                                   |
| **Static pre-filter** — exclude channels the oracle rarely picks, so the probe reads fewer rows (a straight multiplier on its cost) | forbidding the bottom**25%** of channels by held-out keep-frequency already loses **12–18%** of the oracle top-B mass; the bottom 50% loses 29–39% | Keep-frequency is nearly uniform: mass retained tracks the surviving fraction almost linearly (q=0.5 → 0.61–0.71 kept). A 1.33× probe saving costs more mass (12–18%) than the entire probe error it would fund (13.6%). This is the same fact the doc's M1 result rests on — the headroom is*per-token*, so there is no static component to bank. `scripts/probe_prefilter_diag.py`. |

The unifying reason: **the top-B set is decided by the fine structure of individual
weight rows, and those rows carry essentially maximal information per weight.**
Low rank (measured dead earlier), cross-expert bases (dead), subvector codebooks
(dead here), and rotations (no-op) all try to find structure that is not there.
Precision — how many bits per weight, on how many coordinates — is the only axis,
and its cost is what it is.

## The already-quantized-serving case — where the storage objection disappears

The standing objection to the whole scheme is that the probe is an *extra* copy of
`up`/`gate` (+13% of expert weights at 3 bits). That objection vanishes if the model
is already served at low precision: the probe then reads the **served weights
themselves**, and the only thing separating it from the exact oracle is input
sparsity. Measured with `--serve-bits 4` (all three matrices quantized first, so
"exact" means the served weights; layers 22+46, 1024 tokens):

| probe           | extra storage  | rel_err | excess vs served oracle | ≈pt | traffic cut vs**4-bit** dense |
| --------------- | -------------- | ------- | ----------------------- | ---- | ----------------------------------- |
| 4-bit, dense x  | none           | 0.3322  | **0.0000**        | 0.00 | −20.8%                             |
| 4-bit, keep 50% | none           | 0.3517  | 0.0195                  | 0.5  | **−54.2%**                   |
| 4-bit, keep 25% | none           | 0.4178  | 0.0856                  | 2.1  | −70.8%                             |
| 3-bit, keep 25% | +50% of served | 0.4421  | 0.1099                  | 2.7  | −74.9%                             |

The first row is the sanity check that the setup is right: RTN is idempotent on its
own output, so a 4-bit probe on dense `x` reproduces the served oracle's selection
*exactly*, excess 0.0000.

**The honest accounting matters here.** Against a 4-bit dense FFN the denominator
shrinks by 3.9×, so the probe's *relative* traffic cost rises and the cuts are less
dramatic than the fp16 numbers: −54.2% rather than −78.9%. What you get in exchange
is that the +13% storage overhead is gone entirely and the ~0.5pt operating point is
much closer to free. Note also that going *below* the serving precision (last row)
re-introduces a storage cost worse than the fp16 case — at 4-bit serving a 3-bit
probe costs +50% of served expert weights — so the probe should be built at the
serving bit-width, not below it.

## Prior art

Prox (arXiv:2607.27591) published quantized-proxy + input-sparsity → rank →
exact-compute for SwiGLU FFNs, with the same cost model, on ten **dense** LLMs and
zero MoE models. The proxy mechanism is theirs and must be cited as such. What is
specific here is the MoE instantiation — the proxy has to compare channels *across*
K different experts through `g_e` on one pooled scale — plus the negative map above.

## Implementation

`src/dynamic_active_param/sparse_probe.py`, registered as criterion
`sparse_probe`. Config knobs under `prune_kwargs.dynamic_alloc.probe`:
`bits`, `group`, `input_keep`, `use_gate`, `lam`. `src/dynamic_active_param/tests/test_sparse_probe.py`
(18 tests) pins the contract; the load-bearing one is that `bits=16, input_keep=1.0`
reproduces `oracle_mag_noW` **bit-exactly**, and that a candidate pool of `K·I`
makes the cascade reproduce it too.

Storage caveat, stated plainly: the probe is an *extra* b-bit copy of `up`/`gate`,
so total memory **rises** by 13.0% of expert weights at 3 bits even as per-token
traffic falls by 84.2%. The special case that removes the objection — a model
already served at 3–4 bits, where the probe *is* the served weight — is untested.

## Status and what to do next

Configs: `configs/eval/qwen3_30b_a3b_probe_{q3_k25,q4_k100,q3_k50,q3_k25_lam15}_875_hellaswag.yaml`,
plus `probe_q3_k445_r90_hellaswag` (the refined split) and `probe_q3_k25_875_mmlu`.
HellaSwag `q3_k25` / `q4_k100` / `q3_k50` are **done** (74.56 / 76.95 / **76.37**),
as is `q3_k25` MMLU (**77.65**).

**The storage objection is now moot** — see the reuse regime in
`../efficient_scorer.md`: at `bits >= 16` the probe aliases the served weights, so
extra storage is exactly 0 and a sub-serving-precision proxy is strictly dominated
on both cost *and* accuracy. Do not build the probe below serving precision.

Open, in value order:

1. ~~**MMLU at the goal point.**~~ **Done: 77.65** (dense 80.91). The hypothesis
   that per-token activation matters *more* on MMLU is **not supported** — the gap
   to the full-width oracle is 1.79pt on MMLU vs 2.55pt on HellaSwag, i.e. MMLU
   degrades *less*. The −3.26pt total loss at −72.9% is smaller than HellaSwag's
   −4.00pt.
2. **LoRA recovery.** A 2.55pt gap at −84.2% is exactly the size the repo's
   stage-2 LoRA fine-tune closes elsewhere. This is the most likely route to
   "parity at −84%", and the pipeline already exists.
3. ~~**A HellaSwag eval in the zero-extra-storage setting.**~~ **Done** — the reuse
   regime (`bits >= 16`, probe = served weight) is measured across three budgets in
   `../efficient_scorer.md`: **−70.8% → 76.47 / −75.0% → 74.64** (with `router`
   input allocation, itself worth +0.58pt). Zero extra storage, and it beats
   `oracle_up` by +3.34pt while cutting 16.7 more points of the FFN.
   *Superseded text below, kept for the reasoning:* a HellaSwag eval in the
   served-at-4-bit setting, to confirm the ~0.5pt
   estimate for the zero-extra-storage operating point (4-bit probe, keep 50%).
   That is the cleanest deployment story available and it is currently backed only
   by the rel_err ladder.
4. **Wall-clock.** Everything here is an active-parameter/byte-traffic claim. A
   gathered-expert microbenchmark is needed before any latency claim.
