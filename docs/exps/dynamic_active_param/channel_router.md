# Channel-Level Router for Sparse MLP Activation — results

Implementation of `plan/channel_router.md`. Model **Qwen3-30B-A3B-Thinking-2507**
(48 MoE layers, E=128 experts, I=768, top-K=8, H=2048, so D = K·I = 6144 activated
channels per token). Calibration data c4: 1.05M tokens/layer for the two deep-dive layers
(22 = mid, 46 = deep) and 0.52M tokens/layer for all 48 layers. End metric: wikitext2
perplexity, 64 windows × 2048 tokens, **all 48 MoE layers masked**, paired across specs.

Code: `src/channel_router/` (package + tests), `scripts/channel_router/` (studies,
training, eval, benchmark). Raw results: `docs/results/channel_router/`. Phase-0 exit
doc: `docs/results/channel_router/phase0/summary.md`.

---

## 0. Verdict

**The plan's goal is not met, and the measurements say why with three independent
numbers.**

1. **The target budget is infeasible for any selector.** At the plan's k=768 (ρ=0.125)
   the *oracle* mask — exact `imp` top-B, an unreachable ceiling — already costs
   **+6.06% PPL**. The §0.2 gate is ΔPPL < 1%, so k is re-derived: **k=1536 (ρ=0.25),
   where the oracle costs +0.74%**.
2. **The tolerance is ~20× tighter than the plan assumed.** Controlled degradation of the
   oracle mask gives **≈ +575 to +610 % excess PPL per unit of importance-mass lost**:
   losing 3% of the mass costs +6.7% PPL. Near-lossless at a fixed budget needs
   mass-recall ≳ **0.99**, not the plan's 0.95.
3. **The achievable mass-recall of this architecture class is 0.63 (model mean), 0.75
   (best layer).** So the trained router at the oracle's own budget costs **+84% PPL**,
   and needs 3× the budget (4608 of 6144 channels, a 25% cut) to get to **+3.9%**.
   P1's free-embedding *ceiling* — the best any rank-r per-token embedding scorer could
   do, with the embedding solved per token rather than predicted — saturates at 0.79
   (layer 46) and **0.62 (layer 22)** even at r=256, so this is a property of the problem,
   not of the training.

**What the router does deliver.** It is the best cheap selector measured here: it beats
the free static prior, every training-free weight-derived scorer, LSH, product-key, a
25.7M-parameter VQ table and a 51.5M-parameter Deja-Vu MLP — at 9.7M parameters (1.6% of
the layer's stored FFN) and 1.44 MFLOP/token (2.2% of the FLOPs the pruning saves). Both
week-3/4 gates of §3.3 pass. And the mechanism has a clean layer signature: **the router
buys real per-token signal in the deep layers and essentially nothing in the middle
band**, where it ties the zero-parameter static prior.

---

## 1. Phase 0 — all five priors tested, four are false

Full tables: `docs/results/channel_router/phase0/summary.md`.

| study | measurement | plan's rule | verdict |
| --- | --- | --- | --- |
| §0.2 oracle ΔPPL | +6.06% @ ρ=0.125; +0.74% @ ρ=0.25; −0.25% @ 0.375 | < 1% at k=768 | **fail → k re-derived to 1536** |
| §0.3 calibration | 575–610% excess PPL per unit mass lost (R² 0.94/0.96) | — | mass-recall ≥ 0.99 for near-lossless |
| P1 logistic rank | free-U ceiling 0.79 (L46) / 0.62 (L22) @ r=256; wsvd 0.37/0.57 | r* ≤ 128 @ ≥95% mass | **fail on 95%**; free embeddings justified by ≥2× over SVD |
| P2 gate sufficiency | gate-only 0.84 (L46) / 0.75 (L22) | ≥ 0.97 ⇒ distil gate | **fail → target stays the full bilinear `imp`** |
| P3 input screening | 64 global coords reach 0.535 of full-input mass-recall | ≥ 0.90 ⇒ passthrough | **fail** (kept as an ablation: worth +1.8 pts anyway) |
| P4 static/dynamic | 6.25% hottest channels hold 24% of the mask; no knee | knee in coverage | **no hot set** |
| P5 tiles | top-10 of 64 co-activation tiles hold 0.687 of the mass | ≥ 0.95 ⇒ 2-level | **fail → single level** |
| P6 temporal | adjacent-position mask IoU 0.11–0.17 | ≥ 0.70 ⇒ reuse variant | **fail → dropped** |

Two results worth stating independently of the router:

- **The mask is concentrated nowhere.** No static subset, no tile and no temporal
  neighbour carries it: 58 of a token's 64 tiles are touched, the 6.25% hottest channels
  hold a quarter of the mass, and consecutive positions share ~1/8 of their masks
  (reuse-previous-mask mass-recall 0.33 — *below* the static prior). This is the
  channel-level version of the repo's expert-level finding: the exploitable structure is
  per-token, and it must be predicted, not looked up.
- **Co-activation tiles beat weight-similarity tiles.** Balanced Sinkhorn k-means on the
  spectral embedding of the co-activation matrix (DOT-MoE's balanced-assignment machinery
  reused offline with a co-activation cost) concentrates the mask better than
  MoEfication-style `W_g`-row clustering at every tile size: **+9.3 mass points** at 8
  tiles/expert (0.687 vs 0.594), +7.5 at 16, both far above a balanced random split. It
  just does not concentrate enough to build a two-level router on.

---

## 2. Architecture as built, and its cost

```
phi(h)   = concat[Q^T h, h[S_out]]         Q = Σ^{-1/2}V ∈ R^{2048×32}, |S_out| = 16
score_i  = log|c_i·phi| + log|c2_i·phi| + beta_i + log g_e
select   = per-token top-B of score        (no tiles, no hot set — P4/P5 said no)
```

- Channel embeddings are keyed by **physical** channel (`E·I = 98304`), not by activated
  slot: slot 3 is a different expert for every token. Only the K selected experts' blocks
  are touched online.
- `beta_i` is initialized to `log‖W_d[:,i]‖`, the oracle's own per-channel constant.
- `log g_e` is in the oracle's definition and free at inference, so **every baseline gets
  it too**; withholding it would manufacture a win.

**§1.2 accounting.** 9.667M params = **1.60% of the layer's stored FFN** (3·E·I·H = 604M);
1.44 MFLOP/token = **2.18% of the FLOPs the pruning saves**. Both inside the plan's
2%/3% standard. Against the *activated* FFN (37.7M) the parameters are 25.6% — the honest
way to say this is a table, not a matmul: it costs memory bandwidth, not arithmetic.
Per-token scorer traffic is 590K values = **1.56% of the activated FFN's weight traffic**.

### Stage A reproduces the whitened-SVD baseline exactly

The plan's checkpoint standard is "within 0.5 pt"; measured **within 0.06 pt** (init
recall 0.2267 vs wsvd_r32 0.2274, mass 0.3248 vs 0.3251; layer 46, ρ=0.125). Every later
gain is attributable to training.

### The scoring head: SiLU asymmetry is real, but free parameters wash it out

The oracle is `g_e · |silu(W_g h)| · |W_u h| · ‖W_d[:,j]‖`. Four heads, layer 46,
short-budget Stage B (262K tokens, 3 epochs), test mass-recall:

| head | init (training-free) | trained @ρ=0.125 | trained @ρ=0.25 |
| --- | --- | --- | --- |
| `abs` — log\|a·φ\| | 0.268 | 0.605 | — |
| `bilinear` — log\|a·φ\| + log\|u·φ\| | 0.325 | **0.627** | **0.709** |
| `linear` — signed a·φ | 0.562 | 0.651 | 0.695 |
| `swiglu` — log·silu(a·φ) + log\|u·φ\| | **0.677** | — | 0.700 |

- **A large negative gate pre-activation is a dead channel, not a strong one**, so `|·|`
  on the gate factor is actively wrong: `abs` is the worst head, and the plain *signed*
  `linear` head's structural init beats it by **+29 points**.
- The `swiglu` head is the rank-r truncation of the true oracle and has by far the best
  **training-free** init (0.677 at ρ=0.25 — above the static floor 0.587 *and* above a
  trained whitened-SVD). Use it if you want a zero-training selector.
- Once the embeddings are free, the algebraic form washes out and `bilinear` ends
  marginally ahead — so `bilinear` is the default.
- Gotcha worth recording: `swiglu` with `|silu(a)|` **trains worse than its own init**
  (0.563 < 0.658) because `log|silu(a)|` has a singularity at `a=0`; flooring with
  `clamp_min` instead fixes it (0.677 init → 0.700 trained).

---

## 3. Stage B / Stage C

Layer 46, full budget (918K train tokens, 6 epochs, boundary-focused margin ranking with
Δ=256, w_fn=3), test mass-recall:

| budget | init (= whitened SVD) | static prior (free, with `g`) | **Stage B** | free-U ceiling (P1) | oracle |
| --- | --- | --- | --- | --- | --- |
| ρ=0.125 | 0.325 | 0.568 | **0.680** | 0.789 | 1.0 |
| ρ=0.25 | 0.456 | 0.671 | **0.747** | — | 1.0 |

- vs whitened SVD at equal rank: **+35.5 / +29.1 points** → passes the ≥ +3 gate.
- vs the static prior: **+11.2 / +7.6 points**. Against the *weakest* static form (no
  `g`) it is +19.4/+16.0 and the plan's ≥ +10 gate passes at both budgets; against the
  strongest (free) static form it passes at ρ=0.125 (+11.2) and **fails at ρ=0.25
  (+7.6)** — the static prior gets relatively stronger as the budget grows.
- The router at r=32 reaches 86% of P1's free-U ceiling at r=256, i.e. a linear feature
  map captures most of what any rank-r embedding scorer could.

**Stage C** (Sinkhorn soft top-k + STE, ε annealed 1.0→0.05, block-output-error objective
instead of the plan's logit KL — a per-layer KL would need a 30B backward per step; the
rationale and the substitution are documented in the script) improves the deciding
surrogate: layer 46, ρ=0.125, 2000 steps, **block-output rel_err 0.5584 → 0.4616
(−17.3%)** and test mass-recall **0.672 → 0.710 (+3.8 pts)**, with the oracle at 0.2792.
So the plan's Stage-C standard is met in the surrogate; it was not re-run for all 48
layers (5.6 GPU-h), so the full-model ΔPPL below is Stage-B only.

### §3.2 ablations (layer 46, ρ=0.125, short budget, test mass-recall)

| row | mass-recall | Δ vs reference |
| --- | --- | --- |
| reference (`bilinear`, m=16, Stage-A init, margin loss) | 0.6268 | — |
| head `linear` | 0.6507 | +0.024 |
| head `abs` | 0.6051 | −0.022 |
| no outlier passthrough (m=0) | 0.6085 | −0.018 |
| random init instead of Stage A | 0.5976 | −0.029 |
| no static bias `beta` | 0.6246 | −0.002 |
| loss: BCE over all channels | 0.6227 | −0.004 |
| loss: listwise (softmax CE on the boundary window) | 0.5412 | −0.086 |

The **Stage-A init is worth +2.9 points at equal training budget** and is free; the
boundary-focused pairwise margin beats BCE slightly and listwise clearly; the static bias
is nearly irrelevant once the embeddings are trained (they absorb it).

---

## 4. §1.3 baseline table (one protocol, identical budgets)

ρ=0.125, test slice, per-token top-B; `s1.5` = the same scorer given 1.5× the budget
against the same reference. `rel_err` is the block-output error (lower is better).

**Layer 46**

| method | params | recall | mass-recall | rel_err | s1.5 mass | s1.5 rel_err |
| --- | --- | --- | --- | --- | --- | --- |
| oracle (ceiling) | 0 | 1.0000 | 1.0000 | 0.2945 | 1.183 | 0.2200 |
| **router (ours)** | 9.67M | **0.5021** | **0.6796** | **0.5509** | 0.855 | 0.4639 |
| Deja-Vu MLP h=512 | 51.48M | 0.4197 | 0.5793 | 0.6315 | 0.755 | 0.5433 |
| static by frequency | 0 | 0.4219 | 0.5680 | 0.6569 | 0.748 | 0.5755 |
| VQ table, 256 centroids | 25.69M | 0.4176 | 0.5662 | 0.6543 | 0.739 | 0.5835 |
| `oracle_up` (reads full-width `up`) | 0 | 0.4339 | 0.5386 | 0.6795 | 0.737 | 0.5794 |
| LSH SimHash 64 bit | 0 | 0.2425 | 0.4056 | 0.7777 | 0.573 | 0.6908 |
| product-key | 0.08M | 0.2446 | 0.3836 | 0.8067 | 0.551 | 0.7212 |
| whitened SVD r=32 | 0 | 0.2314 | 0.3237 | 0.8395 | 0.468 | 0.7765 |
| plain SVD r=32 | 0 | 0.2320 | 0.3221 | 0.8638 | 0.455 | 0.8160 |
| random projection r=32 | 0 | 0.1637 | 0.2788 | 0.8806 | 0.403 | 0.8294 |

**Layer 22** — same ordering except that `oracle_up` (which needs a full-width `up_proj`)
jumps to 0.7754 while the router gets 0.5340 and the static prior 0.4835. The information
is in `h`; a rank-32 predictor cannot extract it in the middle band.

Reads: the router beats **every** learned and training-free baseline on both metrics and
at every slack, including a Deja-Vu-style MLP with **5.3× more parameters** (+10.0 mass
points at L46, +8.1 at L22) — the plan's "is our structure better than a generic
predictor" question answers *yes*. Whitening is worth +0.002/+0.032 mass over plain SVD
(small but consistent), and a random projection of the same rank is 4.5 points worse than
SVD, so the basis matters.

---

## 5. The end metric: full-model ΔPPL and the frontier

All 48 MoE layers carry their own router (48 Stage-B runs, ρ=0.25, 262K tokens each).
wikitext2, 64×2048 tokens, dense PPL **7.0429**.

| spec | channels kept / 6144 | mask mass-recall | PPL | ΔPPL |
| --- | --- | --- | --- | --- |
| dense | 6144 | — | 7.0429 | — |
| oracle | 1536 (ρ=0.25) | 1.000 | 7.0953 | **+0.743%** |
| oracle | 3072 (ρ=0.50) | 1.000 | 7.0173 | **−0.363%** |
| router, slack 1.0 | 1536 | 0.632 | 12.9750 | +84.23% |
| router, slack 1.5 | 2304 | 0.826 | 9.3937 | +33.38% |
| router, slack 2.0 | 3072 | 0.987 | 8.2095 | +16.56% |
| router, slack 3.0 | 4608 | 1.242 | 7.3148 | **+3.861%** |

(mass-recall > 1 at slack ≥ 2 because the reference is the ρ=0.25 oracle's 1536 channels.)

- The router's mask quality **transfers from c4 to wikitext2 unchanged** (mean mass-recall
  0.6315 on the eval stream vs 0.6353 on held-out c4) — §3.1.4 robustness, for free.
- **Layer profile of the difficulty** (mass-recall per layer, slack 1.0): 0.62 at the
  embedding end, a trough of **0.576 at L24**, then a monotone climb to **0.776 at L47**.
  The middle band is where prediction fails, matching the sparse-probe study's finding
  that L17–L30 are the expensive layers.
- At equal *channel count* the router is 17–85 points of PPL behind the oracle, and it
  needs 3× the oracle's budget to get within 4%.

### Where that puts it on the used-parameter frontier

The router decides from `h` alone, so all three expert matmuls can be gathered: realized
weight traffic = ρ_eff + 1.56% (the scorer). `oracle_mag`-style criteria must compute
`gate` and `up` at full width, so their traffic floor is `(1+1+ρ)/3 ≥ 0.667`.

| method | channels kept | realized read fraction | ΔPPL |
| --- | --- | --- | --- |
| router | 0.25 | 0.266 | +84.2% |
| router | 0.375 | 0.391 | +33.4% |
| router | 0.50 | 0.516 | +16.6% |
| router | 0.75 | 0.766 | +3.9% |
| `oracle_mag` (ceiling) | 0.25 | 0.750 | +0.7% |
| `oracle_mag` (ceiling) | 0.50 | 0.833 | −0.4% |

So: **below the 0.667 traffic floor the router is the only one of these that can operate
at all, but its quality there (+16.6% PPL at 0.516) is not usable**, and above the floor
`oracle_mag` dominates it (0.750 traffic at +0.7% vs 0.766 at +3.9%). Compared with the
repo's `sparse_probe` — a quantized, input-sparse proxy that *reads* low-precision weights
(read ≈ 0.5 at p=0.5/ρ=0.125, predicted HellaSwag 76.6 vs dense 78.56) — the router is
dominated at comparable traffic. That is the same conclusion the low-rank scorer study
reached, now for a *learned* predictor: **bytes spent on precision beat bytes spent on
rank**, and a scorer that never reads the weights cannot reach the ≥0.99 mass-recall that
near-losslessness demands.

---

## 6. §3.1.3 wall-clock — no FLOP-only claims

One real MoE block, bf16, A100-40GB, ρ=0.25, layer 46. `pregathered` rows exclude the
gather (the upper bound a fused kernel could reach); `router_only` is today's per-expert
PyTorch loop.

| batch | dense | sparse_gather | sparse_pregathered | router loop | router pregathered | net speedup (loop / pregathered) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.306 ms | 1.850 ms | 1.008 ms | 2.959 ms | 0.333 ms | 0.33× / **0.97×** |
| 4 | 5.084 ms | 7.489 ms | 4.036 ms | 9.111 ms | 0.347 ms | 0.39× / **1.16×** |
| 16 | 20.470 ms | 29.780 ms | 16.105 ms | 22.539 ms | 0.347 ms | 0.53× / **1.24×** |

Reads: (i) `index_select`-based gathering is **slower than dense** (0.68–0.71×) because it
copies the weights it reads; (ii) the arithmetic-only upper bound is 1.24–1.30×, not the
4× the FLOP count suggests, because a 6144-channel FFN at batch ≤16 is launch- and
bandwidth-bound, not FLOP-bound; (iii) the router's *arithmetic* is negligible (0.35 ms,
flat in batch) but its per-expert Python loop costs 2.9–22.5 ms, so a fused
gather-and-matvec kernel is a prerequisite, not an optimization. **§3.3's week-7 gate
(end-to-end speedup > 1) is met only in the pregathered projection at batch ≥ 4, not by
any implementation measured here.**

---

## 7. Gate-by-gate scorecard

| gate (plan §3.3) | result |
| --- | --- |
| W1–2: oracle ΔPPL < 1% at k=768 | **fail** (+6.06%); k re-derived to 1536 |
| W1–2: P1 r* ≤ 128 at ≥95% mass-recall | **fail** (ceiling 0.79 @ r=256, L46; 0.62, L22) |
| W3–4: beat static baseline by ≥ 10 mass pts | **pass** vs static-without-`g` at both budgets; vs the strongest static form, pass at ρ=0.125 (+11.2), fail at ρ=0.25 (+7.6) |
| W3–4: beat whitened SVD by ≥ 3 pts | **pass** (+35.5 / +29.1) |
| W5–6: Stage C ≤ Stage B error | **pass** in the surrogate (rel_err −17.3%, mass +3.8 pts) |
| W5–6: full-model ΔPPL < 2% at k=768 | **fail** (+84% at k=1536; +3.9% at k=4608) |
| W7: end-to-end speedup > 1 | **partial** — 1.16–1.24× projected with a fused kernel, 0.68× as implemented |

---

## 8. What would have to change

- **The premise that a per-token top-k of the *exact* importance is nearly free is false
  at aggressive budgets.** It is true at ρ ≥ 0.375 and breaks fast below: the accuracy
  lives in the tail of the importance distribution, and the tail is exactly what a cheap
  predictor cannot rank.
- **Cheap prediction from `h` is layer-local in value.** Deep layers (L37–L47) are
  predictable (0.66–0.78 mass-recall, clearly above the free static prior); the middle
  band (L11–L26) is not (0.58–0.60, tied with the static prior). A per-layer *policy* —
  router where it pays, static or full width where it does not — is the obvious next
  experiment and is cheap to run with the artifacts in `docs/results/channel_router/`.
- **If a router is to reach ≥0.99 mass-recall it must read some of the weights.** The two
  measured families that do (`sparse_probe`'s quantized proxy, `oracle_up`'s exact
  `up_proj`) dominate every weight-free predictor at comparable traffic. The interesting
  hybrid is a router that *nominates* a candidate set and a low-precision probe that
  refines it, which is the cascade already implemented as `probe.lam` in
  `src/dynamic_active_param/sparse_probe.py`.

---

## 9. Deviations from the plan, and what was not run

Stated explicitly so the tables are not read as more than they are.

- **§2 Stage C objective.** The plan asks for `KL(full ‖ masked)` on the next-token
  distribution. For a per-layer router that needs a 30B forward *and backward* per step;
  this implementation minimizes the block-output error the KL is driven by (documented in
  `train_stage_c.py`). Stage C was run on layer 46 only, so §5's full-model numbers are
  Stage-B routers.
- **§2.4 conformal budget calibration** is not implemented as a separate calibration step.
  The slack sweep in §5 measures the same object directly — the budget multiplier needed
  for a given mask quality — and given the measured 600%-per-unit-mass slope the useful
  statement is the frontier, not a per-token slack rule. A per-token rule remains untested.
- **§3.1.2 downstream tasks: HellaSwag (0-shot) and MMLU (5-shot) are running** at the three
  frontier points, full test sets, all 48 layers, via the standard eval path
  (`configs/eval/qwen3_30b_a3b_channel_router_keep{25,50,75}_{hellaswag,mmlu}.yaml`). The
  other 1–3 tasks of the plan's "3–5 downstream tasks" were skipped: these two are the pair
  every other method in this repo is measured on, so they are the ones that place the router
  on the existing frontier table. Reference rows to compare against (same protocol, same
  model): dense **HS 78.56 / MMLU 80.91**; `oracle_mag` at ρ=0.25 **78.28 / 80.53** and at
  ρ=0.5 **78.54 / 80.22**; Level-1 `pivchol` at ρ=0.25 **63.60 / 70.81**; `sparse_probe`
  3-bit/keep-25% at ρ=0.125 **74.56 / —**. The informative question is whether the router at
  the *same channel budget* (keep25) lands above Level-1's 63.60 — PPL is much more
  sensitive than these benchmarks, so a +84% ΔPPL does not imply a collapse here.
- **§0.2's `imp`/`mask_idx` on-disk format** was replaced by "store `h` + the weights,
  recompute the oracle" (rationale in `collect_activations.py`): 24 GB/layer of `imp`
  versus a sub-second recompute, and one code path for the importance definition.
- **Two layers, not all 48, for the deep-dive studies**; all 48 for the end metric, trained
  at a smaller per-layer budget (262K tokens, 4 epochs) than the deep-dive (918K, 6
  epochs). Layer 46 gets 0.747 mass-recall at the large budget vs 0.706 at the small one,
  so the full-model numbers would improve by ~0.04 mass-recall with more training — far
  less than the gap to the ≥0.99 that near-losslessness requires.
- **Captured activations (~110 GB) live on the GPU boxes only**; the repo carries the
  JSONs, figures and router artifacts' reports, not the tensors.

## Reproduction

```bash
# 1. data (one model load per token stream)
python scripts/channel_router/collect_activations.py --layers 22,46 --tokens 1048576
# 2. Phase 0
python scripts/channel_router/p1_logistic_rank.py --layers 46,22
python scripts/channel_router/p{2,3,4,5,6}_*.py --layers 46,22
python scripts/channel_router/ppl_ladder.py --specs "dense,oracle:0.5,...,degrade:0.25:0.4"
python scripts/channel_router/phase0_summary.py
# 3. Stage A+B, Stage C, baselines, benchmark
python scripts/channel_router/train_stage_b.py --layers 46,22 --ratio 0.25
python scripts/channel_router/train_stage_c.py --ckpt <stage_b.pt> --layers 46
python scripts/channel_router/eval_protocol.py --router-ckpt <stage_b.pt>
python scripts/channel_router/bench_sparse_ffn.py --router-ckpt <stage_b.pt>
# 4. all 48 layers + the end metric
bash scripts/channel_router/run_all_layers.sh <group> 6 0.25       # x6 in parallel
python scripts/channel_router/merge_router_artifacts.py --glob '...g*.pt' --out router_all.pt
python scripts/channel_router/ppl_ladder.py \
  --specs "dense,oracle:0.25,router:0.25:1.0,router:0.25:1.5,router:0.25:2.0,router:0.25:3.0" \
  --router-ckpt router_all.pt
```

Downstream (lm-eval-harness) evaluation of a trained router is wired into the standard
eval path: `configs/eval/qwen3_30b_a3b_channel_router_25_hellaswag.yaml`
(`prune_kwargs.dynamic_alloc.criterion: channel_router`).

Unit tests: `src/channel_router/tests/` — 12 tests, the load-bearing ones being that
`data.oracle_scores` reproduces the deployed `oracle_mag` keep-mask bit-for-bit and that
the Sinkhorn top-k has an exact budget and a finite-difference-correct gradient.
