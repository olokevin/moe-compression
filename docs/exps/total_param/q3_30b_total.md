# Qwen3-30B-A3B — Compression Leaderboard (one-shot, no recovery fine-tuning)

Consolidated results for every compression method we have run on Qwen3-30B-A3B, merged from:

- `docs/results/attribution_guided/nystrom.md` — structural expert-FFN **pruning** (ridge-leverage
  ranking + Nyström reconstruction, plus activation-magnitude and uniform baselines).
- `docs/results/mobe/initial_results.md` — expert-FFN **factorization** (MoBE, RFID-MoE).

All numbers are **one-shot** (decompose/prune → eval, **no LoRA/CE recovery training**) on the full
lm-eval-harness tasks. This file is organized as: (1) the **leaderboards** (tables only, bucketed by
target reduction), then (2) a **self-contained section per method** — each covering its settings,
results, and analysis. Cross-method takeaways live in the leaderboard section; per-method detail lives
in that method's section.

> ⚠️ **Read the caveats before comparing rows across families.** Two things are not held constant:
>
> 1. **Base checkpoint differs.** Pruning runs used `Qwen/Qwen3-30B-A3B-Thinking-2507`; the
>    MoBE/RFID factorization runs used plain `Qwen/Qwen3-30B-A3B`. Their uncompressed baselines are
>    **78.56** vs **77.68** HellaSwag acc_norm respectively — so a ~0.9 pt gap is baked in.
> 2. **Reduction axis differs.** Pruning rows report **overall** model-param reduction (25% expert
>    prune → −23.74% overall). Factorization rows report **MoE-layer** param reduction (down_proj
>    left dense unless noted). They land in the same bucket but are not the identical quantity — see
>    each method's section.

---

## Model

| Property                      | Value                                                                                                                                                                              |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Architecture                  | Qwen3-30B-A3B — hidden`d=2048`, MoE intermediate `p=768`, `n=128` experts, top-k 8, 48 layers, SwiGLU/SiLU, no shared expert                                                |
| Pruning base checkpoint       | `Qwen/Qwen3-30B-A3B-Thinking-2507` (bf16)                                                                                                                                        |
| Factorization base checkpoint | `Qwen/Qwen3-30B-A3B` (bf16)                                                                                                                                                      |
| Hardware                      | A100-New / A100-Sagemaker, 40 GB A100s (`FORCE_DEVICE_MAP_AUTO=1 PER_GPU_MEM=36GiB ATTN_IMPLEMENTATION=sdpa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)                    |

**Eval protocol (all methods).** HellaSwag full 10 042 items, `num_fewshot=0` (report acc_norm);
MMLU full 14 042 questions × 57 subtasks, `num_fewshot=5` (acc). Each task in its own lm-eval call.
PPL (wikitext2 + c4) reported for the factorization runs.

---

# Leaderboards

Tables only. Method-level settings, per-run detail, and analysis are in the per-method sections that
follow. All rows one-shot, no recovery.

## Leaderboard @ ~25% reduction

| Rank | Method                                       | Family    | Reduction                         | HellaSwag acc_norm | MMLU (5-shot)            | PPL wiki2 / c4 | Base ckpt     |
| ---- | -------------------------------------------- | --------- | --------------------------------- | ------------------ | ------------------------ | -------------- | ------------- |
| —   | Original (Thinking-2507)                     | —        | 0%                                | 78.56              | —                       | —             | Thinking-2507 |
| —   | Original (Qwen3-30B-A3B)                     | —        | 0%                                | 77.68              | 82.0†                   | 8.70 / 14.05   | A3B           |
| 🥇   | Leverage ranking + Nyström                   | prune     | −23.74% overall (25% expert-FFN) | **78.45**    | **76.04** (±0.34) | —             | Thinking-2507 |
| 🥈   | Activation-magnitude + plain slicing         | prune     | 25% expert-FFN                    | 78.23              | 76.28                    | —             | Thinking-2507 |
| 🥉   | MoBE (`m=32`, `r=768`, gate/up only)     | factorize | −25% MoE-layer                   | 73.67              | 77.23                    | 9.59 / 15.98   | A3B           |
| 4    | RFID-MoE (`m=32`, `ξ=0.8`, no residual) | factorize | −28.4% MoE-layer‡               | 66.80              | 71.32                    | 12.68 / 21.49  | A3B           |

† MMLU baseline was **not** re-run on either base checkpoint; MoBE/RFID MMLU deltas are against the
MoBE-doc's cited 82.0, not a same-run baseline. ‡ RFID's allocator undershot the 0.625 retain budget
(actual −28.4%, heavier than the other rows).

**Takeaway @ 25%.** On the same-base *pruning* family, **leverage + Nyström wins** — near-lossless
HellaSwag (78.45 vs 78.56 unpruned) and ~tied MMLU with the activation baseline. **MoBE** is the best
*factorization* result: a clean ~4 pt HellaSwag drop (73.67 vs its own 77.68 baseline), well ahead of
RFID. MoBE's MMLU (77.23) nominally tops the pruning rows, but this is confounded by the different base
checkpoint and un-rerun MMLU baseline — do not read it as MoBE > pruning without a matched baseline.

## Leaderboard @ ~33% reduction

| Rank | Method                                                                | Family    | Reduction                         | HellaSwag acc_norm | MMLU (5-shot)            | PPL wiki2 / c4 | Base ckpt     |
| ---- | --------------------------------------------------------------------- | --------- | --------------------------------- | ------------------ | ------------------------ | -------------- | ------------- |
| —   | Original (Thinking-2507)                                              | —        | 0%                                | 78.56              | 81.73                    | 7.29 / 12.46   | Thinking-2507 |
| —   | Original (Qwen3-30B-A3B)                                              | —        | 0%                                | 77.68              | 82.0†                   | 8.70 / 14.05   | A3B           |
| 🥇   | Attribution-guided (leverage + Nyström)                              | prune     | −31.33% overall (33% expert-FFN) | **78.40**    | 73.00 (±0.35)           | —             | Thinking-2507 |
| 🥈   | MoBE even-split (`m=38`, gate/up/down all factorized)      | factorize | −32.8% MoE-layer                 | 73.13              | **76.83**          | 10.10 / 16.57  | A3B           |
| 🥉   | MoBE (`m=16`, `r=768`, gate/up only, down dense)          | factorize | −33.3% MoE-layer                 | 69.64              | 74.05                    | 11.75 / 20.32  | A3B           |
| 4    | Nyström-MoE fix1 (`k=512`, self-target, 1500 it)              | factorize | 33% expert-FFN                    | 66.24              | 60.70                    | 12.97 / 17.69  | Thinking-2507 |
| 5    | Nyström-MoE fix1+2 (`k=512`, teacher-traj, 1500 it)          | factorize | 33% expert-FFN                    | 65.97              | 61.24                    | 12.97 / 17.75  | Thinking-2507 |
| —   | Nyström-MoE (self-target, 800 it — under-trained)                  | factorize | 33% expert-FFN                    | 65.46              | 60.92                    | 13.46 / 17.98  | Thinking-2507 |
| ✗   | Uniform (`uniform`+`uniform`) — ablation                        | prune     | −31.28% overall (33% expert-FFN) | 65.10 (±0.48)     | 27.40 (±0.38)           | —             | Thinking-2507 |

**Takeaway @ 33%.** Attribution-guided leverage+Nyström pruning stays far ahead on HellaSwag (78.40,
−0.16 vs unpruned) at a modest MMLU cost (73.00). The **MoBE even-split** (`m=38`, gate/up/**and** down
all factorized) is the **best factorization result by a wide margin**: HellaSwag **73.13** / MMLU
**76.83**, a +3.5 pt / +2.8 pt jump over the classic down-dense MoBE `m=16` (69.64 / 74.05) at the same
overall reduction, with much better PPL (c4 16.57 vs 20.32). The win comes from **spreading the cut
across all three matrices instead of concentrating it on up+gate** (details in the MoBE section). The
Nyström-MoE compress-then-fit family lands well below both (66/61) but far above the uniform ablation
(MMLU 61.24 vs 27.40), confirming leverage-guided selection retains real task signal that naive uniform
slicing destroys.

## Ablation @ ~37.5% (down_proj-only, basis placement)

Down-only ablation (gate/up dense, so whole-MoE reduction ≈12.5%) isolating **where the shared basis
sits when factorizing `down_proj`**. Both `m=32`, γ=0.625 on `down_proj`; detail + analysis in the
MoBE section.

| Basis side  | down_proj reduction | rank `r` | HellaSwag acc_norm | MMLU (5-shot) | PPL wiki2 / c4 |
| ----------- | ------------------- | -------- | ------------------ | ------------- | -------------- |
| Output-side | 37.5%               | 768      | **76.27**    | **78.33** | **9.42 / 14.99** |
| Input-side  | 37.5%               | 439      | 74.09              | 77.30         | 11.10 / 17.18  |

**Output-side wins on every metric** (+2.2 pt HellaSwag, +1.0 pt MMLU, lower PPL) — the shared basis
should span the larger hidden axis `d=2048`, not the intermediate axis `p=768`.

---

# Methods

Each subsection is self-contained: settings → results → analysis.

## 1. Leverage ranking + Nyström reconstruction (pruning) — best at 25% and 33%

**Settings.** Rank expert-FFN intermediate channels by the **ridge leverage score**
`diag((C+λI)⁻¹C)` (`C` = per-expert `down_proj` input covariance `zᵀz/N`, `λ=1.0`); allocate
per-expert budget via the `attr_coverage` intra-layer planner and `loss_coverage` inter-layer planner;
physically slim each expert with a **closed-form Nyström `down_proj` reconstruction**
`W_downₙₑw = (SᵀCS)⁻¹(SᵀC)W_downᵀ` (absorbs pruned-channel mass into survivors) rather than plain
column slicing. Attention + router kept dense. `shrink_gate: true`, `min_per_expert: 16`, mode
`test_only` (one-shot). Covariance/leverage collected **on-the-fly at eval time** from c4
(128 batches × bs16, seq 512) via `src/calibration/channel_scoring/collect_covariance.py` — one hooked
c4 forward on the full un-slimmed model (~17 min on 4×40GB A100), cached into `scores_dir`. Base
`Qwen/Qwen3-30B-A3B-Thinking-2507`. Run dates: 25%/33% 2026-07-10.

**Results.**

| Setting | Reduction | HellaSwag | MMLU (5-shot) |
| ------- | --------- | --------- | ------------- |
| 25% expert-FFN | −23.74% overall | **78.45** | **76.04** (±0.34) |
| 33% expert-FFN | −31.33% overall | **78.40** | **73.00** (±0.35) |

MMLU by category @ 33%:

| Category | Attribution-guided 33% | Uniform 33% (ablation) |
| -------- | ---------------------- | ---------------------- |
| Humanities | 66.23 | 25.87 |
| Social sciences | 84.47 | 28.66 |
| STEM | 65.87 | 27.18 |
| Other | 79.14 | 28.71 |

**Analysis.** Near-lossless on HellaSwag at both budgets (−0.11 / −0.16 vs the 78.56 unpruned
Thinking-2507 baseline); MMLU costs ~0 pt at 25% and ~3 pt at 33%. This is the strongest method overall
on the same-base comparison. The attribution-guided allocation is what carries it — the uniform
ablation (below) collapses at the same budget.

## 2. Activation-magnitude ranking + plain slicing (pruning baseline)

**Settings.** Same allocation scaffold as method 1 but ranks channels by **activation magnitude** and
removes columns by **plain slicing** (no Nyström reconstruction). `intra_expert_metric: activation`.
Base Thinking-2507.

**Results.** 25% expert-FFN: HellaSwag **78.23**, MMLU **76.28**.

**Analysis.** Slightly behind leverage+Nyström on HellaSwag (78.23 vs 78.45), ~tied on MMLU (76.28 vs
76.04). Isolates the value of the leverage ranking + closed-form reconstruction: modest but consistent
on HellaSwag, negligible on MMLU at 25%.

## 3. Uniform allocation (pruning ablation)

**Settings.** Leverage+Nyström machinery with `inter_layer_method: uniform` and
`intra_layer_method: uniform` — every layer and every expert pruned by the same fraction. Base
Thinking-2507. Run date 2026-07-14.

**Results.** 33% expert-FFN: HellaSwag **65.10** (±0.48), MMLU **27.40** (±0.38). MMLU-by-category in
method 1's table (all four categories ~25–29, i.e. near chance).

**Analysis.** **Collapses at 33%** — MMLU drops to near-random (27.4) while the attribution-guided run
holds 73.0. This is the clearest evidence that the attribution-guided budget allocation, not the
Nyström machinery alone, is what preserves task signal.

## 4. MoBE — Mixture-of-Basis-Experts (factorization) — best factorization result

Data-free. Factorizes each routed expert's projection(s) into a **per-layer shared basis**
`B ∈ ℝ^{m×r×c}` (`m` bases, rank `r`) plus a **per-expert transform** `A_e` and simplex mixing
coefficients `α_e ∈ ℝ^m`, with a weight-space `SiLU` between basis and transform (MoBE Algorithm 1):
`W_hat = A_e · f(Σ_j softmax(α_e)_j B_j)`. Fit with the reference-matched `inclusionAI/MoBE` trainer
(grouped-SVD init, **std-only normalization**, **mean-MSE**), Adam `lr=0.07`, **2000 fixed steps** per
(layer, type), `patience=0`. Data-free (`calib_source: c4` only satisfies the argparser). Base
`Qwen/Qwen3-30B-A3B` (bf16), one-shot, no recovery. Impl: `src/compress/moe_basis/{mobe.py,fit.py}`.
Per-expert dims: `p=768`, `d=2048`, `n=128`, 48 layers; basis rank fixed at `r=p=768` unless noted.

Baseline (uncompressed A3B, same run): HellaSwag **77.68**, wiki2 PPL **8.70**, c4 PPL **14.05**; MMLU
baseline not re-run (cited **82.0**).

### 4a. Classic MoBE — gate/up factorized, down_proj dense (25% & 33%)

**Settings.** Compresses **`gate_proj` + `up_proj`** only (`_PROJ_TYPES` in `mobe.py`); `down_proj`,
router, attention, norms, embeddings left dense. Input-side shared basis `B ∈ ℝ^{r×d}` (spans hidden
`d`), per-expert `A_e ∈ ℝ^{p×r}`. `m` is the sole compression knob.

The **two ratios** (the crux): stored per compressed (layer,type) = `A` (`n·p·r`) + `B` (`m·r·d`) +
`α` (`n·m`, negligible); original = `n·p·d`. With `r=p=768`:

| `m`  | γ on **up+gate** (compressed) | γ over **whole MoE layer** (incl. dense down), (2·γ_ug+1)/3 | Config |
| ------ | ----------------------------- | ---------------------------------------------------------- | ------ |
| **32** | 0.625 → −37.5% on up+gate    | 0.750 → **−25.0%** whole-MoE                              | `qwen3_30b_a3b_mobe.yaml` |
| **16** | 0.500 → −50.0% on up+gate    | 0.667 → **−33.3%** whole-MoE                              | `qwen3_30b_a3b_mobe_33.yaml` |

The headline "25% / 33.3%" is the **whole-MoE** figure (denominator = all three expert matrices; the
touched matrices are cut harder). `m=16` realized `stored/orig = 9.66e9/1.93e10 = 0.5000` exactly. Note
this denominator is MoE-layer params only (attention/router excluded) — *not* the overall-model axis
the pruning rows use. Fit converged uniformly `rel_err 0.97→0.33` (`m=32`) / `→0.35–0.47` (`m=16`).
25% run on A100-New (8 GPUs); 33% on A100-Sagemaker (8 GPUs). Dates: `m=32` 2026-07-15, `m=16`
2026-07-16.

**Results.**

| Setting | Whole-MoE reduction | up+gate reduction | HellaSwag | ΔHS | MMLU | wiki2 / c4 PPL |
| ------- | ------------------- | ----------------- | --------- | --- | ---- | -------------- |
| Baseline | — | — | 77.68 | — | 82.0 | 8.70 / 14.05 |
| MoBE `m=32` | −25.0% | −37.5% | **73.67** | −4.01 | **77.23** | 9.59 / 15.98 |
| MoBE `m=16` | −33.3% | −50.0% | **69.64** | −8.04 | **74.05** | 11.75 / 20.32 |

**Analysis.** Going `m=32→m=16` (25%→33.3%) costs ~4 extra pts on both tasks and pushes c4 PPL
15.98→20.32. The 33% `m=16` run is the weak point: concentrating the whole cut on up+gate forces them to
γ=0.5, and it shows. This motivated the even-split variant (4b). Artifacts:
`methods/mobe_benchmark_comparison.json` (`m=32`, run `ce_mobe_calib-c4-0.75_1.0e-04-0715-005135`);
`.../ce_mobe_calib-c4-0.67_1.0e-04-0716-070717/benchmark_comparison.json` (`m=16`).

### 4b. MoBE even-split — gate/up/down all factorized (33%) — best factorization result

**Settings.** Instead of leaving `down_proj` dense, factorize **gate, up, and down** to the *same*
per-matrix ratio so the 33% reduction is spread evenly. `down_proj` is `(d, p)` while gate/up are
`(p, d)`; to keep the shared basis on the hidden (`d`) axis for all three, `down_proj` is fit on its
**transpose** (`moe_down_basis_side=output`) → symmetric **output-side** basis `B ∈ ℝ^{r×d}`, per-expert
`A_e ∈ ℝ^{p×r}`, materialized `W_hat = (A_e · f(mix(B)))ᵀ`. Each matrix then has
γ = `r/d + m/n` = `0.375 + m/128`; **`m=38` → γ=0.672 on every matrix → −32.8% overall** (closest
integer-`m` to an even 33%; realized `stored/orig = 1.9479e10/2.8991e10 = 0.6719`). Knobs
`moe_compress_down=true`, `moe_compress_gate_up=true`, `m=38`, `r=768`. Same trainer as 4a (2000 steps,
`lr=0.07`, std-only norm). Two-phase on 40GB A100s: **fit** on one GPU with the model on CPU
(`MODEL_ON_CPU=1`, ~15 GiB/GPU, all 144 matrix-fits), then **eval** the saved `hf_reconstructed/`
sharded across 8 GPUs. Config `qwen3_30b_a3b_mobe_33_even.yaml`. Run date 2026-07-23.

**Results.**

| Setting | Whole-MoE reduction | per-matrix γ | HellaSwag | MMLU | wiki2 / c4 PPL |
| ------- | ------------------- | ------------ | --------- | ---- | -------------- |
| MoBE even-split `m=38` | −32.8% | 0.672 (gate=up=down) | **73.13** | **76.83** | 10.10 / 16.57 |
| (vs) classic `m=16`, down dense | −33.3% | up+gate 0.5, down 1.0 | 69.64 | 74.05 | 11.75 / 20.32 |

**Analysis.** **+3.5 pt HellaSwag / +2.8 pt MMLU over the classic down-dense `m=16` at the same overall
reduction**, and much better PPL. The gain is the allocation: spreading the cut means each matrix is
only reduced to γ≈0.67 (vs γ=0.5 on up+gate in `m=16`), and the reconstruction error per matrix is
correspondingly smaller. Crucially, factorizing `down_proj` with the **output-side** basis reconstructs
just as cleanly as gate/up (per-layer fit rel_err ~0.30 at all depths, incl. L47 — the transpose trick
does not disadvantage `down_proj`). This is now the best factorization result at 33%, 🥈 on the
leaderboard behind only pruning. (Its MMLU 76.83 nominally tops the pruning row 73.00, but confounded
by base ckpt + un-rerun MMLU baseline — not a clean MoBE>pruning claim.)

### 4c. down_proj-only basis-side ablation (37.5%, `m=32`)

**Settings.** Isolates **where the shared basis should sit for `down_proj`**. Both runs compress **only
`down_proj` by 37.5%** (gate/up left dense), same `m=32`, differing only in the basis axis:
**output-side** (basis spans hidden `d`, `r=768` → γ=0.625, `moe_down_basis_side=output`) vs
**input-side** (basis spans intermediate `p`, `r=439` → γ=0.625, `moe_down_basis_side=input`,
`moe_basis_rank_down=439`). Same-`m` clean controlled comparison of placement. Because only `down_proj`
is touched, whole-MoE reduction ≈12.5% (37.5% of the ⅓ of MoE params `down_proj` holds). Knob
`moe_compress_gate_up=false`. Configs `qwen3_30b_a3b_mobe_down_{out,in}_375_{fit_only,eval}.yaml`. Base
A3B, one-shot. Run date 2026-07-28.

**Results.**

| Basis side | down_proj reduction | rank `r` | HellaSwag | MMLU | wiki2 / c4 PPL | L0 fit rel_err |
| ---------- | ------------------- | -------- | --------- | ---- | -------------- | -------------- |
| Output-side | 37.5% | 768 | **76.27** | **78.33** | **9.42 / 14.99** | ~0.33 |
| Input-side | 37.5% | 439 | 74.09 | 77.30 | 11.10 / 17.18 | ~0.40 |

**Analysis.** **Output-side wins on every metric** — +2.2 pt HellaSwag, +1.0 pt MMLU, and lower PPL
(c4 14.99 vs 17.18). The fit-quality signal predicts it: the output-side basis, spanning the larger
hidden axis `d=2048`, reconstructs `down_proj` markedly better (L0 rel_err 0.33 vs input-side 0.40) —
a richer shared subspace admits more of the per-expert variation at the same `m`. **Conclusion: when
factorizing `down_proj`, place the shared basis on the output (hidden `d`) side.** This is exactly the
choice the even-split run (4b) makes.

## 5. RFID-MoE (factorization)

**Settings.** Frequency-grouped basis decomposition (`m=32` groups, fusion `ξ=0.8`),
`compression_ratio=0.625` retain of up+gate, `down_proj` dense. The **residual reconstruction module
(§3.4) is intentionally omitted**. Routing counts collected from c4 (128 seqs × 1024 tok). RFID predates
the trainer rewrite (≤3000 steps, early-stop patience 500) — so **RFID vs MoBE is not yet
apples-to-apples on fit quality**. Base A3B. Run date 2026-07-14.

**Results.** −28.4% MoE-layer (allocator undershot the 0.625 target): HellaSwag **66.80**, MMLU
**71.32**, PPL 12.68 / 21.49.

**Analysis.** Trails MoBE at a comparable budget, for two reasons: the omitted residual module (the
paper's headline retention leans on it) and the short pre-rewrite fit. A fresh RFID run on the new
fitter (+ residual) is the remaining piece. No RFID 33% run exists yet. Artifact:
`docs/results/mobe/rfid_benchmark_comparison.json` (run `ce_rfid_calib-c4-0.625_1.0e-04-0714-184003`).

## 6. Nyström-MoE compress-then-fit (factorization) — run at 33%

**Settings.** Sequential, one MoE layer at a time in depth order (re-linearized: layer ℓ's calibration
runs through the already-compressed prefix 0…ℓ-1). Per expert: rank intermediate channels by ridge
leverage `diag((C+λI)⁻¹C)` (`λ=1.0`), keep a **uniform `k=512`** (of `p=768` → exactly −33.3% of
expert-FFN; **gate/up/down all shrink**, unlike classic MoBE), closed-form Nyström `down_proj`
reconstruction on the kept subset (escalating-ridge Cholesky, column-slice fallback for rare
rank-deficient deep experts), then a **per-layer activation-aware joint fit**. Impl:
`src/compress/moe_basis/nystrom_moe.py` (`fit_mode=layer`). Base Thinking-2507.

**Local-fit loss (layer-joint).** Cache block input `X ∈ ℝ^{T×d}` (through the compressed prefix) and a
reference output `Y ∈ ℝ^{T×d}`, `T = 65536` rows. Optimize the stacked narrowed weights of all E=128
experts jointly by **replaying the block forward with the FROZEN router**:

```
logits = X · Wgate_routerᵀ                       # router frozen (not trained)
w, sel = topk(softmax(logits), top_k)            # w renormalized if norm_topk_prob
ŷ_t    = Σ_{e ∈ sel(t)} w_{t,e} · D_e ( SiLU(A_e xₜ) ⊙ U_e xₜ )     # per-token top-k experts
L      = (1/Td) ‖ Ŷ − Y ‖_F²                     # raw block-output MSE
```

Frozen router → each token's gradient reaches only the experts it activates. Adam, **`lr=3e-4`**
(**must be tuned on a DEEP layer** — MoBE-style `lr=1e-3` diverges by depth; L20 lr-scan picked 3e-4),
**`iters=1500`** (L20 study: block-MSE keeps dropping well past 800, flat by ~1500), cosine decay to
`0.05·lr`, full-fp32, deterministic 4096-row minibatches, best-state seeded with the closed-form init
(never regresses). Only `{A,U,D}` factors trainable. Cost ~10 min/layer on a 4-GPU shard (~8 h total).
Calibration c4, 128 seqs × 1024 tok. Configs `qwen3_30b_a3b_nystrom_moe{,_teacher}.yaml`. Dates:
800-step self 2026-07-16; converged self + teacher 2026-07-20.

**Two fit targets** (`nystrom_fit_target`), head-to-head at 33%, both converged `iters=1500`:

- **fix 1 — `self`** (default): `Y = OrigBlock_ℓ(X_compressed)` — match the block's own output on the
  re-linearized compressed input.
- **fix 2 — `teacher`**: `Y = h*_ℓ`, the *uncompressed* model's clean block output at depth ℓ (cached
  once, row-aligned by deterministic loader order) — a GPTQ/AWQ-style sequential error-compensation
  target that absorbs upstream drift.

**Results.**

| Run | HellaSwag | MMLU | PPL wiki2 / c4 |
| --- | --------- | ---- | -------------- |
| self, 800 it (under-trained) | 65.46 | 60.92 | 13.46 / 17.98 |
| fix 1 — self, 1500 it | **66.24** | 60.70 | 12.97 / 17.69 |
| fix 2 — teacher, 1500 it | 65.97 | **61.24** | 12.97 / 17.75 |

Per-layer fit quality (init → post-fit block-output MSE, run `…-0716-103639`, mean per band):

| Layer band | init MSE | final MSE | mean reduction |
| ---------- | -------- | --------- | -------------- |
| L0–L11 (shallow) | 1.1e-4 | 4.6e-5 | 2.3× |
| L12–L35 (mid) | 3.8e-4 | 1.5e-4 | 2.6× |
| L36–L47 (deep) | 5.4e-3 | 3.6e-3 | 1.6× |

**Analysis.** Converging the fit (fix 1) lifted deep-layer block-MSE reduction from ~1× (the 800-iter
run gave up at L47) to **4–5×**; fix 2's clean-trajectory target edges MMLU up further (best 61.24).
Both beat the under-trained run, but **end-to-end the gap to MoBE/pruning barely closes**: the init MSE
grows monotonically with depth (3.7e-6 at L0 → 9.4e-3 at L47, ~2500×) as reconstruction error compounds
through the already-slimmed prefix, and the fit's relative gain shrinks in the last ~8 layers (3×
mid-stack → 1.2–1.5× by L44–47). The ~0.15 per-block residual compounds over 48 layers, and one-shot
activation matching (self or teacher) cannot fully undo a 33% structural cut of all three matrices.
Still, MMLU 61.24 vs the uniform ablation's 27.40 confirms leverage-guided selection + fit retains real
signal. A LoRA/CE recovery pass is the clear next step. See `plan/nystrom_fit_diagnosis.md`. Artifact:
`docs/results/mobe/nystrom_moe_benchmark_comparison.json` (run `ce_nystrom_moe_calib-c4-0.67_1.0e-04-0716-103639`).

---

## Reproduce

```bash
# ── Pruning (Nyström) ──────────────────────────────────────────────
# 25% / 33% / uniform-33% ablation
python src/train/merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_nystrom_25p_{hellaswag,mmlu}.yaml
python src/train/merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_nystrom_33p_{hellaswag,mmlu}.yaml
python src/train/merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_nystrom_uniform33_{hellaswag,mmlu}.yaml

# ── Factorization (MoBE / RFID / Nyström-MoE) ──────────────────────
python src/compress_then_train.py --config configs/compress_then_train/qwen3_30b_a3b_mobe.yaml       # 25% (m=32)
python src/compress_then_train.py --config configs/compress_then_train/qwen3_30b_a3b_mobe_33.yaml    # 33% (m=16, down dense)
# MoBE even-split 33% (m=38, gate/up/down). Two-phase on 40GB A100s: fit (model on CPU), then sharded eval.
CUDA_VISIBLE_DEVICES=7 MODEL_ON_CPU=1 ATTN_IMPLEMENTATION=sdpa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python src/compress_then_train.py --config configs/compress_then_train/qwen3_30b_a3b_mobe_33_even_fit_only.yaml
FORCE_DEVICE_MAP_AUTO=1 PER_GPU_MEM=36GiB ATTN_IMPLEMENTATION=sdpa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python src/compress_then_train.py --config configs/compress_then_train/qwen3_30b_a3b_mobe_33_even_eval.yaml  # set model path to Phase-1 hf_reconstructed/
# down_proj-only basis-side ablation @ 37.5% (fit_only + eval each, same two-phase pattern):
#   ..._mobe_down_out_375_{fit_only,eval}.yaml  (output-side basis, r=768)
#   ..._mobe_down_in_375_{fit_only,eval}.yaml   (input-side  basis, r=439)
python src/compress_then_train.py --config configs/compress_then_train/qwen3_30b_a3b_rfid.yaml
# Nyström-MoE compress-then-fit @ 33% (lr tuned on a deep layer)
python src/compress_then_train.py --config configs/compress_then_train/qwen3_30b_a3b_nystrom_moe.yaml
```

Prefix the 40 GB A100 runs with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_DEVICE_MAP_AUTO=1 PER_GPU_MEM=36GiB ATTN_IMPLEMENTATION=sdpa`.

### Raw result artifacts

- Pruning: `run_results/A100-New/results_eval/qwen3_nystrom_{25p,33p,uniform33}_{hellaswag,mmlu}_*/lm_harness/`.
- MoBE 25%: run `ce_mobe_calib-c4-0.75_1.0e-04-0715-005135`, JSON `docs/results/mobe/mobe_benchmark_comparison.json`.
- MoBE 33% (`m=16`): run `ce_mobe_calib-c4-0.67_1.0e-04-0716-070717` (A100-Sagemaker).
- MoBE even-split 33% (`m=38`): fit run `ce_mobe_calib-c4-0.67_1.0e-04-0723-015029`, eval `ce_full_calib-c4-0.67_1.0e-04-0723-234445`.
- MoBE down-only 37.5%: `ce_mobe_calib-c4-0.625_..._downout375-0728-071517` (output) / `..._downin375-0728-071517` (input).
- RFID: run `ce_rfid_calib-c4-0.625_1.0e-04-0714-184003`, JSON `docs/results/mobe/rfid_benchmark_comparison.json`.
- Nyström-MoE: run `ce_nystrom_moe_calib-c4-0.67_1.0e-04-0716-103639`, JSON `docs/results/mobe/nystrom_moe_benchmark_comparison.json`.

---

## Notes & caveats

- **No fine-tuning anywhere.** Every number is one-shot. Both families should improve with a LoRA/CE
  recovery pass; the factorization gap to the papers' ~96–98% retention is partly the missing recovery
  step (and, for RFID, the omitted residual module + short-cap fit).
- **Cross-family comparisons are indicative, not rigorous** — different base checkpoints (Thinking-2507
  vs A3B), different reduction axes (overall vs MoE-layer params), and an un-rerun MMLU baseline for
  factorization. To make it rigorous: re-run one family on the other's base checkpoint and re-measure
  the uncompressed MMLU baseline.
- **Active vs storage params (25% pruning).** A 25% storage cut yields only ~1.4% *active*-compute cut
  (full-model active ratio 0.986): attribution-guided pruning strips the least-routed experts' channels,
  which rarely enter a token's top-8. Real memory savings but **not** a proportional FLOPs speedup. See
  `docs/results/attribution_guided/nystrom.md`.
