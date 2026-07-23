# Qwen3-30B-A3B — Compression Leaderboard (one-shot, no recovery fine-tuning)

Consolidated results for every compression method we have run on Qwen3-30B-A3B, merged from:

- `docs/results/attribution_guided/nystrom.md` — structural expert-FFN **pruning** (ridge-leverage
  ranking + Nyström reconstruction, plus activation-magnitude and uniform baselines).
- `docs/results/mobe/initial_results.md` — expert-FFN **factorization** (MoBE, RFID-MoE).

All numbers are **one-shot** (decompose/prune → eval, **no LoRA/CE recovery training**) on the full
lm-eval-harness tasks. Leaderboards are bucketed by target reduction (~25% and ~33%).

> ⚠️ **Read the caveats before comparing rows across families.** Two things are not held constant:
>
> 1. **Base checkpoint differs.** Pruning runs used `Qwen/Qwen3-30B-A3B-Thinking-2507`; the
>    MoBE/RFID factorization runs used plain `Qwen/Qwen3-30B-A3B`. Their uncompressed baselines are
>    **78.56** vs **77.68** HellaSwag acc_norm respectively — so a ~0.9 pt gap is baked in.
> 2. **Reduction axis differs.** Pruning rows report **overall** model-param reduction (25% expert
>    prune → −23.74% overall). Factorization rows report **MoE-layer** param reduction (down_proj
>    left dense). They land in the same bucket but are not the identical quantity — see per-row notes.

---

## Model

| Property                      | Value                                                                                                                                                                              |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Architecture                  | Qwen3-30B-A3B — hidden`d=2048`, MoE intermediate `p=768`, `n=128` experts, top-k 8, 48 layers, SwiGLU/SiLU, no shared expert                                                |
| Pruning base checkpoint       | `Qwen/Qwen3-30B-A3B-Thinking-2507` (bf16)                                                                                                                                        |
| Factorization base checkpoint | `Qwen/Qwen3-30B-A3B` (bf16)                                                                                                                                                      |
| Hardware                      | A100-New, 40 GB A100s (`FORCE_DEVICE_MAP_AUTO=1 PER_GPU_MEM=36GiB ATTN_IMPLEMENTATION=sdpa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`); pruning/RFID on 4 GPUs, MoBE on 8 |

**Eval protocol (all methods).** HellaSwag full 10 042 items, `num_fewshot=0` (report acc_norm);
MMLU full 14 042 questions × 57 subtasks, `num_fewshot=5` (acc). Each task in its own lm-eval call.
PPL (wikitext2 + c4) reported only for the factorization runs.

---

## Leaderboard @ ~25% reduction

| Rank | Method                                       | Family    | Reduction                         | HellaSwag acc_norm | MMLU (5-shot)            | PPL wiki2 / c4 | Base ckpt     |
| ---- | -------------------------------------------- | --------- | --------------------------------- | ------------------ | ------------------------ | -------------- | ------------- |
| —   | Original (Thinking-2507)                     | —        | 0%                                | 78.56              | —                       | —             | Thinking-2507 |
| —   | Original (Qwen3-30B-A3B)                     | —        | 0%                                | 77.68              | 82.0†                   | 8.70 / 14.05   | A3B           |
| 🥇   | **Leverage ranking + Nyström**        | prune     | −23.74% overall (25% expert-FFN) | **78.45**    | **76.04** (±0.34) | —             | Thinking-2507 |
| 🥈   | Activation-magnitude + plain slicing         | prune     | 25% expert-FFN                    | 78.23              | 76.28                    | —             | Thinking-2507 |
| 🥉   | **MoBE** (`m=32`, `r=768`)         | factorize | −25% MoE-layer                   | 73.67              | 77.23                    | 9.59 / 15.98   | A3B           |
| 4    | RFID-MoE (`m=32`, `ξ=0.8`, no residual) | factorize | −28.4% MoE-layer‡               | 66.80              | 71.32                    | 12.68 / 21.49  | A3B           |

- † MoBE-doc cites an uncompressed MMLU of 82.0 as a reference point; the MMLU baseline was **not**
  re-run on either base checkpoint, so MoBE/RFID MMLU deltas are against this cited value, not a
  same-run baseline. Treat with caution.
- ‡ RFID's adaptive per-group rank allocator undershoots the 0.625 retain budget, so it actually
  removed **28.4%** of MoE-layer params (beyond the 25% target) — its row is a mildly conservative
  read for the 25% point, and it is heavier than the other three rows.
- **Note on MMLU ordering:** MoBE's MMLU (77.23) nominally exceeds the pruning rows (76.04/76.28),
  but this is confounded by the different base checkpoint and the un-rerun MMLU baseline; do not
  read it as MoBE > pruning on MMLU without a matched baseline.

**Takeaway @ 25%.** On the same-base *pruning* family, **leverage + Nyström is the winner** —
near-lossless HellaSwag (78.45 vs 78.56 unpruned) and essentially tied MMLU with the activation
baseline. **MoBE** is the best *factorization* result: a clean ~4 pt HellaSwag drop (73.67 vs its own
77.68 baseline) with zero fine-tuning, well ahead of RFID.

---

## Leaderboard @ ~33% reduction

The pruning family, **MoBE** factorization, and the **Nyström-MoE compress-then-fit** factorization
method have all been run at 33%. The uniform-allocation row is an ablation, not a competitive method —
it shares the exact leverage+Nyström machinery but allocates budget uniformly.

| Rank | Method                                                                                   | Family    | Reduction                         | HellaSwag acc_norm | MMLU (5-shot)            | PPL wiki2 / c4 | Base ckpt     |
| ---- | ---------------------------------------------------------------------------------------- | --------- | --------------------------------- | ------------------ | ------------------------ | -------------- | ------------- |
| —   | Original (Thinking-2507)                                                                 | —        | 0%                                | 78.56              | 81.73                    | 7.29 / 12.46   | Thinking-2507 |
| —   | Original (Qwen3-30B-A3B)                                                                 | —        | 0%                                | 77.68              | 82.0†                   | 8.70 / 14.05   | A3B           |
| 🥇   | **Attribution-guided** (leverage + Nyström, `loss_coverage`+`attr_coverage`)  | prune     | −31.33% overall (33% expert-FFN) | **78.40**    | **73.00** (±0.35) | —             | Thinking-2507 |
| 🥈   | **MoBE** (`m=16`, `r=768`)                                                     | factorize | −33.3% MoE-layer                 | 69.64              | 74.05                    | 11.75 / 20.32  | A3B           |
| 🥉   | **Nyström-MoE** fix1 (`k=512`, layer-joint fit, self-target, 1500 it)           | factorize | 33% expert-FFN                    | 66.24              | 60.70                    | 12.97 / 17.69  | Thinking-2507 |
| 4    | **Nyström-MoE** fix1+2 (`k=512`, layer-joint fit, teacher-traj target, 1500 it) | factorize | 33% expert-FFN                    | 65.97              | **61.24**          | 12.97 / 17.75  | Thinking-2507 |
| —   | Nyström-MoE (self-target, 800 it — under-trained)                                      | factorize | 33% expert-FFN                    | 65.46              | 60.92                    | 13.46 / 17.98  | Thinking-2507 |
| ✗   | Uniform (`uniform`+`uniform`) — ablation                                            | prune     | −31.28% overall (33% expert-FFN) | 65.10 (±0.48)     | 27.40 (±0.38)           | —             | Thinking-2507 |

**Takeaway @ 33%.** Attribution-guided leverage+Nyström pruning remains far ahead on HellaSwag:
**almost no loss** (78.40, −0.16 vs unpruned) at a modest MMLU cost (73.00, ~3 pt). **MoBE** at 33%
(`m=16` — half the 25% run's basis count) is the **best factorization result**: HellaSwag 69.64 (−8.0
vs its own 77.68 baseline) and MMLU 74.05, one-shot with no recovery. Its MMLU nominally *exceeds* the
pruning row (74.05 vs 73.00), but this is confounded by the different base checkpoint (A3B vs
Thinking-2507) and the un-rerun MMLU baseline — do not read it as MoBE > pruning without a matched
baseline. Going 25% → 33% costs MoBE ~4 extra pts on both tasks (HellaSwag 73.67 → 69.64, MMLU
77.23 → 74.05) and pushes PPL up (c4 15.98 → 20.32). The **Nyström-MoE compress-then-fit** — which
shrinks *all three* expert matrices (gate/up/down) to `k=512` via per-expert ridge-leverage channel
selection + closed-form `down_proj` reconstruction + a per-layer activation-aware joint fit — lands
below both (66.24 HellaSwag, 61.24 MMLU) but **far above the uniform ablation on MMLU** (61.24 vs
27.40), confirming the leverage-guided selection + fit retains real task signal that naive uniform
slicing destroys.

**Fit-quality fixes (2026-07-20).** A diagnosis of the original 800-iter run
(`docs/results/total_param/plan/nystrom_fit_diagnosis.md`) showed the fit was *under-trained* and
*collapsed at deep layers*: block-MSE at L20 kept dropping to 3000 steps (rel 0.21→0.156), and deep
layers L44–L47 barely improved (1.0–1.7×) at 800 iters. Two fixes were run head-to-head at 33%:

- **fix 1** — converged iters (800 → 1500), self-target (match the block's own output on the
  compressed-prefix input). Deep-layer block-MSE reduction jumped from 1.0–1.7× to **4–5×** (L47:
  9.9e-3→3.7e-3). **HellaSwag 65.46→66.24, MMLU 60.92→60.70** vs the 800-iter run.
- **fix 2** — fix 1 + *sequential teacher-trajectory target*: cache the uncompressed model's clean
  per-block outputs `h*_ℓ` once, and fit each block on its (drifted) compressed-prefix input to match
  `h*_ℓ`, so downstream layers absorb accumulated upstream drift. **HellaSwag 65.97, MMLU 61.24** —
  best Nyström-MoE MMLU, slightly behind fix-1 on HellaSwag.

**Verdict.** The convergence + drift fixes each move the needle a little (fix 1 best HellaSwag 66.24,
fix 2 best MMLU 61.24; both ~+0.5–1 pt over the under-trained run) and the deep-layer reconstruction is
now genuinely solved (4–5× vs the earlier 1× collapse). **But end-to-end the gap to MoBE/pruning barely
closes** — confirming the diagnosis's core point: the ~0.15 per-block residual still compounds over 48
layers, and one-shot activation matching (self *or* teacher target) cannot fully undo a 33% structural
cut. A LoRA/CE recovery pass on top is the clear next step for both factorization methods. No RFID 33%
run exists yet.

### MMLU by category (pruning, 33%)

| Category        | Attribution-guided 33% | Uniform 33% |
| --------------- | ---------------------- | ----------- |
| Humanities      | 66.23                  | 25.87       |
| Social sciences | 84.47                  | 28.66       |
| STEM            | 65.87                  | 27.18       |
| Other           | 79.14                  | 28.71       |

### Nyström-MoE fitting quality (per-layer, 33%)

Per-layer block-output MSE **before** (closed-form Nyström init) vs **after** the activation-aware
joint fit (lr=3e-4, 800 steps, 65 536 calib tokens), from run `…-0716-103639`. The fit improves
**every** layer (best-state seeding guarantees no regression), by an average **2.3×** (peak 3.2× at
L23–24). Two structural trends: (1) the closed-form init MSE **grows monotonically with depth**
(3.7e-6 at L0 → 9.4e-3 at L47, ~2500×) as reconstruction error compounds through the already-slimmed
prefix; (2) the fit's **relative gain shrinks in the last ~8 layers** (3× mid-stack → ~1.2–1.5× by
L44–47, and L47 exactly 1.00× = fit fell back to init), where the loss landscape is stiffest and the
absolute error is largest. This depth-compounding is the primary reason the one-shot result trails
pruning; a recovery pass would most help the deep layers.

| Layer band        | init MSE (mean) | final MSE (mean) | mean reduction |
| ----------------- | --------------- | ---------------- | -------------- |
| L0–L11 (shallow) | 1.1e-4          | 4.6e-5           | 2.3×          |
| L12–L35 (mid)    | 3.8e-4          | 1.5e-4           | 2.6×          |
| L36–L47 (deep)   | 5.4e-3          | 3.6e-3           | 1.6×          |

Selected rows (init → final, ×reduction): L0 3.66e-6→2.53e-6 (1.4×), L12 1.81e-4→6.52e-5 (2.8×),
L23 3.32e-4→1.03e-4 (**3.2×**), L36 7.88e-4→2.65e-4 (3.0×), L44 7.99e-3→5.21e-3 (1.5×),
L47 9.44e-3→9.44e-3 (1.0×). Full trajectory in the run log
(`grep 'joint lr=' run_logs/nys_full_v2_*.log`).

---

## MoBE — full breakdown (settings, what's compressed, ratios, eval)

MoBE (Mixture-of-Basis-Experts) is the **factorization** method that leads the factorization family
at both 25% and 33%. This section is self-contained: it states the exact run settings, which weight
matrices are touched, the compression ratio *on the compressed matrices* vs. *over the whole MoE
layer*, and every eval number we have. Impl: `src/compress/moe_basis/mobe.py` (param accounting) +
`src/compress/moe_basis/fit.py` (fitter). Base checkpoint: `Qwen/Qwen3-30B-A3B` (bf16), one-shot
(decompose → eval), **no LoRA/CE recovery**.

### What gets compressed

- **Compressed:** every routed expert's **`gate_proj` and `up_proj`** (`_PROJ_TYPES = ("gate_proj", "up_proj")` in `mobe.py`). Each is factorized into a **per-layer shared basis** `B ∈ ℝ^{m×r×d}`
  (`m` bases, rank `r`) plus a **per-expert transform** `A_e ∈ ℝ^{p×r}` and mixing coefficients
  `α_e ∈ ℝ^m`, with a weight-space `SiLU` activation between basis and transform (MoBE Algorithm 1).
- **Left dense (untouched):** every expert's **`down_proj`**, the **router/gate** `Wgate`, all
  **attention** weights, norms, and embeddings. This is the same dense-`down_proj` convention RFID uses.
- Per-expert dims (Qwen3-30B-A3B): `p = 768` (MoE intermediate), `d = 2048` (hidden), `n = 128`
  experts/layer, `48` layers. Basis rank is fixed at `r = p = 768` (the paper fixes `r=p` and varies
  `m`), so **`m` is the only compression knob**.

### The two ratios (this is the crux)

Stored params per compressed (layer, type): `A` = `n·p·r` + `B` = `m·r·d` + `α` = `n·m`
(negligible). Original per (layer, type): `n·p·d`. With `r = p = 768`:

| `m`        | Ratio on**compressed matrices** (up+gate only), γ_ug = stored/orig | Ratio over**whole MoE layer** (incl. dense down_proj), (2·γ_ug + 1)/3 | Config                         |
| ------------ | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------ |
| **32** | **0.625** → −37.5% on up+gate                                     | **0.750** → **−25.0%** whole-MoE                                | `qwen3_30b_a3b_mobe.yaml`    |
| **16** | **0.500** → −50.0% on up+gate                                     | **0.667** → **−33.3%** whole-MoE                                | `qwen3_30b_a3b_mobe_33.yaml` |

The "(2·γ_ug + 1)/3" formula is exact because up+gate are 2/3 of expert-FFN params and the dense
`down_proj` (kept at 1.0) is the remaining 1/3. So the **headline "25%" / "33.3%" is the whole-MoE
figure** (denominator = all three expert matrices); the matrices we actually touch are cut harder
(−37.5% / −50%). The `m=16` run realized `stored/orig = 9.66e9 / 1.93e10 = 0.5000` exactly.

> **Scope of the denominator:** this is **MoE-layer** params only — attention and router are
> excluded entirely. It is *not* the overall-model reduction that the pruning rows report (33%
> expert-FFN → −31.33% overall). Do not compare the MoBE "33.3%" against the pruning "31.33%" as if
> they were the same axis.

### Run settings (both `m=32` and `m=16`)

- **Fitter:** reference-matched `inclusionAI/MoBE` trainer — grouped-SVD init, **std-only
  normalization** (`moe_z_norm: true`, no mean subtraction), **mean-MSE** objective.
- **Optimization:** Adam, `moe_fit_lr = 0.07`, **2000 fixed steps** per (layer, type)
  (`moe_fit_patience: 0` → no early stop; reference uses 30 000). Fit runs per-layer over the
  stacked `(n, p, d)` expert weights. `seed = 42 + layer_idx`.
- **Data-free:** MoBE fits weights directly, so `calib_source: c4` is set only to satisfy the
  argparser; no calibration tokens enter the decomposition.
- **Fit convergence:** every `gate_proj`/`up_proj` converged uniformly from `rel_err ≈ 0.97` (step 0)
  to `rel_err ≈ 0.33` (mse ≈ 0.11) at step 2000 for `m=32`. For `m=16` the residual is larger —
  per-layer `rel_err ≈ 0.35–0.47` (shallow layers ~0.35, mid ~0.47), full trace in
  `methods/data/mobe_33_perlayer_relerr.txt`.
- **Eval:** lm-eval-harness — HellaSwag full 10 042 items `num_fewshot=0` (acc_norm), MMLU full
  14 042 × 57 subtasks `num_fewshot=5` (acc), each task in its own call; PPL on wikitext2 + c4
  (`eval_ppl_seqlen: 2048`). `eval_before_compression: true` gives the same-run baseline HellaSwag;
  MMLU baseline was **not** re-run (`baseline_skip_tasks: mmlu`) — deltas use the cited 82.0.
- **Hardware:** 25% run on A100-New (8 GPUs, ~9 GiB/GPU); 33% run on A100-Sagemaker (8 GPUs). Env:
  `FORCE_DEVICE_MAP_AUTO=1 PER_GPU_MEM=36GiB ATTN_IMPLEMENTATION=sdpa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Run dates: `m=32` 2026-07-15, `m=16` 2026-07-16.

### Eval results

Baseline (uncompressed `Qwen/Qwen3-30B-A3B`, same run): HellaSwag **77.68**, wikitext2 PPL **8.70**,
c4 PPL **14.05**. MMLU baseline not re-run (cited **82.0**).

| Setting                 | Whole-MoE reduction | up gate reduction | HellaSwag acc_norm | ΔHellaSwag | MMLU (5-shot)   | ΔMMLU vs 82.0 | wiki2 / c4 PPL |
| ----------------------- | ------------------- | ----------------- | ------------------ | ----------- | --------------- | -------------- | -------------- |
| Baseline                |                     |                   | **77.68**    |             | 82.0            |                |                |
| **MoBE `m=32`** | −25.0%             | -37.5%            | **73.67**    | −4.01      | **77.23** | −4.77         | 9.59 / 15.98   |
| **MoBE `m=16`** | −33.3%             | -50%              | **69.64**    | −8.04      | **74.05** | −7.95         | 11.75 / 20.32  |

Going `m=32 → m=16` (25% → 33.3%) costs **~4 extra pts** on both HellaSwag (73.67 → 69.64) and MMLU
(77.23 → 74.05) and pushes c4 PPL from 15.98 → 20.32. Both are one-shot with **no recovery**; a
LoRA/CE recovery pass is the clear next step toward the paper's ~96–98% retention. Raw artifacts:
`methods/mobe_benchmark_comparison.json` (`m=32`, run `ce_mobe_calib-c4-0.75_1.0e-04-0715-005135`) and
`run_results/A100-Sagemaker/.../ce_mobe_calib-c4-0.67_1.0e-04-0716-070717/benchmark_comparison.json`
(`m=16`).

---

## Methods tested

### 1. Leverage ranking + Nyström reconstruction (pruning) — **best at 25% and 33%**

Rank expert-FFN intermediate channels by the **ridge leverage score** `diag((C+λI)⁻¹C)` (`C` =
per-expert `down_proj` input covariance `zᵀz/N`, `λ=1.0`); allocate per-expert budget via the
`attr_coverage` planner; physically slim each expert with a **closed-form Nyström `down_proj`
reconstruction** `W_downₙₑw = (SᵀCS)⁻¹(SᵀC)W_downᵀ` (absorbs pruned-channel mass into survivors)
rather than plain column slicing. Attention + router kept dense.

### 2. Activation-magnitude ranking + plain slicing (pruning baseline)

Same allocation scaffold but ranks channels by activation magnitude and removes columns by plain
slicing (no reconstruction). Slightly behind leverage+Nyström on HellaSwag, ~tied on MMLU.

### 3. Uniform allocation (pruning ablation)

Leverage+Nyström machinery with `inter_layer_method: uniform` and `intra_layer_method: uniform` —
every layer and every expert pruned by the same fraction. Isolates the value of attribution-guided
allocation; **collapses at 33%**.

### 4. MoBE — Mixture-of-Basis-Experts (factorization) — **best factorization result**

Data-free. Factorizes every routed expert's `gate_proj`/`up_proj` into a shared per-layer basis
(`m` bases) + per-expert transform at rank `r=p=768`. The basis count `m` is the compression knob
(paper fixes `r=p`): `m=32` → up+gate γ=0.625 → total MoE **−25%**; `m=16` → up+gate γ=0.5 →
total MoE **−33.3%**. `down_proj` left dense in both. Fit with the reference-matched trainer
(`inclusionAI/MoBE`: std-only norm, mean-MSE), 2000 fixed steps/(layer,type). Impl in
`src/compress/moe_basis/`. Run at both 25% (`m=32`) and 33% (`m=16`).

### 5. RFID-MoE (factorization)

Frequency-grouped basis decomposition (`m=32` groups, fusion `ξ=0.8`), `compression_ratio=0.625`
retain of up+gate. The **residual reconstruction module (§3.4) is intentionally omitted**, which is
the main reason it trails MoBE here — the paper's headline retention leans on that module. Routing
counts collected from C4 (128 seqs × 1024 tok).

### 6. Nyström-MoE compress-then-fit (factorization) — new, run at 33%

Sequential, one MoE layer at a time in depth order (re-linearized: layer ℓ's calibration runs
through the already-compressed prefix 0…ℓ-1). Per expert: rank intermediate channels by ridge
leverage `diag((C+λI)⁻¹C)`, keep a **uniform `k=512`** (of `p=768` → exactly −33.3% of expert-FFN;
gate/up/down **all** shrink, unlike MoBE which keeps `down_proj` dense), closed-form Nyström
`down_proj` reconstruction on the kept subset, then a **per-layer activation-aware joint fit**: Adam
refines all 128 experts' narrowed `{gate,up,down}` against the MoE **block-output MSE**, replaying
the frozen router so each token's gradient reaches only its top-k experts. Best-state seeded with the
closed-form init so the fit never regresses. Impl in `src/compress/moe_basis/nystrom_moe.py`
(`fit_mode=layer`).

**Two fit targets (`nystrom_fit_target`), run head-to-head at 33%, both with converged `iters=1500`,
`lr=3e-4` tuned on a deep layer:**

- **fix 1 — `self`** (default): `Y_ref = OrigBlock_ℓ(X_compressed)`, i.e. match the block's own output
  on the re-linearized compressed input. HellaSwag **66.24**, MMLU **60.70**, PPL 12.97 / 17.69.
- **fix 2 — `teacher`**: `_collect_teacher_block_outputs` caches the *uncompressed* model's clean
  per-block outputs `h*_ℓ` in one forward pass (CPU fp16, row-aligned with the per-layer compressed
  inputs by deterministic loader order), and the fit targets `h*_ℓ` from the drifted compressed input
  — a GPTQ/AWQ-style sequential error-compensation that lets each block absorb upstream drift.
  HellaSwag **65.97**, MMLU **61.24** (best Nyström-MoE MMLU), PPL 12.97 / 17.75.

Converging the fit (fix 1) lifted deep-layer block-MSE reduction from ~1× (the 800-iter run gave up at
L47) to **4–5×**; fix 2's clean-trajectory target edges MMLU up further. Both beat the original
under-trained run (65.46 / 60.92) but the end-to-end gap to MoBE/pruning barely closes — the per-block
residual compounds over 48 layers. See the takeaway and `plan/nystrom_fit_diagnosis.md`.

---

## Settings

### Pruning family (Nyström, activation, uniform)

- `intra_expert_metric: leverage` (activation baseline uses `activation`); `intra_layer_method: attr_coverage` (uniform ablation: `uniform`); `inter_layer_method: loss_coverage` (uniform
  ablation: `uniform`); `nystrom_reconstruct: true`, `lambda_ridge: 1.0`, `shrink_gate: true`,
  `min_per_expert: 16`. Mode: `test_only` (one-shot, no fine-tuning).
- **Covariance/leverage** collected **on-the-fly at eval time** from c4 (128 batches × bs16,
  seq 512) via `src/calibration/channel_scoring/collect_covariance.py`; a single hooked c4 forward
  sweep on the full un-slimmed model (~17 min on 4×40GB A100), cached into `scores_dir` for reuse.
- Run dates: 25%/33% attribution-guided 2026-07-10; uniform ablation 2026-07-14.

### Factorization family (MoBE, RFID)

- **Compression target (MoBE):** exactly 25% or 33.3% of total MoE-layer params (down_proj dense),
  set by the basis count `m` at fixed `r=768`. `m=32` → up+gate γ=0.625 → −25%; `m=16` → up+gate
  γ=0.5 → −33.3% (realized `stored/orig=9.66e9/1.93e10=0.5000`, exact).
- **Compression target (RFID):** `m=32`, `compression_ratio=0.625` retain, `ξ=0.8` (actual −28.4%).
- **Fit:** reference-matched trainer (std-only norm, mean-MSE), Adam lr 0.07. MoBE = **2000 fixed
  steps/(layer,type)** (patience 0), converging `rel_err≈0.97 → ≈0.33` (mse≈0.11) uniformly across
  all 48 layers. RFID row predates the trainer rewrite (≤3000 steps, early-stop patience 500) — so
  **RFID vs MoBE is not yet apples-to-apples on fit quality**; a fresh RFID run on the new fitter is
  the remaining piece.
- Mode: `one_shot_eval_only` (no recovery training). Both MoBE checkpoints save loadable artifacts
  (`compressed_model/{mobe_native, hf_reconstructed}`). Run dates: MoBE 25% + RFID 2026-07-15,
  MoBE 33% 2026-07-16.

### Nyström-MoE compress-then-fit

- **Compression target:** exactly 33% of expert-FFN params — uniform `keep_ratio=0.67`,
  `align_to=128` → `k=512` (gate/up/down all shrink; realized reduction 33.3%).
- **Selection + init:** ridge-leverage channel ranking (`λ=1.0`) + closed-form Nyström `down_proj`
  reconstruction on the kept subset (escalating-ridge Cholesky, with a column-slice fallback for the
  rare singular/rank-deficient deep-layer expert).

**Local-fit loss (layer-joint, `fit_mode=layer`).** For MoE block ℓ, cache the block input
`X ∈ ℝ^{T×d}` (re-linearized: through the already-compressed prefix 0…ℓ-1) and a reference output
`Y ∈ ℝ^{T×d}`, capped at `T = layer_fit_tokens = 65536` token rows. The fit optimizes the **stacked
narrowed weights of all E=128 experts jointly** — `{A_e = gate_k[e], U_e = up_k[e] ∈ ℝ^{k×d}, D_e = down_k[e] ∈ ℝ^{d×k}}` — by **replaying the block's own forward with the FROZEN router**:

```
logits = X · Wgate_routerᵀ                       # router frozen (not trained)
w, sel = topk(softmax(logits), top_k)            # w renormalized if norm_topk_prob
ŷ_t    = Σ_{e ∈ sel(t)} w_{t,e} · D_e ( SiLU(A_e xₜ) ⊙ U_e xₜ )     # per-token top-k experts
L      = (1/Td) ‖ Ŷ − Y ‖_F²                     # raw block-output MSE (mean over T·d)
```

Because the router is frozen and each token routes to only its top-k experts, **each token's gradient
reaches only the experts it activates** — the truncation loss updates exactly the experts responsible
for that token. `rel_loss` (divide by `‖Y‖²`) is available but **off by default** (Adam is
scale-invariant, so it's a near no-op within a layer). The two `nystrom_fit_target` choices differ only
in `Y`:

- **`self` (fix 1):** `Y = OrigBlock_ℓ(X)` — the block's own output on the compressed-prefix input.
- **`teacher` (fix 2):** `Y = h*_ℓ`, the *uncompressed* model's clean block output at depth ℓ (cached
  once, row-aligned) — a sequential error-compensation target that absorbs upstream drift.

**Training settings (both fixes):** Adam, **`lr = 3e-4`**, **`iters = 1500` steps/layer**, cosine LR
decay to `0.05·lr` (`CosineAnnealingLR`, `T_max=iters`), full-fp32 fit, deterministic minibatch of
`4096` rows cycled by `s = (step·4096) mod T` (no RNG). Best-state snapshotted every
`snapshot_every=300` steps on the **full-`T` MSE** and **seeded with the closed-form init**, so the fit
can never regress below the Nyström closed form. No weight decay, no gradient clipping; only the
`{A,U,D}` factors are trainable (router `Wgate`, attention, norms, embeddings untouched).

- **LR must be tuned on a DEEP layer.** The MoBE-style `lr=1e-3` (tuned on shallow L0) diverges by
  depth (block-output magnitude grows ~10× through the prefix); an L20 lr-scan `{3e-4,1e-4,3e-5}`
  picked **3e-4**. `lr=1e-3` blows up at L20 (MSE 18.8 @ step 99, never recovers below init).
- **`iters=1500` (not the earlier 800).** The L20 convergence study showed block-MSE keeps dropping
  well past 800 (`1.05e-4 @800 → 6.0e-5 @3000`, rel `0.21 → 0.156`, flat by ~1500). At 1500 the fit
  gives a **~3.5× (mid) to 4–5× (deep)** block-MSE reduction over the closed-form init at every depth
  — vs the 800-step run which collapsed to ~1× on the deepest layers.
- Cost: ~10 min/layer at 1500 steps on a 4-GPU shard (~8 h compression + baseline/post eval). fix 2
  adds one extra uncompressed forward pass upfront to cache `h*_ℓ` (CPU fp16, bounded by
  `layer_fit_tokens`).
- Calibration: C4, 128 seqs × 1024 tok. Mode: `one_shot_eval_only` (no recovery). Impl:
  `_fit_layer_joint` / `_collect_teacher_block_outputs` in `src/compress/moe_basis/nystrom_moe.py`.
  Configs `qwen3_30b_a3b_nystrom_moe.yaml` (self) / `..._teacher.yaml` (teacher). Run dates: 800-step
  self 2026-07-16; converged self + teacher 2026-07-20.

---

## Reproduce

```bash
# ── Pruning (Nyström) ──────────────────────────────────────────────
# 25%
python src/train/merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_nystrom_25p_hellaswag.yaml
python src/train/merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_nystrom_25p_mmlu.yaml
# 33%
python src/train/merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_nystrom_33p_hellaswag.yaml
python src/train/merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_nystrom_33p_mmlu.yaml
# Uniform 33% ablation
python src/train/merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_nystrom_uniform33_hellaswag.yaml
python src/train/merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_nystrom_uniform33_mmlu.yaml

# ── Factorization (MoBE / RFID / Nyström-MoE) ──────────────────────
python src/compress_then_train.py --config configs/compress_then_train/qwen3_30b_a3b_mobe.yaml     # 25% (m=32)
python src/compress_then_train.py --config configs/compress_then_train/qwen3_30b_a3b_mobe_33.yaml  # 33% (m=16)
python src/compress_then_train.py --config configs/compress_then_train/qwen3_30b_a3b_rfid.yaml
# Nyström-MoE compress-then-fit @ 33% (lr tuned on a deep layer; see Settings)
python src/compress_then_train.py --config configs/compress_then_train/qwen3_30b_a3b_nystrom_moe.yaml
# Deep-layer lr sweep helper (closed-form prefix to L19, lr-scan at L20):
python src/compress_then_train.py --config configs/compress_then_train/qwen3_30b_a3b_nystrom_moe_sweep.yaml \
  --nystrom_max_layers 21 --nystrom_fit_from_layer 20 --nystrom_fit_lr_scan 3e-4,1e-4,3e-5
```

Prefix the 40 GB A100 runs with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_DEVICE_MAP_AUTO=1 PER_GPU_MEM=36GiB ATTN_IMPLEMENTATION=sdpa`.

### Raw result artifacts

- Pruning: `run_results/A100-New/results_eval/qwen3_nystrom_{25p,33p}_{hellaswag,mmlu}_*/lm_harness/`;
  uniform `run_results/A100-New/.../qwen3_nystrom_uniform33_{hellaswag,mmlu}_*/lm_harness/`.
- MoBE 25%: run `ce_mobe_calib-c4-0.75_1.0e-04-0715-005135`, JSON at `docs/results/mobe/mobe_benchmark_comparison.json`.
- MoBE 33%: run `ce_mobe_calib-c4-0.67_1.0e-04-0716-070717` (A100-Sagemaker, 8 GPUs), checkpoint at
  `outputs/compress_then_train/ce_mobe_calib-c4-0.67_1.0e-04-0716-070717/compressed_model/{mobe_native, hf_reconstructed}`.
- RFID: run `ce_rfid_calib-c4-0.625_1.0e-04-0714-184003`, JSON at `docs/results/mobe/rfid_benchmark_comparison.json`.
- Nyström-MoE: run `ce_nystrom_moe_calib-c4-0.67_1.0e-04-0716-103639`, JSON at
  `docs/results/mobe/nystrom_moe_benchmark_comparison.json`.

---

## Notes & caveats

- **No fine-tuning anywhere.** Every number is one-shot. Both families should improve with a LoRA/CE
  recovery pass; the factorization gap to the papers' ~96–98% retention is partly the missing
  recovery step (and, for RFID, the omitted residual module + short-cap fit).
- **Cross-family comparisons are indicative, not rigorous** — different base checkpoints (Thinking-2507
  vs A3B), different reduction axes (overall vs MoE-layer params), and an un-rerun MMLU baseline for
  factorization. To make it rigorous: re-run one family on the other's base checkpoint and re-measure
  the uncompressed MMLU baseline.
- **Active vs storage params (25% pruning).** A 25% storage cut yields only ~1.4% *active*-compute cut
  (full-model active ratio 0.986): attribution-guided pruning strips the least-routed experts'
  channels, which rarely enter a token's top-8. This checkpoint delivers real memory savings but
  **not** a proportional FLOPs speedup. See `docs/results/attribution_guided/nystrom.md` for the full
  active-param distribution.

```

```
