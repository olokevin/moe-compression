# MoBE & RFID-MoE — Initial Results (Qwen3-30B-A3B, ~25% MoE-layer param reduction)

**Status:** RFID-MoE complete (2026-07-15, A100-Sagemaker). MoBE complete (2026-07-15,
A100-New) — reference-matched fitter, 2000 fit steps/(layer,type). See both rows below.

## Setup

- **Model:** Qwen/Qwen3-30B-A3B (hidden `d=2048`, `moe_intermediate p=768`, `n=128` experts,
  top-k 8, 48 layers, SwiGLU/SiLU, no shared expert).
- **Methods:** MoBE (data-free) and RFID-MoE (residual reconstruction **omitted**), both
  factorizing every routed expert's `gate_proj`/`up_proj` into a shared per-layer basis +
  per-expert transform; `down_proj` left dense. Implemented in `src/compress/moe_basis/`.
- **Compression target:** exactly **25% reduction of the total MoE-layer parameters**.
  - MoBE: `m=32` shared bases, rank `r=p=768` → up+gate shrink to γ=0.625 → total MoE −25%.
  - RFID: `m=32` frequency groups, `compression_ratio=0.625` retain of up+gate → total MoE −25%.
    Fusion `ξ=0.8`. Routing counts collected from C4 calibration (128 seqs × 1024 tok).
- **Fit (RFID run):** Adam lr 0.07, ≤3000 steps/(layer,type,group), early-stop patience 500,
  Z-score norm on. (The fitter was subsequently rewritten to match the reference `inclusionAI/MoBE`
  trainer — grouped-SVD init, std-only norm, mean-MSE — so a fresh run would fit longer.)
- **Fit (MoBE run):** reference-matched trainer (std-only norm, mean-MSE), Adam lr 0.07,
  **2000 fixed steps/(layer,type)** (patience 0), full 48 layers. Every `gate_proj`/`up_proj`
  converged uniformly from `rel_err≈0.97` at step 0 to `rel_err≈0.33` (mse≈0.11) at step 2000.
- **Mode:** one-shot decompose + eval, **no LoRA/CE recovery training** (`one_shot_eval_only`).
- **Eval:** lm-eval-harness — HellaSwag (0-shot, acc_norm) and MMLU (5-shot, acc); PPL on
  wikitext2 + c4. Full tasks (`lm_eval_limit=-1`).
- **Launch env (30B on 40GB A100):** `FORCE_DEVICE_MAP_AUTO=1 PER_GPU_MEM=36GiB
  ATTN_IMPLEMENTATION=sdpa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; RFID sharded across
  4 GPUs, MoBE across 8 GPUs (~9 GiB/GPU).

## Results

Eval protocol matches `docs/results/attribution_guided/nystrom.md`: HellaSwag full 10042 items
`num_fewshot=0` (acc_norm), MMLU full 14042×57 subtasks `num_fewshot=5`; each task in its own
lm-eval call. Baseline PPL: **wikitext2=8.7029, c4=14.0549**. Baseline MMLU not re-run (known
82.0 from the nystrom doc); baseline HellaSwag from this run.

| Model | HellaSwag (acc_norm) | MMLU (acc, 5-shot) | wikitext2 PPL | c4 PPL |
|---|---|---|---|---|
| Baseline (uncompressed) | **77.68** | 82.0 (nystrom doc) | 8.7029 | 14.0549 |
| **MoBE (−25% MoE params)** | **73.67** | **77.23** | **9.5937** | **15.9809** |
| RFID-MoE (−28.4% MoE params) | 66.80 | 71.32 | 12.6776 | 21.4943 |

Sources: MoBE `benchmark_comparison.json` (run `ce_mobe_calib-c4-0.75_1.0e-04-0715-005135`), copied
to `docs/results/mobe/mobe_benchmark_comparison.json`; RFID (run
`ce_rfid_calib-c4-0.625_1.0e-04-0714-184003`), copied to
`docs/results/mobe/rfid_benchmark_comparison.json`.

**MoBE one-shot (no recovery training) deltas vs baseline:**
- HellaSwag: 77.68 → **73.67** (−4.0 pts)
- MMLU (5-shot): 82.0 → **77.23** (−4.8 pts, vs the nystrom-doc baseline; MMLU baseline not re-run)
- wikitext2 PPL: 8.70 → **9.59**; c4 PPL: 14.05 → **15.98**

**RFID one-shot (no recovery training) deltas vs baseline:**
- HellaSwag: 77.68 → **66.80** (−10.9 pts)
- MMLU (5-shot): 82.0 → **71.32** (−10.7 pts)
- wikitext2 PPL: 8.70 → **12.68**; c4 PPL: 14.05 → **21.49**

**RFID actual compression:** the fit reported up+gate `stored/orig = 0.5739` → **28.4% total
MoE-layer reduction** (a bit beyond the 25% target — the adaptive per-group rank allocator rounds/
caps ranks and undershoots the 0.625 retain budget). So these numbers are a mildly conservative
read for the −25% point.

**Interpretation.** MoBE with the reference-matched fitter (one-shot, **no recovery fine-tuning**,
2000 fit steps) loses only **~4–5 pts** at exactly 25% MoE-layer compression (HellaSwag −4.0, MMLU
−4.8), holding up markedly better than RFID's ~11-pt drop at 28% compression. MoBE also lands close
to the attribution-guided *pruning* baseline at 25% (HellaSwag 78.45, MMLU 76.04, see
`docs/results/attribution_guided/nystrom.md`) — competitive on MMLU (77.23 vs 76.04), a few points
behind on HellaSwag (73.67 vs 78.45). The MoBE↔RFID gap is expected: (1) RFID's headline retention
relies on the **residual reconstruction module (§3.4), which we intentionally omitted**, and (2) the
RFID row used the earlier short-cap fitter, whereas the MoBE row used the rewritten reference-matched
trainer (std-only norm, mean-MSE). A fresh RFID run on the new fitter (and/or longer fits toward the
paper's ≤50k steps + a recovery pass) would tighten both rows further toward the paper's ~96–98%
retention.

**Note (2026-07-15):** MoBE has now been re-run to completion on the reference-matched fitter (all
48 layers, 2000 steps each, run `ce_mobe_calib-c4-0.75_1.0e-04-0715-005135`). The RFID row above
still comes from the earlier short-cap fitter, so the two rows are **not yet apples-to-apples** on
fit quality — a fresh RFID run on the new fitter is the remaining piece for a matched comparison.

Paper reference points (Qwen3-30B-A3B, ~24% total reduction, WITH residual + full fit): MoBE
retains ~96–98% relative performance; RFID ≥ MoBE. Papers report the 30B-A3B-**2507** checkpoint
and a broader suite; our HellaSwag/MMLU one-shot numbers are the directly comparable slice.
