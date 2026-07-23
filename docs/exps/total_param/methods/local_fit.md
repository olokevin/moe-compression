# Nyström-MoE local-fit — settings, behaviors, results (Qwen3-30B-A3B, −33% MoE FFN)

**Status:** one-shot factorize-and-fit complete (2026-07-16/17, A100-New,
Qwen3-30B-A3B-**Thinking-2507**). Diagnosis of the small fit→accuracy gap done
2026-07-17. Full diagnosis: `docs/results/total_param/plan/nystrom_fit_diagnosis.md`;
plan: `docs/results/total_param/plan/nystrom_fit.md`.

## What the method is

`nystrom_moe` (`src/compress/moe_basis/nystrom_moe.py`) — a MoE analog of MoBE's
one-layer-at-a-time flow, but instead of a shared basis it **narrows every expert**
(gate/up/down all shrink) and re-fits it. Per MoE block, in depth order (so layer ℓ
sees the already-compressed prefix — re-linearization):

1. **Capture** each expert's `down_proj`-input covariance `C = zᵀz/N` in one calib
   sweep (C4, 128 seqs × 1024 tok). Layer mode also captures the block's
   `(input, output)` pairs.
2. **Select** the top-`k` intermediate channels per expert by ridge leverage
   `diag((C+λI)⁻¹C)`, uniform `k` across experts.
3. **Closed-form init**: reconstruct `down_proj` on the kept subset
   (`W_down^T = (SᵀCS)⁻¹(SᵀC)W_down^T`); row-slice `gate/up`.
4. **Activation-aware local fit** (the subject of this doc): Adam-refine the
   narrowed weights to reduce block-output reconstruction error, best-state seeded
   with the closed-form init (fit can never regress below it).

Result is a **standard HF checkpoint** with `moe_intermediate_size = k` (no custom
module) — `save_pretrained`, loads in lm-eval/vLLM directly.

## Settings

- **Target:** `keep_ratio=0.67`, `align_to=128` → `k=512/768` → **exactly −33.3%**
  of expert-FFN params (gate+up+down all shrink; unlike MoBE which keeps `down`
  dense and only factorizes gate/up).
- **Fit mode** (`nystrom_fit_mode`, default `layer`):
  - **`layer`** — fit ALL experts of a block **jointly** against the block-output
    MSE, replaying the FROZEN router gate so each token's gradient reaches only its
    top-k experts (`_moe_forward_functional`, mirrors `Qwen3MoeSparseMoeBlock`).
    One Adam run/layer over stacked `(E,k,H)`/`(E,H,k)` weights. `layer_fit_tokens`
    (65536) caps captured block-I/O rows. ~5 min/layer on a 4-GPU 40GB shard.
  - **`expert`** — fit each expert separately vs its own output MSE (128 Adam
    runs/layer; slower, higher per-expert fidelity).
- **Optimizer:** Adam, **lr=3e-4** (tuned on a deep probe layer — see below),
  **800 iters**, cosine lr decay to 5%, best-state snapshot every 200 steps.
- **Robustness:** `_robust_leverage_scores` (escalating ridge + `diag(C)` fallback)
  for singular covariances; closed-form reconstruct wrapped with a finiteness
  guard → column-slice fallback; fit skips non-finite `Y_ref`/init. On healthy C4
  only ~1 isolated deep expert is singular; **mass singularity ⇒ bad calibration
  draw** (seen during an HF C4 504 outage — every expert from L14 went singular).
- **LR must be tuned on a DEEP layer, not L0.** Sweep knobs `nystrom_fit_from_layer`
  (closed-form prefix so a deep probe is reachable in ~25 min) + `nystrom_fit_lr_scan`
  (try each lr from the same init, keep best). L20 scan `{3e-4,1e-4,3e-5}` → **3e-4
  wins** (block-MSE 3.30e-4→1.25e-4, 2.6×); 1e-4→2.2×; 3e-5→1.6×. lr=1e-3 (tuned on
  shallow L0) **diverges at depth** and falls back to closed-form.
- **Launch env (30B on 40GB):** `FORCE_DEVICE_MAP_AUTO=1 PER_GPU_MEM=36GiB
  ATTN_IMPLEMENTATION=sdpa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, 4 GPUs.
- **wandb curves:** set `NYSTROM_WANDB_PROJECT=q3_30b_nystrom_local_fit`
  (`WANDB_MODE=offline` on the box — no auth); `nystrom_snapshot_every` is the
  curve cadence.

## Behaviors (per-layer reconstruction, full 48-layer run, lr=3e-4/800, layer mode)

The fit **does work** — every layer's block-MSE drops, best-state guarantees it
never regresses. But the benefit is mis-allocated across depth:

| depth bucket | mean MSE improvement (init→final) | final rel-error |
|---|---|---|
| L0–L11 (shallow) | 2.26× | ~0.11–0.23 |
| L12–L35 (mid) | **2.59×** | init ~0.32–0.38 → **39–44% drop** to ~0.19–0.23 |
| L36–L47 (deep) | **1.74×** (L47 = 1.00×, no-op) | ~0.15–0.20 |

- **Absolute reconstruction MSE grows ~2579× from L0 (3.7e-6) to L47 (9.4e-3).**
  Deep experts are higher-rank; uniform k=512 is too aggressive there.
- The fit gives its **biggest relative gains in the middle** (L11–L35, 39–44% rel
  drop) but **barely touches the deep layers** (L36–L47, ~1.7× and falling), which
  carry the **largest absolute error**. It fixes the cheap layers and fails the
  expensive ones.
- **Residual per-block rel-error ~0.20 even after fitting**, compounding over 48
  layers → the network output is heavily corrupted regardless.

## Results

Baseline (Thinking-2507): HellaSwag 78.57, PPL wikitext2 7.29 / c4 12.46.
One-shot, **no LoRA/CE recovery**.

| Setting | HellaSwag | MMLU (5-shot) | wiki2 PPL | c4 PPL |
|---|---|---|---|---|
| Baseline (uncompressed) | **78.57** | — | 7.29 | 12.46 |
| Nyström-MoE −33%, **fit** (lr=3e-4/800, layer mode) | **65.46** | 60.92 | 13.46 | 17.98 |

The fitted vs closed-form-only HellaSwag gap is **small** — the finding under
investigation. Cause: the fit helps mid layers (already the cheapest) and fails
the deep layers that dominate end-to-end error; the surviving ~0.20 per-block
error compounds over depth. Source: run
`ce_nystrom_moe_calib-c4-0.67_1.0e-04-0716-103639`,
`docs/results/mobe/nystrom_moe_benchmark_comparison.json`.

### Comparison — MoBE at 33% (partial, per-layer only)

The dedicated **MoBE 33%** run (`m=16`, `r=768` → γ_ug=0.5 → −33% MoE FFN, config
`qwen3_30b_a3b_mobe_33.yaml`) **OOM-crashed at ~layer 9** (shared-GPU memory
pressure) — no eval produced. Its per-layer `rel_err` (gate/up only; `down` dense):

| layer | gate_proj rel_err | up_proj rel_err |
|---|---|---|
| L0 | 0.354 | 0.376 |
| L2 | 0.449 | 0.463 |
| L4 | 0.455 | 0.470 |
| L6 | 0.453 | 0.473 |
| L8 | 0.458 | 0.470 |

**MoBE 33% per-layer rel-error (0.35–0.47, rising with depth) is ~2× WORSE than
Nyström-MoE's fitted 0.11–0.24** — Nyström reconstructs each block far better
per-layer (it fits all three matrices with activation-aware selection, vs MoBE's
data-free shared-basis factorization of gate/up only). Yet the two land near the
same one-shot accuracy tier, underscoring that **per-layer reconstruction quality
at this magnitude does not translate to end-to-end accuracy** because of depth
compounding. (The only COMPLETE MoBE run is at **25%**: HellaSwag 73.67, MMLU
77.23 — see `mobe_rfid.md`; no complete MoBE 33% eval exists.)

## Corrections to earlier hypotheses (2026-07-17)

- **"Relative-MSE loss fixes the deep-layer collapse" — WRONG.** Adam is already
  scale-invariant (a constant `1/‖y‖²` factor cancels in the `m/√v` update), so
  `rel_loss` is a no-op *within* a layer — confirmed empirically: an L20 rel-loss
  scan at lr {1e-3,3e-3} gave final==init (all diverged/fell back). The deep-layer
  failure is about **lr transfer across depth** (a fixed lr mis-steps
  different-scale layers), fixed by tuning lr on a deep probe — not by rescaling
  the loss. `nystrom_rel_loss` is retained as an option but is not the fix.
- **"Iterations too few" — not the primary cause.** Mid-layer fits converge and
  still leave ~0.20 rel-error; more iters won't cross the compounding barrier.

## Proposed better fitting strategy (next)

1. **Sequential hidden-state target** (fixes compounding): fit block ℓ so the
   *compressed* model's hidden state tracks the *original* model's cached hidden
   state at depth ℓ, so later layers correct upstream drift (GPTQ/AWQ-style
   sequential calibration). Same one-shot cost (one cached teacher forward).
2. **Non-uniform keep budget** (fixes depth error growth): allocate more channels
   to deep, high-error layers under the same global 33% (RFID-style adaptive rank
   by per-layer reconstruction error / effective rank).
3. **LoRA/CE recovery** after compression — one-shot factorize-and-fit is not
   competitive with attribution-guided pruning at 33% (HS 78.40 / MMLU 73.00) on
   this model; a short recovery pass is the standard remedy and is pipeline-supported.
