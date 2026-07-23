# Nyström compress-then-fit for MoE experts (activation-aware, sequential)

## Context

The repo already has two Nyström surfaces (see `docs/plans/mobe-rfid-moe-decomposition.md`):
a **dense-MLP** intermediate-dim compressor (`src/compress/structured/nystrom.py`, MoDeGPT
Alg. 1) and a **per-expert** closed-form `down_proj` reconstruction used inside the *pruning*
eval path (`src/prune/apply/slimming/expert_slim.py`). Neither is wired into the MoE
**decomposition** pipeline (`src/compress_then_train.py` → `compress_model.py` → `moe_basis/`)
the way MoBE/RFID are.

We want a new decomposition method — call it **`nystrom_moe`** — that mirrors MoBE's
*one-layer-at-a-time* structure but, per layer, does:

1. **Nyström channel select** — rank each expert's intermediate channels by ridge
   leverage `diag((C+λI)⁻¹C)` on its `down_proj`-input covariance `C = zᵀz/N`, keep the
   top `k` (uniform `k` across experts so the compressed model is a standard, smaller
   Qwen3-MoE).
2. **Closed-form init** — reconstruct `down_proj` on the kept subset via the existing
   Nyström solve; slice `gate_proj`/`up_proj` rows to the kept channels.
3. **Activation-aware local fit** — Adam-refine the narrowed `{gate_k, up_k, down_k}` of
   each expert to minimize `‖expert_k(X) − Y_ref‖²` on that expert's captured routed
   inputs `X`, with `Y_ref` = the *original* expert's output. Settings aligned with
   MoBE's fitter (Adam lr≈0.07, ~2–3k iters, best-state snapshot, float32).

**Sequential:** compress decoder layers in depth order; before compressing layer ℓ, push
the calibration batch through the *already-compressed* prefix `0..ℓ-1` so layer ℓ's expert
inputs (and thus `C`, `X`, `Y_ref`) reflect the distribution it will actually receive at
inference (same re-linearization idea as `src/compress/sequential/relinearized.py`).

Unlike MoBE, experts stay **plain dense** gated MLPs (just narrower), so **no custom
`nn.Module`, no factor checkpoint, no `save.py` changes** — the result is a standard HF
model with a smaller `moe_intermediate_size`.

First target: **Qwen3-30B-A3B, 33% reduction of expert-FFN params** (`k = round(0.67·p)`,
aligned to a multiple of 128 → `p=768 → k=512`, giving exactly 1−512/768 = 33.3%; gate/up/down
all shrink). C4 calibration. Tune lr/iters on the first few MoE layers, then full-model run
on A100.

## Files to create / modify

### 1. New compressor — `src/compress/moe_basis/nystrom_moe.py`

The core, mirroring `mobe.py`'s layer loop. Reuse helpers rather than reinvent:
- `_iter_moe_blocks`, `_stack_expert_weights` from `mobe.py`.
- `compute_ridge_leverage_scores` from `src/calibration/channel_scoring/leverage.py`.
- `_nystrom_reconstruct_down_proj` from `src/prune/apply/slimming/expert_slim.py` (import it;
  it already does the escalating-ridge Cholesky solve). If a cross-package import is awkward,
  copy the ~50-line function into this module with a comment pointing back to the source.
- Per-expert I/O capture: adapt the forward-hook pattern from
  `src/calibration/channel_scoring/collect_covariance.py` (`_make_cov_hook`) — but capture the
  **expert input `x`** (input to `gate_proj`), not just the `down_proj`-input covariance.

Key functions:

```python
def nystrom_moe_compress_model(
    model, calib_loader_factory, *,
    keep_ratio: float = 0.67,      # fraction of intermediate channels kept (0.67 = -33%)
    align_to: int = 128,           # round k to a multiple for hardware-friendly shapes
    lambda_ridge: float = 1.0,
    fit: bool = True,              # False => closed-form only (no Adam)
    fit_iters: int = 3000, fit_lr: float = 0.07, fit_patience: int = 0,
    snapshot_every: int = 200,
    max_fit_tokens: int = 8192,    # per-expert cap on captured rows X (memory bound)
    device: str = "cuda",
    max_layers: Optional[int] = None,   # sweep aid: compress only first N MoE layers
    seed: int = 0, log_every: int = 500,
) -> nn.Module:
    for layer_idx, block in _iter_moe_blocks(model):   # depth order
        if max_layers and processed >= max_layers: break
        experts = _get_experts(block)
        # (a) capture per-expert routed inputs X through the COMPRESSED prefix:
        #     one hooked forward over calib_loader_factory(); accumulate down_proj-input
        #     covariance C (zᵀz/N) AND a capped sample of gate_proj inputs X per expert.
        # (b) for each expert: leverage-rank C, keep top-k (uniform k), closed-form down init,
        #     row-slice gate/up. If fit: compute Y_ref = orig_expert(X), Adam-fit narrowed
        #     {gate_k, up_k, down_k} to MSE(expert_k(X), Y_ref), snapshot best.
        # (c) swap expert.gate_proj/up_proj/down_proj for the new narrower nn.Linear (reuse
        #     nystrom.py:_make_linear); free X/Y_ref; torch.cuda.empty_cache().
    model.config.moe_intermediate_size = k     # uniform shrink -> standard HF model
    # log realized reduction = 1 - 3kd / 3pd
```

Notes:
- **Capture step (a)** registers, per expert, a `gate_proj` forward-pre-hook (grab `x`,
  reservoir-cap at `max_fit_tokens`) and a `down_proj` forward-hook (accumulate `zᵀz/N` for
  `C`). Both come from the *same* forward sweep. `_get_experts`/`_is_moe_block` skip dense
  layers (DeepSeek layer 0) naturally.
- **`Y_ref`** is computed from the original (not-yet-swapped) expert weights on the captured
  `X`, in chunks, right before the swap.
- The per-expert Adam fit is small (k×d + d×k params, ≤8k rows) — batch across experts on GPU
  where it fits, else loop. Match MoBE's snapshot-best-state + float32 conventions from
  `fit.py`.
- Uses `calib_loader_factory` (zero-arg, returns a fresh loader) because a fresh pass is needed
  per layer — same contract as `relinearized.py`. The C4 loader
  (`compress/loaders.py:build_c4_calib_loader`) materializes windows, so a factory that rebuilds
  it is cheap.

### 2. Register the method — `src/compress/compress_model.py`

- Add `"nystrom_moe"` to `_MOE_ONLY_METHODS` (line ~75). It needs calibration data, so do
  **not** add it to `_MOE_CALIB_FREE_METHODS`.
- In `_compress_moe_basis` (line ~501), add a branch dispatching `nystrom_moe` to
  `nystrom_moe_compress_model`, reading its knobs from `moe_kwargs`. It needs a loader
  **factory**; the simplest wiring is to pass the already-built `calib_loader` plus a
  rebuild thunk, or accept the loader and internally re-iterate (C4 loader is re-iterable
  since it's a materialized `DataLoader`, not a stream — confirmed in `loaders.py`). Prefer
  passing a factory built in `decompose_model`.

### 3. Wire config knobs — `src/compress_then_train.py`

- Add `"nystrom_moe"` to `_VALID_TRAIN_MODES`, `_MOE_ONLY_TRAIN_MODES`, and the `train_mode`
  `choices` list. Leave it **out** of `_CALIB_FREE_TRAIN_MODES` (needs C4).
- Add `KDDecompositionConfig` fields (near the existing MoBE knobs, line ~285):
  `nystrom_keep_ratio` (0.67), `nystrom_align_to` (128), `nystrom_lambda_ridge` (1.0),
  `nystrom_fit` (True), `nystrom_fit_iters` (3000), `nystrom_fit_lr` (0.07),
  `nystrom_fit_patience` (0), `nystrom_max_fit_tokens` (8192), `nystrom_max_layers` (None,
  sweep aid). Reuse `moe_fit_log_every`/`seed`/`calib_*` that already exist.
- In `decompose_model` (line ~1145), extend the `moe_kwargs` block so these are forwarded when
  `train_mode == "nystrom_moe"`. Build the loader factory here (it already builds a
  `calib_loader` via `_build_calib_loader`; wrap that call in a lambda).
- **Save:** since experts stay dense and `config.moe_intermediate_size` is updated, the model
  is a standard smaller HF checkpoint. Add a small branch so that for `nystrom_moe` the
  post-compress save uses `model.save_pretrained` (+ tokenizer) into
  `<run_dir>/compressed_model/hf_reconstructed/` instead of the MoBE `save_compressed_model`
  path (which scans for `MoBEProjection`). The existing one-shot eval + PPL + benchmark JSON
  flow needs no change.
- Mirror the `_VALID_TRAIN_MODES` addition in `src/compress/decomposition.py`
  (`VALID_TRAIN_MODES` / `MOE_ONLY_TRAIN_MODES`) for parity.

### 4. Configs — `configs/compress_then_train/`

- `qwen3_30b_a3b_nystrom_moe_sweep.yaml` — sweep: `train_mode: nystrom_moe`,
  `nystrom_max_layers: 3`, `one_shot_eval_only: true`, `run_lm_eval: false`,
  `eval_ppl_after_compression: true` (fast signal). This run just prints per-layer/expert
  `rel_err` for the lr/iters grid.
- `qwen3_30b_a3b_nystrom_moe.yaml` — full run, modeled on `qwen3_30b_a3b_mobe.yaml`:
  `model_name_or_path: Qwen/Qwen3-30B-A3B`, `nystrom_keep_ratio: 0.67`,
  `nystrom_align_to: 128`, tuned `nystrom_fit_lr`/`nystrom_fit_iters`, `calib_source: c4`,
  `one_shot_eval_only: true`, `baseline_skip_tasks: mmlu`, `lm_eval_tasks: hellaswag,mmlu`,
  `lm_eval_limit: -1`. Set the same A100 sharding env flags as the MoBE run.

### 5. Test — `src/compress/tests/test_nystrom_moe.py`

Mirror `test_moe_basis.py` on a tiny synthetic gated-MoE (or the smallest available):
(a) captured-`X` activation MSE decreases over the Adam fit; (b) after compress, each expert's
`gate/up/down` have the narrowed `k`; (c) `config.moe_intermediate_size == k`; (d) closed-form
(`fit=False`) path runs and produces a lower activation error than plain column-slicing;
(e) dense (non-MoE) layers are left untouched.

## Verification

1. **Unit tests** — `python -m pytest src/compress/tests/test_nystrom_moe.py` (repo has no
   global suite; run this file directly with `PYTHONPATH=$(pwd):$(pwd)/src`).
2. **Local smoke** — run the sweep config with `nystrom_max_layers: 1`, `nystrom_fit_iters: 100`
   on a small MoE if one fits locally, or on A100; confirm the log shows decreasing per-expert
   `rel_err` and the "realized reduction ≈ 33%" line.
3. **lr/iters sweep on A100** (via `launch-on-a100` skill) — run
   `qwen3_30b_a3b_nystrom_moe_sweep.yaml` over a small grid
   (`nystrom_fit_lr ∈ {0.07, 0.02, 0.005}`, `nystrom_fit_iters ∈ {1500, 3000}`), compare the
   post-compression PPL / mean per-expert `rel_err` on the first 3 MoE layers. Pick the best.
4. **Full run on A100** — launch `qwen3_30b_a3b_nystrom_moe.yaml` with the tuned settings
   (env: `FORCE_DEVICE_MAP_AUTO=1 PER_GPU_MEM=36GiB ATTN_IMPLEMENTATION=sdpa
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, 4 GPUs). Compare HellaSwag/MMLU/PPL in
   `benchmark_comparison.json` against the MoBE/RFID 25% numbers and the attribution-guided
   pruning results at ~33% in `docs/results/`.

## Key reuse map (do not reinvent)

| Need | Reuse |
|---|---|
| Layer loop over MoE blocks | `moe_basis/mobe.py:_iter_moe_blocks`, `_stack_expert_weights` |
| Ridge leverage scoring | `calibration/channel_scoring/leverage.py:compute_ridge_leverage_scores` |
| Closed-form down_proj reconstruct | `prune/apply/slimming/expert_slim.py:_nystrom_reconstruct_down_proj` |
| Per-expert forward-hook capture | `calibration/channel_scoring/collect_covariance.py:_make_cov_hook` |
| Sequential re-linearization pattern | `compress/sequential/relinearized.py` |
| Building narrowed `nn.Linear` | `compress/structured/nystrom.py:_make_linear` |
| Adam fit conventions (snapshot, fp32, z-norm) | `moe_basis/fit.py:fit_layer_basis` |
| Model getters | `base/shared_utils/safe_isinstance.py` (`_get_experts`, `_get_moe_intermediate_size`, ...) |
