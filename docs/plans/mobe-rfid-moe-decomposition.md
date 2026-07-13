# Plan: Add MoBE and RFID-MoE decomposition methods

## Context

This repo has **two** compression subsystems:
- a **pruning** pipeline (`score → train(mask+LoRA) → merge_slim_eval`), and
- a **decomposition** pipeline: `src/compress/` + `src/compress_then_train.py` + `configs/compress_then_train/`, which already implements SVD / BTT / Nyström as one-shot factorizations followed by post-compression PPL + lm-eval benchmarking.

MoBE (`docs/papers/26_MoBE.pdf`) and RFID-MoE (`docs/papers/26_RFID-MOE.pdf`) are **decomposition** methods — both papers benchmark directly against SVD/D²-MoE that this subsystem already hosts — so they belong in `src/compress/`. Neither exists today (grep for `mobe`/`rfid` is empty).

**Scope (per user decisions):**
- Integrate into the `src/compress` decomposition path only.
- **One-shot decomposition only** — no LoRA/CE recovery fine-tuning wiring. (`compress_then_train.py`'s existing optional CE fine-tune stays available but off; configs set `kd_loss_type: ce`-free by using `train_mode` + benchmarks, or we run decompose-then-eval only.)
- Primary target: **Qwen3-30B-A3B** (128 experts/layer, SwiGLU SiLU, **no shared expert** — the clean case). DeepSeek/Qwen1.5 handling is deferred (their shared experts + DeepSeek dense layer-0 are noted below but not built now).

**Intended outcome:** two new `train_mode` values, `mobe` and `rfid`, that factorize every routed expert's `up_proj`/`gate_proj` (down_proj left dense — both papers do this) into a per-layer shared basis + per-expert transforms, replace those Linears in-place, and run the existing post-compression PPL + lm-eval benchmark. RFID-MoE is implemented **without** the residual-reconstruction module (per request), i.e. adaptive-rank + frequency-grouped MoBE.

## The shared math (why the two methods share a core)

Both reconstruct each expert up/gate weight `W_i ∈ R^{p×d}` (p = `moe_intermediate_size`, d = `hidden_size`) as
`Ŵ_i = A_i · f( Σ_j α_{i,j} B_j )`, with per-layer shared basis `{B_j ∈ R^{r×d}}_{j=1..m}`, per-expert transform `A_i ∈ R^{p×r}`, per-expert nonneg simplex coeffs `α_i ∈ R^m`, and `f = SiLU` (MoBE §3.3, RFID Eq. 4). Because `f` acts in **weight space** (on a fixed `r×d` matrix, not on activations), `Ŵ_i` is a **fixed matrix after fitting** — so for eval we can materialize `Ŵ_i` and run the stock expert forward; accuracy is exact. Compression ratio is measured from the **stored** factor param count (`2·n·p·r + m·r·d` per type vs `n·p·d`), not from the materialized weights.

**Differences (RFID = MoBE + 3 additions), residual omitted:**
1. **Frequency grouping:** collect per-expert routing counts on a small calibration set, sort experts by frequency, partition into `m` groups of `k=n/m` (RFID §3.1). MoBE instead shares one basis set across *all* experts in the layer.
2. **Effective rank** per group from its stacked-matrix SVD spectrum (`R_eff = exp(entropy(σ²/Σσ²))`, RFID §3.2).
3. **Adaptive rank allocation:** fuse normalized effective rank `E_g` and normalized frequency `F_g` as `C_g = ξ·E_g + (1−ξ)·F_g` (ξ≈0.8), allocate per-group basis rank `K_g ∝ C_g` under a total budget (RFID §3.3, Eq. 13).

MoBE uses uniform `r` (paper sets `r = d`) and uniform `m` (e.g. 32 for Qwen3). Z-score normalization (MoBE §3.4): fold σ into `A_i`, drop µ — implement as an optional flag defaulting on.

## Implementation

### 1. New module: `src/compress/moe_basis/` (new package)

`src/compress/moe_basis/basis_expert.py` — the factorized module, modeled on `SVDCompressedLinear` (`src/compress/svd/svd_linear.py`) contract (`.in_features`/`.out_features` props + `materialize_dense_weight()`):
- `MoBEProjection(nn.Module)` replacing one expert's `up_proj` **or** `gate_proj` Linear. Holds `A` (`p×r`, learnable during fit), a **reference to the layer-shared basis** `B` (`nn.ParameterList` of `m` × `r×d`, registered once per layer and shared across experts/type), `alpha` (`m`, softmax→simplex), optional `sigma` scale. `forward(x)`: compute `Ŵ = A @ f(Σ softmax(alpha)_j · B_j)` (cache it in eval), return `x @ Ŵ.T` (+ bias if any). `materialize_dense_weight()` returns `Ŵ.T` so downstream save/eval that expects dense weights still works.
- A small container that owns the per-layer, per-type (`up`/`gate`) shared basis so multiple `MoBEProjection`s point at the same `B`.

`src/compress/moe_basis/fit.py` — the fitter (data-free, Adam), following MoBE Algorithm 1 / RFID Eq. 5:
- `fit_layer_basis(W_stack, m, r, f, iters, lr, z_norm) -> (A_list, B_list, alpha_list)`: stack a layer's `n` expert matrices `[n,p,d]`, init (SVD warm-start of `A`, random `B`, uniform `alpha`), Adam-minimize `Σ_i ||W_i − A_i f(Σ_j α_{ij} B_j)||_F²`. Paper hyperparams: Adam lr 0.07, up to 50k steps, early-stop patience 2k on train loss, batch = n. Expose these as config knobs with paper defaults; use a smaller default iter cap for smoke runs.
- Reuse existing helpers where possible; this is pure weight-space optimization (no calibration data, no model forward).

`src/compress/moe_basis/mobe.py`:
- `mobe_compress_model(model, *, m, r, iters, lr, z_norm, skip_layers, device)`: iterate MoE blocks via `_is_moe_block` (`src/base/shared_utils/safe_isinstance.py`), get experts via `_get_experts`, read dims via `_get_moe_intermediate_size`/`_get_num_hidden_size`. For each layer and each type in {`gate_proj`,`up_proj`}: gather the `n` expert weight matrices, call `fit_layer_basis`, build the shared basis container, and `setattr(expert, "gate_proj"/"up_proj", MoBEProjection(...))` (the in-place replacement idiom from `nystrom_compress_model`, `src/compress/structured/nystrom.py:379` / `convert_linear_to_svd_compress`). Leave `down_proj` untouched. Log stored-vs-original param counts.

`src/compress/moe_basis/rfid.py`:
- `collect_expert_routing_counts(model, calib_loader, device) -> {layer_idx: Tensor[n]}`: forward-hook each MoE **router/gate** (`_get_router_gate`) on the calib loader, argtop-k the routing logits (`_get_topk`), accumulate per-expert selection counts. (Qwen gate is `nn.Linear`; DeepSeek `MoEGate` — handle both, but Qwen3 is the target.) This mirrors the routing-frequency infra that already exists as `gate_scores["usage"]` in the prune scorer (`src/calibration/channel_scoring/collector/attn_mlp.py`) but is self-contained inside `compress/`.
- `effective_rank(W_stack)`, `allocate_group_ranks(freq, eff_rank, ratio, xi, m)` (Eq. 8–13).
- `rfid_compress_model(model, calib_loader, *, m, ratio, xi, iters, lr, z_norm, ...)`: per layer, sort experts by frequency → `m` groups; per group compute effective rank + `C_g` + `K_g`; run `fit_layer_basis` **per group** with that group's rank `K_g`; replace `up`/`gate` on each expert in the group with a `MoBEProjection` bound to its group's basis. **No residual reconstruction** (the `η`/`P` mechanism from RFID §3.4 is intentionally omitted).

### 2. Register the methods (string-keyed dispatch, matching existing pattern)

Add `"mobe"`, `"rfid"` to the registries — these are **MoE-only** (like `nystrom` is MLP-only), so add a new `_MOE_ONLY_METHODS = {"mobe","rfid"}` rather than the SVD/BTT covariance sets:
- `src/compress/compress_model.py`: add `_MOE_ONLY_METHODS`; route it in `compress_model_with_loader` (mobe = calib-free like `svd`/`btt`; rfid needs the calib loader only for routing counts). Update `_normalize_method_name` to accept them.
- `src/compress_then_train.py`: add both to `_VALID_TRAIN_MODES` and the `train_mode` `choices` list; add a `_MOE_ONLY_TRAIN_MODES` set and branch in `decompose_model()` (dispatch to `mobe_compress_model`/`rfid_compress_model`; mobe skips the calib loader, rfid builds it). Add config knobs: `moe_basis_count` (m), `moe_basis_rank` (r), `moe_fit_iters`, `moe_fit_lr`, `moe_z_norm`, `rfid_xi`, and reuse `compression_ratio`.
- `src/compress/decomposition.py`: add both to `VALID_TRAIN_MODES` for library-level parity.

### 3. Config

`configs/compress_then_train/qwen3_30b_a3b_mobe.yaml` and `..._rfid.yaml`:
- `model_name_or_path: Qwen/Qwen3-30B-A3B`, `train_mode: mobe`/`rfid`, `moe_basis_count: 32`, `moe_basis_rank: 768` (≈ hidden_size for Qwen3), `compression_ratio` per target (RFID uses ~0.6–0.8 retain), `calib_source: wikitext2`/`c4` (rfid only), `eval_ppl_after_compression: true`, `kd_loss_type` unused (one-shot: run decompose + `run_lm_eval` + PPL, no training steps — set `ce_steps: 0` / skip training path), `lm_eval_tasks: hellaswag,mmlu` (+ PPL on wikitext2/ptb/c4 via `evaluate_model_ppl`), `skip_layers: lm_head`. Follow the 30B sharding env flags from memory ([[qwen3-30b-on-40gb-a100]]).

## Deferred / noted (not built now)
- **DeepSeek/Qwen1.5** shared experts (`mlp.shared_expert(s)`) and DeepSeek dense **layer-0** — `_get_experts` returns only routed experts and there is no shared-expert getter; `_is_moe_block` naturally skips DeepSeek layer-0. When extending, add a `_get_shared_expert` getter in `safe_isinstance.py` and decide whether to factorize shared experts.
- **RFID residual reconstruction** (`η` + sparse projection `P`, RFID §3.4) — omitted per request; leave a clearly-named stub/flag (`rfid_residual: false`) so it can be added later.
- **True inference-time param savings** (keeping factors un-materialized + a fused kernel) — MoBE's own stated limitation. For accuracy eval we materialize `Ŵ`; report compression from stored factor counts.

## Verification
End-to-end smoke (small first, following [[launch-on-a100-skill]] / [[a100-new-remote-runs]]):
1. **Unit / shape check** (local, CPU or 1 GPU): build a tiny synthetic MoE layer (or load Qwen1.5-MoE-A2.7B for speed), run `mobe_compress_model` with small `m,r,iters`, assert (a) `up_proj`/`gate_proj` became `MoBEProjection`, (b) `materialize_dense_weight()` shape `= [p,d]`, (c) reconstruction MSE decreases over fit iters, (d) `down_proj` unchanged. Add under `src/compress/tests/` (mirrors `test_nystrom.py`).
2. **Routing-count check** for RFID: run `collect_expert_routing_counts` on a few calib batches, assert per-layer counts sum to `num_tokens × top_k` and are non-uniform.
3. **Full pipeline on Qwen3-30B-A3B** (A100 box): `bash scripts/compress_then_train.sh` with `CONFIG=configs/compress_then_train/qwen3_30b_a3b_mobe.yaml` (and `_rfid`). Confirm the run reports post-compression **PPL (wikitext2/ptb/c4)** and **lm-eval (hellaswag/mmlu)** in `benchmark_comparison.json`, and that numbers are in the papers' ballpark (MoBE ~96% relative on Qwen3-30B at 24%; RFID ≥ MoBE). Compare `mobe` vs `rfid` at the same `compression_ratio` — RFID should match or beat MoBE on PPL.

## Key files to touch
- New: `src/compress/moe_basis/{__init__,basis_expert,fit,mobe,rfid}.py`, `src/compress/tests/test_moe_basis.py`, `configs/compress_then_train/qwen3_30b_a3b_{mobe,rfid}.yaml`.
- Edit: `src/compress/compress_model.py` (registries + dispatch), `src/compress_then_train.py` (`_VALID_TRAIN_MODES`, `decompose_model`, config fields), `src/compress/decomposition.py` (`VALID_TRAIN_MODES`).
- Reuse (no edit): `src/base/shared_utils/safe_isinstance.py` (getters), `src/compress/svd/svd_linear.py` (module contract), `src/compress/structured/nystrom.py` (replacement idiom), `src/compress/ppl_eval.py` + `eval/lm_harness/` (benchmarks), `src/compress/loaders.py` (calib loader for RFID routing counts).
