# Plan: `mix_btt` — Mixture-of-Basis BTT decomposition (MoBE-style), on Qwen3-8B FFN

## Context

We want to port MoBE's *mixture-of-basis + weight-space nonlinearity* idea onto the
**BTT (Block-Tensor-Train)** decomposition that already exists in this repo
(`src/compress/btt/`). BTT in `output_one_block` mode factorizes a Linear
`W:(d_out,d_in)` into per-input-block cores via block-wise SVD:

```
out = Σ_j L_jᵀ · (R_j · x_j)      # n input blocks; R_j:(rank,b) small/input core, L_j:(rank,d_out) large/output core
```
(confirmed: `resolve_btt_block_dims(d_out,d_in,"output_one_block") -> (m=1, n, a=d_out, b)` with
`n,b = _closest_factor_pair(d_in)`; forward = `BTTLinear.forward`, `btt_linear.py:110`.)

**`mix_btt` generalization** — treat the `n` small cores `{R_k}` as a *shared basis*, give
each block a learnable mixing row `α_j` over that basis, and insert the Qwen3 activation
`f = SiLU` between the small-core and large-core stages:

```
z_k = R_k · x_k                          # n basis latents
u_j = f( Σ_k α_{j,k} · z_k )             # per-block mixture + nonlinearity   (α: n×n, init = I)
out = Σ_j L_jᵀ · u_j
```

This is **exactly the MoBE reconstruction** `Ŵ_i = A_i · f(Σ_j α_{i,j} B_j)` with the
correspondence **expert ↔ input-block**: `A_j = L_jᵀ:(d_out,rank)`, `B_k = R_k:(rank,b)`,
`α:(n,n)`. Plain BTT is the exact special case `α = I, f = identity`, so the block-wise SVD
stays a valid (and, for the linear cell, exact) warm-start.

Two ways to insert `f` give the two fit regimes the user asked for:
- **Weight-space (MoBE-faithful):** `f` applied to the *mixed cores* → `Ŵ_j = L_jᵀ·f(Σ_k α_{j,k} R_k)`
  is a **dense** weight block (layer stays linear in `x`, materializable). Fit is **data-free** `‖W−Ŵ‖²`.
- **Activation-space (BTT data-path):** `f` applied on the *data path* → nonlinear in `x`, no dense `Ŵ`.
  Fit is `E_x‖W·x − mixBTT(x)‖²` over captured calibration activations (lr-sensitive → needs tuning).

Target model is **Qwen3-8B — a DENSE model** (`Qwen3MLP`, one FFN per layer, `hidden_act="silu"`);
`α` mixes over the `n` BTT input-blocks *within each Linear* (per-Linear analog of MoBE, not cross-expert).
Confirmed with the user.

## First launch: gate_proj only, 33% per-projection reduction, local-fit only (no LoRA/CE)

Compress **only** `model.layers.*.mlp.gate_proj` at `compression_ratio = 0.67` (retain 67% ⇒ 33%
fewer params in that projection), evaluate after local fit **directly** (no CE recovery), on
**HellaSwag (0-shot) + MMLU (5-shot)**. Three ablation cells:

| Cell | `mix_space` | `f` | `α` (mix basis) | Local fit objective | Fitter reused |
|---|---|---|---|---|---|
| **1** weight-space, nonlin+mix | weight | SiLU (on mixed cores) | learnable | `‖W−Ŵ‖²` data-free | `fit_layer_basis` (verbatim) |
| **2** act-space, nonlin only | activation | SiLU (data path) | fixed = I | `E_x‖Wx−mixBTT(x)‖²` | new `fit_mix_btt_layer` |
| **3** act-space, nonlin+mix | activation | SiLU (data path) | learnable | `E_x‖Wx−mixBTT(x)‖²` | new `fit_mix_btt_layer` |

(Cell 1 vs 3 probes weight-space-MoBE vs activation-space-BTT for the full config; cell 2 vs 3
isolates the mixture-of-basis contribution in activation space.) Later phases (out of scope for
launch 1) extend to `up_proj` / `down_proj` and an all-projections run for the headline 33%.

## Implementation

### New file: `src/compress/btt/mix_btt_linear.py`

**`MixBTTLinear(nn.Module)`** — duck-typed to the BTTLinear / MoBEProjection contract
(`forward`, `in_features`/`out_features` props, `materialize_dense_weight`, `topology_spec`/`from_topology_spec`).
Params: `btt_l:(1,n·rank,d_out)`, `btt_r:(n,b,rank)` (reused verbatim from the SVD init packing),
`alpha:(n,n)` init `torch.eye(n)`. Flags `mix_space∈{"weight","activation"}`, `use_mix`, `use_nonlin`;
`_act_fn = _resolve_activation("silu" if use_nonlin else "identity")` (import from
`compress.moe_basis.basis_expert`).
- **When `use_mix=False`, register `alpha` as a buffer (fixed I), not a Parameter** — so it never
  enters the optimizer or the trainable-param count (cell 2).
- **`mix_space="activation"` forward** — clone `BTTLinear.forward` (`btt_linear.py:110-134`) but insert
  the mix between the R-reshape and the L-`bmm`:
  `right = einsum("xnb,nbk->xnk", x_blocks, btt_r)` → `z=(B,n,m,rank)` →
  `u = einsum("jk,bkmr->bjmr", alpha, z)` → `u = act_fn(u)` → permute to `(B,m,n·rank)` → `bmm(btt_l)`.
  **Mandatory unit test:** `α=I, f=identity` ⇒ byte-exact to `BTTLinear` output.
- **`mix_space="weight"` forward** — materialize once (cache in eval, invalidate on `train()`, like
  `MoBEProjection`): `Ŵ_j = L_jᵀ·f(Σ_k α_{j,k} R_k)`, concat over `j` → dense `Ŵ:(d_out,d_in)`, then `F.linear(x,Ŵ)`.
- **`materialize_dense_weight()`** — valid for `mix_space="weight"` (any `f`) and for
  `mix_space="activation"` only if `f=identity`; otherwise return `None` (nonlinear data-path has no dense equiv).

**`decompose_to_mix_btt(weight, rank, bias, *, mix_space, use_mix, use_nonlin, decomp_mode="output_one_block", device)`**
— call the existing `btt_llm_v2_decompose_layer(..., precomputed_whitening=None, decomp_mode="output_one_block")`
(`btt_llm_v2.py:143`) to get `btt_l/btt_r` via per-block SVD + `_pack_btt`, wrap into `MixBTTLinear` with `alpha=I`.

**Fit — reuse, don't rebuild:**
- *Cell 1 (weight-space)* reuses **`fit_layer_basis`** (`moe_basis/fit.py:106`) **verbatim**: reshape `W`
  into block stack `W_stack:(n, d_out, b)` (`W_j = W[:, j·b:(j+1)·b]`), call
  `fit_layer_basis(W_stack, m=n, r=rank, activation="silu", iters=…, lr=0.07)`. With `m=n`,
  `_grouped_svd_init` reproduces the plain-BTT block-SVD init with one-hot (=identity) `α`. Returns
  `A:(n,d_out,rank), B:(n,rank,b), alpha:(n,n)` → pack into `btt_l/btt_r`/`alpha`.
- *Cells 2 & 3 (activation-space)* — new **`fit_mix_btt_layer(module, X, W_orig, bias_orig, *, iters=1500,
  lr=3e-4, patience=0, snapshot_every=200, minibatch=4096, rel_loss=False, cosine=True, dev, tag)`**,
  modeled on `nystrom_moe.py::_fit_expert`/`_fit_layer_joint`: `Y_ref = X@W_orgᵀ(+b)` computed once in
  no-grad minibatches; `opt = Adam([p for p in module.parameters() if p.requires_grad])` (auto-respects the
  frozen-`α` cell); mean-MSE loss; deterministic minibatch cycling + optional `CosineAnnealingLR`;
  **best-state seeded with the SVD init so the fit can never regress**; load best back before returning.

**`capture_linear_inputs(model, calib_loader, target_names, cap_per_module, device)`** — reuse
`_InputSampler` + `_run_calib_sweep` from `nystrom_moe.py` (capped CPU fp32 collector +
`register_forward_pre_hook` capturing `inputs[0]`); build the loader via `build_c4_calib_loader`
(`src/compress/loaders.py`). `cap_per_module≈8192`, freed per-module right after its fit.

**`mix_btt_compress_modules(model, calib_loader, names, compression_ratio, *, mix_space, use_mix, use_nonlin, fit_lr, fit_iters, fit_patience, decomp_mode, device)`** — the driver: resolve rank via
`_resolve_ranks` (`btt_llm_v2.py:46`), capture X once for the `names` (activation cells only), then per
module: grab `module.weight.data` (the dense `W_orig`) → `decompose_to_mix_btt` → fit (cell 1 → `fit_layer_basis`;
cells 2/3 → `fit_mix_btt_layer`) → `setattr(parent, leaf, mixbtt)` (same pattern as `btt_llm_v2_compress_model`,
`btt_llm_v2.py:485-565`). Log stored/orig params **including `α` (n² per Linear)**.

### Integration edit points (primary path = `compression_rules`)

1. `src/compress/compress_model.py` — add `"mix_btt"` to `SUPPORTED_METHODS` (so the rules validator at
   `compress_then_train.py:1062` accepts it); add set `_MIX_BTT_METHODS = {"mix_btt"}`.
2. `src/compress_then_train.py::apply_compression_rules` (~L1054 flavour detection) — `mix_btt` does its
   own raw-X capture, so **skip** the covariance-flavour branches for it (don't set `need_forward`).
3. `src/compress_then_train.py::apply_compression_rules` (~L1114 dispatch loop) — add branch:
   `if method == "mix_btt": mix_btt_compress_modules(model, calib_loader, names, ratio, mix_space=…, use_mix=…, use_nonlin=…, fit_lr=…, fit_iters=…, fit_patience=…, decomp_mode=decomp_args.decomp_mode, device=device); continue`.
   `calib_loader` is already in scope.
4. New `KDDecompositionConfig` fields (mirror the `nystrom_*` block ~L328): `mix_btt_space:str="activation"`,
   `mix_btt_use_mix:bool=True`, `mix_btt_use_nonlin:bool=True`, `mix_btt_fit_lr:float=3e-4`,
   `mix_btt_fit_iters:int=1500`, `mix_btt_fit_patience:int=0`, `mix_btt_cap_tokens:int=8192`.
   **First-pass shortcut:** read these globally from `decomp_args` (one cell per run) rather than
   extending `_match_rule`/`_plan_compression_rules` to carry per-rule flags — fewer edits, and each
   launch uses a single cell anyway.
5. `src/compress_then_train.py` post-decomp trainability — add `configure_mix_btt_trainability`
   (analogous to `decomposition.py:189`) OR, since launch 1 is **fit-only / no training**, just ensure the
   `else` branch doesn't mis-handle `MixBTTLinear`; note `configure_btt_trainability` checks
   `isinstance(module, BTTLinear)` and will skip `MixBTTLinear` (fine for fit-only).

(Optional, deferred: top-level `--train_mode mix_btt` via `decomposition.py::VALID_TRAIN_MODES` +
`compress_model.py::compress_model_with_loader` dispatch — not needed for the FFN-only experiment.)

### Param math (retain 0.67; confirm Qwen3-8B dims from the checkpoint at runtime — public: H=4096, I=12288, 36 layers)

Per-Linear BTT params `≈ rank·(n·d_out + d_in)`; `α` adds `n²` (negligible, ~4 orders smaller).
gate_proj `(d_out=I=12288, d_in=H=4096)` → `(m=1, n=64, a=12288, b=64)`, `rank≈42` ⇒ retain ≈0.66,
`α`=64² . Rank auto-resolves via `_resolve_ranks` from the float ratio and clamps to `min(a,b)` — log the realized rank.

## Pre-study: local-fit lr / iterations (run BEFORE the full-model launch)

Per `docs/results/total_param/methods/local_fit.md` ("tune lr on a DEEP layer; shallow-tuned lr diverges at depth").
Pick 3 layers spanning depth (`layers.2`, `layers.18`, `layers.32`), target their `gate_proj`. Capture X once
per layer, then from the same SVD seed sweep `lr∈{1e-4,3e-4,1e-3,3e-3} × iters∈{500,1500,3000}` (amortize the
expensive capture, like `nystrom_moe`'s `fit_lr_scan`). Report `init_mse / final_mse / rel_error` per layer;
lock the best `(lr,iters)` per activation cell (cell 1's weight-space fit uses the fixed `lr=0.07/iters=30000`
MoBE default and doesn't need this sweep). Implement as a tiny standalone script
`scripts/mix_btt_fit_scan.py` (imports the new module + capture helper) — no full pipeline needed.

## Configs + A100 launch

Add three configs under `configs/compress_then_train/`, all cloned from `qwen2_5_0_5b_c4.yaml` with:
`model_name_or_path: Qwen/Qwen3-8B`, `attn_implementation: "sdpa"`, `one_shot_eval_only: true`
(fit-only, no CE), `compression_rules: [{pattern: "layers\\.\\d+\\.mlp\\.gate_proj$", method: mix_btt,
compression_ratio: 0.67}]`, `lm_eval_tasks: hellaswag,mmlu`, `lm_eval_limit: -1`, plus the `mix_btt_*` knobs:

- `qwen3_8b_mixbtt_gate_ws.yaml` — cell 1: `mix_btt_space: weight, use_mix: true, use_nonlin: true`
- `qwen3_8b_mixbtt_gate_act_nl.yaml` — cell 2: `mix_btt_space: activation, use_mix: false, use_nonlin: true`
- `qwen3_8b_mixbtt_gate_act_nl_mix.yaml` — cell 3: `mix_btt_space: activation, use_mix: true, use_nonlin: true`

Launch via the **launch-on-a100** skill / `a100.sh` (rsync up — never git; the `src/compress` submodule and
`loguru` must ship; A100-New first). Qwen3-8B bf16 ≈16 GB fits on **1 GPU per config** → run the 3 cells in
parallel on 3 GPUs. Example:
```
bash <skill>/scripts/a100.sh sync
bash <skill>/scripts/a100.sh launch -n 1 --name mixbtt_ws \
  --cmd '.venv/bin/python src/compress_then_train.py --config configs/compress_then_train/qwen3_8b_mixbtt_gate_ws.yaml'
# repeat for _act_nl and _act_nl_mix; then status / pull
```
(If the HF hellaswag cache errors with `Feature type 'List' not found`, `rm -rf ~/.cache/huggingface/datasets/hellaswag`.)

## Verification

1. **Unit (local, tiny):** in `src/compress/tests/`, build a small Linear; assert `MixBTTLinear(α=I,f=identity)`
   forward == `BTTLinear` forward (bit-exact); assert `fit_mix_btt_layer` reduces activation-MSE below init;
   assert cell-1 weight-space `materialize_dense_weight()` reconstructs and `fit_layer_basis` lowers `‖W−Ŵ‖`.
2. **Smoke (A100):** run one config with `lm_eval_limit: 16` and a couple layers to confirm the rules path
   dispatches `mix_btt`, capture+fit runs, and eval produces `benchmark_comparison.json`.
3. **Pre-study:** `scripts/mix_btt_fit_scan.py` prints per-layer rel-error grid; pick `(lr,iters)`.
4. **Full launch:** 3 configs, full HellaSwag+MMLU; compare the three cells' acc vs the uncompressed
   Qwen3-8B baseline (`eval_before_compression`) in `benchmark_comparison.json`; `pull` results.

## Risks / gotchas

- **α-axis placement:** mix must sit between the R-reshape `(B,n,m,rank)` and the L-collapse `(B,m,n·rank)`;
  mixing after the collapse would wrongly blend the rank dim. The `α=I,f=identity`==BTT test guards this.
- **`use_mix=False` → α as buffer** (not Parameter), else it leaks into the optimizer/param count.
- **Nonlinear (act-space) cells cannot materialize to dense** — keep the live `MixBTTLinear`; eval uses the
  in-memory model (as BTT/nystrom_moe do), so disk-save is best-effort/deferred (topology.py pattern if needed).
- **Fit target = weight at decomposition time:** grab `module.weight.data` before swapping.
- **Qwen3-8B dims unverified locally** — confirm H/I/#layers from the checkpoint config; ranks/pre-study layer
  indices are dim-driven and auto-adjust.
- **Rank clamp** `min(a,b)`: for gate_proj `b=64` bounds rank ≤64 (r≈42 ok); log realized rank.
