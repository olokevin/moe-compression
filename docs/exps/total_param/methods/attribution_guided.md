# Attribution-Guided Loss Approximation (ALA): per-expert scoring

## My addings

loss: 给expert加上随机扰动，这样才有最终的loss（而不是不perturb，得到的ntp loss）

用这个loss来得到contribution

## What ALA computes

ALA assigns each expert a **single calibration-averaged scalar** that estimates
*how much the model loss depends on that expert's output*, weighted by how often
the expert is actually routed. This scalar is stored on each expert under the key
**`expert_out_token_contrib`**.

Downstrea m, the intra-layer planner's default `attr_coverage` method uses this
scalar as the per-expert **coverage weight** to allocate the channel-keep budget
across experts within a layer (see `src/prune/generate/planners/intra_layer/`).

It is distinct from the per-channel *intra-expert* scores (`activation`,
`saliency`, `token_contrib`, `grad`, `wa`, `wg`, `weight`) that rank channels
*within* one expert — ALA operates at the whole-expert granularity.

## Where it is computed

`src/calibration/channel_scoring/collector/attn_mlp.py:61-64`

```python
total_tokens = float(attn_mask.sum().item())
usage = float(down_output.shape[0]) / max(total_tokens, 1.0)
expert_out_token_contrib = token_contrib(down_out_grad, down_output).sum() * usage
safe_add_with_ema(expert, ema, expert_out_token_contrib, "expert_out_token_contrib")
```

Helper `token_contrib` is defined in
`src/calibration/channel_scoring/collector/utils.py:31-66`.

## Step by step

1. **Per-token, per-channel attribution.**
   `token_contrib(g, z)` forms the elementwise product `c = g ⊙ z`, where

   - `z = down_output` — the expert's output (the `down_proj` output, i.e. the
     expert's contribution to the residual stream),
   - `g = down_out_grad` — the gradient of the loss w.r.t. that output.

   So `g ⊙ z ≈ (∂L/∂output) · output` is the **first-order Taylor approximation
   of the loss change if that output were zeroed** (i.e. if the expert were
   removed). This is the "loss approximation" at the heart of ALA.

   `token_contrib` then clips the top 1% of values per channel by absolute value
   (`trim_head=0.01`, robust to outlier tokens) and takes the **mean over the
   token dimension**, returning a per-output-channel vector of shape `[H]`.
2. **Sum over channels.**
   `.sum()` collapses the per-channel attribution into a single scalar: the total
   approximated loss contribution of the whole expert output.
3. **Scale by usage.**
   `usage = (# tokens routed to this expert) / (total valid tokens)`.
   Multiplying by `usage` down-weights rarely-routed experts, so the final scalar
   reflects the expert's **actual contribution across the calibration set**, not
   just its per-token effect.
4. **EMA accumulation.**
   `safe_add_with_ema(..., ema=0.9, key="expert_out_token_contrib")`
   (`utils.py:74-92`) maintains a running exponential moving average of this
   scalar across calibration batches.

## Summary formula

For expert `e` in a given batch:

```
ALA(e) = ( Σ_channels  mean_over_tokens[ clip( g_e ⊙ z_e ) ] ) × usage_e
```

- `z_e` = `down_proj` output of expert `e`  (its residual contribution)
- `g_e` = gradient of loss w.r.t. `z_e`
- `clip(·)` = per-channel top-1%-by-magnitude clipping
- `usage_e` = fraction of calibration tokens routed to expert `e`

accumulated across batches with EMA (`ema=0.9`).

## Consumers
