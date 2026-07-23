# Level 1 implementation plan — Global g²-weighted nested channel selection

Realizes `plan_level1.md` as a new **`pivchol_global`** criterion in the existing
`src/dynamic_active_param/` package (masking simulation, no fine-tuning,
`real_slim: false`), reusing the block/install/keep-mask machinery.

## Key reuse insight

The existing block keeps channels via `keep = _dyn_ranks[e] < k_col`. If we set
`_dyn_ranks = pivrank` (position of each physical channel in the expert's pivot
order π) and `k_col = t_k` (per-token prefix length), then `keep` selects exactly
**the first `t_k` pivoted-Cholesky channels** — no block changes beyond the
allocation call. The only new online piece is computing `t_k` from a single
global g²·σ threshold instead of a per-expert quota.

## Offline artifact (Phase B; Phase A = cached `expert_covariances.pth`)

`G_k = expert_covariances.pth[layer][eid]` is already the uncentered activation
Gram `E[φ_k φ_kᵀ]` (m×m). New builder `scripts/warm_pivchol_cache.py` +
`src/dynamic_active_param/pivchol.py`:

Per layer (batched over the E=128 experts on GPU):
1. `Wd (E,d,m)` = stacked `expert.down_proj.weight` (d=2048, m=768).
2. `B = bmm(Wd.transpose(1,2), Wd)` → `(E,m,m)` weight Gram (H = I).
3. `G (E,m,m)` = stacked covariances (identity fallback if an expert is missing).
4. `Θ = G ⊙ B` (Hadamard; PSD by Schur product theorem).
5. **Batched ridge-pivoted Cholesky** (`λ_r` added to diagonal, shared across all
   experts/layers): 768 batched steps → `perm (E,m)` pivot order, `gains (E,m)`
   marginal gains (residual diagonal at each pivot; monotone non-increasing).
6. `pivrank = inverse-permutation of perm` (E,m). Store `pivrank (L,E,m) int` and
   `gains_pos (L,E,m) float` (gains in pivot-position order).

Cache → `scores_dir/pivchol_artifact.pth` (tens of MB). Cost ≈ minutes
(~1.4e12 flops total) + one model load.

## Code changes

- **`pivchol.py`** (new): `pivoted_cholesky_batched(theta, lambda_r)` →
  `(perm, gains)`; `build_pivchol_artifact(model, scores_dir, lambda_r, device,
  save)` → `AllocArtifact` with `channel_rank=pivrank`, `gains=gains_pos`.
- **`precompute.py`**: add optional `gains: torch.Tensor = None` field to
  `AllocArtifact`.
- **`allocate.py`**: add `"pivchol_global"` to `_VALID_CRITERIA`; new
  `_pivchol_allocate(routing_weights, selected_experts, gains, B)` — per token
  (chunked): `score = (g²)[:,:,None] * gains[sel]` `(t,K,m)`, flatten to
  `(t, K·m)`, `topk(B)`, count selected per expert block → `t_k (t,K)`. Conserves
  `Σ t_k = B` exactly; **no k_min floor** (a losing expert gets `t_k=0`, the
  intended expert-dropping-emerges behavior). Branch runs before the
  `K*k_min ≤ B` feasibility check.
- **`block.py`**: pass `gains=getattr(self, "_dyn_gains", None)` into
  `allocate_budgets`. Keep-mask + linear-g output scaling unchanged (g² is used
  only for selection, output still weighted by linear g).
- **`install.py`**: for `criterion == "pivchol_global"`, attach
  `_dyn_gains = artifact.gains[mask_idx]`; `B = round((1-prune_ratio)·K·I)`.
- **`merge_slim_eval.py`**: when `criterion == "pivchol_global"`, build/load the
  pivchol artifact (build needs the loaded model + covariances; skip if
  `pivchol_artifact.pth` cache exists) and install.

## Config

`configs/eval/qwen3_30b_a3b_dynamic_pivchol_lev50_hellaswag.yaml`:
`prune_ratio: 0.50`, `dynamic_alloc.criterion: pivchol_global`,
`dynamic_alloc.lambda_r: 1.0`, `real_slim: false`. `channel_metric` unused (gains
come from the pivchol artifact, not leverage/activation).

## Tests (`src/dynamic_active_param/tests/`)

- `test_pivchol.py`: pivoted Cholesky reproduces `Θ+λI` (`L Lᵀ ≈ Θ+λI`); `perm`
  is a permutation; `gains` monotone non-increasing; batched == per-matrix loop;
  a near-duplicate channel pair gets a collapsed second gain (redundancy fix).
- `test_allocate.py`: `pivchol_global` conserves `Σt_k = B`; `0 ≤ t_k ≤ m`;
  higher-g expert gets ≥ budget of lower-g (via g² weighting); ρ=1 (B=K·m) keeps
  all; a dominated expert can receive `t_k=0`.

## Validation

Three-way at 50% active budget on HellaSwag 0-shot (Qwen3-30B-A3B-Thinking-2507):
expert-drop (75.96%, done) vs current coverage/router_prob (done) vs **Level 1
pivchol_global**. Expected: L1 ≥ current-method, L1 ≈ expert-drop. Update
`docs/results/dynamic_active_param/q3_30b_dynamic_active.md`.
