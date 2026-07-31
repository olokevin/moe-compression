# Level-2 implementation plan — Allocate cross-experts, score cross-experts

Turns `plan_level2.md` (M1 oracle ladder, M3 redundancy structure, M4 regime
diagnostic) into concrete code, configs, and runs, reusing the
`src/dynamic_active_param/` package and the Level-1 `pivchol_global` machinery.

## The tractability split (why the plan is shaped this way)

`plan_level2.md` writes its oracles as **reconstruction diagnostics on ~1–2k
tokens, ignoring efficiency**. Two of those objects are infeasible at full
benchmark scale:

- **Oracle-A** (exact per-token `h`, full off-diagonal `Θ(x)`) is a per-token
  greedy OMP over `K·I = 6144` channels — ~1e17 flops across full HellaSwag.
- **Oracle-B**'s literal ingredient, full pairwise cross-expert covariance, is
  `(K·I)² · d ≈ 19 GB/layer` dense — unstorable.

And a strong prior already exists: a full stacked cross-expert Nyström study
(memory `full-cross-expert-nystrom-covariance`, layer 46) found **70.6% of the
covariance energy is off-diagonal (cross-expert)** yet channel **selection barely
changes — 98.5% top-512 overlap, Spearman 0.995**. So Level-2's realistic upside
is small, and the job is to *measure* it decisively, not to over-build.

We therefore split Level-2 into:

- **(A) Runnable selectors** — produce the benchmark-accuracy tables (HellaSwag
  ×4 budgets + MMLU@75%). All stay in the masking-simulation block forward, touch
  no expert weights beyond cached column norms / a tiny public basis, and are
  prefix-contiguity-friendly.
- **(B) Subset diagnostics** — the full M1 oracle ladder (incl. the infeasible-at-
  scale Oracle-A OMP) is run as **reconstruction relative-error on ~2k C4 tokens**,
  plus M3 structure and M4 regime plots.

## (A) Runnable selectors (new criteria in `allocate.py` / `block.py`)

All three below select a per-token global top-`B` over the pooled `K·I` channels
of the token's active experts (same budget `B = round((1−ρ)·K·I)` as Level-1,
no `k_min` floor — a dominated expert may get 0).

### A1. `oracle_mag` — Oracle-A (magnitude), the exactly-runnable ceiling

Per token, score each channel by its **exact per-token output magnitude**

```
s_{e,j}(x) = g_e · |inter_{e,j}(x)| · ‖W_down[:, j]‖₂
```

where `inter = act_fn(gate)·up` is already computed by the block and the column
norms `‖W_down[:,j]‖` are a cached `(L,E,I)` constant. Keep the global top-`B`.

This is the block-diagonal (no off-diagonal) oracle that sees the **true
per-token activation** rather than only the router `g`. It upper-bounds every
router-only offline method (Level-1, `pubsub`): the gap `oracle_mag − Level-1` is
the value of per-token activation information; the gap `oracle_mag − pubsub`
tells us how much of that a cheap offline scheme recovers. Cheap: `inter` is
free, top-`B` over `(T,K,I)` is the only extra work.

### A2. `pubsub` — Level-2 method: shared-public-subspace redundancy penalty

The one runnable cross-expert method, realizing the "cover shared knowledge once,
spend the rest on unique channels" idea in the **prefix-contiguity-preserving**
form M3 licenses (low-rank public correction).

**Offline artifact `pubsub_artifact.pth`** (built from cached
`expert_covariances.pth` + `down_proj` weights, per layer):

1. Aggregate output-space second moment `M_ℓ = Σ_e W_down_e G_e W_down_eᵀ`
   (`d×d`), `G_e` = cached activation Gram. Top-`r` eigenvectors give the shared
   **public basis** `U_ℓ ∈ R^{d×r}` (directions many experts write into = public).
2. Per channel, split its activation-scaled output vector
   `v_{e,j} = √(G_e[j,j]) · W_down[:,j]` into public `‖Uᵀv‖²` and private
   `‖v‖² − ‖Uᵀv‖²`. Store: private pivoted-Cholesky gains `σ^priv` (run the
   Level-1 batched pivoted Cholesky on the **public-deflated** coupling
   `Θ^priv_e = G_e ⊙ (W̃_downᵀ W̃_down)`, `W̃_down = (I−U Uᵀ)W_down`), the
   per-direction public-carrier coefficients `c_{e,j,·} = Uᵀ v_{e,j}` (only the
   argmax carrier per `(e, dir)` is needed online), and `U_ℓ`. Sizes: `σ^priv`
   like Level-1 (~57 MB), `U` and carriers negligible (`r≤16`).

**Online (router-only, per token):** reserve `p = round(κ·B)` "public" slots —
for each of the top public directions, keep the single co-activated channel with
the largest `|c|` onto it (dedup: each public direction covered once, by whichever
active expert carries it best); fill the remaining `B−p` "private" slots by the
Level-1 rule on the residual, `g_e² · σ^priv_{e,r}` global top. `κ` and `r` are
config knobs (`r=8, κ` chosen so `p≈r`). Emerges per-expert prefix lengths, no
weight reads beyond the offline artifact + free `g`.

**Fallback contract:** if the prior holds and `pubsub ≈ Level-1`, that is the
decisive M1 result (small `Oracle-B − Level-1`) and terminates the engineering —
the oracle ladder still delivers the paper's conclusion. Code is structured so
the oracle ladder is independent of `pubsub` succeeding.

### Shared block change

Level-1's block loops experts independently; `oracle_mag`/`pubsub` need a
**global (cross-expert) top-B per token**, so add a generalized path in
`block.py`: compute `inter` for all K active experts of each token into a
`(T,K,I)` buffer (gather via the existing `expert_mask` loop, write into a padded
tensor), run the selector to get the `(T,K)` keep-counts / masks, then apply. The
router-only criteria (`router_prob`, `coverage_alloc`, `pivchol_global`) keep the
existing per-expert loop unchanged (they need no cross-expert activation).

## (B) Subset diagnostics (scripts, ~2k C4 tokens, reconstruction rel-error)

`scripts/level2_oracle_ladder.py` — capture one layer's per-token `inter` and
`g` on ~2k C4 tokens (reuse the hook pattern in `full_nystrom_cov_analysis.py`),
then for each budget `B` and each selector compute the **layer-output
reconstruction relative error** `‖ŷ − y‖ / ‖y‖` averaged over tokens:

| selector | online info | coupling | note |
| --- | --- | --- | --- |
| `level1` | router `g` | block-diag `Θ_e` | current |
| `pubsub` | router `g` | offline + public basis | **Oracle-B proxy** |
| `oracle_mag` | exact `h` | block-diag (magnitude) | runnable ceiling |
| `oracle_exact` | exact `h` | **full off-diag OMP** | Oracle-A, infeasible at scale, feasible here |

`oracle_exact` = per-token greedy OMP: repeatedly add the channel most reducing
the residual `‖y − Σ_{sel} g_e inter_{e,j} W_down[:,j]‖` until `B` chosen. Gives
the true A/B/Level-1 ladder in reconstruction space. **Decision:** negligible
`(oracle_exact − pubsub)` and `(pubsub − level1)` at all budgets ⇒ terminate.

`scripts/level2_m3_structure.py` — reuse `full_nystrom_cov_analysis.py`'s
push-through leverage tooling: (i) coherence `μ` bucketed by pivoted-Cholesky
rank (monotone-decay test → head-public/tail-private), (ii) principal angles
between leading `Θ_e` eigen-subspaces of frequent co-activated pairs.

**M4 regime diagnostic** — reuse the runnable path: a `beta` knob on
`pivchol_global` scoring `g^{2β}·σ` (β∈{1,1.5,2,3}); β=1 is Level-1, β→∞ →
reduce-top-k. Config-only sweep on HellaSwag. Plus emergent prefix-length `t_e`
histogram (logged by the ladder script) and per-token entropy bucketing.

## Code changes (summary)

- `allocate.py`: add `oracle_mag`, `pubsub` to `_VALID_CRITERIA`; the global
  top-B counting reuses `_pivchol_allocate`'s topk-and-scatter shape. Add optional
  `beta` arg threaded to `pivchol_global` scoring.
- `block.py`: generalized cross-expert forward for `oracle_mag`/`pubsub`
  (buffer `inter (T,K,I)`, global select, mask); existing loop for the rest.
- `precompute.py`: `AllocArtifact` gains `col_norm`, `pub_basis`, `pub_carrier`,
  `sigma_priv` optional fields.
- `pivchol.py` / new `pubsub.py`: `build_pubsub_artifact(model, scores_dir, r, λ)`
  → deflated `σ^priv` + `U` + carriers; `col_norm` cached from `down_proj`.
- `install.py`: attach the new per-layer tensors for the new criteria; `beta`.
- `merge_slim_eval.py`: build/load `pubsub_artifact.pth` when
  `criterion=pubsub`; `oracle_mag` needs only `col_norm` (build inline). β passthrough.
- `tests/`: `oracle_mag` conserves `Σk=B`, ties broken deterministically, ρ=1
  keeps all; `pubsub` conserves budget, public dedup keeps ≤1 carrier/direction,
  reduces to Level-1 when `r=0`; `build_pubsub_artifact` deflation is orthogonal
  (`U'ᵀ W̃ ≈ 0`); `beta` monotonic sharpening.

## Configs

Reuse the Level-1 YAML shape (`test_only`, `real_slim:false`, `sdpa`, 30B-Thinking,
`scores_dir` the cached c4 scores). Per budget ρ∈{0.50,0.625,0.75,0.875}:

- `qwen3_30b_a3b_dynamic_oracle_mag_{50,625,75,875}_hellaswag.yaml`
- `qwen3_30b_a3b_dynamic_pubsub_{50,625,75,875}_hellaswag.yaml`
- MMLU@75%: `qwen3_30b_a3b_dynamic_{oracle_mag,pubsub}_75_mmlu.yaml`
- M4: `qwen3_30b_a3b_dynamic_pivchol_beta{15,2,3}_50_hellaswag.yaml`

Level-1 and reduce-top-k numbers are already in the report — reused as baselines,
not re-run.

## Runs (A100, `launch-on-a100`)

Warm `pubsub_artifact.pth` once (CPU factorization, per the cublas-crash memory).
Then eval waves of 2 jobs × 4 GPUs (mirror `run_level1_sweep.sh`): 8 HellaSwag +
2 MMLU + 3 M4 = 13 evals. Subset diagnostics run on one 4-GPU box (~minutes).
Pull results, write the report section.

## Report

New "Level 2 — cross-expert" section in `q3_30b_dynamic_active.md`: the accuracy
table (Level-1 vs `oracle_mag` vs `pubsub` vs reduce-top-k across 4 budgets +
MMLU@75%), the M1 reconstruction ladder, and the M3/M4 takeaways — concise.
