
# Channel-Level Router for Sparse MLP Activation: Experiment Plan

**Goal.** Learn a parameter-efficient router that maps a token's hidden state `h ∈ R^2048` directly to the set of **768 MLP intermediate channels** (out of 6144 activated channels = 8 experts × 768) that must be computed, such that the sparse forward pass matches the full model's output. The original FFN weights are **frozen and untouched** — we learn an *index*, not a compressed model.

**Positioning vs. DOT-MoE (arXiv:2606.01666).** DOT-MoE assigns neurons→experts (compute-structure partitioning) via balanced Sinkhorn OT + STE joint training of assignment and router. We push granularity to expert-size-1 (channel level): the partition becomes the router's *internal hierarchy* (offline tile construction), and the Sinkhorn+STE recipe transfers to the **token→channel** selection level (differentiable exact-budget top-k). Weights stay frozen.

**Key priors to exploit** (each must be *verified* in Phase 0 before the corresponding architecture component is enabled):

1. Input sparsity: a few outlier dims of `h` may dominate selection.
2. Static hot channels: some channels are near-always selected.
3. Co-activation block structure: selected channels cluster into groups.
4. Gate dominance: `silu(W_g h)` alone may determine the top-k set.
5. Low intrinsic complexity: the mask pattern may have far lower logistic rank than `W_g`'s spectral rank.

---

## 0. Setup, Data, and Global Standards

### 0.1 Notation

| Symbol    | Meaning                                                                        | Shape / value                    |
| --------- | ------------------------------------------------------------------------------ | -------------------------------- |
| `d`     | hidden dim                                                                     | 2048                             |
| `D`     | activated intermediate dim (8 experts × 768)                                  | 6144                             |
| `k`     | channel budget per token                                                       | 768 (12.5%)                      |
| `h`     | hidden state at MLP input (post-LN, the exact tensor fed to W_g/W_u)           | `[N, d]`                       |
| `a`     | oracle activation`silu(W_g h) ⊙ (W_u h)` per activated expert, concatenated | `[N, D]`                       |
| `imp_i` | channel importance = `                                                         | a_i                              |
| `M`     | oracle mask = top-k of`imp` per token                                        | `[N, D]` binary                |
| `N`     | number of tokens in the dataset                                                | ≥ 2M train, 200K val, 200K test |

**IMPORTANT — importance definition.** The oracle mask must be computed from `imp = |silu(gate) ⊙ up| · ||W_d column||`, NOT raw activation magnitude. Store per-expert gate values alongside (for MoE models, include the router gate weight in `imp` if experts are gated: `imp_i = |g_e · a_i| · ||W_d[:,i]||`).

### 0.2 Data collection spec (implement first)

```
collect_activations.py
  input : model (HF checkpoint), calibration corpus (e.g. 2M+ tokens of pretraining-like mix), layer indices L = {mid, deep} (e.g. layer 12 and 24 of a ~30-layer model)
  output: per layer, sharded .pt/.npy files:
    h        float16  [N, 2048]
    imp      float16  [N, 6144]        # or store topk indices+values to save space
    mask_idx int16    [N, 768]         # argsort-derived oracle indices
    meta.json: model name, layer, corpus, seq positions, dtype, W_d column norms, channel marginal freq
```

- Deduplicate near-identical `h` (e.g. BOS positions) or keep but flag; report both.
- Sanity check (mandatory, blocks everything downstream): run the model with oracle mask applied (zero out non-selected channels before down-proj) and confirm **ΔPPL vs full model < agreed tolerance** (target: < 1% relative PPL increase at k=768). If this fails, the premise is wrong — stop and re-derive k.

### 0.3 Global evaluation standards (fixed now, used everywhere)

Two metrics, always reported together:

1. **recall@k** = |predicted ∩ oracle| / k, averaged over tokens. Also report **importance-mass recall** = sum of `imp` covered / sum of oracle top-k `imp` (more forgiving, more meaningful).
2. **End metric**: KL(full ‖ masked) on next-token distribution, and ΔPPL on held-out text, with the *predicted* mask applied in the real forward pass.

**Calibration curve (do once, early):** degrade oracle masks synthetically (drop x% of oracle channels, replace with next-ranked) and plot ΔPPL vs mass-recall. This converts recall targets into PPL guarantees and prevents overfitting to the proxy metric. **Standard: all later methods are judged by predicted-mask ΔPPL; recall is diagnostic only.**

Report at budgets k ∈ {768, 0.9k, 1.15k, 1.25k} (slack sweep).

---

## Phase 0 — Preliminary Studies (no training of the router; ~1 week)

Each study has: Goal → Method → Implementation notes → Deliverable → Decision rule.
All run on the collected `(h, imp, mask)` dataset. All plots go into `results/phase0/`.

### P1. Intrinsic logistic rank of the mask matrix  ★ project go/no-go

- **Goal:** Measure the information-theoretic difficulty of the set-prediction problem *independently of W_g*. This is the feasibility certificate for learning free channel embeddings instead of approximating W_g.
- **Method:** Fit logistic matrix factorization `M ≈ σ(U V^T + b)` with `U ∈ R^{N×r}`, `V ∈ R^{D×r}`, bias `b ∈ R^D`; per-token top-k of the reconstructed scores → recall@k. Sweep `r ∈ {8, 16, 32, 64, 128, 256}`. On the same axes, plot recall@k of rank-r truncation of `W_g` under (a) plain SVD, (b) whitened SVD (`Σ^{1/2}-weighted`, Σ = empirical covariance of h).
- **Implementation:** Alternating minimization or joint Adam on GPU; subsample N to 500K for fitting, evaluate on held-out 100K (U for held-out tokens obtained by solving the per-token logistic regression given fixed V — this makes it a fair "best possible linear-in-embedding scorer" bound). Use the bias term to absorb marginal frequency.
- **Deliverable:** One figure: recall@768 vs rank, 3 curves (logistic-MF r*, plain SVD, whitened SVD). This is the paper's motivation figure.
- **Decision:** Let r* = smallest rank with mass-recall ≥ 95%. If r*(logistic-MF) ≤ 0.5 × r(whitened SVD @ same recall) → free-embedding router justified, set embedding dim from r*. If curves coincide → fall back to W_g-distillation-only router.

### P2. Gate sufficiency

- **Goal:** Decide the distillation target: is `silu(gate)` alone enough to determine top-k, or is the bilinear `gate⊙up` (and W_d norm) needed?
- **Method:** recall@k of top-k by `|silu(W_g h)|` (and by `|W_g h|`, `|W_u h|`) against oracle mask. All computable from stored tensors.
- **Decision:** gate-only mass-recall ≥ 97% → Stage-B target = gate scores (linear problem, whole lm_head machinery transfers). Else → keep full `imp` as target; consider TensorSketch-style bilinear features later.

### P3. Input screening (outlier dims / ANOVA sensitivity)

- **Goal:** Quantify the input-sparsity prior; choose router input form.
- **Method:** (a) Zero all but top-m dims of `h` by per-token magnitude, and separately by a *global* fixed dim set (rank dims by E[h_j^2] and by ANOVA main sensitivity below); recompute oracle-style selection through frozen W_g/W_u; measure recall vs m ∈ {8, 16, 32, 64, 128, 512, 2048}. (b) Anchored-ANOVA screening: anchor q = E[h]; for each dim j, main effect = Var over the empirical marginal of `score(q with dim j replaced by h_j)`; rank dims by main-sensitivity share (cheap: only requires forward scores through W_g, batched).
- **Decision:** If m ≤ 64 global dims reach ≥ 90% of full-input recall → enable the outlier-passthrough branch with that dim set; else input = low-rank projection only.

### P4. Static / dynamic decomposition

- **Goal:** Size the free "hot set" bypass and the true dynamic budget.
- **Method:** Channel marginal frequency f_i = P(i ∈ oracle mask). Coverage curve: for hot set H(q) = top-q channels by f_i, plot E[|M ∩ H|]/k vs q. Also compute the recall of the *pure static* predictor (always output top-768 by frequency) as the zero-parameter baseline.
- **Decision:** Choose |H| at the knee (e.g. where marginal coverage gain < 0.1 per added channel). Router then predicts only k − E[|M∩H|] dynamic slots. Report static-baseline recall — every learned method must beat it by a stated margin (≥ +10 mass-recall points) to justify its parameters.

### P5. Tile-ability via balanced Sinkhorn clustering  ★ connects to DOT-MoE

- **Goal:** Test whether channels can be re-grouped into balanced tiles such that each token's mask concentrates in few tiles → two-level router + coalesced memory access.
- **Method:** Build co-activation statistics C_ij = P(i,j both in mask) (estimate on 500K tokens; sparse — accumulate over per-token masks). Partition D=6144 channels into K tiles of exactly T (sweep (K,T) ∈ {(64,96), (128,48), (48,128)}) by **balanced OT / Sinkhorn k-means** on channel co-activation embeddings (spectral embedding of C, then Sinkhorn-constrained assignment — this is DOT-MoE's balanced-assignment machinery reused offline with a co-activation cost instead of weight similarity). Baselines for tile construction: (a) weight-similarity clustering of W_g rows (= MoEfication's heuristic), (b) random balanced split, (c) native expert boundaries (the existing 8×768 grouping).
- **Metrics:** per-token tile-count histogram of the oracle mask; recall@k when restricted to top-n tiles (oracle tile choice and frequency-based tile choice), n sweep.
- **Decision:** If top-10 tiles (of 64) cover ≥ 95% oracle mass → enable level-1 tile scorer; output space shrinks 96×. Also record: co-activation tiles vs weight-similarity tiles vs native expert boundaries — a standalone result regardless of the router.

### P6. Temporal coherence (optional, cheap)

- **Goal:** Decide if mask caching / delta prediction is worth it.
- **Method:** IoU of oracle masks between adjacent positions in the same sequence; IoU vs token distance.
- **Decision:** mean adjacent IoU ≥ 70% → add "reuse-previous-mask + top-up" variant to the ablation list; else drop.

### Phase-0 exit criteria

Produce `results/phase0/summary.md` containing: the P1 figure, the static baseline number (P4), the chosen (m, r, |H|, K, T), the calibration curve (§0.3), and an explicit go/no-go: **GO iff (i) oracle-mask ΔPPL < 1% and (ii) P1 shows r* ≤ 128 with mass-recall ≥ 95%.**

---

## Phase 1 — Core Router Design

### 1.1 Architecture (default; every component gated by Phase-0 results and individually ablatable)

```
Inputs (from Phase 0): S_out = outlier dim set (|S_out| = m, from P3)
                       P ∈ R^{d×r} = whitened low-rank projection (init: whitened SVD of W_g, r from P1)
                       tiles: balanced partition {T_1..T_K} (from P5)
                       H = static hot set (from P4)

phi(h) = concat[ h[S_out],  P^T Σ^{-1/2} h ]            # r' = m + r dims
score_i = c_i^T phi(h) + b_i                             # c ∈ R^{D×r'} free embeddings, b ∈ R^D static bias
tile_score_t = logsumexp_{i ∈ T_t}(score_i)  (or learned pooling)
Selection:
  1. keep all of H (free, no prediction)
  2. select top-n tiles by tile_score (n from P5; adaptive-n optional)
  3. within selected tiles, take top-(k − |H∩selected|) channels by score_i
  4. output mask with budget slack s (default 1.15×; conformal-calibrated in §2.4)
```

### 1.2 Parameter and FLOP budget (standard to hold)

With r=32, m=16, r'=48: params = d·r + D·r' + D + K·(pool params) ≈ 65K + 295K + 6K ≈ **0.37M ≈ 1.0% of one FFN layer** (3·2048·6144 ≈ 37.7M). **Hard standard: router params ≤ 2% of FFN params; router online FLOPs ≤ 3% of the FLOPs it saves** (saved ≈ 7/8 of FFN matmuls). Report both numbers in every results table.

### 1.3 Mandatory baselines

| Baseline                                        | Tests the claim                                | Params        |
| ----------------------------------------------- | ---------------------------------------------- | ------------- |
| Static top-768 by frequency (P4)                | is learning needed at all                      | 0             |
| Plain SVD of W_g, rank r                        | is whitening needed                            | 0 (derived)   |
| Whitened SVD of W_g, rank r (no training)       | is*training* needed                          | 0 (derived)   |
| Deja-Vu-style 2-layer MLP (d→1024→D)          | is our structure better than generic predictor | ~8.4M         |
| Product-key scorer (2×√D sub-keys)            | structural-trick alternative                   | ~0.4M         |
| LSH (SimHash, R bits, bucket→channel table)    | training-free retrieval alternative            | ~0            |
| VQ: k-means centroids + per-centroid mask table | lookup alternative (P5-adjacent)               | ~centroids·d |
| Random Gaussian projection rank r + top-k       | sketching lower bound / JL baseline            | 0             |

All evaluated under the §0.3 protocol at identical budgets. A results table row = (method, params, recall@k, mass-recall, KL, ΔPPL, router overhead).

---

## Phase 2 — Training Algorithm

### Stage A — Structural initialization (offline, no gradients)

- `P` ← top-r right-singular vectors of `Σ^{1/2} W_g^T` (whitened SVD; reuse lm_head pipeline code).
- `c_i` ← projection of whitened gate row: `P^T Σ^{-1/2} g_i` (padded with zeros on the m outlier coords, or small random). **Guarantee:** initial scores ≈ rank-r whitened gate scores, so pre-training recall equals the whitened-SVD baseline. Every training gain is then attributable.
- `b_i` ← λ · logit(f_i) with λ fit by 1-D line search on val recall.
- Tiles ← P5 Sinkhorn balanced clustering (frozen for Stage B).
- **Checkpoint standard:** log recall@k at init; must match whitened-SVD baseline within 0.5 pt.

### Stage B — Supervised set distillation (fast; main workhorse)

- Data: `(phi(h), mask_idx)` pairs.
- **Loss:** asymmetric boundary-focused margin ranking. For each token, form pairs (i ∈ oracle top-k, j ∉) restricted to the boundary window (ranks k−Δ..k+Δ, Δ≈256 by oracle imp); loss = w_fn·hinge(margin − (s_i − s_j)) with w_fn > 1 for pushing true positives up (false negatives cost more). Optionally listwise (LambdaRank-style) — ablate.
- Alternative target if P2 passed: regress/rank against gate scores (smoother signal) — ablate vs mask target.
- Optimizer: AdamW, lr 1e-3 embeddings / 3e-4 projection, cosine, ~3 epochs over 2M tokens. Cheap: whole stage is small-matrix; runs on 1 GPU in hours.
- **Stopping standard:** train until val mass-recall plateaus (< 0.1 pt / epoch). Do NOT chase recall beyond plateau — oracle mask is sufficient-not-necessary; residual errors may be harmless. Record plateau recall.

### Stage C — End-to-end task distillation (DOT-MoE recipe transferred)

- Training graph: `score(h)` → **Sinkhorn soft top-k** (entropic OT from D scores to the two-point marginal {selected: k, dropped: D−k}; exact budget by construction; ε annealed 1.0→0.03) → soft mask multiplies activations inside the *frozen* FFN → **KL(full model logits ‖ masked model logits)**.
- Forward uses hard top-k via STE; backward uses Sinkhorn soft gradients (train/inference consistency — same STE pattern DOT-MoE uses for its discrete assignments, applied at token→channel level).
- Trainable: `c, b, P` (low lr), optionally per-tile temperature. FFN and all model weights frozen.
- Optional joint refinement (**DOT-MoE's "joint > two-stage" hypothesis, our version**): allow tile assignment to update every 500 steps via Sinkhorn re-balancing on current co-activation of *predicted* masks. Ablate frozen-tiles vs co-adapted-tiles.
- Scope control: train on single layers first (mid + deep); full-model = per-layer routers trained in parallel with layerwise KL, then joint finetune only if layerwise composition degrades PPL by > 0.5%.
- **Standard:** Stage C must improve predicted-mask ΔPPL over Stage B at equal budget; if it doesn't within 5K steps, report and keep Stage B (negative result is still a finding: mask distillation suffices).

### 2.4 Conformal budget calibration (final, cheap)

On a held-out calibration set, compute per-token conformity = (mass captured at budget k·s); choose the smallest global slack s (or a per-token rule on score-distribution entropy) such that P(mass-recall ≥ 1−α) ≥ 90%, α = 0.02. Report the guarantee alongside average budget actually used. This replaces hand-tuned slack.

---

## Phase 3 — Evaluation, Ablations, Timeline

### 3.1 Final evaluation protocol

1. Per-layer: recall/mass-recall/KL vs budget curves for all methods (table + one figure).
2. Full model: ΔPPL (WikiText-2 + a held-out pretraining-mix shard) and 3–5 downstream tasks (lm-eval-harness) with routers on all FFN layers.
3. Efficiency: measured wall-clock decode tok/s with a gather-based sparse FFN kernel (or index_select fallback), router overhead isolated; report FLOPs and bytes-moved alongside. **No FLOP-only claims.**
4. Robustness: recall on out-of-calibration-domain text (e.g. code if calibrated on web text).

### 3.2 Ablation matrix (one row each, vs full default)

free `c` vs frozen-at-init | whitened vs plain init | outlier passthrough on/off | static bias `b` on/off | hot-set bypass on/off | tile level on/off | tile construction: co-activation vs weight-sim vs native-expert vs random | Stage C on/off | co-adapted vs frozen tiles | ranking loss vs BCE | boundary window vs all-pairs | budget slack sweep | conformal vs fixed slack.

### 3.3 Timeline & gates

| Week | Work                                                    | Gate                                                                         |
| ---- | ------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 1–2 | §0.2 data + Phase 0 (P1–P5; P6 if time)               | GO/NO-GO: oracle ΔPPL < 1% AND P1 r* ≤ 128                                 |
| 3–4 | Stage A+B on 2 layers + all baselines                   | Beat static baseline by ≥ 10 mass-recall pts; beat whitened-SVD by ≥ 3 pts |
| 5–6 | Stage C, tile ablations, all-layer deployment           | Stage C ΔPPL ≤ Stage B ΔPPL; full-model ΔPPL < 2% at k=768               |
| 7    | Kernels + wall-clock, conformal calibration, robustness | End-to-end speedup > 1 (router overhead included)                            |

### 3.4 Suggested repo layout

```
channel-router/
  data/collect_activations.py          # §0.2, includes oracle-mask sanity check
  phase0/p1_logistic_rank.py           # GPU alternating-min logistic MF + SVD curves
  phase0/p2_gate_sufficiency.py
  phase0/p3_input_screening.py         # magnitude + anchored-ANOVA sensitivity
  phase0/p4_static_dynamic.py
  phase0/p5_tiles_sinkhorn.py          # co-activation stats + balanced OT clustering
  phase0/p6_temporal.py
  phase0/calibration_curve.py          # recall→ΔPPL conversion (§0.3)
  router/model.py                      # architecture §1.1 (all components flag-gated)
  router/init_stage_a.py
  router/train_stage_b.py              # ranking losses
  router/train_stage_c.py             # sinkhorn soft-topk + STE + KL distillation
  router/sinkhorn_topk.py              # standalone differentiable top-k module + tests
  baselines/{static,svd,dejavu,product_key,lsh,vq,random_proj}.py
  eval/protocol.py                     # §0.3 metrics; §3.1 full protocol
  eval/sparse_ffn.py                   # masked forward w/ frozen weights; gather kernel
  results/phase0/summary.md            # exit criteria doc
```

Implementation conventions for Claude Code: every script takes `--layer`, `--data_dir`, `--out_dir`; every experiment writes a JSON of its metrics + a PNG; seeds fixed (=0) and logged; fp32 for covariance/whitening (Σ estimation and Σ^{-1/2} via eigh with 1e-5 jitter), fp16/bf16 elsewhere; unit tests for sinkhorn_topk (budget exactness, gradient check) and for the oracle-mask forward (matches recomputed activations).
