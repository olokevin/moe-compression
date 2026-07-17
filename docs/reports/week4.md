# Week 4 — MoE Compression Progress

Model throughout: **Qwen3-30B-A3B** (hidden `d=2048`, MoE intermediate `p=768`, 128
experts/layer, top-8, 48 layers). All numbers below are **one-shot** (compress → eval,
**no recovery fine-tuning**) unless stated. HellaSwag = 0-shot acc_norm, MMLU = 5-shot acc.

---

## 1. Heterogeneous budget allocation

We prune the expert-FFN intermediate dimension and let two orthogonal planners decide *where*
the budget goes: **inter-layer** (which layers absorb the cut, via `loss_coverage`) and
**intra-layer** (which experts keep channels, via coverage-weighted `attr_coverage`). Channels
are ranked by ridge leverage and `down_proj` is reconstructed with a closed-form Nyström solve.

### Current results (one-shot, no fine-tuning)

| Method @ 33% expert-FFN                                | HellaSwag                | MMLU             | MoE param ↓    | Active MoE param ↓ |
| ------------------------------------------------------ | ------------------------ | ---------------- | --------------- | ------------------- |
| Original (unpruned)                                    | 78.56                    | —               | —              | —                  |
| **Attribution-guided** (heterogeneous)           | **78.40** (−0.16) | **73.00**  | **−33%** | **−3%**      |
| Uniform (every layer/expert same fraction) — ablation | 65.10 (−13.3)           | 27.40 (≈chance) | −33%           | −33%               |

<sub>Reductions are on MoE (expert-FFN) params.  is storage;  ismeasured on real C4 top-8 routing (§ "Conclusion"). Uniform prunes every expert equally, so itsactive cut tracks the storage cut (−33%); heterogeneous pruning strips the least-routed experts,so its active cut is only −3%.</sub>

At 25% the attribution-guided model is essentially lossless (HellaSwag 78.45 vs 78.56, MMLU
76.04) and beats the activation-magnitude + plain-slicing baseline (78.23).

The per-expert statistics explain *why* heterogeneity is necessary: **Layer 0/1 are uniquely
compressible** (peaked leverage, truncatable spectra, low effective rank ≈595), the **middle
stack L5–L40 is uniformly high-rank** (effective rank ≈680, near-zero per-expert contribution),
and the **top layers L44–L47 re-concentrate importance into a few high-contribution experts**
(L47 is the single most important layer). Spending cuts where rank/leverage is low and
protecting the sparse high-contribution experts is exactly what the two-axis planner does.

### Conclusion

**Heterogeneous budget allocation is what makes aggressive one-shot pruning viable** — it holds
HellaSwag flat to 33% while the uniform ablation collapses by 13 points (and MMLU to chance) at
the identical storage cut. **However, it does not translate into an active-parameter reduction.**
Because the method strips the *least-contributing* channels — which live in the *least-routed*
experts — the removed capacity rarely enters a token's active top-8. Measured on real C4 routing,
a **25% storage cut yields only ~1.4% average active-compute reduction** (full-model active ratio
0.986; expert-FFN active ratio 0.974). The checkpoint delivers genuine memory savings but **not a
proportional FLOPs/latency speedup** — which motivates §3.

---

## 2. MoBE and progressive-heal Nyström

### MoBE (Mixture-of-Basis-Experts) — brief summary

MoBE is a **factorization** (not pruning) of the expert FFN:

- **Initialization** — group several experts (e.g. `k=8`) and SVD them **together** to seed a
  shared basisd.
- **Structure** — every expert's `gate_proj`/`up_proj` is expressed as a **mix of a shared
  per-layer basis** (`m=32` bases) plus a **per-expert transform**, adding an activation between
  them to improve expressiveness. `down_proj` is left dense.
- **Fit** — a **layer-wise local fit** (Adam, std-only norm, mean-MSE) that directly minimizes
  the reconstruction/truncation error of that layer's experts.

*(figure placeholder — MoBE architecture diagram, `!image.png`)*

### MoBE results (one-shot, −25% MoE-layer params)

| Model                                | HellaSwag               | MMLU            | wiki2 PPL | c4 PPL | MoE param ↓    | Active MoE param ↓ |
| ------------------------------------ | ----------------------- | --------------- | --------- | ------ | --------------- | ------------------- |
| Baseline (Qwen3-30B-A3B)             | 77.68                   | 82.0†          | 8.70      | 14.05  | —              | —                  |
| **MoBE** (`m=32`, `r=768`) | **73.67** (−4.0) | **77.23** | 9.59      | 15.98  | **−25%** | **−25%**     |
| RFID-MoE (`m=32`, `ξ=0.8`)      | 66.80 (−10.9)          | 71.32           | 12.68     | 21.49  | −28.4%         | −28.4%             |

<sub>Reductions are on MoE params (up+gate factorized to γ=0.625,  left dense). BecauseMoBE/RFID factorize  expert uniformly, the active cut equals the storage cut. † MMLUbaseline not re-run on this checkpoint (cited 82.0), so MMLU deltas are indicative. MoBE =reference-matched fitter, 2000 fit steps/(layer,type), all 48 layers. RFID row predates the fitterrewrite and omits the residual-reconstruction module, so the two are not yet apples-to-apples.</sub>

MoBE is our **best factorization result** — a clean ~4-pt HellaSwag drop with zero fine-tuning,
well ahead of RFID. It lands close to the same-budget attribution-guided *pruning* result on MMLU
(77.23 vs 76.04) but a few points behind on HellaSwag (73.67 vs 78.45).

### Borrowing MoBE's idea: layer-wise healing for the Nyström-compressed model

**Rationale.** The current Nyström derivation only does **column selection on up/gate** and a
closed-form `down_proj` reconstruction — which is *optimal for `down_proj` given the fixed
selection*, but it leaves the up/gate matrices untouched. We can go further and **update all
three matrices** jointly, exactly as MoBE heals its factorization with a local fit.

**Current implementation (`nystrom_moe`).** Sequential, one layer at a time (each layer sees the
already-compressed prefix, so its expert inputs match inference):

1. **Nyström channel select** — leverage-rank each expert's channels, keep a uniform top-`k`
   (→ a standard, smaller Qwen3-MoE; `p=768 → k=512` for 33%).
2. **Closed-form init** — Nyström `down_proj` reconstruction on the kept subset; row-slice
   `gate_proj`/`up_proj`.
3. **Activation-aware local heal** — for each expert, Adam-refine `{gate_k, up_k, down_k}`
   **together** to minimize `‖expert_k(X) − Y_ref‖²` on captured routed inputs, with `Y_ref`
   the original expert's output (MoBE-aligned fitter: lr≈0.07, fp32, best-state snapshot).

Unlike MoBE, experts stay plain dense MLPs (just narrower), so the output is a standard HF
checkpoint — no custom module or factor format. Implementation and configs are landed
(`src/compress/moe_basis/nystrom_moe.py`, `configs/compress_then_train/qwen3_30b_a3b_nystrom_moe*.yaml`);
the lr/iters sweep and full 33% A100 run are the remaining step to fill in the results table.

---

## 3. Reducing active parameters

Since §1 shows heterogeneous pruning barely cuts *active* compute, we prototyped a scheme that
targets active parameters directly: **dynamic, per-token, per-expert active-parameter allocation**.

**Idea.** Keep a fixed *per-token* channel budget (`B = 0.67·K·I`, a 33% active cut), but
distribute it **unevenly across each token's top-K experts** — more channels to the experts that
matter more for *that token*, less to the rest, conserving the budget exactly (largest-remainder
water-filling with a per-expert floor). Two orthogonal knobs:

- **criterion** (how much budget each expert gets): `router_prob` (per-token softmax weight),
  `contribution` (per-expert attribution scalar), or `uniform` (even-split baseline).
- **channel_metric** (which channels each expert keeps): `activation` (repo default) or
  `leverage` (Nyström ridge-leverage score, score-only, no `down_proj` reconstruction).

Realized as **masking simulation** (zero channels beyond a token's budget), so it measures exact
accuracy at the target *active* budget without variable-width matmuls, and reuses ranking
statistics already saved by the scoring stage — nothing new to collect. Implemented as a
self-contained, unit-tested package (`src/dynamic_active_param/`) wired into eval; five configs
cover the criterion × channel_metric grid at 33% on HellaSwag.

### Dynamic active-param results (HellaSwag 0-shot, 33% active cut, no fine-tuning)

| Config                                       | criterion     | channel_metric | acc            | acc_norm       | MoE param ↓ | Active MoE param ↓ |
| -------------------------------------------- | ------------- | -------------- | -------------- | -------------- | ------------ | ------------------- |
| Dense baseline (unpruned)                    | —            | —             | —             | 78.56          | —           | —                  |
| Static one-shot 33%                          | attr_coverage | activation     | —             | 78.23          | −33%        | −3%                |
| Static Nyström 33%                          | attr_coverage | leverage       | 59.18          | 78.40          | −33%        | −3%                |
| Dynamic prob × activation                   | router_prob   | activation     | 57.31          | 75.96          | 0%\*         | **−33%**     |
| Dynamic prob × leverage                     | router_prob   | leverage       | 57.65          | 76.13          | 0%\*         | **−33%**     |
| Dynamic contrib × activation                | contribution  | activation     | *re-running* | *re-running* | 0%\*         | −33%               |
| Dynamic contrib × leverage                  | contribution  | leverage       | *re-running* | *re-running* | 0%\*         | −33%               |
| Dynamic uniform × activation (dyn baseline) | uniform       | activation     | 49.55          | 66.29          | 0%\*         | −33%               |

<sub>Reductions are on MoE (expert-FFN) params. The  rows physically slim the model (bigstorage cut, but active cut only −3% because heterogeneous pruning removes the least-routedexperts). The  rows are  (), so storage isunchanged (0%), but each token keeps only 67% of its top-8 expert-FFN channels — a true −33%active cut.  A real (non-simulated) implementation would need variable-width matmuls to realizethe storage/latency saving.</sub>

**Reads.**

- **Per-token heterogeneity is decisive, again.** The dynamic-path uniform baseline (even split
  across the top-K) collapses to **66.29**, while routing budget by `router_prob` recovers to
  **75.96–76.13** — the same "*where* the budget goes matters" story as the §1 storage ablation,
  now on the *active*-param axis.
- **`leverage` edges `activation`** as the channel metric (76.13 vs 75.96) even without the
  Nyström `down_proj` correction.
- **The static-pruning ceiling (78.40) is not yet matched.** Distributing a *per-token* active
  budget costs ~2–2.5 pts vs. the static 33% storage prune — but that static cut barely reduces
  active compute (§1), whereas these dynamic configs deliver a genuine ~33% active-param cut.
  The `contribution` arms (re-running) will complete the grid.
