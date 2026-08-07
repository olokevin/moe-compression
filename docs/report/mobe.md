use c

# Week 6 — MoBE (Mixture-of-Basis-Experts) Factorization

Model throughout: **Qwen3-30B-A3B** (hidden `d=2048`, MoE intermediate `p=768`, 128 experts/layer, top-8, 48 layers). All numbers below are **one-shot** (compress → eval, **no recovery fine-tuning**). HellaSwag = 0-shot acc_norm, MMLU = 5-shot acc.

---

## 1. MoBE formulation

MoBE is a **factorization** (not pruning) approach: instead of removing channels, it decomposes each expert's weight matrix into a **per-layer shared basis** plus a **per-expert transform**.

![MoBE mechanism](../exps/mobe/figures/mobe_mechanism.svg)

**MoBE factorization.** Each up/gate matrix `W^i ∈ ℝ^{p×d}` is first rank-decomposed, then the larger factor is re-parameterized as a mixture of shared basis matrices:

$$
W^i = A^i B^i,\qquad A^i \in \mathbb{R}^{p\times r},\; B^i \in \mathbb{R}^{r\times d},\; r \le \min\{p,d\}=p,
$$

$$
B^i = \sum_{j=1}^{m}\alpha_{i,j}\,B_j,\qquad \alpha_{i,j}\ge 0,\;\; \sum_{j=1}^{m}\alpha_{i,j}=1,\qquad \{B_j \in \mathbb{R}^{r\times d}\}_{j=1}^{m}\ \text{shared in one MoE layer}.
$$

Adding a weight-space nonlinearity `f` (SiLU) between the mixed basis and the transform gives the final reconstruction (Eq. 4 in the paper):

$$
\boxed{\;\hat W^i = A^i\, f\!\Big(\textstyle\sum_{j=1}^{m}\alpha_{i,j}B_j\Big)\;}
$$

The factors are learned by minimizing the per-layer reconstruction error over the `n` experts (Eq. 5):

$$
\min_{A^i,\,B_j,\,\alpha_{i,j}}\ \sum_{i=1}^{n}\big\lVert W^i - \hat W^i\big\rVert^2 = \sum_{i=1}^{n}\Big\lVert W^i - A^i f\big(\textstyle\sum_{j=1}^{m}\alpha_{i,j}B_j\big)\Big\rVert^2 .
$$

- `A^i ∈ ℝ^{p×r}` — **per-expert transform** (unique per expert, encodes specialized information).
- `α_i ∈ ℝ^m` — **per-expert mixing coefficients** (simplex-constrained, selects which bases matter).
- `f(·)` — **weight-space SiLU** (bipolar activation; ReLU is suboptimal as it over-sparsifies `B^i`).

**Shape / parameter analysis.** The saving comes entirely from *sharing* the basis. Consider one compressed matrix type (e.g. gate) in a layer with `n=128` experts:

- **Each basis matrix** `B_j ∈ ℝ^{r×d}` holds `r·d` parameters.
- **The combined shared basis** `{B_j}_{j=1}^{m}` holds `m·r·d` parameters — stored **once** for the whole layer, not per expert.
- **Per-expert transforms** `{A^i}` hold `n·p·r` parameters; the `α` coefficients (`n·m`) are negligible.

**Fit procedure:** Grouped-SVD initialization, then per-(layer, matrix-type) Adam optimization with std-only normalization and mean-MSE loss, `lr=0.07`, 2000 fixed steps. Data-free (no calibration data needed beyond weight initialization). Following the paper, classic MoBE leaves `down_proj` dense (it stores critical knowledge and is less compressible); our even-split variant factorizes all three via an output-side basis (§3).

---

## 2. Results

Per-type γ = kept fraction of each matrix (`1.0` = dense/untouched). Whole-MoE ↓ is the reduction over all three expert matrices combined.

| Method                                  | `m` | γ (gate / up / down) | Whole-MoE ↓ | HellaSwag       | MMLU            | PPL wiki2 / c4          |
| --------------------------------------- | ----- | --------------------- | ------------ | --------------- | --------------- | ----------------------- |
| Baselines -33%                          |       |                       |              |                 |                 |                         |
| **Uncompressed** (Qwen3-30B-A3B)  | —    | 1.0 / 1.0 / 1.0       | —           | 77.68           | 82.0†          | 8.70 / 14.05            |
| Nyström uniform (prune)                | —    | 0.67 / 0.67 / 0.67    | −33%        | 65.10           | 70.35           | —                      |
| Nyström uniform (prune+heal)           |       | 0.67 / 0.67 / 0.67    | −33%        | 77.3            | 75.3            |                         |
| MoBE -33%                               |       |                       |              |                 |                 |                         |
| Classic MoBE, gate/up only              | 16    | 0.500 / 0.500 / 1.0   | −33%        | 69.64           | 74.05           | 11.75 / 20.32           |
| **Even-split MoBE, gate/up/down** | 38    | 0.672 / 0.672 / 0.672 | −33%        | **73.13** | **76.83** | **10.10 / 16.57** |
|                                         |       |                       |              |                 |                 |                         |
| Classic MoBE, gate/up only              | 32    | 0.625 / 0.625 / 1.0   | −25.0%      | 73.67           | 77.23           | 9.59 / 15.98            |
| Down-only MoBE (output-side basis)      | 32    | 1.0 / 1.0 / 0.625     | ~12.5%       | 76.27           | 78.33           | 9.42 / 14.99            |

† MMLU baseline not re-run on this checkpoint (cited 82.0); MoBE MMLU deltas are indicative.

---

## 3. Key findings

1. **Even-split is the best factorization result.** Spreading the 33% reduction across gate/up/down (each at γ≈0.67) beats concentrating it on gate/up alone (each at γ=0.5): **+3.5 pt HellaSwag, +2.8 pt MMLU** at the same total compression.
2. **Output-side basis wins for down_proj.** When factorizing `down_proj`, an ablation showed placing the shared basis on the output (hidden `d=2048`) side beats the input (intermediate `p=768`) side by +2.2 pt HellaSwag (76.27 vs 74.09) — the larger axis provides a richer shared subspace. The even-split run uses this output-side choice for `down_proj`.
3. **MoBE vs pruning.** Even-split MoBE (73.13 HS / 76.83 MMLU) still trails attribution-guided pruning (78.40 HS / 73.00 MMLU) on HellaSwag by ~5 pts, but edges it on MMLU by ~4 pts. The comparison is confounded by different base checkpoints (A3B vs Thinking-2507).
4. **MoBE delivers proportional active-param reduction.** Unlike heterogeneous pruning (which strips least-routed experts and barely cuts active compute), MoBE factorizes every expert uniformly — so the storage cut equals the active-parameter cut.

## Next steps

* Combine mobe and nystrom: orthogonal compression axi
