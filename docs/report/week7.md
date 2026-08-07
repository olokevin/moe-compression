# Channel Experts — Experts at Finest Granularity

**TL;DR.** An MoE expert's intermediate channels are themselves micro-experts: each `(gate_row, up_row, down_column)` triplet is a self-contained unit that either fires or stays silent for a given token. Treating channels as the atomic expert unit reveals that (1) an offline router cannot determine which channels a token needs, (2) the `up_proj` output already acts as a per-channel router, and (3) this perspective reduces the number of unique routing decisions and eliminates additional parameters.

## Formulation

A standard MoE FFN expert computes

$$
y_e(x) \;=\; W_{\text{down}}^{(e)}\;\bigl[\,\text{SiLU}(W_{\text{gate}}^{(e)} x)\;\odot\;(W_{\text{up}}^{(e)} x)\,\bigr]
$$

with intermediate dimension $I$. The $j$-th **channel expert** of expert $e$ is the rank-1 computation path

$$
y_{e,j}(x) \;=\; \bigl[\text{SiLU}(w_{\text{gate},j}^{(e)\top} x)\cdot(w_{\text{up},j}^{(e)\top} x)\bigr]\;\cdot\;W_{\text{down}}^{(e)}[:,j]
$$

where $w_{\text{gate},j}^{(e)}$ is row $j$ of `gate_proj`, $w_{\text{up},j}^{(e)}$ is row $j$ of `up_proj`, and $W_{\text{down}}^{(e)}[:,j]$ is column $j$ of `down_proj`. The block output is $\sum_{e \in \text{top-}K} g_e \sum_{j=1}^{I} y_{e,j}(x)$ — a sum over $K \cdot I$ channel experts.

The gating value $\text{SiLU}(w_{\text{gate},j}^{(e)\top} x)$ acts as a **soft on/off switch** for channel expert $j$: when it is near zero, the entire path contributes nothing regardless of `up_proj` or `down_proj`.

## Why experts at the finest granularity

Traditional MoE routing selects $K$ whole experts per token. But:

1. **Experts contain overlapping features.** The intermediate subspaces of co-activated experts share significant structure — cross-expert covariance holds ~70% off-diagonal energy. When two experts both carry the same feature, routing at expert granularity forces redundant computation.
2. **Partial experts suffice.** Masking experiments show that keeping only 12.5% of a token's $K \cdot I$ channels (selected per-token by activation magnitude) loses < 2 pts vs. dense — 7/8 of the channel experts are dispensable for any given token, but *which* 7/8 changes token-to-token.
3. **No universal sparse set.** Per-channel activation-frequency analysis over 70k tokens shows 0.3% of channels are "always on" and 0.4% "always off" — the remaining 99.3% have genuinely token-dependent utility, precluding a static keep-set.

## Determining the channel experts for each token

### Offline calibration — and its fundamental limitation

The MoE router produces $g_e(x) \in [0,1]$ per expert but has **no information about expert parameters or channel-level capabilities** — it cannot tell *which* channels within an expert the token needs. Offline calibration scores channels from corpus statistics and the online decision reduces to $\text{score}_{e,j}(x) = r(g_e) \cdot s_{e,j}$ (router reweight $\times$ static channel importance).

The key limitation: the channel ranking is fixed (cannot bear a look-up table for each token), so the best offline methods still **degrade sharply under high reduction ratios.**


**Budget sweep — Level 1 vs the winning 33% baseline (`router_prob × act`), HellaSwag acc_norm:**

| Reduction | Reduce k | MoSE  | **Ours (pivchol)** |
| --------- | -------- | ----- | ------------------------ |
| 50%       |          | 71.46 | **74.26**          |
| 62.5%     |          | 61.00 | **70.54**          |
| 75%       |          | 43.66 | **63.60**          |
| 87.5%     |          | 30.32 | **44.15**          |

### Online — token-specific channel selection via `up_proj`

To pick the best channel experts for each token online (rather than relying on a fixed offline ranking), the naive approach is an additional learned router of size $d \times (N \cdot I)$ that directly predicts which channels to activate — but this is as large as the MoE parameters themselves and requires full re-training.

Instead, we observe that `up_proj` already *is* a per-channel router: its output $u_{e,j}(x) = w_{\text{up},j}^{(e)\top} x$ determines which channel experts will fire for each token. We use its magnitude to select the active channels, then retrieve only the corresponding rows of `gate_proj` and columns of `down_proj`.

Scoring channels by $g_e \cdot |u_{e,j}(x)| \cdot \|W_{\text{down}}[:,j]\|_2$ and keeping the global top-B across the $K$ active experts:

Dense baseline: HellaSwag 78.56 acc_norm, MMLU 5-shot $\approx$ 79.5.

| up_proj | gate_proj | down_proj | Whole-FFN active cut | HellaSwag | MMLU |
| :-----: | :-------: | :-------: | :------------------: | :-------: | :---: |
|  full  |   full   |   -75%   |       −25.0%       |   78.28   | 80.53 |
|  full  |   full   |  -87.5%  |       −29.2%       |   76.84   | 79.48 |
|  full  |   -75%   |   -75%   |       −50.0%       |   75.31   | 79.47 |
|  full  |  -87.5%  |  -87.5%  |       −58.3%       |   71.30   | 76.43 |

The `up_proj` proxy moves the selection decision **before** `gate_proj`, so both `gate_proj` and `down_proj` run at reduced width — achieving **2$\times$ the actual active-param reduction** at a cost of 3–5.5 pts vs. the full-intermediate oracle. This is still far above any offline method at comparable real reduction. Further ablation shows the weight-norm factor $\|W_{\text{down}}[:,j]\|$ is negligible ($<$ 0.3 pt contribution); the activation magnitude alone carries the signal.

### Limitation and next steps

**Limitation — sequential dependency.** Using `up_proj` as the channel router introduces a strict sequential dependency in the per-layer critical path:

```
route(x) → pick experts → compute u = x·W_up → threshold → know M → fetch W_gate[:,M], W_down[M,:] → compute
```

Nothing can be overlapped: `gate_proj`/`down_proj` parameters cannot be fetched until the channel mask $\mathcal{M}$ is known, and $\mathcal{M}$ requires the full `up_proj` forward. The GPU may idle while waiting for the selection result before loading the next matrices.

**Next step 1 — low-rank compression on `up_proj`.** Since `up_proj` must run at full width (it *is* the router), reducing its cost via factorization is critical. MoBE decomposes each expert matrix into a per-layer shared basis + per-expert transform: $\hat{W}^{(e)} = A_e \cdot f\bigl(\sum_j \text{softmax}(\alpha_e)_j B_j\bigr)$, with shared bases $B_j \in \mathbb{R}^{r \times d}$ and per-expert $A_e \in \mathbb{R}^{p \times r}$. Per-matrix keep-fraction = $r/d + m/E$.

We sweep the reduction on **`up_proj` alone** (gate and down left dense), varying rank $r$ at fixed $m=16$:

| up_proj reduction | wikitext PPL | MMLU | HellaSwag (acc/norm) |
| :---------------: | :----------: | :---: | :------------------: |
|   0% (baseline)   |    10.89    | 0.796 |        0.786        |
|        50%        |    12.91    | 0.766 |        0.726        |
|        60%        |    15.04    | 0.748 |        0.687        |
|        70%        |    19.52    | 0.717 |        0.639        |
|        80%        |    33.33    | 0.640 |        0.550        |

**Next step 2 — predict active channels from the residual path (break the sequential chain).** The hidden state entering layer $i$ and layer $i+1$ has cosine similarity > 0.95 (measured across all layers except layer 0). This enables predicting $(\text{experts}, \mathcal{M})$ for layer $i+1$ *during* layer $i$'s compute:

- **Inter-expert predictor (which experts):** learned MLP on the hidden state + historical trajectory; average precision 0.88. Mispredictions force synchronous reload but are rare.
- **Intra-expert predictor (which channels):** parameter-free — feed the *current* hidden state through the *next* layer's `up_proj`: $\hat{a}_{\text{up}}^{(i+1)} \approx x^{(i)} W^{\text{up},(i+1)}$, then threshold to get $\hat{\mathcal{M}}$. Average recall 0.95 (the right metric — a missed channel costs accuracy, a spurious one only wastes bandwidth).

This breaks the sequential chain: while layer $i$ computes, the predictor identifies which channel experts layer $i+1$ needs, enabling **prefetching** `gate_proj`/`down_proj` parameters from memory before they are needed — converting dynamic selection into a latency-free memory-access pattern.

## What we have achieved

1. **Experts identified at the finest granularity.** Each intermediate channel is a self-contained computation path; the relevant "expert set" per token is a subset of $K \cdot I$ channel experts rather than $K$ whole experts.
2. **The number of unique routers is much smaller.** Instead of requiring a separate $d \times (N \cdot I)$ routing matrix to select among all channel experts, the existing `up_proj` (already computed as part of the FFN) serves as the per-channel router — $K$ projections of dimension $d \times I$ that the model already performs.
3. **Protocol to find channel experts per token — without additional compute or parameters.** The `up_proj` activation magnitude ranks channel experts for each token with no extra weight reads, no calibration artifacts, and no learned router. The decision is made before `gate_proj`, enabling both `gate_proj` and `down_proj` to run at reduced width.

## Next steps

- **Low-rank compression on `up_proj`.** MoBE (Mixture-of-Basis-Experts) factorization decomposes `gate_proj`/`up_proj` into shared bases + per-expert transforms. Even-split MoBE at −33% storage achieves 73.13 HellaSwag / 76.83 MMLU (vs. dense 77.68 / 82.0) with proportional active-param reduction — providing the rank-compressed `up_proj` that still produces a valid channel-routing signal.
- **Bring reduced active parameters to actual efficiency.** The masking simulation demonstrates the accuracy ceiling; realizing the latency gain requires (1) pre-sorting channels so the kept set is a contiguous prefix for regular GEMM slices, and (2) a two-phase forward: full-width `up_proj` $\to$ select top-B $\to$ reduced-width `gate_proj`/`down_proj`.
- **Predict channel experts for future layers from the residual path.** The residual stream $x_\ell$ feeding layer $\ell$ is available before the MoE block runs. A lightweight predictor (linear probe or low-rank projection on the residual) that approximates which channels will fire at layer $\ell$ — or even at layer $\ell+1$ — enables **prefetching** the selected channel experts from memory before compute begins, converting the dynamic selection into a latency-free memory-access pattern.
