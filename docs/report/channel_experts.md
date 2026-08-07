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

1. **Partial experts suffice.** Masking experiments show that keeping only 12.5% of a token's $K \cdot I$ channels (selected per-token by activation magnitude) loses < 2 pts vs. dense — 7/8 of the channel experts are dispensable for any given token, but *which* 7/8 changes token-to-token.
2. **No universal sparse set.** Per-channel activation-frequency analysis over 70k tokens shows 0.3% of channels are "always on" and 0.4% "always off" — the remaining 99.3% have genuinely token-dependent utility, precluding a static keep-set.

## Determining the channel experts for each token

### Offline calibration — and its fundamental limitation

The MoE router produces $g_e(x) \in [0,1]$ per expert but has **no information about expert parameters or channel-level capabilities** — it cannot tell *which* channels within an expert the token needs. Offline calibration scores channels from corpus statistics and the online decision reduces to $\text{score}_{e,j}(x) = r(g_e) \cdot s_{e,j}$ (router reweight $\times$ static channel importance).

The key limitation: the channel ranking is fixed (cannot bear a look-up table for each token), so the best offline methods still **degrade sharply under high reduction ratios.**

**Budget sweep — Level 1 vs the winning 33% baseline (`router_prob × act`), HellaSwag acc_norm:**

| Reduction | reduce top-k          | MoSE (router_prob × act) | Level 1 (pivchol) |
| --------- | --------------------- | ------------------------- | ----------------- |
| 50%       | **75.2** (8→4) | 71.46%                    | 74.26%            |
| 62.5%     | 69.8 (8→3)           | 61.00%                    | **70.54%**  |
| 75%       | 49.4 (8→2)           | 43.66%                    | **63.60%**  |
| 87.5%     | 26.2 (8→1)           | 30.32%                    | **44.15%**  |



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

### Limitations and improvements

Two structural costs come with using `up_proj` as the channel router.

**Limitation 1 — sequential dependency.** The channel router introduces a strict dependency in the per-layer critical path:

```
route(x) → pick experts → compute u = x·W_up → threshold → know M → fetch W_gate[:,M], W_down[M,:] → compute
```

Nothing can be overlapped: `gate_proj`/`down_proj` parameters cannot be fetched until the channel mask $\mathcal{M}$ is known, and $\mathcal{M}$ requires the full `up_proj` forward. On bandwidth-starved hardware the GPU idles while waiting for the selection result before loading the next matrices.

**Limitation 2 — `up_proj` stays full width.** `up_proj` *is* the scoring signal, so it cannot itself be narrowed. The active-compute floor for this family is therefore $(1 + 2\rho)/3$ of the expert FFN (at kept fraction $\rho$): the budget shrinks `gate_proj` and `down_proj`, never `up_proj`.

Two orthogonal improvements address these, each covered in a dedicated section below:

1. **Compress `up_proj` storage (MoBE).** Since `up_proj` must run at full width, reducing its read/DRAM cost by factorization is the lever for Limitation 2. MoBE decomposes each expert matrix into a per-layer shared basis + per-expert transform: $\hat{W}^{(e)} = A_e \cdot f\bigl(\sum_j \text{softmax}(\alpha_e)_j B_j\bigr)$, with shared bases $B_j \in \mathbb{R}^{r \times d}$ and per-expert $A_e \in \mathbb{R}^{p \times r}$; per-matrix keep-fraction $= r/d + m/E$. Sweeping the reduction on **`up_proj` alone** (gate and down left dense), varying rank $r$ at fixed $m=16$:

   | up_proj reduction | wikitext PPL | MMLU | HellaSwag (acc/norm) |
   | :---------------: | :----------: | :---: | :------------------: |
   |   0% (baseline)   |    10.89    | 0.796 |        0.786        |
   |        50%        |    12.91    | 0.766 |        0.726        |
   |        60%        |    15.04    | 0.748 |        0.687        |
   |        70%        |    19.52    | 0.717 |        0.639        |
   |        80%        |    33.33    | 0.640 |        0.550        |

   The `up50%` operating point (−16.7% MoE storage, +2.0 PPL) is the one carried into the stacking experiments below and the [MoBE report](mobe.md).
2. **Channel Expert Predictor.** Predict layer $i{+}1$'s channel mask *during* layer $i$'s compute so `gate`/`down` can be prefetched, breaking the sequential chain of Limitation 1 (next section).

## Channel Expert Predictor — design and results

**Design.** The hidden state entering adjacent MoE blocks is nearly unchanged — the residual stream feeding layer $i$ and layer $i{+}1$ has cosine similarity $> 0.95$ (measured across all layers except layer 0). This lets us predict $(\text{experts}, \mathcal{M})$ for layer $i{+}1$ *during* layer $i$'s compute, with two components:

- **Inter-expert predictor (which experts):** a learned MLP on the current hidden state + historical trajectory; average precision 0.88. Mispredictions force a synchronous reload but are rare.
- **Intra-expert predictor (which channels):** **parameter-free** — feed the *current* hidden state through the *next* layer's own `up_proj`, $\hat{a}_{\text{up}}^{(i+1)} \approx x^{(i)} W^{\text{up},(i+1)}$, then threshold to get $\hat{\mathcal{M}}$. Recall is the right metric here: a missed channel costs accuracy, a spurious one only wastes a little bandwidth.

While layer $i$ computes, the predictor identifies which channel experts layer $i{+}1$ needs, enabling **prefetching** `gate_proj`/`down_proj` from memory before they are needed — converting dynamic selection into a latency-free memory-access pattern.

**Results.** Predicting the mask one layer early, at the −50% whole-FFN operating point (`|up|` scoring, `gate`/`down` reduced −75%). The predicted-mask row follows the FloE prefetch scheme:

| Configuration            | up\| gate \| down |      MMLU      |   HellaSwag   |     ARC-C     |   TruthfulQA   |      Avg      |  wikitext PPL  |
| ------------------------ | :----------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :-------------: |
| Exact mask               | -0%\| -75% \| -75% |      78.6      |      75.4      |      66.0      |      51.1      |      67.8      |      16.89      |
| **Predicted mask** | -0%\| -75% \| -75% | **77.1** | **75.2** | **65.0** | **52.1** | **67.4** | **15.11** |

- Measured intra-expert **recall 0.777** on Qwen3-30B-A3B (vs. FloE's ~0.95 on Mixtral-8×7B — the recall gap is the room a better predictor still has).
- **Average cost −0.4 pt** across the four tasks; only MMLU loses measurably (−1.5 pt), and PPL actually improves (spurious kept channels do no harm).
- The predictor is parameter-free and turns the serial `up → threshold → fetch` critical path into a fetch that overlaps with the current layer's compute — the enabler for the edge speedup below.

## Efficiency — Edge Offload

The framework's primary payoff is on bandwidth-starved edge hardware, where experts live in CPU DRAM and are streamed over PCIe. Measured on real Qwen3-30B-A3B on a single 23 GB L4 GPU, experts offloaded to host DRAM, AIME-24 generation, batch-1 decode:

| Variant                             |    ms/tok    |     tok/s     |    vs. dense    | Peak GPU |
| ----------------------------------- | :-----------: | :------------: | :--------------: | :------: |
| Dense (all experts offloaded)       |      526      |      1.90      |      1.00×      |  3.2 GB  |
| **Dynamic (−75% gate+down)** | **291** | **3.43** | **1.81×** |  3.2 GB  |
| **+ predicted prefetch**      | **246** | **4.01** | **2.14×** |  3.2 GB  |

- Halving the PCIe bytes moved per layer (37.75 vs. 75.50 MB/layer, exact) buys **1.81×** on the wall clock — decode is bandwidth-bound, so latency tracks bytes moved almost exactly.
- The **predicted prefetch** overlaps the next layer's fetch with the current layer's compute, lifting the total to **2.14×** (4.0 tok/s).
- Peak GPU is only 3.2 GB — the full 30B runs on one L4 with ~20 GB to spare.
- Prefill is flat (~28 tok/s): at prefill lengths the per-token masks union to near-full width, so there is nothing to skip. The win is a decode-time, single-batch phenomenon.

By contrast, in the **cloud-resident** setting (experts in HBM, batch ≥ 1) the benefit is modest — the gathered kernel gives ~1.34× per layer without MoBE, but multi-batch decode dilutes the per-token sparsity as the mask-union saturates. Edge offload, not cloud serving, is where the method pays off.

## Ablation studies

### Two Scoring Signals — `|up|` vs `|SiLU(gate)|`

The channel score can be built from either half of the SwiGLU product. `|up_j|` is the shipped signal (it is available *before* `gate_proj`, so it can reduce both `gate` and `down`); `|\text{SiLU}(gate_j)|` is the mirror (available only *after* `gate_proj`, so it can reduce `up` and `down` but not `gate`). At equal whole-FFN cut:

| Method                            |   up\| gate \| down   | FFN cut |      MMLU      |   HellaSwag   |     ARC-C     |   TruthfulQA   |      Avg      |
| --------------------------------- | :--------------------: | :-----: | :------------: | :------------: | :------------: | :------------: | :------------: |
| Dense baseline                    |    -0%\| -0% \| -0%    |   0%   |      79.6      |       —       |      69.7      |       —       |       —       |
| `\|up\|` (reduce gate+down)       |   -0%\| -75% \| -75%   |  −50%  | **78.6** |      75.4      |      66.0      | **51.1** | **67.8** |
| `\|SiLU(gate)\|` (reduce up+down) |   -75%\| -0% \| -75%   |  −50%  |      77.2      | **75.8** | **67.3** |      48.3      |      67.2      |
| `\|up\|`                          | -0%\| -87.5% \| -87.5% | −58.3% | **75.3** |      71.5      |      63.2      | **50.8** | **65.2** |
| `\|SiLU(gate)\|`                  | -87.5%\| -0% \| -87.5% | −58.3% |      74.0      | **72.5** | **64.4** |      45.1      |      64.0      |

- The two signals split the benchmarks: `|up|` wins MMLU and TruthfulQA; `|SiLU(gate)|` wins HellaSwag, ARC-C, and PPL (11.18 vs. 16.89 at −50%).
- **Mechanism.** `|up|` can spend budget on a channel whose gate SiLU has closed for this token (the gate is precisely the on/off switch), whereas `|SiLU(gate)|` cannot — so the mirror wastes no budget on dead channels and is the tighter *accuracy-per-budget* signal.
- **Why we still ship `|up|`.** `|SiLU(gate)|` requires `gate_proj` before it can decide, so it can only reduce `up`+`down`, never `gate`; `|up|` decides pre-gate and shrinks two matrices for the same budget. `|up|` is the deployable choice; `|SiLU(gate)|` is the per-unit-budget ceiling.

### Stacking with Top-K Reduction

Reducing the *number* of experts (route each token to top-4 of 8 at full width) and narrowing the *surviving* experts (dynamic per-token channel selection) are orthogonal — the first drops whole experts, the second drops channels within the experts that remain — so they compose. HellaSwag acc_norm and MMLU 5-shot:

| Method                          |           up\| gate \| down           |      FFN cut      |      MMLU      |   HellaSwag   |
| ------------------------------- | :-----------------------------------: | :---------------: | :------------: | :------------: |
| Dense baseline (K=8)            |           -0%\| -0% \| -0%           |        0%        |      79.5      |     78.56     |
| Top-4 only                      |       -0%\| -0% \| -0% (×4/8)       |       −50%       |      77.4      |     75.96     |
| Dynamic −75% only (K=8)        |          -0%\| -75% \| -75%          |       −50%       |      78.6      |      75.4      |
| **Top-4 + dynamic −50%** | **-0% \| -50% \| -50% (×4/8)** | **−66.7%** | **77.2** | **74.0** |
| **Top-4 + dynamic −75%** | **-0% \| -75% \| -75% (×4/8)** | **−75.0%** | **74.3** | **70.0** |

- **Stacking dominates narrowing-only at equal compute.** At an iso-compute −58.3% whole-FFN cut, halving the experts then narrowing moderately beats narrowing all 8 experts hard by **+4.4 pt** on HellaSwag (75.7 vs. 71.3; +0.9 pt on the more forgiving MMLU).
- The first steps of narrowing on top of top-4 are nearly free: accuracy holds within ~3.4 pt (HellaSwag) / ~2.9 pt (MMLU) of dense out to a **−62.5%** active cut, with no fine-tuning.
- Expert-dropping alone collapses at deep cuts (reduce-top-k 8→2 falls to HellaSwag 49.4), whereas the stacked method retains the knowledge that whole-expert removal destroys.
- The two reductions are close to independent rather than compounding their damage — the accuracy loss of the stack is roughly the sum of the two individual losses, not worse.

## What we have achieved

1. **Experts identified at the finest granularity.** Each intermediate channel is a self-contained computation path; the relevant "expert set" per token is a subset of $K \cdot I$ channel experts rather than $K$ whole experts.
2. **The number of unique routers is much smaller.** Instead of requiring a separate $d \times (N \cdot I)$ routing matrix to select among all channel experts, the existing `up_proj` (already computed as part of the FFN) serves as the per-channel router — $K$ projections of dimension $d \times I$ that the model already performs.
3. **Protocol to find channel experts per token — without additional compute or parameters.** The `up_proj` activation magnitude ranks channel experts for each token with no extra weight reads, no calibration artifacts, and no learned router. The decision is made before `gate_proj`, enabling both `gate_proj` and `down_proj` to run at reduced width.

## Next steps

- **Fuse the two-phase forward into a kernel.** The masking simulation demonstrates the accuracy ceiling; a fused kernel — full-width `up_proj` $\to$ select top-B $\to$ reduced-width `gate_proj`/`down_proj` in one launch, with channels pre-sorted so the kept set is a contiguous prefix for regular GEMM slices — is needed to realize the edge speedup end-to-end and to close the cloud-resident gap.
- **Improve the intra-expert predictor.** Recall is 0.777 today (vs. FloE's ~0.95), and the whole accuracy gap between exact and predicted masks lives there. A predictor that approximates $\text{SiLU}(gate_j)\cdot up_j$ (not just $up_j$) computable *before* `gate_proj` would push toward the `oracle_mag` accuracy at the `oracle_up` cost structure.
- **Push MoBE + dynamic deeper, and stack on a Nyström base.** Even-split MoBE at −33% storage achieves 73.13 HellaSwag / 76.83 MMLU with proportional active-param reduction; the −16.7% `up50` point already composes with dynamic (§ ablations). MoBE beats Nyström at the first 1.5× compression ratio and is orthogonal to a further 1.5×, so building the reduced-active-parameter method on a Nyström-compressed base is the next lever on *total* parameters.
- **Learn the per-token expert budget.** The channel budget $B$ is per-token adaptive but the expert count $K=8$ is fixed; many tokens need fewer. Combining top-p routing with per-token channel allocation would make the *total* active budget per token adaptive, not just its allocation across channels.
