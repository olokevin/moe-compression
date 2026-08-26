# Week 8 — Efficient Channel Scoring: Input Sparsity

Model: **Qwen3-30B-A3B** (hidden `d=2048`, MoE intermediate `p=768`, 128
experts/layer, top-8, 48 layers). All numbers are **one-shot** (masking simulation →
eval, **no recovery fine-tuning**). HellaSwag = 0-shot acc_norm, MMLU = 5-shot acc.
Dense baseline: HellaSwag 78.56, MMLU 80.91.

---

## The problem

The best per-token channel selector is `oracle_mag` — rank channels by
`g_e·|SiLU(gate)⊙up|` and keep the global top-B across a token's K experts. But it
runs `gate_proj` + `up_proj` at full width just to decide which channels to keep.
Its whole-FFN "used" fraction is `(1+1+ρ)/3 ≥ 2/3`, so **−33.3% is its hard floor**
regardless of channel budget. Goal: score channels with something costing **≪ full
gate+up**, emit indices, gather only the kept channels of all three matrices.

---

## 1. The `input_sparse` method — one method, two sparsities

**Idea.** Instead of reading all `d=2048` coordinates of the token's input `x` to
compute the full gate/up activations, read only the top-`ρ_input` coordinates (by
`|x_i|`) of the *served* weight — then score channels from those partial activations.
Nothing is quantized, nothing extra is stored — the probe is a **view** onto the
served expert tensors (measured extra allocation: **0.00 MB**).

The score is `g_e·|SiLU(g̃ate)⊙ũp|` (same formula as the oracle, but on the
sparse-input approximation). Keep the global top-B channels across the token's K
experts, then gather all three matrices to those channels.

### Two sparsities — same kind of quantity, different roles

| symbol                   | what it keeps                                  | paid to           |
| ------------------------ | ---------------------------------------------- | ----------------- |
| **`ρ_input`**   | fraction of the token's input coordinates read | **scoring** |
| **`ρ_channel`** | fraction of the pooled K·I channels kept      | **compute** |

**Used-parameter accounting** (in units of one expert FFN = 3 matrices):

```
scoring : 2·ρ_input     (up + gate, all I rows, ρ_input of the H columns)
compute : 3·ρ_channel   (up, gate, down gathered to the kept channels)
used    = ρ_channel + 2·ρ_input/3
```

This billing is conservative — overlap is counted twice. Zero extra storage.

### Why the two sparsities are not interchangeable

Cutting `ρ_input` is **far cheaper** than cutting `ρ_channel`. Input sparsity enters
discounted 3× (two branches spread over a three-matrix FFN), while compute pays at
face value. Measured: dropping `ρ_input` from 1.0 → 0.25 (75% of coordinates gone)
loses only **0.48pt** of HellaSwag while freeing 47.5pp of budget. Dropping `ρ_channel`
from 0.20 → 0.10 at fixed `ρ_input=0.25` costs **2.66pt**.

---

## 2. Key results

### Budget sweep (`ρ_input=0.25`, `router` allocation)

| `ρ_channel` | used-param cut    | HellaSwag       | MMLU  | Δ HS vs dense |
| -------------- | ----------------- | --------------- | ----- | -------------- |
| 0.200          | −63.3%           | **76.72** | —    | −1.84         |
| 0.150          | −68.3%           | **76.47** | 78.63 | −2.09         |
| 0.100          | **−73.3%** | **74.64** | 77.67 | −3.92         |

**No cliff.** Degradation is smooth — roughly 0.24pt per percentage-point of extra cut.

## 3. How to split a fixed budget between the two sparsities

Given a budget `C = ρ_channel + 2·ρ_input/3`, many splits are feasible. Solving the
Lagrangian on the cached 8192-token per-layer error surface (no GPU, seconds):

| target cut | budget C | `ρ_input`* | `ρ_channel`* | HellaSwag |
| ---------- | -------- | ------------- | --------------- | --------- |
| −63.3%    | 0.3667   | 0.2500        | 0.2000          | 76.61     |
| −68.3%    | 0.3167   | 0.2400        | 0.1567          | 75.78     |
| −73.3%    | 0.2667   | 0.1875        | 0.1417          | 74.63     |
| −75.0%    | 0.2500   | 0.1875        | 0.1250          | 74.08     |
| −77.5%    | 0.2250   | 0.1875        | 0.1000          | 73.33     |
| −80.0%    | 0.2000   | 0.1575        | 0.0950          | 72.55     |

**`ρ_input`* = 0.1875 at every budget ≤ −73.3%** — the optimum keeps the split near
**50/50 scoring/compute**, not the 67/33 that the hand-picked `ρ_input=0.25` forces.

### The split is a flat direction

Measured iso-cost test at −73.3%: hand-picked (0.25/0.10) → 74.64, solved
(0.1875/0.1417) → 74.63 — a **dead tie**. Three tests average **+0.24pt** for
solving, inside noise. The split is a flat direction near the optimum: solving it
costs nothing and protects against corners, but there is no material accuracy to
collect.

**What it does buy is depth.** The solved split at −75.0% (74.08) ties the uniform
baseline at −73.3% (74.06) while cutting 1.7pp more — so relative to the uniform
baseline, the extra cut is free.

---

## 4. Where the method sits — the full frontier

Six budget points at best-practice settings (`bits=16`, `use_gate=true`,
`input_alloc=router`, solved split). The curve from −63.3% to −80.0%:

| cut     | HellaSwag | Δ dense | MMLU  | Δ dense |
| ------- | --------- | -------- | ----- | -------- |
| −63.3% | 76.61     | −1.95   | 79.45 | −1.46   |
| −68.3% | 75.78     | −2.78   | 78.98 | −1.93   |
| −73.3% | 74.63     | −3.93   | 77.94 | −2.97   |
| −75.0% | 74.08     | −4.48   | 77.77 | −3.14   |
| −77.5% | 73.33     | −5.23   | 76.81 | −4.10   |
| −80.0% | 72.55     | −6.01   | 76.11 | −4.80   |

Roughly **0.24pt per pp of cut**, no cliff. For contrast: Level-1 `pivchol` is at
44.15 by −87.5%, and activate-fewer-experts (top-4 of 8) is at 75.96 at −50%. At
−80.0% this method is still within 6pt of dense while using a fifth of the FFN
parameters per token.

---

## 5. Unstructured sparsity (`weight_sparse`)

**Idea.** Instead of reading whole coordinates (columns), spend the scoring budget on
individual `(channel, coordinate)` entries — the greedy-optimal read set is the top of
`|W_ji|·|x_i|`, which neither a column rule (`|x_i|` only) nor a static mask
(`|W_ji|` only) reaches. Implemented as a threshold rule: bisect a per-token `τ` until
read count matches budget. No stored mask — thresholds are derived from served weights.

**Key result at −80.0% (used = 0.200):**

| variant                    | HellaSwag | vs column-only control |
| -------------------------- | --------- | ---------------------- |
| `tau` uniform              | **74.87** | **+3.46**              |
| rectangle `0.25×0.45`     | 74.47     | +3.06                  |
| `tau` + `router` alloc    | 74.41     | +3.00                  |
| `input_sparse` + `router` | 71.41     | (control)              |

74.87 at −80.0% beats the old `input_sparse` headline (74.64 at −73.3%).

**Why it's not the default — the metadata catch.** Unstructured reads need positional
metadata (which entries to read). At 8 levels: 4 bits/weight = 17.4% of the expert FFN.
At 1 rectangle level: 1 bit/weight = 4.3%. Channel blocks of 8 cut storage to 2.8% but
give back half the gain (+0.14pt over columns). `input_sparse` remains the only
**zero-overhead** option. Unstructured sparsity is the frontier *if* metadata storage is
acceptable.

---

## 6. Low-rank / activation-aware SVD (closed negative)

**Setting.** Factor `W_up` (optionally `W_gate`) into `L·R` at rank r; score =
`|L·(R·x)|`. Tested: plain SVD, BTT (block-grid factorization), activation-aware SVD
(`W Σ^{1/2}`), shared cross-expert basis, quantized factors.

**Key reads (nominal −75% channel cut):**

| scorer                            | cost (cB) | recall | HellaSwag |
| --------------------------------- | --------- | ------ | --------- |
| SVD r32 up-only                   | 0.057     | 0.444  | 63.94     |
| BTT m2n2 r32 up+gate             | 0.229     | 0.479  | 66.83     |
| `actbasis_r32` (cheapest aware)  | 0.037     | 0.353  | —         |
| `awsvd_r128` (expensive aware)   | 0.458     | 0.535  | —         |
| `sparse_probe` q3/k25 (control)  | 0.098     | 0.629  | —         |
| Static prior (reads **nothing**)  | 0         | 0.363  | —         |

**Why it fails (structurally):**

1. **Low-rank is an averaging operator; the top-B channel set is the per-token
   *deviation* from the average.** A rank-32 activation-aware basis scores *below* the
   free static prior (0.353 vs 0.363) — its first component reconstructs the mean
   token, producing a per-token-constant score profile. More rank buys only diminishing
   returns on the residual.
2. **Effective rank of the score-error metric is 1.8–3.5 directions.** The top
   eigenvector *is* the mean token (`cos² ≥ 0.998`). Activation weighting concentrates
   budget onto this common mode — the one part a per-token ranking cannot use.
3. **Recall grows only log with cost** — matching the oracle needs rank-256 (cost 0.917),
   worse than running the oracle itself.
4. **Bytes on rank are strictly worse than bytes on precision:** quantized 3-bit
   rank-448 @ cB 0.10 → recall 0.457 vs probe 0.678 at same budget.
5. **Activation-awareness works but doesn't matter:** +0.035 recall over plain SVD at
   every rank (free, confirmed analytically), but the family is too far behind sparse
   probing for this to close the gap.

Analytic bonus: output-side SVD ≡ activation-aware input SVD (verified 3.8e-13 in
fp64) — don't run them separately.

---

## 7. Other negative results (closed)

| direction                        | result                                               |
| -------------------------------- | ---------------------------------------------------- |
| Cross-layer budget allocation    | Uniform wins; −0.16pt unweighted, −3.29pt weighted |
| Quantized probe (3-bit separate) | +13% storage and worse — dominated on both axes     |
| Column-norm coordinate weighting | CV = 0.022; nothing to weight                        |
| Learned channel router           | Oracle fails at k=768; cheap predictors cap 0.63     |

---

## Summary and current status

`input_sparse` is the established channel scorer: **−73.3% used parameters at 74.64
HellaSwag / 77.94 MMLU with zero extra storage** — the best zero-overhead result. A
solved budget split extends the frontier smoothly to −80.0% (72.55 / 76.11) with no cliff.
The two sparsities (`ρ_input` for scoring, `ρ_channel` for compute) are the minimal
description: one method, one formula, two knobs. Every alternative (low-rank, quantized,
offline) is dominated.

Full detail: `docs/exps/dynamic_active_param/efficient_scorer.md`.
