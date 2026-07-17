# Final Proposal — GATE: per-token Global cross-expert channel selection for MoE compression

**Problem anchor (frozen).** Take a *frozen pretrained* MoE LLM and reduce its **active** expert-FFN parameters per token by ~33%, recovering accuracy with lightweight LoRA — *without* slimmable pretraining (the MoSE requirement). The deployable honest floor is real uniform-Nyström-slim ≈ **65% HellaSwag**; the target is to close most of the gap to dense (~80%) at a *real* reduction.

**Method thesis (one sentence).** For each token, rank *all* K×I FFN intermediate channels of its routed experts in a single global pool by a routing-weight-modulated ridge-leverage score and keep the global top-B (block-granular, with a per-expert floor), so per-expert width emerges from one ranking instead of being allocated per expert.

**Dominant contribution.** The *per-token global cross-expert channel selection* mechanism — verified novel: every prior neuron/channel MoE method (MoNE-neuron 2510.05781, FlexMoE 2606.27866) selects within each expert; per-token dynamic methods (AnyExperts, Alloc-MoE, MC#) prune at whole-expert/slot granularity.

## What changed after adversarial review (3/10 → target ≥7)

The review's three decisive corrections, all adopted:

1. **Never zero a routed expert.** Enforce a per-expert floor of ≥1 block. Emergent-zero-width was strictly worse than static (drops an expert the router deliberately weighted, leaving softmax mass unaccounted) — the likely cause of the naive pilot's 75.96 < 78.23. Keep "allow zero" only as an *ablation*; if it ever helps, reframe honestly as joint expert-drop + width and compare head-to-head with AnyExperts/Alloc-MoE.

2. **The routing weight `g_e` is the CORRECT cross-expert scale, not a double-count.** The marginal contribution of channel (e,i) to the token output is `g_e · SiLU(gate_{e,i})·up_{e,i}·(down row i)`. So the global score should be **`g_e`-weighted leverage** (or the full marginal-‖Δy‖² contribution score), and the earlier "avoid double-counting" instinct is likely backwards — dropping `g_e` under-weights high-routing experts. This is now a *derived* score, not a hand-wave, and a central ablation.

3. **Honest scope on "compression."** Masking-sim is the **oracle upper bound** (accuracy if any per-token width were realizable), explicitly labeled as such. The paper must carry at least the winning config to a **real** block-granular gathered-GEMM realization with a wall-clock number, or reframe as "active-FLOP reduction at iso-accuracy" and defend it. Block granularity (16/128 channels) is what makes a real speedup path exist at all; per-token *arbitrary*-width kills tensor-core utilization.

Additional honesty item (verified against code): the leverage score uses an **aggregate** covariance → it is a **static** per-channel scalar. "Per-token" applies to selection/budget, **not** to the score. State plainly; do not claim token-adaptive scoring.

## Method (final)

Per token t, routed to experts E_t (|E_t|=K), with routing weights g_{t,e} (norm-topk softmax):
- **Global score** for channel (e,i): `S_{t,e,i} = g_{t,e} · ℓ_{e,i}`, where `ℓ_{e,i} = diag((C_e+λI)⁻¹C_e)_i` is the (static, aggregate-covariance) ridge-leverage score. Ablate against: activation metric, unweighted leverage, random, and full marginal-Δy² score.
- **Block-granular global top-B**: group channels into blocks of size b∈{16,128}; keep the top ⌈B/b⌉ blocks across the K×I pool by summed block score, subject to a per-expert floor of ≥1 block. B = round(0.67·K·I).
- **Realize**: (i) masking-sim (oracle, exact accuracy at budget); (ii) real gathered variable-width GEMM at block granularity → real active-param/FLOP cut + latency.
- **Recover**: LoRA on {gate,up,down}, argued as *amortized reconstruction* over the expected per-token kept-set distribution (replaces infeasible per-token closed-form Nyström reconstruction). Report select-only+LoRA vs select+reconstruct+LoRA at a fixed mask to justify skipping reconstruction.

## Positioning vs closest prior art
- **MoSE (ICML 2026)**: router-conf→per-expert width, but needs slimmable *pretraining*; GATE is post-hoc on frozen weights + LoRA, and uses a *global pool* not per-expert budgeting.
- **FlexMoE / MoNE-neuron**: per-expert channel selection, static (FlexMoE) or within-expert (MoNE); GATE pools across experts per token.
- **AnyExperts / Alloc-MoE**: per-token but whole-expert/slot; GATE is sub-expert (channel/block) width.
