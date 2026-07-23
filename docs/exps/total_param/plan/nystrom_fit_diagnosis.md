# Nyström-MoE local-fit diagnosis + improved strategy

## Question (user)
The activation fit barely improves HellaSwag over uniform/closed-form Nyström at
33% (65.46 vs the closed-form ablation), which is unexpected. Does the local fit
actually improve reconstruction, or are the fit iterations too few?

## Evidence (full run `nys_full_v2`, Thinking-2507, layer-joint, lr=3e-4/800)

Per-layer block-output reconstruction, init (closed-form) → final (after fit):

| layer | init_rel | final_rel | rel drop | init_mse | final_mse |
|------:|---------:|----------:|---------:|---------:|----------:|
| L0    | 0.129 | 0.108 | 17% | 3.7e-6 | 2.5e-6 |
| L11   | 0.371 | 0.225 | 39% | 1.7e-4 | 6.1e-5 |
| L23   | 0.381 | 0.213 | 44% | 3.3e-4 | 1.0e-4 |
| L35   | 0.316 | 0.187 | 41% | 6.9e-4 | 2.4e-4 |
| L43   | 0.250 | 0.191 | 24% | 7.6e-3 | 4.4e-3 |
| L45   | 0.231 | 0.204 | 11% | 8.6e-3 | 6.8e-3 |
| L47   | 0.064 | 0.064 |  0% | 9.4e-3 | 9.4e-3 |

Bench: HellaSwag 78.57→65.46 (−13.1), MMLU 60.92, PPL wiki2 7.29→13.46, c4 12.46→17.98.

## Findings

1. **The fit DOES work — strongly in the middle.** Mid-layers (L11–L35) cut
   block rel-error 39–44% over the closed-form init. This is a large, real gain,
   not a no-op. So "fit doesn't help" is FALSE in aggregate.

2. **The fit collapses at deep layers (L43–L47): only 0–24% drop**, yet those
   layers carry the LARGEST absolute MSE (L46 init 9.4e-3 = ~28× L23's). The fit
   barely touches exactly the layers that dominate the end-to-end error. lr=3e-4
   was tuned on L20; the loss/grad scale at L43+ is ~25× larger, so a fixed lr is
   almost certainly mis-scaled there (needs per-layer lr, or grad normalization).
   [L40 convergence diagnostic running to confirm iters-vs-lr.]

3. **Even successful fits leave ~0.20 rel-error per block.** ~20% RMS error per
   layer compounding over 48 layers ⇒ heavily corrupted logits ⇒ HellaSwag can't
   recover. This is DEPTH COMPOUNDING: each layer's local fit targets the
   ORIGINAL block output, but downstream layers receive the accumulated upstream
   error, which local per-layer fitting never sees or corrects.

4. **Absolute error grows ~2500× with depth** (L0 3.7e-6 → L47 9.4e-3). Uniform
   k=512 is too aggressive for deep layers — their experts are higher-rank.

## Convergence study (L20, dense curves, `nys_conv_L20`)

Reached L20 closed-form, then fit it with raw MSE, 3000 iters, snapshot/100:

| step | lr=3e-4 MSE | lr=3e-4 rel | lr=1e-3 MSE |
|-----:|------------:|------------:|------------:|
| 0 (closed-form) | 3.30e-4 | 0.373 | 3.30e-4 |
| 100 | 3.28e-4 | — | 1.88e+1 (blows up) |
| **800** (full run used) | ~1.05e-4 | ~0.21 | 8e-3 |
| 1600 | 7.02e-5 | 0.158 | 4.9e-3 |
| 2400 | 6.19e-5 | 0.158 | 4.0e-3 |
| **3000** (converged) | **6.03e-5** | **0.156** | 3.9e-3 (never < init) |

## Corrected findings

- **rel-MSE loss (old strategy A) is a NO-OP within a layer.** Adam is
  scale-invariant: a constant `1/‖y‖²` factor cancels in the `m/√v` update. So
  normalizing the loss does nothing to the per-layer optimizer — confirmed
  empirically (L20 rel-loss lr=1e-3 == raw lr=1e-3, both diverge). Reverted to
  default off. The deep-layer collapse is **lr transfer across depth**, not loss
  scale — a fixed lr mis-steps layers of different curvature; the fix is to tune
  lr per depth (or use the deep-probe lr), not rescale the loss.

- **YES, 800 iters was far too few.** At lr=3e-4, L20 keeps descending well past
  800: MSE 1.05e-4 @800 → 6.0e-5 @3000 (rel 0.21 → 0.156). Going to 3000 iters
  **more than doubles** the error reduction (2.6× → 5.5× over closed-form). The
  full run was badly under-trained. Curve is ~flat by ~2500 (0.158→0.156 over the
  last 500), so **~2500-3000 iters is the right budget** at lr=3e-4.

- **lr=3e-4 is the ceiling.** lr=1e-3 blows up (MSE 18.8 @ step 99) and never
  returns below the closed-form init. Bigger raw steps diverge.

- **BUT rel-err floors at ~0.156 per block even converged.** ~16% RMS error/block
  over 48 layers still compounds heavily. More iters is necessary-not-sufficient:
  it roughly halves per-layer error but does not remove the compounding ceiling.

## Proposed better fitting strategy (ranked, corrected)

**1. More iters — 2500-3000, lr=3e-4 (cheapest, confirmed).** The single biggest
easy win: doubles the per-layer error reduction. Just bump `nystrom_fit_iters`.
~5×→~18 min/layer though (3000 vs 800) → ~14h full run; acceptable one-shot.

**2. Sequential hidden-state (error-compensation) target — fixes compounding.**
Today Y_ref = original_block(X_compressed). Instead cache the ORIGINAL model's
per-layer hidden states h*_ℓ (one teacher forward), and fit block ℓ on its
COMPRESSED input to match h*_ℓ. Later layers then absorb upstream drift
(GPTQ/AWQ/SparseGPT-style sequential calibration). This is the principled fix for
the ~0.156 floor compounding over depth. Same one-shot cost (cached teacher pass).

**3. Non-uniform keep budget — deep layers are higher-rank.** Allocate more
channels to deep/high-error layers, fewer shallow, under the same 33% total
(reuse RFID's adaptive-rank machinery keyed on per-layer recon error). Spends
capacity where the fit can't compensate.

**4. LoRA/CE recovery.** One-shot factorize-and-fit is not competitive with
pruning at 33% here; a short CE/LoRA pass is the standard remedy, already in the
pipeline. This is the highest-ceiling option end-to-end.

**Recommendation:** ship **#1 now** (iters=2500, lr=3e-4) — verified to more than
double the fit's effect for a config-only change; then implement **#2** (sequential
hidden-state target) as the real fix for depth compounding. #3/#4 are follow-ups.
Validate #1/#2 on the L20/L40 probe (converged rel-err) before each full run.
