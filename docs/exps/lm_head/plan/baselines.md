# LM-Head Baselines — Implementation & Test Plan

**Target models**: `Qwen/Qwen3-30B-A3B-Thinking-2507` (primary) + one SLM (`Qwen/Qwen3-0.6B`) where the
payoff is 2–3× larger.
**Goal**: reduce **used** lm_head parameters — storage compression *or* sparse activation both count.
**Eval**: HellaSwag (0-shot `acc_norm`) + MMLU (5-shot), via the existing
`src/train/merge_slim_eval.py` → `eval_dispatch` path, plus C4 PPL as the sensitive metric.
**Dense references** (already measured, `Thinking-2507`): **HellaSwag 78.56**, **MMLU 80.91**,
stderr ≈ 0.41–0.45 pt.
**Evidence base**: `idea-stage/IDEA_REPORT.md` (5 pilots, 2026-08-14).

---

## 1. Accounting: what the ceiling actually is

Read this before designing anything — it sets what "success" can mean.

| Model | V | d | lm_head params | % of **total** | % of **active** | **max achievable saving** |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 151 936 | 2048 | 311.16 M | 1.02 % | **9.28 %** | **9.28 % of active params** |
| Qwen3-30B-A3B after −73 % expert pruning | — | — | 311.16 M | — | **~15.4 %** | ~15.4 % of active |
| Qwen3-0.6B | 151 936 | 1024 | 155.58 M | **20.70 %** | 20.70 % | **20.70 %** |
| Qwen2.5-0.5B | 151 936 | 896 | 136.13 M | **27.56 %** | 27.56 % | 27.56 % |

Effective active params for Qwen3-30B-A3B if the head is stored/read at `b` bits (BF16-equivalent):

| head treatment | effective head | effective active | Δ active |
|---|---|---|---|
| BF16 (dense) | 311.2 M | 3.353 B | — |
| INT4 (25 % storage) | 77.8 M | 3.119 B | **−6.96 %** |
| INT2 / CARVQ 1.6 bit | 38.9 / 31.1 M | 3.081 / 3.073 B | **−8.12 % / −8.35 %** |
| sparse-activate top-4096 rows only, BF16 | 8.4 M read | 3.050 B | **−9.03 %** |

**Consequences that shape the plan:**
1. Every method lives inside a **0 → 9.28 %** band of active-param reduction on the 30B. The
   difference between "good" (INT4, −6.96 %) and "perfect" (free head, −9.28 %) is **2.3 pp of active
   params**. So the decisive question is not *how much* you save but **whether accuracy moves at all**:
   a method that costs >1 stderr (0.45 pt) of MMLU is not worth 2 pp.
2. The SLM arm matters — there the same mechanism buys **20.7–27.6 %**. Run it.
3. On the 30B this composes with the repo's expert pruning, which *raises* the head's share to ~15.4 %.
   Report both denominators explicitly or the numbers read as trivial.

---

## 2. Shortlist — the 3 baselines to implement

Selected on measured evidence, not popularity. Each row states the number it must reproduce.

### B1 — Frequency-tiered head (the free static prior) — **implement first**

`COMPACT`-style rare-vocab treatment ⊕ `adaptive-softmax`-style tiering.
Partition V by **calibration unigram usage frequency** into a head tier `H` (top-`T`) and a tail.

Three sub-variants, all cheap, sharing one implementation:
- **B1-s (storage)**: head tier BF16, tail tier INT4/INT2. No rows dropped → safe for any task.
- **B1-p (prune)**: drop tail rows entirely (logit = −inf). Real param reduction; risks task tokens.
- **B1-a (sparse activation)**: store everything, but *read* only the head tier per position; tail
  contributes only via a fallback.

**Why it is #1, not an afterthought**: in the pilots this free prior beat every learned or geometric
mechanism tried. The exact argmax is in the top-4096 rows (**2.7 % of V**) on **92.5 %** of steps;
top-1024 rows carry **71.7 %** of corpus mass. An adaptive per-sequence cache *lost* to it at K≥4096
(89.4 % vs 92.5 %). Row norms are near-uniform (p99/p50 = 1.19–1.33) and `corr(log freq, ‖w‖) = −0.13`,
so **frequency is the only usable tiering axis — magnitude is not**.
**Target to beat**: itself — B1 is the floor that B2/B3 and any new method must clear.

### B2 — ARCHead (quantized low-rank core + group-INT4 residual + activation-metric correction)

arXiv:2608.02703 (Aug 2026), code `github.com/suayptalha/archead`.
**Reproduce**: **25.6 % of BF16 head storage at 1.007× relative PPL (+0.7 %)** on Qwen3-8B-Base;
3.7–3.9× head storage reduction; storage-matched naive INT4 = 1.14–1.16×.
**Why**: the best published storage/quality point anywhere in the survey, and it was validated on a
**Qwen3 model with the identical vocabulary (V=151936)** — the strongest prior that it transfers to
30B-A3B. Its win comes from fitting the correction in an **activation-derived metric**, which the
pilots independently confirmed is the dominant factor (whitening moved low-rank PPL 1477 → 42).
**This is the SOTA bar on the storage axis.**

### B3 — Residual / product VQ head (CARVQ-style; VQ-Logits as the extreme point)

CARVQ arXiv:2510.12721 — corrective adaptor (linear + non-linear) + **group residual VQ**, post-training,
**≈1.6 bits/param** with no sub-4-bit hardware requirement.
VQ-Logits arXiv:2505.10202 — one shared codebook of `K` vectors, **up to 99 %** of output-layer params
removed, 6× logit-compute speedup, **+4 % PPL** (the extreme, low-quality end).
**Why**: the highest ceiling for param reduction, and the codebook family is where the pilots point —
quantization/codebook methods beat low-rank by ~**50×** in excess PPL at matched storage.
**Reproduce**: CARVQ's ~1.6 bit/param at "reasonable" accuracy; VQ-Logits' +4 % PPL as the ceiling of
how aggressive is tolerable.

### Explicitly excluded, with the measurement that killed each

| Not implementing | Why (measured in the pilots) |
|---|---|
| Low-rank / SVD / ASVD / GroupReduce head | At matched ~25 % storage: low-rank PPL **129.6** vs 4-bit RTN **39.2**. Even at **75 %** storage the best whitened low-rank head gives only **73.9 % top-1 agreement**. 90 % of spectral energy needs 81 % of `d`. |
| CSV-Decode certified sub-vocabulary | Centroid+radius bounds certify **nothing**: 99.33 % of V survives even with an **oracle** lower bound. Slack `R·‖h‖`=62.3 vs required gap 19.7; `R ∝ C^−0.33` → needs C≈10⁷ ≫ V. Kept only as a **falsification run** (§6, F1). |
| Softmax-gauge (mean-row) removal as a *mechanism* | Subsumed by whitening for low-rank (42.26 → 42.18); **harmful** to quantization (−51 % at 3-bit, −80 % at 2-bit per-row). Keep as a reporting diagnostic only. |
| Adaptive per-sequence row cache | Loses to B1's free static prior at K≥4096. |
| Magnitude-based row pruning | Row norms near-uniform; `corr(freq, ‖w‖) = −0.13`. |

---

## 3. Metrics — and a caveat that changes the eval design

> ⚠️ **MMLU and HellaSwag cannot, by themselves, evaluate a sparse-activation head.** Both are
> **loglikelihood** tasks in lm-eval-harness: MMLU scores 4 single-token continuations (" A".." D"),
> HellaSwag scores 4 short endings. Those target tokens are all *high-frequency*, so they sit inside
> **any** frequency-tiered read set — B1-a will score ~identically to dense while reading 2.7 % of the
> head. That is a property of the benchmark, not evidence of quality. Two consequences:
>
> 1. **C4 perplexity is mandatory**, not optional. It is a full-vocabulary token-level metric and is
>    the only one of the three that is actually sensitive to head approximation.
> 2. Sparse-activation variants must be run in **strict mode** (tokens outside the read set get −inf,
>    no oracle fallback) so that a missed target token is a real, visible failure.

Report every run as a row in one frontier table:

| metric | how |
|---|---|
| **used lm_head params / token** (BF16-equiv) | primary efficiency axis; also as % of active params |
| **head storage bytes** | secondary efficiency axis (differs from the above for sparse activation) |
| **HellaSwag acc_norm** (0-shot) | vs dense **78.56**, stderr ≈0.45 |
| **MMLU acc** (5-shot, full) | vs dense **80.91**, stderr ≈0.41 |
| **C4 PPL** | sensitivity metric; `configs/eval/qwen3_30b_a3b_baseline_c4.yaml` |
| top-1 agreement + KL vs dense head | cheap diagnostics, computed on the calibration split |

---

## 4. Implementation

### 4.1 Code layout (mirrors `src/dynamic_active_param/`)

```
src/lm_head/
  __init__.py          # exports install_lm_head
  calib.py             # unigram frequency counts + hidden-state 2nd moment Sigma = E[h h^T]
  tiering.py           # B1: frequency partition, per-tier bit assignment, row pruning
  quant.py             # B1-s / B2 numerics: group RTN, INT4 residual, activation-metric fit
  vq.py                # B3: group residual VQ + corrective adaptor
  archead.py           # B2 wrapper (port or vendor from github.com/suayptalha/archead)
  install.py           # install_lm_head(model, cfg) -> binds a replacement forward
  accounting.py        # used-params / bytes-read accounting, printed like print_wsparse_accounting
  tests/
    test_tiering.py    # tier partition + strict-mode masking correctness
    test_quant.py      # dequant round-trip; accounting matches the analytic byte count
```

`install.py` follows the `install_dynamic_alloc` pattern exactly: rebind
`model.lm_head.forward` via `types.MethodType`, move per-tier tensors to the module's own device so it
works under `device_map='auto'` sharding (required — 30B-A3B is sharded across GPUs).

### 4.2 Config schema (new `prune_kwargs.lm_head` block)

`prune_kwargs` is already a free-form dict consumed by `merge_slim_eval.py`, so no
`E2EArguments` change is needed. Add:

```yaml
prune_kwargs:
  prune_ratio: 0.0            # keep the MoE path untouched for the head-only arm
  lm_head:
    enabled: true
    method: "freq_tier"       # freq_tier | archead | rvq
    calib_dir: ""             # reuse the existing scores dir; unigram counts cached here
    # --- B1 ---
    tier_size: 4096           # T
    tail_bits: 4              # 16 => storage-neutral tiering; 0 => prune tail rows (B1-p)
    head_bits: 16
    sparse_activate: false    # B1-a: read only the head tier
    strict: true              # no fallback: outside-tier logits = -inf
    # --- B2 / B3 ---
    rank: 64                  # archead low-rank core
    residual_bits: 4
    vq_groups: 8              # rvq
    vq_bits: 2
    activation_metric: true   # fit in Sigma^{1/2}; pilots say this dominates
```

### 4.3 Dispatch hook

In `src/train/merge_slim_eval.py`, immediately before the `eval_dispatch(...)` call on **every**
branch (the `test_only` branch and the dynamic branches), insert:

```python
lm_head_cfg = args.prune_kwargs.get("lm_head", {}) or {}
if lm_head_cfg.get("enabled", False):
    from src.lm_head import install_lm_head
    model = install_lm_head(model, lm_head_cfg, tokenizer=tokenizer, args=args)
    _print("[Step 4b] ✅ lm_head method installed")
```

Placing it on all branches is what makes the head arm **compose** with the existing expert-pruning
arm — needed for the ~15.4 % denominator in §1.

### 4.4 Calibration

Reuse the existing C4 calibration path. Two artifacts, both cheap and cached next to `scores_dir`:
- `unigram.pt` — token counts. **Must be counted on ≥5 M tokens**, not the ~25 k used in the pilots
  (that sample saw only 5945 distinct types and underestimates tail coverage).
- `sigma_lm_head.pt` — `Σ = E[h hᵀ]` for the post-final-norm hidden state, `d×d` = 2048² (16 MB fp32).
  Collect with a forward hook on `model.model.norm`; 128 × 512-token sequences is enough.

---

## 5. Test matrix

Naming follows the existing convention: `configs/eval/qwen3_30b_a3b_lmhead_<variant>_{hellaswag,mmlu}.yaml`.

### Phase 0 — correctness gates (no GPU-hours, local)
| # | Check | Pass condition |
|---|---|---|
| 0a | `install_lm_head` with `method=freq_tier, tier_size=V, tail_bits=16` | bit-exact vs dense logits |
| 0b | accounting vs analytic byte count | match to <0.1 % |
| 0c | strict-mode masking | tokens outside the tier get exactly −inf |
| 0d | sharded `device_map='auto'` | no cross-device tensor errors |

### Phase 1 — B1 sweep (the floor), Qwen3-30B-A3B
| Run | T | tail_bits | sparse | used head params | expected Δ active |
|---|---|---|---|---|---|
| B1-s-4 | 4096 | 4 | no | 84.1 M | −6.77 % |
| B1-s-2 | 4096 | 2 | no | 46.2 M | −7.90 % |
| B1-p-32k | 32768 | 0 (pruned) | no | 67.1 M | −7.28 % |
| B1-p-8k | 8192 | 0 (pruned) | no | 16.8 M | −8.78 % |
| B1-a-4k | 4096 | — | yes, strict | 8.4 M read | −9.03 % |
| B1-a-16k | 16384 | — | yes, strict | 33.6 M read | −8.28 % |

Each run: HellaSwag + MMLU + C4 PPL → 18 eval jobs. **B1-p is the run most likely to break MMLU**
(pruning may remove tokens the 5-shot prompt format needs) — inspect which tokens got dropped before
blaming the method.

### Phase 2 — B2 ARCHead, storage-matched to B1
Storage-match to B1-s-4 (25.6 % target from the paper) and one aggressive point.
| Run | storage | must beat |
|---|---|---|
| B2-25 | 25.6 % of BF16 | B1-s-4 on all three metrics |
| B2-15 | ~15 % | B1-s-2 |

### Phase 3 — B3 residual VQ
| Run | bits/param | note |
|---|---|---|
| B3-rvq-1.6 | ~1.6 | CARVQ operating point |
| B3-vql | ~0.1 (codebook only) | VQ-Logits extreme; expect the +4 % PPL failure |

### Phase 4 — SLM arm (where the payoff is 2–3×)
Repeat the winner of Phases 1–3 + B1-s-4 on **Qwen3-0.6B** (20.7 % of total params).
Note Qwen3-0.6B is **tied** (`tie_word_embeddings: true`) — compressing the head also changes the
input embedding unless they are untied first. **Untie before compressing**, and report that the untie
itself costs 155.6 M params, or the accounting is wrong.

### Phase 5 — composition with expert pruning
Best head method ⊕ the repo's `input_sparse` −73 % expert config, to test the ~15.4 % denominator and
whether the two compressions interact.

**Total**: ~30 eval jobs. At the observed 30B bf16 eval cost these are the dominant expense — run
Phase 0/1 first and gate the rest on Phase 1.

---

## 6. Falsification runs (do these, they protect the conclusions)

| # | Run | Purpose |
|---|---|---|
| **F1** | Clone `github.com/FastLM/CSV-Decode`, run on one shared model | The pilot says its bound certifies 99.33 % of V. Either reproduce its 18.4 % \|S\|/V or refute it. **Do not publish the criticism without this.** |
| **F2** | Replicate the pilot low-rank ladder on 30B-A3B (untied, d=2048) | The "low-rank is dead" claim rests on Qwen3-0.6B, which is **tied**. Kill criterion: if low-rank at 25 % storage lands within +5 % PPL, the exclusion in §2 is wrong and low-rank returns to the shortlist. |
| **F3** | Plain group-RTN INT4 at 25.78 % storage | The honest naive floor. Pilot: PPL 37.40 → 39.18 (+4.8 %) on 0.6B. If B2/B3 cannot beat this, the sophistication is not buying anything. |

---

## 7. Pass / fail criteria (pre-registered)

- **Headline success**: ≥ **6.9 %** active-param reduction on 30B-A3B (i.e. head ≤ INT4-equivalent)
  with **HellaSwag ≥ 78.1** and **MMLU ≥ 80.5** (both within 1 stderr of dense) **and** C4 PPL within
  +1 %.
- **Acceptable**: same budget within 2 stderr (HS ≥ 77.7, MMLU ≥ 80.1).
- **Fail / stop**: no method reaches INT4-equivalent storage within 2 stderr on both tasks → the head
  is not compressible at this scale without finetuning; escalate to a LoRA-recovery arm rather than
  pushing bits lower.
- **Sparse-activation-specific**: B1-a must be reported with C4 PPL **and** strict mode. If it shows
  dense-level MMLU/HellaSwag *and* dense-level C4 PPL at 2.7 % of head reads, verify the install is
  actually taking effect (0a/0c) before believing it.

---

## 8. Risks

1. **The ceiling is 9.28 %** on the primary target. If a reviewer expects large end-to-end wins, the
   30B arm cannot deliver them; the SLM arm (20.7–27.6 %) and the post-expert-pruning composition
   (~15.4 %) are where the story is. An Amdahl bound caps B=1 decode speedup at **1.10×** (30B) /
   **1.18×** (post-pruning) / **1.26×** (0.6B) — plan claims accordingly.
2. **MMLU/HellaSwag insensitivity** to sparse activation (§3) — mitigated by mandatory C4 PPL + strict mode.
3. **All pilot evidence is Qwen3-0.6B, tied.** F2 is the guard.
4. **`Thinking-2507` config not verified locally** — confirm `vocab_size=151936`, `hidden_size=2048`,
   `tie_word_embeddings=false` before trusting the §1 accounting.
5. **ARCHead port cost** — if vendoring its kernels is slow, run it as a numerics-only simulation
   (dequantized BF16 matmul) first; accuracy transfers, throughput claims do not.
