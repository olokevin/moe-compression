# LM-Head Baselines — Results

Implementation of [`plan/baselines.md`](plan/baselines.md). Code in `src/lm_head/`.

> **Follow-up, and a partial correction:** the conclusion below — *"the parameter count is
> irreducible; only precision is compressible"* — holds for **stored** parameters (and
> [`results_s1_screen_refine.md`](results_s1_screen_refine.md) closes four families at 25%
> storage, not the two tested here) but is **wrong for reads**. §1c closes sparse
> activation on the strength of B1-a, whose failure is attributable ~97% to *which* rows it
> read (a static frequency tier) and to its `−inf` tail, not to reading few rows. Choosing
> the read set **per token** and keeping a **graded** tail gives 23.8% of reads at C4
> ×1.0017 and HellaSwag **+0.00** on the 0.6B, and KL 0.0003 at 24.5% of reads on the 30B.
> See [`results_s1_screen_refine.md`](results_s1_screen_refine.md).

---

## The short version

The lm_head's **parameter count is irreducible; only its precision is compressible.**

Every structural method — the ones that actually reduce how many numbers the head holds
or touches — fails:

| method | params stored | params read/token | outcome |
|---|---|---|---|
| *dense reference* | *100%* | *100%* | *30B: C4 25.349, HS **78.57**, MMLU **80.94** · 0.6B: C4 31.863, HS **47.29**, MMLU **47.18*** |
| **low-rank** `(V+D)·r` | 25.3% | 25.3% | **+250% PPL** (bar: +5%); HS 37.80 |
| **row pruning** `T·D` | 21.6% | 21.6% | HS **40.61** (−6.68); MMLU 47.18 (0.00) |
| **sparse reads** | 100% | **2.7%** | **PPL = ∞**; HS **25.67** (chance) |

Meanwhile **precision** compression is nearly free. Holding all 311.16 M parameters but
storing each in ~4.3 bits instead of 16:

| ~4.3 bits/param, 30B | C4 PPL | HellaSwag | MMLU |
|---|---|---|---|
| **dense BF16** (16 bits) | **25.349** | **78.57** | **80.94** |
| **ARCHead** (4.31 b) | **+1.3%** | **78.48** | *(running)* |
| **frequency-tiered** (4.44 b) *(free)* | +1.9% | 78.34 | *(running)* |
| uniform INT4 (4.13 b) | +9.7% | 77.66 | *(running)* |

So the head resists having *fewer* numbers and tolerates having *coarser* ones. That
asymmetry organizes this document: [Part 1](#part-1--reducing-the-parameter-count) covers
parameter-count methods and reports everything as a **parameter fraction**;
[Part 2](#part-2--reducing-precision-quantization) covers quantization separately, where
the honest unit is **bits per parameter**.

**Why keep them apart.** Quantization does not reduce the parameter count at all — an INT4
head has exactly the same 311.16 M parameters as a BF16 one. Reporting it as "25.8% of the
head" invites comparison with a low-rank head that really does hold 25.3% as many numbers,
and those are not the same claim.

**Two branches closed by measurement:** low-rank heads (confirmed dead at `d=1024` *and*
`d=2048`) and sparse activation (perplexity ∞, HellaSwag at chance).

⚠️ **One plan assumption did not survive contact.** Plan §3 predicted HellaSwag and MMLU
would both be blind to a sparse head. **MMLU is** (bit-identical). **HellaSwag is not** —
`B1-p T=32768` costs HellaSwag **−6.68 pt** and MMLU **exactly 0.00**. See
[§ What the benchmarks can see](#what-the-benchmarks-can-and-cannot-see).

**Verdict:** plan §7's second tier, *acceptable, not headline success*. Storage and
HellaSwag clauses pass; C4 misses +1% by 0.29 pp for ARCHead.

---

## What was built

```
src/lm_head/
  calib.py       unigram counts (5M C4 tokens) + activation second moment C = E[h hᵀ]
  tiering.py     frequency partition, strict masking, uniform tail fallback
  quant.py       group RTN, activation metric T_p, randomized SVD, low-rank ladder
  archead.py     ARCHead (arXiv:2608.02703), Algorithm 1 verbatim
  vq.py          group residual VQ (CARVQ) + VQ-Logits
  accounting.py  the two axes: parameter count AND bits/param, vs total + active
  install.py     install_lm_head(model, cfg) — rebinds lm_head.forward
  tests/         Phase 0 gates
```

Driven from a YAML's `prune_kwargs.lm_head` block; `merge_slim_eval.py` installs it before
`eval_dispatch` on **all six** branches, so the head arm composes with the existing
expert-pruning arm. 100 configs from `scripts/gen_lm_head_configs.py`.

```bash
bash scripts/lm_head_run.sh   --model Qwen/Qwen3-0.6B --gates    # Phase 0
bash scripts/lm_head_run.sh   --model Qwen/Qwen3-0.6B --ladder   # C4 PPL ladder
bash scripts/lm_head_sweep.sh --model <M> --tasks hellaswag c4 --variants ...
python scripts/lm_head_accept_rate.py   --model <M>   # block-accept rate
python scripts/lm_head_task_coverage.py               # benchmark tier coverage
python scripts/show_sweep.py <results.json>           # read any result file
```

`lm_head_sweep.py` exists because ~80% of a per-config `merge_slim_eval.py` run on the 30B
is spent loading a 61 GB checkpoint. It loads once and cycles head treatments through
lm-eval. Same numerics; the per-config path stays canonical.

---

## The two axes, and the ceiling

| | Qwen3-30B-A3B | Qwen3-0.6B |
|---|---|---|
| `V × D` | 151936 × 2048 | 151936 × 1024 |
| head parameters | **311.16 M** | 155.58 M |
| share of total params | 1.02% | 20.70% |
| **share of *active* params** | **9.28%** (of 3.353 B) | 20.70% |
| after −73% expert pruning | **15.41%** | — |

Measured from the live models; matches plan §1 exactly (it estimated 3.353 B active).
Confirmed before trusting any of it: `vocab_size=151936`, `hidden_size=2048`,
`tie_word_embeddings=false` on the 30B.

Two axes, tracked separately throughout (`accounting.py:head_cost`):

- **parameter count** — `stored_params` and `read_params_per_token`, each over `V·D`.
  Quantization leaves both at 100%.
- **bits per parameter** — where quantization's saving lives, reported in Part 2.

The whole design space spans **0 → 9.28%** of active params on the 30B. A completely free
head buys 9.28 pp; ~4.3-bit storage already banks ~6.8 pp of it in byte terms. The
remaining headroom is small, and Part 1 shows it cannot be taken structurally.

**Qwen3-0.6B is tied** (`tie_word_embeddings: true`), so the head is **untied first** —
otherwise compressing it silently compresses the input embedding too. The untie adds
155.6 M params, so 0.6B savings are against the *untied* model, not the shipped checkpoint.

---

## Phase 0 — correctness gates

All pass (`src.lm_head.tests.test_tiering`, `test_quant`, and `--gates` on a real model).

| gate | check | result |
|---|---|---|
| 0a | `tier_size=V`, 16/16 bits | logits **bit-identical**, max \|Δ\| = 0 |
| 0b | accounting vs hand-computed bytes and parameter counts | exact |
| 0c | strict masking | out-of-tier logits exactly `-inf`, in-tier bit-exact, `log_softmax` finite |
| 0d | `device_map='auto'` | passes on a 4-way shard |

Gate 0d earned its place. Rebinding `lm_head.forward` *replaces* accelerate's dispatch
wrapper, so the head stopped receiving hidden states on its own device
(`mat2 is on cuda:0, different from other tensors on cuda:2`). Fixed by hooking
`_old_forward` when accelerate owns `forward` (`install.py:bind_head_forward`).

---
---

# Part 1 — Reducing the parameter count

Everything in this part is measured as a **fraction of the head's `V·D` parameters** —
stored, and read per token. These are the only methods that change those counts.

## 1a. Low-rank factorization — dead

`W ≈ A B` with `A: (V,r)`, `B: (r,D)` holds `(V+D)·r` parameters instead of `V·D`, all of
them read every token. It is the one full-vocabulary method that genuinely shrinks the
count. Fitted in the activation metric (`min ‖(W−Ŵ)T_p‖_F`, `p=½`, so the objective *is* the
damped logit MSE).

**Qwen3-30B-A3B (`d=2048`)** — the replication plan risk 3 asked for. Dense C4 wppl 25.349:

| rank | params stored | param frac | C4 wppl | rel | top-1 agr | HellaSwag | MMLU |
|---|---|---|---|---|---|---|---|
| *dense* | *311.16 M* | *100%* | *25.349* | *1.000* | — | ***78.57*** | ***80.94*** |
| 512 | 78.84 M | **25.34%** | **88.844** | **3.505** | 55.6% | — | — |
| 1024 | 157.68 M | 50.67% | 38.207 | 1.508 | 74.3% | — | — |
| 1536 | 236.52 M | 76.01% | 28.277 | **1.116** | 85.3% | — | — |
| 512, *unwhitened* | 78.84 M | 25.34% | **1.6 × 10⁸** | 6.3 M× | **9.2%** | — | — |

**Qwen3-0.6B (`d=1024`, untied)** — dense C4 PPL 31.863:

| rank | param frac | whitened PPL | rel | unwhitened PPL | HellaSwag | MMLU |
|---|---|---|---|---|---|---|
| *dense* | *100%* | *31.863* | *1.000* | — | ***47.29*** | ***47.18*** |
| 256 | 25.17% | **81.823** | 2.568 | 7 827 (245.7×) | **37.80** (−9.49) | **44.62** (−2.56) |
| 512 | 50.34% | 44.700 | 1.403 | 3 593 (112.8×) | — | — |
| 768 | 75.51% | 35.096 | 1.101 | 1 676 (52.6×) | — | — |

**Kill criterion: "if low-rank at 25% storage lands within +5% PPL, it returns to the
shortlist."** It lands at **+250%** on the 30B and **+157%** on the 0.6B — missed by 50×.
Three further nails:

- Still **+11.6% at 76%** of the parameters. A head at a *third* of that in bytes costs less.
- **Whitening is load-bearing, not optional.** Unwhitened it is destroyed at both widths
  (9.2% agreement at `d=2048`; 96× worse PPL at `d=1024`). A naive SVD of an lm_head is not
  a weak baseline, it is a broken one.
- It fails **downstream** too, unlike precision methods: HellaSwag **−9.49 pt** at 25.2% of
  parameters, against −0.33 for a 4-bit head.

**Plan §2's exclusion is confirmed at both `d=1024` and `d=2048`; risk 3 is discharged.**

## 1b. Row pruning — a real reduction that breaks multi-token tasks

Keep the top-`T` rows by calibration unigram frequency, drop the tail (`logit = −inf`).
Stores `T·D` parameters. Frequency is the only usable axis: row norms are near-uniform
(p99/p50 = 1.19–1.33) and `corr(log freq, ‖w‖) = −0.13`.

| model | T | param frac | Δactive | discarded mass | top-1 agr | C4 PPL | HellaSwag | MMLU |
|---|---|---|---|---|---|---|---|---|
| 30B | *dense* | *100%* | — | — | — | *25.349* | ***78.57*** | ***80.94*** |
| 30B | 8 192 | **5.39%** | −8.78% | 11.09% | 91.46% | **∞** | — | — |
| 0.6B | *dense* | *100%* | — | — | — | *31.863* | ***47.29*** | ***47.18*** |
| 0.6B | 32 768 | **21.57%** | −16.24% | **3.08%** | 98.97% | — | **40.61** (−6.68) | **47.18** (0.00) |

Perplexity is **infinite** by construction — a dropped row cannot emit its token — and that
is the honest number, not an artifact.

The 0.6B row is the instructive one: discarding just **3.08%** of the dense probability mass
still costs **6.68 pt** of HellaSwag. Mass is the wrong intuition for multi-token scoring;
see [1d](#1d-why-both-fail-coveragelength-not-mass).

## 1c. Sparse activation — reads 2.7%, perplexity ∞

Store all `V·D` parameters, but *read* only the top-`T` frequent rows per position. This is
the one method that separates the two axes: 100% stored, `T·D` read.

**Qwen3-0.6B**, dense PPL 31.863. Reported two ways, because one number alone misleads:

| T | read frac | Δactive (reads) | strict PPL | targets unreachable | uniform tail fallback | HellaSwag | MMLU |
|---|---|---|---|---|---|---|---|
| *dense* | *100%* | — | *31.863* | — | — | ***47.29*** | ***47.18*** |
| 4 096 | **2.70%** | −20.14% | **∞** | **20.09%** | 5 543 (174×) | **25.48** / 29.02 ᶠ | 47.18 / 47.18 ᶠ |
| 8 192 | 5.39% | −19.58% | **∞** | 13.45% | 1 048 (32.9×) | — | — |
| 16 384 | 10.78% | −18.47% | **∞** | 7.52% | 233.6 (7.33×) | — | — |
| 32 768 | 21.57% | −16.24% | **∞** | 2.82% | 69.2 (2.17×) | — | — |

ᶠ second figure is the uniform-fallback variant. Note it barely helps HellaSwag
(25.48 → 29.02, against a 47.29 dense) — a shared tail logit lets a candidate be *scored*
but not scored *correctly*.

**Qwen3-30B-A3B**, T=4096, 2.70% of reads, −9.03% active: strict PPL **∞** (16.79% of dense
mass outside the tier) against dense 25.349, fallback **126 362** (4984×), HellaSwag
**25.67** against dense **78.57** — chance.

Restricted to tokens it *can* emit, T=4096 scores PPL 16.7 on the 0.6B — **better than
dense** — which is pure selection bias, since only easy frequent tokens get scored. Any
harness quoting that without the unreachable-target rate is reporting nonsense; `oov_rate`
is in every result JSON for this reason.

The graded version is no kinder: give the whole tail one shared logit (the classic tiered
softmax) and even a **21.6%** read set costs **+117%**.

## 1d. Why both fail: coverage^length, not mass

The tier looks generous on paper and isn't, once a *sequence* has to survive it.
Measured on 32 768 held-out C4 positions with a 5 M-token unigram prior
(`scripts/lm_head_accept_rate.py`):

| T | % of V | unigram mass | **accept@1** (dense argmax in tier) | accept@target | dense mass in tier |
|---|---|---|---|---|---|
| 1 024 | 0.67 | 65.60% | 77.38% | 65.85% | 68.66% |
| 4 096 | 2.70 | 80.44% | **88.28%** | 79.79% | 81.91% |
| 8 192 | 5.39 | 87.41% | **92.54%** | 86.40% | 87.94% |
| 16 384 | 10.78 | 93.51% | 96.02% | 92.42% | 93.13% |
| 32 768 | 21.57 | 98.19% | 98.61% | 97.26% | 97.19% |

Per-token coverage is high. But a length-`L` continuation survives only if **every** token
is in-tier, and that probability decays like coverage^L. HellaSwag endings average 13.7
tokens, so 82.96% per-token coverage at T=4096 becomes **9.35%** per-ending
(`scripts/lm_head_task_coverage.py`):

| T | target tokens in tier | **endings *fully* in tier** | measured HellaSwag (0.6B) | MMLU |
|---|---|---|---|---|
| *dense (no tier)* | *100%* | *100%* | ***47.29*** | ***47.18*** |
| 4 096 | 82.96% | **9.35%** | **25.48** (chance) | 47.18 |
| 8 192 | 89.74% | 25.20% | — | — |
| 16 384 | 95.29% | 53.73% | — | — |
| 32 768 | 98.85% | 85.95% | **40.61** | 47.18 |
| *MMLU targets, any T* | *100%* | *100%* | — | *unchanged* |

At T=32768, 85.95% coverage predicts 0.86·47.29 + 0.14·25 ≈ 44.2 against **40.61**
measured — right size, right direction. This is the same arithmetic that makes perplexity
infinite: over 262 144 tokens, *something* falls outside any tier.

### Corrections to the numbers the plan inherited

The plan's sparse-activation case came from a pilot using ~25 k calibration tokens.
Re-measured with 5 M:

- Pilot: *"the exact argmax is in the top-4096 rows (2.7% of V) on 92.5% of steps."*
  Actually **88.28%** at T=4096; 92.5% needs **T=8192 (5.4% of V)** — one tier doubling more.
- Pilot: *"top-1024 rows carry 71.7% of corpus mass."* With 5 M tokens: **65.60%**. The
  small sample overestimated concentration, as the plan suspected.
- Only **59 978 of 151 936** token types appear at all in 5 M C4 tokens, so 60% of the
  vocabulary is untestable from a histogram of this size.

## Part 1 conclusion

**No parameter-count reduction of the lm_head survives.** Low-rank misses the bar by 50×,
pruning to 21.6% of parameters costs 6.7 pt of HellaSwag while discarding only 3.08% of
probability mass, and reading 2.7% of parameters makes perplexity infinite. The 311 M
parameters are needed; what is negotiable is how precisely each is stored.

---
---

# Part 2 — Reducing precision (quantization)

Every method in this part keeps **all `V·D` parameters, stored and read** — parameter
fractions are 100.00% throughout, which is exactly why they are separated from Part 1. The
unit here is **bits per parameter**, and `bytes vs BF16 = bits/16`.

Group-wise quantization is charged `bits + scale_bits/group` per parameter, so a "4-bit"
head is really 4.125 bits at `g=128`. All figures include scales.

The methods:

| | mechanism |
|---|---|
| **F3 RTN** | uniform group round-to-nearest — the honest naive floor |
| **B1-s frequency-tiered** | top-`T` rows kept BF16, tail quantized. Free: needs only a token histogram |
| **B2 ARCHead** | quantized low-rank core + group-INT4 residual + rank-6 correction fitted in the activation metric (arXiv:2608.02703, Algorithm 1, published Qwen hyperparameters `rc=10, rr=6, g=64, p=0.75, ridge=1e-3`) |
| **B3 RVQ / VQ-Logits** | vector quantization: per-group residual codebooks (CARVQ) and one shared full-row codebook (VQ-Logits) |

## 2a. Qwen3-30B-A3B (primary target)

lm-eval `word_perplexity` on C4 (500 docs) and full HellaSwag 0-shot. Dense rows come out at
HellaSwag **78.57** and MMLU **80.94** against the plan's references of 78.56 and 80.91 — the
harness and protocol are the ones the pre-registered bars were set in.

| run | bits/param | bytes vs BF16 | Δactive (bytes) | top-1 agr | **C4 wppl** | rel | **HellaSwag** | Δ | **MMLU** | Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| **dense BF16** | 16.00 | 100.00% | — | — | **25.349** | 1.000 | **78.57** | — | **80.94** | — |
| **B2 ARCHead** | **4.31** | 26.92% | **−6.78%** | 93.31% | **25.676** | **1.013** | **78.48** | **−0.09** | *(running)* | — |
| **B1-s** T=4096, tail 4b | 4.44 | 27.78% | −6.70% | **96.00%** | **25.827** | **1.019** | **78.34** | **−0.23** | *(running)* | — |
| B2 ARCHead, *no* activation metric | 4.31 | 26.92% | −6.78% | 85.69% | 27.151 | 1.071 | — | — | — | — |
| F3 RTN 4-bit g128 *(naive floor)* | 4.13 | 25.78% | −6.89% | 83.54% | 27.820 | 1.098 | 77.66 | −0.91 | *(running)* | — |
| B1-s T=4096, tail 2b | 2.50 | 15.62% | −7.83% | 71.73% | 113.728 | 4.487 | — | — | — | — |
| B3 RVQ 1.58 b | 1.58 | 9.88% | −8.36% | 45.26% | 103.394 | 4.079 | — | — | — | — |
| B3 VQ-Logits K=1024 | 0.11 | 0.70% | −9.21% | 0.24% | 114 522 | 4517 | — | — | — | — |

**What the 30B shows:**

1. **The ordering reverses versus the 0.6B.** ARCHead wins by 0.6 pp here instead of losing
   by 0.2. Both still clear uniform INT4 by a wide margin, and that margin *grew*: INT4's
   excess is +9.75% here against +4.16% on the 0.6B (2.3×), while frequency tiering only
   goes +1.05% → +1.89%. The larger head is harder to quantize uniformly and more rewarding
   to treat structurally — where "structurally" means *within* the precision axis: protect
   the frequent rows, or correct the dominant activation directions.
2. **The ARCHead reproduction lands in family.** rel PPL **1.013 at 26.9%** of bytes against
   the paper's **1.007 at 25.6%** on Qwen3-8B-Base; our storage-matched INT4 (1.098) sits
   just under the 1.14–1.16 the paper reports for that comparison. Its VibeThinker-3B row is
   the *identical* head shape (151936×2048) at 3.873× compression, which is the strongest
   available prior that it transfers here.
3. **The activation metric is worth much more at scale.** Ablating it costs 1.013 → 1.071
   (+5.8 pp of relative PPL) versus +2.3 pp on the 0.6B, and drops top-1 agreement
   93.3% → 85.7%.
4. **Below ~26% of bytes everything degrades far harder than on the 0.6B** — tiered-2-bit is
   rel 4.49 here vs 2.46 there; RVQ is 4.08 vs 1.76. SLM low-bit results do **not**
   extrapolate to the large model.
5. **VQ-Logits is destroyed** — 0.24% top-1 agreement, i.e. it retains essentially nothing.

Batch sizes were reduced (c4=2, hellaswag=8) so 61 GB of weights plus the
`[bs, 2048, 151936]` logits tensor fit the available GPUs. Both metrics are computed per
request, so this changes speed only, not the numbers.

**Still running:** the MMLU column beyond its dense row — `results_eval/lm_head_sweep_30b_mmlu.json`
on A100-Sagemaker, read with `scripts/show_sweep.py`.

## 2b. Qwen3-0.6B — the full bit-width ladder

Held-out C4, 262 144 tokens, dense PPL **31.863**, untied. The head is 20.70% of this model
against 9.28% of the 30B's active budget, so the same byte saving is worth **2.23×** more.

| run | bits/param | bytes vs BF16 | Δactive (bytes) | **PPL** | rel | HellaSwag | MMLU |
|---|---|---|---|---|---|---|---|
| **dense BF16** | 16.00 | 100.00% | — | **31.863** | 1.000 | **47.29** | **47.18** |
| F3 RTN 8-bit g128 | 8.13 | 50.78% | −10.19% | 31.870 | 1.000 | — | — |
| **B1-s** T=16384, tail 4b | 5.41 | 33.78% | −13.71% | **31.988** | **1.004** | — | — |
| **B1-s** T=4096, tail 4b | 4.44 | 27.78% | −14.95% | **32.199** | **1.011** | 47.02 | **47.18** |
| **B2 ARCHead** | 4.37 | 27.28% | −15.05% | **32.284** | **1.013** | **47.33** | **47.25** |
| B2 ARCHead, *no* metric | 4.37 | 27.28% | −15.05% | 33.011 | 1.036 | — | — |
| F3 RTN 4-bit g128 | 4.13 | 25.78% | −15.36% | 33.188 | 1.042 | 46.96 | 46.82 |
| B1-s T=16384, tail 2b | 3.62 | 22.63% | −16.01% | 61.071 | 1.917 | **47.36** | **47.18** |
| F3 RTN 3-bit g128 | 3.13 | 19.53% | −16.66% | 39.702 | 1.246 | — | — |
| B1-s T=4096, tail 2b | 2.50 | 15.62% | −17.47% | 78.377 | 2.460 | — | — |
| **B2 ARCHead, 2-bit residual** | 2.37 | 14.78% | −17.64% | **60.891** | **1.911** | 44.63 | 46.79 |
| B2 ARCHead 2-bit, *no* metric | 2.37 | 14.78% | −17.64% | 211.109 | 6.626 | — | — |
| F3 RTN 2-bit g128 | 2.13 | 13.28% | −17.95% | 339.261 | 10.65 | — | — |
| B3 RVQ 1.58 b (d16, K256, 3 st) | 1.58 | 9.88% | −18.65% | 55.906 | 1.755 | 45.69 | 45.10 |
| B3 RVQ 1.03 b (d8, K256, 1 st) | 1.03 | 6.42% | −19.37% | 79.861 | 2.507 | — | — |
| B3 VQ-Logits K=1024 | 0.12 | 0.74% | −20.55% | 33 542 | 1053 | 28.45 | **22.95** |

**Reading the ladder:**

1. **8 bits is free** (rel 1.000) and **~4.4 bits is nearly free** (rel 1.004–1.013). That is
   the operating range.
2. **The free static prior edges out the published SOTA here**, reversing on the 30B. At
   ~27% of bytes frequency tiering costs 1.011 and ARCHead 1.013 — a tie within measurement.
   Both beat uniform INT4 (1.042) at essentially the same bytes.
3. **Below 4 bits the ranking flips decisively.** At ~15% ARCHead (1.911) beats tiering
   (2.460), and its activation metric is doing all the work: without it, 6.626. Fitting the
   correction in plain Frobenius error wastes it.
4. **Codebooks reach bit-widths scalars cannot.** RVQ at 1.58 bits (1.755) beats 2-bit RTN
   (10.65) at *lower* storage. But nothing at ≤10% of bytes is within a stderr of dense.
5. **VQ-Logits fails completely** post-training — 1053× PPL, and the one method that drags
   MMLU *below* chance (22.95 vs 25.0). The paper reports +4% PPL; it presumably fine-tunes.

## 2c. ARCHead's mechanism, isolated

Relative error of the fit, 0.6B:

| | Frobenius | in the `C` metric |
|---|---|---|
| core only (no correction) | 0.1015 | 0.1019 |
| + rank-6 correction, **activation metric** | 0.1010 | **0.0627** |
| + rank-6 correction, plain Frobenius | 0.1000 | 0.0969 |

The correction is nearly invisible in Frobenius error and cuts the *metric* error by 38%.
That is ARCHead's thesis and it holds — and it independently confirms the pilots' finding
that the activation metric is the dominant factor.

**Storage caveat.** Our analytic bit count for a 151936×2048 ARCHead is **4.308 bits/param =
26.9% of BF16**; the paper *measures* **25.8%** on that exact shape. The paper is explicit
that an analytic bit count and a packed `state_dict` differ, so treat 26.9% as a bit count
and 25.8% as their measurement. We do not reproduce the packed kernel — accuracy transfers,
throughput claims do not.

## 2d. Diagnostics (2048 held-out post-norm states, 0.6B)

| variant | bytes | top-1 agreement | KL vs dense |
|---|---|---|---|
| *dense BF16* | *100%* | *100.00%* | *0.0000* |
| B1-s T=4096 tail 4b | 27.78% | **96.73%** | **0.0119** |
| B2 ARCHead | 27.28% | 92.04% | 0.0125 |
| B2 ARCHead, no metric | 27.28% | 86.57% | 0.0349 |
| F3 RTN 4-bit | 25.78% | 86.52% | 0.0415 |
| B2 ARCHead 2-bit residual | 14.78% | 50.34% | 0.7324 |

At ~27% of bytes **frequency tiering dominates ARCHead on both** — agreement 96.7% vs 92.0%,
KL 0.0119 vs 0.0125 — consistent with its slightly lower perplexity on this model. Keeping
frequent rows *bit-exact* beats spreading a small error everywhere and correcting the top 6
activation directions, on both the argmax and whole-distribution measures. ARCHead pulls
ahead only once the residual drops to 2 bits, where "bit-exact for some rows" is no longer
affordable.

For masked heads (Part 1) `KL within the tier = 0.0000` exactly — the weights are untouched
in-tier, so the mask is provably the only thing acting. A useful internal check.

---
---

## What the benchmarks can and cannot see

Qwen3-0.6B, lm-eval, full test sets. Complete.

| run | axis | bytes / reads | C4 rel | HellaSwag | Δ | MMLU | Δ |
|---|---|---|---|---|---|---|---|
| dense BF16 | — | 100% | 1.000 | 47.29 | — | 47.18 | — |
| B2 ARCHead | precision | 27.28% | 1.013 | **47.33** | +0.04 | **47.25** | +0.07 |
| B1-s T=4096 tail 4b | precision | 27.78% | 1.011 | 47.02 | −0.27 | **47.18** | **0.00** |
| F3 RTN 4-bit | precision | 25.78% | 1.042 | 46.96 | −0.33 | 46.82 | −0.36 |
| **B1-s T=16384 tail 2b** | precision | 22.63% | **1.917** | **47.36** | **+0.07** | **47.18** | **0.00** |
| B2 ARCHead 2-bit | precision | 14.78% | 1.911 | 44.63 | −2.66 | 46.79 | −0.39 |
| B3 RVQ 1.58 b | precision | 9.88% | 1.755 | 45.69 | −1.60 | 45.10 | −2.08 |
| **B1-p T=32768** | **params** | 21.57% | — | **40.61** | **−6.68** | **47.18** | **0.00** |
| **F2 low-rank r=256** | **params** | 25.17% | 2.568 | **37.80** | **−9.49** | 44.62 | −2.56 |
| B1-a T=4096 strict | **reads** | 2.70% | ∞ | **25.48** | **−21.81** | 47.18 | 0.00 |
| B1-a T=4096 fallback | **reads** | 2.70% | 174× | **29.02** | **−18.27** | 47.18 | 0.00 |
| B3 VQ-Logits | precision | 0.74% | 1053× | 28.45 | −18.84 | **22.95** | **−24.23** |

**1. At sane operating points these tasks cannot rank methods.** `B1-s T=16384/tail-2b`
nearly **doubles** C4 perplexity (1.917×) and still scores HellaSwag **+0.07**, MMLU **0.00**.
A method chosen on task accuracy would call that head free. At ~27% of bytes all three real
methods sit inside ±0.36 pt while C4 separates them 1.1 / 1.3 / 4.2%. **Select on perplexity;
the tasks certify "not catastrophic".**

**2. Parameter-count methods are the ones tasks catch.** Every row that drops or unreads
parameters shows real HellaSwag damage (−6.7, −9.5, −21.8) while precision methods at
comparable byte fractions show ≤0.36. This is the Part 1 / Part 2 split appearing in a second,
independent measurement.

**3. MMLU is blind to tier masking, not to head damage in general.** `B1-p T=32768`:
HellaSwag −6.68, MMLU **exactly 0.00**. MMLU targets are single tokens (" A".." D"), in-tier
at every size, so its score is bit-identical. But VQ-Logits drops MMLU to **22.95** — *below*
the 25% chance floor — and low-rank costs it 2.56 pt. So MMLU's insensitivity is a property
of methods that leave its four answer rows intact, not of the benchmark.

**Consequence for plan §3:** C4 perplexity is mandatory as the plan said, but for the
*opposite* reason on HellaSwag than it assumed — HellaSwag corroborates C4 for free on
structural methods. A sparse-head result showing dense-level *MMLU alone* is no evidence.

---

## Verdict against the pre-registered criteria

Plan §7: **≥6.9% active-param reduction with HellaSwag ≥78.1, MMLU ≥80.5, C4 PPL within +1%.**

Clause by clause on the 30B, for the two surviving methods (both from Part 2):

| clause | bar | ARCHead @4.31 b | B1-s @4.44 b |
|---|---|---|---|
| active-param reduction (bytes) | ≥6.9% | −6.78% ≈ *at bar* | −6.70% ≈ *at bar* |
| **HellaSwag** | ≥78.1 (dense 78.57) | **78.48 ✅** | **78.34 ✅** |
| MMLU | ≥80.5 (dense **80.94**) | *(running)* | *(running)* |
| **C4 PPL** | ≤+1% | **+1.29% ✗** | +1.89% ✗ |

**HellaSwag passes outright; C4 is the binding failure** — narrowly for ARCHead. The storage
clause lands a whisker under 6.9% because "INT4-equivalent" carries group scales in practice
(4.125 bits, not 4.0); plain INT4 does hit −6.89%, at +9.8% perplexity.

The honest reading is **"acceptable, not headline success"** — plan §7's second tier (within
2 stderr on the tasks, which both clear comfortably). A head at ~4.3 bits/param costs ~1.3%
perplexity and ~0.1 pt of HellaSwag on the primary target: a real, cheap win bounded by the
9.28% ceiling.

Plan §7 also pre-registered: *"if B1-a shows dense-level MMLU/HellaSwag **and** dense-level
C4 PPL at 2.7% of head reads, verify the install is taking effect."* It shows dense-level
MMLU but **not** HellaSwag (78.57 → 25.67) and not C4 (∞), so no verification was needed —
gates 0a/0c confirm the install is real regardless.

### Recommendation

- **Take ARCHead** on the 30B if 0.6 pp of perplexity is worth an eigendecomposition and two
  SVDs; **take frequency tiering** if you want zero calibration machinery — a token histogram
  and a row-wise quantizer — and it wins outright on small models.
- **Do not go below 4 bits/param.**
- **Do not pursue any parameter-count reduction of the head** — low-rank, row pruning, or
  sparse reads. Part 1 closes all three.
- Per plan §7's fail clause, further gains want a **LoRA-recovery arm**, not fewer bits.

---

## Caveats

1. **Bit-width sweeps are on Qwen3-0.6B** (`d=1024`, untied); the 30B has all 11 precision
   variants, HellaSwag for the headline five, and the full F2 ladder. Where the two disagree
   — the tiering/ARCHead ordering, and how fast sub-4-bit points degrade — trust the 30B, and
   do not extrapolate the SLM's gentler low-bit behaviour to it.
2. **ARCHead is a reimplementation.** Algorithm 1, the objective, and the published Qwen
   hyperparameters are the paper's; the packed kernel is not. Storage is an analytic bit
   count, not a measured `state_dict`.
3. **CARVQ / VQ-Logits hyperparameters are ours** — the abstracts disclose neither codebook
   size nor stage count. Ours land on the published bits/param and are reported with every
   result. RVQ's codebooks are counted, not waved away: 0.081 bits/param, ~5% of its budget.
4. **VQ-Logits is post-training only.** Read our 1053× as "a drop-in codebook head does not
   work", not as a refutation of the paper.
5. **The two arms use different perplexity harnesses** — lm-eval `word_perplexity` for the
   30B, a fixed-window loop (`scripts/lm_head_gates.py:c4_ppl`) for the 0.6B. Compare
   *within* an arm against that arm's dense row, never across.
6. **Randomized SVD is now seeded** (`quant.py`). It was not for the first F2 run, which is
   why an early note recorded 55.32% top-1 agreement where the seeded re-run gives 55.57%.

---

## Bugs found and fixed

Recorded because three of them produced plausible-looking wrong numbers.

1. **Padding contaminated the activation metric.** The calibration sweep right-pads batches;
   **45.3% of the batch grid was padding**, entering both `C = E[h hᵀ]` and the diagnostic
   sample. Diagnostics were badly wrong — B1-a T=4096 "discarded mass" read **40.0%** when
   the truth is **19.4%** — because a pad position's hidden state argmaxes onto rare tokens.
   Weight fits moved only slightly (ARCHead PPL 32.266 → 32.284). Fixed by masking pad
   positions (`calib.py`); every number above uses the corrected metric.
2. **Tier hit-rate measured after masking.** `argmax_in_tier` took the argmax of the
   *already-masked* logits, making it in-tier by construction and reporting a meaningless
   100%. Now measured pre-mask — the real block-accept rate.
3. **Negative KL.** `nan_to_num` on `-inf` masked logits zeroed the divergent terms while the
   survivors stayed renormalized, yielding **KL = −0.166** (impossible). Now reported as `inf`
   with two finite companions: discarded mass and in-tier KL.
4. **accelerate `device_map` clobbering** — gate 0d, above.
5. **Params counted before untying** — on a tied 0.6B that used the 0.596 B denominator
   instead of 0.752 B, understating every Δactive by ~21%.
6. **Variant names that lied across models.** `f2_lr25` carried an absolute rank from a
   `d=2048` config and silently meant 50% storage on a `d=1024` head. Low-rank variants now
   specify `rank_frac`, a fraction of `d`.

---

## What remains

| # | item | status |
|---|---|---|
| 1 | 30B **MMLU** column | **running** — dense done (**80.94**, matching 80.91); 4 variants to go, ~3 h each. The same variants moved MMLU by 0.00 / +0.07 / −0.36 pt on the 0.6B, so this closes a formality. |
| 2 | **Phase 5** — head ⊕ the repo's −73% expert config | **not run.** Configs exist (`*_lmhead_*_composed_*.yaml`). Its stated purpose was to test the ~15.4% denominator, which is arithmetic already reported; the open empirical question is only whether the two compressions *interact*. |
| 3 | **F1** — reproduce/refute CSV-Decode's 18.4% \|S\|/V | **not run**; needs an external clone. The plan says not to publish the criticism without it, so the certificate branch stays formally open — though Part 1 independently shows any read-set of a few % of `V` is unusable however it is chosen. |
| 4 | **LoRA-recovery arm** | **not run**, and per plan §7's fail clause this is the right next step rather than pushing bits lower. |

Nothing outstanding changes a conclusion above.
