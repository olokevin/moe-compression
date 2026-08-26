# `input_sparse` — Port Reference

> Self-contained spec for reimplementing the **`input_sparse`** channel scorer in another
> codebase. Distilled from `docs/exps/dynamic_active_param/efficient_scorer.md` (the results
> report) with pointers into this repo's implementation. Read this to build it; read the report
> for the evidence behind every design choice.

## Scope — one method, two ways to set its two sparsities

`input_sparse` (code name `sparse_probe`) is a **cross-expert, per-token channel selector** for
a top-k MoE FFN. It has exactly **two knobs, both *keep* fractions**:

| symbol | what it keeps | paid to |
|---|---|---|
| **`rho_input`** | fraction of the token's input coordinates read | **scoring** |
| **`rho_channel`** (`= 1 - prune_ratio`) | fraction of the pooled `K·I` channels kept | **compute** |

The two things to port are the same scorer with two ways of *choosing* that pair:

1. **Basic — hand-set both.** You pick `rho_input` and `rho_channel` directly. Simplest; the
   knobs move the split *and* the total budget together.
2. **Best practice — solve the split for a unified budget.** You pick a single budget `C`
   (= used-parameter fraction) and *solve* for the `(rho_input*, rho_channel*)` that minimizes
   error at that cost. This is the derived Lagrange solve in §4, plus the free `router`
   cross-expert allocation.

Both are one method; §3 is the shared algorithm, §4 and §5 are the two ways to configure it.
`input_sparse` needs **no calibration/scoring pass and no saved artifact** — everything is
derived from the served expert weights at install time (`install_dynamic_alloc(..., artifact=None, ...)`,
`src/train/merge_slim_eval.py:307`).

Headline (Qwen3-30B-A3B, masking sim, no finetune; dense = **78.56** HellaSwag acc_norm 0-shot
/ **80.91** MMLU 5-shot): **−73.3% used params → 74.64 HS / 77.67 MMLU**, descending smoothly to
−80.0% → 72.55 / 76.11 with no cliff.

---

## 1. The setting and the goal

A SwiGLU MoE expert computes `down( SiLU(gate·x) ⊙ (up·x) )` over an intermediate of `I`
channels. Top-k routing sends each token to `K` experts, so the activated FFN parameters per
token are `K` experts × 3 matrices `(I,H)`.

The idea: **per token, keep only a `rho_channel` fraction of the `K·I` activated channels**,
pooled across the token's K experts on one global scale, and gather all three matrices to just
those channels. The hard part is *deciding which channels* cheaply. `oracle_mag` ranks by the
true SwiGLU intermediate `g_e·|SiLU(gate·x)⊙(up·x)|` — the best selector, but it must run `gate`
and `up` at **full width** just to decide, so its realized whole-FFN cut floors at
`(1+1+rho_channel)/3 ≥ 2/3` (only −29% at `rho_channel=0.125`).

`input_sparse` produces the *same ranking signal* from a proxy cheap enough that the decision
precedes every full-width matmul: **read the served `up`/`gate` weights on only the token's
top-`rho_input` input coordinates by `|x|`.** The proxy *is* the served weight (a view, not a
copy — a stacked fp16 copy of up+gate is ~39 GB on this model), so the only error is input
sparsity and there is **zero extra storage**.

---

## 2. Used-parameter accounting (this *is* the unified budget)

Units of one expert `(I,H)` matrix; a dense expert FFN is 3. `n = 2` (score both up and gate).

```
scoring  = n · rho_input           (both branches, all I rows, rho_input of the H columns)
compute  = 3 · rho_channel         (up, gate, down gathered to the kept channels)
used  =  rho_channel + n · rho_input / 3   =   rho_channel + 2·rho_input/3    ≡  C
used_param_cut = 1 - C
```

Reference impl: `sparse_probe.used_param_fraction` (`src/dynamic_active_param/sparse_probe.py:400`).

The accounting is deliberately **conservative**: scoring and compute overlap on the kept rows,
and this bills that overlap twice. The consequence that drives everything else: **the scoring
term carries a 3× discount that the compute term does not** — a unit of `rho_input` costs only
**two-thirds** of a unit of `rho_channel`. So cutting `rho_input` is the cheap axis, cutting
`rho_channel` is expensive, and `rho_input=1` is hopeless regardless of `rho_channel` (scoring
alone is then 0.667). This asymmetry is exactly what §4's solve exploits.

### The masking-simulation caveat — READ BEFORE PORTING

This repo evaluates via a **masking simulation**: it computes the full-width intermediate,
zeros the non-kept channels, then runs `down_proj` at full width. The *arithmetic* is identical
to a real gather-then-compute; only the *accounting* differs. See `dynamic_moe_block_forward`
(`src/dynamic_active_param/block.py:311`) — it never actually skips a matmul.

**A real deployment must do the opposite:** run only the cheap `input_sparse` scoring pass, take
the global top-B, then **gather** `up`/`gate`/`down` to those B channels and compute at reduced
width. That is where the parameter/wall-clock savings the report *claims* actually come from.
The scoring math and the selection math port verbatim; the FFN compute is the part you replace
with a real gather. The selector's job ends at "here is the `(T,K,I)` boolean keep-mask."

---

## 3. The algorithm (per MoE block, per forward)

Inputs: `x (T,H)` flattened tokens; router gives `g (T,K)` weights and `sel (T,K)` expert ids
(standard softmax-topk, `norm_topk_prob` as upstream). Fixed budget `B = round(rho_channel·K·I)`.

1. **Sort once per token.** `sorted_abs, ranks = descending_abs_ranks(x)` — the descending `|x|`
   order, shared by all K experts and both branches (`sparse_probe.py:171`).
2. **Allocate coordinate reads across the token's K experts** — `allocate_input_reads`
   (`sparse_probe.py:186`):
   - `uniform` (β=0): every expert reads the same `round(rho_input·H)` coordinates.
   - `router` (β=1, **best practice**): rank `(slot, coord)` pairs by `g_e·|x_i|`, keep the
     pooled top-`K·round(rho_input·H)`. Because the score factors as `g_e^β · |x_i|`, each slot
     keeps a **prefix** of the one shared `|x|` order — so this is a single bisection on a
     threshold τ plus a tiny top-up, *not* a per-expert sort. Returns read counts `n_e (T,K)`.
3. **Score each expert's channels** on its sparse input — `probe_expert_scores`
   (`sparse_probe.py:350`). With `x_sp` = `x` zeroed outside the kept coordinates:
   ```
   up_hat   = x_sp @ W_up[e].T
   gate_hat = x_sp @ W_gate[e].T
   score_e  = |SiLU(gate_hat) ⊙ up_hat|          # (T, I);  up-only: |up_hat|
   ```
4. **Pool and select.** `score = g[:,:,None] · score_all  (T,K,I)`; keep the global top-B per
   token — `select_global_topB` (`allocate.py:55`), a topk over the flattened `K·I` axis.
5. **Gather + compute** the real FFN on the B kept channels. (Repo: mask + full `down_proj`,
   `block.py:337–345` — replace with a real gather in a deployment.)

---

## 4. Best practice — solving the split for a unified budget

This is the derived allocation the user asked to port. **Do not hand-pick `rho_input`** at deep
budgets: the historically hand-picked `rho_input=0.25` was never optimal below −73%.

### The derivation

Given a fixed budget `C = rho_channel + 2·rho_input/3`, choose the split that minimizes the
block-output error. One Lagrange multiplier gives the stationarity condition:

```
minimize  rel_err(rho_input, rho_channel)   s.t.   rho_channel + 2·rho_input/3 = C
   =>     (3/2) · ∂rel_err/∂rho_input   ==   ∂rel_err/∂rho_channel
```

The `3/2` is exactly the discount `rho_input` carries in the cost (§2). This is a **selector**
change (one global pair for all layers), which is the regime where the offline rel_err ladder is
validated — unlike a *cross-layer* budget schedule, which is a **closed negative result** (do not
attempt it: unweighted −0.16pt, slope-weighted −3.29pt, uniform beats both).

### How to run it

`rel_err(rho_input, rho_channel)` is read off a cached, per-model offline surface (bilinear in
`(p, rho)`), measured once with no GPU cost at eval time:

```bash
python scripts/probe_split_solve.py --budgets <C>     # seconds, no GPU
```

The surface itself is built by `scripts/probe_layer_surface.py` (this repo caches it as
`layer_surface_8k.json`, 8192 C4 tokens). **In a new codebase you must measure your own surface**
(the split is model-specific) — or use the shortcut below.

**Practical shortcut (no surface needed):** the solved `rho_input*` sits at **≈0.1875 across the
whole deep-budget range**, i.e. a roughly **50/50 scoring/compute split**. So absent a measured
surface, split a budget `C` as `rho_input ≈ 0.75·C/2` and `rho_channel ≈ C/2` and you land on the
optimum to within a grid step.

### Solved optima (measured end-to-end on both metrics)

| target cut | budget `C` | `rho_input*` | `rho_channel*` | B (of K·I=6144) | reads/expert | HellaSwag | MMLU |
|---|---|---|---|---|---|---|---|
| −63.3% | 0.3667 | 0.2500 | 0.2000 | 1229 | 512 | 76.61 | 79.45 |
| −68.3% | 0.3167 | 0.2400 | 0.1567 | 963 | 492 | 75.78 | 78.98 |
| −73.3% | 0.2667 | 0.1875 | 0.1417 | 871 | 384 | 74.63 | 77.94 |
| −75.0% | 0.2500 | 0.1875 | 0.1250 | 768 | 384 | 74.08 | 77.77 |
| −77.5% | 0.2250 | 0.1875 | 0.1000 | 614 | 384 | 73.33 | 76.81 |
| −80.0% | 0.2000 | 0.1575 | 0.0950 | 584 | 323 | 72.55 | 76.11 |

**How much solving is worth is depth-dependent** (five iso-cost tests): a **wash** down to −75%
(−0.01 HS / +0.27 MMLU at −73.3%; +0.28 / +0.44 at −75%), and a **real +1.14pt** by −80%, where
there is no slack to absorb budget spent on the wrong axis. The solve costs seconds offline, so
just always run it. Caveat: `0.1875` is the best *grid point* (neighbours 0.125 / 0.25), not a
continuum optimum.

### The best-practice config block

`configs/eval/qwen3_30b_a3b_probe_bp_cut750_hellaswag.yaml` (−75.0%, solved split, router alloc):

```yaml
prune_kwargs:
  prune_ratio: 0.87500            # = 1 - rho_channel (MUST match rho_channel)
  dynamic_alloc:
    enabled: true
    criterion: "sparse_probe"     # == input_sparse (historical code name)
    k_min: 0                      # no per-expert floor; the top-B is global by design
    probe:
      bits: 16                    # serving precision -> probe ALIASES the served weight
      group: 128                  # ignored at bits>=16
      rho_input: 0.1875           # SCORING budget  (solved for this C)
      rho_channel: 0.12500        # COMPUTE budget  (solved for this C)
      use_gate: true              # score BOTH up and gate
      lam: 1.0                    # no candidate cascade
      input_alloc: "router"       # split coordinate reads across experts by g_e·|x_i|
```

Single best operating point to quote: **−73.3%, `rho_input=0.25, rho_channel=0.10`, router →
74.64 HS / 77.67 MMLU** (best raw accuracy); the solved `0.1875/0.1417` at the same −73.3% ties
on HS and is +0.27 on MMLU.

---

## 5. Basic version — hand-set both sparsities

Identical scorer; you just write `rho_input` and `rho_channel` directly instead of solving them.
Because the two knobs move the split *and* the budget together, this is fine for exploration and
for budgets ≥ −75% (where the split is a flat direction), but leaves ~1pt on the table by −80%.
Measured sweep at the historically hand-picked `rho_input=0.25`, `uniform` allocation:

| `rho_channel` | used-param cut | HellaSwag | MMLU |
|---|---|---|---|
| 0.200 | −63.3% | 76.72 | — |
| 0.150 | −68.3% | 76.47 | 78.63 |
| 0.100 | −73.3% | 74.06 | 77.20 |
| 0.100 + `router` | −73.3% | 74.64 | 77.67 |

Config: same block as §4, with your chosen `rho_input`/`rho_channel` and (optionally)
`input_alloc: "uniform"` for the plainest form.

---

## 6. Knob settings and why (get these wrong and the method breaks)

| knob | set to | why |
|---|---|---|
| `bits` | **16** | Probe aliases the served weight → **zero extra storage** (measured 0.00 MB). A separate 3-bit copy is dominated on *both* axes (+13% storage, worse ranking). **⚠ the code default is 3, not 16** — omitting `bits` silently gives the worse variant. |
| `use_gate` | **true** | Score both branches (`g_e·|SiLU(gate)⊙up|`). Up-only is `oracle_up`: −3.34pt at comparable depth. The gate is load-bearing here. |
| `input_alloc` | **`router`** | Split coordinate reads across a token's K experts by `g_e·|x_i|`. **+0.58pt HS / +0.47pt MMLU** at identical cost, won all 16 offline layer×budget cells. Free (one bisection over a sort the layer already computes). **Do NOT** use `router2` (g² overshoots) or `colnorm` (a wash, column-norm CV=0.022). |
| `lam` | **1.0** | No relaxed-candidate cascade; extra exact reads buy less than spending them on `rho_input`. |
| `k_min` | **0** | The pooled top-B is global; a dominated expert may legitimately get 0 channels. (Code default is 16 — override it.) |
| `rho_channel` | `1 - prune_ratio` | Redundant with `prune_ratio` and **not** cross-checked at load; a mismatch silently changes B. `probe.rho_channel` wins if both set (`merge_slim_eval.py:243`). |
| per-layer schedule | **do not use** | Cross-layer budget reallocation is a **closed negative result**. One global `(rho_input, rho_channel)` for all layers. |

Note two allocation concepts, both live in best practice and both distinct:
- **budget split** (§4) — how a unified budget `C` is divided between `rho_input` and `rho_channel`.
- **`input_alloc` = router** — how, within a token, the coordinate-read budget is divided across
  its K experts. Orthogonal to the split; keep both on.

---

## 7. Porting guardrails (unit-test these in the new codebase)

Mirrors `src/dynamic_active_param/tests/test_sparse_probe.py`:

- **`input_sparse` at `bits=16, rho_input=1.0, uniform` ≡ `oracle_mag_noW`** (the exact
  full-width magnitude selector) — `test_probe_at_full_precision_is_oracle_mag_noW`,
  `test_reuse_probe_dense_input_is_oracle_mag_noW`.
- **The probe is a view, not a copy** — `data_ptr` equality with the served weights
  (`test_reuse_probe_aliases_served_weights_without_copying`); materializing a stacked copy OOMs.
- **`router` allocation conserves the pooled budget** and matches an explicit pooled top-N
  (`test_alloc_matches_bruteforce_pooled_topk`, `test_alloc_conserves_the_pooled_budget`);
  `uniform` gives every expert the same count
  (`test_alloc_uniform_gives_every_expert_the_same_count`).
- **Accounting closed form** — `used = rho_channel + 2·rho_input/3`
  (`test_used_param_fraction_closed_form`).

## 8. Verifying a real run

Confirm the realized budget from the log, not the YAML (`print_probe_accounting`,
`sparse_probe.py:512`). A correct −75.0% run prints:

```
[input_sparse] rho_input=0.1875 rho_channel=0.125 (scoring 0.1250 + compute 0.1250)
  -> USED PARAMS=0.2500, cut 75.0%; ... extra proxy storage = 0.0% of expert weights
[DynamicAlloc] Installing: ... prune_ratio=0.875, B=768 (of K*I=6144)
[DynamicAlloc] ✅ Installed dynamic forward on 48 MoE blocks
```

Check `USED PARAMS`, `extra proxy storage = 0.0%`, and that `B` is what you intended.

---

## 9. Pointer index

**Core implementation** (`src/dynamic_active_param/`)

| file | what to read |
|---|---|
| `sparse_probe.py` | `SparseProbe`, `build_layer_probe` (:288), `probe_expert_scores` (:350), `sparsify_input_topk` (:145), `descending_abs_ranks` (:171), `allocate_input_reads` (router split, :186), `sparsify_input_by_count` (:374), accounting `used_param_fraction` (:400) / `report_probe_accounting` (:448) / `print_probe_accounting` (:512). Module docstring is the concept spec. |
| `block.py` | `dynamic_moe_block_forward` (:311) and `_cross_expert_keep` (:71) — the per-token forward (the `need_probe` branch) that applies the keep-mask (**masking sim; replace with a real gather**). |
| `allocate.py` | `select_global_topB` (:55) — global top-B over `(T,K,I)`; `_CROSS_EXPERT_CRITERIA` (:49). |
| `install.py` | `install_dynamic_alloc` (:42) — binds the forward onto every MoE block, computes `B`, builds per-layer probes (`sparse_probe` branch :217–234). Called with `artifact=None` for this criterion. |
| `tests/test_sparse_probe.py` | the guardrails in §7. |

**Wiring / configs / scripts**

- Entry point + config→kwargs: `src/train/merge_slim_eval.py:230–262`.
- Best-practice configs: `configs/eval/qwen3_30b_a3b_probe_bp_cut{633,683,733,750,775,800}_*.yaml`
  (the solved curve), `..._probe_router_k25_r10_*.yaml` (−73.3% best raw accuracy).
- **Budget-split solve: `scripts/probe_split_solve.py`** (§4); rel_err surface builder:
  `scripts/probe_layer_surface.py`.
- Offline rel_err screen before spending a ~10h eval (4 min/GPU): `scripts/probe_output_error.py`.
  Validated ladder slope **−26.4 pt/unit** (R²=0.985) for changes to *which channels a layer
  selects*; **invalid** for cross-layer budget moves.

**Source report:** `docs/exps/dynamic_active_param/efficient_scorer.md` — the leaderboard, the
`rho_input`/`rho_channel` trade, the split-solve derivation and iso-cost tests, the `router`
allocation study, and the closed cross-layer-schedule negative result.

## 10. Minimal port plan (recommended order)

1. **Basic `input_sparse`, uniform, bits=16** — §3 steps 1–5 with `sparsify_input_topk` +
   `probe_expert_scores` + `select_global_topB`, hand-set `(rho_input, rho_channel)`. Pin the
   `rho_input=1.0 ≡ oracle_mag_noW` invariant. ~80 lines; this is the whole basic method.
2. **Add `router` allocation** — `descending_abs_ranks` + `allocate_input_reads` +
   `sparsify_input_by_count`. Pin the pooled-budget test. +0.58pt, free.
3. **Add the budget-split solve** (§4) — either port `probe_split_solve.py` against a rel_err
   surface you measure on your model, or use the ≈50/50 shortcut (`rho_input ≈ 0.75·C/2`).
4. **Replace the masking sim with a real gather** in your FFN — the selector already yields the
   `(T,K,I)` keep-mask; gather up/gate rows and down columns to the B kept channels and compute
   at reduced width. This is where the claimed savings become real.
