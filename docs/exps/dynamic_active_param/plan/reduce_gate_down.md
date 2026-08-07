# Reducing `gate_proj` + `down_proj` active params per token — implementation plan

**Purpose.** Self-contained porting spec for the best-performing dynamic
active-parameter recipe found in
[`../q3_30b_dynamic_active.md`](../q3_30b_dynamic_active.md) (§ "`oracle_mag`
ablations", lines 436–531): **`oracle_up`** — use the `up_proj` output to decide,
per token, which intermediate channels to activate in `gate_proj` **and**
`down_proj`. Includes the formulation, the measured numbers, exact pointers into
this repo's reference implementation, and a step-by-step port plan.

Read this if you are implementing the method elsewhere. Nothing here requires
reading the rest of the repo.

---

## 1. TL;DR — what to implement

Per token, per MoE layer:

1. Route as usual → top-K experts with normalized weights `g_{t,e}`.
2. For each of the token's K experts, compute `up_proj` at **full width**
   (`I` channels). This is unavoidable — it is the ranking signal.
3. Score every one of the `K·I` pooled channels on **one global scale**:
   `s_{t,e,j} = g_{t,e} · |up_{e,j}(x_t)| · ‖W_down_e[:, j]‖₂`.
4. Keep the **global top-`B`** channels (`B = round((1−prune_ratio)·K·I)`),
   pooled across the token's K experts. No per-expert floor, no per-expert
   quota — quotas emerge; a dominated expert may get 0 channels.
5. Compute `gate_proj` and `down_proj` **only on the kept channels**
   (gather `W_gate` rows / `W_down` columns).

Measured (Qwen3-30B-A3B-Thinking-2507, no fine-tuning, masking simulation):
**−50% of the whole expert FFN active params for −3.3pt HellaSwag acc_norm**
(75.31 vs 78.56 dense) and **−1.1pt MMLU 5-shot** (79.47 vs ≈79.5 dense
— essentially free on MMLU). At −58.3% whole-FFN: 71.30 / 76.43.

The recipe is **deployable**: unlike `oracle_mag` (which needs the SwiGLU
intermediate, i.e. `gate_proj`, before it can decide to skip `gate_proj`), the
`up_proj` output is available before the decision, so the selection sits ahead of
two of the three matrices. Despite the `oracle_` prefix inherited from the
ablation family, `oracle_up` reads nothing unavailable at inference time.

---

## 2. Notation and the block being modified

A SwiGLU MoE block (Qwen2-MoE / Qwen3-MoE / DeepSeek-MoE shape):

- `H` — hidden size (Qwen3-30B-A3B: 2048)
- `I` — per-expert intermediate size, `moe_intermediate_size` (768)
- `E` — number of experts (128), `K` — experts routed per token (8)
- `L` — number of MoE layers (48)
- `x_t ∈ R^H` — one token's block input
- Per expert `e`: `W_gate_e, W_up_e ∈ R^{I×H}`, `W_down_e ∈ R^{H×I}`

Dense forward, for token `t` with routed set `E_t` (|E_t| = K) and normalized
router weights `g_{t,e}`:

```
gate_{e}    = W_gate_e x_t                 ∈ R^I
up_{e}      = W_up_e   x_t                 ∈ R^I
inter_{e,j} = SiLU(gate_{e,j}) · up_{e,j}   (the SwiGLU intermediate)
y_t         = Σ_{e∈E_t} g_{t,e} · W_down_e · inter_e
```

Rewrite the output as a sum over the pooled `K·I` **channels**:

```
y_t = Σ_{e∈E_t} Σ_{j=1}^{I}  g_{t,e} · inter_{e,j} · W_down_e[:, j]
          └────────────── one channel's rank-1 contribution ──────────────┘
```

This is the whole method in one line: each channel contributes an additive
vector to the block output, and we keep the `B` largest contributions per token.

---

## 3. Formulation

### 3.1 The exact criterion (`oracle_mag`) and why top-B is the right rule

Let `S_t ⊆ E_t × [I]` be the kept set, `|S_t| = B`, and `ŷ_t` the output with the
non-kept channels zeroed. Then exactly:

```
‖y_t − ŷ_t‖₂  =  ‖ Σ_{(e,j)∉S_t} g_{t,e} · inter_{e,j} · W_down_e[:,j] ‖₂
              ≤    Σ_{(e,j)∉S_t} g_{t,e} · |inter_{e,j}| · ‖W_down_e[:,j]‖₂
                   └──────────────── s_{t,e,j} ────────────────┘
```

So defining the **per-token channel score**

```
s_{t,e,j} = g_{t,e} · |inter_{e,j}(x_t)| · ‖W_down_e[:, j]‖₂          (oracle_mag)
```

and keeping the global top-`B` by `s` **exactly minimizes this upper bound on
the block-output error**, over all keep-sets of size `B`, jointly across the
token's K experts. Three properties fall out of that, and they are the design:

- **Global, not per-expert.** `g_{t,e}` is a per-expert constant that puts all
  `K·I` channels on one comparable scale. Selection is a single top-`B` over the
  pooled pool — per-expert budgets `k_{t,e} = |S_t ∩ ({e}×[I])|` are an
  *output*, not an input. This is what beats every fixed-quota scheme.
- **Per-token.** `|inter_{e,j}(x_t)|` is the token's own activation. This is
  where the accuracy comes from (see §5): router-only offline methods, which
  know `g` and weight geometry but not `|inter(x)|`, top out ~15pt lower at
  tight budgets.
- **Exact budget.** Every token spends exactly `B`, so the active-param count is
  deterministic — no calibration of a threshold, no variance across tokens.

### 3.2 Two ablations that define the recipe

**Q1 — drop the weight term (`oracle_mag_noW`):** `s = g_{t,e}·|inter_{e,j}|`.
Measured **within 1 stderr** of the full score at both −75% and −87.5%, on both
benchmarks (if anything marginally better). Within an expert the column norms
`‖W_down[:,j]‖` vary far less than the per-token activations and are static, so
they rarely flip a top-`B` decision. **Consequence: the activation magnitude
carries essentially all the signal; the weight geometry is optional.** Drop it
if it simplifies your kernel (it removes a per-expert `(I,)` table).

**Q2 — move the decision before `gate_proj` (`oracle_up`):** replace `|inter|`
with the pre-gate `|up|`:

```
s_{t,e,j} = g_{t,e} · |up_{e,j}(x_t)| · ‖W_down_e[:, j]‖₂               (oracle_up)
```

`up` is computable without `gate`, so the same budget `B` now shrinks **two**
matrices instead of one. Costs 3.0pt (HellaSwag, −75% nominal) / 5.5pt
(−87.5%) / 1.1pt (MMLU, −75%) versus `oracle_mag` at equal nominal `ρ`, while
cutting **twice** the active parameters. At equal *whole-FFN* reduction the trade
is clearly favourable (§4).

Why `|up|` is weaker, and why the gap widens as budget tightens: the SwiGLU
output is `SiLU(gate_j)·up_j`, and the gate is precisely the multiplicative
switch deciding whether the channel is on for this token. A channel with large
`|up_j|` but near-zero `SiLU(gate_j)` wastes budget; at small `B` selection must
be more precise, so the penalty grows (−2.97 → −5.54pt on HellaSwag).

### 3.3 The recommended recipe

**Use `oracle_up`** — it is the only member of the family that reduces
`gate_proj` as well as `down_proj`, and it is realizable at inference time.
Keep the `‖W_down[:,j]‖` factor (it is what was measured; it is one static
`(E, I)` table and free to apply), but per Q1 you may drop it if it costs you
anything structurally — expect no measurable accuracy change.

`up_proj` stays at full width. That is inherent: you cannot rank by `|up|`
without computing `up`. So the floor for this family is `(1 + 2ρ)/3` of the
expert FFN. Going below it requires predicting the ranking rather than computing
it (§10).

---

## 4. Budget accounting — state this explicitly, it is easy to misreport

`prune_ratio` sets `B = round((1 − prune_ratio) · K · I)`, i.e. it measures the
**intermediate dimension** (equivalently `down_proj`'s active columns), written
`ρ = B/(K·I)`. Which *matrices* that budget actually shrinks differs per method:

| Method | ranking signal | `up_proj` | `gate_proj` | `down_proj` | active FFN kept | active FFN cut |
| ------ | -------------- | --------- | ----------- | ----------- | --------------- | -------------- |
| `oracle_mag` / `oracle_mag_noW` | `\|inter\|` (needs gate **and** up) | full | full | ρ | `(2+ρ)/3` | `(1−ρ)/3` |
| **`oracle_up`** | `\|up\|` (pre-gate) | full | ρ | ρ | `(1+2ρ)/3` | `2(1−ρ)/3` |

Concretely:

| nominal cut | ρ | B (of K·I = 6144) | `oracle_mag` whole-FFN cut | `oracle_up` whole-FFN cut |
| ----------- | --- | ----------------- | -------------------------- | ------------------------- |
| −75%   | 0.250 | 1536 | −25.0% | **−50.0%** |
| −87.5% | 0.125 | 768  | −29.2% | **−58.3%** |

**So `oracle_up` and `oracle_mag` rows at the same nominal ρ are not
iso-compute** — `oracle_up` buys 2× the real reduction. Always report the
whole-FFN column alongside the nominal one.

Scaling to whole-model active params (Qwen3-30B-A3B, derived from the config,
recompute for your model): per layer the active expert FFN is
`K·3·H·I = 8·3·2048·768 ≈ 37.7M`, ×48 layers ≈ **1.81B**; attention
`(H·n_h·d_h)·2 + (H·n_kv·d_h)·2 ≈ 18.9M`/layer ≈ 0.91B; router ≈ 0.013B. Expert
FFN is therefore ≈66% of the ≈2.73B active non-embedding params, so a −50%
whole-FFN cut ≈ **−33% of total active non-embedding params** per token.

---

## 5. Measured results (source of truth for the recipe)

Qwen3-30B-A3B-Thinking-2507, K=8, E=128, I=768, 48 MoE layers. Masking
simulation (arithmetically identical to real gathering, §7.1), **no
fine-tuning**, `k_min = 0`, `real_slim: false`. HellaSwag 0-shot, MMLU 5-shot.
Dense: HellaSwag 78.56 acc_norm, MMLU ≈79.5. stderr ≈0.41–0.45pt (HellaSwag
acc_norm), ≈0.32–0.34pt (MMLU acc).

| nominal | Method | whole-FFN cut | HS acc | HS acc_norm | MMLU acc |
| ------- | ------ | ------------- | ------ | ----------- | -------- |
| −75%   | `oracle_mag` (ref)      | −25.0% | 59.71 | 78.28 | 80.53 |
| −75%   | `oracle_mag_noW` (Q1)   | −25.0% | 59.77 | **78.36** | **80.70** |
| −75%   | **`oracle_up` (Q2)**    | **−50.0%** | 57.81 | **75.31** | **79.47** |
| −87.5% | `oracle_mag` (ref)      | −29.2% | 58.40 | 76.84 | _pending_ |
| −87.5% | `oracle_mag_noW` (Q1)   | −29.2% | 58.60 | **77.11** | **79.44** |
| −87.5% | **`oracle_up` (Q2)**    | **−58.3%** | 54.51 | **71.30** | **76.43** |

**Context — why this family is worth porting at all.** At the same *nominal*
budget, the best router-only offline method in the study (Level-1
`pivchol_global`: global `g²·σ` threshold over a pivoted-Cholesky nested order)
reaches 63.60 acc_norm at −75% and 44.15 at −87.5%, and reduce-top-k (route to
fewer full-width experts) reaches 49.4 / 26.2. `oracle_mag` stays at 78.3 / 76.8
— i.e. **the entire remaining headroom is per-token activation information**, not
cross-expert structure or offline statistics. (`pubsub`, a cross-expert method
built on a shared public subspace, matched Level-1 to within 1 stderr at every
budget: cross-expert coupling buys nothing measurable.) `oracle_up` is the
cheapest way to cash in most of that per-token headroom while also cutting
`gate_proj`.

Supporting structure (from the activation-frequency study, same doc): at ρ=0.5
only **0.3%** of channels are kept >95% of the time and **0.4%** <5%; **73.9%**
of keep-frequency variance is *within*-expert. There is no stable sparse
keep-set — the decision genuinely has to be re-made per token, which is why a
static mask cannot substitute. Per-token concentration also varies 2.7–3.2× at
fixed depth, while mean keep-frequency is flat across all 48 layers (0.45–0.49 at
ρ=0.5) — **a single global per-token budget `B`, uniform across layers, is
well-matched; do not bother with a depth schedule.**

---

## 6. Reference implementation in this repo

Package `src/dynamic_active_param/` — pure-PyTorch, no custom kernels, installed
by monkey-patching the MoE block forward. Unit-tested; no model surgery.

| Concern | File / symbol |
| ------- | ------------- |
| Criterion registry (`oracle_mag`, `oracle_mag_noW`, `oracle_up`, `pubsub`) | `src/dynamic_active_param/allocate.py:40` `_CROSS_EXPERT_CRITERIA` |
| Per-token global top-`B` over pooled `K·I` (token-chunked) | `src/dynamic_active_param/allocate.py:43-70` `select_global_topB` |
| **Scoring + keep-mask (the core)** | `src/dynamic_active_param/block.py:51-133` `_cross_expert_keep` |
| — materialize `(T,K,I)` `inter` and (for `oracle_up`) `up` | `block.py:64-84` |
| — `oracle_mag` / `oracle_mag_noW` score | `block.py:86-95` |
| — **`oracle_up` score** | `block.py:97-106` |
| Patched block forward; cross-expert vs router-only branch | `block.py:136-218` `dynamic_moe_block_forward` (cross-expert branch `153-170`) |
| Install onto every MoE block; `B` computation; `k_min` policy | `src/dynamic_active_param/install.py:29-139` (budget `55-62`) |
| Static `‖W_down[:,j]‖` table (`(E,I)`, built inline) | `install.py:113-117` `_dyn_col_norm` |
| Eval-time wiring (config → install → lm-eval) | `src/train/merge_slim_eval.py:115-161` |
| Unit tests (ρ=1 identity, Q1 differs, `oracle_up` matches hand-computed score) | `src/dynamic_active_param/tests/test_level2.py:129-239` |
| Minimal fake MoE block for tests | `src/dynamic_active_param/tests/test_block.py:22-54` `TinyMoEBlock` |
| Configs (4 for `oracle_up`) | `configs/eval/qwen3_30b_a3b_dynamic_oracle_up_{75,875}_{hellaswag,mmlu}.yaml` |
| Sweep orchestrator (8 jobs, waves of 2 × 4 GPUs) | `scripts/run_oracle_q1q2_sweep.sh` |
| Model-family abstraction (`_get_topk`, `_get_moe_intermediate_size`, `_get_experts`, `_get_moe_block`) | `src/base/shared_utils/safe_isinstance.py` |
| Results narrative this plan condenses | `docs/exps/dynamic_active_param/q3_30b_dynamic_active.md:436-531` |

**No offline artifact is needed.** `oracle_mag_noW` uses no offline statistics at
all; `oracle_up` / `oracle_mag` need only `‖W_down[:,j]‖`, computed from the
weights at install time in a few lines. (Contrast the Level-1/`pubsub` paths,
which need a ~57MB pivoted-Cholesky artifact derived from cached activation
covariances — not part of this recipe.)

The reference scoring block, verbatim (`block.py:97-106`):

```python
if self._dyn_criterion == "oracle_up":
    g = routing_weights.to(torch.float32)                  # (T, K)
    score = g.unsqueeze(-1) * up_all.abs().float() * self._dyn_col_norm[selected_experts]
    keep = select_global_topB(score, self._dyn_B)
    return inter_all, keep
```

and the budget (`install.py:55-62`):

```python
B = int(round((1.0 - prune_ratio) * K * I))
cross_expert = criterion in ("oracle_mag", "oracle_mag_noW", "oracle_up", "pubsub")
if not cross_expert:
    B = max(K * k_min, min(B, K * I))
else:
    B = min(B, K * I)          # no k_min floor: quotas emerge from the global top-B
```

---

## 7. Port plan

### Phase 0 — decide what you are measuring

Two distinct deliverables; pick deliberately, they need different code.

- **(A) Accuracy at budget** — a *masking simulation*: compute everything at full
  width, zero the non-kept intermediate before `down_proj`. Exact same numbers as
  a real gathered kernel (proof in §7.1), ~100 lines, no kernel work. This is
  what produced every number in §5.
- **(B) Realized speedup** — a *gathered* implementation that actually skips the
  work. Required to claim latency/FLOPs; **this repo never built it, so there is
  no measured wall-clock number anywhere in the doc.** Do not report a speedup
  you have not measured.

Recommendation: build (A) first, lock the accuracy numbers, then build (B) and
assert bit-comparable outputs against (A).

### 7.1 The equivalence that makes (A) legitimate

For a kept set `S_t`, the gathered computation

```
kept channels J = {j : (e,j) ∈ S_t}
gate_J = W_gate_e[J, :] x_t ;  up_J = up_e[J]
out    = W_down_e[:, J] · (SiLU(gate_J) ⊙ up_J)
```

is **algebraically identical** to computing `inter_e` at full width, zeroing
`j ∉ J`, and applying full `W_down_e` — because `down_proj` is linear in `inter`
and the dropped terms are multiplied by exactly 0. Crucially the selection itself
depends only on `up` and `g`, both computed before any of this, so it is
unaffected. Therefore the masking simulation's accuracy is **exact at budget**,
not an approximation. (`oracle_up`'s `up_proj` is full width in both, so nothing
is skipped there either way.)

Consequence for testing: at `ρ = 1` (`B = K·I`) the dynamic forward must
reproduce the dense forward **bit-for-bit up to fp tolerance** — that is the
first test to port (`test_level2.py:162-175`).

### Phase 1 — masking simulation (accuracy path)

Replace the MoE block forward. Per block, keep these pieces of state:

```
_dyn_col_norm : (E, I) fp32 — ‖W_down_e[:, j]‖₂, built once at install
_dyn_B        : int         — round((1−prune_ratio)·K·I), clamped to ≤ K·I
_dyn_I        : int         — moe_intermediate_size
```

Forward (mirrors `block.py:136-218`):

```python
def moe_forward(self, hidden_states):                 # (Bsz, S, H)
    B_, S, H = hidden_states.shape
    hs = hidden_states.view(-1, H)                    # (T, H), T = Bsz*S
    router_logits = self.gate(hs)

    # --- routing: byte-for-byte the upstream path, do not "improve" it ---
    rw = F.softmax(router_logits, dim=1, dtype=torch.float)
    rw, sel = torch.topk(rw, self.top_k, dim=-1)      # (T,K), (T,K)
    if self.norm_topk_prob:
        rw /= rw.sum(dim=-1, keepdim=True)
    rw = rw.to(hs.dtype)

    T, K, I = sel.shape[0], self.top_k, self._dyn_I

    # --- pass 1: materialize each token's K intermediates and up outputs ---
    inter_all = torch.zeros((T, K, I), dtype=hs.dtype, device=hs.device)
    up_all    = torch.zeros((T, K, I), dtype=hs.dtype, device=hs.device)
    expert_mask = F.one_hot(sel, num_classes=self.num_experts).permute(2, 1, 0)
    for eid in torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero():
        eid = int(eid)
        ex = self.experts[eid]
        slot, tok = torch.where(expert_mask[eid].squeeze(0))   # slot∈[0,K), token id
        cur  = hs[tok]
        up   = ex.up_proj(cur)
        gate = ex.gate_proj(cur)
        inter_all[tok, slot] = (ex.act_fn(gate) * up).to(hs.dtype)
        up_all[tok, slot]    = up.to(hs.dtype)

    # --- score + global top-B (fp32) ---
    g     = rw.to(torch.float32)                                     # (T,K)
    score = g.unsqueeze(-1) * up_all.abs().float() * self._dyn_col_norm[sel]
    keep  = select_global_topB(score, self._dyn_B)                   # (T,K,I) bool

    # --- pass 2: mask, then down_proj + router-weight scatter ---
    final = torch.zeros((T, H), dtype=hs.dtype, device=hs.device)
    for eid in torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero():
        eid = int(eid)
        slot, tok = torch.where(expert_mask[eid].squeeze(0))
        inter = inter_all[tok, slot] * keep[tok, slot].to(hs.dtype)
        out   = self.experts[eid].down_proj(inter) * rw[tok, slot, None]
        final.index_add_(0, tok, out.to(hs.dtype))

    # shared expert (Qwen2-MoE / DeepSeek), if present — NOT budget-pruned
    if getattr(self, "shared_expert", None) is not None:
        se = self.shared_expert(hs)
        final = final + F.sigmoid(self.shared_expert_gate(hs)) * se

    return final.view(B_, S, H), router_logits
```

with (`allocate.py:43-70`):

```python
def select_global_topB(score, B, chunk=4096):
    T, K, I = score.shape
    if B > K * I: raise ValueError("infeasible budget")
    keep = torch.zeros((T, K, I), dtype=torch.bool, device=score.device)
    for a in range(0, T, chunk):                      # bound the (chunk, K*I) topk
        b = min(a + chunk, T)
        flat = score[a:b].reshape(b - a, K * I)
        idx  = torch.topk(flat, B, dim=1, sorted=False).indices
        m    = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, idx, True)
        keep[a:b] = m.reshape(b - a, K, I)
    return keep
```

Notes that matter:

- The two-pass structure exists **because selection is global across experts**:
  you cannot finish an expert until every co-activated expert's `up` is known.
  That is the one real structural difference from a stock MoE forward.
- Score in **fp32** even in a bf16 model. bf16 has ~8 mantissa bits; a top-`B`
  over 6144 values with heavy ties near the threshold is exactly where that
  bites.
- `topk(..., sorted=False)` — you need membership, not order.
- The `+inf` / `−inf` sentinel convention (`allocate.py:48-50`) lets you
  force-keep or forbid channels; unused by `oracle_up`, useful if you add e.g. a
  per-expert floor.

### Phase 2 — install / wiring

Mirror `install.py:29-139`:

1. Walk layers in order; skip non-MoE layers (dense-first-`k` architectures are
   common); for each MoE block get its expert list.
2. `col_norm = torch.stack([e.down_proj.weight.detach().float().norm(dim=0) for e in experts])`
   → `(E, I)`. Move it to **that block's own device** — this is what makes the
   whole thing work unchanged under `device_map='auto'` sharding.
3. Attach `_dyn_col_norm`, `_dyn_B`, `_dyn_I`, and bind the forward
   (`types.MethodType`).
4. `B = min(round((1−prune_ratio)·K·I), K·I)`; **no `k_min` floor** (`k_min: 0`)
   — the emerging-quota behaviour is load-bearing.

If your codebase uses fused/grouped expert weights (one `(E, I, H)` tensor rather
than `nn.ModuleList`), the col-norm becomes `W_down.norm(dim=1)` over the right
axis and Phase 1's per-expert loop becomes a grouped GEMM — the scoring and
selection code is unchanged.

### Phase 3 — gathered kernel (realized savings)

Only needed for deliverable (B). What changes: `select_global_topB` should return
**indices**, not a `(T,K,I)` mask, and the per-token kept sets are **ragged**
(variable count per `(token, expert)`), which is the entire difficulty.

- **Decode (batch 1, one token):** easiest and highest-value case. `T=1`, so you
  have one index list of length `B` over `K·I`. Gather `W_gate_e[J,:]` rows and
  `W_down_e[:,J]` columns and run narrow GEMVs. Decode is memory-bandwidth bound,
  so gathering rows/cols cuts the bytes moved roughly in proportion to `ρ` —
  this is where the win is real.
- **Prefill / batched:** ragged widths per token fight tensor cores. Options, in
  increasing order of accuracy cost: (i) keep the masking form for prefill and
  gather only in decode; (ii) round each token's per-expert count up to a
  multiple of 8/16 and pad (small budget overshoot — report it); (iii) convert
  the global top-`B` into per-expert quotas `k_{t,e}` once, then take each
  expert's own top-`k_{t,e}` (identical selection, easier to block).
- **Scoring overhead** is negligible and worth stating: `K·I = 6144` multiplies
  plus one top-`B` per token per layer, against `3·K·I·H ≈ 37.7M` MACs of expert
  FFN work. The `up_proj` you must compute anyway.
- Memory: the `(T,K,I)` buffers are ~50MB each in bf16 at `T=4096, K=8, I=768`,
  and the fp32 score ~100MB. Chunk over tokens (the reference chunks at 4096) or
  you will OOM on long prefills.

### Phase 4 — validation (port these tests)

From `tests/test_level2.py` / `tests/test_block.py`, against a tiny fake block
(`TinyMoEBlock`, `test_block.py:22-54` — 3 `nn.Linear`s per expert, upstream-shaped
reference forward):

1. **ρ=1 identity** (`test_level2.py:162-175`): `B = K·I` ⇒ output matches the
   dense reference within `atol=1e-5`. Catches routing-path drift, dtype bugs,
   scatter/index errors. Non-negotiable.
2. **Exact budget** (`test_block.py:85-115` pattern): every token's keep-mask has
   exactly `B` `True` entries: `keep.reshape(T,-1).sum(1) == B`.
3. **Score fidelity** (`test_level2.py:210-239`): recompute
   `g·|up|·‖W_down[:,j]‖` in a naive double loop from `up_proj` outputs, take
   top-`B`, assert the keep-mask is **exactly equal**. This is the test that
   pins "ranked by `up_proj` output, not by the SwiGLU intermediate".
4. **Q1/Q2 are actually different** (`test_level2.py:178-207`): with strongly
   non-uniform `‖W_down[:,j]‖`, dropping the factor must change the selection —
   guards against silently ignoring `_dyn_col_norm`.
5. **Gathered ≡ masked** (new, for Phase 3): same inputs, same seed, assert the
   gathered kernel's output matches the masking simulation.
6. **Sanity on the real model**: at `prune_ratio` ≈ 0 accuracy must equal dense;
   monotone degradation as `ρ` falls; a −75% run should land near 75.3 HellaSwag
   acc_norm on the same base model, else the port differs.

---

## 8. Config surface

Four knobs. Reference YAML (`configs/eval/qwen3_30b_a3b_dynamic_oracle_up_75_hellaswag.yaml`):

```yaml
prune_kwargs:
  prune_ratio: 0.75          # ρ = 0.25 ⇒ B = 1536 of K·I = 6144
  dynamic_alloc:
    enabled: true
    criterion: "oracle_up"   # oracle_up | oracle_mag | oracle_mag_noW
    k_min: 0                 # no per-expert floor — quotas emerge

real_slim: false             # masking simulation; weights untouched
shrink_gate: false
dtype: "bf16"
attn_implementation: "sdpa"
model_name_or_path: "Qwen/Qwen3-30B-A3B-Thinking-2507"
eval_task_names: "hellaswag" # num_fewshot: 0 ; mmlu uses num_fewshot: 5
```

`scores_dir` is present in these configs but **unused** by `oracle_up` (no
offline artifact) — an artifact of the shared config dataclass. Don't port that
dependency.

---

## 9. Pitfalls

- **Report the whole-FFN cut, not the nominal one.** `oracle_up` at "−75%" is
  −50% of the expert FFN. Comparing it to an `oracle_mag` row at the same nominal
  ρ is a 2× iso-compute error in `oracle_up`'s favour, and the doc's numbers only
  make sense with §4's table alongside.
- **Do not add a `k_min` floor.** Reserving channels for a token's weakest
  experts is exactly the failure mode of the earlier per-expert-quota criteria
  (`router_prob × activation` collapses to 43.66 acc_norm at −75%, vs 63.60 for a
  global threshold). Let dominated experts get 0.
- **Selection must be global across the K experts on one scale.** Per-expert
  top-`k` with a fixed `k` is a different, much worse method.
- **Keep the routing path byte-identical to upstream** — softmax in fp32, top-k,
  `norm_topk_prob` renormalization, cast to input dtype, and the same
  `g`-scaling of expert outputs. `g` enters both the score and the output; a
  mismatch silently changes selection.
- **Shared experts / dense layers are not pruned** by this scheme (`block.py:211-215`).
  Decide explicitly and note it in your accounting.
- **Dead experts exist.** 33 of 48 layers had ≥1 expert that never fired over
  69k calibration tokens. Code must tolerate zero-token experts (the
  `expert_hit` nonzero-mask loop does).
- **Score in fp32**, and chunk tokens in the top-`B`.
- **No fine-tuning was used** for any number in §5, and no LoRA recovery was
  run for this family. Fine-tuning is unexplored upside, not a claimed result.
- **`oracle_up`'s `up_proj` stays full width.** If your accounting assumes all
  three matrices shrink, you will overstate the saving by 3×.

---

## 10. Where the remaining headroom is

The doc's conclusion, worth carrying over so the port isn't optimized in the
wrong direction:

- **Q1 says the target is the activation magnitude alone.** A future online
  predictor needs to predict only `|inter_{e,j}(x)|` (or its rank) — it can
  ignore `W_down` geometry entirely.
- **Q2 says a pre-gate proxy already works well enough to double the realized
  saving**, at a 3pt cost — but a raw `|up_j|` proxy leaves ~3–6pt on the table.
  The headroom is a cheap predictor of `SiLU(gate_j)·up_j` (not of `up_j`) that
  is computable before `gate_proj` — e.g. a low-rank/quantized sketch of
  `W_gate` giving an approximate gate sign/magnitude, used only for ranking. That
  would push toward `oracle_mag`'s accuracy (78.3 at −75% nominal) at
  `oracle_up`'s cost structure.
- **Offline cross-expert statistics are a dead end** — `pubsub` matched Level-1
  to within 1 stderr at every budget, despite cross-expert coupling holding ~70%
  of covariance energy. It changes <2% of selections. Don't spend effort there.
- **A single per-token budget `B`, uniform across layers, is already
  well-matched** to depth. If you do experiment with a schedule, early/late
  layers (peakier, 2.7–3.2× more heterogeneous per token) are where a per-token
  *adaptive* budget would pay off, not mid layers.
