# Midpoint presentation figures

Every figure exists as both `.pdf` (vector — use this in the deck) and `.png`
(preview). Regenerate with:

```bash
# schematic figures — no model needed, instant
python scripts/presentation_illustrations.py

# measured figures — needs the capture .npz + the leverage scores (see below)
python scripts/presentation_plot.py

# motivation single-expert figures — needs expert_activation.npz
python scripts/expert_activation_plot.py

# motivation sweep figures — no model needed, numbers hard-coded from result docs
python scripts/presentation_motivation_plot.py
```

Numbers quoted in the slide text are all in `stats.json`.

## Motivation figures (`docs/presentation/motivation.md`)

Four figures back the motivation slides:

### `fig_sparse_suffices` — motivation slide 1

Single panel (linear y-axis): histogram of one expert's SwiGLU output
`h_j = SiLU(gate_j·x)·(up_j·x)` over 8,000 WikiText-2 tokens (layer 0, expert 0),
with the bottom 80% by `|h|` shaded grey (deactivated) and the large-magnitude tail
blue (activated). 43% of activations are ~0. From `scripts/expert_activation_plot.py`;
numbers in `stats_activation.json`.

### `fig_token_specific` — motivation slide 2

Single panel (y-axis a 0–100% keep-ratio): per-neuron keep-frequency for **one**
expert, neurons in natural index order (unsorted). Each token keeps its own top-25%
channels (ρ = 0.25); almost every neuron hovers around ρ, not a step — 0% of
channels are kept >95% of the time, 0.1% <5%. Establishes: no fixed within-expert
keep-set works.

### `fig_prune_sweep_mmlu` / `fig_prune_sweep_hellaswag` — motivation slide 3

MMLU (5-shot acc) and HellaSwag (0-shot acc_norm), **one metric per figure**, vs
nominal reduction (50–90%) of the active intermediate dimension, per-token selection
on the **dense** Qwen3-30B-A3B (`reduce = gate+down`, no fine-tuning). y-axis starts
at 50; each has its dashed dense reference. Source: `test/results/mobe/mobe_30b.md`
§"Results — dense Qwen3-30B-A3B, prune-ratio sweep". **Caveat:** the dense HellaSwag
dashed line uses the arc-era baseline (0.6971) because the base-A3B HellaSwag eval
crashed — it is a placeholder, not a measured dense HellaSwag.

### `fig_offline_vs_online` — motivation slide 4

HellaSwag 0-shot acc_norm vs active-param reduction (50/62.5/75/87.5%), on
`Qwen3-30B-A3B-Thinking-2507` (dense = 78.56). Two black offline curves
(reduce-top-k dashed, Level-1 pivoted-Cholesky solid) vs one blue online curve
(per-token `oracle_mag`). Source: `docs/exps/dynamic_active_param/q3_30b_dynamic_active.md`
and `docs/report/level2.md`. This figure and the two prune-sweep figures above are
all from `scripts/presentation_motivation_plot.py` (numbers hard-coded from the
result docs).

## Measured figures

Data comes from three captures:

| Artifact | Produced by | Contents |
|---|---|---|
| `docs/results/presentation/pres_exps.npz` | `scripts/presentation_capture.py` (A100-New, 8×A100-40GB, ~6 min) | expert-overlap experiments, per-token keep masks, union-vs-prefill |
| `docs/results/level2/oracle_mag_freq.npz` | `scripts/oracle_mag_freq_capture.py` | 69,764-token keep-frequency + per-token score concentration |
| `docs/results/presentation/expert_scores_50p.pth` | scoring stage (`scores_50p/`), pulled from A100-New | measured ridge leverage per (layer, expert, channel) |

The capture command used:

```bash
bash ~/.claude/skills/launch-on-a100/scripts/a100.sh launch \
  --cmd '.venv/bin/python scripts/presentation_capture.py --layers 1,24,46 \
         --seq-len 2048 --n-seqs 6 --doc-pool 6000 --probe-experts 12 \
         --func-tokens 192 --out docs/results/presentation/pres_exps.npz' \
  -n 8 --name pres_exps
```

Model: `Qwen/Qwen3-30B-A3B-Thinking-2507` (E=128, K=8, I=768, 48 MoE layers),
bf16 + sdpa, C4 calibration. Channel scoring in the capture reproduces
`src/dynamic_active_param/block.py::_cross_expert_keep` exactly (verified against
a mock block: identical score tensor and selection).

### `fig_expert_overlap` — slide 6

Three panels, and the third is deliberately a *negative* result:

- **(a)** cosine between two experts' mean routed input (0.59) vs between two
  individual token states (0.27). The router does not partition token space.
- **(b)** leave-one-out shared subspace: fraction of an expert's `up_proj` energy
  explained by a rank-`r` eigenbasis of the *other* 127 experts, against a
  norm-matched Gaussian control (2.4× chance at rank 70).
- **(c)** substitution damage: relative output error when a token's top-1 expert
  is replaced by (i) another expert the router chose, (ii) one it did not, (iii) a
  random-weight expert. All ≈1.1–1.5 — i.e. **whole experts are not
  interchangeable**, and the least-squares fit of an expert from all 127 others
  reaches only R²=0.03.

This is why the argument moves to channel granularity: the redundancy that exists
(a, b) cannot be harvested at expert granularity (c).

### `fig_load_balance` — slide 6 companion

Routing load over 69,764 tokens. The auxiliary loss does **not** equalise usage:
load CV ≈ 0.9–1.4 at every depth, 36% of experts get under ¼ of a uniform share,
and 66 experts never fire at all.

### `fig_leverage_spectrum` — slide 7

Measured ridge leverage `diag((C+λI)⁻¹C)`, λ=1, from the C4 calibration run — not
an SVD proxy. Early layers are genuinely peaked (L0: top 8% of channels hold 45%
of the leverage, effective width 376/768); depth flattens this to ~19% by L47.

### `fig_channel_granularity` — slide 8

(a) distribution over tokens of the score mass captured by the top ρ of channels
(median 50% at ρ=0.125, 90% at ρ=0.5, from the 69,764-token capture).
(b) the three granularities compared, each labelled with the measurement backing
it. The 87.5% bar is the `oracle_mag` operating point at ρ=0.125, whose accuracy
cost (−1.7 pt HellaSwag acc_norm) comes from the Level-2 sweep, not this run.

### `fig_fixed_fails` — slides 8 / 17

(a) exact per-token keep masks for the three most-routed experts at layer 24,
ρ=0.125: rows are consecutive tokens routed to that expert, columns are channels.
Consecutive tokens share only 7–20% of their kept channels.
(b) running union of channels any token has activated, vs prefill length: 12.5%
per token but 74% of the expert's channels after 2048 tokens.

### `fig_union_budgets` — slide 17 companion

The same union curve at all three budgets (ρ = 0.5 / 0.25 / 0.125).

## Schematic figures

Hand-laid diagrams (matplotlib patches, no data), from
`scripts/presentation_illustrations.py`.

### `fig_channel_activation` — slide 9

Two tokens ("Where", "is") through the same 4-expert MoE layer, one band each.
Both bands show the same four experts in the same order, so the shared expert
(E2) sits at the same x in both — routed by both tokens, but with a *different*
lit channel subset. Channel orientation is technically faithful: a channel is a
`gate` row + an `up` row + a `down` column, so `gate`/`up` light up as horizontal
stripes and `down` as vertical ones.

### `fig_framework` — slide 11

Two concurrent lanes. Top: layer *i+1*'s **full-width** `up_proj` (MoBE-compressed
storage) produces the per-channel score `|up·x|`, the top-B mask selects channels,
and only those rows/columns are fetched from DRAM/HBM and staged. Bottom: layer
*i* computes on channels staged a layer earlier, with `gate`/`down` narrowed and
only `up` at full width. Dashed ghost outlines show the bytes *not* loaded.

**Note on orientation:** the slide outline's older wording said "gate_proj as the
key". The implemented method (`oracle_up` in `src/dynamic_active_param/block.py`)
and slides 11/13 use **`up_proj`** as the scoring signal, with `gate_proj` and
`down_proj` the ones reduced. The figure follows the implementation.
