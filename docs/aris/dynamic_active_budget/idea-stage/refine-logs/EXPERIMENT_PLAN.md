# Experiment Plan — GATE

Model: Qwen3-30B-A3B-Thinking-2507 (K=8, I=768, B=round(0.67·8·768)=4116). Scores dir exists. A100-New, 4 GPUs (see launch-on-a100 skill). All runs at matched ~33% active-param reduction, same calibration data, same LoRA budget.

## The paper lives or dies on Block B2. Run it first.

### Block A — Cheap masking-sim pilots (oracle, ~1 GPU-run each, code ~1h to add)
Implementation: add `criterion: "global"` / `global_pool: true` to `src/dynamic_active_param/block.py` — stack the K experts' `(I,)` scores into `(K,I)`, apply `g_e` weighting, block-group, keep global top-⌈B/b⌉ blocks with a ≥1-block/expert floor, build the keep-mask. Reuse existing config/eval/leverage-collection infra.

| Run | criterion | score | floor | Purpose |
|---|---|---|---|---|
| A0 | static uniform (real-slim Nyström) | leverage | — | **honest floor ~65%** (real) |
| A0' | static uniform (masking-sim) | activation | — | sim baseline 78.23 |
| A1 | naive per-expert dynamic | router_prob×act | k_min | reproduce 75.96 (negative baseline) |
| A2 | **global pool** | g_e·leverage | ≥1 block | 🏆 core method (sim oracle) |
| A3 | global pool | unweighted leverage | ≥1 block | test the g_e claim |
| A4 | global pool | activation | ≥1 block | leverage-vs-activation |
| A5 | global pool | random | ≥1 block | control: does scoring matter? |
| A6 | global pool | g_e·leverage | allow-zero | zeroing-experts failure-mode test |

**Gate to proceed:** A2 must beat BOTH A0' (78.23) and A1 (75.96) in masking-sim by >1 pt. If it ties/loses → negative result; stop and report honestly.

### Block B — The isolating ablation that IS the paper
- **B2 (non-negotiable):** global-pool (A2) vs matched per-expert allocation (A1 upgraded to same g_e·leverage score + same budget), *everything else fixed*. Global must win by >1 pt on ≥3 tasks. This is the contribution; if it fails there is no paper.

### Block C — Real reduction + recovery (deployable claim)
- C1: physically slim the winning block-granular config → real active-param cut; LoRA recovery; report HellaSwag must reach ≥77–78 (within 2–3 pt of dense) vs the 65% floor.
- C2: wall-clock latency/throughput (prefill + decode) vs dense top-K at block granularity; target ≥1.2× or reframe as active-FLOP@iso-accuracy.
- C3: select-only+LoRA vs select+reconstruct+LoRA at a fixed mask (justify skipping Nyström reconstruction).

### Block D — Generality (clears the bar)
- D1: tasks beyond HellaSwag: MMLU, ARC-c, WinoGrande, GSM8K, wikitext PPL.
- D2: ≥2 model families: add DeepSeek-MoE-16B and/or Qwen1.5-MoE-A2.7B.

## Run order
1. Add `global` criterion + block-granularity + floor to block.py; extend unit tests (budget conservation, ≥1-block floor, g_e weighting, ρ=1 ⇒ baseline).
2. Block A pilots (A2/A3/A1/A0' first) → check the gate.
3. If gate passes: Block B2, then Block C (real), then Block D (generality).
4. If gate fails: write it up as "a negative result on per-token dynamic allocation" + the leverage-select analysis.

## Compute budget
Each 30B HellaSwag masking-sim eval ≈ full-model 4-GPU run (dynamic path = un-slimmed). Block A ≈ 7 runs; keep within MAX_TOTAL_GPU_HOURS by running A2/A1/A0'/A3 first.
