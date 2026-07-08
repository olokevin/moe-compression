# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Official implementation of "Attribution-Guided and Coverage-Maximized Pruning for Structural MoE Compression" (ICML 2026 Spotlight). It structurally prunes the expert FFN intermediate dimension of MoE LLMs (Qwen1.5-MoE-A2.7B, DeepSeek-MoE-16B, DeepSeek-V2-Lite, Qwen3-30B-A3B), then recovers accuracy with LoRA fine-tuning.

## The three-stage pipeline

The workflow is **score → train (generate mask + fine-tune) → eval (physically slim + benchmark)**. Each stage feeds artifacts to the next via directories set in YAML configs.

1. **Scoring** (`src/calibration/channel_scoring/main.py`) — Runs calibration data through the model layer-by-layer, collecting per-channel attribution/importance scores. Writes `expert_scores.pth`, `gate_scores.pth`, `layerwise_loss.pth` to `<output-dir>/<model>/<dataset>/scores/`. That `scores/` directory is what later stages consume as `scores_dir`.

2. **Mask generation + training** (`src/train/train.py`) — Calls `generate_masks()` to plan which channels to keep, applies **fake pruning** (masking, not removal) via `mask_expert()`, wraps experts in LoRA, and fine-tunes with `SlimTrainer`. Masks are saved alongside checkpoints (`masks.pth`) by `SavePruningArtifactsCallback`.

3. **Eval** (`src/train/merge_slim_eval.py`) — Loads the LoRA adapter, `merge_and_unload()`s it, then calls `build_real_slim_model()` to **physically remove** the pruned channels (real slimming), and benchmarks via lm-eval-harness.

**Fake vs. real pruning is a core distinction:** during scoring and training, channels are masked (`src/prune/apply/masking/`, multiply intermediate activations by 0/1) so gradients still flow and masks can anneal. Only at eval time does `src/prune/apply/slimming/` actually shrink the weight tensors to produce a smaller dense checkpoint.

## Mask generation internals (`src/prune/generate/`)

`generate_masks()` (in `pipeline.py`) is the orchestrator. If `mask_dir` is set it just loads a saved mask and skips planning. Otherwise it runs `prepare_scores → init_mask_for_I → adjust_masks`. Planning is split into two orthogonal decisions, each a dispatcher over pluggable algorithms:

- **inter-layer** (`planners/inter_layer/`) — how much budget each *layer* gets. Dispatched by `inter_layer_planner()` on `inter_layer_method`. Default `loss_coverage` uses smoothed per-layer calibration loss + binary search.
- **intra-layer** (`planners/intra_layer/`) — within a layer, how channels are split *across experts*. Dispatched by `build_masks()` on `intra_layer_method`. Default `attr_coverage` uses per-expert coverage weights (from `expert_out_token_contrib` scores) to allocate channels via binary search + cumulative-score thresholding.

The `intra_expert_metric` (default `activation`) selects *which* score tensor ranks channels within an expert; the `*_method` names select the *allocation algorithm*. These are set under `prune_kwargs.mask_method_kwargs` in the YAML. `adjust_masks_kwargs` (`align_inter`, `min_per_expert`) then rounds kept-channel counts to hardware-friendly multiples and enforces a per-expert floor.

## Model abstraction layer

The pipeline is model-agnostic through `src/base/shared_utils/safe_isinstance.py`, which provides `_is_moe_block`, `_get_experts`, `_get_moe_intermediate_size`, `_get_num_experts`, etc. These branch on Qwen2-MoE / Qwen3-MoE / DeepSeek(-V2) internals. **To support a new MoE architecture, extend the detectors and getters here** rather than touching the pruning logic. `src/base/models/get_model.py` handles HF loading, quantization config, and tokenizer setup.

## Commands

Scripts assume they are run from the repo root and set `PYTHONPATH="$(pwd)"` themselves. All modules are imported as `src.*`, so that PYTHONPATH is required.

```bash
# Stage 1 — collect channel scores (edit model/dataset inline in the script)
bash scripts/scoring.sh

# Stage 2 — generate masks + LoRA fine-tune (multi-GPU via torchrun)
bash scripts/train.sh          # runs: torchrun ... src/train/train.py --config configs/train/<model>.yaml

# Stage 3 — physically slim + evaluate on lm-eval-harness tasks
bash scripts/eval.sh           # runs: python src/train/merge_slim_eval.py --config configs/eval/<model>.yaml
```

**Note:** the eval entry point is `src/train/merge_slim_eval.py`. It must **not** be
named `eval.py` — running a script named `eval.py` puts its directory on `sys.path[0]`
and shadows the top-level `eval/` package that `eval_dispatch.py` imports (`import eval`
would resolve to the script, raising "'eval' is not a package").

There is no test suite and no lint/format config in this repo. `src/base/shared_utils/throughput_test.py` is a throughput utility, not a unit test.

## Configuration

Everything for train/eval is driven by a single YAML config parsed into the `E2EArguments` dataclass (`src/base/argparser/e2e_args.py`); `post_init()` converts `dtype` strings to `torch.dtype`, seeds RNGs, and appends a timestamp to `output_dir`/`wandb_name`. To change a run, edit the YAML — CLI only accepts `--config`. Before running you must fill in the placeholder paths (`scores_dir`, `resume_path`, `mask_dir`, `output_dir`) in the chosen config.

Key config knobs: `prune_kwargs.prune_ratio` (target fraction removed), `real_slim`/`shrink_gate` (whether eval physically slims and shrinks the gate), `enable_gate_lora`/`enable_attn_lora` (staged LoRA targets beyond the default expert `gate_proj/up_proj/down_proj`), and `train_router_aux_loss`.

## Dependencies

`requirements.txt` is the light set; `requirements_full.txt` pins the full environment. Note two dependencies not in the light file but required at runtime: `lm_eval==0.4.9.1` (eval harness, imported in `eval/lm_harness/`) and an `alignment` package (`get_kbit_device_map`, imported in `get_model.py`).
