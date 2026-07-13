# Launching Qwen3-30B-A3B eval/train jobs on the 40GB A100 boxes

This is the battle-tested recipe for running the moe-compression pipeline on the
A100-New / A100-Sagemaker boxes, where each GPU is **40 GB** but Qwen3-30B-A3B is
**~61 GB in bf16** (the paper used 96 GB H20s). Everything here was learned the
hard way on 2026-07-09; follow it to avoid re-hitting the same OOMs.

## TL;DR launch commands

Use the `launch-on-a100` skill (`scripts/a100.sh`). Always prefix the `--cmd`
with these env vars for the 30B model:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_DEVICE_MAP_AUTO=1 PER_GPU_MEM=36GiB
```

- **`FORCE_DEVICE_MAP_AUTO=1`** — shards the model across all visible GPUs even
  when not `test_only` (scoring needs this). Added to `get_model.py`.
- **`PER_GPU_MEM=36GiB`** — caps accelerate's per-GPU budget so it actually shards
  (default hardcoded "95GiB" packs everything onto GPU0 → OOM). Use `30GiB` when
  you need extra eval headroom on the full (un-slimmed) model.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — reduces fragmentation OOMs.

Interpreter: `.venv/bin/python` for eval/scoring, `.venv/bin/torchrun` for training
(plain `torchrun` / `python` are NOT on PATH).

Every qwen3 config must set `attn_implementation: "sdpa"` (flash_attn isn't installed).

## GPU count per job (40GB cards)

| Job | Model in memory | GPUs | Notes |
|---|---|---|---|
| Channel scoring | full 61GB bf16, sharded | 8 | ~8.7GB/GPU; layer-by-layer, ~2h for 180 calib-batches |
| Baseline eval (no prune) | full 61GB bf16 | **4** | 2 GPUs OOMs during MMLU long contexts / eval spikes |
| One-shot / fine-tuned eval | slimmed (~46GB) | 3 | real-slim peak needs ≥3; 2 OOMs in `build_real_slim_model` |
| LoRA training (4-bit QLoRA) | ~15GB/rank | 8 | DDP, one 4-bit replica per rank |

Rule of thumb: **full model → 4 GPUs, slimmed model → 3 GPUs, training → 8 GPUs (DDP).**
Never run eval on 2 GPUs for this model.

## Eval task splitting (critical)

`eval_fn` runs all tasks in ONE `simple_evaluate` with a single `num_fewshot`, and
`simple_evaluate` only emits results after ALL tasks finish. Therefore:

- **Split c4 / hellaswag / mmlu into separate configs / runs.** Mixing them means
  one crash loses everything, and 0-shot vs 5-shot can't be mixed.
- HellaSwag & C4: `num_fewshot: 0`. MMLU: `num_fewshot: 5`.
- **C4 PPL**: cap `eval_sample_limit: 500` (the rolling-PPL outer loop is over
  documents — 45k docs = ~20h; 500 docs ≈ 15min). Reports `word_perplexity`.
- **MMLU**: it's 57 subtasks; `eval_sample_limit` is PER-SUBTASK. Use `50` (~2850
  questions, ±1% SE) for a fast run, or `-1` for the full 14k (slow, ~7h).
- HellaSwag full = 40168 requests (10042 × 4 choices); ~70min on 3-4 GPUs.

## Batch sizes for the LMEvalAdaptor

`batch_size` in the config drives the eval adaptor (NOT `per_device_eval_batch_size`).
- HellaSwag / C4: `batch_size: 16`.
- MMLU 5-shot (long contexts): `batch_size: 8`.

## Two adaptor bugs that were fixed (eval/lm_harness/eval_utils_lm_2.py)

1. **`loglikelihood` was unbatched** (one request per forward → ~1.1s/item, 13h for
   hellaswag). Rewrote to pre-tokenize, sort by length, left-pad, and run
   `batch_size` requests per forward → ~9 it/s (~12x). Correctness verified against
   the reference (hellaswag acc_norm identical).
2. **Full-vocab softmax OOM on MMLU.** `torch.log_softmax(logits.float())` over the
   whole `[batch, seq, vocab≈152k]` tensor spikes to ~7.5GB on long 5-shot contexts
   → OOM at 94-96%. Fixed by gathering ONLY the continuation-position logits across
   the batch, then one small log_softmax. Keep this — do NOT softmax the full seq.

Also fixed: `max_length` now honours the passed cap (Qwen3 reports
`max_position_embeddings=262144`, which blows up c4 rolling windows).

## Known failure modes → cause

- `mat1 and mat2 shapes cannot be multiplied (Nx2048 and 768x2048)` during training
  → the fake-prune wrapper's forward. Fixed to compute the full SwiGLU expert
  (`down_proj(act(gate_proj(x))*up_proj(x)) * mask`), see
  `src/prune/apply/masking/expert/fake_prune_wrapper.py`.
- `tensors on cuda:0 and cuda:1` in scoring → deepcopy keeps accelerate hooks;
  `remove_hook_from_module(copied_block, recurse=True)` (already in main.py).
- `FlashAttention2 ... not installed` → set `attn_implementation: "sdpa"`.
- A100-Sagemaker `/opt/conda` env is stale (transformers has no qwen3_moe,
  lm_eval 0.4.3, bitsandbytes/triton broken). Use A100-New.

## Timing / cost reference (25% prune + 4-bit QLoRA, this box)

- Scoring (180 calib-batches, ~0.74M tokens): ~2h on 8 GPUs.
- Training (80 steps, bs4 × accum2 × 8 GPUs = 64/step): ~137s/step, ~3h.
- HellaSwag eval: ~70min. C4(500): ~15min. MMLU(50/subtask): ~60min.
