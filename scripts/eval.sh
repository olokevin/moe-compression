#!/usr/bin/env bash
# Stage 3: Evaluation — real slimming (CBA + AAR + physical channel removal),
# then benchmark on lm-evaluation-harness. Set resume_path/mask_dir/eval_task_names
# in the config; leave resume_path empty for the one-shot (no fine-tuning) model.
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/eval/qwen1_5_moe_a2_7b.yaml}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python src/train/merge_slim_eval.py --config "${CONFIG}"
