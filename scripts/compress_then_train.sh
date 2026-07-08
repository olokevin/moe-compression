#!/usr/bin/env bash
# Compress-then-train pipeline for Qwen/Qwen2.5-0.5B:
#   1) C4-calibrated compression, 2) continue training on C4 (CE),
#   3) eval C4 PPL + hellaswag + MMLU (lm-eval-harness), before vs. after.
#
# Everything is driven by the YAML config; the env vars below override
# individual fields on the CLI (CLI flags win over YAML).
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

CONFIG="${CONFIG:-configs/compress_then_train/qwen2_5_0_5b_c4.yaml}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}"
TRAIN_MODE="${TRAIN_MODE:-svd_llm_v2}"       # svd, svd_llm, svd_llm_v2, btt, btt_llm_v2, ...
COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.8}"  # 0.8 => retain 80% (20% compression)
KD_LOSS_TYPE="${KD_LOSS_TYPE:-ce}"           # ce = teacher-free CE on C4
CE_STEPS="${CE_STEPS:-200}"
LR="${LR:-1e-4}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-hellaswag,mmlu}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-200}"        # per-task cap; -1 = full task

python src/compress_then_train.py --config "$CONFIG" \
    --model_name_or_path "$MODEL_NAME" \
    --train_mode "$TRAIN_MODE" \
    --compression_ratio "$COMPRESSION_RATIO" \
    --kd_loss_type "$KD_LOSS_TYPE" \
    --ce_steps "$CE_STEPS" \
    --lr "$LR" \
    --lm_eval_tasks "$LM_EVAL_TASKS" \
    --lm_eval_limit "$LM_EVAL_LIMIT" \
    --save_steps none
