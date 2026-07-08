#!/usr/bin/env bash
# Stage 1: Attribution-Guided Loss Approximation (ALA, paper §4.1)
# Collects per-channel importance scores + the expert-wise loss proxy on a
# calibration set. Outputs go to <output-dir>/<model>/<dataset>/scores/.
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Overridable knobs (defaults target a single-GPU CUDA box).
MODEL="${MODEL:-Qwen/Qwen1.5-MoE-A2.7B}"
CALIB_DATASETS="${CALIB_DATASETS:-c4}"
CALIB_BATCHES="${CALIB_BATCHES:-200}"   # 200 ≈ full calibration (~3M tokens); lower for a quick test
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python src/calibration/channel_scoring/main.py \
    --model-name-or-path "${MODEL}" \
    --calib-datasets ${CALIB_DATASETS} \
    --calib-batches "${CALIB_BATCHES}" \
    --batch-size "${BATCH_SIZE}" \
    --max-seq-length "${MAX_SEQ_LENGTH}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --trust-remote-code \
    --output-dir "${OUTPUT_DIR}" \
    --verbose
