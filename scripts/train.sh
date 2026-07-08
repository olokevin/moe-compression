#!/usr/bin/env bash
# Stage 2: mask generation (CBA §4.2 + AAR §4.3) + LoRA fine-tuning.
# Set scores_dir and output_dir in the config before running.
set -euo pipefail

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/train/qwen1_5_moe_a2_7b_e2e_alpaca.yaml}"
NPROC="${NPROC:-8}"                 # number of GPUs / processes
MASTER_PORT="${MASTER_PORT:-29502}"

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    src/train/train.py --config "${CONFIG}"
