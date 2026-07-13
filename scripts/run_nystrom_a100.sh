#!/usr/bin/env bash
# Launch the nystrom compress-then-train run on A100-New in the background.
# Self-contained: sets PYTHONPATH (repo root + src/), pins one GPU, logs to file.
set -euo pipefail

cd "$HOME/yequan/moe-compression"
export PYTHONPATH="$(pwd):$(pwd)/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1

CONFIG="${1:-configs/compress_then_train/05B_mlp_nystrom.yaml}"
LOGDIR="$HOME/yequan/moe-compression/run_logs"
mkdir -p "$LOGDIR"
STAMP=$(date +%m%d-%H%M%S)
LOG="$LOGDIR/nystrom_${STAMP}.log"

echo "config=$CONFIG log=$LOG gpu=$CUDA_VISIBLE_DEVICES" > "$LOGDIR/nystrom_latest.meta"
echo "$LOG" > "$LOGDIR/nystrom_latest.path"

nohup .venv/bin/python src/compress_then_train.py --config "$CONFIG" \
  > "$LOG" 2>&1 &
echo $! > "$LOGDIR/nystrom_latest.pid"
echo "LAUNCHED pid=$(cat "$LOGDIR/nystrom_latest.pid") log=$LOG"
