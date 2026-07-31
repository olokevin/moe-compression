#!/usr/bin/env bash
# Orchestrator for the two down_proj-only 37.5% MoBE experiments (output-side vs
# input-side shared basis). Runs both fits in parallel (model on CPU, one GPU
# each), then both sharded PPL+HellaSwag+MMLU evals sequentially.
#
# Distinct name_suffix in each fit config guarantees distinct run dirs even when
# both launch in the same second. Runs from repo root in its own screen.
set -u
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
export PYTHONPATH="$REPO"
LOGDIR="$REPO/run_logs"
mkdir -p "$LOGDIR"
STAMP="$(date +%m%d-%H%M%S)"
ORCH="$LOGDIR/down375_orch_${STAMP}.log"
PY="$REPO/.venv/bin/python"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ORCH"; }

FIT_OUT_LOG="$LOGDIR/down_out375_fit_${STAMP}.log"
FIT_IN_LOG="$LOGDIR/down_in375_fit_${STAMP}.log"

# ── 1. Launch both fits in parallel (GPU 0 = output-side, GPU 1 = input-side) ─
log "launching output-side fit on GPU 0 -> $FIT_OUT_LOG"
screen -dmS down_out375_fit bash -lc "cd $REPO && env CUDA_VISIBLE_DEVICES=0 MODEL_ON_CPU=1 ATTN_IMPLEMENTATION=sdpa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$REPO $PY src/compress_then_train.py --config configs/compress_then_train/qwen3_30b_a3b_mobe_down_out_375_fit_only.yaml > $FIT_OUT_LOG 2>&1"
log "launching input-side fit on GPU 1 -> $FIT_IN_LOG"
screen -dmS down_in375_fit bash -lc "cd $REPO && env CUDA_VISIBLE_DEVICES=1 MODEL_ON_CPU=1 ATTN_IMPLEMENTATION=sdpa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$REPO $PY src/compress_then_train.py --config configs/compress_then_train/qwen3_30b_a3b_mobe_down_in_375_fit_only.yaml > $FIT_IN_LOG 2>&1"

sleep 30
while screen -ls 2>/dev/null | grep -qE 'down_(out|in)375_fit'; do sleep 60; done
log "both fits finished."

# ── 2. Resolve checkpoint dirs from the fit logs and patch eval configs ──────
get_run_dir() { grep -oE 'Run directory: [^ ]+' "$1" | head -1 | awk '{print $3}'; }
OUT_RUN="$(get_run_dir "$FIT_OUT_LOG")"
IN_RUN="$(get_run_dir "$FIT_IN_LOG")"
OUT_CKPT="$OUT_RUN/compressed_model/hf_reconstructed"
IN_CKPT="$IN_RUN/compressed_model/hf_reconstructed"
log "output-side ckpt: $OUT_CKPT"
log "input-side  ckpt: $IN_CKPT"
[ -f "$REPO/$OUT_CKPT/config.json" ] || log "ERROR: output ckpt missing config.json"
[ -f "$REPO/$IN_CKPT/config.json" ]  || log "ERROR: input ckpt missing config.json"

sed -i "s#^model_name_or_path: .*#model_name_or_path: $OUT_CKPT#" configs/compress_then_train/qwen3_30b_a3b_mobe_down_out_375_eval.yaml
sed -i "s#^model_name_or_path: .*#model_name_or_path: $IN_CKPT#"  configs/compress_then_train/qwen3_30b_a3b_mobe_down_in_375_eval.yaml

# ── 3. Launch both evals sequentially (each shards all 8 GPUs) ───────────────
EVAL_OUT_LOG="$LOGDIR/down_out375_eval_${STAMP}.log"
EVAL_IN_LOG="$LOGDIR/down_in375_eval_${STAMP}.log"

log "launching OUTPUT-side eval -> $EVAL_OUT_LOG"
env FORCE_DEVICE_MAP_AUTO=1 PER_GPU_MEM=36GiB ATTN_IMPLEMENTATION=sdpa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$REPO" \
    "$PY" src/compress_then_train.py --config configs/compress_then_train/qwen3_30b_a3b_mobe_down_out_375_eval.yaml > "$EVAL_OUT_LOG" 2>&1
log "output-side eval done."

log "launching INPUT-side eval -> $EVAL_IN_LOG"
env FORCE_DEVICE_MAP_AUTO=1 PER_GPU_MEM=36GiB ATTN_IMPLEMENTATION=sdpa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$REPO" \
    "$PY" src/compress_then_train.py --config configs/compress_then_train/qwen3_30b_a3b_mobe_down_in_375_eval.yaml > "$EVAL_IN_LOG" 2>&1
log "input-side eval done. ALL COMPLETE."
