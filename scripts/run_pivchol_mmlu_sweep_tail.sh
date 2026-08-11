#!/usr/bin/env bash
# Tail wave of the Level-1 (pivchol_global) MMLU budget sweep on one A100 box.
# Jobs at 50% (GPUs 0-3) and 62.5% (GPUs 4-7) are launched separately and run
# concurrently; this waiter blocks on the 50% job's PID, then reuses GPUs 0-3 to
# run the 87.5% job. Run under nohup on the remote box so it survives an SSH drop.
#   Usage: run_pivchol_mmlu_sweep_tail.sh <pid_of_50pct_job>
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/src"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p run_logs

wait_pid="${1:?need the PID of the 50% job to wait on}"
echo "[tail] waiting for 50% job pid=$wait_pid to exit ..."
while kill -0 "$wait_pid" 2>/dev/null; do sleep 60; done
echo "[tail] pid=$wait_pid gone; GPUs 0-3 free. Launching 87.5% MMLU."

stamp=$(date +%m%d-%H%M%S)
log="run_logs/pivchol_875_mmlu_${stamp}.log"
echo "$log" > run_logs/pivchol_875_mmlu_latest.path
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python src/train/merge_slim_eval.py \
  --config configs/eval/qwen3_30b_a3b_dynamic_pivchol_875_mmlu.yaml > "$log" 2>&1
echo "[tail] 87.5% MMLU done (rc $?)"
echo "[tail] PIVCHOL_MMLU_SWEEP_DONE"
