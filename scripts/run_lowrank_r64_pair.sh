#!/usr/bin/env bash
# svd_r64 in both modes: the equal-cost partners of btt_m2n2_r32 up+gate/up-only.
# Settles BTT-vs-SVD on ACCURACY (Inv. 1 settled it only on recall).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/src"
export WANDB_MODE=disabled
mkdir -p run_logs
P=qwen3_30b_a3b_dynamic_lowrank
stamp=$(date +%m%d-%H%M%S)
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python src/train/merge_slim_eval.py \
  --config configs/eval/${P}_svd_r64_upgate_75_hellaswag.yaml \
  > run_logs/svd_r64_upgate_${stamp}.log 2>&1 &
a=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python src/train/merge_slim_eval.py \
  --config configs/eval/${P}_svd_r64_uponly_75_hellaswag.yaml \
  > run_logs/svd_r64_uponly_${stamp}.log 2>&1 &
b=$!
wait $a; echo "[r64] upgate rc $?"
wait $b; echo "[r64] uponly rc $?"
echo "[r64] ALL_DONE"
