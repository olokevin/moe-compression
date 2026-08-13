#!/usr/bin/env bash
# Orchestrate the lowrank_scorer sweep on one A100 box: 6 eval jobs in waves of
# 2 (GPUs 0-3 and 4-7). Run from repo root on the box.
#
# All at a nominal −75% channel cut, HellaSwag 0-shot, masking simulation:
#
#   up+gate scorer (proxy ~ SiLU(gate_hat)*up_hat, targets oracle_mag_noW)
#     svd_r16_upgate      global rank-16 SVD of W_up and W_gate
#     svd_r32_upgate      global rank-32 SVD of W_up and W_gate
#     btt_m2n2_r32_upgate 2x2 block grid, per-block rank 32
#   up-only scorer (proxy ~ |up_hat|, half the scorer cost)
#     svd_r16_uponly / svd_r32_uponly / btt_m2n2_r32_uponly
#
# No offline artifacts to warm: the scorer cores are factorized from the expert
# weights inside install_dynamic_alloc (randomized SVD, matmul-only).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/src"
export WANDB_MODE=disabled
mkdir -p run_logs

run_pair() {  # name_a cfg_a name_b cfg_b
  local stamp; stamp=$(date +%m%d-%H%M%S)
  echo "[sweep] wave: $1 (gpu 0-3) + $3 (gpu 4-7) @ $stamp"
  CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python src/train/merge_slim_eval.py \
    --config "configs/eval/$2" > "run_logs/${1}_${stamp}.log" 2>&1 &
  local pa=$!
  CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python src/train/merge_slim_eval.py \
    --config "configs/eval/$4" > "run_logs/${3}_${stamp}.log" 2>&1 &
  local pb=$!
  wait $pa; echo "[sweep] $1 done (rc $?)"
  wait $pb; echo "[sweep] $3 done (rc $?)"
}

P=qwen3_30b_a3b_dynamic_lowrank

# Wave 1 — the two rank-32 headline points (best recall in each mode)
run_pair svd_r32_upgate ${P}_svd_r32_upgate_75_hellaswag.yaml \
         svd_r32_uponly ${P}_svd_r32_uponly_75_hellaswag.yaml
# Wave 2 — rank-16 (half the cost) in both modes
run_pair svd_r16_upgate ${P}_svd_r16_upgate_75_hellaswag.yaml \
         svd_r16_uponly ${P}_svd_r16_uponly_75_hellaswag.yaml
# Wave 3 — BTT block grid in both modes
run_pair btt_r32_upgate ${P}_btt_m2n2_r32_upgate_75_hellaswag.yaml \
         btt_r32_uponly ${P}_btt_m2n2_r32_uponly_75_hellaswag.yaml

echo "[sweep] ALL_WAVES_DONE"
