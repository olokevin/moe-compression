#!/usr/bin/env bash
# Orchestrate the Q1 (oracle_mag_noW) + Q2 (oracle_up) sweep on one A100 box:
# 8 eval jobs in waves of 2 (GPUs 0-3 and 4-7). Run from repo root on the box.
#
#   Q1 oracle_mag_noW : rank by g*|inter| only (drop the ||W_down|| factor)
#   Q2 oracle_up      : rank by up_proj output, cut gate_proj + down_proj
#   both @ {75%, 87.5%} x {HellaSwag 0-shot, MMLU 5-shot}
#
# No artifacts to warm: oracle_mag_noW needs nothing offline, oracle_up only
# builds down_proj column norms inline in install_dynamic_alloc.
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

# Wave 1 — Q1 noW 75 HS + Q2 up 75 HS
run_pair noW_75_hs qwen3_30b_a3b_dynamic_oracle_mag_noW_75_hellaswag.yaml \
         up_75_hs  qwen3_30b_a3b_dynamic_oracle_up_75_hellaswag.yaml
# Wave 2 — Q1 noW 87.5 HS + Q2 up 87.5 HS
run_pair noW_875_hs qwen3_30b_a3b_dynamic_oracle_mag_noW_875_hellaswag.yaml \
         up_875_hs  qwen3_30b_a3b_dynamic_oracle_up_875_hellaswag.yaml
# Wave 3 — Q1 noW 75 MMLU + Q2 up 75 MMLU
run_pair noW_75_mmlu qwen3_30b_a3b_dynamic_oracle_mag_noW_75_mmlu.yaml \
         up_75_mmlu  qwen3_30b_a3b_dynamic_oracle_up_75_mmlu.yaml
# Wave 4 — Q1 noW 87.5 MMLU + Q2 up 87.5 MMLU
run_pair noW_875_mmlu qwen3_30b_a3b_dynamic_oracle_mag_noW_875_mmlu.yaml \
         up_875_mmlu  qwen3_30b_a3b_dynamic_oracle_up_875_mmlu.yaml

echo "[sweep] ALL_WAVES_DONE"
