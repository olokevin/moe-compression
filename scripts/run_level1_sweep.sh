#!/usr/bin/env bash
# Orchestrate the Level-1 / router_prob×activation comparison sweep on one A100
# box: 6 eval jobs in 3 waves of 2 (GPUs 0-3 and 4-7), waiting for each wave.
# Run from the repo root on the remote box.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/src"
export WANDB_MODE=disabled
mkdir -p run_logs

run_pair() {
  local name_a="$1" cfg_a="$2" name_b="$3" cfg_b="$4"
  local stamp; stamp=$(date +%m%d-%H%M%S)
  echo "[sweep] wave start: $name_a (gpu 0-3) + $name_b (gpu 4-7) @ $stamp"
  CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python src/train/merge_slim_eval.py \
    --config "configs/eval/$cfg_a" > "run_logs/${name_a}_${stamp}.log" 2>&1 &
  local pid_a=$!
  CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python src/train/merge_slim_eval.py \
    --config "configs/eval/$cfg_b" > "run_logs/${name_b}_${stamp}.log" 2>&1 &
  local pid_b=$!
  wait $pid_a; echo "[sweep] $name_a done (rc $?)"
  wait $pid_b; echo "[sweep] $name_b done (rc $?)"
}

# Wave 1 — HellaSwag router_prob×activation 62.5% + 75%
run_pair prob_act_625_hs qwen3_30b_a3b_dynamic_prob_act_625_hellaswag.yaml \
         prob_act_75_hs  qwen3_30b_a3b_dynamic_prob_act_75_hellaswag.yaml

# Wave 2 — HellaSwag router_prob×activation 87.5% + Level-1 87.5%
run_pair prob_act_875_hs qwen3_30b_a3b_dynamic_prob_act_875_hellaswag.yaml \
         pivchol_875_hs   qwen3_30b_a3b_dynamic_pivchol_875_hellaswag.yaml

# Wave 3 — MMLU Level-1 75% + router_prob×activation 75%
run_pair pivchol_75_mmlu  qwen3_30b_a3b_dynamic_pivchol_75_mmlu.yaml \
         prob_act_75_mmlu qwen3_30b_a3b_dynamic_prob_act_75_mmlu.yaml

echo "[sweep] ALL_WAVES_DONE"
