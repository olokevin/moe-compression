#!/usr/bin/env bash
# Orchestrate the "top-4 experts x narrower experts" stacking sweep on one A100
# box: 8 eval jobs in waves of 2 (GPUs 0-3 and 4-7). Run from repo root.
#
# Every job sets reduce_topk: 4 (route to top-4 of 8 experts) AND a Level-2
# cross-expert dynamic budget, so the two active-param reductions multiply:
#
#   oracle_mag_noW : rank by g*|inter|, budget cuts down_proj only
#   oracle_up      : rank by g*|up|*||W_down||, budget cuts gate_proj + down_proj
#   both @ {50%, 75%} of the reduced K_new*I budget x {HellaSwag 0-shot, MMLU 5-shot}
#
# Budgets are measured against the already-halved active path
# (B = (1-prune_ratio) * 4 * 768), so nominal 50%/75% here means 58.3%/62.5%
# (noW) and 66.7%/75.0% (up) of the *dense top-8* expert FFN.
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

# Wave 1 — HellaSwag @ nominal 50%
run_pair k4_noW_50_hs qwen3_30b_a3b_dynamic_topk4_oracle_mag_noW_50_hellaswag.yaml \
         k4_up_50_hs  qwen3_30b_a3b_dynamic_topk4_oracle_up_50_hellaswag.yaml
# Wave 2 — HellaSwag @ nominal 75%
run_pair k4_noW_75_hs qwen3_30b_a3b_dynamic_topk4_oracle_mag_noW_75_hellaswag.yaml \
         k4_up_75_hs  qwen3_30b_a3b_dynamic_topk4_oracle_up_75_hellaswag.yaml
# Wave 3 — MMLU @ nominal 50%
run_pair k4_noW_50_mmlu qwen3_30b_a3b_dynamic_topk4_oracle_mag_noW_50_mmlu.yaml \
         k4_up_50_mmlu  qwen3_30b_a3b_dynamic_topk4_oracle_up_50_mmlu.yaml
# Wave 4 — MMLU @ nominal 75%
run_pair k4_noW_75_mmlu qwen3_30b_a3b_dynamic_topk4_oracle_mag_noW_75_mmlu.yaml \
         k4_up_75_mmlu  qwen3_30b_a3b_dynamic_topk4_oracle_up_75_mmlu.yaml

# The plain reduce-top-k MMLU reference row (the "fewer experts only" baseline
# these four MMLU rows are read against) is NOT in this sweep — it needs no
# artifact and is independent, so run it in parallel on the other box:
#   configs/eval/qwen3_30b_a3b_reduce_topk4_mmlu.yaml
# The HellaSwag counterpart already exists (reduce_topk4_hellaswag, 75.96).

echo "[sweep] ALL_WAVES_DONE"
