#!/usr/bin/env bash
# Wave 2 of the reuse-probe study: the two *principled* allocation gains, both at
# the SAME -75.0% used-parameter budget as the hand-crafted p=0.25 / rho=0.10 row
# (configs/eval/qwen3_30b_a3b_probe_reuse_k25_r10_*), so the comparison is
# iso-cost by construction:
#
#   router      -- split the pooled coordinate-read budget across the token's K
#                  experts by g_e*|x_i| instead of an equal top-p*H set per expert.
#                  Offline screen: -0.025 rel_err vs uniform (~+0.66 pt).
#   sched       -- per-layer (p, rho) from the all-layer rel_err surface, solved
#                  by one Lagrange multiplier at a fixed mean budget.
#                  Offline: -0.020 rel_err vs the best uniform (~+0.52 pt).
#   sched+router-- both stacked.
#
# 6 jobs in 3 waves of 2, 4 GPUs each. The schedule file
# docs/results/idea_pilot/schedule_cut75.json must be present (it is committed).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/src"
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

# Wave 1 — router input allocation (the single cheapest win)
run_pair router_hs   qwen3_30b_a3b_probe_router_k25_r10_hellaswag.yaml \
         router_mmlu qwen3_30b_a3b_probe_router_k25_r10_mmlu.yaml
# Wave 2 — per-layer schedule
run_pair sched_hs   qwen3_30b_a3b_probe_sched_cut75_hellaswag.yaml \
         sched_mmlu qwen3_30b_a3b_probe_sched_cut75_mmlu.yaml
# Wave 3 — both stacked
run_pair schedrouter_hs   qwen3_30b_a3b_probe_sched_router_cut75_hellaswag.yaml \
         schedrouter_mmlu qwen3_30b_a3b_probe_sched_router_cut75_mmlu.yaml

echo "[sweep] ALL_WAVES_DONE"
