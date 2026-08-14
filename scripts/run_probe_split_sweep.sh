#!/usr/bin/env bash
# ISO-COST SPLIT SWEEP: how should a fixed used-parameter budget be divided
# between SCORING (rho_input) and COMPUTE (rho_channel)?
#
# Every prior probe row hand-picked rho_input=0.25 and then varied rho_channel,
# which conflates the split with the budget. Here the budget is PINNED at
# used = rho_channel + 2*rho_input/3 = 0.2500 (-75.0%) and only the split moves:
#
#   opt  rho_in=0.1875, rho_ch=0.12500   50.0% scoring / 50.0% compute  <- solved optimum
#   mis  rho_in=0.2500, rho_ch=0.08333   66.7% scoring / 33.3% compute  <- hand-picked value
#
# Principle: min rel_err(p,rho) s.t. rho+2p/3=C equalizes marginal rel_err per
# unit budget, (3/2)*d(relerr)/dp == d(relerr)/d(rho). Solved on the cached 8192
# -token surface docs/results/idea_pilot/layer_surface_8k.json.
#
# PRE-REGISTERED: opt beats mis by ~0.58 pt HellaSwag (rel_err 0.4613 vs 0.4833,
# -26.4 pt/unit). If it does not, the allocation principle is wrong.
# Both use input_alloc=router and bits=16 (probe = served weight, 0 extra storage).
#
# 4 jobs in 2 waves of 2, 4 GPUs each.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/src"
mkdir -p run_logs

run_pair() {  # name_a cfg_a name_b cfg_b gpus_a gpus_b
  local stamp; stamp=$(date +%m%d-%H%M%S)
  echo "[split] wave: $1 (gpu $5) + $3 (gpu $6) @ $stamp"
  CUDA_VISIBLE_DEVICES=$5 .venv/bin/python src/train/merge_slim_eval.py \
    --config "configs/eval/$2" > "run_logs/${1}_${stamp}.log" 2>&1 &
  local pa=$!
  CUDA_VISIBLE_DEVICES=$6 .venv/bin/python src/train/merge_slim_eval.py \
    --config "configs/eval/$4" > "run_logs/${3}_${stamp}.log" 2>&1 &
  local pb=$!
  wait $pa; echo "[split] $1 done (rc $?)"
  wait $pb; echo "[split] $3 done (rc $?)"
}

GA="${GPUS_A:-4,5,6,7}"
GB="${GPUS_B:-4,5,6,7}"

# Wave 1 -- the solved optimum
run_pair splitopt_hs   qwen3_30b_a3b_probe_router_split_opt_hellaswag.yaml \
         splitopt_mmlu qwen3_30b_a3b_probe_router_split_opt_mmlu.yaml "$GA" "$GB"
# Wave 2 -- the mis-allocated control at the identical budget
run_pair splitmis_hs   qwen3_30b_a3b_probe_router_split_mis_hellaswag.yaml \
         splitmis_mmlu qwen3_30b_a3b_probe_router_split_mis_mmlu.yaml "$GA" "$GB"

echo "[split] ALL_WAVES_DONE"
