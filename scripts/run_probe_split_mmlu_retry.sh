#!/usr/bin/env bash
# RETRY of the iso-cost split sweep's MMLU arms.
#
# The first attempt (scripts/run_probe_split_sweep.sh on A100-Sagemaker) gave each
# job only 2 GPUs; 5-shot MMLU OOMed at 99% (55480/56168) after ~9h55m because a
# 30B shard plus 5-shot contexts does not fit on 2x40GB. Every earlier MMLU probe
# eval in this study used 4 GPUs. This reruns both arms at 4 GPUs each, serially.
#
# Both arms are iso-cost at used = rho_ch + 2*rho_in/3 = 0.2500 (-75.0%):
#   opt  rho_in=0.1875, rho_ch=0.12500   (the solved optimum, 50/50 split)
#   mis  rho_in=0.2500, rho_ch=0.08333   (hand-picked rho_input, 67/33 split)
#
# HellaSwag already measured: opt 74.08, mis 73.80 (gap +0.28, predicted +0.58).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/src"
mkdir -p run_logs

GPUS="${GPUS:-0,1,2,3}"

run_one() {  # name cfg
  local stamp; stamp=$(date +%m%d-%H%M%S)
  echo "[retry] $1 on gpu $GPUS @ $stamp"
  CUDA_VISIBLE_DEVICES=$GPUS .venv/bin/python src/train/merge_slim_eval.py \
    --config "configs/eval/$2" > "run_logs/${1}_${stamp}.log" 2>&1
  echo "[retry] $1 done (rc $?)"
}

run_one splitopt_mmlu_retry qwen3_30b_a3b_probe_router_split_opt_mmlu.yaml
run_one splitmis_mmlu_retry qwen3_30b_a3b_probe_router_split_mis_mmlu.yaml

echo "[retry] ALL_DONE"
