#!/usr/bin/env bash
# Orchestrate the Level-2 cross-expert sweep on one A100 box: 13 eval jobs in
# waves of 2 (GPUs 0-3 and 4-7). Run from the repo root on the remote box.
#
#   8 HellaSwag : oracle_mag + pubsub @ {50,625,75,875}
#   2 MMLU@75%  : oracle_mag + pubsub
#   3 M4 beta   : pivchol_global beta={1.5,2,3} @ 50%
#
# The pubsub 50% job runs FIRST and alone so its CPU pivoted-Cholesky
# factorization warms pubsub_artifact_r8.pth before any parallel pubsub job.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/src"
export WANDB_MODE=disabled
mkdir -p run_logs

run() {  # name cfg gpus
  local name="$1" cfg="$2" gpus="$3" stamp
  stamp=$(date +%m%d-%H%M%S)
  CUDA_VISIBLE_DEVICES="$gpus" .venv/bin/python src/train/merge_slim_eval.py \
    --config "configs/eval/$cfg" > "run_logs/${name}_${stamp}.log" 2>&1
  echo "[sweep] $name done (rc $?)"
}

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

# Wave 0 — pubsub 50% ALONE first (warms pubsub_artifact_r8.pth on CPU).
echo "[sweep] warming pubsub artifact via pubsub_50 (gpu 0-3)"
run pubsub_50_hs qwen3_30b_a3b_dynamic_pubsub_50_hellaswag.yaml 0,1,2,3

# Wave 1 — oracle_mag 50 + pubsub 62.5
run_pair oracle_mag_50_hs qwen3_30b_a3b_dynamic_oracle_mag_50_hellaswag.yaml \
         pubsub_625_hs     qwen3_30b_a3b_dynamic_pubsub_625_hellaswag.yaml
# Wave 2 — oracle_mag 62.5 + pubsub 75
run_pair oracle_mag_625_hs qwen3_30b_a3b_dynamic_oracle_mag_625_hellaswag.yaml \
         pubsub_75_hs       qwen3_30b_a3b_dynamic_pubsub_75_hellaswag.yaml
# Wave 3 — oracle_mag 75 + pubsub 87.5
run_pair oracle_mag_75_hs qwen3_30b_a3b_dynamic_oracle_mag_75_hellaswag.yaml \
         pubsub_875_hs      qwen3_30b_a3b_dynamic_pubsub_875_hellaswag.yaml
# Wave 4 — oracle_mag 87.5 + MMLU pubsub 75
run_pair oracle_mag_875_hs qwen3_30b_a3b_dynamic_oracle_mag_875_hellaswag.yaml \
         pubsub_75_mmlu     qwen3_30b_a3b_dynamic_pubsub_75_mmlu.yaml
# Wave 5 — MMLU oracle_mag 75 + M4 beta1.5
run_pair oracle_mag_75_mmlu qwen3_30b_a3b_dynamic_oracle_mag_75_mmlu.yaml \
         pivchol_beta15_hs   qwen3_30b_a3b_dynamic_pivchol_beta15_50_hellaswag.yaml
# Wave 6 — M4 beta2 + beta3
run_pair pivchol_beta2_hs qwen3_30b_a3b_dynamic_pivchol_beta2_50_hellaswag.yaml \
         pivchol_beta3_hs qwen3_30b_a3b_dynamic_pivchol_beta3_50_hellaswag.yaml

echo "[sweep] ALL_WAVES_DONE"
