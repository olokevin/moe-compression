#!/usr/bin/env bash
# Wave 1 of the reuse-probe study: no quantization (the probe reads the *served*
# up/gate weights), input sparsity p=25% for scoring, channel keep rho in
# {10%, 15%, 20%} for the actual compute. In the reuse frame
# (kept = rho + 2*p*(1-rho)/3) those are -75.0% / -70.8% / -66.7% used-parameter
# cuts, with zero extra storage.
#
# 6 jobs (3 budgets x {HellaSwag 0-shot, full MMLU 5-shot}) in 3 waves of 2,
# 4 GPUs each. Nothing to warm offline: the probe aliases the served weights and
# needs no scores artifact. Run from repo root on the box.
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

# Wave 1 — the target operating point (-75.0%) on both benchmarks
run_pair reuse_r10_hs   qwen3_30b_a3b_probe_reuse_k25_r10_hellaswag.yaml \
         reuse_r10_mmlu qwen3_30b_a3b_probe_reuse_k25_r10_mmlu.yaml
# Wave 2 — -70.8%
run_pair reuse_r15_hs   qwen3_30b_a3b_probe_reuse_k25_r15_hellaswag.yaml \
         reuse_r15_mmlu qwen3_30b_a3b_probe_reuse_k25_r15_mmlu.yaml
# Wave 3 — -66.7%
run_pair reuse_r20_hs   qwen3_30b_a3b_probe_reuse_k25_r20_hellaswag.yaml \
         reuse_r20_mmlu qwen3_30b_a3b_probe_reuse_k25_r20_mmlu.yaml

echo "[sweep] ALL_WAVES_DONE"
