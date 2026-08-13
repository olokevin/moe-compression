#!/usr/bin/env bash
# Tail of the lowrank_scorer sweep: the 4 configs NOT covered by wave 1 of
# run_lowrank_scorer_sweep.sh (which runs svd_r32 up+gate / up-only on 4 GPUs
# each). Waits for those two to exit, then runs all 4 remaining jobs
# **concurrently on 2 GPUs each** — the 30B fits in 2x40GB (verified), so 4-way
# concurrency replaces two sequential 2-job waves and roughly halves wall clock.
#
#   svd_r16      up+gate (gpu 0,1)   svd_r16      up-only (gpu 2,3)
#   btt_m2n2_r32 up+gate (gpu 4,5)   btt_m2n2_r32 up-only (gpu 6,7)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/src"
export WANDB_MODE=disabled
mkdir -p run_logs

# Wait for any in-flight merge_slim_eval (wave 1) to finish so we don't
# oversubscribe the GPUs.
while pgrep -f "merge_slim_eval.py --config configs/eval/qwen3_30b_a3b_dynamic_lowrank" \
        > /dev/null 2>&1; do
  echo "[tail] waiting for wave-1 jobs to finish ($(date +%H:%M:%S))"
  sleep 120
done
echo "[tail] GPUs free, launching 4 concurrent jobs @ $(date +%H:%M:%S)"

P=qwen3_30b_a3b_dynamic_lowrank
stamp=$(date +%m%d-%H%M%S)
pids=()

launch() {  # name gpus cfg
  CUDA_VISIBLE_DEVICES=$2 .venv/bin/python src/train/merge_slim_eval.py \
    --config "configs/eval/$3" > "run_logs/${1}_${stamp}.log" 2>&1 &
  pids+=($!)
  echo "[tail] launched $1 on GPU $2 (pid ${pids[-1]})"
}

launch svd_r16_upgate  0,1 ${P}_svd_r16_upgate_75_hellaswag.yaml
launch svd_r16_uponly  2,3 ${P}_svd_r16_uponly_75_hellaswag.yaml
launch btt_r32_upgate  4,5 ${P}_btt_m2n2_r32_upgate_75_hellaswag.yaml
launch btt_r32_uponly  6,7 ${P}_btt_m2n2_r32_uponly_75_hellaswag.yaml

fail=0
for p in "${pids[@]}"; do
  wait "$p" || { echo "[tail] pid $p FAILED (rc $?)"; fail=1; }
done
echo "[tail] ALL_TAIL_DONE (fail=$fail)"
