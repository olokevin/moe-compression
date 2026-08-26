#!/usr/bin/env bash
# Fill the oracle_mag_noW ceiling sweep for the presentation curve figures.
#
# oracle_mag_noW is the FULL-WIDTH scoring ceiling: it ranks channels by the exact
# g_e*|SiLU(gate)*up| (no ||W_down|| factor), so gate+up must run at full width just
# to decide. Its used-param cut is therefore (1+1+rho)/3 kept -> floors at -33.3%
# however deep the channel cut goes:
#
#   nominal channel cut   rho     used-param cut
#   -50.0%                0.500   -16.67%
#   -62.5%                0.375   -20.83%
#   -75.0%                0.250   -25.00%   (already measured)
#   -87.5%                0.125   -29.17%   (already measured)
#
# This runs the two missing rungs (-50%, -62.5%) on both benchmarks so the dashed
# ceiling curve spans the same nominal grid as fig_offline_vs_online.pdf.
#
# MMLU needs 4 GPUs (2x40GB OOMs); HellaSwag is fine on 2.
# Usage: GPUS=0,1,2,3 bash scripts/run_oracle_now_sweep.sh [hs|mmlu|all]
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/src"
mkdir -p run_logs
GPUS="${GPUS:-0,1,2,3}"
MODE="${1:-all}"
TAGS="${TAGS:-50 625}"

run_one() {
  local stamp; stamp=$(date +%m%d-%H%M%S)
  echo "[onow] start $1 on gpu $3 @ $stamp"
  CUDA_VISIBLE_DEVICES=$3 .venv/bin/python src/train/merge_slim_eval.py \
    --config "configs/eval/$2" > "run_logs/${1}_${stamp}.log" 2>&1
  echo "[onow] done  $1 (rc $?)"
}

if [ "$MODE" = "hs" ] || [ "$MODE" = "all" ]; then
  for t in $TAGS; do
    run_one "onow${t}_hs" "qwen3_30b_a3b_dynamic_oracle_mag_noW_${t}_hellaswag.yaml" "$GPUS"
  done
fi
if [ "$MODE" = "mmlu" ] || [ "$MODE" = "all" ]; then
  for t in $TAGS; do
    run_one "onow${t}_mmlu" "qwen3_30b_a3b_dynamic_oracle_mag_noW_${t}_mmlu.yaml" "$GPUS"
  done
fi
echo "[onow] ALL_DONE"
