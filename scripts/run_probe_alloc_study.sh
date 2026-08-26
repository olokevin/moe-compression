#!/usr/bin/env bash
# End-to-end evals for the input-ALLOCATION study (the doc's "Which input
# coordinates, and how many per expert?" section, which was offline-only for two of
# its four terms).
#
# Fills exactly the gaps. Already measured end-to-end at rho_input=0.25 -- do NOT
# re-run these:
#   rho_ch  uniform          router
#   0.10    74.06 / 77.20    74.64 / 77.67   (leaderboard rows 6, 7)
#   0.15    76.47 / 78.63    <- this script
#   0.20    76.72 / 79.10    76.61 / 79.45   (reuse_k25_r20 vs bp_cut633; iso, B=1229)
#
# Jobs:
#   router  @ rho_ch=0.15  x {hs, mmlu}   fills the middle of the trend -- LAUNCHED then
#                                         CANCELLED; still the default arm if rerun
#   router2 @ rho_ch=0.10  x {hs, mmlu}   available but DELIBERATELY NOT RUN: predicted
#   colnorm @ rho_ch=0.10  x {hs, mmlu}   below `router`, neither is a best-practice
#                                         candidate, and the ladder is well-calibrated
#                                         at this budget (router resid -0.08pt). Opt in
#                                         with ARMS if the question is revisited:
#                                           ARMS="router2_k25_r10 colnorm_k25_r10" ...
#
# Configs: scripts/gen_probe_alloc_configs.py. Harvest: scripts/probe_alloc_collect.sh.
#
# GPU RULES: MMLU 5-shot needs 4 GPUs (2 OOMs at 99% after ~10h); HellaSwag 0-shot is
# fine on 2. The GPU set is split into blocks accordingly and each wave runs as many
# jobs concurrently as there are blocks.
#
# Usage:
#   GPUS=0,1,2,3,4,5,6,7 bash scripts/run_probe_alloc_study.sh all
#   GPUS=0,1,2,3,4,5 bash scripts/run_probe_alloc_study.sh hs     # the 3 HellaSwag jobs
#   GPUS=0,1,2,3,4,5,6,7 bash scripts/run_probe_alloc_study.sh mmlu
# GPUS are indices *within* CUDA_VISIBLE_DEVICES when the a100.sh launcher pins it,
# so 0,1,2,.. is almost always what you want.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/src"
mkdir -p run_logs

GPUS="${GPUS:-0,1,2,3}"
MODE="${1:-all}"
# config tags, in run order. router2/colnorm are intentionally NOT in the default --
# opt in explicitly (see the header).
ARMS="${ARMS:-rtr_k25_r15}"

IFS=',' read -r -a G <<< "$GPUS"
NG="${#G[@]}"
if { [ "$MODE" = "mmlu" ] || [ "$MODE" = "all" ]; } && [ "$NG" -lt 4 ]; then
  echo "[alloc] ERROR: MMLU needs 4 GPUs (2 OOMs at 99%); got $GPUS" >&2; exit 1
fi
if [ "$NG" -lt 2 ]; then
  echo "[alloc] ERROR: need >=2 GPUs, got $GPUS" >&2; exit 1
fi

blocks() {  # sz -> space-separated CSV GPU blocks
  local sz="$1" i j b out=""
  for ((i = 0; i + sz <= NG; i += sz)); do
    b="${G[i]}"
    for ((j = 1; j < sz; j++)); do b="$b,${G[i + j]}"; done
    out="$out $b"
  done
  echo "$out"
}

run_one() {  # name cfg gpus
  local stamp; stamp=$(date +%m%d-%H%M%S)
  echo "[alloc] start $1 on gpu $3 @ $stamp"
  CUDA_VISIBLE_DEVICES=$3 .venv/bin/python src/train/merge_slim_eval.py \
    --config "configs/eval/$2" > "run_logs/${1}_${stamp}.log" 2>&1
  echo "[alloc] done  $1 (rc $?)"
}

run_list() {  # tag taskfile blocksize arms...
  local tag="$1" taskfile="$2" sz="$3"; shift 3
  local BLK; BLK=$(blocks "$sz")
  echo "[alloc] $tag: $# job(s) over $(wc -w <<< "$BLK") block(s) of $sz GPU(s):$BLK"
  local pids a b p
  while [ "$#" -gt 0 ]; do
    pids=()
    for b in $BLK; do
      [ "$#" -gt 0 ] || break
      a="$1"; shift
      run_one "palloc_${a}_${tag}" \
        "qwen3_30b_a3b_probe_alloc_${a}_${taskfile}.yaml" "$b" &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
  done
}

if [ "$MODE" = "hs" ] || [ "$MODE" = "all" ]; then
  run_list hs hellaswag 2 $ARMS
  echo "[alloc] HELLASWAG_DONE"
fi
if [ "$MODE" = "mmlu" ] || [ "$MODE" = "all" ]; then
  run_list mmlu mmlu 4 $ARMS
  echo "[alloc] MMLU_DONE"
fi
echo "[alloc] ALL_DONE"
