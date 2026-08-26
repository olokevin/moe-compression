#!/usr/bin/env bash
# `input_only` sweep -- ONE-PASS input sparsity at three used-parameter budgets.
#
# The method: gate/up read only the token's top-rho_input coordinates by |x| and
# that sparse intermediate IS the compute (no proxy, no exact re-read); down_proj
# is gathered to the pooled top-B channels. used = (2*rho_input + rho_channel)/3,
# and both knobs are set to the budget itself (symmetric split), so
#
#   cut      rho_in  rho_ch     B    reads/expert
#   -70.0%   0.3000  0.3000   1843    614
#   -75.0%   0.2500  0.2500   1536    512
#   -80.0%   0.2000  0.2000   1229    410
#
# Two arms per budget, which is the second question this sweep answers:
#   uni  input_alloc=uniform -- all K experts read the same coordinate set
#   rtr  input_alloc=router  -- pooled reads split across the token's K experts
#                               by g_e*|x_i|; identical cost and reads/token.
# Offline this is worth +2.4 to +3.6pt here (vs +0.75 to +1.1pt for the two-pass
# scorer), because a sparse read now corrupts the expert's VALUES, not just its
# ranking -- see scripts/input_only_error.py.
#
# Configs: scripts/gen_input_only_configs.py.
#
# GPU RULES (from scripts/run_probe_curve_sweep.sh, learned the hard way):
#   * MMLU 5-shot needs 4 GPUs. On 2x40GB it OOMs -- one run died at 99% after
#     ~10h. Never economize on MMLU.
#   * HellaSwag 0-shot is fine on 2 GPUs, so HS jobs are paired 2+2 on a 4-GPU
#     block and all six can be in flight at once across three blocks.
#
# The GPU set is split into blocks automatically -- 2 per HellaSwag job, 4 per MMLU
# job -- and each wave runs as many jobs concurrently as there are blocks. So an
# 8-GPU box runs 4 HellaSwag jobs at once, then 2 MMLU jobs at once.
#
# Usage:
#   GPUS=0,1,2,3,4,5,6,7 bash scripts/run_input_only_sweep.sh all
#   GPUS=0,1,2,3 CUTS=800 bash scripts/run_input_only_sweep.sh all
#   GPUS=0,1 bash scripts/run_input_only_sweep.sh hs          # HS only on 2 GPUs
# Note GPUS are indices *within* CUDA_VISIBLE_DEVICES if that is already set (the
# a100.sh launcher pins it), so 0,1,2,3.. is almost always what you want.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/src"
mkdir -p run_logs

GPUS="${GPUS:-0,1,2,3}"
MODE="${1:-all}"
CUTS="${CUTS:-700 750 800}"
ARMS="${ARMS:-uni rtr}"

IFS=',' read -r -a G <<< "$GPUS"
NG="${#G[@]}"
if { [ "$MODE" = "mmlu" ] || [ "$MODE" = "all" ]; } && [ "$NG" -lt 4 ]; then
  echo "[io] ERROR: MMLU needs 4 GPUs (2 OOMs at 99%); got $GPUS" >&2; exit 1
fi
if [ "$NG" -lt 2 ]; then
  echo "[io] ERROR: need >=2 GPUs, got $GPUS" >&2; exit 1
fi

# blocks(sz) -> space-separated CSV GPU blocks of size sz, e.g. "0,1 2,3 4,5 6,7"
blocks() {
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
  echo "[io] start $1 on gpu $3 @ $stamp"
  CUDA_VISIBLE_DEVICES=$3 .venv/bin/python src/train/merge_slim_eval.py \
    --config "configs/eval/$2" > "run_logs/${1}_${stamp}.log" 2>&1
  echo "[io] done  $1 (rc $?)"
}

# Run a work list in waves: as many concurrent jobs per wave as there are blocks.
run_list() {  # tag taskfile blocksize jobs...
  local tag="$1" taskfile="$2" sz="$3"; shift 3
  local BLK; BLK=$(blocks "$sz")
  local nb; nb=$(wc -w <<< "$BLK")
  echo "[io] $tag: $# job(s) over $nb block(s) of $sz GPU(s):$BLK"
  local pids j c a b p
  while [ "$#" -gt 0 ]; do
    pids=()
    for b in $BLK; do
      [ "$#" -gt 0 ] || break
      j="$1"; shift; c="${j%%:*}"; a="${j##*:}"
      run_one "io_${a}_cut${c}_${tag}" \
        "qwen3_30b_a3b_inputonly_${a}_cut${c}_${taskfile}.yaml" "$b" &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
  done
}

# the (cut, arm) work list, flattened
JOBS=""
for c in $CUTS; do for a in $ARMS; do JOBS="$JOBS ${c}:${a}"; done; done

# ---- HellaSwag: 2 GPUs per job ------------------------------------------- #
if [ "$MODE" = "hs" ] || [ "$MODE" = "all" ]; then
  run_list hs hellaswag 2 $JOBS
  echo "[io] HELLASWAG_DONE"
fi

# ---- MMLU: 4 GPUs per job ------------------------------------------------ #
if [ "$MODE" = "mmlu" ] || [ "$MODE" = "all" ]; then
  run_list mmlu mmlu 4 $JOBS
  echo "[io] MMLU_DONE"
fi

echo "[io] ALL_DONE"
