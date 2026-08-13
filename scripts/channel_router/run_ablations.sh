#!/usr/bin/env bash
# §3.2 ablation matrix — one Stage-B run per row, identical token/step budget so the
# rows are comparable to each other (not to the full-budget main run, which trains
# ~3.5x longer; the main row is re-run here at the short budget as the reference).
#
# Usage: bash scripts/channel_router/run_ablations.sh <group>   # group = a|b|c
# Each group is sized for one GPU; run the three groups on three GPUs in parallel.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--layers 46 --epochs 3 --fit-tokens 262144 --eval-every 300 --eval-tokens 8192 \
  --out-dir docs/results/channel_router/ablate"

run() { echo "=== $*"; $PY scripts/channel_router/train_stage_b.py $COMMON "$@"; }

case "${1:-a}" in
  a)  # reference + head family
    run --name ab_ref            --r 32 --m 16 --head bilinear --loss margin
    run --name ab_head_abs       --r 32 --m 16 --head abs      --loss margin
    run --name ab_head_linear    --r 32 --m 16 --head linear   --loss margin
    ;;
  b)  # structure switches
    run --name ab_m0             --r 32 --m 0  --head bilinear --loss margin
    run --name ab_init_random    --r 32 --m 16 --head bilinear --loss margin --init random
    run --name ab_no_bias        --r 32 --m 16 --head bilinear --loss margin --no-bias
    run --name ab_no_g           --r 32 --m 16 --head bilinear --loss margin --no-g
    run --name ab_bias_freq      --r 32 --m 16 --head bilinear --loss margin --bias-init both
    ;;
  c)  # loss family + rank sweep
    run --name ab_loss_bce       --r 32 --m 16 --head bilinear --loss bce
    run --name ab_loss_listwise  --r 32 --m 16 --head bilinear --loss listwise
    run --name ab_allpairs       --r 32 --m 16 --head bilinear --loss margin --delta 2048 --pairs 8192
    run --name ab_r16            --r 16 --m 16 --head bilinear --loss margin
    run --name ab_r64            --r 64 --m 16 --head bilinear --loss margin
    ;;
esac
echo "ablation group ${1:-a} done"
