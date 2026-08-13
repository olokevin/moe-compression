#!/usr/bin/env bash
# Train one router per MoE layer (all 48) so the full-model ΔPPL of §3.1.2 can be
# measured. Layers are split into groups; run one group per GPU in parallel, then
# merge_router_artifacts.py combines them into a single artifact.
#
# Usage: bash scripts/channel_router/run_all_layers.sh <group_index> <n_groups> [ratio]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GI=${1:-0}
NG=${2:-6}
RATIO=${3:-0.25}
TOKENS=${TOKENS:-524288}

LAYERS=""
for ((l = GI; l < 48; l += NG)); do
  LAYERS="${LAYERS:+$LAYERS,}$l"
done
echo "group $GI/$NG -> layers $LAYERS (ratio $RATIO)"

$PY scripts/channel_router/train_stage_b.py \
  --layers "$LAYERS" --tokens "$TOKENS" --ratio "$RATIO" \
  --r 32 --m 16 --head bilinear --loss margin \
  --epochs 4 --fit-tokens 262144 --batch 512 --eval-every 1000 --eval-tokens 4096 \
  --out-dir docs/results/channel_router/all_layers \
  --name "all_g${GI}_rho${RATIO}"
echo "group $GI done"
