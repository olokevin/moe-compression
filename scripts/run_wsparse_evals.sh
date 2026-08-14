#!/usr/bin/env bash
# Run the unstructured-scorer evals on a busy box, politely.
#
# Meant to be started on the A100 host itself (nohup'd): it *waits* until enough
# GPUs are genuinely idle before starting, so it can be queued behind other people's
# (or one's own) multi-hour jobs without OOM-killing them. Then it runs the given
# eval configs sequentially on the GPUs it claimed.
#
#   NEED=4 MINFREE=30000 bash scripts/run_wsparse_evals.sh cfg_a.yaml cfg_b.yaml
#
# Every config here is at used-params = 0.125 + 2*0.1125/3 = 0.20, i.e. 20% of what
# the dense model activates including the scorer, with the 12.5% channel budget.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/src"
mkdir -p run_logs

NEED=${NEED:-4}
MINFREE=${MINFREE:-30000}          # MiB free per GPU; a 30B bf16 shard + activations
POLL=${POLL:-300}

pick_gpus() {
  nvidia-smi --query-gpu=index,memory.used,memory.total \
      --format=csv,noheader,nounits \
    | awk -F', *' -v m="$MINFREE" '($3 - $2) >= m {print $1}' \
    | head -n "$NEED" | paste -sd,
}

echo "[queue] waiting for $NEED GPUs with >= ${MINFREE}MiB free (poll ${POLL}s)"
while :; do
  GPUS=$(pick_gpus)
  N=$(echo "$GPUS" | tr ',' '\n' | grep -c '[0-9]' || true)
  if [ "${N:-0}" -ge "$NEED" ]; then break; fi
  sleep "$POLL"
done
echo "[queue] claimed GPUs $GPUS at $(date -u +%FT%TZ)"

for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  stamp=$(date +%m%d-%H%M%S)
  log="run_logs/${name}_${stamp}.log"
  echo "[queue] START $name -> $log ($(date -u +%FT%TZ))"
  CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/python src/train/merge_slim_eval.py \
      --config "configs/eval/${cfg}" > "$log" 2>&1
  echo "[queue] DONE  $name rc=$? ($(date -u +%FT%TZ))"
  grep -E "acc_norm|realized scoring density|USED PARAMS" "$log" | tail -5 || true
done
echo "[queue] ALL_DONE"
