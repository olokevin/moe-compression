#!/usr/bin/env bash
# 0.5B mixed-compression sweep: attention (qkvo) left dense, MLP compressed at
# ratio 0.8 under four methods, one job per GPU. Each job trains 1000 CE steps on
# C4 and benchmarks MMLU (5% subset) + C4/WikiText PPL + hellaswag right after
# compression (step 0) and every 200 steps, logging to W&B project
# yequan-train_aware-05B.
#
# Usage:  bash scripts/compress_then_train_05B_sweep.sh
# Logs:   /tmp/ctt_05B_<method>.log   (one per job)
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$(pwd)"

# This host has an NVML driver/library version mismatch. PyTorch's default
# expandable-segments allocator calls nvmlInit and crashes during the backward
# pass (hit by the *_combined methods' backward covariance collection). Disable
# it so allocation avoids the NVML code path.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# Prefer the project venv interpreter (repo uses uv; there is no system torch).
PYTHON="$(pwd)/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="python"

METHODS=(nystrom nystrom_combined btt_llm_v2 btt_llm_v2_combined)

pids=()
for i in "${!METHODS[@]}"; do
    method="${METHODS[$i]}"
    gpu="$i"
    config="configs/compress_then_train/05B_mlp_${method}.yaml"
    log="/tmp/ctt_05B_${method}.log"
    echo "Launching method=${method} on GPU ${gpu}  (config=${config}, log=${log})"
    CUDA_VISIBLE_DEVICES="${gpu}" HF_ALLOW_CODE_EVAL=1 \
        nohup "${PYTHON}" src/compress_then_train.py --config "${config}" \
        > "${log}" 2>&1 &
    pids+=($!)
done

echo "Launched ${#pids[@]} jobs: PIDs ${pids[*]}"
echo "Waiting for all jobs to finish ..."
fail=0
for pid in "${pids[@]}"; do
    wait "${pid}" || fail=1
done
echo "All jobs done (fail=${fail})."
exit "${fail}"
