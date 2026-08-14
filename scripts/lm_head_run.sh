#!/usr/bin/env bash
# Thin runner for the lm_head baselines: sets PYTHONPATH (all imports are `src.*`)
# and picks the repo venv, then forwards every argument to scripts/lm_head_gates.py.
#
#   bash scripts/lm_head_run.sh --model Qwen/Qwen3-0.6B --gates
#   bash scripts/lm_head_run.sh --model Qwen/Qwen3-0.6B --ladder --ppl-tokens 262144
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
PY="${LMHEAD_PY:-/local/home/yequan/moe-compression/.venv/bin/python}"
export PYTHONPATH="$REPO"
export HF_HOME="${HF_HOME:-/home/yequan/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
exec "$PY" "$REPO/scripts/lm_head_gates.py" "$@"
