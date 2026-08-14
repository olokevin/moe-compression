#!/usr/bin/env bash
# Runner for scripts/lm_head_sweep.py -- many head treatments, one model load.
#   bash scripts/lm_head_sweep.sh --model Qwen/Qwen3-0.6B --tasks c4 --limit 20
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
PY="${LMHEAD_PY:-$REPO/.venv/bin/python}"
[[ -x "$PY" ]] || PY="/local/home/yequan/moe-compression/.venv/bin/python"
export PYTHONPATH="$REPO"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
exec "$PY" "$REPO/scripts/lm_head_sweep.py" "$@"
