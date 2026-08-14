#!/usr/bin/env bash
# Unstructured vs column-structured scoring at an identical read budget.
#
# Every row below reads the SAME number of (channel, coordinate) entries per token
# per branch, so the comparison is iso-cost by construction. The reference density
# is 0.1125, which with a 12.5% channel budget is exactly
#     used = 0.125 + 2*0.1125/3 = 0.20
# i.e. 20% of what the dense model activates, scorer included. Two extra densities
# (0.15 / 0.075) let the same 20% total be re-split against rho_channel 0.10 / 0.15.
#
# One layer per GPU (4 L4s or 4 A100s), then merged. ~5 min/layer.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
OUT=docs/results/idea_pilot
mkdir -p "$OUT" run_logs

# the converged threshold ladder: geometric ratio 0.7, 8 levels (1.0 -> 0.082).
# Measured within 0.04pt of the exact |W_ji*x_i| top-N rule (scripts/wsparse_screen.py
# --variants prod:...), at 8 masked GEMMs instead of a per-token (I,H) mask.
LADDER="1.0|0.7|0.49|0.343|0.24|0.168|0.118|0.082"

VARIANTS="exact,random,\
stair:0.1125x1.0,\
stair:0.225x0.5,\
stair:0.25x0.45,\
stair:0.5x0.225,\
stair:1.0x0.1125,\
geo:6:0.1125,\
glob:0.1125,\
tau:0.1125:${LADDER},\
tau:0.1125:${LADDER}:router,\
tau:0.1125:${LADDER}:mf,\
stair:0.1125x1.0:router,\
stair:0.1125x1.0:mf,\
prod:0.1125,\
stair:0.15x1.0,\
tau:0.15:${LADDER},\
stair:0.075x1.0,\
tau:0.075:${LADDER}"

i=0
for L in 6 22 38 46; do
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH="$(pwd)" .venv/bin/python scripts/wsparse_screen.py \
      --layers "$L" --max-tokens "${TOKENS:-4096}" --chunk 256 \
      --rhos 0.10,0.125,0.15 --variants "$VARIANTS" --prod-tokens 1024 \
      --out "$OUT/wsparse_screen_L${L}.json" \
      > "run_logs/wsparse_screen_L${L}.log" 2>&1 &
  i=$((i + 1))
done
wait

.venv/bin/python scripts/wsparse_screen.py --merge \
    "$OUT"/wsparse_screen_L{6,22,38,46}.json --out "$OUT/wsparse_screen.json"
