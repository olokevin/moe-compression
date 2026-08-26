#!/usr/bin/env bash
# Harvest the input_sparse curve evals off the A100 boxes into one JSON.
#
# Writes docs/results/btt_dynamic/probe_curve.json, which scripts/probe_curve_plot.py
# reads and merges over its literal fallbacks. Only *finished* runs contribute -- a
# result file exists only after lm-eval writes it, so an in-flight or OOM-killed run
# is simply absent rather than silently half-counted.
#
# Handles both result-JSON layouts seen in this repo: metrics at the top level keyed
# by task, and nested under a "results" key.
#
# Usage: bash scripts/probe_curve_collect.sh
set -u
cd "$(dirname "$0")/.."
OUT=docs/results/btt_dynamic/probe_curve.json
mkdir -p "$(dirname "$OUT")"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

for h in A100-New A100-3 A100-Sagemaker; do
  echo "[collect] $h"
  # bp_cut* = the best-practice curve; split_opt = the -75.0% point (same settings)
  ssh -o ConnectTimeout=15 "$h" 'cd ~/yequan/moe-compression/results_eval 2>/dev/null && \
    find . \( -path "*probe_bp_cut*" -o -path "*probe_router_split_opt*" \
               -o \( -path "*dynamic_oracle_mag_noW*" -a ! -path "*topk4*" \) \) \
      -name "*results.json" 2>/dev/null | while read -r j; do
        echo "===FILE=== $j"
        cat "$j"
      done' > "$TMP/$h.raw" 2>/dev/null || echo "  (unreachable)"
done

python3 - "$TMP" "$OUT" <<'PY'
import json, os, re, sys

tmp, out = sys.argv[1], sys.argv[2]

# config tag -> used-param cut %. bp_cut<tag> encodes it; split_opt is the -75.0% point.
# oracle_mag_noW runs gate+up at FULL width to score, so kept=(1+1+rho)/3 and the
# used-param cut is 100*(1-(2+rho)/3) -- NOT the nominal channel cut in the dir name.
ONOW_CUT = {"50": 16.67, "625": 20.83, "75": 25.00, "875": 29.17}

def cut_of(path):
    m = re.search(r"probe_bp_cut(\d{3})_", path)
    if m:
        t = m.group(1)
        return float(f"{t[:2]}.{t[2]}")
    if "probe_router_split_opt" in path:
        return 75.0
    m = re.search(r"dynamic_oracle_mag_noW_(\d+)_", path)
    if m:
        return ONOW_CUT.get(m.group(1))
    return None

def is_oracle(path):
    return "dynamic_oracle_mag_noW" in path

acc = {"hellaswag": {}, "mmlu": {},
       "oracle_now_hellaswag": {}, "oracle_now_mmlu": {}}
for host in os.listdir(tmp):
    raw = open(os.path.join(tmp, host)).read()
    # split on the sentinel; each chunk is "<path>\n<json>"
    for chunk in raw.split("===FILE=== ")[1:]:
        nl = chunk.find("\n")
        path, blob = chunk[:nl].strip(), chunk[nl:]
        cut = cut_of(path)
        if cut is None:
            continue
        try:
            d = json.loads(blob)
        except json.JSONDecodeError:
            continue
        r = d.get("results", d)
        pre = "oracle_now_" if is_oracle(path) else ""
        for task, metric in (("hellaswag", "acc_norm,none"), ("mmlu", "acc,none")):
            if task in r and isinstance(r[task].get(metric), float):
                val = round(100 * r[task][metric], 2)
                key = pre + task
                prev = acc[key].get(cut)
                if prev is not None and abs(prev - val) > 1e-9:
                    print(f"  WARNING: {key} {cut}% has two values {prev} vs {val} ({path})")
                acc[key][cut] = val

with open(out, "w") as f:
    json.dump({k: {f"{c}": v for c, v in sorted(d.items())} for k, d in acc.items()},
              f, indent=2)

for task in ("hellaswag", "mmlu"):
    got = sorted(acc[task].items())
    print(f"{task:10s} {len(got)}/6 measured: " +
          ", ".join(f"-{k}%={v}" for k, v in got))
for task in ("oracle_now_hellaswag", "oracle_now_mmlu"):
    got = sorted(acc[task].items())
    print(f"{task:22s} {len(got)}/4 measured: " +
          ", ".join(f"-{k}%={v}" for k, v in got))
print(f"wrote {out}")
PY
