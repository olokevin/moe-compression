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
    find . \( -path "*probe_bp_cut*" -o -path "*probe_router_split_opt*" \) \
      -name "*results.json" 2>/dev/null | while read -r j; do
        echo "===FILE=== $j"
        cat "$j"
      done' > "$TMP/$h.raw" 2>/dev/null || echo "  (unreachable)"
done

python3 - "$TMP" "$OUT" <<'PY'
import json, os, re, sys

tmp, out = sys.argv[1], sys.argv[2]

# config tag -> used-param cut %. bp_cut<tag> encodes it; split_opt is the -75.0% point.
def cut_of(path):
    m = re.search(r"probe_bp_cut(\d{3})_", path)
    if m:
        t = m.group(1)
        return float(f"{t[:2]}.{t[2]}")
    if "probe_router_split_opt" in path:
        return 75.0
    return None

acc = {"hellaswag": {}, "mmlu": {}}
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
        for task, metric in (("hellaswag", "acc_norm,none"), ("mmlu", "acc,none")):
            if task in r and isinstance(r[task].get(metric), float):
                val = round(100 * r[task][metric], 2)
                prev = acc[task].get(cut)
                if prev is not None and abs(prev - val) > 1e-9:
                    print(f"  WARNING: {task} {cut}% has two values {prev} vs {val} ({path})")
                acc[task][cut] = val

with open(out, "w") as f:
    json.dump({"hellaswag": {f"{k}": v for k, v in sorted(acc["hellaswag"].items())},
               "mmlu": {f"{k}": v for k, v in sorted(acc["mmlu"].items())}}, f, indent=2)

for task in ("hellaswag", "mmlu"):
    got = sorted(acc[task].items())
    print(f"{task:10s} {len(got)}/6 measured: " +
          ", ".join(f"-{k}%={v}" for k, v in got))
print(f"wrote {out}")
PY
