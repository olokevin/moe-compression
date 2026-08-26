#!/usr/bin/env bash
# Harvest the `input_only` sweep evals off the A100 boxes into one JSON + table.
#
# Only *finished* runs contribute: lm-eval writes its result JSON at the end, so an
# in-flight or OOM-killed run is simply absent rather than silently half-counted.
# Handles both result-JSON layouts in this repo (metrics at top level keyed by task,
# and nested under a "results" key).
#
# Prints a markdown-ready table with the offline pre-registered predictions and the
# residual, so a transferred / non-transferred ladder call is immediate. The
# predictions are the doc's: anchor rel_err 0.4121 <-> HS 74.08 / MMLU 77.77
# (two-pass input_sparse 0.1875/0.125, doc row 7b), slopes -26.4 / -20.3.
#
# Usage: bash scripts/input_only_collect.sh
set -u
cd "$(dirname "$0")/.."
OUT=docs/results/idea_pilot/input_only_evals.json
mkdir -p "$(dirname "$OUT")"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

for h in A100-3 A100-Sagemaker A100-New; do
  echo "[collect] $h"
  ssh -o ConnectTimeout=15 "$h" 'cd ~/yequan/moe-compression/results_eval 2>/dev/null && \
    find . -path "*inputonly*" -name "*results.json" 2>/dev/null | while read -r j; do
        echo "===FILE=== $j"
        cat "$j"
      done' > "$TMP/$h.raw" 2>/dev/null || echo "  (unreachable)"
done

python3 - "$TMP" "$OUT" docs/results/idea_pilot/input_only_error.json <<'PY'
import json, os, re, sys

tmp, out, screen_path = sys.argv[1], sys.argv[2], sys.argv[3]

# offline rel_err per (cut, arm), from the iso-cost screen at the symmetric split
REL = {}
try:
    s = json.load(open(screen_path))["summary"]
    for x in s:
        if (x["method"] == "input_only"
                and abs(x["rho_input"] - x["rho_channel"]) < 1e-9):
            cut = round(100 * (1 - x["used"]), 1)
            REL[(cut, "router" if x["alloc"] == "router" else "uniform")] = x["rel_err"]
except (OSError, KeyError):
    pass

ANCHOR_REL = 0.4121                       # two-pass input_sparse 0.1875/0.125
ANCHOR = {"hellaswag": 74.08, "mmlu": 77.77}
SLOPE = {"hellaswag": -26.4, "mmlu": -20.3}
ARM = {"uni": "uniform", "rtr": "router"}

acc = {}
for host in os.listdir(tmp):
    for chunk in open(os.path.join(tmp, host)).read().split("===FILE=== ")[1:]:
        nl = chunk.find("\n")
        path, blob = chunk[:nl].strip(), chunk[nl:]
        m = re.search(r"inputonly_(uni|rtr)_cut(\d{3})_", path)
        if not m:
            continue
        arm, tag = ARM[m.group(1)], m.group(2)
        cut = float(f"{tag[:2]}.{tag[2]}")
        try:
            d = json.loads(blob)
        except json.JSONDecodeError:
            continue
        r = d.get("results", d)
        for task, metric in (("hellaswag", "acc_norm,none"), ("mmlu", "acc,none")):
            v = r.get(task, {}).get(metric)
            if isinstance(v, float):
                key = f"{cut}|{arm}|{task}"
                val = round(100 * v, 2)
                if key in acc and abs(acc[key] - val) > 1e-9:
                    print(f"  WARNING: {key} has two values {acc[key]} vs {val}")
                acc[key] = val

json.dump(acc, open(out, "w"), indent=2, sort_keys=True)

DENSE = {"hellaswag": 78.56, "mmlu": 80.91}
print()
print("| cut | arm | rho | rel_err | HellaSwag | pred | resid | MMLU | pred | resid |")
print("|---|---|---|---|---|---|---|---|---|---|")
n = 0
for cut in (70.0, 75.0, 80.0):
    for arm in ("uniform", "router"):
        rel = REL.get((cut, arm))
        cells = []
        for task in ("hellaswag", "mmlu"):
            v = acc.get(f"{cut}|{arm}|{task}")
            if rel is None:
                cells += ["?" if v is None else f"{v:.2f}", "-", "-"]
                continue
            pred = ANCHOR[task] + SLOPE[task] * (rel - ANCHOR_REL)
            if v is None:
                cells += ["*running*", f"{pred:.1f}", "-"]
            else:
                n += 1
                cells += [f"**{v:.2f}** ({v - DENSE[task]:+.2f})",
                          f"{pred:.1f}", f"{v - pred:+.2f}"]
        rho = round(1 - cut / 100, 4)
        rs = "-" if rel is None else f"{rel:.4f}"
        print(f"| -{cut:.1f}% | `{arm}` | {rho} | {rs} | " + " | ".join(cells) + " |")
print(f"\n{n}/12 evals complete -> {out}")

# the two headline deltas, as soon as both arms of a budget are in
print()
for cut in (70.0, 75.0, 80.0):
    for task in ("hellaswag", "mmlu"):
        u = acc.get(f"{cut}|uniform|{task}")
        r = acc.get(f"{cut}|router|{task}")
        if u is not None and r is not None:
            print(f"  router gain, -{cut:.1f}% {task:9s}: {r - u:+.2f}pt "
                  f"({u:.2f} -> {r:.2f})")
PY
