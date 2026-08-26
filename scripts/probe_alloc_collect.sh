#!/usr/bin/env bash
# Harvest the input-ALLOCATION study evals (router2 / colnorm / router@rho_ch=0.15).
#
# Prints the four-term table with the already-measured uniform and router rows folded
# in from the earlier runs, plus the pre-registered predictions and residuals.
# Only finished runs contribute (lm-eval writes its JSON at the end).
#
# Usage: bash scripts/probe_alloc_collect.sh
set -u
cd "$(dirname "$0")/.."
OUT=docs/results/idea_pilot/probe_alloc_evals.json
mkdir -p "$(dirname "$OUT")"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

for h in A100-New A100-3 A100-Sagemaker; do
  echo "[collect] $h"
  ssh -o ConnectTimeout=15 "$h" 'cd ~/yequan/moe-compression/results_eval 2>/dev/null && \
    find . -path "*probe_alloc_*" -name "*results.json" 2>/dev/null | while read -r j; do
        echo "===FILE=== $j"; cat "$j"; done' > "$TMP/$h.raw" 2>/dev/null || echo "  (unreachable)"
done

python3 - "$TMP" "$OUT" <<'PY'
import json, os, re, sys
tmp, out = sys.argv[1], sys.argv[2]
# tag -> (alloc, rho_channel, offline rel_err)
META = {"router2_k25_r10": ("router2", 0.10, 0.4272),
        "colnorm_k25_r10": ("colnorm", 0.10, 0.4429),
        "rtr_k25_r15":     ("router",  0.15, 0.3592)}
# already measured elsewhere: (alloc, rho_ch) -> (HS, MMLU)
KNOWN = {("uniform", 0.10): (74.06, 77.20), ("router", 0.10): (74.64, 77.67),
         ("uniform", 0.15): (76.47, 78.63),
         ("uniform", 0.20): (76.72, 79.10), ("router", 0.20): (76.61, 79.45)}
# anchors for the ladder, per rho_channel: uniform's (rel_err, HS, MMLU)
ANCH = {0.10: (0.4434, 74.06, 77.20), 0.15: (0.3820, 76.47, 78.63)}
SLOPE = {"hellaswag": -26.4, "mmlu": -20.3}

acc = {}
for host in os.listdir(tmp):
    for chunk in open(os.path.join(tmp, host)).read().split("===FILE=== ")[1:]:
        nl = chunk.find("\n"); path, blob = chunk[:nl].strip(), chunk[nl:]
        m = re.search(r"probe_alloc_(\w+?)_(hellaswag|mmlu)_\d", path)
        if not m or m.group(1) not in META:
            continue
        tag, task = m.group(1), m.group(2)
        try: d = json.loads(blob)
        except json.JSONDecodeError: continue
        r = d.get("results", d)
        v = r.get(task, {}).get("acc_norm,none" if task == "hellaswag" else "acc,none")
        if isinstance(v, float): acc[f"{tag}|{task}"] = round(100 * v, 2)
json.dump(acc, open(out, "w"), indent=2, sort_keys=True)

n = 0
for rho in (0.10, 0.15):
    print(f"\n### rho_input=0.25, rho_channel={rho:g}  (cut {100*(1-(rho+2*0.25/3)):.1f}%)")
    print("| term | rel_err | HellaSwag | pred | resid | MMLU | pred | resid |")
    print("|---|---|---|---|---|---|---|---|")
    a_rel, a_hs, a_mm = ANCH[rho]
    rows = [("uniform", a_rel, KNOWN.get(("uniform", rho)))]
    for tag, (alloc, r, rel) in META.items():
        if r == rho: rows.append((alloc, rel, (acc.get(f"{tag}|hellaswag"), acc.get(f"{tag}|mmlu"))))
    if ("router", rho) in KNOWN and not any(x[0] == "router" for x in rows):
        rows.append(("router", 0.4183 if rho == 0.10 else None, KNOWN[("router", rho)]))
    elif ("router", rho) in KNOWN and rho == 0.10:
        rows.insert(1, ("router", 0.4183, KNOWN[("router", rho)]))
    for alloc, rel, vals in rows:
        cells = []
        for i, task in enumerate(("hellaswag", "mmlu")):
            v = None if vals is None else vals[i]
            base = a_hs if task == "hellaswag" else a_mm
            pred = None if rel is None else base + SLOPE[task] * (rel - a_rel)
            if v is None:
                cells += ["*running*", "-" if pred is None else f"{pred:.1f}", "-"]
            else:
                if alloc not in ("uniform", "router"): n += 1
                cells += [f"**{v:.2f}**", "-" if pred is None else f"{pred:.1f}",
                          "-" if pred is None else f"{v - pred:+.2f}"]
        rs = "-" if rel is None else f"{rel:.4f}"
        print(f"| `{alloc}` | {rs} | " + " | ".join(cells) + " |")
print(f"\n{n}/6 new evals complete -> {out}")
PY
