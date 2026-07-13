#!/usr/bin/env python
"""Summarize the 0.5B compress-then-train sweep into a markdown report.

Reads the benchmark_comparison.json written by each run under
outputs/compress_then_train/ and emits docs/results/compress_then_train/05B.md
with (a) a headline before/after table across the four methods and (b) the
per-200-step MMLU/PPL curves.
"""
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_MD = os.path.join(REPO, "docs/results/compress_then_train/05B.md")

# (method key in run-dir name, display label)
METHODS = [
    ("mlp-nystrom-0.8", "nystrom"),
    ("mlp-nystrom_combined-0.8", "nystrom_combined"),
    ("mlp-btt_llm_v2-0.8", "btt_llm_v2"),
    ("mlp-btt_llm_v2_combined-0.8", "btt_llm_v2_combined"),
]

KEYS = ["ppl/c4", "ppl/wikitext2", "lm_eval/hellaswag", "lm_eval/mmlu"]


def _find_run(method_key):
    pat = os.path.join(REPO, "outputs/compress_then_train", f"05B_{method_key}_*", "benchmark_comparison.json")
    matches = sorted(glob.glob(pat))
    return matches[-1] if matches else None


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def main():
    data = {}
    for method_key, label in METHODS:
        path = _find_run(method_key)
        if path is None:
            data[label] = None
            continue
        with open(path) as f:
            data[label] = json.load(f)

    lines = []
    lines.append("# Qwen2.5-0.5B — mixed-compression sweep (attention dense, MLP @ ratio 0.8)")
    lines.append("")
    lines.append("Setting: attention (`q/k/v/o_proj`) left **uncompressed**; every layer's MLP "
                 "(`gate/up/down_proj`) compressed to **80% params retained** under four methods. "
                 "C4-calibrated compression -> continue-train 1000 CE steps on C4. "
                 "Eval: MMLU (5% subset, 5-shot), HellaSwag (5% subset), C4/WikiText-2 PPL, "
                 "right after compression (step 0) and every 200 steps. "
                 "W&B project `yequan-train_aware-05B`.")
    lines.append("")

    # Baseline (same uncompressed model for all) — take from first available run.
    base = next((d["before_compression"] for d in data.values() if d and d.get("before_compression")), None)
    if base:
        lines.append("**Uncompressed baseline:** "
                     + ", ".join(f"{k.split('/')[-1]}={_fmt(base.get(k))}" for k in KEYS))
        lines.append("")

    # Headline table: post-compression (step 0) vs final (after 1000 steps).
    lines.append("## Results")
    lines.append("")
    lines.append("| method | MMLU post-comp | MMLU final | HellaSwag final | C4 PPL post-comp | C4 PPL final | WikiText2 PPL final |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, label in METHODS:
        d = data.get(label)
        if not d:
            lines.append(f"| {label} | (run missing) | | | | | |")
            continue
        hist = d.get("history", {})
        step0 = hist.get("0", {})
        final = d.get("after_compress_train", {})
        lines.append(
            f"| {label} "
            f"| {_fmt(step0.get('lm_eval/mmlu'))} "
            f"| {_fmt(final.get('lm_eval/mmlu'))} "
            f"| {_fmt(final.get('lm_eval/hellaswag'))} "
            f"| {_fmt(step0.get('ppl/c4'), 2)} "
            f"| {_fmt(final.get('ppl/c4'), 2)} "
            f"| {_fmt(final.get('ppl/wikitext2'), 2)} |"
        )
    lines.append("")

    # Per-step curves.
    lines.append("## Training curves (MMLU / C4 PPL by step)")
    lines.append("")
    for _, label in METHODS:
        d = data.get(label)
        if not d:
            continue
        hist = d.get("history", {})
        steps = sorted(hist.keys(), key=lambda s: int(s))
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| step | MMLU | HellaSwag | C4 PPL | WikiText2 PPL |")
        lines.append("|---|---|---|---|---|")
        for s in steps:
            m = hist[s]
            tag = {"-1": "base", "0": "post-comp"}.get(s, s)
            lines.append(
                f"| {tag} | {_fmt(m.get('lm_eval/mmlu'))} | {_fmt(m.get('lm_eval/hellaswag'))} "
                f"| {_fmt(m.get('ppl/c4'), 2)} | {_fmt(m.get('ppl/wikitext2'), 2)} |"
            )
        lines.append("")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")
    # Also echo the headline table to stdout.
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
