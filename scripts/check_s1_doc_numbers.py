"""Verify every S1 ladder number in results_lm_head.md against the result JSONs.

The doc's §3e tables are transcribed by hand from sweep output, which is exactly the
step where a digit gets dropped. This re-derives each (reads, ARC-C, KL) triple from the
JSONs and diffs it against what the markdown claims.

    python scripts/check_s1_doc_numbers.py
"""

import json
import os
import re
import sys

DOC = "docs/exps/lm_head/results_lm_head.md"
SOURCES = {
    "30B": ["results_eval/lm_head_s1_30b_lowread_arc.json",
            "results_eval/lm_head_s1_30b_c4arc.json"],
    "0.6B": ["results_eval/lm_head_s1_0_6b_lowread_arc.json",
             "results_eval/lm_head_s1_0_6b_tasks.json"],
}


def measured(paths):
    """(screen_rank, cand_size) -> measured row. Keyed on the SCREEN CONFIG, not on the
    read fraction: `s1_r12_n8k` and the `s1_r12_n8k_mag` ablation share a budget exactly,
    so a reads-keyed lookup silently compares a ladder row against an ablation."""
    out = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for row in json.load(open(p))["rows"]:
            raw = row.get("arc_challenge_raw")
            if not isinstance(raw, dict) or row.get("screen_rank") is None:
                continue
            if row.get("screen_use_col_norm") is False or row.get("basis") != "ceig":
                continue                      # ablations are not ladder points
            inner = (raw.get("arc_challenge")
                     if isinstance(raw.get("arc_challenge"), dict) else raw)
            if "acc_norm,none" not in inner:
                continue
            out.setdefault((row["screen_rank"], row["cand_size"]), {
                "variant": row["variant"],
                "reads": 100 * row.get("read_param_frac", row.get("read_frac", 0)),
                "arc": 100 * inner["acc_norm,none"],
                "kl": row.get("kl_vs_dense"),
            })
    return out


ROW = re.compile(
    r"^\|\s*\*{0,2}(\d+)\*{0,2}\s*\|\s*\*{0,2}(\d+)\*{0,2}\s*\|"          # r0 | N
    r"\s*\*{0,2}([\d.]+)%\*{0,2}\s*\|"                                     # reads
    r"(?:\s*\*{0,2}[−+-][\d.]+%\*{0,2}\s*\|)?"                             # optional Δactive
    r"\s*\*{0,2}(\d\d\.\d\d)\*{0,2}\s*\|"                                  # ARC-C
    r"\s*\*{0,2}([−+-][\d.]+)\*{0,2}\s*\|"                                 # Δ
    r"\s*\.?(\d+)\s*\|", re.M)


def main():
    text = open(DOC).read()
    bad = checked = 0
    for model, paths in SOURCES.items():
        m = measured(paths)
        for r0, N, reads, arc, delta, kl in ROW.findall(text):
            key = (int(r0), int(N))
            if key not in m:
                continue
            v = m[key]
            if abs(v["reads"] - float(reads)) > 0.02:
                continue                      # this table row belongs to the other model
            checked += 1
            if abs(v["arc"] - float(arc)) > 0.006:
                print(f"MISMATCH {model} r0={r0} N={N} ({reads}%): "
                      f"doc ARC-C {arc}, measured {v['arc']:.2f}  [{v['variant']}]")
                bad += 1
            doc_kl = float("0." + kl)
            if v["kl"] and abs(v["kl"] - doc_kl) > 10 ** -len(kl):
                print(f"MISMATCH {model} r0={r0} N={N}: doc KL .{kl}, "
                      f"measured {v['kl']:.5f}  [{v['variant']}]")
                bad += 1
    print(f"\n{'FAILED' if bad else 'OK'}: {checked} ladder rows checked, {bad} mismatched")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
