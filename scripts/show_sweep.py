"""Print a sweep/ladder JSON as a readable frontier table.

    python scripts/show_sweep.py results_eval/lm_head_sweep_30b_c4.json
"""

import json
import sys


def main(paths):
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"{p}: {e}")
            continue
        print(f"\n=== {p} ===")
        if "active_params" in d:
            print(f"model={d.get('model')}  total={d['total_params'] / 1e9:.3f}B  "
                  f"active={d['active_params'] / 1e9:.3f}B  tied={d.get('was_tied')}")
        rows = d.get("rows", [])
        print(f"{len(rows)} rows")
        hdr = (f"{'variant':22s} {'store%':>7s} {'read%':>7s} {'dact%':>7s} "
               f"{'agr%':>6s} {'lost%':>6s} {'c4_wppl':>11s} {'hellaswag':>10s} {'mmlu':>8s}")
        print(hdr)
        for r in rows:
            name = r.get("variant") or r.get("run")
            c4 = (r.get("c4_raw") or {}).get("word_perplexity,none")
            hs = (r.get("hellaswag_raw") or {}).get("acc_norm,none")
            mm = (r.get("mmlu_raw") or {}).get("acc,none")
            if c4 is None and "ppl" in r:
                c4 = r["ppl"]
            lost = r.get("dense_mass_outside_tier")
            fmt = lambda v, n=4: ("-" if v is None else
                                  ("inf" if v == float("inf") else f"{v:.{n}f}"))
            print(f"{name:22s} {100 * r.get('storage_frac', float('nan')):7.2f} "
                  f"{100 * r.get('read_frac', float('nan')):7.2f} "
                  f"{r.get('delta_active_pct', float('nan')):7.2f} "
                  f"{100 * r.get('top1_agreement', 0):6.2f} "
                  f"{('-' if lost is None else f'{100 * lost:.2f}'):>6s} "
                  f"{fmt(c4, 3):>11s} {fmt(hs):>10s} {fmt(mm):>8s}"
                  + (f"  oov={100 * r['oov_rate']:.3f}%" if r.get("oov_rate") else ""))


if __name__ == "__main__":
    main(sys.argv[1:] or ["results_eval/lm_head_sweep_30b_c4.json"])
