"""Print an lm_head sweep JSON as a comparison table (S1 columns included)."""

import argparse
import json


def get(row, task):
    """acc_norm for the multiple-choice tasks, word_perplexity for c4."""
    raw = row.get(f"{task}_raw")
    if not isinstance(raw, dict):
        return None
    inner = raw.get(task) if isinstance(raw.get(task), dict) else raw
    for k in ("acc_norm,none", "word_perplexity,none", "acc,none"):
        if k in inner:
            return inner[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tasks", nargs="+", default=["hellaswag", "arc_challenge"])
    a = ap.parse_args()

    for f in a.files:
        d = json.load(open(f))
        print(f"\n=== {f}  ({d.get('model')}, V={d.get('V')} D={d.get('D')}) ===")
        hdr = (f"{'variant':<22}{'store%':>8}{'read%':>8}{'Δactive%':>10}"
               + "".join(f"{t[:9]:>11}" for t in a.tasks)
               + f"{'KL':>9}{'top1%':>8}{'aic%':>8}")
        print(hdr)
        print("-" * len(hdr))
        base = {}
        for r in d["rows"]:
            if r["variant"] == "dense":
                base = {t: get(r, t) for t in a.tasks}
        for r in d["rows"]:
            cells = ""
            for t in a.tasks:
                v = get(r, t)
                if v is None:
                    cells += f"{'-':>11}"
                elif t == "c4":
                    # perplexity: absolute, plus the ratio to dense (lower is better)
                    cells += (f"{v:>7.3f}x{v / base[t]:<3.3f}" if base.get(t)
                              else f"{v:>11.3f}")
                elif base.get(t) and r["variant"] != "dense":
                    cells += f"{100 * v:>6.2f}{100 * (v - base[t]):>+5.2f}"
                else:
                    cells += f"{100 * v:>11.2f}"
            sp = r.get("stored_param_frac", r.get("storage_frac"))
            rp = r.get("read_param_frac", r.get("read_frac"))
            kl = r.get("kl_vs_dense")
            t1 = r.get("top1_agreement")
            aic = r.get("argmax_in_tier")
            print(f"{r['variant']:<22}"
                  f"{100 * sp:>8.2f}" if sp is not None else f"{r['variant']:<22}{'-':>8}", end="")
            print(f"{100 * rp:>8.2f}" if rp is not None else f"{'-':>8}", end="")
            print(f"{r.get('delta_active_pct', 0):>+10.2f}" + cells
                  + (f"{kl:>9.4f}" if isinstance(kl, float) else f"{'-':>9}")
                  + (f"{100 * t1:>8.2f}" if isinstance(t1, float) else f"{'-':>8}")
                  + (f"{100 * aic:>8.3f}" if isinstance(aic, float) else f"{'-':>8}"))


if __name__ == "__main__":
    main()
