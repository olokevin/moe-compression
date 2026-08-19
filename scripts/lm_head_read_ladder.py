"""Print the S1 read-fraction ladder for both models, from the real cost function.

The read fraction is ``r0/D + N*(D-r0)/(V*D) + D/V``, which has two floors that are
easy to get wrong by hand: the candidate set alone costs ``N/V`` and the rotation costs
``D/V``. This prints what each variant actually budgets on each head shape, so a sweep
is never launched against a read fraction that was guessed.

    python scripts/lm_head_read_ladder.py
"""

from scripts.gen_lm_head_configs import S1_LADDER, VARIANTS
from src.lm_head.screen_refine import screen_refine_cost

V = 151936
SHAPES = {"Qwen3-30B-A3B (D=2048)": 2048, "Qwen3-0.6B (D=1024)": 1024}


def main():
    hdr = f"{'variant':<16}{'r0/D':>9}{'N':>8}" + "".join(f"{k:>26}" for k in SHAPES)
    print(hdr)
    print("-" * len(hdr))
    for name in S1_LADDER:
        cfg = VARIANTS[name]
        frac, N = cfg["screen_rank_frac"], cfg["cand_size"]
        row = f"{name:<16}{frac:>9.5f}{N:>8}"
        for label, D in SHAPES.items():
            sc = screen_refine_cost(V, D, max(1, round(frac * D)), N)
            row += (f"   r0={sc['screen_rank']:<5} reads={100 * sc['read_param_frac']:>6.2f}%")
        print(row)
    print(f"\nfloors: candidate set N/V, rotation D/V "
          f"({100 * 2048 / V:.2f}% on the 30B, {100 * 1024 / V:.2f}% on the 0.6B)")


if __name__ == "__main__":
    main()
