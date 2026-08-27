"""Do benchmark target tokens fall inside a frequency tier?

Plan section 3 asserts that MMLU and HellaSwag "cannot evaluate a sparse-activation
head" because their target tokens "are all high-frequency, so they sit inside any
frequency-tiered read set". That is true for MMLU and **false for HellaSwag**, and the
difference is worth pinning down: HellaSwag scores multi-token continuations of natural
text, so a single out-of-tier token anywhere in an ending sends its loglikelihood to
-inf and the accuracy to chance.

This needs no model -- only the tokenizer and the cached unigram counts.

    python scripts/lm_head_task_coverage.py --calib-dir ./calib/lm_head_qwen3_0_6b
"""

import argparse
import json

import torch

from src.base.shared_utils import _print
from src.lm_head.calib import ensure_unigram
from src.lm_head.tiering import build_tiers

SIZES = (4096, 8192, 16384, 32768)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--calib-dir", default="./calib/lm_head_qwen3_0_6b")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--out", default="./results_eval/lm_head_task_coverage.json")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    V = len(tok)
    counts = ensure_unigram(tok, {}, a.calib_dir, 151936)
    masks = {T: build_tiers(counts, T, verbose=False).keep_mask for T in SIZES}

    # HellaSwag: 4 endings scored as continuations of a context.
    hs = load_dataset("Rowan/hellaswag", split="validation")
    hs_ends = []
    for ex in hs.select(range(min(a.limit, len(hs)))):
        for e in ex["endings"]:
            hs_ends.append(" " + e.strip())

    # MMLU: 4 single-token continuations " A".." D".
    mmlu_targets = [" A", " B", " C", " D"]

    out = {"model": a.model, "rows": []}
    _print(f"\n{'task':12s} {'T':>7s} {'targets in tier':>16s} "
           f"{'sequences fully in tier':>24s}")
    for T, m in masks.items():
        # --- HellaSwag: per-token and per-ending ---
        n_tok = n_tok_in = 0
        n_seq = n_seq_in = 0
        for s in hs_ends:
            ids = tok(s, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            inm = m[torch.tensor(ids)]
            n_tok += len(ids)
            n_tok_in += int(inm.sum())
            n_seq += 1
            n_seq_in += int(bool(inm.all()))
        # --- MMLU ---
        mm_tok = mm_in = 0
        for s in mmlu_targets:
            ids = tok(s, add_special_tokens=False)["input_ids"]
            inm = m[torch.tensor(ids)]
            mm_tok += len(ids)
            mm_in += int(inm.sum())
        row = {
            "tier_size": T,
            "hellaswag_token_coverage": n_tok_in / max(n_tok, 1),
            "hellaswag_ending_fully_covered": n_seq_in / max(n_seq, 1),
            "hellaswag_endings": n_seq,
            "hellaswag_mean_tokens_per_ending": n_tok / max(n_seq, 1),
            "mmlu_token_coverage": mm_in / max(mm_tok, 1),
        }
        out["rows"].append(row)
        _print(f"{'hellaswag':12s} {T:7d} {100 * row['hellaswag_token_coverage']:15.2f}% "
               f"{100 * row['hellaswag_ending_fully_covered']:23.2f}%")
        _print(f"{'mmlu':12s} {T:7d} {100 * row['mmlu_token_coverage']:15.2f}% "
               f"{100.0:23.2f}%")
    _print(
        f"\nHellaSwag endings average "
        f"{out['rows'][0]['hellaswag_mean_tokens_per_ending']:.1f} tokens over "
        f"{out['rows'][0]['hellaswag_endings']:,} endings; MMLU targets are 1 token.\n"
        "A strict sparse head scores an ending at -inf unless EVERY token is in-tier, so\n"
        "the 'fully covered' column is what its accuracy tracks."
    )
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    _print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
