"""Block-accept rate of a frequency-tiered head: is the dense argmax inside the tier?

This is the single number B1-a (sparse activation) lives or dies on, and the plan
inherits it from a pilot that reported **92.5% at T=4096** measured on ~25k
calibration tokens with a unigram prior estimated from that same small sample. It
is worth re-measuring properly, on held-out text with a 5M-token prior, because
every sparse-activation claim is downstream of it.

Reports, per tier size, over held-out C4 positions:
  accept@1   dense argmax lies in the tier   (exact-decode accept rate)
  accept@tgt the *target* token lies in the tier (what perplexity depends on)
  mass       dense probability mass inside the tier (the graded version)

    python scripts/lm_head_accept_rate.py --model Qwen/Qwen3-0.6B \
        --calib-dir ./calib/lm_head_qwen3_0_6b
"""

import argparse
import json

import torch
import torch.nn.functional as F

from src.base.datasets import load_datasets
from src.base.shared_utils import _print
from src.lm_head.calib import ensure_unigram, get_lm_head
from src.lm_head.tiering import build_tiers

SIZES = (1024, 2048, 4096, 8192, 16384, 32768, 65536)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--calib-dir", default="./calib/lm_head_qwen3_0_6b")
    ap.add_argument("--positions", type=int, default=32768)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--skip-texts", type=int, default=20000)
    ap.add_argument("--out", default="./results_eval/lm_head_accept_rate.json")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", device_map="auto",
    ).eval()
    V = get_lm_head(model).weight.shape[0]

    counts = ensure_unigram(tok, {}, a.calib_dir, V)
    masks = {}
    for T in SIZES:
        if T <= V:
            masks[T] = build_tiers(counts, T, verbose=False).keep_mask.to("cuda")

    # held-out slice, past whatever the calibration used
    texts = load_datasets("c4", tok, max_samples=a.skip_texts + 6000, max_length=None)
    texts = [t for t in texts[a.skip_texts:] if isinstance(t, str) and len(t) > 200]
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        if len(ids) >= a.positions + a.seq:
            break
    ids = torch.tensor(ids[: a.positions + 1], dtype=torch.long)
    n_win = max(1, (ids.numel() - 1) // a.seq)

    hit1 = {T: 0 for T in masks}
    hitT = {T: 0 for T in masks}
    mass = {T: 0.0 for T in masks}
    n = 0
    for w in range(n_win):
        win = ids[w * a.seq: w * a.seq + a.seq + 1]
        if win.numel() != a.seq + 1:
            continue
        chunk = win.unsqueeze(0).to("cuda")
        logits = model(chunk[:, :-1]).logits[0].float()      # (S, V)
        tgt = chunk[0, 1:]
        am = logits.argmax(-1)
        p = F.softmax(logits, dim=-1)
        n += am.numel()
        for T, m in masks.items():
            hit1[T] += int(m[am].sum())
            hitT[T] += int(m[tgt].sum())
            mass[T] += float(p[:, m].sum())

    _print(f"\nHeld-out C4, {n:,} positions, model {a.model} (V={V})")
    _print(f"{'T':>7s} {'%ofV':>6s} {'unigram mass':>13s} {'accept@1':>9s} "
           f"{'accept@target':>14s} {'dense mass':>11s}")
    rows = []
    for T in sorted(masks):
        t = build_tiers(counts, T, verbose=False)
        row = {
            "tier_size": T, "frac_of_V": T / V, "unigram_mass": t.head_mass,
            "accept_at_1": hit1[T] / n, "accept_at_target": hitT[T] / n,
            "dense_mass_in_tier": mass[T] / n, "n_positions": n,
        }
        rows.append(row)
        _print(f"{T:7d} {100 * T / V:5.2f}% {100 * t.head_mass:12.2f}% "
               f"{100 * row['accept_at_1']:8.2f}% {100 * row['accept_at_target']:13.2f}% "
               f"{100 * row['dense_mass_in_tier']:10.2f}%")
    with open(a.out, "w") as f:
        json.dump({"model": a.model, "V": V, "n_positions": n, "rows": rows}, f, indent=2)
    _print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
