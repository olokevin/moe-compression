"""B1 -- frequency-tiered head: the free static prior.

Partition the vocabulary by calibration unigram frequency into a head tier ``H``
(the top ``T`` rows) and a tail. Three sub-variants share this one partition and
differ only in what happens to the tail:

``B1-s`` (storage)
    Tail rows kept, stored at ``tail_bits``. No row is droppable, so no task can
    break; the only cost is quantization error on rare rows.
``B1-p`` (prune)
    Tail rows removed. Real parameter reduction, but the model provably cannot
    emit a tail token.
``B1-a`` (sparse activation)
    All rows stored, only the head tier *read* per position. Same arithmetic as
    B1-p -- the difference is purely which axis the saving lands on (bytes stored
    vs bytes read per token), which is why :mod:`src.lm_head.accounting` reports
    both. In ``strict`` mode a tail target token is a real, visible failure; the
    ``uniform`` fallback instead gives the whole tail one shared logit, which is
    the classic tiered-softmax construction and keeps perplexity finite.

Why frequency and not magnitude: the pilots measured row norms near-uniform
(p99/p50 = 1.19-1.33) and ``corr(log freq, ||w||) = -0.13``. Frequency is the only
axis with signal, and it is free.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from src.base.shared_utils import _print

__all__ = ["FreqTiers", "build_tiers", "tier_stats"]


@dataclass
class FreqTiers:
    """A frequency partition of the vocabulary.

    ``keep_mask`` is ``(V,) bool`` -- True for head-tier rows. ``head_idx`` is the
    head tier in descending-frequency order (useful for reporting, not needed by
    the forward).
    """

    keep_mask: torch.Tensor
    head_idx: torch.Tensor
    tier_size: int
    vocab_size: int
    head_mass: float          # fraction of calibration token occurrences in the tier
    tail_mass: float
    n_unseen_in_tier: int     # tier rows with zero calibration count (T > #types)
    tail_logit_offset: float  # log(tail_mass / #tail) - for the uniform fallback


def build_tiers(counts: torch.Tensor, tier_size: int, verbose: bool = True) -> FreqTiers:
    """Split ``counts`` into a top-``tier_size`` head tier and a tail.

    Ties are broken by token id (``torch.topk`` is deterministic given a fixed
    input), so two runs with the same ``unigram.pt`` produce the same partition.
    """
    counts = counts.to(torch.float64).cpu()
    V = counts.numel()
    T = int(min(max(int(tier_size), 1), V))

    order = torch.argsort(counts, descending=True, stable=True)
    head_idx = order[:T].contiguous()
    keep_mask = torch.zeros(V, dtype=torch.bool)
    keep_mask[head_idx] = True

    total = counts.sum().clamp_min(1.0)
    head_mass = float(counts[head_idx].sum() / total)
    tail_mass = 1.0 - head_mass
    n_unseen = int((counts[head_idx] == 0).sum())
    n_tail = V - T
    # One shared logit for the whole tail, placed so that the tail's total
    # probability equals its calibration mass. Used only by tail_fallback="uniform".
    if n_tail > 0 and tail_mass > 0:
        offset = float(torch.log(torch.tensor(tail_mass / n_tail, dtype=torch.float64)))
    else:
        offset = float("-inf")

    tiers = FreqTiers(
        keep_mask=keep_mask, head_idx=head_idx, tier_size=T, vocab_size=V,
        head_mass=head_mass, tail_mass=tail_mass, n_unseen_in_tier=n_unseen,
        tail_logit_offset=offset,
    )
    if verbose:
        _print(
            f"[lm_head/B1] tier T={T} ({100 * T / V:.2f}% of V={V}): "
            f"calibration mass in tier={100 * head_mass:.3f}%, "
            f"tail mass={100 * tail_mass:.3f}% over {n_tail:,} rows"
            + (f"; {n_unseen:,} tier rows are unseen in calibration" if n_unseen else "")
        )
    return tiers


@torch.no_grad()
def tier_stats(counts: torch.Tensor, sizes=(1024, 4096, 8192, 16384, 32768)) -> dict:
    """Corpus mass covered by each candidate tier size -- the cheap sizing sweep."""
    counts = counts.to(torch.float64).cpu()
    total = counts.sum().clamp_min(1.0)
    srt = torch.sort(counts, descending=True).values
    cum = torch.cumsum(srt, dim=0) / total
    V = counts.numel()
    return {int(t): float(cum[min(int(t), V) - 1]) for t in sizes}
