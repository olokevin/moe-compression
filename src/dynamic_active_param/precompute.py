"""Offline allocation artifact: channel ranks + per-expert contributions.

Everything needed to rank channels within an expert and to weight experts by
their calibration contribution is already saved by the scoring stage in
``scores_dir/expert_scores.pth``. This module loads it once and reshapes it
into two dense tensors keyed by (layer, expert):

- ``channel_rank`` ``(L, E, I)`` int: rank of each channel by *descending*
  channel score (rank 0 = most important). Built as
  ``argsort(argsort(score, descending))`` per ``(l, e)``.
- ``contrib`` ``(L, E)`` float: ``expert_out_token_contrib`` clamped >= 0,
  used by the ``contribution`` allocation criterion.

``channel_metric`` selects which score tensor ranks channels:
``activation`` (repo default) or ``leverage`` (Nyström ridge-leverage; the
score only picks columns/rows — no down_proj weight correction).
"""

import os
from dataclasses import dataclass

import torch

from src.base.shared_utils.dict_to_tensor import dict_to_tensor
from src.base.shared_utils import _print

__all__ = ["AllocArtifact", "build_alloc_artifact"]


@dataclass
class AllocArtifact:
    """Precomputed ranking statistics for dynamic allocation."""

    channel_rank: torch.Tensor  # (L, E, I) long, rank by descending score
    contrib: torch.Tensor       # (L, E) float, expert_out_token_contrib >= 0
    prefix_sums: torch.Tensor   # (L, E, I) float, cumsum of descending-sorted score
    L: int
    E: int
    I: int
    channel_metric: str


def _channel_rank_from_scores(scores: torch.Tensor) -> torch.Tensor:
    """Per (l, e), rank channels by descending score. Rank 0 = highest score.

    ``argsort(argsort(-score))`` maps each channel to its 0-based position in
    the descending sort — a valid permutation of ``0..I-1`` for each expert.
    """
    # descending sort => negate; stable keeps ties deterministic.
    order = torch.argsort(-scores, dim=-1, stable=True)      # positions -> channel
    rank = torch.argsort(order, dim=-1, stable=True)         # channel -> position
    return rank.to(torch.long)


def _prefix_sums_from_scores(scores: torch.Tensor) -> torch.Tensor:
    """Per (l, e), prefix sums of the *descending-sorted* channel scores.

    ``prefix[..., n-1] = S_e(n) = sum of the top-n scores``. This must use the
    same descending sort as ``_channel_rank_from_scores`` so that the channel at
    rank ``r`` contributes exactly the ``r -> r+1`` prefix increment — the
    coverage curve and the keep-set (top-k by rank) then agree. Used by the
    ``coverage_alloc`` criterion (paper §4.2: ``rho_e(n) = S_e(n) / S_tot_e``).
    """
    # clamp >= 0 so the cumulative curve is non-decreasing (required for the
    # searchsorted in coverage allocation). Leverage/activation are already
    # non-negative; this only guards against tiny negative float noise.
    sorted_desc = torch.sort(scores.clamp_min(0.0), dim=-1, descending=True).values
    return sorted_desc.cumsum(dim=-1).to(torch.float32)


def build_alloc_artifact(
    scores_dir: str,
    channel_metric: str = "activation",
    device: str = "cpu",
    save: bool = True,
    verbose: bool = True,
) -> AllocArtifact:
    """Build (or load) the dynamic-allocation artifact from ``scores_dir``.

    Args:
        scores_dir: directory holding ``expert_scores.pth``.
        channel_metric: ``activation`` | ``leverage`` — which score ranks channels.
        device: device to place the returned tensors on.
        save: if True, cache the artifact to
            ``scores_dir/dynamic_alloc_<metric>.pth`` for reuse.
        verbose: print progress.

    Returns:
        AllocArtifact with ``channel_rank`` (L,E,I) and ``contrib`` (L,E).
    """
    # v2: schema gained ``prefix_sums`` (for coverage_alloc). The bumped filename
    # ensures pre-v2 caches (which lack it) are not silently reused.
    cache_path = os.path.join(scores_dir, f"dynamic_alloc_{channel_metric}_v2.pth")
    if os.path.exists(cache_path):
        if verbose:
            _print(f"[DynamicAlloc] Loading cached artifact from {cache_path}")
        payload = torch.load(cache_path, map_location=device)
        return AllocArtifact(
            channel_rank=payload["channel_rank"].to(device),
            contrib=payload["contrib"].to(device),
            prefix_sums=payload["prefix_sums"].to(device),
            L=int(payload["L"]),
            E=int(payload["E"]),
            I=int(payload["I"]),
            channel_metric=payload["channel_metric"],
        )

    scores_path = os.path.join(scores_dir, "expert_scores.pth")
    if verbose:
        _print(f"[DynamicAlloc] Building artifact (metric={channel_metric}) from {scores_path}")
    expert_scores = torch.load(scores_path, map_location=device)

    if channel_metric not in expert_scores:
        raise KeyError(
            f"channel_metric {channel_metric!r} not in expert_scores.pth "
            f"(keys: {list(expert_scores.keys())})"
        )
    scores = dict_to_tensor(expert_scores[channel_metric]).to(device)  # (L, E, I)
    if scores.ndim != 3:
        raise ValueError(f"Expected (L,E,I) scores for {channel_metric!r}, got {tuple(scores.shape)}")
    L, E, I = scores.shape

    channel_rank = _channel_rank_from_scores(scores)   # (L, E, I) long
    prefix_sums = _prefix_sums_from_scores(scores)     # (L, E, I) float

    # expert_out_token_contrib is stored as a (calibration-averaged) *negative*
    # per-expert scalar — a more-important expert is more negative. The repo's
    # static attr_coverage path negates it to get a positive coverage weight
    # (src/prune/generate/stages/prepare_scores.py:116); we follow the same
    # convention so the 'contribution' criterion has meaningful (nonzero) weights.
    # (Clamping the raw negatives to >=0 would zero everything -> uniform fallback.)
    contrib = dict_to_tensor(expert_scores["expert_out_token_contrib"]).to(device)  # (L, E)
    contrib = (-contrib).clamp_min(0.0)
    if contrib.shape != (L, E):
        raise ValueError(
            f"expert_out_token_contrib shape {tuple(contrib.shape)} != expected {(L, E)}"
        )

    if verbose:
        _print(
            f"[DynamicAlloc] channel_rank {tuple(channel_rank.shape)}, "
            f"contrib {tuple(contrib.shape)}, prefix_sums {tuple(prefix_sums.shape)}"
        )

    if save:
        torch.save(
            {
                "channel_rank": channel_rank.cpu(),
                "contrib": contrib.cpu(),
                "prefix_sums": prefix_sums.cpu(),
                "L": L,
                "E": E,
                "I": I,
                "channel_metric": channel_metric,
            },
            cache_path,
        )
        if verbose:
            _print(f"[DynamicAlloc] Cached artifact to {cache_path}")

    return AllocArtifact(
        channel_rank=channel_rank,
        contrib=contrib,
        prefix_sums=prefix_sums,
        L=L,
        E=E,
        I=I,
        channel_metric=channel_metric,
    )
