import os

import torch

from src.dynamic_active_param.precompute import (
    _channel_rank_from_scores,
    _prefix_sums_from_scores,
    build_alloc_artifact,
)


def test_rank_is_valid_permutation():
    L, E, I = 3, 5, 16
    torch.manual_seed(0)
    scores = torch.randn(L, E, I)
    rank = _channel_rank_from_scores(scores)
    assert rank.shape == (L, E, I)
    # each (l,e) row is a permutation of 0..I-1
    for l in range(L):
        for e in range(E):
            assert torch.equal(torch.sort(rank[l, e]).values, torch.arange(I))


def test_top_ranked_is_argmax():
    L, E, I = 2, 3, 20
    torch.manual_seed(1)
    scores = torch.randn(L, E, I)
    rank = _channel_rank_from_scores(scores)
    # channel with rank 0 must be the argmax score
    for l in range(L):
        for e in range(E):
            top_channel = (rank[l, e] == 0).nonzero().item()
            assert top_channel == int(scores[l, e].argmax())


def test_prefix_sums_non_decreasing_and_total():
    L, E, I = 2, 3, 16
    torch.manual_seed(4)
    scores = torch.rand(L, E, I)  # non-negative
    prefix = _prefix_sums_from_scores(scores)
    assert prefix.shape == (L, E, I)
    # cumulative -> non-decreasing along channels
    assert torch.all(prefix[..., 1:] >= prefix[..., :-1] - 1e-5)
    # final prefix == total score mass per (l,e)
    assert torch.allclose(prefix[..., -1], scores.sum(dim=-1), atol=1e-4)


def test_prefix_increments_match_rank_order():
    # The channel at rank r contributes exactly the r->r+1 prefix increment,
    # i.e. prefix increments equal the descending-sorted scores.
    L, E, I = 2, 2, 12
    torch.manual_seed(5)
    scores = torch.rand(L, E, I)
    prefix = _prefix_sums_from_scores(scores)
    rank = _channel_rank_from_scores(scores)
    inc = prefix.clone()
    inc[..., 1:] = prefix[..., 1:] - prefix[..., :-1]  # (L,E,I) marginal at each rank
    for l in range(L):
        for e in range(E):
            # scatter each channel's score to its rank position, compare to inc
            by_rank = torch.zeros(I)
            by_rank[rank[l, e]] = scores[l, e]
            assert torch.allclose(by_rank, inc[l, e], atol=1e-4)


def test_build_artifact_roundtrip(tmp_path):
    L, E, I = 4, 6, 32
    torch.manual_seed(2)
    expert_scores = {
        "activation": {l: torch.rand(E, I) for l in range(L)},
        "expert_out_token_contrib": {l: torch.randn(E) for l in range(L)},
    }
    scores_dir = str(tmp_path)
    torch.save(expert_scores, os.path.join(scores_dir, "expert_scores.pth"))

    art = build_alloc_artifact(scores_dir, channel_metric="activation", device="cpu", save=True)
    assert art.L == L and art.E == E and art.I == I
    assert art.channel_rank.shape == (L, E, I)
    assert art.contrib.shape == (L, E)
    assert art.prefix_sums.shape == (L, E, I)
    assert torch.all(art.contrib >= 0), "contrib must be clamped >= 0"

    # cache exists (v2 schema with prefix_sums) and reloads identically
    cache = os.path.join(scores_dir, "dynamic_alloc_activation_v2.pth")
    assert os.path.exists(cache)
    art2 = build_alloc_artifact(scores_dir, channel_metric="activation", device="cpu", save=True)
    assert torch.equal(art.channel_rank, art2.channel_rank)
    assert torch.equal(art.contrib, art2.contrib)
    assert torch.equal(art.prefix_sums, art2.prefix_sums)
