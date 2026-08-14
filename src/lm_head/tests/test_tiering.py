"""Phase 0 gates 0a / 0c: tier partition and strict-mode masking correctness.

Run: ``.venv/bin/python -m src.lm_head.tests.test_tiering`` from the repo root.
"""

import torch

from src.lm_head.tiering import build_tiers, tier_stats

V = 512


def _counts():
    g = torch.Generator().manual_seed(0)
    # Zipf-ish, plus a block of genuinely unseen tokens so the tail is realistic
    c = (1.0 / torch.arange(1, V + 1).float()).mul(1e5).round().long()
    c[400:] = 0
    perm = torch.randperm(V, generator=g)
    return c[perm]


def test_partition_is_exact():
    c = _counts()
    for T in (1, 8, 64, 400, V):
        t = build_tiers(c, T, verbose=False)
        assert int(t.keep_mask.sum()) == T, (T, int(t.keep_mask.sum()))
        assert t.head_idx.numel() == T
        # the tier is exactly the T largest counts
        assert torch.equal(
            torch.sort(c[t.head_idx], descending=True).values,
            torch.sort(c, descending=True).values[:T],
        )
        # mass adds up
        assert abs(t.head_mass + t.tail_mass - 1.0) < 1e-9
    print("  ✅ partition is exactly the top-T by frequency, masses sum to 1")


def test_tier_size_clamped_and_monotone():
    c = _counts()
    t = build_tiers(c, 10 * V, verbose=False)
    assert t.tier_size == V and t.tail_mass == 0.0 and t.tail_logit_offset == float("-inf")
    masses = [build_tiers(c, T, verbose=False).head_mass for T in (1, 8, 64, 256, V)]
    assert all(b >= a - 1e-12 for a, b in zip(masses, masses[1:])), masses
    assert abs(masses[-1] - 1.0) < 1e-9
    print("  ✅ T > V clamps to V; head mass is monotone in T and reaches 1.0")


def test_unseen_accounting():
    c = _counts()
    n_types = int((c > 0).sum())
    t = build_tiers(c, n_types + 50, verbose=False)
    assert t.n_unseen_in_tier == 50, t.n_unseen_in_tier
    t2 = build_tiers(c, n_types, verbose=False)
    assert t2.n_unseen_in_tier == 0 and abs(t2.head_mass - 1.0) < 1e-9
    print("  ✅ unseen-in-tier count is exact; a tier of all seen types holds all mass")


def test_strict_mask_gives_exact_neg_inf():
    """Gate 0c -- the masked forward must produce exactly -inf outside the tier."""
    from src.lm_head.install import _tiered_lm_head_forward

    torch.manual_seed(0)
    lin = torch.nn.Linear(16, V, bias=False)
    mask = torch.zeros(V, dtype=torch.bool)
    mask[build_tiers(_counts(), 64, verbose=False).head_idx] = True

    lin._lmh_keep_mask = mask
    lin._lmh_tail_logit = None            # strict
    lin._lmh_stats = {"tokens": 0, "argmax_in_tier": 0}
    import types
    lin.forward = types.MethodType(_tiered_lm_head_forward, lin)

    x = torch.randn(3, 5, 16)
    out = lin(x)
    assert torch.isinf(out[..., ~mask]).all() and (out[..., ~mask] < 0).all()
    assert torch.isfinite(out[..., mask]).all()
    # in-tier logits must be untouched
    ref = torch.nn.functional.linear(x, lin.weight)
    assert torch.equal(out[..., mask], ref[..., mask])
    # log_softmax must stay finite in-tier (no nan from -inf arithmetic)
    lsm = torch.log_softmax(out.float(), dim=-1)
    assert torch.isfinite(lsm[..., mask]).all(), "log_softmax produced nan/inf in-tier"
    # The hit-rate stat must count every scored position, and must be measured on
    # the PRE-mask logits -- a post-mask argmax is in-tier by construction and would
    # report a meaningless 100%. With a random 64/512 tier, chance is ~12.5%, so the
    # count must be well below the 15 positions scored.
    assert lin._lmh_stats["tokens"] == 15
    n_hit = lin._lmh_stats["argmax_in_tier"]
    assert n_hit < 15, (
        f"argmax_in_tier={n_hit}/15 -- the stat is being taken after masking"
    )
    ref_hits = int(mask[torch.nn.functional.linear(x, lin.weight).reshape(-1, V).argmax(-1)].sum())
    assert n_hit == ref_hits, (n_hit, ref_hits)
    print("  ✅ strict mode: out-of-tier logits are exactly -inf, in-tier bit-exact, "
          "log_softmax finite")


def test_uniform_fallback_is_finite():
    from src.lm_head.install import _tiered_lm_head_forward
    import types

    torch.manual_seed(0)
    lin = torch.nn.Linear(16, V, bias=False)
    tiers = build_tiers(_counts(), 64, verbose=False)
    lin._lmh_keep_mask = tiers.keep_mask
    lin._lmh_tail_logit = tiers.tail_logit_offset
    lin._lmh_stats = None
    lin.forward = types.MethodType(_tiered_lm_head_forward, lin)

    out = lin(torch.randn(2, 4, 16))
    assert torch.isfinite(out).all(), "uniform fallback must keep every logit finite"
    tail = out[..., ~tiers.keep_mask]
    assert torch.allclose(tail, torch.full_like(tail, tiers.tail_logit_offset))
    print("  ✅ uniform tail fallback: all logits finite, tail is one shared constant")


def test_tier_stats_matches_build():
    c = _counts()
    st = tier_stats(c, sizes=(8, 64, 256))
    for T, m in st.items():
        assert abs(m - build_tiers(c, T, verbose=False).head_mass) < 1e-9
    print("  ✅ tier_stats sizing sweep agrees with build_tiers")


if __name__ == "__main__":
    print("Phase 0 gates 0a/0c -- tiering + strict masking")
    test_partition_is_exact()
    test_tier_size_clamped_and_monotone()
    test_unseen_accounting()
    test_strict_mask_gives_exact_neg_inf()
    test_uniform_fallback_is_finite()
    test_tier_stats_matches_build()
    print("ALL TIERING TESTS PASSED")
