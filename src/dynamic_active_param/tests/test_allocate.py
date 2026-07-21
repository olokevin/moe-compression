import torch
import pytest

from src.dynamic_active_param.allocate import allocate_budgets


def _rand_topk(T, K, E, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, E, generator=g)
    weights = torch.softmax(logits, dim=1)
    w, sel = torch.topk(weights, K, dim=1)
    w = w / w.sum(dim=1, keepdim=True)
    return w, sel


@pytest.mark.parametrize("criterion", ["router_prob", "contribution", "uniform"])
def test_budget_conservation_and_bounds(criterion):
    T, K, E, I = 200, 4, 32, 128
    k_min = 8
    B = round(0.67 * K * I)
    w, sel = _rand_topk(T, K, E, seed=1)
    contrib = torch.rand(E)

    k = allocate_budgets(w, sel, contrib, B, k_min, I, criterion=criterion)

    assert k.shape == (T, K)
    assert k.dtype == torch.long
    assert torch.all(k.sum(dim=1) == B), "budget must be met exactly per token"
    assert torch.all(k >= k_min), "floor violated"
    assert torch.all(k <= I), "cap violated"


def test_monotonicity_router_prob():
    # Larger routing weight => at least as many channels (single token).
    I, K = 100, 4
    k_min = 4
    B = round(0.67 * K * I)
    w = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    sel = torch.tensor([[0, 1, 2, 3]])
    k = allocate_budgets(w, sel, None, B, k_min, I, criterion="router_prob")[0]
    # weights are sorted descending => budgets should be non-increasing
    assert torch.all(k[:-1] >= k[1:]), f"budgets not monotone with weight: {k}"


def test_uniform_even_split():
    I, K = 128, 4
    k_min = 4
    B = K * 64  # divisible so split is exactly even
    w, sel = _rand_topk(50, K, 16, seed=3)
    k = allocate_budgets(w, sel, None, B, k_min, I, criterion="uniform")
    assert torch.all(k == 64), "uniform criterion should split evenly when divisible"


def test_rho_one_keeps_all_channels():
    I, K = 64, 4
    k_min = 4
    B = K * I  # rho = 1.0
    w, sel = _rand_topk(30, K, 20, seed=5)
    contrib = torch.rand(20)
    for crit in ["router_prob", "contribution", "uniform"]:
        k = allocate_budgets(w, sel, contrib, B, k_min, I, criterion=crit)
        assert torch.all(k == I), f"rho=1.0 must keep all channels ({crit})"


def test_degenerate_contrib_falls_back_uniform():
    I, K = 128, 4
    k_min = 4
    B = K * 80
    w, sel = _rand_topk(40, K, 16, seed=7)
    contrib = torch.zeros(16)  # all-zero => uniform fallback
    k = allocate_budgets(w, sel, contrib, B, k_min, I, criterion="contribution")
    assert torch.all(k == 80), "all-zero contrib should fall back to uniform even split"


def test_infeasible_budget_raises():
    I, K = 32, 4
    with pytest.raises(ValueError):
        # k_min too high: K*k_min > B
        allocate_budgets(
            torch.rand(5, K), torch.zeros(5, K, dtype=torch.long), None,
            B=10, k_min=8, I=I, criterion="uniform",
        )


def test_unknown_criterion_raises():
    with pytest.raises(ValueError):
        allocate_budgets(
            torch.rand(5, 4), torch.zeros(5, 4, dtype=torch.long), None,
            B=100, k_min=4, I=64, criterion="bogus",
        )


# --------------------------------------------------------------------------
# coverage_alloc
# --------------------------------------------------------------------------

def _prefix_from_scores(scores):
    """(E,I) descending-sorted cumulative sums, matching precompute helper."""
    sorted_desc = torch.sort(scores.clamp_min(0.0), dim=-1, descending=True).values
    return sorted_desc.cumsum(dim=-1).to(torch.float32)


def test_coverage_requires_prefix():
    with pytest.raises(ValueError):
        allocate_budgets(
            torch.rand(3, 4), torch.zeros(3, 4, dtype=torch.long), None,
            B=100, k_min=4, I=64, criterion="coverage_alloc", prefix_sums=None,
        )


def test_coverage_budget_conservation_and_bounds():
    T, K, E, I = 200, 4, 32, 128
    k_min = 8
    B = round(0.5 * K * I)
    w, sel = _rand_topk(T, K, E, seed=11)
    scores = torch.rand(E, I) + 1e-3
    prefix = _prefix_from_scores(scores)

    k = allocate_budgets(w, sel, None, B, k_min, I,
                         criterion="coverage_alloc", prefix_sums=prefix)

    assert k.shape == (T, K)
    assert k.dtype == torch.long
    assert torch.all(k.sum(dim=1) == B), "coverage budget must be met exactly per token"
    assert torch.all(k >= k_min), "floor violated"
    assert torch.all(k <= I), "cap violated"


def test_coverage_rho_one_keeps_all_channels():
    I, K, E = 64, 4, 20
    k_min = 4
    B = K * I  # rho = 1.0 -> every expert must take all channels
    w, sel = _rand_topk(30, K, E, seed=13)
    prefix = _prefix_from_scores(torch.rand(E, I) + 1e-3)
    k = allocate_budgets(w, sel, None, B, k_min, I,
                         criterion="coverage_alloc", prefix_sums=prefix)
    assert torch.all(k == I), "rho=1.0 must keep all channels (coverage_alloc)"


def test_coverage_concentration_effect():
    # Two experts, EQUAL router prob. Expert 0's leverage is concentrated in a
    # few channels; expert 1's is flat. The concentrated expert should reach its
    # coverage target with FEWER channels, so it gets fewer of the budget.
    I, K = 64, 2
    k_min = 1
    B = 40  # < K*I so allocation is non-trivial
    w = torch.tensor([[0.5, 0.5]])
    sel = torch.tensor([[0, 1]])

    concentrated = torch.zeros(I)
    concentrated[:4] = 10.0            # nearly all mass in 4 channels
    concentrated[4:] = 0.01
    flat = torch.ones(I)               # mass spread evenly
    scores = torch.stack([concentrated, flat], dim=0)  # (E=2, I)
    prefix = _prefix_from_scores(scores)

    k = allocate_budgets(w, sel, None, B, k_min, I,
                         criterion="coverage_alloc", prefix_sums=prefix)[0]
    assert k.sum() == B
    assert k[0] < k[1], f"concentrated expert should get fewer channels: {k}"


def test_coverage_equal_scores_monotone_in_prob():
    # All-equal scores => linear coverage curve => coverage target (and thus
    # channel count) is monotone in router prob. Use a single token with
    # sorted-descending probs and assert non-increasing per-expert counts.
    I, K = 100, 4
    k_min = 1
    B = round(0.5 * K * I)
    w = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    sel = torch.tensor([[0, 1, 2, 3]])
    prefix = _prefix_from_scores(torch.ones(4, I))
    k = allocate_budgets(w, sel, None, B, k_min, I,
                         criterion="coverage_alloc", prefix_sums=prefix)[0]
    assert k.sum() == B
    assert torch.all(k[:-1] >= k[1:]), f"equal-score coverage not monotone in prob: {k}"


def test_coverage_monotone_in_budget():
    # Larger total budget B => per-expert counts are non-decreasing.
    I, K, E = 96, 4, 24
    k_min = 2
    w, sel = _rand_topk(50, K, E, seed=19)
    prefix = _prefix_from_scores(torch.rand(E, I) + 1e-3)
    k_small = allocate_budgets(w, sel, None, round(0.4 * K * I), k_min, I,
                               criterion="coverage_alloc", prefix_sums=prefix)
    k_large = allocate_budgets(w, sel, None, round(0.7 * K * I), k_min, I,
                               criterion="coverage_alloc", prefix_sums=prefix)
    assert torch.all(k_large >= k_small), "counts must not shrink as budget grows"
