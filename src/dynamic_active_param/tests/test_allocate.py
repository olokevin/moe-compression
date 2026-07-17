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
