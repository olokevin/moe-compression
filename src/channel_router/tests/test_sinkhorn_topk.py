"""Unit tests for the differentiable exact-budget top-k (plan §3.4: required)."""

import torch

from src.channel_router.sinkhorn_topk import (
    hard_topk, marginal_error, sinkhorn_topk, sinkhorn_topk_ste,
)


def test_budget_exactness_soft():
    torch.manual_seed(0)
    s = torch.randn(7, 64)
    for B in (1, 8, 32, 63):
        p = sinkhorn_topk(s, B, eps=0.1)
        assert torch.allclose(p.sum(1), torch.full((7,), float(B)), rtol=1e-4), \
            f"budget drift at B={B}: {p.sum(1)}"
        assert (p >= 0).all()


def test_channel_marginal_holds_at_moderate_eps():
    """``plan ≤ 1`` needs convergence; check the regime Stage C actually anneals into."""
    torch.manual_seed(0)
    s = torch.randn(4, 48) * 3
    for eps in (1.0, 0.3, 0.1, 0.03):
        berr, cerr = marginal_error(sinkhorn_topk(s, 12, eps=eps, n_iter=600), 12)
        assert berr < 1e-3, (eps, berr)
        assert cerr < 1e-3, (eps, cerr)


def test_ranking_is_always_preserved():
    """The plan is a monotone function of the score, so top-B by plan == top-B by score.

    This is what lets Stage C use the soft plan for gradients while the forward pass
    takes the hard top-B: the two never disagree about *which* channels win.
    """
    torch.manual_seed(0)
    s = torch.randn(6, 40) * 2
    B = 9
    hard = hard_topk(s, B)
    for eps in (2.0, 0.5, 0.1, 0.03):
        p = sinkhorn_topk(s, B, eps=eps, n_iter=300)
        assert torch.equal(hard_topk(p, B), hard), eps


def test_mass_concentrates_as_eps_anneals():
    torch.manual_seed(0)
    s = torch.randn(4, 48) * 3
    B = 12
    hard = hard_topk(s, B)
    prev = 0.0
    for eps in (2.0, 1.0, 0.3, 0.1, 0.03):
        p = sinkhorn_topk(s, B, eps=eps, n_iter=600)
        mass = float(((p * hard).sum(1) / p.sum(1)).mean())
        assert mass > prev, (eps, mass, prev)
        prev = mass
    assert prev > 0.95


def test_large_eps_converges_to_uniform():
    s = torch.randn(3, 40)
    B = 10
    p = sinkhorn_topk(s, B, eps=1e4, n_iter=400)
    assert torch.allclose(p, torch.full_like(p, B / 40), atol=1e-3)


def test_ste_forward_is_hard_and_budget_exact():
    torch.manual_seed(0)
    s = torch.randn(5, 32)
    for B in (1, 4, 16):
        h = hard_topk(s, B)
        st = sinkhorn_topk_ste(s.clone().requires_grad_(True), B, eps=0.2)
        assert torch.equal(st.detach(), h)
        assert (st.detach().sum(1) == B).all()


def test_gradient_matches_finite_difference():
    """d(Σ w·plan)/d(score) against central differences on the soft plan."""
    torch.manual_seed(0)
    s = torch.randn(1, 12, dtype=torch.double, requires_grad=True)
    w = torch.randn(1, 12, dtype=torch.double)
    B, eps = 4, 0.5

    def f(x):
        return (sinkhorn_topk(x, B, eps=eps, n_iter=500, tol=0.0) * w).sum()

    f(s).backward()
    ana = s.grad.clone()
    num = torch.zeros_like(s)
    h = 1e-5
    for i in range(s.shape[1]):
        sp, sm = s.detach().clone(), s.detach().clone()
        sp[0, i] += h
        sm[0, i] -= h
        num[0, i] = (f(sp) - f(sm)) / (2 * h)
    assert torch.allclose(ana, num, atol=1e-4, rtol=1e-3), f"{ana}\n{num}"


def test_ste_gradient_is_nonzero_and_finite():
    s = torch.randn(3, 20, requires_grad=True)
    out = sinkhorn_topk_ste(s, 5, eps=0.3)
    out.sum().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0
