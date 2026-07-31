"""Tests for the Level-2 cross-expert selection primitives.

Covers ``select_global_topB`` (the pooled global top-B keep-mask used by both
``oracle_mag`` and ``pubsub``), the ``beta`` sharpness knob on ``pivchol_global``
(M4), and the offline pubsub artifact math (public deflation orthogonality).
"""

import torch
import pytest

from src.dynamic_active_param.allocate import select_global_topB, allocate_budgets


def _rand_topk(T, K, E, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, E, generator=g)
    weights = torch.softmax(logits, dim=1)
    w, sel = torch.topk(weights, K, dim=1)
    w = w / w.sum(dim=1, keepdim=True)
    return w, sel


# --------------------------------------------------------------------------
# select_global_topB
# --------------------------------------------------------------------------

def test_global_topB_conserves_budget():
    T, K, I = 100, 4, 64
    B = round(0.5 * K * I)
    score = torch.rand(T, K, I)
    keep = select_global_topB(score, B)
    assert keep.shape == (T, K, I)
    assert keep.dtype == torch.bool
    assert torch.all(keep.reshape(T, -1).sum(dim=1) == B), "must keep exactly B per token"


def test_global_topB_picks_highest():
    # Single token, hand-checkable: the B highest scores are kept.
    score = torch.tensor([[[3.0, 1.0], [4.0, 2.0]]])  # (1, K=2, I=2)
    keep = select_global_topB(score, B=2)[0]
    # top-2 of {3,1,4,2} = {4,3} -> (slot0,ch0) and (slot1,ch0)
    assert keep[1, 0] and keep[0, 0]
    assert not keep[0, 1] and not keep[1, 1]


def test_global_topB_forced_keep_inf():
    # +inf entries are always kept.
    score = torch.rand(5, 3, 8)
    score[:, 2, 7] = float("inf")
    keep = select_global_topB(score, B=6)
    assert torch.all(keep[:, 2, 7]), "+inf channel must always be kept"


def test_global_topB_rho_one_keeps_all():
    T, K, I = 10, 3, 16
    keep = select_global_topB(torch.rand(T, K, I), B=K * I)
    assert torch.all(keep), "B = K*I keeps everything"


def test_global_topB_chunking_matches():
    T, K, I = 9000, 4, 32  # > default chunk 4096
    B = round(0.5 * K * I)
    score = torch.rand(T, K, I)
    keep = select_global_topB(score, B)
    assert torch.all(keep.reshape(T, -1).sum(dim=1) == B)


def test_global_topB_infeasible_raises():
    with pytest.raises(ValueError):
        select_global_topB(torch.rand(3, 2, 4), B=100)


# --------------------------------------------------------------------------
# beta sharpness knob (M4) on pivchol_global
# --------------------------------------------------------------------------

def _monotone_gains(E, I, seed=0):
    g = torch.Generator().manual_seed(seed)
    raw = torch.rand(E, I, generator=g) + 1e-3
    return torch.sort(raw, dim=1, descending=True).values


def test_beta_one_matches_default():
    T, K, E, I = 50, 4, 16, 64
    B = round(0.5 * K * I)
    w, sel = _rand_topk(T, K, E, seed=31)
    gains = _monotone_gains(E, I, seed=31)
    k1 = allocate_budgets(w, sel, None, B, 0, I, criterion="pivchol_global", gains=gains)
    kb = allocate_budgets(w, sel, None, B, 0, I, criterion="pivchol_global", gains=gains, beta=1.0)
    assert torch.equal(k1, kb), "beta=1.0 must reproduce the default g^2 path"


def test_beta_sharpens_toward_top_expert():
    # Higher beta concentrates budget on the highest-g expert (-> reduce-top-k).
    I, K = 64, 4
    B = round(0.5 * K * I)
    w = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    sel = torch.tensor([[0, 1, 2, 3]])
    gains = torch.ones(4, I)  # equal gains: g alone decides
    k_lo = allocate_budgets(w, sel, None, B, 0, I, criterion="pivchol_global", gains=gains, beta=1.0)[0]
    k_hi = allocate_budgets(w, sel, None, B, 0, I, criterion="pivchol_global", gains=gains, beta=3.0)[0]
    # top expert should get at least as many channels under higher beta.
    assert k_hi[0] >= k_lo[0], f"beta should sharpen toward top expert: {k_lo} -> {k_hi}"
    assert k_hi.sum() == B and k_lo.sum() == B


# --------------------------------------------------------------------------
# pubsub artifact math (CPU, tiny synthetic model-free check)
# --------------------------------------------------------------------------

def test_public_deflation_is_orthogonal():
    # (I - U U^T) W has zero projection onto U's columns.
    d, m, r = 32, 16, 4
    torch.manual_seed(3)
    W = torch.randn(d, m)
    M = W @ W.t()
    M = 0.5 * (M + M.t())
    _, evecs = torch.linalg.eigh(M)
    U = evecs[:, -r:]                       # (d, r)
    Wt = W - U @ (U.t() @ W)                # deflated
    resid = U.t() @ Wt                      # should be ~0
    assert resid.abs().max() < 1e-4, f"deflation not orthogonal: {resid.abs().max()}"
