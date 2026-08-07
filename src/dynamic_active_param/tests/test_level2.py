"""Tests for the Level-2 cross-expert selection primitives.

Covers ``select_global_topB`` (the pooled global top-B keep-mask used by both
``oracle_mag`` and ``pubsub``), the ``beta`` sharpness knob on ``pivchol_global``
(M4), and the offline pubsub artifact math (public deflation orthogonality).
"""

import torch
import pytest

from src.dynamic_active_param.allocate import (
    select_global_topB,
    allocate_budgets,
    _CROSS_EXPERT_CRITERIA,
)


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


# --------------------------------------------------------------------------
# Q1/Q2 cross-expert criteria: oracle_mag_noW, oracle_up (block-level)
# --------------------------------------------------------------------------

import types

import torch.nn.functional as F

from src.dynamic_active_param.block import dynamic_moe_block_forward, _cross_expert_keep
from src.dynamic_active_param.tests.test_block import TinyMoEBlock


def test_new_criteria_registered():
    for c in ("oracle_mag", "oracle_mag_noW", "oracle_up"):
        assert c in _CROSS_EXPERT_CRITERIA


def _install_cross(block, B, I, criterion, col_norm=None):
    block._dyn_B = B
    block._dyn_k_min = 0
    block._dyn_I = I
    block._dyn_criterion = criterion
    if col_norm is not None:
        block._dyn_col_norm = col_norm
    block.forward = types.MethodType(dynamic_moe_block_forward, block)


def _col_norm(block, E, I):
    return torch.stack(
        [e.down_proj.weight.detach().float().norm(dim=0) for e in block.experts], dim=0
    )  # (E, I)


@pytest.mark.parametrize("criterion", ["oracle_mag", "oracle_mag_noW", "oracle_up"])
def test_cross_expert_rho_one_equals_reference(criterion):
    # B = K*I keeps every channel, so any global-top-B selector must reproduce
    # the full-width reference output exactly (masking is a no-op at rho=1).
    torch.manual_seed(0)
    H, I, E, K = 16, 32, 8, 4
    block = TinyMoEBlock(H, I, E, K)
    x = torch.randn(2, 5, H)
    ref_out, _ = block.forward(x)

    cn = _col_norm(block, E, I)
    _install_cross(block, B=K * I, I=I, criterion=criterion, col_norm=cn)
    dyn_out, _ = block.forward(x)
    assert torch.allclose(ref_out, dyn_out, atol=1e-5), f"{criterion} rho=1 must match reference"


def test_oracle_mag_noW_differs_from_oracle_mag():
    # With a non-uniform down_proj column norm, dropping the ||W_down|| factor
    # (Q1) must change which channels are kept for at least some tokens.
    torch.manual_seed(1)
    H, I, E, K = 16, 48, 8, 4
    block = TinyMoEBlock(H, I, E, K)
    # make column norms strongly non-uniform so the factor matters.
    for e in block.experts:
        with torch.no_grad():
            e.down_proj.weight.mul_(torch.linspace(0.1, 3.0, I).unsqueeze(0))
    x = torch.randn(1, 40, H)
    hs = x.view(-1, H)
    rl = block.gate(hs)
    rw = F.softmax(rl, dim=1, dtype=torch.float)
    rw, sel = torch.topk(rw, K, dim=-1)
    rw /= rw.sum(dim=-1, keepdim=True)
    rw = rw.to(hs.dtype)

    cn = _col_norm(block, E, I)
    B = round(0.5 * K * I)

    block._dyn_B, block._dyn_I, block._dyn_col_norm = B, I, cn
    block._dyn_criterion = "oracle_mag"
    _, keep_full = _cross_expert_keep(block, hs, rw, sel)
    block._dyn_criterion = "oracle_mag_noW"
    _, keep_noW = _cross_expert_keep(block, hs, rw, sel)

    assert not torch.equal(keep_full, keep_noW), "dropping ||W_down|| should change selection"
    assert torch.all(keep_full.reshape(hs.shape[0], -1).sum(1) == B)
    assert torch.all(keep_noW.reshape(hs.shape[0], -1).sum(1) == B)


def test_oracle_up_ranks_by_up_activation():
    # oracle_up scores by g*|up|*||W_down||; verify the keep-mask matches a hand
    # recomputation from up_proj outputs (not the SwiGLU intermediate).
    torch.manual_seed(2)
    H, I, E, K = 16, 40, 6, 3
    block = TinyMoEBlock(H, I, E, K)
    x = torch.randn(1, 20, H)
    hs = x.view(-1, H)
    rl = block.gate(hs)
    rw = F.softmax(rl, dim=1, dtype=torch.float)
    rw, sel = torch.topk(rw, K, dim=-1)
    rw /= rw.sum(dim=-1, keepdim=True)
    rw = rw.to(hs.dtype)

    cn = _col_norm(block, E, I)
    B = round(0.4 * K * I)
    block._dyn_B, block._dyn_I, block._dyn_col_norm = B, I, cn
    block._dyn_criterion = "oracle_up"
    _, keep = _cross_expert_keep(block, hs, rw, sel)

    # reference score from up_proj outputs directly.
    T = hs.shape[0]
    up_all = torch.zeros((T, K, I))
    for t in range(T):
        for j in range(K):
            e = int(sel[t, j])
            up_all[t, j] = block.experts[e].up_proj(hs[t])
    ref_score = rw.float().unsqueeze(-1) * up_all.abs() * cn[sel]
    ref_keep = select_global_topB(ref_score, B)
    assert torch.equal(keep, ref_keep), "oracle_up keep-mask must match up-based score"


# --------------------------------------------------------------------------
# reduce-top-k stacked with a cross-expert criterion (fewer AND narrower)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("criterion", ["oracle_mag_noW", "oracle_up"])
def test_cross_expert_under_reduced_topk(criterion):
    # Lowering block.top_k must shrink the pooled candidate set to K_new*I, so a
    # budget B = rho*K_new*I keeps exactly B channels drawn from only the token's
    # K_new highest-probability experts.
    torch.manual_seed(4)
    H, I, E, K0, K_new = 16, 32, 8, 8, 4
    block = TinyMoEBlock(H, I, E, K0)
    block.top_k = K_new  # what merge_slim_eval's reduce_topk does per block

    x = torch.randn(1, 24, H)
    T = x.shape[1]
    B = round(0.5 * K_new * I)
    _install_cross(block, B=B, I=I, criterion=criterion, col_norm=_col_norm(block, E, I))
    out, _ = block.forward(x)

    assert out.shape == (1, T, H)
    # keep-mask is over (T, K_new, I), so budget conservation is against K_new*I.
    hs = x.view(-1, H)
    rl = block.gate(hs)
    rw = F.softmax(rl, dim=1, dtype=torch.float)
    rw, sel = torch.topk(rw, block.top_k, dim=-1)
    rw /= rw.sum(dim=-1, keepdim=True)
    _, keep = _cross_expert_keep(block, hs, rw.to(hs.dtype), sel)
    assert keep.shape == (T, K_new, I)
    assert torch.all(keep.reshape(T, -1).sum(1) == B)


def test_reduced_topk_rho_one_equals_topk_reference():
    # At rho=1 the stacked config must equal plain reduce-top-k (masking is a
    # no-op), which is what makes the two reductions cleanly multiplicative.
    torch.manual_seed(5)
    H, I, E, K0, K_new = 16, 32, 8, 8, 4
    block = TinyMoEBlock(H, I, E, K0)
    block.top_k = K_new
    x = torch.randn(2, 6, H)
    ref_out, _ = TinyMoEBlock.forward(block, x)  # reduce-top-k only

    _install_cross(block, B=K_new * I, I=I, criterion="oracle_mag_noW")
    dyn_out, _ = block.forward(x)
    assert torch.allclose(ref_out, dyn_out, atol=1e-5)
