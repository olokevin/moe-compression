"""Tests for the cheap block-low-rank channel scorers.

Covers the factorization contract (shapes, singular values living in the
input-side core, exactness at full rank, BTT > SVD at equal cost), the online
proxy kernel, the cost accounting, and the ``lowrank_scorer`` block forward.
"""

import types

import torch
import torch.nn.functional as F
import pytest

from src.dynamic_active_param.allocate import _CROSS_EXPERT_CRITERIA
from src.dynamic_active_param.block import dynamic_moe_block_forward
from src.dynamic_active_param.lowrank_scorer import (
    build_layer_scorer,
    factorize_blocks,
    report_scorer_accounting,
    resolve_scorer_grid,
    scorer_cost_fraction,
    scorer_proxy,
)
from src.dynamic_active_param.tests.test_block import TinyMoEBlock


# --------------------------------------------------------------------------
# grid resolution
# --------------------------------------------------------------------------

def test_resolve_grid_ok():
    assert resolve_scorer_grid(64, 32, 4, 2) == (4, 2, 16, 16)
    assert resolve_scorer_grid(64, 32, 1, 1) == (1, 1, 64, 32)


def test_resolve_grid_rejects_non_divisor():
    with pytest.raises(ValueError, match="must divide"):
        resolve_scorer_grid(64, 32, 5, 1)
    with pytest.raises(ValueError, match="positive"):
        resolve_scorer_grid(64, 32, 0, 1)


# --------------------------------------------------------------------------
# factorization
# --------------------------------------------------------------------------

def test_factorize_shapes_and_channel_order():
    torch.manual_seed(0)
    E, I, H = 3, 64, 32
    W = torch.randn(E, I, H)
    sc = factorize_blocks(W, m=4, n=2, rank=8)
    assert sc.L_core.shape == (E, 4, 2, 16, 8)
    assert sc.R_core.shape == (E, 4, 2, 8, 16)
    assert (sc.m, sc.n, sc.a, sc.b, sc.rank) == (4, 2, 16, 16, 8)


def test_full_rank_factorization_is_exact():
    """At r = min(a,b) the block factorization reproduces W exactly, so the
    proxy equals the true projection output — the key sanity check that the
    (m,n) block reshape and the einsum channel order agree with ``W @ x``."""
    torch.manual_seed(0)
    E, I, H = 2, 32, 16
    W = torch.randn(E, I, H, dtype=torch.float64)
    for (m, n) in [(1, 1), (2, 2), (4, 1), (1, 2)]:
        a, b = I // m, H // n
        sc = factorize_blocks(W, m=m, n=n, rank=min(a, b), niter=8)
        x = torch.randn(5, H, dtype=torch.float64)
        for e in range(E):
            got = scorer_proxy(x, sc.L_core[e], sc.R_core[e])
            want = x @ W[e].t()
            assert torch.allclose(got, want, atol=1e-6), f"grid m={m},n={n}"


def test_singular_values_live_in_input_core():
    """The request: merge S into the smaller input-side core. So R's rows carry
    the singular values (norms = S, descending) while L's columns are
    orthonormal."""
    torch.manual_seed(0)
    W = torch.randn(1, 32, 16, dtype=torch.float64)
    sc = factorize_blocks(W, m=1, n=1, rank=6, niter=10)
    L = sc.L_core[0, 0, 0]        # (a, r)
    R = sc.R_core[0, 0, 0]        # (r, b)

    # L orthonormal columns
    assert torch.allclose(L.t() @ L, torch.eye(6, dtype=L.dtype), atol=1e-6)
    # R row norms == top-6 singular values, descending
    S_true = torch.linalg.svdvals(W[0])[:6]
    assert torch.allclose(R.norm(dim=1), S_true, rtol=1e-5)
    assert torch.all(R.norm(dim=1)[:-1] >= R.norm(dim=1)[1:] - 1e-9)


def _rel_err(W, m, n, rank, T=512, seed=0):
    torch.manual_seed(seed)
    sc = factorize_blocks(W, m=m, n=n, rank=rank, niter=10)
    x = torch.randn(T, W.shape[-1])
    got = scorer_proxy(x, sc.L_core[0], sc.R_core[0])
    want = x @ W[0].t()
    return ((got - want).norm() / want.norm()).item()


def test_btt_expresses_higher_rank_at_equal_cost():
    """The structural motivation for BTT over plain SVD: at the *same* FLOP cost a
    block grid can express a much higher-rank map. Each row-block spans rank
    ``min(a, n*r)``, so the grid reaches ``m*min(a, n*r)`` versus the global SVD's
    ``r``. Whether the extra rank *helps* depends on the matrix (see below) — the
    rank budget itself is unconditional."""
    I, H = 64, 64
    # equal cost: c = r(mH + nI)/(IH). m=n=1,r=8 vs m=n=2,r=4 -> same c.
    assert scorer_cost_fraction(I, H, 1, 1, 8) == pytest.approx(
        scorer_cost_fraction(I, H, 2, 2, 4)
    ), "test premise: equal cost"
    svd_max_rank = 8
    btt_max_rank = 2 * min(I // 2, 2 * 4)
    assert btt_max_rank > svd_max_rank


def test_btt_beats_svd_on_block_structured_matrix():
    """Where BTT is designed to win: a matrix whose blocks are individually
    low-rank. Here the grid is exact at r=4 while a global rank-8 SVD is not."""
    torch.manual_seed(0)
    I = H = 64
    a = b = 32
    # 2x2 grid of exactly-rank-4 blocks -> global rank up to 16.
    blocks = [[torch.randn(a, 4) @ torch.randn(4, b) for _ in range(2)] for _ in range(2)]
    W = torch.cat([torch.cat(row, dim=1) for row in blocks], dim=0).unsqueeze(0)

    err_btt = _rel_err(W, m=2, n=2, rank=4)
    err_svd = _rel_err(W, m=1, n=1, rank=8)
    assert err_btt < 1e-4, "BTT is exact on its own structure"
    assert err_btt < err_svd


def test_svd_not_beaten_on_unstructured_matrix():
    """Counterpoint, recorded so the BTT claim is not over-read: on an iid
    Gaussian (flat spectrum, no block structure) BTT buys nothing over SVD at
    equal cost — the two are within a few % of each other. Which one wins on real
    expert weights is an empirical question, measured in
    ``scripts/lowrank_scorer_recall.py``."""
    torch.manual_seed(0)
    W = torch.randn(1, 64, 64)
    err_btt = _rel_err(W, m=2, n=2, rank=4)
    err_svd = _rel_err(W, m=1, n=1, rank=8)
    assert abs(err_btt - err_svd) < 0.05 * err_svd


def test_build_layer_scorer_from_experts():
    torch.manual_seed(0)
    block = TinyMoEBlock(H=32, I=64, E=4, K=2)
    sc = build_layer_scorer(block.experts, "up_proj", m=2, n=2, rank=4)
    assert sc.L_core.shape == (4, 2, 2, 32, 4)
    # cores must match the stacked weights they came from
    W = torch.stack([e.up_proj.weight.detach() for e in block.experts], 0)
    full = build_layer_scorer(block.experts, "up_proj", m=1, n=1, rank=32, niter=10)
    x = torch.randn(4, 32)
    for e in range(4):
        got = scorer_proxy(x, full.L_core[e], full.R_core[e])
        assert torch.allclose(got, x @ W[e].t(), atol=1e-3)


# --------------------------------------------------------------------------
# cost accounting
# --------------------------------------------------------------------------

def test_cost_fraction_formula():
    I, H, m, n, r = 768, 2048, 4, 2, 16
    want = r * (m * H + n * I) / (I * H)
    assert scorer_cost_fraction(I, H, m, n, r) == pytest.approx(want)


def test_accounting_beats_oracle_variants():
    """At the same nominal rho the scorer scheme must keep less of the FFN than
    both oracle_mag ((1+1+rho)/3) and oracle_up ((1+2rho)/3), otherwise the whole
    exercise is pointless."""
    acc = report_scorer_accounting(
        I=768, H=2048, m=1, n=1, rank=16, n_scorers=2, prune_ratio=0.75
    )
    assert acc["rho"] == pytest.approx(0.25)
    assert acc["kept_fraction"] < acc["oracle_up_kept"] < acc["oracle_mag_kept"]
    # scorer overhead should be small (a few % of the FFN) at rank 16
    assert acc["scorer_overhead_ffn"] < 0.05


# --------------------------------------------------------------------------
# block forward
# --------------------------------------------------------------------------

def _install_scorer(block, B, I, use_gate=True, m=1, n=1, rank=None):
    if rank is None:
        rank = min(I // m, block.gate.in_features // n)  # full rank => exact
    block._dyn_sc_up = build_layer_scorer(
        block.experts, "up_proj", m=m, n=n, rank=rank, niter=10
    )
    block._dyn_sc_gate = (
        build_layer_scorer(block.experts, "gate_proj", m=m, n=n, rank=rank, niter=10)
        if use_gate else None
    )
    block._dyn_B = B
    block._dyn_k_min = 0
    block._dyn_I = I
    block._dyn_criterion = "lowrank_scorer"
    block.forward = types.MethodType(dynamic_moe_block_forward, block)


def test_lowrank_scorer_registered():
    assert "lowrank_scorer" in _CROSS_EXPERT_CRITERIA


def test_rho_one_equals_reference():
    """B = K*I keeps every channel, so the output must match the dense block
    exactly regardless of how good the proxy is."""
    torch.manual_seed(0)
    H, I, E, K = 32, 64, 8, 4
    block = TinyMoEBlock(H, I, E, K)
    x = torch.randn(2, 6, H)
    ref, _ = TinyMoEBlock.forward(block, x)
    _install_scorer(block, B=K * I, I=I)
    got, _ = block.forward(x)
    assert torch.allclose(got, ref, atol=1e-5)


def test_full_rank_proxy_matches_oracle_selection():
    """With full-rank cores the proxy IS the true intermediate, so
    lowrank_scorer must reproduce oracle_mag_noW exactly. This pins the proxy
    formula (SiLU(gate_hat) * up_hat) to the oracle it approximates."""
    torch.manual_seed(0)
    H, I, E, K = 32, 32, 8, 4
    block = TinyMoEBlock(H, I, E, K)
    x = torch.randn(2, 6, H)
    B = K * I // 4

    _install_scorer(block, B=B, I=I, use_gate=True, rank=min(I, H))
    got, _ = block.forward(x)

    block2 = TinyMoEBlock(H, I, E, K)
    block2.load_state_dict(block.state_dict())
    block2._dyn_B, block2._dyn_k_min, block2._dyn_I = B, 0, I
    block2._dyn_criterion = "oracle_mag_noW"
    block2.forward = types.MethodType(dynamic_moe_block_forward, block2)
    want, _ = block2.forward(x)

    assert torch.allclose(got, want, atol=1e-5)


def test_budget_is_respected_and_output_changes():
    torch.manual_seed(0)
    H, I, E, K = 32, 64, 8, 4
    block = TinyMoEBlock(H, I, E, K)
    x = torch.randn(2, 6, H)
    ref, _ = TinyMoEBlock.forward(block, x)
    _install_scorer(block, B=K * I // 8, I=I, m=2, n=2, rank=4)
    got, _ = block.forward(x)
    assert got.shape == ref.shape
    assert not torch.allclose(got, ref, atol=1e-3), "a 1/8 budget must change output"


def test_up_only_mode_runs_and_differs_from_up_gate():
    torch.manual_seed(0)
    H, I, E, K = 32, 64, 8, 4
    block = TinyMoEBlock(H, I, E, K)
    x = torch.randn(2, 6, H)
    B = K * I // 4

    _install_scorer(block, B=B, I=I, use_gate=True, m=2, n=2, rank=4)
    with_gate, _ = block.forward(x)
    _install_scorer(block, B=B, I=I, use_gate=False, m=2, n=2, rank=4)
    up_only, _ = block.forward(x)
    assert not torch.allclose(with_gate, up_only, atol=1e-4)
