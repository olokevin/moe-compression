"""Tests for the low-precision / input-sparse channel probe.

The load-bearing test is :func:`test_probe_at_full_precision_is_oracle_mag_noW`:
at ``bits>=16`` and ``rho_input=1.0`` the probe *is* the exact intermediate, so
the criterion must reproduce ``oracle_mag_noW`` bit-for-bit. Everything else the
criterion claims (that a b-bit, input-sparse proxy picks nearly the same channels)
is an empirical question measured by ``scripts/probe_frontier.py``; this file
guarantees the plumbing around it is not silently different.
"""

import types

import pytest
import torch
import torch.nn.functional as F

from src.dynamic_active_param.allocate import _CROSS_EXPERT_CRITERIA
from src.dynamic_active_param.block import dynamic_moe_block_forward
from src.dynamic_active_param.sparse_probe import (
    allocate_input_reads,
    build_layer_probe,
    descending_abs_ranks,
    probe_cost_per_matrix,
    probe_expert_scores,
    quantize_rtn_dequant,
    report_probe_accounting,
    used_param_fraction,
    sparsify_input_by_count,
    sparsify_input_topk,
)
from src.dynamic_active_param.tests.test_block import TinyMoEBlock


def _install_probe(block, B, I, bits=3, group=8, rho_input=0.25,
                   use_gate=True, lam=1.0, criterion="sparse_probe",
                   input_alloc="uniform"):
    block._dyn_B = B
    block._dyn_k_min = 0
    block._dyn_I = I
    block._dyn_criterion = criterion
    block._dyn_ranks = torch.arange(I).unsqueeze(0).repeat(block.num_experts, 1).long()
    block._dyn_contrib = torch.rand(block.num_experts)
    block._dyn_prefix = None
    block._dyn_gains = None
    block._dyn_beta = 1.0
    if criterion == "sparse_probe":
        block._dyn_probe = build_layer_probe(
            block.experts, bits=bits, group=group, use_gate=use_gate,
            rho_input=rho_input, input_alloc=input_alloc)
        block._dyn_probe_lam = lam
    block.forward = types.MethodType(dynamic_moe_block_forward, block)
    return block


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def test_criterion_is_registered_cross_expert():
    assert "sparse_probe" in _CROSS_EXPERT_CRITERIA


# --------------------------------------------------------------------------
# quantizer
# --------------------------------------------------------------------------

def test_rtn_is_exact_at_16_bits():
    W = torch.randn(2, 8, 16)
    assert torch.equal(quantize_rtn_dequant(W, 16, 8), W)


def test_rtn_error_shrinks_with_bits():
    torch.manual_seed(0)
    W = torch.randn(2, 32, 64)
    errs = [float((quantize_rtn_dequant(W, b, 32) - W).norm() / W.norm())
            for b in (2, 3, 4, 8)]
    assert errs == sorted(errs, reverse=True), f"error must fall with bits: {errs}"
    assert errs[-1] < 0.01, "8-bit RTN should be near-exact"


def test_rtn_group_size_falls_back_when_not_divisor():
    W = torch.randn(1, 4, 10)                      # H=10 not divisible by 4
    out = quantize_rtn_dequant(W, 4, 4)            # must not raise
    assert out.shape == W.shape


def test_rtn_rejects_one_bit():
    with pytest.raises(ValueError, match="too small"):
        quantize_rtn_dequant(torch.randn(1, 2, 4), 1, 4)


# --------------------------------------------------------------------------
# input sparsification
# --------------------------------------------------------------------------

def test_sparsify_keeps_exactly_k_largest():
    x = torch.tensor([[1.0, -5.0, 2.0, 0.5, -3.0, 0.1, 0.2, 4.0]])
    out = sparsify_input_topk(x, 0.25)              # k = 2
    assert int((out != 0).sum()) == 2
    assert out[0, 1] == -5.0 and out[0, 7] == 4.0   # kept by |.|, sign preserved


def test_sparsify_identity_at_keep_one():
    x = torch.randn(4, 16)
    assert torch.equal(sparsify_input_topk(x, 1.0), x)
    assert torch.equal(sparsify_input_topk(x, None), x)


def test_sparsify_keeps_at_least_one():
    x = torch.randn(3, 16)
    assert int((sparsify_input_topk(x, 1e-9) != 0).sum()) == 3


# --------------------------------------------------------------------------
# probe kernel
# --------------------------------------------------------------------------

def test_probe_matches_exact_intermediate_at_full_precision():
    torch.manual_seed(0)
    block = TinyMoEBlock(H=16, I=8, E=3, K=2)
    probe = build_layer_probe(block.experts, bits=16, group=8, use_gate=True,
                             rho_input=1.0)
    x = torch.randn(5, 16)
    for e in range(3):
        got = probe_expert_scores(x, probe, e)
        ex = block.experts[e]
        want = (F.silu(ex.gate_proj(x)) * ex.up_proj(x)).abs()
        assert torch.allclose(got, want, atol=1e-6), f"expert {e}"


def test_up_only_probe_ignores_gate():
    torch.manual_seed(0)
    block = TinyMoEBlock(H=16, I=8, E=2, K=2)
    probe = build_layer_probe(block.experts, bits=16, group=8, use_gate=False,
                             rho_input=1.0)
    assert probe.Wg_q is None
    x = torch.randn(4, 16)
    want = block.experts[0].up_proj(x).abs()
    assert torch.allclose(probe_expert_scores(x, probe, 0), want, atol=1e-6)


# --------------------------------------------------------------------------
# the anchor: full precision + dense input == oracle_mag_noW, bit-exactly
# --------------------------------------------------------------------------

def test_probe_at_full_precision_is_oracle_mag_noW():
    """``bits=16, rho_input=1.0`` makes the proxy the exact intermediate, so the
    keep-mask -- and hence the block output -- must equal ``oracle_mag_noW``'s.

    This is the gate before any eval: it is the same semantics guarantee that made
    the earlier ``lowrank_scorer`` results trustworthy.
    """
    torch.manual_seed(0)
    H, I, E, K = 16, 8, 4, 2
    x = torch.randn(2, 5, H)

    def run(criterion, **kw):
        torch.manual_seed(0)
        blk = TinyMoEBlock(H, I, E, K)
        _install_probe(blk, B=round(0.5 * K * I), I=I, criterion=criterion, **kw)
        with torch.no_grad():
            return blk(x)[0]

    want = run("oracle_mag_noW")
    got = run("sparse_probe", bits=16, group=8, rho_input=1.0, use_gate=True)
    assert torch.equal(got, want), "full-precision probe must reproduce oracle_mag_noW"


def test_cascade_lambda_one_equals_non_cascade():
    torch.manual_seed(0)
    H, I, E, K = 16, 8, 4, 2
    x = torch.randn(2, 5, H)

    def run(lam):
        torch.manual_seed(0)
        blk = TinyMoEBlock(H, I, E, K)
        _install_probe(blk, B=round(0.5 * K * I), I=I, bits=3, group=8,
                       rho_input=0.5, lam=lam)
        with torch.no_grad():
            return blk(x)[0]

    assert torch.equal(run(1.0), run(1.0))
    # lam = 1 must take the non-cascade branch and agree with it
    assert torch.allclose(run(1.0), run(0.999), atol=0)


def test_cascade_at_large_lambda_recovers_oracle():
    """With a candidate pool as large as K*I the exact re-rank sees every channel,
    so the cascade must reproduce ``oracle_mag_noW`` no matter how bad the probe."""
    torch.manual_seed(0)
    H, I, E, K = 16, 8, 4, 2
    x = torch.randn(2, 5, H)

    torch.manual_seed(0)
    ref = TinyMoEBlock(H, I, E, K)
    _install_probe(ref, B=round(0.5 * K * I), I=I, criterion="oracle_mag_noW")
    torch.manual_seed(0)
    cas = TinyMoEBlock(H, I, E, K)
    _install_probe(cas, B=round(0.5 * K * I), I=I, bits=2, group=8,
                   rho_input=0.125, lam=float(K * I))
    with torch.no_grad():
        assert torch.equal(cas(x)[0], ref(x)[0])


def test_budget_is_conserved():
    torch.manual_seed(0)
    H, I, E, K = 16, 8, 4, 2
    B = round(0.375 * K * I)
    blk = _install_probe(TinyMoEBlock(H, I, E, K), B=B, I=I, bits=3, group=8,
                         rho_input=0.5)
    seen = {}
    orig = dynamic_moe_block_forward

    from src.dynamic_active_param import block as blkmod
    real = blkmod._cross_expert_keep

    def spy(self, hs, rw, se):
        inter, keep = real(self, hs, rw, se)
        seen["keep"] = keep
        return inter, keep

    blkmod._cross_expert_keep = spy
    try:
        with torch.no_grad():
            blk(torch.randn(2, 5, H))
    finally:
        blkmod._cross_expert_keep = real
    keep = seen["keep"]
    T = keep.shape[0]
    assert torch.all(keep.reshape(T, -1).sum(dim=1) == B)


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------

def test_cost_per_matrix_closed_form():
    # 3 bits + fp16 scale per 128 -> 3.125 bits/weight; read on 25% of x.
    assert probe_cost_per_matrix(3, 128, 0.25) == pytest.approx(3.125 / 16 * 0.25)
    # bits >= 16 is the reuse regime: the probe IS the served weight, so it reads
    # p of a matrix and pays for no group scale of its own.
    assert probe_cost_per_matrix(16, 128, 1.0) == pytest.approx(1.0)
    assert probe_cost_per_matrix(16, 128, 0.25) == pytest.approx(0.25)


def test_accounting_beats_oracle_mag_and_oracle_up():
    a = report_probe_accounting(bits=3, group=128, rho_input=0.25,
                                use_gate=True, rho_channel=0.125)
    assert a["rho"] == pytest.approx(0.125)
    # probe = 2 * 3.125/16 * 0.25 = 0.09766 of one matrix
    assert a["probe_bytes_total"] == pytest.approx(0.09766, abs=1e-4)
    assert a["kept_fraction"] == pytest.approx(0.125 + 0.09766 / 3, abs=1e-4)
    assert a["whole_ffn_cut"] > 0.84
    assert a["kept_fraction"] < a["oracle_up_kept"] < a["oracle_mag_kept"]


def test_accounting_cascade_costs_more():
    base = report_probe_accounting(3, 128, 0.25, True, 0.125, lam=1.0)
    casc = report_probe_accounting(3, 128, 0.25, True, 0.125, lam=1.5)
    # extra = 2(lam-1)*rho / 3
    assert casc["kept_fraction"] - base["kept_fraction"] == pytest.approx(
        2 * 0.5 * 0.125 / 3, abs=1e-6)


def test_accounting_up_only_is_half_the_probe():
    both = report_probe_accounting(3, 128, 0.25, True, 0.125)
    up = report_probe_accounting(3, 128, 0.25, False, 0.125)
    assert up["probe_bytes_total"] == pytest.approx(both["probe_bytes_total"] / 2)


# --------------------------------------------------------------------------
# reuse regime: bits >= 16 aliases the served weights (zero extra storage)
# --------------------------------------------------------------------------

def test_reuse_probe_aliases_served_weights_without_copying():
    """At ``bits>=16`` the probe must *view* the expert weights, not copy them.

    A stacked fp16 copy of up+gate is ~39 GB for Qwen3-30B, so this is a hard
    memory requirement, not a nicety: ``data_ptr`` equality is the check.
    """
    block = TinyMoEBlock(H=16, I=8, E=3, K=2)
    probe = build_layer_probe(block.experts, bits=16, group=8, use_gate=True,
                              rho_input=0.25)
    for e, ex in enumerate(block.experts):
        assert probe.Wu_q[e].data_ptr() == ex.up_proj.weight.data_ptr()
        assert probe.Wg_q[e].data_ptr() == ex.gate_proj.weight.data_ptr()


def test_reuse_regime_has_zero_extra_storage():
    a = report_probe_accounting(bits=16, group=128, rho_input=0.25,
                                use_gate=True, rho_channel=0.125)
    assert a["extra_storage_frac_of_experts"] == 0.0
    q = report_probe_accounting(bits=3, group=128, rho_input=0.25,
                                use_gate=True, rho_channel=0.125)
    assert q["extra_storage_frac_of_experts"] > 0.12


def test_used_param_fraction_closed_form():
    """Used params = **scoring params + compute params**, in units of a 3-matrix FFN.

    Both arguments are **keep** fractions: ``rho_input`` of the input coordinates are
    read for scoring, ``rho_channel`` of the pooled channels are kept for compute::

        used = (n*rho_input + 3*rho_channel) / 3 = rho_channel + n*rho_input/3

    Deliberately conservative: the two passes overlap on the kept rows and this
    bills that overlap twice rather than discounting it. Hence at ``rho_input -> 1``
    it must come out ABOVE a single-pass exact scorer (``oracle_mag``'s
    ``(1+1+rho_channel)/3``) by exactly ``2*rho_channel/3`` -- a two-pass scheme that
    reads everything really does read the kept rows twice. ``rho_input -> 0`` gives
    bare ``rho_channel``.
    """
    # rho_channel=0.125, rho_input=0.25, n=2 -> 0.125 + 2*0.25/3
    assert used_param_fraction(0.25, 0.125) == pytest.approx(0.125 + 2 * 0.25 / 3)
    # the headline operating point: rho_input=0.25, rho_channel=0.10 -> 0.2667 (-73.3%)
    assert used_param_fraction(0.25, 0.10) == pytest.approx(0.10 + 2 * 0.25 / 3)
    for rc in (0.0625, 0.10, 0.125, 0.25, 0.5):
        # rho_input=0 costs nothing beyond the kept channels themselves
        assert used_param_fraction(0.0, rc) == pytest.approx(rc), rc
        # rho_input=1 exceeds the single-pass oracle by exactly 2*rho_channel/3 (the
        # double-billed overlap): the frame is conservative, not flattering.
        assert used_param_fraction(1.0, rc) == pytest.approx(
            (1 + 1 + rc) / 3 + 2 * rc / 3), rc
        assert used_param_fraction(1.0, rc) >= (1 + 1 + rc) / 3 - 1e-12, rc
        # up-only (n=1) halves the scoring term
        assert used_param_fraction(0.5, rc, 1) == pytest.approx(rc + 0.5 / 3), rc
    # strictly increasing in rho_input, never below rho_channel, and the scoring term
    # must not depend on rho_channel at all (no (1-rho) coupling).
    prev = -1.0
    for ri in (0.0, 0.125, 0.25, 0.5, 1.0):
        k = used_param_fraction(ri, 0.10)
        assert k > prev and k >= 0.10 - 1e-12
        prev = k
    for ri in (0.125, 0.25, 0.5):
        a = used_param_fraction(ri, 0.10) - 0.10
        b = used_param_fraction(ri, 0.20) - 0.20
        assert a == pytest.approx(b), "scoring term must be independent of rho_channel"
    # keeping fewer channels always costs less
    assert used_param_fraction(0.25, 0.10) < used_param_fraction(0.25, 0.15)
    # both axes are keep fractions, so the symmetry check: swapping them is NOT the
    # same function (scoring is discounted by 1/3 per branch pair, compute is not).
    assert used_param_fraction(0.30, 0.10) != pytest.approx(
        used_param_fraction(0.10, 0.30))


def test_reuse_probe_dense_input_is_oracle_mag_noW():
    """The reuse probe at ``rho_input=1.0`` must still be the exact oracle."""
    H, I, E, K = 16, 8, 4, 2
    x = torch.randn(2, 5, H)

    def run(criterion, **kw):
        torch.manual_seed(0)
        blk = TinyMoEBlock(H, I, E, K)
        _install_probe(blk, B=round(0.5 * K * I), I=I, criterion=criterion, **kw)
        with torch.no_grad():
            return blk(x)[0]

    torch.manual_seed(0)
    want = run("oracle_mag_noW")
    got = run("sparse_probe", bits=16, group=8, rho_input=1.0, use_gate=True)
    assert torch.equal(got, want)


# --------------------------------------------------------------------------
# input-read allocation across a token's K experts
# --------------------------------------------------------------------------

def test_descending_ranks_are_the_inverse_permutation():
    x = torch.randn(4, 12)
    ranks, sorted_abs = descending_abs_ranks(x)
    # rank 0 is the largest-|x| coordinate
    assert torch.equal(x.abs().gather(-1, ranks.argmin(dim=-1, keepdim=True)),
                       sorted_abs[:, :1])
    # ranks is a permutation per row
    for r in ranks:
        assert sorted(r.tolist()) == list(range(12))
    # sorted_abs is descending
    assert torch.all(sorted_abs[:, :-1] >= sorted_abs[:, 1:])


def test_alloc_uniform_gives_every_expert_the_same_count():
    x = torch.randn(6, 32)
    _, sorted_abs = descending_abs_ranks(x)
    g = torch.rand(6, 4)
    n = allocate_input_reads(sorted_abs, g, keep=0.25, beta=0.0)
    assert torch.all(n == 8)


def test_alloc_conserves_the_pooled_budget():
    """Every term must spend exactly ``K*round(keep*H)`` reads per token."""
    torch.manual_seed(0)
    H, K = 64, 4
    x = torch.randn(20, H)
    _, sorted_abs = descending_abs_ranks(x)
    g = torch.softmax(torch.randn(20, K), dim=-1)
    for beta in (0.0, 1.0, 2.0):
        for keep in (0.125, 0.25, 0.5):
            n = allocate_input_reads(sorted_abs, g, keep=keep, beta=beta)
            want = K * max(1, round(keep * H))
            assert torch.all(n.sum(dim=1) == want), f"beta={beta} keep={keep}"
            # a dominated slot may get 0 reads (as in Level-1 pivchol); the only
            # hard bounds are non-negativity and the per-expert cap H.
            assert torch.all(n >= 0) and torch.all(n <= H)


def test_alloc_router_favors_high_routing_weight_experts():
    """Under beta>0 the dominant expert must read at least as much as the weakest."""
    torch.manual_seed(0)
    H, K = 64, 4
    x = torch.randn(32, H)
    _, sorted_abs = descending_abs_ranks(x)
    g = torch.softmax(torch.randn(32, K) * 2.0, dim=-1)
    top = g.argmax(dim=1)
    bot = g.argmin(dim=1)
    rows = torch.arange(g.shape[0])
    for beta in (1.0, 2.0):
        n = allocate_input_reads(sorted_abs, g, keep=0.25, beta=beta)
        assert torch.all(n[rows, top] >= n[rows, bot])
    # router2 concentrates at least as hard as router
    n1 = allocate_input_reads(sorted_abs, g, keep=0.25, beta=1.0)
    n2 = allocate_input_reads(sorted_abs, g, keep=0.25, beta=2.0)
    assert n2[rows, top].float().mean() >= n1[rows, top].float().mean()


def test_alloc_matches_bruteforce_pooled_topk():
    """The threshold form must equal an explicit pooled top-N over (slot, coord)."""
    torch.manual_seed(0)
    H, K, T = 32, 3, 8
    x = torch.randn(T, H)
    ranks, sorted_abs = descending_abs_ranks(x)
    g = torch.softmax(torch.randn(T, K), dim=-1)
    keep, beta = 0.25, 1.0
    n = allocate_input_reads(sorted_abs, g, keep=keep, beta=beta)
    N = K * max(1, round(keep * H))
    # brute force: score every (slot, coord) by g_e^beta * |x_i|, take top-N
    score = g.pow(beta).unsqueeze(-1) * x.abs().unsqueeze(1)      # (T,K,H)
    idx = score.reshape(T, K * H).topk(N, dim=1).indices
    want = torch.zeros(T, K, dtype=torch.long)
    want.scatter_add_(1, idx // H, torch.ones_like(idx))
    assert torch.equal(n, want)


def test_sparsify_by_count_keeps_the_requested_prefix():
    x = torch.randn(5, 16)
    ranks, _ = descending_abs_ranks(x)
    n = torch.tensor([1, 4, 8, 16, 2])
    out = sparsify_input_by_count(x, ranks, n)
    assert torch.equal((out != 0).sum(dim=-1), n)
    # and it keeps the LARGEST coordinates
    for t in range(5):
        kept = x[t][out[t] != 0].abs()
        dropped = x[t][out[t] == 0].abs()
        if dropped.numel():
            assert kept.min() >= dropped.max()


def test_uniform_alloc_matches_plain_topk_sparsify():
    """``input_alloc='uniform'`` must be exactly the original top-|x| path."""
    x = torch.randn(7, 32)
    ranks, sorted_abs = descending_abs_ranks(x)
    g = torch.rand(7, 4)
    n = allocate_input_reads(sorted_abs, g, keep=0.25, beta=0.0)
    got = sparsify_input_by_count(x, ranks, n[:, 0])
    assert torch.equal(got, sparsify_input_topk(x, 0.25))


def test_block_runs_for_every_input_alloc_term():
    """Plumbing: each term produces a finite output with the budget conserved."""
    H, I, E, K = 32, 8, 4, 2
    x = torch.randn(2, 6, H)
    outs = {}
    for alloc in ("uniform", "router", "router2", "colnorm"):
        torch.manual_seed(0)
        blk = TinyMoEBlock(H, I, E, K)
        _install_probe(blk, B=round(0.25 * K * I), I=I, bits=3, group=8,
                       rho_input=0.25, input_alloc=alloc)
        with torch.no_grad():
            y = blk(x)[0]
        assert torch.isfinite(y).all(), alloc
        outs[alloc] = y
    # the terms are genuinely different selectors, not aliases of each other
    assert not torch.equal(outs["uniform"], outs["router"])
    assert not torch.equal(outs["router"], outs["router2"])
