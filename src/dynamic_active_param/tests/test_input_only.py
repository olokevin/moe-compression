"""Tests for `input_only` — one-pass input sparsity (the sparse read IS the compute).

Two load-bearing anchors:

* :func:`test_rho_input_one_is_oracle_mag_noW` — at ``rho_input=1.0`` the sparse
  read is the dense read, so the block output must equal ``oracle_mag_noW``'s
  bit-for-bit. This is the one setting where the method has a known-correct
  reference, and it also pins that the sparse input actually reaches ``gate``/``up``
  (a bug that dropped it would pass every other test in this file).
* :func:`test_cost_agrees_with_oracle_mag_noW_at_rho_input_one` — the *accounting*
  must agree with ``oracle_mag_noW``'s at that same point. The two-pass frame in
  ``sparse_probe`` deliberately does not (it double-bills the kept rows), so this
  is the test that catches someone reusing the wrong closed form.

Everything else it claims — that a sparse-input intermediate is accurate enough to
serve as the FFN's actual output — is empirical, measured by
``scripts/input_only_error.py`` and the evals in
``docs/exps/dynamic_active_param/efficient_scorer.md``.
"""

import types

import pytest
import torch

from src.dynamic_active_param.allocate import _CROSS_EXPERT_CRITERIA
from src.dynamic_active_param.block import dynamic_moe_block_forward
from src.dynamic_active_param.input_only import (
    InputOnlyCfg,
    N_BRANCHES,
    report_input_only_accounting,
    solve_symmetric,
    used_param_fraction,
)
from src.dynamic_active_param.sparse_probe import (
    used_param_fraction as two_pass_used,
)
from src.dynamic_active_param.tests.test_block import TinyMoEBlock


def _install_io(block, B, I, rho_input=0.25, input_alloc="uniform",
                criterion="input_only"):
    block._dyn_B = B
    block._dyn_k_min = 0
    block._dyn_I = I
    block._dyn_criterion = criterion
    block._dyn_ranks = torch.arange(I).unsqueeze(0).repeat(block.num_experts, 1).long()
    block._dyn_contrib = torch.rand(block.num_experts)
    block._dyn_prefix = None
    block._dyn_gains = None
    block._dyn_beta = 1.0
    if criterion == "input_only":
        block._dyn_io = InputOnlyCfg(rho_input=rho_input, input_alloc=input_alloc)
    block.forward = types.MethodType(dynamic_moe_block_forward, block)
    return block


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def test_criterion_is_registered_cross_expert():
    assert "input_only" in _CROSS_EXPERT_CRITERIA


# --------------------------------------------------------------------------
# the anchors: rho_input=1 == oracle_mag_noW, in output AND in cost
# --------------------------------------------------------------------------

@pytest.mark.parametrize("input_alloc", ["uniform", "router"])
def test_rho_input_one_is_oracle_mag_noW(input_alloc):
    """Dense input => the "approximate" intermediate is the exact one."""
    torch.manual_seed(0)
    H, I, E, K, T = 16, 12, 4, 2, 6
    x = torch.randn(1, T, H)

    def run(criterion, **kw):
        torch.manual_seed(1)
        blk = TinyMoEBlock(H, I, E, K)
        _install_io(blk, B=round(0.5 * K * I), I=I, criterion=criterion, **kw)
        return blk(x)[0]

    got = run("input_only", rho_input=1.0, input_alloc=input_alloc)
    want = run("oracle_mag_noW")
    assert torch.equal(got, want), (
        "input_only at rho_input=1 must reproduce oracle_mag_noW exactly"
    )


def test_cost_agrees_with_oracle_mag_noW_at_rho_input_one():
    """(2*1 + r)/3 == (1 + 1 + r)/3 — the frames must coincide where the methods do."""
    for r in (0.0625, 0.125, 0.25, 0.5):
        a = report_input_only_accounting(1.0, r)
        assert a["used_param_fraction"] == pytest.approx(a["oracle_mag_noW_used"])
        # and the two-pass frame does NOT coincide: it bills the kept rows twice,
        # by exactly 2*r/3. This asymmetry is the reason input_only exists.
        assert two_pass_used(1.0, r, N_BRANCHES) == pytest.approx(
            a["used_param_fraction"] + 2.0 * r / 3.0)


def test_sparse_input_actually_changes_the_output():
    """Guard against the sparsification silently not being applied."""
    torch.manual_seed(0)
    H, I, E, K, T = 16, 12, 4, 2, 8
    x = torch.randn(1, T, H)

    def run(rho_input):
        torch.manual_seed(1)
        blk = TinyMoEBlock(H, I, E, K)
        _install_io(blk, B=K * I, I=I, rho_input=rho_input)   # B=K*I: no channel cut
        return blk(x)[0]

    # B = K*I means the keep-mask is all-True, so the ONLY difference between
    # these two runs is that gate/up read 25% of the coordinates.
    assert not torch.allclose(run(0.25), run(1.0), atol=1e-4)


def test_rho_channel_one_keeps_every_channel():
    """B = K*I is pure input sparsity: no channel is dropped."""
    torch.manual_seed(0)
    H, I, E, K, T = 16, 12, 4, 2, 6
    x = torch.randn(1, T, H)
    torch.manual_seed(1)
    blk = TinyMoEBlock(H, I, E, K)
    _install_io(blk, B=K * I, I=I, rho_input=1.0)
    # rho_input=1 AND no channel cut => the unmodified reference forward.
    torch.manual_seed(1)
    ref = TinyMoEBlock(H, I, E, K)
    assert torch.allclose(blk(x)[0], ref(x)[0], atol=1e-5)


# --------------------------------------------------------------------------
# accounting closed form
# --------------------------------------------------------------------------

def test_used_param_fraction_closed_form():
    for p, r in ((0.25, 0.25), (0.1875, 0.125), (0.5, 1.0), (0.0, 0.3)):
        assert used_param_fraction(p, r) == pytest.approx((2 * p + r) / 3.0)


def test_symmetric_point_lands_on_the_budget():
    """rho_input = rho_channel = C  =>  used = C exactly."""
    for c in (0.30, 0.25, 0.20):
        p = solve_symmetric(c)
        assert p == pytest.approx(c)
        assert used_param_fraction(p, p) == pytest.approx(c)


def test_rho_channel_is_the_cheap_axis():
    """The marginal costs are 2/3 for rho_input and 1/3 for rho_channel.

    This inverts sparse_probe's ordering (there rho_channel cost 1 and rho_input
    2/3), so "cut rho_input first" does not carry over. Pinned because a doc claim
    rests on it.
    """
    eps = 1e-6
    base = used_param_fraction(0.25, 0.25)
    d_in = (used_param_fraction(0.25 + eps, 0.25) - base) / eps
    d_ch = (used_param_fraction(0.25, 0.25 + eps) - base) / eps
    assert d_in == pytest.approx(2.0 / 3.0, abs=1e-4)
    assert d_ch == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert d_ch < d_in


def test_no_extra_storage():
    a = report_input_only_accounting(0.25, 0.25)
    assert a["extra_storage_frac_of_experts"] == 0.0


def test_budget_targets_hit_their_cuts():
    """The three benchmarked budgets, symmetric split."""
    for rho, cut in ((0.30, 0.70), (0.25, 0.75), (0.20, 0.80)):
        a = report_input_only_accounting(rho, rho)
        assert a["used_param_cut"] == pytest.approx(cut)


# --------------------------------------------------------------------------
# router allocation reaches the forward
# --------------------------------------------------------------------------

def test_router_alloc_changes_the_output_but_not_the_budget():
    """`router` spends the same pooled reads, differently — so output must move."""
    torch.manual_seed(0)
    H, I, E, K, T = 32, 12, 4, 2, 16
    x = torch.randn(1, T, H)

    def run(alloc):
        torch.manual_seed(1)
        blk = TinyMoEBlock(H, I, E, K)
        _install_io(blk, B=round(0.25 * K * I), I=I, rho_input=0.25,
                    input_alloc=alloc)
        return blk(x)[0]

    assert not torch.allclose(run("uniform"), run("router"), atol=1e-4)
    # cost is identical: the pooled read budget is K*round(rho_input*H) either way
    # (allocate_input_reads conserves it; pinned in test_sparse_probe.py).
    assert used_param_fraction(0.25, 0.25) == pytest.approx(0.25)


# --------------------------------------------------------------------------
# the shipped configs: prune_ratio and rho_channel are redundant and NOT
# cross-checked at load, so a mismatch silently changes B. Catch it here.
# --------------------------------------------------------------------------

def test_shipped_configs_are_self_consistent():
    import glob
    import os

    import yaml

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    paths = sorted(glob.glob(os.path.join(
        repo, "configs/eval/*inputonly*.yaml")))
    assert paths, "no input_only eval configs found"
    seen = set()
    for p in paths:
        cfg = yaml.safe_load(open(p))
        pk = cfg["prune_kwargs"]
        da = pk["dynamic_alloc"]
        io = da["input_only"]
        assert da["criterion"] == "input_only", p
        assert da["k_min"] == 0, f"{p}: the pooled top-B is global by design"
        # the mismatch the doc warns about
        assert pk["prune_ratio"] == pytest.approx(1.0 - io["rho_channel"]), p
        # symmetric split -> used == rho, exactly on the target cut in the filename
        cut = int(os.path.basename(p).split("cut")[1][:3]) / 10.0
        used = used_param_fraction(io["rho_input"], io["rho_channel"])
        assert used == pytest.approx(1.0 - cut / 100.0, abs=1e-9), (
            f"{p}: used={used} does not match the filename's {cut}% cut")
        assert io["input_alloc"] in ("uniform", "router"), p
        seen.add((cut, io["input_alloc"], cfg["eval_task_names"]))
    # 3 budgets x 2 allocations x 2 benchmarks
    assert len(seen) == 12, sorted(seen)


def test_forward_shape_and_finiteness_under_deep_sparsity():
    torch.manual_seed(0)
    H, I, E, K, T = 64, 24, 8, 4, 10
    x = torch.randn(1, T, H)
    for alloc in ("uniform", "router"):
        torch.manual_seed(1)
        blk = TinyMoEBlock(H, I, E, K)
        _install_io(blk, B=round(0.20 * K * I), I=I, rho_input=0.20,
                    input_alloc=alloc)
        out, logits = blk(x)
        assert out.shape == (1, T, H)
        assert torch.isfinite(out).all()
        assert logits.shape == (T, E)
