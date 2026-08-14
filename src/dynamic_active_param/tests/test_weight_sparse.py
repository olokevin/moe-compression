"""Tests for the unstructured (entry-level) sparse channel proxy.

Three anchors carry most of the weight:

* :func:`test_single_full_row_level_is_input_sparse` — a one-level staircase
  ``((p, 1.0),)`` reads whole columns, so it must reproduce `input_sparse` at
  ``rho_input=p`` *exactly*. That pins the two methods to one family and makes the
  col-vs-row budget sweep in the docs a within-family comparison.
* :func:`test_full_staircase_is_oracle_mag_noW` — at density 1 the proxy is the
  exact intermediate, so the criterion must reproduce ``oracle_mag_noW``.
* :func:`test_mean_fix_is_exact_at_the_mean` — with the mean-fix on and ``x = mu``,
  every sparse read is zero and the score comes entirely from the precomputed
  ``W mu``, which must equal the exact score. That is the claim the mean-fix makes.

Everything about *which* read set ranks channels best is an empirical question
measured by ``scripts/wsparse_screen.py``; these tests guarantee the plumbing and
the accounting are not silently different.
"""

import types

import pytest
import torch
import torch.nn.functional as F

from src.dynamic_active_param.allocate import _CROSS_EXPERT_CRITERIA
from src.dynamic_active_param.block import dynamic_moe_block_forward
from src.dynamic_active_param.sparse_probe import (
    build_layer_probe,
    descending_abs_ranks,
    probe_expert_scores,
    used_param_fraction,
)
from src.dynamic_active_param.weight_sparse import (
    block_scores,
    build_layer_wsparse,
    level_row_counts,
    level_thresholds,
    levels_density,
    parse_levels,
    report_wsparse_accounting,
    wsparse_expert_scores,
    wsparse_used_param_fraction,
)
from src.dynamic_active_param.tests.test_block import TinyMoEBlock


def _install_wsparse(block, B, I, levels="0.25x0.5", use_gate=True, mu=None,
                     input_alloc="uniform", criterion="weight_sparse"):
    block._dyn_B = B
    block._dyn_k_min = 0
    block._dyn_I = I
    block._dyn_criterion = criterion
    block._dyn_ranks = torch.arange(I).unsqueeze(0).repeat(block.num_experts, 1).long()
    block._dyn_contrib = torch.rand(block.num_experts)
    block._dyn_prefix = None
    block._dyn_gains = None
    block._dyn_beta = 1.0
    block._dyn_wsparse = build_layer_wsparse(
        block.experts, levels=levels, use_gate=use_gate, mu=mu,
        input_alloc=input_alloc,
    )
    block.forward = types.MethodType(dynamic_moe_block_forward, block)
    return block


# --------------------------------------------------------------------------
# registration / spec parsing
# --------------------------------------------------------------------------

def test_criterion_is_registered_cross_expert():
    assert "weight_sparse" in _CROSS_EXPERT_CRITERIA


def test_parse_levels_sorts_by_descending_row_fraction():
    assert parse_levels("0.25x0.2+0.0625x1.0") == ((0.0625, 1.0), (0.25, 0.2))


def test_parse_levels_accepts_pairs_and_rejects_overfull_columns():
    assert parse_levels([(0.5, 0.5)]) == ((0.5, 0.5),)
    with pytest.raises(ValueError):
        parse_levels("0.7x1.0+0.7x0.5")
    with pytest.raises(ValueError):
        parse_levels("1.5x1.0")


def test_density_is_the_sum_of_level_areas():
    assert levels_density("0.0625x1.0+0.25x0.2") == pytest.approx(0.1125)
    assert levels_density("1.0x0.1125") == pytest.approx(0.1125)
    assert levels_density("0.1125x1.0") == pytest.approx(0.1125)


# --------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------

def test_thresholds_keep_exactly_the_requested_rows_per_column():
    torch.manual_seed(0)
    I, H = 32, 8
    W = torch.randn(I, H)
    thr = level_thresholds(W, [1.0, 0.5, 0.25, 0.0])
    keep = W.abs() >= thr[1].unsqueeze(0)
    assert keep.sum(dim=0).tolist() == [16] * H
    keep = W.abs() >= thr[2].unsqueeze(0)
    assert keep.sum(dim=0).tolist() == [8] * H
    # the extremes are finite weight magnitudes (the threshold mode prices with
    # them), not +-inf: row_frac=1 is the column minimum, row_frac=0 sits above
    # the column maximum.
    assert (W.abs() >= thr[0].unsqueeze(0)).sum(dim=0).tolist() == [I] * H
    assert (W.abs() >= thr[3].unsqueeze(0)).sum(dim=0).tolist() == [0] * H
    assert torch.isfinite(thr).all()
    assert torch.allclose(thr[0], W.abs().amin(dim=0))


def test_thresholds_are_per_column_not_global():
    # column 1 is scaled up 100x; a global threshold would drop all of column 0.
    W = torch.tensor([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])
    thr = level_thresholds(W, [0.5])
    keep = W.abs() >= thr[0].unsqueeze(0)
    assert keep.sum(dim=0).tolist() == [2, 2]


def test_probe_aliases_served_weights_without_copying():
    torch.manual_seed(0)
    block = TinyMoEBlock(H=16, I=12, E=4, K=2)
    wsp = build_layer_wsparse(block.experts, levels="0.5x0.5")
    for e, expert in enumerate(block.experts):
        assert wsp.Wu[e].data_ptr() == expert.up_proj.weight.data_ptr()
        assert wsp.Wg[e].data_ptr() == expert.gate_proj.weight.data_ptr()


# --------------------------------------------------------------------------
# the family anchors
# --------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0.25, 0.5, 1.0])
def test_single_full_row_level_is_input_sparse(p):
    """``((p, 1.0),)`` reads whole columns == input_sparse at rho_input=p."""
    torch.manual_seed(1)
    H, I = 24, 10
    block = TinyMoEBlock(H=H, I=I, E=3, K=2)
    x = torch.randn(7, H)

    wsp = build_layer_wsparse(block.experts, levels=((p, 1.0),))
    probe = build_layer_probe(block.experts, bits=16, rho_input=p,
                             input_alloc="uniform")
    from src.dynamic_active_param.sparse_probe import sparsify_input_topk
    for eid in range(3):
        got = wsparse_expert_scores(x, wsp, eid)
        want = probe_expert_scores(sparsify_input_topk(x, p), probe, eid)
        assert torch.allclose(got, want, atol=1e-5), f"expert {eid}"


def test_full_staircase_is_oracle_mag_noW():
    torch.manual_seed(2)
    H, I, E, K = 16, 12, 4, 2
    x = torch.randn(2, 5, H)
    B = K * I // 2

    a = _install_wsparse(TinyMoEBlock(H, I, E, K), B, I, levels="1.0x1.0")
    torch.manual_seed(3)
    b = TinyMoEBlock(H, I, E, K)
    b.load_state_dict(a.state_dict())
    b = _install_wsparse(b, B, I, levels="1.0x1.0", criterion="oracle_mag_noW")

    ya, _ = a(x)
    yb, _ = b(x)
    assert torch.allclose(ya, yb, atol=1e-5)


def test_static_mask_level_drops_low_magnitude_entries_only():
    """``((1.0, a),)`` reads every coordinate but only the top-|W| rows of each."""
    torch.manual_seed(4)
    H, I = 20, 16
    block = TinyMoEBlock(H=H, I=I, E=2, K=1)
    x = torch.randn(5, H)
    wsp = build_layer_wsparse(block.experts, levels="1.0x0.5", use_gate=False)
    W = block.experts[0].up_proj.weight.detach()
    thr = wsp.thr_u[0, 0]
    want = (x @ (W * (W.abs() >= thr.unsqueeze(0))).t()).abs()
    assert torch.allclose(wsparse_expert_scores(x, wsp, 0), want, atol=1e-5)


def test_graded_staircase_reads_the_expected_entry_count():
    """Realized read count matches ``levels_density`` (the accounting's claim)."""
    torch.manual_seed(5)
    H, I = 64, 32
    block = TinyMoEBlock(H=H, I=I, E=1, K=1)
    levels = parse_levels("0.25x1.0+0.5x0.25")
    wsp = build_layer_wsparse(block.experts, levels=levels, use_gate=False)
    W = block.experts[0].up_proj.weight.detach()
    x = torch.randn(1, H)
    ranks, _ = descending_abs_ranks(x)
    n_read = 0
    lo = 0
    for l, (cf, rf) in enumerate(levels):
        hi = lo + int(round(cf * H))
        cols = ((ranks[0] >= lo) & (ranks[0] < hi))
        thr = wsp.thr_u[0, l]
        mask = (W.abs() >= thr.unsqueeze(0)) & cols.unsqueeze(0)
        n_read += int(mask.sum())
        lo = hi
    assert n_read == pytest.approx(levels_density(levels) * I * H, rel=0.02)


# --------------------------------------------------------------------------
# threshold mode (the evaluated method)
# --------------------------------------------------------------------------

_LADDER = (1.0, 0.7, 0.49, 0.343, 0.24, 0.168, 0.118, 0.082)


def _tau_probe(experts, density=0.1125, ladder=_LADDER, **kw):
    return build_layer_wsparse(
        experts, levels=tuple((0.0, rf) for rf in ladder),
        alloc_mode="tau", density=density, count_reads=True, **kw)


def test_tau_mode_respects_the_read_budget():
    """The cost claim: reads per token per branch never exceed the budget."""
    torch.manual_seed(10)
    H, I = 256, 128
    block = TinyMoEBlock(H=H, I=I, E=2, K=1)
    probe = _tau_probe(block.experts, density=0.1125)
    x = torch.randn(64, H)
    wsparse_expert_scores(x, probe, 0)
    realized = probe.reads_sum / probe.reads_n
    assert realized <= 0.1125 + 1e-9
    # and it is not leaving the budget mostly unspent
    assert realized > 0.9 * 0.1125


def test_tau_mode_budget_scales_with_density():
    torch.manual_seed(11)
    H, I = 256, 128
    block = TinyMoEBlock(H=H, I=I, E=2, K=1)
    x = torch.randn(32, H)
    got = []
    for d in (0.05, 0.1125, 0.25):
        p = _tau_probe(block.experts, density=d)
        wsparse_expert_scores(x, p, 0)
        got.append(p.reads_sum / p.reads_n)
    assert got[0] < got[1] < got[2]
    for d, g in zip((0.05, 0.1125, 0.25), got):
        assert g == pytest.approx(d, rel=0.15)


def test_tau_mode_spends_reads_where_the_product_is_largest():
    """A coordinate with a huge |x| must be read deeper than a tiny one."""
    torch.manual_seed(12)
    H, I = 64, 64
    block = TinyMoEBlock(H=H, I=I, E=1, K=1)
    probe = _tau_probe(block.experts, density=0.125, ladder=(1.0, 0.5, 0.25, 0.125))
    x = torch.zeros(1, H)
    x[0, 0] = 100.0          # one dominant coordinate
    x[0, 1:] = 0.01
    from src.dynamic_active_param.weight_sparse import _tau_bands
    n_rows = torch.tensor([round(rf * I) for _, rf in probe.levels],
                          dtype=torch.float32)
    lvl, _ = _tau_bands(x.abs(), probe.thr_u[0], n_rows,
                        torch.tensor([0.125 * I * H]), 16)
    assert int(lvl[0, 0]) == 0                       # dominant coord: deepest read
    assert int(lvl[0, 5]) > int(lvl[0, 0])           # a weak one: shallower or unread


def test_tau_mode_is_closer_to_the_product_rule_than_column_sparsity_is():
    """The ladder is a *discretization* of the exact ``|W_ji·x_i|`` top-N rule.

    It cannot match it entrywise (a column snaps to one of the available channel
    fractions, and the freed budget is then re-spent elsewhere), so the claim under
    test is the one the method actually rests on: at equal read count the
    discretized rule tracks the exact rule far more closely than reading whole
    columns does. Offline this shows up as rel_err 0.4147 (ladder) vs 0.4132
    (exact) vs 0.5171 (columns).
    """
    torch.manual_seed(13)
    H, I, d = 128, 64, 0.2
    block = TinyMoEBlock(H=H, I=I, E=1, K=1)
    W = block.experts[0].up_proj.weight.detach()
    x = torch.randn(8, H)

    exact = torch.empty((x.shape[0], I))
    for t in range(x.shape[0]):
        wx = W * x[t].unsqueeze(0)
        thr = wx.abs().flatten().topk(int(round(d * I * H))).values[-1]
        exact[t] = (wx * (wx.abs() >= thr)).sum(dim=1).abs()

    ladder = tuple(0.9 ** k for k in range(40))
    tau = wsparse_expert_scores(
        x, _tau_probe(block.experts, density=d, ladder=ladder, use_gate=False), 0)
    col = wsparse_expert_scores(
        x, build_layer_wsparse(block.experts, levels=((d, 1.0),), use_gate=False), 0)

    err_tau = (tau - exact).norm() / exact.norm()
    err_col = (col - exact).norm() / exact.norm()
    assert err_tau < 0.5 * err_col, f"tau {err_tau:.4f} vs col {err_col:.4f}"


def test_tau_mode_accounting_uses_the_target_density():
    probe_levels = tuple((0.0, rf) for rf in _LADDER)
    a = report_wsparse_accounting(probe_levels, rho_channel=0.125, density=0.1125)
    assert a["used_param_fraction"] == pytest.approx(0.20)
    # the ladder's own "area" is meaningless in this mode
    assert levels_density(probe_levels) == 0.0
    assert wsparse_used_param_fraction(probe_levels, 0.125) == pytest.approx(0.125)


def test_tau_mode_runs_through_the_block():
    torch.manual_seed(14)
    H, I, E, K = 64, 32, 4, 2
    x = torch.randn(2, 5, H)
    for alloc in ("uniform", "router"):
        block = TinyMoEBlock(H, I, E, K)
        block._dyn_B, block._dyn_k_min, block._dyn_I = K * I // 8, 0, I
        block._dyn_criterion = "weight_sparse"
        block._dyn_wsparse = _tau_probe(block.experts, density=0.1125,
                                        input_alloc=alloc)
        block.forward = types.MethodType(dynamic_moe_block_forward, block)
        y, _ = block(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        assert block._dyn_wsparse.reads_n > 0


# --------------------------------------------------------------------------
# mean-fix
# --------------------------------------------------------------------------

def test_mean_fix_is_exact_at_the_mean():
    torch.manual_seed(6)
    H, I = 20, 8
    block = TinyMoEBlock(H=H, I=I, E=2, K=1)
    mu = torch.randn(H)
    # a very thin read set: everything must come from the precomputed W mu
    wsp = build_layer_wsparse(block.experts, levels="0.05x0.05", mu=mu)
    x = mu.unsqueeze(0).repeat(3, 1)
    got = wsparse_expert_scores(x, wsp, 0)
    e = block.experts[0]
    want = (F.silu(e.gate_proj(mu)) * e.up_proj(mu)).abs().unsqueeze(0).repeat(3, 1)
    assert torch.allclose(got, want, atol=1e-4)


def test_mean_fix_off_is_starved_at_the_mean():
    """Control for the test above: without the mean-fix the thin read set misses."""
    torch.manual_seed(6)
    H, I = 20, 8
    block = TinyMoEBlock(H=H, I=I, E=2, K=1)
    mu = torch.randn(H)
    wsp = build_layer_wsparse(block.experts, levels="0.05x0.05", mu=None)
    x = mu.unsqueeze(0)
    e = block.experts[0]
    exact = (F.silu(e.gate_proj(mu)) * e.up_proj(mu)).abs()
    got = wsparse_expert_scores(x, wsp, 0)[0]
    assert (got - exact).abs().max() > 1e-3


# --------------------------------------------------------------------------
# router input allocation
# --------------------------------------------------------------------------

def test_router_alloc_scales_the_ladder_with_the_slot_budget():
    torch.manual_seed(7)
    H, I = 64, 16
    block = TinyMoEBlock(H=H, I=I, E=2, K=2)
    wsp = build_layer_wsparse(block.experts, levels="0.25x1.0", use_gate=False,
                              input_alloc="router")
    x = torch.randn(4, H)
    ranks, _ = descending_abs_ranks(x)
    # a slot granted 0 coordinates scores nothing; one granted all of them is exact.
    zero = wsparse_expert_scores(x, wsp, 0, ranks=ranks,
                                 n_cols=torch.zeros(4, dtype=torch.long))
    full = wsparse_expert_scores(x, wsp, 0, ranks=ranks,
                                 n_cols=torch.full((4,), H, dtype=torch.long))
    assert float(zero.abs().max()) == 0.0
    W = block.experts[0].up_proj.weight.detach()
    assert torch.allclose(full, (x @ W.t()).abs(), atol=1e-4)


def test_block_runs_for_both_input_allocs_and_conserves_the_budget():
    torch.manual_seed(8)
    H, I, E, K = 24, 16, 4, 2
    x = torch.randn(2, 6, H)
    for alloc in ("uniform", "router"):
        block = _install_wsparse(TinyMoEBlock(H, I, E, K), B=K * I // 4, I=I,
                                 levels="0.0625x1.0+0.25x0.25", input_alloc=alloc)
        y, _ = block(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


def test_budget_is_conserved():
    """Exactly B of the K*I pooled channels survive per token."""
    torch.manual_seed(9)
    H, I, E, K = 20, 12, 4, 2
    from src.dynamic_active_param.allocate import select_global_topB
    block = _install_wsparse(TinyMoEBlock(H, I, E, K), B=K * I // 4, I=I)
    x = torch.randn(1, 5, H).view(-1, H)
    rw = torch.rand(5, K)
    sel = torch.randint(0, E, (5, K))
    from src.dynamic_active_param.block import _cross_expert_keep
    _, keep = _cross_expert_keep(block, x, rw, sel)
    assert keep.reshape(5, -1).sum(dim=1).tolist() == [K * I // 4] * 5


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------

def test_used_param_fraction_matches_input_sparse_at_equal_density():
    for p in (0.1125, 0.25, 0.5):
        assert wsparse_used_param_fraction(((p, 1.0),), 0.125) == pytest.approx(
            used_param_fraction(p, 0.125))
        # a rectangle with the same area costs the same
        assert wsparse_used_param_fraction(((1.0, p),), 0.125) == pytest.approx(
            used_param_fraction(p, 0.125))


def test_used_param_fraction_closed_form():
    a = report_wsparse_accounting("0.0625x1.0+0.25x0.2", rho_channel=0.125)
    assert a["density_per_branch"] == pytest.approx(0.1125)
    assert a["used_param_fraction"] == pytest.approx(0.125 + 2 * 0.1125 / 3)
    assert a["used_param_cut"] == pytest.approx(0.8)          # the -80% target frame
    assert a["scoring"] == pytest.approx(0.075)
    assert a["compute"] == pytest.approx(0.125)


def test_up_only_halves_the_scoring_cost():
    both = report_wsparse_accounting("0.25x0.45", 0.125, use_gate=True)
    up = report_wsparse_accounting("0.25x0.45", 0.125, use_gate=False)
    assert up["scoring"] == pytest.approx(both["scoring"] / 2)


def test_level_map_metadata_grows_with_level_count():
    one = report_wsparse_accounting("1.0x0.1125", 0.125)
    three = report_wsparse_accounting("0.0625x1.0+0.125x0.2+0.375x0.0667", 0.125)
    assert one["level_map_bits_per_weight"] == 1
    assert three["level_map_bits_per_weight"] == 2
    assert three["storage_level_map_frac_of_ffn"] > one["storage_level_map_frac_of_ffn"]
    # thresholds themselves are nearly free; the level map is not
    assert one["storage_thresholds_frac_of_ffn"] < 0.01
    assert one["storage_level_map_frac_of_ffn"] == pytest.approx(2 / 16 / 3)


def test_mean_fix_storage_is_negligible():
    a = report_wsparse_accounting("0.25x0.45", 0.125, I=768, H=2048, meanfix=True)
    assert a["storage_mean_fix_frac_of_ffn"] < 0.001


# --------------------------------------------------------------------------
# semi-structured channel blocks
# --------------------------------------------------------------------------

def test_row_block_selects_whole_blocks_of_channels():
    torch.manual_seed(20)
    I, H, r = 64, 8, 8
    W = torch.randn(I, H)
    thr = level_thresholds(W, [0.5], row_block=r)
    keep = block_scores(W, r) >= thr[0].unsqueeze(0)
    assert keep.shape == (I // r, H)
    assert keep.sum(dim=0).tolist() == [(I // r) // 2] * H


def test_row_block_read_counts_are_block_multiples():
    counts = level_row_counts([1.0, 0.7, 0.49, 0.082], I=768, row_block=8)
    assert counts == [768, 536, 376, 64]
    assert all(c % 8 == 0 for c in counts)
    assert level_row_counts([0.45], I=768, row_block=1) == [346]


def test_row_block_still_respects_the_budget():
    torch.manual_seed(21)
    H, I = 256, 128
    block = TinyMoEBlock(H=H, I=I, E=2, K=1)
    p = _tau_probe(block.experts, density=0.1125, row_block=8)
    wsparse_expert_scores(torch.randn(32, H), p, 0)
    assert p.reads_sum / p.reads_n <= 0.1125 + 1e-9
