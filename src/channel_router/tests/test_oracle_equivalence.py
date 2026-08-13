"""The load-bearing test: our oracle definition IS the deployed ``oracle_mag`` score.

Every recall/mass-recall number in the channel-router study is measured against
``src.channel_router.data.oracle_scores``, while every downstream accuracy already in
the repo was measured with ``src/dynamic_active_param/block.py``'s ``oracle_mag``
branch. If the two disagree, the study's targets are not the deployed target. So this
test builds a tiny Qwen3-MoE-shaped block, runs both paths, and requires bit-level
agreement on the resulting keep-mask (and near-exact agreement on the scores).

It also checks the ``channel_router`` criterion in ``mode='oracle'`` reproduces the same
mask, which is what makes the ΔPPL ladder's "oracle" row comparable to the published
``oracle_mag`` rows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.channel_router.data import LayerWeights, oracle_scores
from src.channel_router.metrics import select_topB
from src.channel_router.scorers import ForwardSelector
from src.dynamic_active_param.block import _cross_expert_keep


class _Expert(nn.Module):
    def __init__(self, H, I):
        super().__init__()
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)
        self.act_fn = F.silu


class _Block(nn.Module):
    """Minimal stand-in for Qwen3MoeSparseMoeBlock (only what the forward touches)."""

    def __init__(self, H, I, E, K):
        super().__init__()
        self.experts = nn.ModuleList([_Expert(H, I) for _ in range(E)])
        self.gate = nn.Linear(H, E, bias=False)
        self.num_experts, self.top_k, self.norm_topk_prob = E, K, True


def _fixture(seed=0, H=16, I=12, E=6, K=3, T=32):
    torch.manual_seed(seed)
    blk = _Block(H, I, E, K).to(torch.float64)
    h = torch.randn(T, H, dtype=torch.float64)
    probs = F.softmax(blk.gate(h), dim=1, dtype=torch.float64)
    g, sel = torch.topk(probs, K, dim=-1)
    g = g / g.sum(-1, keepdim=True)
    w = LayerWeights(
        Wg=torch.stack([e.gate_proj.weight.data for e in blk.experts]),
        Wu=torch.stack([e.up_proj.weight.data for e in blk.experts]),
        Wd=torch.stack([e.down_proj.weight.data for e in blk.experts]),
        gate_w=blk.gate.weight.data.float(),
        col_norm=torch.stack([e.down_proj.weight.data.norm(dim=0) for e in blk.experts]),
        top_k=K, norm_topk=True, layer=0)
    return blk, h, g, sel, w


def test_oracle_scores_match_block_oracle_mag():
    blk, h, g, sel, w = _fixture()
    I, K = w.I, w.top_k
    B = (K * I) // 4
    # deployed path
    blk._dyn_criterion = "oracle_mag"
    blk._dyn_I = I
    blk._dyn_B = B
    blk._dyn_col_norm = w.col_norm
    inter_all, keep_dep = _cross_expert_keep(blk, h, g, sel)
    # study path
    imp = oracle_scores(h, sel, g, w)
    keep_ours = select_topB(imp, B)
    assert torch.equal(keep_dep, keep_ours)
    dep_score = (g.unsqueeze(-1) * inter_all.abs() * w.col_norm[sel]).float()
    assert torch.allclose(dep_score, imp.float(), atol=1e-6, rtol=1e-5)


def test_oracle_mag_noW_variant_matches_use_colnorm_false():
    blk, h, g, sel, w = _fixture(seed=1)
    I, K = w.I, w.top_k
    B = (K * I) // 3
    blk._dyn_criterion = "oracle_mag_noW"
    blk._dyn_I, blk._dyn_B = I, B
    blk._dyn_col_norm = w.col_norm
    _, keep_dep = _cross_expert_keep(blk, h, g, sel)
    imp = oracle_scores(h, sel, g, w, use_colnorm=False)
    assert torch.equal(keep_dep, select_topB(imp, B))


def test_forward_selector_oracle_mode_matches():
    blk, h, g, sel, w = _fixture(seed=2)
    I, K = w.I, w.top_k
    B = (K * I) // 2
    blk._dyn_criterion = "oracle_mag"
    blk._dyn_I, blk._dyn_B = I, B
    blk._dyn_col_norm = w.col_norm
    inter_all, keep_dep = _cross_expert_keep(blk, h, g, sel)
    fs = ForwardSelector("oracle", w.col_norm, collect_stats=True)
    keep_fs = fs(h, g, sel, inter_all, B)
    assert torch.equal(keep_dep, keep_fs)
    s = fs.summary()
    assert abs(s["mass_recall"] - 1.0) < 1e-9 and abs(s["recall"] - 1.0) < 1e-9
    assert abs(s["kept_per_token"] - B) < 1e-9


def test_degrade_lowers_mass_recall_monotonically():
    blk, h, g, sel, w = _fixture(seed=3, T=64)
    I, K = w.I, w.top_k
    B = (K * I) // 4
    blk._dyn_criterion = "oracle_mag"
    blk._dyn_I, blk._dyn_B = I, B
    blk._dyn_col_norm = w.col_norm
    inter_all, _ = _cross_expert_keep(blk, h, g, sel)
    prev = 1.0
    for frac in (0.0, 0.1, 0.25, 0.5):
        fs = ForwardSelector("oracle_degrade", w.col_norm, drop_frac=frac)
        keep = fs(h, g, sel, inter_all, B)
        assert (keep.reshape(keep.shape[0], -1).sum(1) == B).all()
        mr = fs.summary()["mass_recall"]
        assert mr <= prev + 1e-9, f"mass recall not monotone at frac={frac}"
        prev = mr
    assert prev < 1.0
