import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dynamic_active_param.block import dynamic_moe_block_forward


class TinyExpert(nn.Module):
    def __init__(self, H, I):
        super().__init__()
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TinyMoEBlock(nn.Module):
    """Mimics Qwen3MoeSparseMoeBlock closely enough for the forward path."""

    def __init__(self, H, I, E, K):
        super().__init__()
        self.num_experts = E
        self.top_k = K
        self.norm_topk_prob = True
        self.gate = nn.Linear(H, E, bias=False)
        self.experts = nn.ModuleList([TinyExpert(H, I) for _ in range(E)])

    def forward(self, hidden_states):
        # upstream-equivalent reference forward (full width, no budget masking)
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        final = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for eidx in hit:
            e = int(eidx)
            idx, top_x = torch.where(expert_mask[e].squeeze(0))
            cur = hidden_states[None, top_x].reshape(-1, hidden_dim)
            ch = self.experts[e](cur) * routing_weights[top_x, idx, None]
            final.index_add_(0, top_x, ch.to(hidden_states.dtype))
        return final.reshape(batch_size, sequence_length, hidden_dim), router_logits


def _install(block, B, k_min, I, criterion="router_prob", prefix_sums=None):
    E = block.num_experts
    # identity ranks: rank[e,c] = c  (channel c has rank c)
    block._dyn_ranks = torch.arange(I).unsqueeze(0).repeat(E, 1).long()
    block._dyn_contrib = torch.rand(E)
    block._dyn_prefix = prefix_sums
    block._dyn_B = B
    block._dyn_k_min = k_min
    block._dyn_I = I
    block._dyn_criterion = criterion
    block.forward = types.MethodType(dynamic_moe_block_forward, block)


def test_rho_one_equals_reference():
    torch.manual_seed(0)
    H, I, E, K = 16, 32, 8, 4
    block = TinyMoEBlock(H, I, E, K)
    x = torch.randn(2, 5, H)

    ref_out, _ = block.forward(x)  # reference (bound method on class)

    # rho=1.0 => B = K*I => keep all channels => must equal reference
    _install(block, B=K * I, k_min=4, I=I, criterion="router_prob")
    dyn_out, _ = block.forward(x)

    assert torch.allclose(ref_out, dyn_out, atol=1e-5), "rho=1.0 dynamic must match reference"


def test_nonzero_channels_equal_budget():
    torch.manual_seed(1)
    H, I, E, K = 8, 40, 6, 3
    block = TinyMoEBlock(H, I, E, K)
    x = torch.randn(1, 12, H)

    B = round(0.67 * K * I)
    k_min = 4

    # Capture the per-token, per-expert nonzero intermediate count by hooking.
    # We re-derive allocation and check keep-mask nonzeros sum to B per token.
    from src.dynamic_active_param.allocate import allocate_budgets

    block_ranks_E_I = torch.arange(I).unsqueeze(0).repeat(E, 1).long()
    _install(block, B=B, k_min=k_min, I=I, criterion="router_prob")

    # recompute routing exactly as the forward does
    hs = x.view(-1, H)
    rl = block.gate(hs)
    rw = F.softmax(rl, dim=1, dtype=torch.float)
    rw, sel = torch.topk(rw, K, dim=-1)
    rw /= rw.sum(dim=-1, keepdim=True)
    k = allocate_budgets(rw.to(hs.dtype), sel, block._dyn_contrib, B, k_min, I, "router_prob")

    # For each token, total kept channels across its K experts == B
    assert torch.all(k.sum(dim=1) == B)
    # and with identity ranks, kept channels per (token,expert) == k
    for t in range(k.shape[0]):
        for j in range(K):
            keep = (block_ranks_E_I[int(sel[t, j])] < k[t, j]).sum().item()
            assert keep == int(k[t, j])


def test_coverage_rho_one_equals_reference():
    torch.manual_seed(2)
    H, I, E, K = 16, 32, 8, 4
    block = TinyMoEBlock(H, I, E, K)
    x = torch.randn(2, 5, H)

    ref_out, _ = block.forward(x)

    # rho=1.0 => B=K*I => coverage_alloc keeps all channels => equals reference.
    prefix = (torch.rand(E, I) + 1e-3).sort(dim=-1, descending=True).values.cumsum(-1)
    _install(block, B=K * I, k_min=4, I=I, criterion="coverage_alloc", prefix_sums=prefix)
    dyn_out, _ = block.forward(x)

    assert torch.allclose(ref_out, dyn_out, atol=1e-5), "coverage rho=1.0 must match reference"
