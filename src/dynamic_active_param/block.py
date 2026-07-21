"""Drop-in dynamic MoE block forward (masking simulation).

``dynamic_moe_block_forward`` replaces ``Qwen3MoeSparseMoeBlock.forward`` /
``Qwen2MoeSparseMoeBlock.forward``. Routing / top-k is identical to upstream;
the only change is that, per token, a fixed channel budget ``B`` is split
across its K experts (``allocate_budgets``) and each expert keeps only its top
``k_{t,e}`` channels by precomputed rank — the rest of the SwiGLU intermediate
is zeroed before ``down_proj`` (fake pruning, so ``down_proj`` runs at full
width with original weights: no Nyström correction).

The block reads per-layer state attached at install time:
    self._dyn_ranks    (E, I) long   channel ranks by descending score
    self._dyn_contrib  (E,)   float   expert_out_token_contrib >= 0
    self._dyn_prefix   (E, I) float   descending-score prefix sums (coverage_alloc)
    self._dyn_B        int             total kept channels per token
    self._dyn_k_min    int             per-expert floor
    self._dyn_I        int             per-expert cap (moe_intermediate_size)
    self._dyn_criterion str            router_prob | contribution | uniform | coverage_alloc
"""

import torch
import torch.nn.functional as F

from src.dynamic_active_param.allocate import allocate_budgets

__all__ = ["dynamic_moe_block_forward"]


def dynamic_moe_block_forward(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    # --- dynamic per-token per-expert channel budgets -----------------------
    # (T, K) long: how many channels each token keeps in each of its K experts.
    k_alloc = allocate_budgets(
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        contrib=self._dyn_contrib,
        B=self._dyn_B,
        k_min=self._dyn_k_min,
        I=self._dyn_I,
        criterion=self._dyn_criterion,
        prefix_sums=getattr(self, "_dyn_prefix", None),
    )

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        eid = int(expert_idx)
        expert_layer = self.experts[eid]
        idx, top_x = torch.where(expert_mask[eid].squeeze(0))

        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

        # SwiGLU intermediate at full width, then zero the channels beyond each
        # token's budget for this expert (keep the top k_{t,e} ranks).
        gate = expert_layer.gate_proj(current_state)
        up = expert_layer.up_proj(current_state)
        inter = expert_layer.act_fn(gate) * up  # (n_e, I)

        k_col = k_alloc[top_x, idx]                       # (n_e,) budget per token
        rank_row = self._dyn_ranks[eid]                   # (I,)
        keep = rank_row.unsqueeze(0) < k_col.unsqueeze(1)  # (n_e, I) bool
        inter = inter * keep.to(inter.dtype)

        current_hidden_states = expert_layer.down_proj(inter)
        current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]

        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    # Shared expert path (Qwen2-MoE); left untouched — it is not budget-pruned.
    if hasattr(self, "shared_expert") and self.shared_expert is not None:
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits
