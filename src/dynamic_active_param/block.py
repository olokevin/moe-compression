"""Drop-in dynamic MoE block forward (masking simulation).

``dynamic_moe_block_forward`` replaces ``Qwen3MoeSparseMoeBlock.forward`` /
``Qwen2MoeSparseMoeBlock.forward``. Routing / top-k is identical to upstream;
the only change is that, per token, a fixed channel budget ``B`` is split
across its K experts and each expert keeps only its top ``k_{t,e}`` channels
(by precomputed rank) — the rest of the SwiGLU intermediate is zeroed before
``down_proj`` (fake pruning, so ``down_proj`` runs at full width with original
weights: no Nyström correction).

Two families of criteria:

- **router-only** (``router_prob`` | ``contribution`` | ``uniform`` |
  ``coverage_alloc`` | ``pivchol_global``): the per-expert keep-count is decided
  by ``allocate_budgets`` from the router weights alone, then applied in the
  standard per-expert loop.
- **cross-expert** (Level-2: ``oracle_mag`` | ``oracle_mag_noW`` | ``oracle_up``
  | ``pubsub`` | ``lowrank_scorer``): channels of all K active experts compete on
  **one global scale** per token, so we materialize each token's ``(K, I)``
  intermediate, score it, and keep the global top-``B``. ``oracle_mag_noW`` drops
  the ``||W_down||`` factor from ``oracle_mag`` (Q1); ``oracle_up`` ranks by the
  ``up_proj`` output instead of the SwiGLU intermediate, so the top-B decision
  precedes ``gate_proj`` and both ``gate_proj`` + ``down_proj`` are cut to budget
  (Q2). ``lowrank_scorer`` goes further: it ranks by a **cheap block-low-rank
  proxy** of the intermediate, so the decision precedes *every* full-width matmul
  and all three expert matrices are gathered to budget.

The block reads per-layer state attached at install time:
    self._dyn_ranks    (E, I) long   channel ranks by descending score
    self._dyn_contrib  (E,)   float   expert_out_token_contrib >= 0
    self._dyn_prefix   (E, I) float   descending-score prefix sums (coverage_alloc)
    self._dyn_gains    (E, I) float   pivoted-Cholesky marginal gains (pivchol_global)
    self._dyn_beta     float          g^{2*beta} sharpness (pivchol_global; M4)
    self._dyn_col_norm (E, I) float   ||W_down[:,j]|| per channel (oracle_mag)
    self._dyn_pub_*    pubsub artifact tensors (pubsub)
    self._dyn_sc_up    LowRankScorer for up_proj    (lowrank_scorer)
    self._dyn_sc_gate  LowRankScorer for gate_proj  (lowrank_scorer; None => up-only)
    self._dyn_probe    SparseProbe                  (sparse_probe / input_sparse)
    self._dyn_wsparse  WeightSparseProbe            (weight_sparse; unstructured)
    self._dyn_B        int             total kept channels per token
    self._dyn_k_min    int             per-expert floor
    self._dyn_I        int             per-expert cap (moe_intermediate_size)
    self._dyn_criterion str            criterion name
"""

import torch
import torch.nn.functional as F

from src.dynamic_active_param.allocate import (
    allocate_budgets,
    select_global_topB,
    _CROSS_EXPERT_CRITERIA,
)
from src.dynamic_active_param.lowrank_scorer import scorer_proxy
from src.dynamic_active_param.sparse_probe import (
    _ALLOC_BETA,
    allocate_input_reads,
    descending_abs_ranks,
    probe_expert_scores,
    sparsify_input_by_count,
    sparsify_input_topk,
)
from src.dynamic_active_param.weight_sparse import (
    wsparse_expert_scores,
    wsparse_layer_bands,
)

__all__ = ["dynamic_moe_block_forward"]


def _cross_expert_keep(self, hidden_states, routing_weights, selected_experts):
    """Level-2 cross-expert keep-mask: gather each token's (K,I) intermediate,
    score all K*I channels on one scale, keep the global top-B.

    Returns ``inter_all`` ``(T, K, I)`` and ``keep`` ``(T, K, I)`` bool, plus the
    per-(token, slot) intermediate so the caller can apply the mask and run
    ``down_proj`` in the standard per-expert scatter loop.
    """
    T, K = selected_experts.shape
    I = self._dyn_I
    device = hidden_states.device
    dtype = hidden_states.dtype

    # oracle_up (Q2) ranks channels by the up_proj output magnitude alone (the
    # signal computable before gate_proj), so it needs each token's (K, I) up
    # activation in addition to the SwiGLU intermediate that feeds down_proj.
    need_up = self._dyn_criterion == "oracle_up"
    # lowrank_scorer ranks by a cheap block-low-rank proxy of the intermediate,
    # computed from the hidden state alone — so it needs a (K, I) proxy tensor.
    need_proxy = self._dyn_criterion == "lowrank_scorer"
    # sparse_probe ranks by a low-precision / input-sparse proxy, also a (K, I)
    # tensor computed from the hidden state alone.
    need_probe = self._dyn_criterion == "sparse_probe"
    # weight_sparse ranks by an *unstructured* (entry-level) sparse proxy: the read
    # budget is spent on (channel, coordinate) pairs, not whole coordinates. Same
    # (K, I) proxy tensor, and the same masking simulation.
    need_wsp = self._dyn_criterion == "weight_sparse"
    probe_ranks = probe_nkeep = hidden_sp = None
    wsp_lvl_u = wsp_lvl_g = None
    if need_wsp:
        wsp = self._dyn_wsparse
        # Coordinate order is a property of the token (shared by its K experts and
        # both branches), so sort once per token — on the *centered* input when the
        # mean-fix is on, since that is what the sparse reads then estimate.
        d = (hidden_states if wsp.mu is None
             else hidden_states - wsp.mu.to(device=device, dtype=dtype))
        probe_ranks, sorted_abs = descending_abs_ranks(d)
        probe_nkeep = (
            allocate_input_reads(sorted_abs, routing_weights, wsp.alloc_keep,
                                 _ALLOC_BETA.get(wsp.input_alloc, 1.0))
            if wsp.input_alloc != "uniform" else None
        )
        if wsp.alloc_mode == "tau":
            # One batched bisection for the whole layer: the per-token threshold
            # depends on the token and its expert choice, not on the expert loop,
            # and resolving it per expert instead is ~50x slower (launch-bound).
            wsp_lvl_u, wsp_lvl_g = wsparse_layer_bands(
                sorted_abs, wsp, selected_experts, n_cols=probe_nkeep)
    if need_probe:
        probe = self._dyn_probe
        alloc = getattr(probe, "input_alloc", "uniform")
        if alloc == "uniform":
            # Sparsify once per token: the kept coordinate set is a property of the
            # token, shared by all K of its experts (and by both branches), which is
            # what the byte accounting in sparse_probe.py charges for.
            hidden_sp = sparsify_input_topk(hidden_states, probe.rho_input)
        elif alloc == "colnorm":
            # Same per-expert budget, but rank coordinates by |x_i|*rms_i(W): the
            # currency that actually perturbs a score is |x_i| * |w_{j,i}|.
            cr = probe.col_rms.to(device=hidden_states.device, dtype=torch.float32)
            k = max(1, int(round(float(probe.rho_input) * hidden_states.shape[-1])))
            idx = (hidden_states.abs().float() * cr).topk(k, dim=-1).indices
            hidden_sp = torch.zeros_like(hidden_states).scatter_(
                -1, idx, hidden_states.gather(-1, idx))
        else:
            # router / router2: the pooled read budget is split across the token's
            # K experts by g_e^beta, so each slot gets its own prefix length of the
            # shared descending-|x| order. Sort once per token, not once per expert.
            probe_ranks, sorted_abs = descending_abs_ranks(hidden_states)
            probe_nkeep = allocate_input_reads(
                sorted_abs, routing_weights, probe.rho_input,
                _ALLOC_BETA.get(alloc, 1.0),
            )                                                   # (T, K)

    # Materialize each token's K active-expert intermediates into (T, K, I).
    inter_all = torch.zeros((T, K, I), dtype=dtype, device=device)
    up_all = torch.zeros((T, K, I), dtype=dtype, device=device) if need_up else None
    proxy_all = (torch.zeros((T, K, I), dtype=torch.float32, device=device)
                 if (need_proxy or need_probe or need_wsp) else None)
    expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        eid = int(expert_idx)
        expert_layer = self.experts[eid]
        idx, top_x = torch.where(expert_mask[eid].squeeze(0))  # idx in 0..K-1, token id
        cur = hidden_states[top_x]
        gate = expert_layer.gate_proj(cur)
        up = expert_layer.up_proj(cur)
        inter_all[top_x, idx] = (expert_layer.act_fn(gate) * up).to(dtype)
        if need_up:
            up_all[top_x, idx] = up.to(dtype)
        if need_proxy:
            # Cheap proxy of this expert's intermediate from the rank-r cores.
            # Computed from `cur` only — in a realized implementation this runs
            # *before* the full gate/up above, and the true matmuls are then
            # gathered to the kept channels. The masking simulation keeps the
            # arithmetic identical while the accounting changes (see docstring).
            sc_up = self._dyn_sc_up
            up_hat = scorer_proxy(cur, sc_up.L_core[eid], sc_up.R_core[eid]).float()
            sc_gate = self._dyn_sc_gate
            if sc_gate is None:
                p = up_hat.abs()                                   # up-only proxy
            else:
                gate_hat = scorer_proxy(
                    cur, sc_gate.L_core[eid], sc_gate.R_core[eid]
                ).float()
                p = (F.silu(gate_hat) * up_hat).abs()              # up+gate proxy
            proxy_all[top_x, idx] = p
        if need_probe:
            # Proxy on the sparsified input. As with lowrank_scorer, in a realized
            # implementation this runs *before* the full gate/up above and the true
            # matmuls are gathered to the kept channels; the masking simulation
            # keeps the arithmetic identical and changes only the accounting (see
            # sparse_probe.report_probe_accounting).
            if hidden_sp is not None:
                cur_sp = hidden_sp[top_x]                       # shared coord set
            else:
                # per-slot read count: this expert reads n_keep[t, idx] coords of
                # the token's shared |x| order.
                cur_sp = sparsify_input_by_count(
                    hidden_states[top_x], probe_ranks[top_x],
                    probe_nkeep[top_x, idx],
                )
            proxy_all[top_x, idx] = probe_expert_scores(cur_sp, probe, eid)
        if need_wsp:
            # Unstructured proxy: per level, the coordinates in that band are read
            # for only the channels whose |W| is above the level's per-column
            # threshold. As with the other proxies this runs *before* the full
            # gate/up above in a realized implementation, so all three matrices are
            # gathered to the kept channels; the masking simulation keeps the
            # arithmetic identical and changes only the accounting.
            proxy_all[top_x, idx] = wsparse_expert_scores(
                hidden_states[top_x], wsp, eid, ranks=probe_ranks[top_x],
                n_cols=None if probe_nkeep is None else probe_nkeep[top_x, idx],
                lvl_u=None if wsp_lvl_u is None else wsp_lvl_u[top_x, idx],
                lvl_g=None if wsp_lvl_g is None else wsp_lvl_g[top_x, idx],
            )

    if self._dyn_criterion == "channel_router":
        # The learned channel router (and, through the same object, the exact-oracle
        # and controlled-degradation references of the calibration curve) decides the
        # keep-mask itself: it may restrict to tiles or force-keep a hot set, so it
        # returns a mask rather than a score. See src/channel_router/scorers.py.
        keep = self._dyn_router(
            hidden_states, routing_weights, selected_experts, inter_all, self._dyn_B
        )
        return inter_all, keep

    # Score all K*I channels on one per-token scale.
    if self._dyn_criterion in ("oracle_mag", "oracle_mag_noW"):
        # exact per-token magnitude of the down_proj input:
        #   oracle_mag     : g_e * |inter| * ||W_down[:,j]||  (full formula)
        #   oracle_mag_noW : g_e * |inter|                    (Q1: drop col-norm)
        g = routing_weights.to(torch.float32)                  # (T, K)
        score = g.unsqueeze(-1) * inter_all.abs().float()
        if self._dyn_criterion == "oracle_mag":
            score = score * self._dyn_col_norm[selected_experts]  # (T, K, I)
        keep = select_global_topB(score, self._dyn_B)
        return inter_all, keep

    if self._dyn_criterion == "lowrank_scorer":
        # Rank by the cheap proxy g_e * |ĥ_{e,j}(x)|. The proxy is a function of
        # the hidden state and the rank-r cores only, so the top-B decision is
        # available before up_proj/gate_proj/down_proj run at full width — all
        # three can be gathered to the kept channels. Output is identical to
        # zeroing the non-kept intermediate before down_proj; what changes versus
        # oracle_up is the realized cost (see lowrank_scorer.report_scorer_accounting).
        g = routing_weights.to(torch.float32)                  # (T, K)
        score = g.unsqueeze(-1) * proxy_all
        keep = select_global_topB(score, self._dyn_B)
        return inter_all, keep

    if self._dyn_criterion == "weight_sparse":
        # Rank by g_e * |SiLU(gate_hat) * up_hat| from an unstructured read set —
        # (channel, coordinate) entries chosen by the |W_ji|*|x_i| product rather
        # than whole coordinates. Same decision point as sparse_probe: before any
        # full-width matmul, so all three matrices are gathered to the kept
        # channels (see weight_sparse.report_wsparse_accounting).
        g = routing_weights.to(torch.float32)                  # (T, K)
        return inter_all, select_global_topB(g.unsqueeze(-1) * proxy_all, self._dyn_B)

    if self._dyn_criterion == "sparse_probe":
        # Rank by g_e * |SiLU(gate_hat) * up_hat| from b-bit weights read on the
        # token's top-|x| coordinates only. The decision is a function of the
        # hidden state and the proxy alone, so it precedes up/gate/down at full
        # width and all three are gathered to the kept channels.
        g = routing_weights.to(torch.float32)                  # (T, K)
        score = g.unsqueeze(-1) * proxy_all
        lam = float(getattr(self, "_dyn_probe_lam", 1.0))
        if lam <= 1.0:
            return inter_all, select_global_topB(score, self._dyn_B)
        # Cascade: the probe only *nominates* lam*B candidates; the exact up/gate
        # are then computed on those and the true top-B taken among them. Final
        # recall therefore equals candidate coverage, which is why a loose probe
        # suffices — at the price of 2(lam-1)*rho extra exact reads.
        C = min(K * I, max(self._dyn_B, int(round(lam * self._dyn_B))))
        cand = select_global_topB(score, C)
        exact = (g.unsqueeze(-1) * inter_all.abs().float()).masked_fill(
            ~cand, float("-inf"))
        return inter_all, select_global_topB(exact, self._dyn_B)

    if self._dyn_criterion == "oracle_up":
        # Q2: rank by the up_proj output magnitude (decided before gate_proj), so
        # keeping the global top-B lets gate_proj AND down_proj be computed only on
        # the kept channels. In the masking sim the output is identical to zeroing
        # the non-kept intermediate before down_proj, but the realized active-param
        # cut now covers gate_proj too (2x the reduction of oracle_mag at same B).
        g = routing_weights.to(torch.float32)                  # (T, K)
        score = g.unsqueeze(-1) * up_all.abs().float() * self._dyn_col_norm[selected_experts]
        keep = select_global_topB(score, self._dyn_B)
        return inter_all, keep

    # pubsub: private score g^2 * sigma_priv(channel), public carriers forced in.
    g = routing_weights.to(torch.float32)                      # (T, K)
    pivrank = self._dyn_pub_pivrank[selected_experts]          # (T, K, I) channel->rank
    gains = self._dyn_pub_gains[selected_experts]              # (T, K, I) rank-ordered
    sigma_chan = torch.gather(gains, 2, pivrank)               # (T, K, I) per physical channel
    score = (g * g).unsqueeze(-1) * sigma_chan                 # (T, K, I) private

    # Public: for each direction, force-keep the single best carrier among the
    # token's K experts (dedup across experts). carrier_idx/val: (r, E).
    cidx = self._dyn_pub_carrier_idx[:, selected_experts]      # (r, T, K) channel idx
    cval = self._dyn_pub_carrier_val[:, selected_experts]      # (r, T, K) |coef|
    r = cidx.shape[0]
    if r > 0:
        # best-carrying expert-slot per (dir, token).
        best_slot = cval.argmax(dim=2)                         # (r, T) in 0..K-1
        rr = torch.arange(r, device=device).unsqueeze(1)       # (r,1)
        tt = torch.arange(T, device=device).unsqueeze(0)       # (1,T)
        sel_chan = cidx[rr, tt, best_slot]                     # (r, T) channel idx
        # scatter +inf into score[token, best_slot, sel_chan] to force-keep it.
        flat_slot = best_slot.reshape(-1)                      # (r*T,)
        flat_tok = tt.expand(r, T).reshape(-1)                 # (r*T,)
        flat_chan = sel_chan.reshape(-1)                       # (r*T,)
        score[flat_tok, flat_slot, flat_chan] = float("inf")

    keep = select_global_topB(score, self._dyn_B)
    return inter_all, keep


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

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    cross_expert = self._dyn_criterion in _CROSS_EXPERT_CRITERIA
    if cross_expert:
        # Level-2: global cross-expert selection needs each token's full (K,I)
        # intermediate. Compute keep-mask once, then scatter down_proj per expert.
        inter_all, keep = _cross_expert_keep(
            self, hidden_states, routing_weights, selected_experts
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            eid = int(expert_idx)
            expert_layer = self.experts[eid]
            idx, top_x = torch.where(expert_mask[eid].squeeze(0))
            inter = inter_all[top_x, idx]                       # (n_e, I)
            inter = inter * keep[top_x, idx].to(inter.dtype)    # apply keep-mask
            current_hidden_states = expert_layer.down_proj(inter)
            current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    else:
        # router-only per-expert budget.
        k_alloc = allocate_budgets(
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            contrib=self._dyn_contrib,
            B=self._dyn_B,
            k_min=self._dyn_k_min,
            I=self._dyn_I,
            criterion=self._dyn_criterion,
            prefix_sums=getattr(self, "_dyn_prefix", None),
            gains=getattr(self, "_dyn_gains", None),
            beta=getattr(self, "_dyn_beta", 1.0),
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
