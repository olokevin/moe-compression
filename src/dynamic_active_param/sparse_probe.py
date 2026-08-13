"""Low-precision, input-sparse **probe** for Level-2 cross-expert channel selection.

`oracle_mag` reaches near-dense accuracy at a 7/8 channel cut, but it ranks
channels by the *true* SwiGLU intermediate, so `gate_proj` and `up_proj` must run
at full width just to decide: a nominal −87.5% channel cut realizes only a −29.2%
whole-FFN cut. This module produces the same ranking from a proxy cheap enough
that all three matrices can be gathered to budget:

    x_sp   = the top ``rho_input`` fraction of x by |x_i|      (per token)
    ũ_e    = W̃_up^(e) x_sp,   h̃_e = SiLU(W̃_gate^(e) x_sp)      (b-bit weights)
    score  = g_e · |ũ_e ⊙ h̃_e|      pooled over the token's K experts
    keep   = global top-B, then the TRUE up/gate/down on the kept channels only

**Why full-rank-but-imprecise rather than low-rank.** The top-B set is decided by
the fine structure of individual weight rows. Spectral truncation deletes exactly
that structure — this repo measured the whole low-rank family dead (`lowrank_scorer`,
recall 0.44 at ρ=0.25), and a screen over cross-expert shared bases, product
quantization (0.5–1.0 bits/weight), Hadamard rotation, 1-bit sign, asymmetric
gate/up precision and router-rank-adaptive precision found *every one* of them
dominated by plain group-wise RTN composed with per-token input sparsity. See
``scripts/probe_frontier.py`` and ``docs/exps/dynamic_active_param/sparse_probe.md``.

**Two sparsities.** The method has exactly two knobs, both *keep* fractions:

    rho_input    fraction of the token's input coordinates read for SCORING
    rho_channel  fraction of the pooled K*I channels kept for COMPUTE

**Cost model** (units of one full-width ``(I,H)`` matrix; a dense expert FFN is 3).
The headline metric is used parameters = scoring + compute
(:func:`used_param_fraction`)::

    used = rho_channel + n_matrices · rho_input / 3

At serving precision that is the whole story. The legacy quantized-proxy variant
(``bits < 16``) additionally charges its own bytes::

    bits/weight  = bits + 16/group          (payload + the fp16 group scale)
    probe/matrix = (bits/weight)/16 · rho_input

Input sparsity reduces bytes because only the *columns* of the proxy matching x's
kept coordinates are read; in batch-1 decode, which is the persistently
memory-bound regime an MoE lives in, that gather is per token. FLOPs scale the
same way. Both are reported by :func:`report_probe_accounting`.

**Prior art.** Prox (arXiv:2607.27591) published quantized-proxy + input-sparsity →
rank → exact-compute for SwiGLU FFNs on ten *dense* LLMs. The proxy mechanism here
is theirs; what is new is the MoE instantiation — the pooled cross-expert budget,
where the proxy must compare channels across K different experts through ``g_e``.

The eval path is a masking simulation: the arithmetic is identical to zeroing the
non-kept intermediate before `down_proj`, while the *accounting* changes, exactly
as for `oracle_up` and `lowrank_scorer`.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from src.base.shared_utils import _print

__all__ = [
    "SparseProbe",
    "quantize_rtn_dequant",
    "sparsify_input_topk",
    "sparsify_input_by_count",
    "descending_abs_ranks",
    "allocate_input_reads",
    "build_layer_probe",
    "probe_expert_scores",
    "probe_cost_per_matrix",
    "n_scored",
    "used_param_fraction",
    "report_probe_accounting",
    "print_probe_accounting",
]

# input_alloc -> the g_e exponent in allocate_input_reads. "colnorm" is handled
# separately (it reweights |x_i| rather than the per-expert budget).
_ALLOC_BETA = {"uniform": 0.0, "router": 1.0, "router2": 2.0, "colnorm": 0.0}


@dataclass
class SparseProbe:
    """Per-layer expert-stacked proxies of ``up_proj``/``gate_proj``.

    ``Wu_q`` ``(E, I, H)`` and ``Wg_q`` ``(E, I, H)`` hold the *dequantized* b-bit
    values, i.e. the numerics the real kernel would produce after unpacking. The
    byte accounting is analytic (:func:`probe_cost_per_matrix`) and deliberately
    independent of this storage choice — packing them would change memory use, not
    the arithmetic being simulated.

    ``bits >= 16`` is the **reuse** regime: the probe *is* the served weight, so
    ``Wu_q``/``Wg_q`` are views onto the experts' own tensors (no copy — a stacked
    fp16 copy of up+gate for Qwen3-30B is ~39 GB) and the only error the probe
    makes is input sparsity. This is the deployable form, because there is no
    extra storage to object to; see :func:`report_probe_accounting`.

    ``Wg_q is None`` selects the up-only probe.

    ``input_alloc`` chooses which input coordinates each token reads (see
    :func:`sparsify_input_topk`): ``"mag"`` ranks by ``|x_i|`` within a fixed
    per-token budget, ``"colnorm"`` by ``|x_i|·rms_j(W[:,i])`` using the offline
    column statistic in ``col_rms``.
    """

    Wu_q: torch.Tensor
    Wg_q: Optional[torch.Tensor]
    bits: int
    group: int
    rho_input: float
    input_alloc: str = "mag"
    col_rms: Optional[torch.Tensor] = None


def quantize_rtn_dequant(W: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    """Group-wise symmetric round-to-nearest quantize+dequantize along the input dim.

    Groups of ``group`` consecutive input coordinates share one fp16 scale, which
    is the standard recipe and what makes 2–3 bit weights usable at all. Returns a
    tensor of ``W``'s dtype holding the reconstructed values.

    The scale is rounded to fp16 because that is what the cost model is charged for
    (``16/group`` bits per weight). ``scripts/idea_pilot_scorers.quantize_rtn``, used
    by the offline screens, keeps the scale in fp32, so the two differ by ~1%
    relative at 3 bits — verified not to move the screen's verdict at all
    (recall 0.6752 / mass 0.8637 either way, L46, 2048 tokens, bits=3, keep=0.25).
    """
    if bits >= 16:
        return W.clone()
    *lead, H = W.shape
    g = group if group and H % group == 0 else H
    Wg_ = W.reshape(*lead, H // g, g).float()
    qmax = 2 ** (bits - 1) - 1
    if qmax < 1:
        raise ValueError(f"bits={bits} too small for symmetric RTN (need >= 2)")
    scale = Wg_.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    # store the scale at fp16, which is what the cost model is charged for
    scale = scale.to(torch.float16).float()
    q = torch.clamp(torch.round(Wg_ / scale), -qmax - 1, qmax)
    return (q * scale).reshape(*lead, H).to(W.dtype)


def sparsify_input_topk(x: torch.Tensor, keep: float) -> torch.Tensor:
    """Zero all but the ``keep`` largest-|·| coordinates of each token's ``x``.

    Per-token (not per-batch): the coordinate set is what the proxy pass gathers,
    and in batch-1 decode there is no batch to amortize a shared set over.

    This is the ``input_alloc="uniform"`` term: every one of the token's K experts
    reads the *same* ``keep·H`` coordinates. See :func:`allocate_input_reads` for
    the cross-expert alternatives.
    """
    if keep is None or keep >= 1.0:
        return x
    k = max(1, int(round(float(keep) * x.shape[-1])))
    idx = x.abs().topk(k, dim=-1).indices
    out = torch.zeros_like(x)
    return out.scatter_(-1, idx, x.gather(-1, idx))


# Bisection iterations for the pooled input-read allocation. 40 halvings resolve
# tau to ~1e-12 of its range, which leaves a residual of at most a few units.
_ALLOC_BISECT_ITERS = 40
# Cap on unit top-up passes after bisection (one read each). Ties at the threshold
# are the only source of shortfall, so this is generous.
_ALLOC_TOPUP_MAX = 64


def descending_abs_ranks(x: torch.Tensor):
    """Per-token rank of each coordinate in the descending ``|x|`` order.

    Returns ``(ranks, sorted_abs)``: ``ranks (T,H) long`` with ``ranks[t,i]=0``
    for the largest-|·| coordinate, and ``sorted_abs (T,H)`` the descending
    magnitudes. Computed once per layer and shared by all K experts of a token —
    the sort is over the token's own hidden state, not per expert.
    """
    sorted_abs, order = torch.sort(x.abs(), dim=-1, descending=True)
    ranks = torch.empty_like(order)
    ar = torch.arange(x.shape[-1], device=x.device).expand_as(order)
    ranks.scatter_(-1, order, ar)
    return ranks, sorted_abs


def allocate_input_reads(
    sorted_abs: torch.Tensor,
    routing_weights: torch.Tensor,
    keep: float,
    beta: float,
) -> torch.Tensor:
    """Split a token's pooled input-read budget across its K experts.

    The probe's job is to rank ``K·I`` channels on one scale, and a channel's
    score is ``g_e·|SiLU(g̃ate)⊙ũp|`` — so a coordinate read spent on a
    *low*-``g_e`` expert moves the pooled ranking less than the same read spent on
    a high-``g_e`` one. That suggests not giving every routed expert the same
    number of coordinates. Three terms, all at the identical pooled budget
    ``K·round(keep·H)`` reads per token:

    - ``beta=0`` (**uniform**): ``n_e = keep·H`` for every expert. The coordinate
      set is a property of the token alone, so all K experts share it.
    - ``beta=1`` (**router**): rank the ``(slot, coord)`` pairs by
      ``g_e·|x_i|`` and take the pooled top-``K·keep·H``. High-``g_e`` experts
      score on more coordinates, dominated ones on fewer.
    - ``beta=2`` (**router²**): rank by ``g_e²·|x_i|``. Motivated by the repo's
      Level-1 result that the marginal value of resolution for an expert scales
      like ``g_e²`` (``pivchol_global`` scores channels by ``g_e²·σ``): an
      expert's channels both *carry* a ``g_e`` factor and *are more numerous*
      near the global top-B threshold in proportion to ``g_e``.

    Because the score is separable (``c_e = g_e^β`` times the shared ``|x_i|``),
    the pooled top-N is a single threshold ``τ``: slot ``e`` keeps the
    coordinates with ``|x_i| > τ/c_e``, so ``n_e`` is a *prefix length* of the
    token's shared descending-|x| order and no per-expert sort is needed. We
    bisect ``τ`` for the largest value meeting the budget, then hand the small
    residual to the slots with the largest next-coordinate score.

    Args:
        sorted_abs: ``(T, H)`` descending ``|x|`` from :func:`descending_abs_ranks`.
        routing_weights: ``(T, K)`` the token's K routing weights.
        keep: per-expert-equivalent read fraction; sets the pooled budget.
        beta: 0 (uniform) | 1 (router) | 2 (router²), or any exponent.

    Returns:
        ``(T, K)`` long ``n_e`` with ``n_e.sum(dim=1) == K·round(keep·H)`` and
        ``0 <= n_e <= H``. A strongly dominated slot can receive 0 reads (its
        proxy scores are then all zero, so it wins no channel) — the same
        behaviour as the Level-1 ``pivchol`` allocator, and why there is no floor.
    """
    T, K = routing_weights.shape
    H = sorted_abs.shape[-1]
    per = max(1, int(round(float(keep) * H)))
    if beta == 0 or keep is None or keep >= 1.0:
        return torch.full((T, K), min(per, H), dtype=torch.long,
                          device=sorted_abs.device)

    budget = K * per
    g = routing_weights.to(torch.float32).clamp_min(0.0)
    c = g.pow(float(beta)).clamp_min(torch.finfo(torch.float32).tiny)   # (T,K)
    s = sorted_abs.to(torch.float32)                                    # (T,H) desc

    # n_e(tau) = #{ i : c_e * s_i > tau }. s is descending, so that count is a
    # prefix length: searchsorted on the ascending flip.
    s_asc = torch.flip(s, dims=[-1]).contiguous()

    def counts(tau):                          # tau (T,1) -> (T,K) long
        thr = (tau / c).contiguous()          # (T,K) per-slot |x| threshold
        # #{ s_i > thr } = H - #{ s_asc <= thr }
        n = H - torch.searchsorted(s_asc, thr, right=True)
        return n.clamp_(0, H)

    hi = (c * s[:, :1]).amax(dim=1, keepdim=True) * 1.000001   # all counts 0
    lo = torch.zeros((T, 1), dtype=torch.float32, device=s.device)
    n_best = torch.full((T, K), H, dtype=torch.long, device=s.device)
    for _ in range(_ALLOC_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        n = counts(mid)
        feasible = n.sum(dim=1, keepdim=True) <= budget
        # feasible -> tau can come down (more reads); else push tau up.
        hi = torch.where(feasible, mid, hi)
        lo = torch.where(feasible, lo, mid)
        n_best = torch.where(feasible, n, n_best)

    # Exact top-up. Bisection lands short only by the number of (slot, coord)
    # pairs tied at the threshold, so a handful of passes suffices; each gives one
    # remaining read to the slot whose next coordinate scores highest,
    # argmax_e c_e·s[n_e] among slots below the cap. Bounded by _ALLOC_TOPUP_MAX
    # rather than unbounded so a pathological all-ties token cannot spin.
    residual = budget - n_best.sum(dim=1)                      # (T,) >= 0
    for _ in range(_ALLOC_TOPUP_MAX):
        active = residual > 0
        if not bool(active.any()):
            break
        nxt = c * torch.gather(s, 1, n_best.clamp(max=H - 1))  # (T,K)
        nxt = nxt.masked_fill(n_best >= H, float("-inf"))
        pick = nxt.argmax(dim=1, keepdim=True)                 # (T,1)
        one = active.long().unsqueeze(1)                       # (T,1) 1 where needed
        n_best = n_best.scatter_add(1, pick, one)
        residual = residual - one.squeeze(1)

    # A slot may legitimately receive 0 reads: its proxy scores are then all zero
    # so it wins no channel, which is the same "dominated expert gets nothing"
    # behaviour the Level-1 pivchol allocator has. Budget stays exact.
    return n_best.clamp_(0, H)


def build_layer_probe(
    experts,
    bits: int,
    group: int = 128,
    use_gate: bool = True,
    rho_input: float = 0.25,
    compute_device=None,
    input_alloc: str = "uniform",
) -> SparseProbe:
    """Build one MoE layer's probe over ``up_proj`` (and optionally ``gate_proj``).

    Args:
        experts: the layer's expert module list.
        bits: proxy weight bit-width. ``>=16`` is the **reuse** regime — the probe
            aliases the served weights (zero extra storage, and with
            ``rho_input=1.0`` it reduces to ``oracle_mag_noW``, the unit-test
            anchor). Below 16 it is a separate RTN copy.
        group: RTN group size along the input dim (ignored when ``bits>=16``).
        use_gate: ``False`` builds the up-only probe (targets ``oracle_up``'s signal).
        rho_input: fraction of ``x``'s coordinates the probe reads.
        compute_device: where quantization runs; ``None`` keeps the weights' device.
        input_alloc: how the pooled input-read budget is split across a token's K
            experts — ``uniform`` | ``router`` | ``router2`` | ``colnorm``. See
            :func:`allocate_input_reads`.
    """
    def _stack(which):
        # bits >= 16 is the reuse regime: the probe *is* the served weight, so
        # alias the experts' own tensors instead of materializing a stacked copy
        # (that copy is ~39 GB for Qwen3-30B's up+gate and would OOM the box).
        # probe_expert_scores indexes per expert, so a python list of views is a
        # valid stand-in for the (E, I, H) tensor and costs no memory.
        if int(bits) >= 16:
            return [getattr(e, which).weight.detach() for e in experts]
        W = torch.stack([getattr(e, which).weight.detach() for e in experts], dim=0)
        home = W.device
        if compute_device is not None and torch.device(compute_device) != home:
            W = W.to(compute_device)
        Q = quantize_rtn_dequant(W, bits, group)
        del W
        return Q.to(home).contiguous()

    col_rms = None
    if input_alloc == "colnorm":
        # rms_j(W[:, i]) over rows and experts — the offline column statistic that
        # converts |x_i| into "how much this coordinate can move any score".
        acc, n = None, 0
        for e in experts:
            for which in (("up_proj", "gate_proj") if use_gate else ("up_proj",)):
                W = getattr(e, which).weight.detach().float()
                sq = (W * W).mean(dim=0)                    # (H,)
                acc = sq if acc is None else acc + sq
                n += 1
        col_rms = (acc / max(n, 1)).sqrt()

    return SparseProbe(
        Wu_q=_stack("up_proj"),
        Wg_q=_stack("gate_proj") if use_gate else None,
        bits=int(bits), group=int(group), rho_input=float(rho_input),
        input_alloc=str(input_alloc), col_rms=col_rms,
    )


def probe_expert_scores(x_sp: torch.Tensor, probe: SparseProbe, eid: int) -> torch.Tensor:
    """Proxy of one expert's channel magnitudes for the tokens routed to it.

    Args:
        x_sp: ``(T, H)`` **already sparsified** hidden states (sparsify once per
            token, not once per expert — under ``input_alloc="uniform"`` the
            coordinate set is a property of the token, and re-doing it per expert
            would misstate the cost).
        probe: this layer's :class:`SparseProbe`.
        eid: expert index.

    Returns:
        ``(T, I)`` float32 proxy of ``|SiLU(gate_j·x)·(up_j·x)|``, or ``|up_j·x|``
        when the probe is up-only.
    """
    Wu = probe.Wu_q[eid]
    up_hat = (x_sp @ Wu.t().to(x_sp.dtype)).float()
    if probe.Wg_q is None:
        return up_hat.abs()
    Wg = probe.Wg_q[eid]
    gate_hat = (x_sp @ Wg.t().to(x_sp.dtype)).float()
    return (F.silu(gate_hat) * up_hat).abs()


def sparsify_input_by_count(
    x: torch.Tensor, ranks: torch.Tensor, n_keep: torch.Tensor
) -> torch.Tensor:
    """Keep each token's top-``n_keep[t]`` coordinates in the shared ``|x|`` order.

    The per-slot read counts from :func:`allocate_input_reads` are prefix lengths
    of one shared descending-|x| order (that is what makes the pooled allocation a
    single threshold), so applying them is a compare against the cached ranks —
    no re-sort per expert.

    Args:
        x: ``(T, H)`` hidden states.
        ranks: ``(T, H)`` long, from :func:`descending_abs_ranks`.
        n_keep: ``(T,)`` long, coordinates this expert reads for each token.
    """
    return x * (ranks < n_keep.unsqueeze(-1)).to(x.dtype)


def probe_cost_per_matrix(bits: int, group: int, rho_input: float) -> float:
    """Probe bytes for **one** matrix as a fraction of one fp16 ``(I,H)`` matrix."""
    if int(bits) >= 16:
        return float(rho_input)          # reuse: no group scale to pay for
    bpw = float(bits) + (16.0 / float(group) if group else 0.0)
    return (bpw / 16.0) * float(rho_input)


def used_param_fraction(
    rho_input: float, rho_channel: float, n_matrices: int = 2
) -> float:
    """Whole-FFN used-parameter fraction: **scoring reads + compute reads**.

    The method has **two sparsities**, and both are *keep* fractions in ``[0, 1]``:

    * ``rho_input``   — fraction of the token's input coordinates read for scoring.
    * ``rho_channel`` — fraction of the ``K·I`` pooled channels kept for compute.

    (Note both arguments are "keep", not "prune". An earlier signature took the
    channel axis as ``prune_ratio`` = ``1 − rho_channel``, which forced every call
    site to write ``1.0 - r`` and invited sign errors.)

    The quantity charged is every parameter the token touches, split into the two
    passes that touch them (units of one expert ``(I, H)`` matrix; a dense expert
    FFN is 3):

    * **scoring** — ``n_matrices`` branches (``up`` + ``gate``) read all ``I`` rows
      but only ``rho_input`` of the input columns: ``n · rho_input``.
    * **compute** — all three matrices are gathered to the ``rho_channel`` selected
      channels (``up``/``gate`` rows, ``down`` columns): ``3 · rho_channel``.

    ::

        used = (n·rho_input + 3·rho_channel) / 3 = rho_channel + n·rho_input/3

    This is deliberately **conservative**: the scoring pass and the compute pass
    overlap (the probe reads part of the same rows the exact pass then reads in
    full), and this frame bills that overlap twice rather than discounting it. The
    discounted variant ``rho_channel + n·rho_input·(1−rho_channel)/3`` is defensible
    for memory traffic under a perfect cache, but it flatters the method and depends
    on an assumption about the kernel, so it is not used. Two consequences worth
    stating plainly:

    * At ``rho_input→1`` this gives ``rho_channel + n/3``, which *exceeds* a
      single-pass exact scorer (``oracle_mag``'s ``(1+1+rho_channel)/3``) by
      ``2·rho_channel/3`` — correctly, since a two-pass scheme that reads everything
      really does read the kept rows twice.
    * It matches :func:`lowrank_scorer.report_scorer_accounting`'s
      ``kept = rho + n·c/3``, so every scorer row in the docs is now on one scale.

    ``rho_input→0`` still gives bare ``rho_channel``.
    """
    ri, rc, n = float(rho_input), float(rho_channel), int(n_matrices)
    return rc + n * ri / 3.0


def report_probe_accounting(
    bits: int, group: int, rho_input: float, use_gate: bool, rho_channel: float,
    lam: float = 1.0,
) -> dict:
    """Whole-FFN active-parameter accounting for the sparse-probe scheme.

    Non-cascade (``lam=1``) the keep-decision precedes every full-width matmul, so
    all three matrices are gathered to ``ρ`` and the probe is the only overhead::

        kept = ρ + n_matrices · probe_per_matrix / 3

    With a cascade (``lam>1``) the probe nominates ``λB`` candidates, the *exact*
    up/gate run on those, and ``down`` runs on the final ``B``::

        kept = (probe_total + 2·λ·ρ + ρ) / 3

    Compare `oracle_mag` ``(1+1+ρ)/3`` and `oracle_up` ``(1+2ρ)/3``.

    ``used_param_fraction`` is the headline number: **scoring params + compute
    params**, ``ρ + n·p/3``. At serving precision (``bits >= 16``, the only regime
    now reported in the docs) ``probe_cost_per_matrix`` is just ``p``, so
    ``kept_fraction`` and ``used_param_fraction`` coincide.
    """
    n = 2 if use_gate else 1
    c1 = probe_cost_per_matrix(bits, group, rho_input)
    probe_total = n * c1
    rho = float(rho_channel)
    lam = float(lam)
    if lam <= 1.0:
        kept = rho + probe_total / 3.0
        kept_flops = rho + (n * float(rho_input)) / 3.0
    else:
        kept = (probe_total + 2.0 * lam * rho + rho) / 3.0
        kept_flops = (n * float(rho_input) + 2.0 * lam * rho + rho) / 3.0
    used = used_param_fraction(rho_input, rho_channel, n)
    return {
        "used_param_fraction": used,
        "used_param_cut": 1.0 - used,
        "bits_per_weight": float(bits) + (16.0 / group if group else 0.0),
        "probe_bytes_per_matrix": c1,
        "probe_bytes_total": probe_total,
        "probe_overhead_ffn": probe_total / 3.0,
        "rho": rho,
        "lam": lam,
        "kept_fraction": kept,
        "kept_fraction_flops": kept_flops,
        "whole_ffn_cut": 1.0 - kept,
        "oracle_mag_kept": (1.0 + 1.0 + rho) / 3.0,
        "oracle_up_kept": (1.0 + 2.0 * rho) / 3.0,
        # storage, not traffic: a sub-serving-precision proxy is an *extra* copy of
        # up/gate. Reported because it is the standing objection to the approach —
        # and it is exactly 0 in the reuse regime (bits >= 16).
        "extra_storage_frac_of_experts": (
            0.0 if int(bits) >= 16
            else n * (float(bits) + (16.0 / group if group else 0.0)) / 16.0 / 3.0
        ),
    }


def n_scored(use_gate: bool) -> int:
    """Number of scored branches: 2 (up+gate) or 1 (up-only)."""
    return 2 if use_gate else 1


def print_probe_accounting(bits, group, rho_input, use_gate, rho_channel, lam=1.0,
                           input_alloc="uniform"):
    a = report_probe_accounting(bits, group, rho_input, use_gate, rho_channel, lam)
    _print(
        f"[input_sparse] rho_input={float(rho_input):g} rho_channel={float(rho_channel):g} "
        f"(scoring {n_scored(use_gate) * float(rho_input) / 3.0:.4f} + compute "
        f"{float(rho_channel):.4f}) -> USED PARAMS="
        f"{a['used_param_fraction']:.4f}, cut {100 * a['used_param_cut']:.1f}%; "
        f"bits={bits} group={group} lam={lam} input_alloc={input_alloc}; "
        f"legacy proxy frame: probe={a['probe_bytes_total']:.4f} of one matrix, "
        f"kept={a['kept_fraction']:.4f} (cut {100 * a['whole_ffn_cut']:.1f}%), "
        f"FLOPs kept {a['kept_fraction_flops']:.4f}; "
        f"oracle_mag would keep {a['oracle_mag_kept']:.4f}, "
        f"oracle_up {a['oracle_up_kept']:.4f}; extra proxy storage = "
        f"{100 * a['extra_storage_frac_of_experts']:.1f}% of expert weights"
    )
    return a
