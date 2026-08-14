"""**Unstructured** (entry-level) sparse proxy for Level-2 cross-expert channel selection.

`input_sparse` (:mod:`src.dynamic_active_param.sparse_probe`) scores channels by
reading *whole columns* of the served ``up``/``gate`` — the token's top-``rho_input``
coordinates by ``|x_i|``. That is **column-structured** sparsity: every one of the
``I`` rows pays for every coordinate that is read. This module lifts that
restriction: a read budget is spent on individual ``(channel, coordinate)`` entries,
so a coordinate can be read for the channels whose weight on it is large and skipped
for the rest.

**Why that should help.** The exact contribution of entry ``(j, i)`` to channel
``j``'s pre-activation is ``W_ji · x_i``, so the greedy-optimal set of entries to
read at a fixed count is the top of ``|W_ji|·|x_i|`` — a *product* criterion over
both axes. Column selection can only see the ``|x_i|`` factor; a static weight mask
can only see the ``|W_ji|`` factor. Neither is the argmax of the product.

**The staircase.** The product rule is realized here as an ``L``-level staircase.
Per token, coordinates are ranked by ``|x_i|`` (descending) and split into ``L``
bands; band ``l`` covers a fraction ``col_frac[l]`` of the coordinates and reads,
for each of them, only the ``row_frac[l]`` fraction of channels whose ``|W_ji|`` is
largest **in that column**::

    level l:   col_frac[l] of coordinates  x  row_frac[l] of channels
    density  = sum_l col_frac[l] * row_frac[l]        (per scored branch)

Two familiar schemes are the extreme points of this one family, which is what makes
the comparison clean:

    ((p, 1.0),)      -> `input_sparse` at rho_input = p    (all rows, few columns)
    ((1.0, a),)      -> a static per-column-balanced weight mask (all columns, few rows)

and a graded staircase (``((0.06, 1.0), (0.25, 0.2))``, say) sits between them,
spending full-width reads on the few coordinates that carry the token's energy and
a thin slice of channels on the many that do not.

**Row order needs no index.** Within column ``i`` the kept channels are those above
a per-column magnitude threshold ``theta_i[l]``, computed offline from the served
weights (:func:`level_thresholds`). Storage is ``L·H`` floats per expert per branch
(``L/I`` of one matrix, ~0.4% at ``L=3``) — the *mask itself* is derived from the
weights, not stored. A kernel that wants to skip the non-kept entries without
reading them still needs positional metadata (a ``ceil(log2(L+1))``-bit level map,
~2 bits/entry); :func:`report_wsparse_accounting` reports that separately, because
it is the one cost `input_sparse` does not pay.

**Mean-fix.** ``x``'s mean over calibration data carries 12–19% of its energy, and
the doc's activation-aware screen found that the top eigenvector of the
score-error metric *is* the mean token — the part a per-token ranking cannot use,
and the part low-rank scorers waste their budget re-deriving. Here it can be had
for free and *exactly*: precompute ``b = W mu`` (``I`` floats per expert per
branch, ``1/H`` of a matrix), score the *centered* input ``delta = x - mu``, and add
``b`` back. The sparse reads are then spent entirely on the per-token deviation.

Cost model (units of one expert ``(I, H)`` matrix; a dense expert FFN is 3),
identical in form to :func:`sparse_probe.used_param_fraction` so every scorer row in
the docs stays on one scale::

    used = rho_channel + n_matrices * density / 3

The eval path is a masking simulation, exactly as for `input_sparse`: the
arithmetic is identical to zeroing the non-kept intermediate before ``down_proj``,
and only the accounting changes.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.base.shared_utils import _print
from src.dynamic_active_param.sparse_probe import descending_abs_ranks

__all__ = [
    "WeightSparseProbe",
    "parse_levels",
    "levels_density",
    "level_thresholds",
    "level_row_counts",
    "block_scores",
    "build_layer_wsparse",
    "wsparse_layer_bands",
    "wsparse_expert_scores",
    "wsparse_used_param_fraction",
    "report_wsparse_accounting",
    "print_wsparse_accounting",
]


# ---------------------------------------------------------------------------
# staircase specification
# ---------------------------------------------------------------------------

def parse_levels(spec) -> Tuple[Tuple[float, float], ...]:
    """Normalize a staircase spec into ``((col_frac, row_frac), ...)``.

    Accepts the string form ``"0.0625x1.0+0.25x0.2"`` (used by configs and the
    offline screen) or any sequence of pairs. Levels are returned sorted by
    *descending* ``row_frac``, which is the order the per-token coordinate bands
    are assigned in (the strongest coordinates get the widest channel slice), and
    is what makes the level thresholds nested.

    ``col_frac`` are *disjoint* band widths, so they may sum to less than 1 (the
    remaining coordinates are not read at all) but not to more.
    """
    if isinstance(spec, str):
        levels = []
        for part in spec.split("+"):
            part = part.strip()
            if not part:
                continue
            cf, rf = part.lower().split("x")
            levels.append((float(cf), float(rf)))
    else:
        levels = [(float(a), float(b)) for a, b in spec]
    if not levels:
        raise ValueError("empty staircase spec")
    for cf, rf in levels:
        if not (0.0 <= cf <= 1.0) or not (0.0 <= rf <= 1.0):
            raise ValueError(f"level fractions must be in [0,1], got ({cf}, {rf})")
    tot = sum(cf for cf, _ in levels)
    if tot > 1.0 + 1e-9:
        raise ValueError(f"column fractions sum to {tot:.4f} > 1")
    return tuple(sorted(levels, key=lambda t: -t[1]))


def levels_density(levels) -> float:
    """Fraction of one ``(I, H)`` matrix read per token, per scored branch."""
    return float(sum(cf * rf for cf, rf in parse_levels(levels)))


def block_scores(W: torch.Tensor, row_block: int) -> torch.Tensor:
    """Per-``(channel-block, coordinate)`` selection score: ``max |W|`` in the block.

    ``row_block=r`` makes the read set **semi**-structured — channels are selected in
    groups of ``r`` consecutive ones — which is what makes the positional metadata
    cheap: one level index per *block* instead of per weight, i.e. ``4/r`` bits per
    weight instead of 4. The max (rather than the mean) is the right pooling because a
    block must be read if *any* of its entries matters.
    """
    I, H = W.shape
    if row_block <= 1:
        return W.detach().abs().float()
    if I % row_block:
        raise ValueError(f"row_block={row_block} does not divide I={I}")
    return W.detach().abs().float().reshape(I // row_block, row_block, H).amax(dim=1)


def level_thresholds(W: torch.Tensor, row_fracs: Sequence[float],
                     row_block: int = 1) -> torch.Tensor:
    """Per-column ``|W|`` thresholds realizing each level's channel fraction.

    Returns ``(L, H)``: entries of column ``i`` with ``|W[:, i]| >= out[l, i]`` are
    exactly the ``round(row_fracs[l] * I)`` largest of that column. With
    ``row_block=r`` the thresholds instead apply to :func:`block_scores`, so a level
    keeps the top ``round(row_frac * I/r)`` *blocks* of ``r`` channels.

    The values are **finite actual weight magnitudes** (``row_frac=1`` gives the
    column minimum, ``row_frac=0`` a hair above the maximum) rather than ``±inf``,
    because the threshold mode (:func:`_tau_bands`) uses them as *prices*: reading
    column ``i`` down to level ``l`` is worth it exactly when
    ``|x_i| · out[l, i] >= tau``. ``±inf`` would break that comparison.

    One descending sort per (expert, branch) serves all levels.
    """
    H = W.shape[1]
    A = block_scores(W, row_block)                            # (I/r, H)
    nb = A.shape[0]
    S, _ = torch.sort(A, dim=0, descending=True)              # descending
    out = torch.empty((len(row_fracs), H), dtype=torch.float32, device=W.device)
    for l, a in enumerate(row_fracs):
        n = int(round(float(a) * nb))
        if n >= nb:
            out[l] = S[nb - 1]                                # keep every block
        elif n <= 0:
            out[l] = S[0] * 1.000001 + 1e-30                  # keep none
        else:
            out[l] = S[n - 1]
    return out


def level_row_counts(row_fracs: Sequence[float], I: int, row_block: int = 1):
    """Channels read at each level — a multiple of ``row_block`` by construction."""
    nb = I // max(row_block, 1)
    return [int(round(float(a) * nb)) * max(row_block, 1) for a in row_fracs]


def _tau_bands(
    ax: torch.Tensor, thr: torch.Tensor, n_rows: torch.Tensor,
    budget: torch.Tensor, iters: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Budget-exact per-token level assignment under the ``|W_ji·x_i|`` product rule.

    The greedy-optimal read set at a fixed count is ``{(j,i) : |W_ji|·|x_i| >= tau}``.
    Snapping it to the ``L`` available channel fractions, column ``i`` is read down to

        level(i) = min { l : theta_i[l] · |x_i| >= tau }        (unread if none)

    — ``theta_i[l]`` ascends with ``l`` (fewer rows = higher bar), so this keeps at
    most the product rule's count in every column. ``tau`` is bisected per token
    until the total read count meets that token's budget, which makes the *cost*
    exact and lets the *shape* float: a token whose ``|x|`` is concentrated reads
    deep into a few coordinates, a flat token reads shallowly into many. (This is
    the axis the repo's threshold study found worth floating — the input criterion,
    not the channel budget.)

    Args:
        ax: ``(T, H)`` ``|x|`` (or ``|x - mu|``) of the scored input.
        thr: ``(L, H)`` per-column thresholds, ascending in ``l``.
        n_rows: ``(L,)`` channels read at each level.
        budget: ``(T,)`` entries this token may read for this branch.
        iters: bisection steps; 16 resolves ``tau`` to 1e-5 of its range.

    Returns:
        ``(lvl, reads)`` — ``lvl (T, H)`` long in ``[0, L]`` (``L`` = unread) and
        ``reads (T,)`` the realized read count, always ``<= budget``.
    """
    price = ax.unsqueeze(1) * thr.unsqueeze(0).to(ax.dtype)    # (T, L, H) asc in l
    return _tau_bands_from_price(price, n_rows, budget, iters)


def _tau_bands_from_price(
    price: torch.Tensor, n_rows: torch.Tensor, budget: torch.Tensor, iters: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """:func:`_tau_bands` on a precomputed ``(N, L, H)`` price tensor.

    Split out so a whole layer's ``(token, slot)`` pairs can be bisected in one
    batched call (:func:`wsparse_layer_levels`) instead of once per expert: the
    bisection is a chain of small elementwise kernels, so doing it 256 times per
    block is launch-bound and ~70x slower than doing it once.
    """
    L = price.shape[1]
    ax_shape = (price.shape[0], price.shape[2])
    nr = n_rows.to(price.dtype)
    lo = torch.zeros_like(budget).unsqueeze(-1)
    hi = price.reshape(price.shape[0], -1).amax(dim=1, keepdim=True) * 1.000001
    best = torch.full(ax_shape, L, dtype=torch.long, device=price.device)
    best_reads = torch.zeros_like(budget)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)                                  # (T,1)
        ok = price >= mid.unsqueeze(-1)                        # (T,L,H)
        any_ok = ok.any(dim=1)
        lvl = ok.to(torch.uint8).argmax(dim=1).long()          # first l with ok
        reads = torch.where(any_ok, nr[lvl], torch.zeros_like(nr[0])).sum(dim=-1)
        feasible = reads <= budget                             # (T,)
        feas = feasible.unsqueeze(-1)                           # (T,1) for lo/hi
        # feasible -> tau can come down (read more); else push tau up.
        hi = torch.where(feas, mid, hi)
        lo = torch.where(feas, lo, mid)
        lvl = torch.where(any_ok, lvl, torch.full_like(lvl, L))
        best = torch.where(feasible.unsqueeze(-1), lvl, best)
        best_reads = torch.where(feasible, reads, best_reads)
    return best, best_reads


# ---------------------------------------------------------------------------
# per-layer artifact
# ---------------------------------------------------------------------------

@dataclass
class WeightSparseProbe:
    """One MoE layer's unstructured-sparse proxy of ``up_proj``/``gate_proj``.

    ``Wu``/``Wg`` are **views** onto the experts' own served weights (``data_ptr``
    equality, pinned by a unit test) — a stacked copy of up+gate is ~39 GB on
    Qwen3-30B, and the whole point of scoring from the served weights is that the
    proxy costs no storage. ``Wg is None`` selects the up-only proxy.

    ``thr_u``/``thr_g`` are ``(E, L, H)`` per-column magnitude thresholds from
    :func:`level_thresholds`; ``bias_u``/``bias_g`` are ``(E, I)`` precomputed
    ``W mu`` for the mean-fix, and ``mu`` is the ``(H,)`` calibration input mean
    (all three ``None`` when the mean-fix is off).
    """

    Wu: List[torch.Tensor]
    Wg: Optional[List[torch.Tensor]]
    levels: Tuple[Tuple[float, float], ...]
    thr_u: torch.Tensor
    thr_g: Optional[torch.Tensor]
    bias_u: Optional[torch.Tensor] = None
    bias_g: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    input_alloc: str = "uniform"
    # "rank": band widths are fixed (col_frac of the token's |x| order).
    # "tau" : band widths float per token under one budget-exact |W·x| threshold,
    #         so only the row_frac ladder and target_density are used.
    alloc_mode: str = "rank"
    row_block: int = 1
    target_density: Optional[float] = None
    tau_iters: int = 16
    # in-run verification of the read budget (the accounting's whole claim);
    # accumulated by :func:`wsparse_expert_scores` when ``count_reads`` is set.
    count_reads: bool = False
    reads_sum: float = 0.0
    reads_n: int = 0

    @property
    def density(self) -> float:
        """Per-branch read density this probe is charged for."""
        if self.alloc_mode in ("tau", "taux"):
            return float(self.target_density)
        return levels_density(self.levels)

    @property
    def alloc_keep(self) -> float:
        """The ``keep`` handed to :func:`sparse_probe.allocate_input_reads`.

        In rank mode the pooled quantity split across a token's K experts is the
        *coordinate* count, so it is the total column fraction. In tau mode the
        columns are emergent and the pooled quantity is the *read* budget, so any
        common scale works and the density is used — either way the per-slot ratio
        ``n_e / (keep·H)`` is what scales the slot's budget, and it averages to 1.
        """
        if self.alloc_mode in ("tau", "taux"):
            return float(self.target_density)
        return float(sum(cf for cf, _ in self.levels))

    # kept as an alias so existing call sites read naturally
    @property
    def col_total(self) -> float:
        return self.alloc_keep


def build_layer_wsparse(
    experts,
    levels,
    use_gate: bool = True,
    mu: Optional[torch.Tensor] = None,
    input_alloc: str = "uniform",
    compute_device=None,
    alloc_mode: str = "rank",
    row_block: int = 1,
    density: Optional[float] = None,
    tau_iters: int = 16,
    count_reads: bool = False,
) -> WeightSparseProbe:
    """Build one MoE layer's unstructured-sparse proxy.

    Args:
        experts: the layer's expert module list.
        levels: staircase spec (see :func:`parse_levels`). In ``alloc_mode="tau"``
            only the ``row_frac`` ladder matters — the column widths float — so a
            bare list of row fractions may be given as ``((0.0, rf), ...)``.
        row_block: select channels in groups of this many consecutive ones
            (semi-structured). ``1`` is fully unstructured; ``8`` costs 8x less
            positional metadata. See :func:`block_scores`.
        alloc_mode: ``rank`` (fixed band widths) or ``tau`` (band widths float per
            token under one budget-exact ``|W_ji·x_i|`` threshold; see
            :func:`_tau_bands`). ``tau`` requires ``density``.
        density: per-branch read budget for ``alloc_mode="tau"``.
        tau_iters: bisection steps for the per-token threshold.
        count_reads: accumulate realized read counts for in-run budget verification.
        use_gate: ``False`` scores from ``up_proj`` alone.
        mu: ``(H,)`` calibration mean of this layer's MoE input. When given, the
            proxy scores the centered input and adds the exact ``W mu`` back
            (the mean-fix); when ``None`` it scores ``x`` directly.
        input_alloc: ``uniform`` (every routed expert reads the same number of
            coordinates) or ``router`` (the pooled coordinate budget is split
            across the token's K experts by ``g_e·|x_i|``, reusing
            :func:`sparse_probe.allocate_input_reads`).
        compute_device: where the offline sorts run; ``None`` keeps the weights'
            device. Thresholds always end up on the weights' device.
    """
    levels = parse_levels(levels)
    row_fracs = [rf for _, rf in levels]
    if str(alloc_mode) in ("tau", "taux") and density is None:
        raise ValueError(f"alloc_mode={alloc_mode!r} needs an explicit density")

    def _views(which):
        return [getattr(e, which).weight.detach() for e in experts]

    def _thr(which):
        out = []
        for e in experts:
            W = getattr(e, which).weight.detach()
            home = W.device
            Wc = W.to(compute_device) if compute_device is not None else W
            out.append(level_thresholds(Wc, row_fracs, row_block).to(home))
        return torch.stack(out, dim=0)                        # (E, L, H)

    def _bias(which):
        out = []
        for e in experts:
            W = getattr(e, which).weight.detach()
            m = mu.to(device=W.device, dtype=torch.float32)
            out.append(W.float() @ m)                         # (I,)
        return torch.stack(out, dim=0)                        # (E, I)

    return WeightSparseProbe(
        Wu=_views("up_proj"),
        Wg=_views("gate_proj") if use_gate else None,
        levels=levels,
        thr_u=_thr("up_proj"),
        thr_g=_thr("gate_proj") if use_gate else None,
        bias_u=_bias("up_proj") if mu is not None else None,
        bias_g=_bias("gate_proj") if (mu is not None and use_gate) else None,
        mu=None if mu is None else mu.detach().clone(),
        input_alloc=str(input_alloc),
        alloc_mode=str(alloc_mode),
        row_block=int(row_block),
        target_density=None if density is None else float(density),
        tau_iters=int(tau_iters),
        count_reads=bool(count_reads),
    )


# ---------------------------------------------------------------------------
# online scoring
# ---------------------------------------------------------------------------

def _band_bounds(levels, H: int, n_cols: Optional[torch.Tensor]) -> List[tuple]:
    """Per-level ``[lo, hi)`` coordinate-rank bounds.

    With ``n_cols=None`` (uniform) the bounds are ints shared by every token: band
    ``l`` is ranks ``[cum·H, (cum+col_frac)·H)`` of the token's descending-|·|
    coordinate order.

    With ``n_cols`` ``(T,)`` — a per-slot total coordinate count from
    :func:`sparse_probe.allocate_input_reads` — the whole ladder is *stretched* by
    ``n_cols / (col_total·H)``: the bands keep their relative widths, so a slot
    granted twice the coordinates reads twice the entries at every level and the
    pooled read budget is preserved exactly.
    """
    ctot = max(sum(cf for cf, _ in levels), 1e-12)
    bounds, cum = [], 0.0
    for cf, _ in levels:
        lo, hi = cum, cum + cf
        cum = hi
        if n_cols is None:
            bounds.append((int(round(lo * H)), int(round(hi * H))))
        else:
            s = n_cols.to(torch.float32).unsqueeze(-1) / ctot      # (T,1)
            bounds.append(((s * lo).round().long(), (s * hi).round().long()))
    return bounds


def wsparse_layer_bands(
    sorted_abs: torch.Tensor,
    probe: WeightSparseProbe,
    selected_experts: torch.Tensor,
    n_cols: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Threshold-mode band edges for a whole layer, in one batched bisection.

    Returns ``(edges_u, edges_g)``, each ``(T, K, L+1)`` long: band ``l`` of slot
    ``(t, k)`` covers coordinate **ranks** ``[edges[t,k,l], edges[t,k,l+1])`` of the
    token's shared descending-``|x|`` order, and is read for ``row_frac[l]`` of the
    channels. ``edges_g`` is ``None`` for the up-only proxy.

    **Why this is cheap.** The exact product rule prices reading column ``i`` down to
    level ``l`` at ``theta_i[l]·|x_i|``, which needs a ``(T·K, L, H)`` tensor — 840 MB
    per branch per layer at eval shapes, and 16 bisection passes over it. But
    ``theta_i[l]`` is a quantile of column ``i``'s weight magnitudes, and this
    model's per-column weight scales barely vary (column-norm CV 0.022), so
    ``theta_i[l] ~ pbar[l]`` and the price ordering collapses onto ``|x_i|`` — the
    order that is **already sorted once per token**. The level of a coordinate is then
    a function of its rank alone, the per-token threshold search is
    ``L`` searchsorteds on that shared order, and the whole pass costs
    ``O(T·K·L)`` instead of ``O(T·K·L·H)``.

    Crucially this approximates only *which* level a column lands in. The read
    **count** stays exact — ``sum_l n_rows[l] · (#coords in band l)`` — so the budget
    claim is unaffected, and the masks still use the true per-column thresholds.
    (``alloc_mode="taux"`` keeps the exact per-column pricing for offline comparison;
    the screen measures the two within 0.01 rel_err of each other.)
    """
    T, K = selected_experts.shape
    H = sorted_abs.shape[-1]
    I = probe.Wu[0].shape[0]
    L = len(probe.levels)
    dev = sorted_abs.device
    n_rows = torch.tensor(
        level_row_counts([rf for _, rf in probe.levels], I, probe.row_block),
        dtype=torch.float32, device=dev)                            # (L,) desc
    budget = torch.full((T, K), probe.density * I * H, dtype=torch.float32,
                        device=dev)
    if n_cols is not None:
        budget = budget * (n_cols.to(torch.float32)
                           / max(probe.alloc_keep * H, 1e-12))
    s_asc = torch.flip(sorted_abs.to(torch.float32), dims=[-1]).contiguous()

    out = []
    for thr in (probe.thr_u, probe.thr_g):
        if thr is None:
            out.append(None)
            continue
        pbar = thr.to(torch.float32).mean(dim=2)[selected_experts]   # (T,K,L) asc
        pbar = pbar.clamp_min(torch.finfo(torch.float32).tiny)

        def cum(tau):                       # tau (T,K,1) -> (T,K,L) coords at level<=l
            q = (tau / pbar).reshape(T, K * L)          # |x| threshold per (slot,level)
            n = H - torch.searchsorted(s_asc, q.contiguous(), right=True)
            return n.reshape(T, K, L).clamp_(0, H)

        def reads_of(c):                    # (T,K,L) cumulative -> (T,K) read count
            per = c - torch.cat([torch.zeros_like(c[..., :1]), c[..., :-1]], dim=-1)
            return (per.to(torch.float32) * n_rows).sum(dim=-1)

        hi = (sorted_abs[:, :1].to(torch.float32).unsqueeze(-1)
              * pbar.amax(dim=-1, keepdim=True)) * 1.000001          # (T,K,1)
        lo = torch.zeros_like(hi)
        best = torch.zeros((T, K, L), dtype=torch.long, device=dev)
        best_reads = torch.zeros((T, K), dtype=torch.float32, device=dev)
        for _ in range(int(probe.tau_iters)):
            mid = 0.5 * (lo + hi)
            c = cum(mid)
            r = reads_of(c)
            feas = (r <= budget).unsqueeze(-1)                        # (T,K,1)
            hi = torch.where(feas, mid, hi)      # feasible -> lower tau, read more
            lo = torch.where(feas, lo, mid)
            best = torch.where(feas, c, best)
            best_reads = torch.where(feas.squeeze(-1), r, best_reads)
        if probe.count_reads:
            probe.reads_sum += float(best_reads.sum()) / (I * H)
            probe.reads_n += int(best_reads.numel())
        out.append(torch.cat([torch.zeros_like(best[..., :1]), best], dim=-1))
    return out[0], out[1]


def wsparse_expert_scores(
    x: torch.Tensor,
    probe: WeightSparseProbe,
    eid: int,
    ranks: Optional[torch.Tensor] = None,
    n_cols: Optional[torch.Tensor] = None,
    lvl_u: Optional[torch.Tensor] = None,
    lvl_g: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Proxy of one expert's channel magnitudes from an unstructured read set.

    Args:
        x: ``(T, H)`` hidden states of the tokens routed to this expert — **not**
            pre-sparsified (the sparsity is applied per level in here).
        probe: this layer's :class:`WeightSparseProbe`.
        eid: expert index.
        ranks: ``(T, H)`` long descending-|·| coordinate ranks of the *scored*
            input (``x`` or ``x - mu``). Computed by the caller once per token and
            shared across the token's K experts, since the coordinate order is a
            property of the token.
        n_cols: ``(T,)`` long, this slot's share of the pooled budget under
            ``input_alloc="router"``; ``None`` for the uniform ladder.
        lvl_u, lvl_g: ``(T, L+1)`` long band edges for this expert's tokens from
            :func:`wsparse_layer_bands` (``alloc_mode="tau"``). Passing them is how
            the eval path runs; without them ``tau`` falls back to the exact
            per-column bisection (``taux``), which is correct but ~50x slower.

    Returns:
        ``(T, I)`` float32 proxy of ``|SiLU(gate_j·x) · (up_j·x)|`` (or ``|up_j·x|``
        for the up-only proxy).
    """
    H = x.shape[-1]
    I = probe.Wu[eid].shape[0]
    delta = x if probe.mu is None else x - probe.mu.to(device=x.device, dtype=x.dtype)
    # "tau" assigns bands by rank against per-token edges (cheap, the eval path);
    # "taux" prices every column exactly and returns a per-coordinate level map.
    band_mode = probe.alloc_mode == "tau"
    exact_mode = probe.alloc_mode == "taux"
    sorted_abs = None
    if ranks is None and not exact_mode:
        ranks, sorted_abs = descending_abs_ranks(delta)
    bounds = (None if (band_mode or exact_mode)
              else _band_bounds(probe.levels, H, n_cols))
    if band_mode and lvl_u is None:
        # standalone fallback: resolve this expert's band edges here. The eval path
        # passes them in from one batched per-layer call instead.
        if sorted_abs is None:
            sorted_abs = delta.abs().sort(dim=-1, descending=True).values
        sel1 = torch.full((x.shape[0], 1), int(eid), dtype=torch.long,
                          device=x.device)
        eu, eg = wsparse_layer_bands(
            sorted_abs, probe, sel1,
            n_cols=None if n_cols is None else n_cols.unsqueeze(-1))
        lvl_u = eu[:, 0]
        lvl_g = None if eg is None else eg[:, 0]
    if exact_mode:
        n_rows = torch.tensor(
            level_row_counts([rf for _, rf in probe.levels], I, probe.row_block),
            dtype=torch.float32, device=x.device)
        # per-slot read budget; router alloc scales it by this slot's share, whose
        # mean over the token's K slots is 1, so the pooled budget is preserved.
        budget = torch.full((x.shape[0],), probe.density * I * H,
                            dtype=torch.float32, device=x.device)
        if n_cols is not None:
            budget = budget * (n_cols.to(torch.float32)
                               / max(probe.alloc_keep * H, 1e-12))

    Lv = len(probe.levels)
    lev_ar = torch.arange(Lv, device=x.device).view(Lv, 1, 1)
    rank_lo = rank_hi = None
    if bounds is not None:
        if torch.is_tensor(bounds[0][0]):                     # per-token (router)
            rank_lo = torch.stack([b[0] for b in bounds], 0)  # (Lv, T, 1)
            rank_hi = torch.stack([b[1] for b in bounds], 0)
        else:
            # shared integer bounds: cache on the probe, since building them per
            # expert is 2 host->device copies x 128 experts x 48 layers per batch.
            key = (H, id(probe.levels))
            cache = getattr(probe, "_rank_bound_cache", None)
            if cache is None or cache[0] != key or cache[1].device != x.device:
                lo_t = torch.tensor([b[0] for b in bounds], device=x.device).view(Lv, 1, 1)
                hi_t = torch.tensor([b[1] for b in bounds], device=x.device).view(Lv, 1, 1)
                object.__setattr__(probe, "_rank_bound_cache", (key, lo_t, hi_t))
            _, rank_lo, rank_hi = probe._rank_bound_cache

    def _branch(Wlist, thr, bias, edges):
        """``sum_l (delta * band_l) @ (W * 1[|W| >= theta_l])^T`` as one bmm.

        Written stacked rather than as a python loop over levels because the loop
        is pure launch overhead at MoE shapes: 128 experts x 2 branches x L levels
        x ~6 kernels is ~12k launches per block, which measured 3x slower than the
        arithmetic. Stacking makes it ~5 launches per (expert, branch).
        """
        W = Wlist[eid]                                        # (I, H)
        if exact_mode:
            lvl, reads = _tau_bands(delta.abs().float(), thr[eid], n_rows, budget,
                                    probe.tau_iters)
            if probe.count_reads:
                probe.reads_sum += float(reads.sum()) / (I * H)
                probe.reads_n += int(reads.numel())
            sel = lvl.unsqueeze(0) == lev_ar                  # (Lv, T, H)
        else:
            if band_mode:
                lo = edges[:, :Lv].t().unsqueeze(-1)          # (Lv, T, 1)
                hi = edges[:, 1:].t().unsqueeze(-1)
            else:
                lo, hi = rank_lo, rank_hi
            r = ranks.unsqueeze(0)
            sel = (r >= lo) & (r < hi)                        # (Lv, T, H)
        xs = delta.unsqueeze(0) * sel.to(delta.dtype)          # (Lv, T, H)
        t = thr[eid].to(W.dtype).unsqueeze(1)                   # (Lv, 1, H)
        if probe.row_block > 1:
            keep = (block_scores(W, probe.row_block).to(W.dtype).unsqueeze(0) >= t)
            keep = keep.repeat_interleave(probe.row_block, dim=1)   # (Lv, I, H)
        else:
            keep = W.unsqueeze(0).abs() >= t
        Wm = W.unsqueeze(0) * keep.to(W.dtype)
        acc = torch.bmm(xs, Wm.transpose(1, 2).to(xs.dtype)).sum(dim=0).float()
        if bias is not None:
            acc = acc + bias[eid].to(acc.dtype).unsqueeze(0)
        return acc

    up_hat = _branch(probe.Wu, probe.thr_u, probe.bias_u, lvl_u)
    if probe.Wg is None:
        return up_hat.abs()
    gate_hat = _branch(probe.Wg, probe.thr_g, probe.bias_g, lvl_g)
    return (F.silu(gate_hat) * up_hat).abs()


# ---------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------

def wsparse_used_param_fraction(levels, rho_channel: float, n_matrices: int = 2,
                                density: Optional[float] = None) -> float:
    """Whole-FFN used-parameter fraction: **scoring reads + compute reads**.

    Same frame as :func:`sparse_probe.used_param_fraction`, with the staircase's
    per-branch read density in place of ``rho_input``::

        used = rho_channel + n_matrices * density / 3

    ``density = sum_l col_frac[l] * row_frac[l]`` (:func:`levels_density`), so a
    single-level ``((p, 1.0),)`` staircase reproduces `input_sparse`'s
    ``rho_channel + n·p/3`` exactly. Conservative in the same way: the scoring
    reads and the compute reads overlap on the kept rows and this bills that
    overlap twice.

    ``density`` overrides the staircase area — that is the threshold mode, where the
    band widths float per token and the budget is set directly.
    """
    d = levels_density(levels) if density is None else float(density)
    return float(rho_channel) + int(n_matrices) * d / 3.0


def report_wsparse_accounting(
    levels, rho_channel: float, use_gate: bool = True,
    I: int = 768, H: int = 2048, meanfix: bool = False,
    density: Optional[float] = None,
) -> dict:
    """Used parameters plus the metadata unstructured sparsity actually needs.

    Three storage terms are reported, all as fractions of the **expert FFN**
    (3 matrices), because they are what distinguishes this from `input_sparse`'s
    zero-overhead column reads:

    * ``thresholds`` — ``L·H`` floats per expert per branch. What the *mask
      definition* costs; tiny (``L/I`` of a matrix).
    * ``mean_fix`` — ``I`` floats per expert per branch for ``W mu``; ``1/H``.
    * ``level_map`` — ``ceil(log2(L+1))`` bits per weight of the scored branches.
      What it costs a kernel to *skip* the non-kept entries without reading them.
      This is the honest price of unstructured sparsity, and it is charged against
      fp16 weights (16 bits). A semi-structured variant (whole blocks of ``r``
      channels) divides it by ``r``.
    """
    levels = parse_levels(levels)
    n = 2 if use_gate else 1
    L = len(levels)
    d = levels_density(levels) if density is None else float(density)
    used = wsparse_used_param_fraction(levels, rho_channel, n, density=d)
    bits = max(1, (L + 1 - 1).bit_length())          # ceil(log2(L+1))
    return {
        "levels": levels,
        "density_per_branch": d,
        "n_matrices": n,
        "col_total": float(sum(cf for cf, _ in levels)),
        "used_param_fraction": used,
        "used_param_cut": 1.0 - used,
        "scoring": n * d / 3.0,
        "compute": float(rho_channel),
        "storage_thresholds_frac_of_ffn": n * (L * H) / (I * H) / 3.0,
        "storage_mean_fix_frac_of_ffn": (n * I / (I * H) / 3.0) if meanfix else 0.0,
        "storage_level_map_frac_of_ffn": n * (bits / 16.0) / 3.0,
        "level_map_bits_per_weight": bits,
        # references on the same scale
        "input_sparse_equivalent_rho_input": d,
        "oracle_mag_kept": (1.0 + 1.0 + float(rho_channel)) / 3.0,
    }


def print_wsparse_accounting(levels, rho_channel, use_gate=True, I=768, H=2048,
                             meanfix=False, input_alloc="uniform",
                             alloc_mode="rank", density=None):
    a = report_wsparse_accounting(levels, rho_channel, use_gate, I, H, meanfix,
                                  density=density)
    lv = "+".join(f"{cf:g}x{rf:g}" for cf, rf in a["levels"])
    if alloc_mode in ("tau", "taux"):
        lv = alloc_mode + "[" + ",".join(f"{rf:g}" for _, rf in a["levels"]) + "]"
    _print(
        f"[weight_sparse] levels={lv} density={a['density_per_branch']:.4f}/branch "
        f"(alloc_mode={alloc_mode}, cols touched {a['col_total']:.3f}) "
        f"rho_channel={float(rho_channel):g} "
        f"-> USED PARAMS={a['used_param_fraction']:.4f} "
        f"(scoring {a['scoring']:.4f} + compute {a['compute']:.4f}), "
        f"cut {100 * a['used_param_cut']:.1f}%; meanfix={meanfix} "
        f"input_alloc={input_alloc}; "
        f"metadata: thresholds {100 * a['storage_thresholds_frac_of_ffn']:.2f}% + "
        f"mean-fix {100 * a['storage_mean_fix_frac_of_ffn']:.2f}% + level-map "
        f"{100 * a['storage_level_map_frac_of_ffn']:.2f}% of expert FFN "
        f"({a['level_map_bits_per_weight']} bit/weight); "
        f"same density as input_sparse rho_input="
        f"{a['input_sparse_equivalent_rho_input']:.4f}"
    )
    return a
