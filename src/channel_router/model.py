"""§1.1 router architecture — every component flag-gated and individually ablatable.

    phi(h)   = concat[ h[S_out] , Q^T h ]              Q = Σ^{-1/2} V (folded whitening)
    score_i  = head_i(phi(h)) + beta_i + log g_e       (log domain)
    select   = hot set ∪ top-B of score, optionally restricted to top-n tiles

Four deviations from the plan's literal default, each forced by the oracle's algebra
or by the MoE's token-dependent activation set:

1. **Channel embeddings are keyed by physical channel** (``E·I = 98304``), not by the
   ``D = K·I = 6144`` activated slots: slot 3 is a different expert for every token, so
   a slot-keyed embedding is meaningless. Only the ``K`` selected experts' blocks are
   gathered online, so the FLOP cost is the plan's; the parameter cost is charged
   against the layer's *stored* FFN (see ``metrics.router_accounting``).

2. **Scores live in the log domain and the default head is ``swiglu``.** The oracle is a
   product, ``imp = g_e · |silu(W_g h)| · |W_u h| · ‖W_d[:,j]‖``, so its logarithm is a
   *sum* of two data terms and two constants — and the gate factor passes through
   ``silu``, which is **not** symmetric: a large *negative* gate pre-activation gives
   ``silu ≈ 0`` (a dead channel) while a large positive one gives ``silu ≈ gate``. So
   ranking by ``|⟨g_i,h⟩|`` is actively wrong, which the ablation confirms (a plain
   signed linear head beats the ``abs`` head by +4.6 mass-recall points, and its
   structural init by +29). ``head='swiglu'`` applies ``silu`` to the gate factor and the
   magnitude to the up factor, so at Stage-A init the router *is* the rank-``r``
   truncated oracle. ``bilinear`` / ``abs`` / ``linear`` are kept as ablation rows.

3. **The routing weight enters as ``+log g_e``.** It is part of the oracle definition
   and is already computed by the model's own gate, so leaving it out would hand the
   baselines a free advantage. Ablatable via ``use_g=False``.

4. **Tiles are built inside an expert.** A tile is a contiguous group of one expert's
   channels, so ``K · n_tiles_per_expert`` tiles are visible to a token and P5's
   "native expert boundary" baseline is the special case of one tile per expert.
   Tiling across experts coalesces nothing (different weight matrices).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.channel_router.metrics import select_topB

__all__ = ["ChannelRouter", "whiten_stats", "stage_a_init"]

_LOG_EPS = 1e-8


@torch.no_grad()
def whiten_stats(X: torch.Tensor, device="cuda", jitter: float = 1e-5,
                 chunk: int = 65536, center: bool = False):
    """Σ, Σ^{1/2}, Σ^{-1/2}, mean of the MLP-input distribution, in fp32/fp64.

    ``center=False`` (default) uses the second moment ``E[h hᵀ]``: the scorer sees
    ``h`` itself, and the common-mode direction of ``h`` is a real part of what the
    expert rows act on.
    """
    H = X.shape[1]
    S = torch.zeros((H, H), dtype=torch.float64, device=device)
    mu = torch.zeros(H, dtype=torch.float64, device=device)
    n = 0
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk].to(device=device, dtype=torch.float64)
        S += x.t() @ x
        mu += x.sum(0)
        n += x.shape[0]
    S /= n
    mu /= n
    if center:
        S -= torch.outer(mu, mu)
    S = S + torch.eye(H, dtype=S.dtype, device=device) * (
        jitter * float(torch.diagonal(S).mean()))
    evals, evecs = torch.linalg.eigh(S)
    evals = evals.clamp_min(1e-12)
    Sh = (evecs * evals.sqrt()) @ evecs.t()
    Sinv = (evecs * evals.rsqrt()) @ evecs.t()
    return S.float(), Sh.float(), Sinv.float(), mu.float()


class ChannelRouter(nn.Module):
    """Predict, from ``h``, which activated expert-FFN channels must be computed.

    Args:
        H, E, I, K: hidden dim, expert count, per-expert intermediate width, top-k.
        r: low-rank projection width; m: number of passthrough outlier dims.
        head: ``swiglu`` (log|silu(a·φ)| + log|u·φ| — the oracle's own form) |
            ``bilinear`` (log|a·φ| + log|u·φ|) | ``abs`` (log|a·φ|) | ``linear``.
        use_bias: per-channel static prior ``beta_i`` (subsumes ‖W_d‖ and P4's prior).
        use_g: add ``log g_e`` — the router weight that the oracle itself contains.
        n_tiles_per_expert: 0 disables the level-1 tile scorer.
    """

    def __init__(self, H: int, E: int, I: int, K: int, *, r: int = 32, m: int = 16,
                 head: str = "swiglu", use_bias: bool = True, use_g: bool = True,
                 n_tiles_per_expert: int = 0):
        super().__init__()
        if head not in ("swiglu", "bilinear", "abs", "linear"):
            raise ValueError(head)
        self.H, self.E, self.I, self.K = H, E, I, K
        self.r, self.m, self.head = r, m, head
        self.rp = r + m
        self.use_g = use_g
        self.n_tiles_per_expert = n_tiles_per_expert

        self.Q = nn.Parameter(torch.zeros(H, r))
        self.register_buffer("outlier_idx", torch.zeros(m, dtype=torch.long))
        self.register_buffer("feat_scale", torch.ones(self.rp))
        self.C = nn.Parameter(torch.zeros(E * I, self.rp))
        self.two_factor = head in ("swiglu", "bilinear")
        self.silu_floor = 1e-3
        self.C2 = nn.Parameter(torch.zeros(E * I, self.rp)) if self.two_factor else None
        self.beta = nn.Parameter(torch.zeros(E * I)) if use_bias else None
        self.register_buffer("hot_mask", torch.zeros(E * I, dtype=torch.bool))
        if n_tiles_per_expert:
            self.register_buffer("tile_of", torch.zeros(E * I, dtype=torch.long))

    # ---------------------------------------------------------------- features
    def features(self, h: torch.Tensor) -> torch.Tensor:
        """``(T, H)`` -> ``(T, r+m)``: one ``(H, r)`` matmul plus an index_select."""
        hh = h.to(self.Q.dtype)
        f = hh @ self.Q
        if self.m:
            f = torch.cat([f, hh[:, self.outlier_idx]], dim=1)
        return f * self.feat_scale

    # ------------------------------------------------------------------ scores
    def score(self, h: torch.Tensor, sel: torch.Tensor,
              g: torch.Tensor | None = None) -> torch.Tensor:
        """``(T,K,I)`` log-domain scores over the token's activated channel set.

        Loops over the experts present in ``sel`` — the loop the forward pass already
        runs — so only the selected experts' embedding blocks are touched.
        """
        T, K = sel.shape
        f = self.features(h)                                          # (T, rp)
        out = torch.zeros((T, K, self.I), dtype=f.dtype, device=f.device)
        C = self.C.view(self.E, self.I, self.rp)
        C2 = self.C2.view(self.E, self.I, self.rp) if self.C2 is not None else None
        beta = self.beta.view(self.E, self.I) if self.beta is not None else None
        for e in sel.unique().tolist():
            tok, slot = (sel == e).nonzero(as_tuple=True)
            ft = f[tok]                                               # (n, rp)
            a = ft @ C[e].t()                                         # (n, I)
            if self.head == "linear":
                s = a
            elif self.head == "abs":
                s = a.abs().clamp_min(_LOG_EPS).log()
            else:
                u = (ft @ C2[e].t()).abs().clamp_min(_LOG_EPS).log()
                if self.head == "swiglu":
                    # Keep the gate factor's SiLU shape: a large *negative* gate is a
                    # dead channel, not a strong one. Floor at ``silu_floor`` rather than
                    # taking |silu|: |silu| has a log-singularity at a=0 and revives
                    # a<0 channels, which made the head untrainable (measured: trained
                    # worse than its own init) even though the init is the best of the
                    # four heads.
                    s = F.silu(a).clamp_min(self.silu_floor).log() + u
                else:
                    s = a.abs().clamp_min(_LOG_EPS).log() + u
            if beta is not None:
                s = s + beta[e].unsqueeze(0)
            out[tok, slot] = s
        if self.use_g and g is not None:
            gg = g.to(out.device, out.dtype).unsqueeze(-1)
            if self.head == "linear":
                out = out * gg
            else:
                out = out + gg.clamp_min(1e-12).log()
        return out

    @torch.no_grad()
    def set_feature_scale(self, phi_std: torch.Tensor):
        """Normalize features to unit std **without changing the function**.

        ``feat_scale ← 1/std`` and ``C ← C · std`` leave every score identical, so the
        Stage-A init guarantee survives while the optimizer sees a well-conditioned
        parameterization (feature stds span ~3 orders of magnitude after whitening
        because the outlier passthrough coords are unwhitened).
        """
        s = phi_std.to(self.feat_scale.device).clamp_min(1e-8)
        self.feat_scale.copy_(1.0 / s)
        self.C.data.mul_(s.to(self.C.dtype))
        if self.C2 is not None:
            self.C2.data.mul_(s.to(self.C2.dtype))

    def tile_scores(self, score: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
        """Level-1 tile score by logsumexp pooling; ``(T, K*n_tiles_per_expert)``.

        In the log domain this is ``log Σ_{i∈tile} score_i`` — the tile's score mass.
        """
        if not self.n_tiles_per_expert:
            raise RuntimeError("tile level disabled")
        T, K, I = score.shape
        nt = self.n_tiles_per_expert
        tid = self.tile_of.view(self.E, self.I)[sel]                  # (T,K,I) in [0,nt)
        off = (torch.arange(K, device=score.device).view(1, K, 1) * nt + tid).reshape(T, -1)
        flat = score.reshape(T, K * I)
        mx = torch.full((T, K * nt), float("-inf"), device=score.device, dtype=score.dtype)
        mx = mx.scatter_reduce(1, off, flat, reduce="amax", include_self=True)
        ex = torch.exp(flat - mx.gather(1, off))
        sm = torch.zeros_like(mx).scatter_add_(1, off, ex)
        return mx + sm.clamp_min(1e-30).log()

    def tile_allowed(self, score: torch.Tensor, sel: torch.Tensor,
                     top_tiles: int) -> torch.Tensor:
        """``(T,K,I)`` bool: channels inside the token's top-``top_tiles`` tiles."""
        T, K, I = score.shape
        nt = self.n_tiles_per_expert
        ts = self.tile_scores(score, sel)
        keep_t = torch.zeros_like(ts, dtype=torch.bool)
        keep_t.scatter_(1, ts.topk(min(top_tiles, ts.shape[1]), dim=1).indices, True)
        tid = self.tile_of.view(self.E, self.I)[sel]
        off = (torch.arange(K, device=score.device).view(1, K, 1) * nt + tid).reshape(T, -1)
        return keep_t.gather(1, off).reshape(T, K, I)

    # --------------------------------------------------------------- selection
    def select(self, h: torch.Tensor, sel: torch.Tensor, B: int, *,
               g: torch.Tensor | None = None, top_tiles: int = 0,
               slack: float = 1.0) -> torch.Tensor:
        """Predicted keep-mask ``(T,K,I)`` with exactly ``round(B*slack)`` kept/token."""
        score = self.score(h, sel, g)
        return self.select_from_score(score, sel, B, top_tiles=top_tiles, slack=slack)

    def select_from_score(self, score: torch.Tensor, sel: torch.Tensor, B: int, *,
                          top_tiles: int = 0, slack: float = 1.0) -> torch.Tensor:
        budget = min(int(round(B * slack)), sel.shape[1] * self.I)
        if top_tiles and self.n_tiles_per_expert:
            score = score.masked_fill(~self.tile_allowed(score, sel, top_tiles),
                                      float("-inf"))
        if bool(self.hot_mask.any()):
            score = score.masked_fill(self.hot_mask.view(self.E, self.I)[sel],
                                      float("inf"))
        return select_topB(score, budget)

    def set_hot(self, freq: torch.Tensor, size: int):
        """P4 hot set: the ``size`` globally most-frequent channels are always kept."""
        self.hot_mask.zero_()
        if size > 0:
            self.hot_mask[freq.argsort(descending=True)[:size]] = True


# ------------------------------------------------------------------ Stage A init
@torch.no_grad()
def stage_a_init(router: ChannelRouter, w, Sh: torch.Tensor, Sinv: torch.Tensor, *,
                 source: str = "gate", outlier_idx: torch.Tensor | None = None,
                 freq: torch.Tensor | None = None, lam: float = 1.0,
                 bias_init: str = "colnorm", n_basis_experts: int = 32,
                 device="cuda") -> dict:
    """Structural initialization (§2 Stage A), no gradients.

    ``V`` = top-``r`` eigenvectors of ``Σ_e (W_e Σ^{1/2})ᵀ(W_e Σ^{1/2})`` — the
    whitened basis in which the linear part of the gate/up scores lives (equivalently
    the top right singular vectors of the stacked whitened weight rows). Then
    ``Q = Σ^{-1/2} V`` and ``c_i = V^T Σ^{1/2} w_i``, so at init

        c_iᵀ φ(h) = w_iᵀ Σ^{1/2} V Vᵀ Σ^{-1/2} h  ≈  ⟨w_i, h⟩

    and the ``bilinear`` head evaluates ``log|⟨g_i,h⟩| + log|⟨u_i,h⟩| + log‖W_d[:,i]‖ +
    log g_e`` — the oracle with ``silu`` replaced by ``|·|`` and each factor truncated
    to rank ``r``. The plan's Stage-A checkpoint standard (init recall matches the
    whitened-SVD baseline) is therefore structural rather than empirical.
    """
    E, I, H = w.Wu.shape
    r = router.r
    dev = torch.device(device)
    Sh, Sinv = Sh.to(dev), Sinv.to(dev)

    def basis(mat_name):
        gram = torch.zeros((H, H), dtype=torch.float64, device=dev)
        step = max(1, E // n_basis_experts)
        for e in range(0, E, step):
            A = getattr(w, mat_name)[e].to(dev, torch.float32) @ Sh   # (I,H)
            gram += (A.t() @ A).double()
        evals, evecs = torch.linalg.eigh(gram)
        return evecs[:, -r:].flip(-1).float(), evals.flip(0).float()

    primary = "Wg" if source in ("gate", "both") else "Wu"
    secondary = "Wu" if primary == "Wg" else "Wg"
    if source == "both" and router.C2 is not None:
        # one shared basis for both factors, built from the concatenation
        Vg, sg = basis("Wg")
        Vu, su = basis("Wu")
        V = torch.linalg.qr(torch.cat([Vg[:, :r // 2], Vu[:, :r - r // 2]], 1))[0][:, :r]
        spec = torch.cat([sg[:r // 2], su[:r - r // 2]])
    else:
        V, spec = basis(primary)
    router.Q.data.copy_((Sinv @ V).to(router.Q.dtype))
    if outlier_idx is not None:
        router.outlier_idx.copy_(outlier_idx.to(router.outlier_idx.device))

    VtSh = V.t() @ Sh                                                 # (r,H)

    def embed(mat_name):
        C = torch.zeros((E * I, router.rp), dtype=torch.float32, device=dev)
        for e in range(E):
            C[e * I:(e + 1) * I, :r] = getattr(w, mat_name)[e].to(dev, torch.float32) @ VtSh.t()
        return C

    router.C.data.copy_(embed(primary).to(router.C.dtype))
    if router.C2 is not None:
        router.C2.data.copy_(embed(secondary).to(router.C2.dtype))

    if router.beta is not None:
        b = torch.zeros(E * I, dtype=torch.float32, device=dev)
        if bias_init in ("colnorm", "both"):
            b += w.col_norm.reshape(-1).to(dev).clamp_min(1e-12).log()
        if bias_init in ("freq", "both") and freq is not None:
            f = freq.reshape(-1).to(dev).clamp(1e-4, 1 - 1e-4)
            b += lam * torch.log(f / (1 - f))
        router.beta.data.copy_(b.to(router.beta.dtype))
    return {"spectrum_head": spec[:r].tolist(), "source": primary,
            "bias_init": bias_init}
