"""§1.3 mandatory baselines, all behind one interface.

Every scorer maps ``(h, sel, g) -> (T, K, I)`` log-domain scores, exactly like
``ChannelRouter.score``, so the §0.3 protocol runs over router and baselines with the
same code path and the same budget.

**Fairness rule.** Two multiplicative factors of the oracle are available online at
zero parameter cost: the routing weight ``g_e`` (the model's own gate already computed
it) and the down-projection column norm ``‖W_d[:,j]‖`` (a per-channel constant, 2 floats
per channel). Withholding them from the baselines while the router uses them would
manufacture a win, so ``prior='both'`` (the default) grants them to *every* method and
the comparison isolates the learned/derived part. The ``prior='none'`` rows are kept for
the record.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["BaseScorer", "StaticFreq", "SvdScorer", "RandomProjScorer", "LshScorer",
           "ProductKeyScorer", "VqScorer", "DejaVuMLP", "build_baseline",
           "BASELINE_NAMES"]

_EPS = 1e-8


def _log(x):
    return x.abs().clamp_min(_EPS).log()


class BaseScorer:
    """Common machinery: the free per-channel/per-slot prior and the expert loop."""

    name = "base"
    params = 0

    def __init__(self, w, prior: str = "both"):
        self.w = w
        self.prior = prior
        self.I = w.I
        self.E = w.E
        self.log_colnorm = w.col_norm.clamp_min(1e-12).log()      # (E, I)

    def _apply_prior(self, s: torch.Tensor, sel: torch.Tensor, g: torch.Tensor):
        if self.prior in ("colnorm", "both"):
            s = s + self.log_colnorm.to(s.device)[sel]
        if self.prior in ("g", "both") and g is not None:
            s = s + g.to(s.device, s.dtype).clamp_min(1e-12).log().unsqueeze(-1)
        return s

    def score(self, h, sel, g=None):
        raise NotImplementedError

    def meta(self):
        return {"name": self.name, "params": int(self.params), "prior": self.prior}


class StaticFreq(BaseScorer):
    """Zero-parameter static predictor: rank by held-out keep frequency (P4).

    The floor every learned method must clear. ``freq`` is ``(E*I,)`` measured on a
    disjoint token slice, so this is a *legitimate* predictor, not an oracle.
    """

    name = "static_freq"

    def __init__(self, w, freq, prior="none"):
        super().__init__(w, prior)
        self.freq = freq.reshape(w.E, w.I).clamp(1e-6, 1 - 1e-6)
        self.logit = torch.log(self.freq / (1 - self.freq))
        self.params = 0                                # derived from calibration data

    def score(self, h, sel, g=None):
        s = self.logit.to(h.device)[sel].to(torch.float32)         # (T,K,I)
        return self._apply_prior(s, sel, g)


class SvdScorer(BaseScorer):
    """Rank-``r`` (optionally whitened) SVD of the expert projections — training-free.

    ``bilinear=True`` reproduces the oracle's algebraic form with rank-``r`` factors,
    ``|⟨ĝ_i,h⟩|·|⟨û_i,h⟩|``; ``bilinear=False`` uses the gate factor alone (the plan's
    "plain/whitened SVD of W_g" rows). Whitening uses ``Σ^{1/2}`` so the retained
    subspace is the one that matters *for the data*, worth a measured +0.03 recall at
    equal rank.
    """

    def __init__(self, w, r: int, *, whitened: bool, Sh=None, Sinv=None,
                 bilinear: bool = True, prior="both", device="cuda",
                 n_basis_experts: int = 32, source: str = "gate"):
        super().__init__(w, prior)
        self.r, self.whitened, self.bilinear = r, whitened, bilinear
        self.name = f"{'w' if whitened else ''}svd_r{r}{'_bil' if bilinear else '_gate'}"
        dev = torch.device(device)
        H = w.H
        A = (Sh.to(dev) if whitened else torch.eye(H, device=dev))
        prim = "Wg" if source == "gate" else "Wu"
        sec = "Wu" if prim == "Wg" else "Wg"
        gram = torch.zeros((H, H), dtype=torch.float64, device=dev)
        step = max(1, w.E // n_basis_experts)
        for e in range(0, w.E, step):
            M = getattr(w, prim)[e].to(dev, torch.float32) @ A
            gram += (M.t() @ M).double()
            if bilinear:
                M2 = getattr(w, sec)[e].to(dev, torch.float32) @ A
                gram += (M2.t() @ M2).double()
        V = torch.linalg.eigh(gram)[1][:, -r:].flip(-1).float()        # (H, r)
        # feature map: f = V^T A^{-1}... folded as Q = Sinv V (whitened) or V (plain)
        self.Q = ((Sinv.to(dev) @ V) if whitened else V)               # (H, r)
        self.Cp = torch.stack([getattr(w, prim)[e].to(dev, torch.float32) @ (V.t() @ A).t()
                               for e in range(w.E)])                   # (E, I, r)
        self.Cs = (torch.stack([getattr(w, sec)[e].to(dev, torch.float32) @ (V.t() @ A).t()
                                for e in range(w.E)]) if bilinear else None)
        self.params = 0                                # derived from frozen weights

    def score(self, h, sel, g=None):
        f = h.to(self.Q.dtype) @ self.Q                                # (T, r)
        T, K = sel.shape
        out = torch.zeros((T, K, self.I), dtype=torch.float32, device=f.device)
        for e in sel.unique().tolist():
            tok, slot = (sel == e).nonzero(as_tuple=True)
            ft = f[tok]
            s = _log(ft @ self.Cp[e].t())
            if self.Cs is not None:
                s = s + _log(ft @ self.Cs[e].t())
            out[tok, slot] = s
        return self._apply_prior(out, sel, g)


class RandomProjScorer(SvdScorer):
    """JL lower bound: the SVD basis replaced by a random Gaussian sketch."""

    def __init__(self, w, r: int, *, Sh=None, Sinv=None, bilinear=True, prior="both",
                 device="cuda", seed: int = 0, source="gate"):
        BaseScorer.__init__(self, w, prior)
        self.r, self.bilinear = r, bilinear
        self.name = f"randproj_r{r}"
        dev = torch.device(device)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        V = torch.linalg.qr(torch.randn(w.H, r, generator=gen))[0].to(dev)
        prim = "Wg" if source == "gate" else "Wu"
        sec = "Wu" if prim == "Wg" else "Wg"
        self.Q = V
        self.Cp = torch.stack([getattr(w, prim)[e].to(dev, torch.float32) @ V
                               for e in range(w.E)])
        self.Cs = (torch.stack([getattr(w, sec)[e].to(dev, torch.float32) @ V
                                for e in range(w.E)]) if bilinear else None)
        self.params = 0


class LshScorer(BaseScorer):
    """SimHash retrieval: score = sign-agreement between ``h`` and each channel's row.

    Training-free. ``R`` random hyperplanes; the score is the Hamming similarity of the
    sign codes, an estimator of ``1 − angle/π`` between ``h`` and the (gate ⊙ up
    surrogate) channel direction. Bucket tables are not needed for *ranking* — the
    similarity is the table lookup's population statistic.
    """

    name = "lsh"

    def __init__(self, w, bits: int = 64, prior="both", device="cuda", seed: int = 0,
                 source="gate"):
        super().__init__(w, prior)
        dev = torch.device(device)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.R = torch.randn(w.H, bits, generator=gen).to(dev)
        mat = "Wg" if source == "gate" else "Wu"
        self.codes = torch.stack([
            torch.sign(getattr(w, mat)[e].to(dev, torch.float32) @ self.R)
            for e in range(w.E)])                                     # (E, I, bits)
        self.bits = bits
        self.name = f"lsh_b{bits}"
        self.params = 0

    def score(self, h, sel, g=None):
        q = torch.sign(h.to(self.R.dtype) @ self.R)                   # (T, bits)
        T, K = sel.shape
        out = torch.zeros((T, K, self.I), dtype=torch.float32, device=q.device)
        for e in sel.unique().tolist():
            tok, slot = (sel == e).nonzero(as_tuple=True)
            out[tok, slot] = (q[tok] @ self.codes[e].t()) / self.bits
        return self._apply_prior(out, sel, g)


class ProductKeyScorer(BaseScorer):
    """Product-key scorer: two sub-key tables of size ``√(E·I)``, learned.

    ``score_c = ⟨q1(h), k1_{c1}⟩ + ⟨q2(h), k2_{c2}⟩`` with ``c = c1·√D + c2``. Params
    ``2·√D·(r/2) + H·r`` — the structural-trick alternative to free per-channel
    embeddings, i.e. the same online cost with ``√D`` instead of ``D`` rows.
    """

    name = "product_key"

    def __init__(self, w, r: int = 32, prior="both", device="cuda", seed: int = 0):
        super().__init__(w, prior)
        dev = torch.device(device)
        D = w.E * w.I
        n = int(D ** 0.5)
        while n * n != D and n * (D // n) != D:
            n -= 1
        self.n1, self.n2 = n, D // n
        gen = torch.Generator(device="cpu").manual_seed(seed)
        h = r // 2
        self.Q = (torch.randn(w.H, r, generator=gen) / w.H ** 0.5).to(dev).requires_grad_(True)
        self.k1 = (torch.randn(self.n1, h, generator=gen) / h ** 0.5).to(dev).requires_grad_(True)
        self.k2 = (torch.randn(self.n2, h, generator=gen) / h ** 0.5).to(dev).requires_grad_(True)
        self.r, self.hh = r, h
        self.params = w.H * r + (self.n1 + self.n2) * h

    def parameters(self):
        return [self.Q, self.k1, self.k2]

    def score(self, h, sel, g=None):
        f = h.to(self.Q.dtype) @ self.Q
        q1, q2 = f[:, :self.hh], f[:, self.hh:2 * self.hh]
        s1 = q1 @ self.k1.t()                                          # (T, n1)
        s2 = q2 @ self.k2.t()                                          # (T, n2)
        T, K = sel.shape
        I = self.I
        gids = sel.unsqueeze(-1) * I + torch.arange(I, device=sel.device).view(1, 1, I)
        c1, c2 = gids // self.n2, gids % self.n2
        out = torch.gather(s1, 1, c1.reshape(T, -1)).reshape(T, K, I) \
            + torch.gather(s2, 1, c2.reshape(T, -1)).reshape(T, K, I)
        return self._apply_prior(out, sel, g)


class VqScorer(BaseScorer):
    """Lookup alternative: k-means the hidden states, store each centroid's mean profile.

    Zero online arithmetic beyond the nearest-centroid search; params
    ``n_centroids·(H + E·I)``, which is the point — it is the *table* end of the
    spectrum against the router's *factorized* end.
    """

    name = "vq"

    def __init__(self, w, centroids: torch.Tensor, table: torch.Tensor, prior="none"):
        super().__init__(w, prior)
        self.mu = centroids                                            # (Cn, H)
        self.table = table                                             # (Cn, E*I) log-mean
        self.name = f"vq_c{centroids.shape[0]}"
        self.params = centroids.numel() + table.numel()

    def score(self, h, sel, g=None):
        d = torch.cdist(h.to(self.mu.dtype), self.mu)
        c = d.argmin(1)                                                # (T,)
        prof = self.table[c].view(-1, self.E, self.I)                  # (T,E,I)
        out = torch.gather(prof, 1, sel.unsqueeze(-1).expand(-1, -1, self.I))
        return self._apply_prior(out, sel, g)


class DejaVuMLP(BaseScorer):
    """Generic 2-layer MLP predictor ``H -> hidden -> E·I`` (Deja Vu's sparsity predictor).

    Tests whether the router's structure (whitened low-rank features + free channel
    embeddings + static prior) beats an unstructured predictor of the same *family*.
    Its parameter count is reported honestly: on an MoE the output layer must cover all
    ``E·I`` physical channels, so it is ~10× over the plan's §1.2 budget.
    """

    name = "dejavu"

    def __init__(self, w, hidden: int = 512, prior="both", device="cuda", seed: int = 0):
        super().__init__(w, prior)
        dev = torch.device(device)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        D = w.E * w.I
        self.W1 = (torch.randn(w.H, hidden, generator=gen) / w.H ** 0.5).to(dev).requires_grad_(True)
        self.W2 = (torch.randn(hidden, D, generator=gen) / hidden ** 0.5).to(dev).requires_grad_(True)
        self.b2 = torch.zeros(D, device=dev, requires_grad=True)
        self.hidden = hidden
        self.name = f"dejavu_h{hidden}"
        self.params = w.H * hidden + hidden * D + D

    def parameters(self):
        return [self.W1, self.W2, self.b2]

    def score(self, h, sel, g=None):
        z = F.relu(h.to(self.W1.dtype) @ self.W1)                      # (T, hidden)
        T, K = sel.shape
        I = self.I
        out = torch.zeros((T, K, I), dtype=z.dtype, device=z.device)
        W2 = self.W2.view(self.hidden, self.E, I)
        b2 = self.b2.view(self.E, I)
        for e in sel.unique().tolist():
            tok, slot = (sel == e).nonzero(as_tuple=True)
            out[tok, slot] = z[tok] @ W2[:, e] + b2[e]
        return self._apply_prior(out, sel, g)


BASELINE_NAMES = ["static_freq", "svd", "wsvd", "randproj", "lsh", "product_key",
                  "vq", "dejavu"]


def build_baseline(name: str, w, *, r=32, Sh=None, Sinv=None, freq=None,
                   device="cuda", prior="both", **kw):
    if name == "static_freq":
        return StaticFreq(w, freq, prior=prior)
    if name == "svd":
        return SvdScorer(w, r, whitened=False, Sh=Sh, Sinv=Sinv, device=device,
                         prior=prior, **kw)
    if name == "wsvd":
        return SvdScorer(w, r, whitened=True, Sh=Sh, Sinv=Sinv, device=device,
                         prior=prior, **kw)
    if name == "randproj":
        return RandomProjScorer(w, r, Sh=Sh, Sinv=Sinv, device=device, prior=prior, **kw)
    if name == "lsh":
        return LshScorer(w, device=device, prior=prior, **kw)
    if name == "product_key":
        return ProductKeyScorer(w, r=r, device=device, prior=prior, **kw)
    if name == "dejavu":
        return DejaVuMLP(w, device=device, prior=prior, **kw)
    raise ValueError(name)
