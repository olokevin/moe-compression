#!/usr/bin/env python
"""Activation-aware channel scorers: can a *data-aware* sketch of ``x`` rank channels
like `oracle_mag` at under 10% of one expert matrix?

Where this picks up. ``docs/exps/dynamic_active_param/btt_dynamic.md`` measured the
plain low-rank family dead: a per-expert truncated SVD of ``W_up``/``W_gate`` reaches
only recall 0.44-0.47 at rho=0.25 no matter the rank, because a Frobenius-optimal
truncation of ``W`` deletes exactly the fine row structure that decides the top-B.
Every scorer there was built from ``W`` **alone**. This script asks the obvious next
question: the scorer does not need to approximate ``W``, it needs to approximate
``W x`` for ``x`` drawn from the data — so bring the input/output activation
statistics in.

The unifying frame. Every cheap scorer in this family computes the true weight
against a **sketch** of the token:

    score_j ~ |SiLU(<g_j, x~>) * <u_j, x~>|,     x~ = a cheap r-dim sketch of x

and the error that perturbs the ranking is ``<w_j, x - x~>``. So all mechanisms are
comparable on one axis: **how much of the score-relevant energy of x survives, per
unit of per-expert cost**. Mechanisms that differ only in *which* r-dim sketch:

  ``insp``       per-token top-|x_i| coordinate subset (Prox stage 1, and what
                 ``sparse_probe`` uses). Adaptive per token, but the basis is the
                 arbitrary canonical one.
  ``pcabasis``   top-r eigenvectors of ``Sigma_x = E[x x^T]``. Fixed, but optimal
                 in expectation for retaining ``x``'s energy.
  ``actbasis``   **the optimum for a shared input factor** (derived below): top-r
                 eigenvectors of ``Sigma^{1/2} M Sigma^{1/2}`` with
                 ``M = sum_e W_e^T W_e``. Weight-aware *and* activation-aware.
  ``awsvd``      per-expert activation-aware SVD, i.e. SVD-LLM: truncate
                 ``W Sigma^{1/2}``. The quality ceiling of the input-side family,
                 but per-expert, so it cannot share (see cost note).
  ``outwhiten``  activation-aware SVD under an output-**whitened** objective:
                 equalize *relative* error across the I channels instead of
                 absolute. The one output-side lever that is not redundant.
  ``adapt``      per-token top-r of r' ``actbasis`` coefficients: a per-token
                 subspace drawn from a data-optimal dictionary (insp x actbasis).
  ``mix``        shared basis of rank r **plus** the top-q coordinates of the
                 residual ``x - P^T P x``. The two mechanisms are not competing
                 for the same energy, so their union may beat either.

Why a *shared* input factor is the point (MoE-native, 3x cheaper). With
``W_e ~= A_e P`` and ``P`` the **same** for every expert, ``h = P x`` is computed
once per token and reused by all K co-activated experts and both matrices, so per
token the scorer costs ``r*H + n_mat*K*r*I`` instead of ``n_mat*K*r*(H+I)``. In
units of one ``(I,H)`` matrix (the repo convention; a dense expert FFN is 3):

    shared:      cB = r/(K*I) + n_mat*r/H          <- r=64, n_mat=2: 0.073
    per-expert:  cB = n_mat*r*(H+I)/(I*H)          <- r=64, n_mat=2: 0.229

i.e. the shared form buys **3.1x the rank at equal cost**. A plain SVD cannot use
this: its ``V_r`` is expert-specific. Only a basis chosen from *layer-level*
statistics is shareable -- so activation-awareness is not merely an accuracy tweak
here, it is what unlocks the amortization. Bytes and FLOPs coincide for this family
(every factor is read once per token), unlike the quantized probe.

Derivation of ``actbasis``. Requiring a shared input factor, minimize the expected
squared score perturbation over all experts,

    min_P  sum_e E_x || (W_e - A_e P) x ||^2
      = min_G tr( (I - Pi_G) Sigma^{1/2} M Sigma^{1/2} (I - Pi_G) ),
        G = P Sigma^{1/2},  M = sum_e W_e^T W_e

whose optimum is ``Pi_G`` = top-r eigenvectors ``Q`` of ``Sigma^{1/2} M Sigma^{1/2}``,
giving

    P = Q^T Sigma^{-1/2}   (r,H) shared,      A_e = W_e Sigma^{1/2} Q   (I,r).

``M = I`` recovers ``pcabasis`` (weight-agnostic); ``Sigma = I`` recovers a
cross-expert weight basis (measured dead in ``idea_pilot_scorers.py``).

**Output-side SVD is not a separate mechanism.** Since ``Sigma_y = W Sigma_x W^T``,
the top-r eigenvectors of ``Y Y^T`` are the left singular vectors ``U_r`` of
``W Sigma_x^{1/2}``, and

    U_r U_r^T W = U_r U_r^T (W Sigma^{1/2}) Sigma^{-1/2} = U_r S_r V_r^T Sigma^{-1/2}

which is exactly the activation-aware input SVD. So "SVD on ``Y Y^T``" and
"activation-aware SVD on ``W``" are the *same operator*; ``--verify-equiv`` checks
this numerically. What is genuinely different on the output side is *whitening*
(``outwhiten``), which changes the objective rather than the side.

Metrics. Recall/mass of the ``oracle_mag_noW`` top-B (comparable to
``btt_dynamic.md`` and ``probe_frontier.py``) **and** block-output ``rel_err``, the
currency that predicts downstream accuracy to ~0.1pt via the fitted ladder in
``docs/exps/dynamic_active_param/sparse_probe.md`` (-24.3 HellaSwag pt per unit
rel_err for a mis-selecting scorer). ``Sigma`` and ``M`` are fit on a **held-out**
token slice, so no statistic is fit on the tokens it is scored on.

Needs the ``_wd`` captures from ``scripts/probe_capture.py``. One GPU, no model load.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts.idea_pilot_scorers import _route, quantize_rtn
from scripts.probe_frontier import rtn_bits_per_weight, topk_mask_input


# ==========================================================================
# second-order statistics
# ==========================================================================

def input_gram(X, dev, chunk=2048):
    """``Sigma_x = E[x x^T]`` ``(H,H)`` in fp64, accumulated in chunks."""
    H = X.shape[-1]
    S = torch.zeros((H, H), dtype=torch.float64, device=dev)
    n = 0
    for s0 in range(0, X.shape[0], chunk):
        x = X[s0:s0 + chunk].to(dev, torch.float64)
        S += x.t() @ x
        n += x.shape[0]
    return S / max(n, 1)


def weight_gram(Ws, dev, chunk=8):
    """``M = mean_e sum_matrices W^T W`` ``(H,H)`` in fp64, chunked over experts.

    The metric that says how much a perturbation of ``x`` moves the *channel
    scores*: ``E_j <w_j, dx>^2 = dx^T M dx / I``.
    """
    H = Ws[0].shape[-1]
    M = torch.zeros((H, H), dtype=torch.float64, device=dev)
    ne = 0
    for W in Ws:
        E = W.shape[0]
        ne += E
        for e0 in range(0, E, chunk):
            Wc = W[e0:e0 + chunk].to(dev, torch.float32)
            M += torch.einsum("eih,eij->hj", Wc, Wc).double()
    return M / max(ne, 1)


def psd_sqrt(S, eps=1e-8, inverse=False):
    """Symmetric PSD (inverse) square root with a relative eigenvalue floor.

    The floor is what keeps ``Sigma^{-1/2}`` usable: hidden-state covariances of an
    LLM are strongly anisotropic and their tail eigenvalues are estimation noise,
    so inverting them unmodified would amplify exactly the directions the data does
    not constrain (the standard SVD-LLM conditioning caveat).
    """
    ev, U = torch.linalg.eigh(S)
    floor = eps * float(ev.max().clamp_min(1e-300))
    ev = ev.clamp_min(floor)
    d = ev.rsqrt() if inverse else ev.sqrt()
    return (U * d.unsqueeze(0)) @ U.t()


def spectrum_report(Sigma, M, ranks):
    """Retained-energy curves vs rank, in both metrics that matter.

    ``plain`` = fraction of ``E||x||^2`` kept by the top-r eigenspace of ``Sigma``.
    ``score`` = fraction of ``E_j <w_j,x>^2`` kept, i.e. the ``M``-weighted energy,
    which is what actually perturbs the ranking. If these two curves differ, the
    weight-aware basis (``actbasis``) is not the same as PCA (``pcabasis``).
    """
    out = {}
    Ssq = psd_sqrt(Sigma)
    ev_pca = torch.linalg.eigvalsh(Sigma).flip(0)
    C = Ssq @ M @ Ssq
    ev_act = torch.linalg.eigvalsh(C).flip(0)
    # score-energy retained by the *PCA* basis (not its own optimum)
    ev, U = torch.linalg.eigh(Sigma)
    order = ev.argsort(descending=True)
    U = U[:, order]
    # diag of U^T Sigma^{1/2} M Sigma^{1/2} U is not the right quantity for a
    # non-optimal basis; compute the true retained score energy per prefix.
    G = U.t() @ Ssq @ M @ Ssq @ U
    pca_score = torch.cumsum(G.diagonal(), 0)          # trace of the prefix block
    tot_plain, tot_score = float(ev_pca.sum()), float(ev_act.sum())
    cp, ca = ev_pca.cumsum(0), ev_act.cumsum(0)
    for r in ranks:
        if r > Sigma.shape[0]:
            continue
        out[str(r)] = {
            "pca_plain_energy": round(float(cp[r - 1]) / tot_plain, 5),
            "pca_score_energy": round(float(pca_score[r - 1]) / tot_score, 5),
            "actbasis_score_energy": round(float(ca[r - 1]) / tot_score, 5),
        }
    out["eff_rank_plain"] = round(tot_plain / float(ev_pca[0]), 2)
    out["eff_rank_score"] = round(tot_score / float(ev_act[0]), 2)
    return out


def input_keep_energy(X, ranks, H, dev, chunk=2048):
    """Score-relevant energy kept by per-token top-|x| coordinate selection.

    Reported at ``keep = r/H`` so it lands on the *same* per-expert cost axis as a
    rank-r sketch: reading r columns of ``W`` costs the same as reading an ``(I,r)``
    factor. This is the apples-to-apples comparison between the adaptive canonical
    subset and the fixed data basis.
    """
    out, tot = {}, 0.0
    kept = {r: 0.0 for r in ranks}
    for s0 in range(0, X.shape[0], chunk):
        x = X[s0:s0 + chunk].to(dev, torch.float32)
        tot += float((x ** 2).sum())
        for r in ranks:
            k = max(1, min(int(r), H))
            kept[r] += float((x.abs().topk(k, -1).values ** 2).sum())
    for r in ranks:
        out[str(r)] = round(kept[r] / max(tot, 1e-30), 5)
    return out


# ==========================================================================
# scorer construction
# ==========================================================================

class Sketch:
    """A cheap scorer: how to sketch ``x``, and the per-expert factors to apply.

    ``kind``:
      ``shared``    ``score ~ A_e (P x)``, ``P`` shared across experts (and matrices).
      ``per_expert`` ``score ~ A_e (R_e x)``, ``R_e`` per expert.
      ``coords``    read ``keep`` fraction of ``x``'s coordinates, exact weights.
      ``exact``     reference: full-width true projections.
      ``rtn``       reference: the measured ``sparse_probe`` point, to anchor the
                    numbers of this script against the published ones.
    """

    def __init__(self, name, kind, cB, meta=None, **kw):
        self.name, self.kind, self.cB = name, kind, float(cB)
        self.meta = dict(meta or {})
        self.__dict__.update(kw)

    def scores(self, x, Wu, Wg, use_gate):
        """``(t, I)`` proxy of ``|SiLU(gate·x)·(up·x)|`` -- per expert, called with
        that expert's slice of the token batch. Subclass-free dispatch keeps the
        cost model and the arithmetic in one place."""
        raise NotImplementedError


def cost_shared(r, K, I, H, n_mat, bpw=16.0):
    """cB for a shared-input-factor sketch, at ``bpw`` bits per stored factor weight.

    Per token the exact path reads ``K`` experts x ``n_mat`` matrices of ``I*H``
    fp16 weights; the repo normalizes cB *per expert*, i.e. divides by ``K`` (which
    is why `oracle_mag` is booked at 2.0, not ``2K``). With a shared input factor:

        P   (r,H)  read ONCE per token   ->  r*H*bpw   / (K*I*H*16)
        A_e (I,r)  per expert, per matrix ->  n_mat*K*I*r*bpw / (K*I*H*16)

    so ``cB = r*bpw/(16*K*I) + n_mat*r*bpw/(16*H)``. At ``bpw=16`` this is
    ``r/(K*I) + n_mat*r/H``. The ``A_e`` term dominates (H < K*I here), so
    quantizing the factors buys rank almost linearly: 3-bit factors fund ~5x the
    rank at the same bytes.
    """
    f = bpw / 16.0
    return f * r / float(K * I) + f * n_mat * r / float(H)


def cost_per_expert(r, I, H, n_mat, bpw=16.0):
    """cB for a per-expert sketch: ``n_mat*r*(H+I)/(I*H)`` (the btt_dynamic model),
    scaled by ``bpw/16`` when the factors are stored at reduced precision."""
    return (bpw / 16.0) * n_mat * r * float(H + I) / float(I * H)


def build_shared_basis(Sigma, M, r, mode, eps):
    """Return ``(P (r,H), pull (H,r))`` so that ``A_e = W_e @ pull``, ``h = P x``.

    ``mode='act'``  optimal shared factor: ``Q`` = top-r eigvecs of
                    ``S^{1/2} M S^{1/2}``, ``P = Q^T S^{-1/2}``, ``pull = S^{1/2} Q``.
    ``mode='pca'``  ``Q`` = top-r eigvecs of ``Sigma``; ``P = Q^T``, ``pull = Q``.
    """
    if mode == "pca":
        ev, U = torch.linalg.eigh(Sigma)
        Q = U[:, ev.argsort(descending=True)[:r]]
        return Q.t().contiguous(), Q.contiguous()
    Ssq = psd_sqrt(Sigma, eps)
    Sinv = psd_sqrt(Sigma, eps, inverse=True)
    ev, U = torch.linalg.eigh(Ssq @ M @ Ssq)
    Q = U[:, ev.argsort(descending=True)[:r]]
    return (Q.t() @ Sinv).contiguous(), (Ssq @ Q).contiguous()


def build_per_expert(W, Sigma, r, mode, eps, chunk=16):
    """Per-expert activation-aware factors ``(A (E,I,r), R (E,r,H))``.

    ``mode='awsvd'``     truncate ``W Sigma^{1/2}`` (SVD-LLM). Equivalently -- and
                         this is the identity that makes the output-side branch
                         redundant -- project ``W`` onto the top-r eigenspace of
                         ``Sigma_y = W Sigma W^T``.
    ``mode='outwhiten'`` truncate ``D^{-1} W Sigma^{1/2}`` with
                         ``D = diag(rms_j(y))``, then fold ``D`` back in: equalizes
                         *relative* error across the I output channels instead of
                         absolute. The distinct output-side lever.
    ``mode='svd'``       plain SVD of ``W`` (the btt_dynamic reference).

    Uses an **exact** batched SVD, not ``svd_lowrank``. Measured on this capture:
    at r=32 the randomized form (``q=r+16, niter=4``, what ``lowrank_scorer.py``
    uses) is 13.8% off the exact rank-r factors of ``W Sigma^{1/2}``, because
    activation-weighting concentrates the spectrum hard (effective rank ~1.7 in the
    score metric). That error is comparable to the truncation being studied, so it
    would understate this whole family. Exact costs ~5s per matrix for E=128.
    """
    E, I, H = W.shape
    dev = Sigma.device
    A = torch.empty((E, I, r), dtype=torch.float32, device=dev)
    R = torch.empty((E, r, H), dtype=torch.float32, device=dev)
    if mode == "svd":
        Ssq = Sinv = None
    else:
        Ssq = psd_sqrt(Sigma, eps).float()
        Sinv = psd_sqrt(Sigma, eps, inverse=True).float()
    for e0 in range(0, E, chunk):
        Wc = W[e0:e0 + chunk].to(dev, torch.float32)
        Bc = Wc if Ssq is None else Wc @ Ssq
        if mode == "outwhiten":
            # rms over the data of channel j's output: || W_j Sigma^{1/2} ||
            d = Bc.norm(dim=2, keepdim=True).clamp_min(1e-12)      # (e,I,1)
            Bc = Bc / d
        U, S, Vh = torch.linalg.svd(Bc, full_matrices=False)
        U, S, Vh = U[..., :r], S[..., :r], Vh[..., :r, :]
        Ae = U * S.unsqueeze(-2)                                    # (e,I,r)
        if mode == "outwhiten":
            Ae = Ae * d
        Re = Vh if Sinv is None else Vh @ Sinv
        A[e0:e0 + chunk], R[e0:e0 + chunk] = Ae, Re
    return A, R


# ==========================================================================
# main screen
# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,22,38,46")
    ap.add_argument("--tokens", type=int, default=8192, help="capture size (filename)")
    ap.add_argument("--fit-tokens", type=int, default=4096,
                    help="leading tokens used ONLY to fit Sigma/M (held out from scoring)")
    ap.add_argument("--score-tokens", type=int, default=2048,
                    help="tokens scored, taken after the fit slice")
    ap.add_argument("--ratios", default="0.25,0.125")
    ap.add_argument("--ranks", default="16,32,64,128,256")
    ap.add_argument("--variants",
                    default="ref,pcabasis,actbasis,awsvd,outwhiten,svd,insp,adapt,mix")
    ap.add_argument("--gate-modes", default="upgate,uponly")
    ap.add_argument("--adapt-mult", type=float, default=4.0,
                    help="dictionary size r' = mult*r for the per-token adaptive sketch")
    ap.add_argument("--mix-split", type=float, default=0.5,
                    help="fraction of the per-expert budget spent on the shared basis "
                         "in the `mix` variant (rest goes to residual coordinates)")
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="relative eigenvalue floor for Sigma^{+-1/2}")
    ap.add_argument("--qgroup", type=int, default=128)
    ap.add_argument("--qbits", default="3,4",
                    help="factor bit-widths for the qbasis/qawsvd variants")
    ap.add_argument("--qranks", default="128,256,448,768",
                    help="ranks for the quantized-factor variants (a 3-bit rank-448 "
                         "shared basis costs the same as fp16 rank-88)")
    ap.add_argument("--verify-equiv", type=int, default=1,
                    help="numerically check output-side PCA == activation-aware SVD")
    ap.add_argument("--capture-dir", default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--out", default=os.path.join(_REPO, "docs/results/actaware/screen.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    layers = [int(v) for v in args.layers.split(",") if v]
    ratios = [float(v) for v in args.ratios.split(",") if v]
    ranks = [int(v) for v in args.ranks.split(",") if v]
    args.qbits = [int(v) for v in args.qbits.split(",") if v]
    args.qranks = [int(v) for v in args.qranks.split(",") if v]
    variants = set(args.variants.split(","))
    gate_modes = [g for g in args.gate_modes.split(",") if g]
    dev = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def log(m):
        print(m, flush=True)

    rows, diags = [], {}
    for layer in layers:
        cap_p = os.path.join(args.capture_dir, f"capture_L{layer}_t{args.tokens}_wd.pt")
        if not os.path.exists(cap_p):
            log(f"[skip] no capture {cap_p}")
            continue
        cap = torch.load(cap_p, map_location="cpu")
        Xall, gate_w = cap["X"], cap["gate_w"]
        Wg_c, Wu_c, Wd_c = cap["Wg"], cap["Wu"], cap["Wd"]
        K, norm_topk = cap["top_k"], cap["norm_topk"]
        E, I, H = Wu_c.shape
        del cap

        Xfit = Xall[:args.fit_tokens]
        X = Xall[args.fit_tokens:args.fit_tokens + args.score_tokens]
        T, KI = X.shape[0], K * I
        budgets = [max(1, min(int(round(r * KI)), KI)) for r in ratios]
        log(f"\n[layer {layer}] E={E} I={I} H={H} K={K} "
            f"fit={Xfit.shape[0]} score={T} B={budgets}")

        # ---- statistics on the held-out fit slice --------------------------
        Sigma = input_gram(Xfit, dev)
        M = weight_gram([Wu_c, Wg_c], dev)
        d = {"spectrum": spectrum_report(Sigma, M, ranks),
             "insp_score_energy_at_equal_cost": input_keep_energy(Xfit, ranks, H, dev)}
        diags[f"L{layer}"] = d
        log(f"  eff_rank plain={d['spectrum']['eff_rank_plain']} "
            f"score={d['spectrum']['eff_rank_score']}")
        for r in ranks:
            s = d["spectrum"].get(str(r))
            if s:
                log(f"  r={r:4d}  actbasis_score_E={s['actbasis_score_energy']:.4f}  "
                    f"pca_score_E={s['pca_score_energy']:.4f}  "
                    f"pca_plain_E={s['pca_plain_energy']:.4f}  "
                    f"insp(keep={r}/{H})_E={d['insp_score_energy_at_equal_cost'][str(r)]:.4f}")

        g, sel = _route(X, gate_w, K, norm_topk, dev)
        Wu = Wu_c.to(dev)
        Wg = Wg_c.to(dev)
        Wd = Wd_c.to(dev)
        del Wu_c, Wg_c, Wd_c

        # ---- output-side identity check -----------------------------------
        if args.verify_equiv:
            # fp64 + exact SVD: the claim is an algebraic identity, so the check
            # must not be limited by the factorizer's own error.
            r0 = min(32, I)
            Ssq = psd_sqrt(Sigma, args.eps)
            Sinv = psd_sqrt(Sigma, args.eps, inverse=True)
            W0 = Wu[0].to(torch.float64)
            B0 = W0 @ Ssq
            U, S, Vh = torch.linalg.svd(B0, full_matrices=False)
            aw = (U[:, :r0] * S[:r0]) @ (Vh[:r0, :] @ Sinv)
            # output-side: project W onto the top-r eigenspace of Sigma_y = W Sigma W^T
            evy, Uy = torch.linalg.eigh(B0 @ B0.t())
            Qy = Uy[:, evy.argsort(descending=True)[:r0]]
            outp = Qy @ (Qy.t() @ W0)
            rel = float((aw - outp).norm() / outp.norm().clamp_min(1e-30))
            diags[f"L{layer}"]["outside_equiv_rel_diff"] = rel
            log(f"  [check] ||awsvd - outside_pca|| / ||.|| = {rel:.2e} "
                f"(0 => output-side PCA is not a separate mechanism)")
            del Ssq, Sinv, W0, B0, U, S, Vh, aw, evy, Uy, Qy, outp

        # ---- oracle + routing bookkeeping, cached once --------------------
        hits_all, oracle = [], torch.empty((T, KI), dtype=torch.float32, device=dev)
        inter_all = []
        y_full_sq = 0.0
        for s0 in range(0, T, args.chunk):
            x = X[s0:s0 + args.chunk].to(dev)
            t = x.shape[0]
            sc_ = sel[s0:s0 + args.chunk].to(dev)
            gc_ = g[s0:s0 + args.chunk].to(dev)
            hits = []
            for e in torch.unique(sc_):
                tok, slot = torch.where(sc_ == int(e))
                hits.append((int(e), slot, tok))
            hits_all.append(hits)
            inter = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
            for e, slot, tok in hits:
                cur = x[tok]
                inter[tok, slot] = (F.silu(cur @ Wg[e].t().float())
                                    * (cur @ Wu[e].t().float()))
            oracle[s0:s0 + t] = (gc_.unsqueeze(-1) * inter.abs()).reshape(t, KI)
            y = torch.zeros((t, H), dtype=torch.float32, device=dev)
            for e, slot, tok in hits:
                y.index_add_(0, tok, (inter[tok, slot] @ Wd[e].t().float())
                             * gc_[tok, slot].unsqueeze(-1))
            y_full_sq += float((y ** 2).sum())
            inter_all.append((inter, y.norm(dim=1).clamp_min(1e-30)))
        o_mask, o_topmass = [], []
        for B in budgets:
            ti = oracle.topk(B, dim=1, sorted=False).indices
            o_mask.append(torch.zeros_like(oracle, dtype=torch.bool).scatter_(1, ti, True))
            o_topmass.append(oracle.gather(1, ti).sum(1).clamp_min(1e-30))
        log(f"  oracle cached ({oracle.numel() * 4 / 2 ** 30:.2f} GiB)")

        # ---- score one sketch ---------------------------------------------
        def run(sk, use_gate):
            hit = np.zeros(len(budgets))
            mass = np.zeros(len(budgets))
            rel = np.zeros(len(budgets))
            for ci, s0 in enumerate(range(0, T, args.chunk)):
                x = X[s0:s0 + args.chunk].to(dev)
                t = x.shape[0]
                gc_ = g[s0:s0 + args.chunk].to(dev)
                inter, ynorm = inter_all[ci]
                up_p = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                gt_p = (torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                        if use_gate else None)
                sk.fill(x, hits_all[ci], up_p, gt_p, Wu, Wg)
                s = ((F.silu(gt_p) * up_p).abs() if use_gate else up_p.abs())
                score = (gc_.unsqueeze(-1) * s).reshape(t, KI)
                del up_p, gt_p, s
                orc, om = oracle[s0:s0 + t], None
                for bi, B in enumerate(budgets):
                    om, tm = o_mask[bi][s0:s0 + t], o_topmass[bi][s0:s0 + t]
                    idx = score.topk(B, dim=1, sorted=False).indices
                    hit[bi] += float((om.gather(1, idx).sum(1).float() / B).sum())
                    mass[bi] += float((orc.gather(1, idx).sum(1) / tm).sum())
                    keep = torch.zeros_like(score, dtype=torch.bool).scatter_(1, idx, True)
                    dropped = inter * (~keep.reshape(t, K, I))
                    y_err = torch.zeros((t, H), dtype=torch.float32, device=dev)
                    for e, slot, tok in hits_all[ci]:
                        y_err.index_add_(0, tok, (dropped[tok, slot] @ Wd[e].t().float())
                                         * gc_[tok, slot].unsqueeze(-1))
                    rel[bi] += float((y_err.norm(dim=1) / ynorm).sum())
                    del keep, dropped, y_err
                del score
            n_mat = 2 if use_gate else 1
            cB = sk.cB if n_mat == 2 else sk.cB_uponly
            row = {"layer": layer, "name": f"{sk.name}_{'upgate' if use_gate else 'uponly'}",
                   "family": sk.name.split("_")[0], "n_matrices": n_mat,
                   "cost_bytes": cB, "n_tokens": T, **sk.meta}
            for bi, r in enumerate(ratios):
                row[f"recall@rho{r}"] = hit[bi] / T
                row[f"mass@rho{r}"] = mass[bi] / T
                row[f"rel_err@rho{r}"] = rel[bi] / T
                row[f"ffn_kept@rho{r}"] = r + cB / 3.0
                row[f"ffn_cut@rho{r}"] = 1.0 - (r + cB / 3.0)
            rows.append(row)
            r0 = ratios[-1]
            log(f"    {row['name']:26s} cB={cB:.4f} cut={100*row[f'ffn_cut@rho{r0}']:5.1f}% "
                f"rec@{r0}={row[f'recall@rho{r0}']:.3f} mass={row[f'mass@rho{r0}']:.3f} "
                f"relerr={row[f'rel_err@rho{r0}']:.4f}")

        # ---- sketch definitions -------------------------------------------
        class Exact(Sketch):
            def fill(self, x, hits, up_p, gt_p, Wu, Wg):
                for e, slot, tok in hits:
                    cur = x[tok]
                    up_p[tok, slot] = cur @ Wu[e].t().float()
                    if gt_p is not None:
                        gt_p[tok, slot] = cur @ Wg[e].t().float()

        class Coords(Sketch):
            def fill(self, x, hits, up_p, gt_p, Wu, Wg):
                xs = topk_mask_input(x, self.keep)
                for e, slot, tok in hits:
                    cur = xs[tok]
                    up_p[tok, slot] = cur @ self.Wu_p[e].t().float()
                    if gt_p is not None:
                        gt_p[tok, slot] = cur @ self.Wg_p[e].t().float()

        class SharedBasis(Sketch):
            def fill(self, x, hits, up_p, gt_p, Wu, Wg):
                h = x.float() @ self.P.t()                       # (t,r) once per token
                for e, slot, tok in hits:
                    hh = h[tok]
                    up_p[tok, slot] = hh @ self.Au[e].t()
                    if gt_p is not None:
                        gt_p[tok, slot] = hh @ self.Ag[e].t()

        class AdaptBasis(Sketch):
            """Per-token top-r of r' shared-basis coefficients: an adaptive subspace
            drawn from a data-optimal dictionary. Cost is r' for the shared
            projection (once per token) but only r per expert per matrix."""

            def fill(self, x, hits, up_p, gt_p, Wu, Wg):
                h = x.float() @ self.P.t()                       # (t,r')
                # rank coefficients by how much they move a typical channel score
                idx = (h.abs() * self.colw).topk(self.r_keep, dim=-1).indices
                hs = torch.zeros_like(h).scatter_(-1, idx, h.gather(-1, idx))
                for e, slot, tok in hits:
                    hh = hs[tok]
                    up_p[tok, slot] = hh @ self.Au[e].t()
                    if gt_p is not None:
                        gt_p[tok, slot] = hh @ self.Ag[e].t()

        class MixBasis(Sketch):
            """Shared rank-r basis **plus** the top-q coordinates of the residual."""

            def fill(self, x, hits, up_p, gt_p, Wu, Wg):
                xf = x.float()
                h = xf @ self.P.t()
                res = xf - h @ self.pull.t()
                k = max(1, self.q)
                ridx = res.abs().topk(k, dim=-1).indices
                rs = torch.zeros_like(res).scatter_(-1, ridx, res.gather(-1, ridx))
                for e, slot, tok in hits:
                    hh, rr = h[tok], rs[tok]
                    up_p[tok, slot] = hh @ self.Au[e].t() + rr @ Wu[e].t().float()
                    if gt_p is not None:
                        gt_p[tok, slot] = hh @ self.Ag[e].t() + rr @ Wg[e].t().float()

        class PerExpert(Sketch):
            def fill(self, x, hits, up_p, gt_p, Wu, Wg):
                xf = x.float()
                for e, slot, tok in hits:
                    cur = xf[tok]
                    up_p[tok, slot] = (cur @ self.Ru[e].t()) @ self.Au[e].t()
                    if gt_p is not None:
                        gt_p[tok, slot] = (cur @ self.Rg[e].t()) @ self.Ag[e].t()

        def refs():
            yield Exact("oracle_magnoW", "exact", 2.0, cB_uponly=1.0)
            qu = quantize_rtn(Wu.float(), 3, args.qgroup)
            qg = quantize_rtn(Wg.float(), 3, args.qgroup)
            cb = 2 * rtn_bits_per_weight(3, args.qgroup) / 16.0 * 0.25
            yield Coords("probe_q3_k25", "rtn", cb, cB_uponly=cb / 2,
                         meta=dict(note="anchor: published 0.675 recall @rho0.125"),
                         keep=0.25, Wu_p=qu, Wg_p=qg)

        def sketches():
            if "ref" in variants:
                yield from refs()
            if "insp" in variants:
                for r in ranks:
                    keep = r / float(H)
                    yield Coords(f"insp_r{r}", "coords", 2.0 * keep, cB_uponly=keep,
                                 meta=dict(rank=r, input_keep=keep),
                                 keep=keep, Wu_p=Wu, Wg_p=Wg)
            for mode, tag in (("pca", "pcabasis"), ("act", "actbasis")):
                if tag not in variants:
                    continue
                for r in ranks:
                    if r > H:
                        continue
                    P, pull = build_shared_basis(Sigma, M, r, mode, args.eps)
                    P, pull = P.float(), pull.float()
                    Au = torch.einsum("eih,hr->eir", Wu.float(), pull)
                    Ag = torch.einsum("eih,hr->eir", Wg.float(), pull)
                    yield SharedBasis(
                        f"{tag}_r{r}", "shared", cost_shared(r, K, I, H, 2),
                        cB_uponly=cost_shared(r, K, I, H, 1),
                        meta=dict(rank=r, shared=True), P=P, Au=Au, Ag=Ag)
                    del P, pull, Au, Ag
                    torch.cuda.empty_cache()
            if "adapt" in variants:
                for r in ranks:
                    rp = min(H, int(round(args.adapt_mult * r)))
                    P, pull = build_shared_basis(Sigma, M, rp, "act", args.eps)
                    P, pull = P.float(), pull.float()
                    Au = torch.einsum("eih,hr->eir", Wu.float(), pull)
                    Ag = torch.einsum("eih,hr->eir", Wg.float(), pull)
                    # weight each coefficient by the RMS channel response it drives,
                    # so "largest coefficient" means "moves the scores most"
                    colw = (Au.pow(2).mean(dim=(0, 1)) + Ag.pow(2).mean(dim=(0, 1))).sqrt()
                    yield AdaptBasis(
                        f"adapt_r{r}x{rp}", "shared", cost_shared(rp, K, I, H, 0) +
                        2 * r / float(H),
                        cB_uponly=cost_shared(rp, K, I, H, 0) + r / float(H),
                        meta=dict(rank=r, dict_rank=rp, adaptive=True),
                        P=P, Au=Au, Ag=Ag, r_keep=r, colw=colw)
                    del P, pull, Au, Ag, colw
                    torch.cuda.empty_cache()
            if "mix" in variants:
                for r in ranks:
                    rb = max(1, int(round(args.mix_split * r)))
                    q = max(1, r - rb)
                    P, pull = build_shared_basis(Sigma, M, rb, "act", args.eps)
                    P, pull = P.float(), pull.float()
                    Au = torch.einsum("eih,hr->eir", Wu.float(), pull)
                    Ag = torch.einsum("eih,hr->eir", Wg.float(), pull)
                    yield MixBasis(
                        f"mix_r{rb}q{q}", "shared",
                        cost_shared(rb, K, I, H, 0) + 2 * (rb + q) / float(H),
                        cB_uponly=cost_shared(rb, K, I, H, 0) + (rb + q) / float(H),
                        meta=dict(rank=rb, resid_coords=q, total_r=r),
                        P=P, pull=pull, Au=Au, Ag=Ag, q=q)
                    del P, pull, Au, Ag
                    torch.cuda.empty_cache()
            for mode, tag in (("awsvd", "awsvd"), ("outwhiten", "outwhiten"),
                              ("svd", "svd")):
                if tag not in variants:
                    continue
                for r in ranks:
                    if r > min(I, H):
                        continue
                    Au, Ru = build_per_expert(Wu, Sigma, r, mode, args.eps)
                    Ag, Rg = build_per_expert(Wg, Sigma, r, mode, args.eps)
                    yield PerExpert(
                        f"{tag}_r{r}", "per_expert", cost_per_expert(r, I, H, 2),
                        cB_uponly=cost_per_expert(r, I, H, 1),
                        meta=dict(rank=r, shared=False),
                        Au=Au, Ru=Ru, Ag=Ag, Rg=Rg)
                    del Au, Ru, Ag, Rg
                    torch.cuda.empty_cache()
            # ---- the two mechanisms composed: reduced-precision factors ------
            # Rank is bought at bpw/16 of the price, so a 3-bit rank-448 shared
            # basis fits the same 0.10 budget as an fp16 rank-88 one. If low rank
            # were merely *underfunded*, this is where it would win; if the
            # limitation is structural (the basis cannot express the per-token
            # part of the signal at all), extra rank changes nothing.
            if "qbasis" in variants:
                for bits in args.qbits:
                    bpw = rtn_bits_per_weight(bits, args.qgroup)
                    for r in args.qranks:
                        if r > H:
                            continue
                        P, pull = build_shared_basis(Sigma, M, r, "act", args.eps)
                        P, pull = P.float(), pull.float()
                        Au = torch.einsum("eih,hr->eir", Wu.float(), pull)
                        Ag = torch.einsum("eih,hr->eir", Wg.float(), pull)
                        # quantize along the rank axis (the axis that is contracted
                        # at run time), matching how RTN groups the input dim of W
                        Au = quantize_rtn(Au, bits, min(args.qgroup, r))
                        Ag = quantize_rtn(Ag, bits, min(args.qgroup, r))
                        yield SharedBasis(
                            f"qbasis_b{bits}_r{r}", "shared",
                            cost_shared(r, K, I, H, 2, bpw),
                            cB_uponly=cost_shared(r, K, I, H, 1, bpw),
                            meta=dict(rank=r, shared=True, factor_bits=bits),
                            P=P, Au=Au, Ag=Ag)
                        del P, pull, Au, Ag
                        torch.cuda.empty_cache()
            if "qawsvd" in variants:
                for bits in args.qbits:
                    bpw = rtn_bits_per_weight(bits, args.qgroup)
                    for r in args.qranks:
                        if r > min(I, H):
                            continue
                        Au, Ru = build_per_expert(Wu, Sigma, r, "awsvd", args.eps)
                        Ag, Rg = build_per_expert(Wg, Sigma, r, "awsvd", args.eps)
                        Au = quantize_rtn(Au, bits, min(args.qgroup, r))
                        Ag = quantize_rtn(Ag, bits, min(args.qgroup, r))
                        yield PerExpert(
                            f"qawsvd_b{bits}_r{r}", "per_expert",
                            cost_per_expert(r, I, H, 2, bpw),
                            cB_uponly=cost_per_expert(r, I, H, 1, bpw),
                            meta=dict(rank=r, shared=False, factor_bits=bits),
                            Au=Au, Ru=Ru, Ag=Ag, Rg=Rg)
                        del Au, Ru, Ag, Rg
                        torch.cuda.empty_cache()

        for sk in sketches():
            for gm in gate_modes:
                if gm == "uponly" and sk.name.startswith("oracle"):
                    sk.name = "oracle_up"          # exact up-only == oracle_up
                run(sk, gm == "upgate")
            del sk
            torch.cuda.empty_cache()

        del oracle, o_mask, o_topmass, inter_all, hits_all, Wu, Wg, Wd, Sigma, M
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "ratios": ratios, "diagnostics": diags,
                   "args": {k: str(v) for k, v in vars(args).items()}}, f, indent=2)
    log(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
