#!/usr/bin/env python
"""Rotated-basis channel screen (lm_head's S1, ported to expert FFNs): block ``rel_err``.

`input_sparse` spends its scoring budget on the token's top-``rho_input``
coordinates **of x**. lm_head's S1 reached dense accuracy at a quarter of the reads
by spending the same budget in a *rotated* basis: rank the coordinates of
``z = Uᵀx`` (``U`` = eigenvectors of ``C = E[x xᵀ]``), keep the top ``r0``, and
screen with ``x̃ = U_S U_Sᵀ x``. This script measures what that does to the MoE
block output error, on the instrument whose published anchors are `input_sparse`
uniform **0.4099** / router **0.3860** at ``rho_input=0.25, rho_channel=0.125`` and
`oracle_mag_noW` **0.3272** (4 layers × 4096 C4 tokens, mean per-token
``||y_full − y_kept|| / ||y_full||``).

Simulation, not deployment: screening with a *projected* hidden state against the
unrotated ``W`` is the identical arithmetic to reading ``r0`` columns of the rotated
``W U`` (``W U_S U_Sᵀ x == W x̃``) — which is what a kernel would touch. Same
convention as ``src/lm_head/screen_refine.py``.

**Three deployable forms, three cost models.** The rotation is only free if the
basis is shared, so the accounting splits (units of one expert ``(I,H)`` matrix per
token, divided by the token's ``K`` experts — a dense expert FFN is 3):

    scoring   2·r0/(3H)          r0 columns of both branches, all I rows
    compute   rho_channel        all three matrices gathered to the kept channels
    rotation  r'/(3·K·I)         H·r' reads of U_{r'}, shared by the K experts

    form                    basis          rotation      extra storage
    rotglob   (free)        one global U    0             0          <- residual rebased offline
    rot_lr    (low-rank U)  per-layer U     r'/(3KI)      2r'/(3H)   <- separate probe W·U_{r'}
    rot_full  (full U)      per-layer U     H/(3KI)=.111  0          <- served weights rebased

``rotglob`` costs nothing because rotating the residual stream once folds ``U``
into every weight that reads or writes it (QuaRot's machinery; RMSNorm commutes
with an orthogonal rotation once ``gamma`` is folded), so ``z`` *is* the residual
state — but then one basis must serve all 48 layers. ``rot_full`` keeps the
per-layer optimum and must materialize ``z = U_ℓᵀ x``, an ``H²`` matvec per layer.
``rot_lr`` stores a separate rank-``r'`` probe, so it pays storage instead.

**Basis sources.** ``C`` is a 2048×2048 matrix and a capture holds 8192 tokens, so
the sample eigenbasis overfits badly (``scripts/probe_rotate_diag.py``: an in-sample
energy gain of +0.21 shrinks to +0.02 held out). Basis quality is therefore a
confound and every rotated arm is run against several:

    heldout   U fit on the 4096 capture tokens not scored     <- noisy, T/H = 2
    calib     U fit on ``--cov`` (probe_cov_collect.py)       <- achievable, T/H ~ 77
    oracle    U fit on the scored tokens themselves           <- unachievable bound
    glob*     one basis pooled over all layers                <- the free form

If an arm loses with the *oracle* basis it cannot be rescued by more calibration,
which is what makes a negative verdict here final rather than provisional; if it
wins only there, the gap to ``calib`` is what calibration would have to close.

Needs the ``_wd`` captures from ``probe_capture.py``. One GPU, or CPU (slower).
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

from src.dynamic_active_param.sparse_probe import (
    allocate_input_reads,
    descending_abs_ranks,
)

# fixed-budget ladder slopes, pt per unit rel_err: HellaSwag -26.4 (R^2=0.985),
# MMLU -20.3. Validated for changes that alter *which* channels a layer selects.
SLOPE_HS, SLOPE_MMLU = -26.4, -20.3
H_MODEL, I_MODEL, K_MODEL = 2048, 768, 8

# (basis_src, r_prime, r0, alloc). r_prime None = raw basis; r_prime H = full U.
# r0 == r_prime means no per-token choice within the subspace (the static control).
DEFAULT_ARMS = [
    # anchors that pin the instrument against the published table
    ("raw", None, 2048, "uniform"),      # == oracle_mag_noW, 0.3272
    ("raw", None, 512, "uniform"),       # == input_sparse rho_input=0.25, 0.4099
    ("raw", None, 512, "router"),        # == input_sparse + router,        0.3860
    # the -75% baseline to beat: rho_input=0.1875 -> r0=384, rho_channel=0.125
    ("raw", None, 384, "router"),
    ("raw", None, 192, "router"),
    ("raw", None, 43, "router"),
    # full per-layer U at the baseline's read count: the mechanism test, unbilled
    ("heldout", 2048, 384, "router"),
    ("calib", 2048, 384, "router"),
    ("oracle", 2048, 384, "router"),
    ("heldout", 2048, 192, "router"),
    ("calib", 2048, 192, "router"),
    ("oracle", 2048, 192, "router"),
    # ... and at the read count it can actually afford once the H^2 rotation is
    # billed at used=0.25 with rho_channel=0.125 (2*r0/3H = 0.0139 -> r0=43)
    ("heldout", 2048, 43, "router"),
    ("calib", 2048, 43, "router"),
    ("oracle", 2048, 43, "router"),
    # low-rank U: iso-cost at used=0.25 solves 6*r0 + r' = 2304 (rho_channel=.125)
    ("heldout", 1152, 192, "router"),
    ("calib", 1152, 192, "router"),
    ("oracle", 1152, 192, "router"),
    ("calib", 768, 256, "router"),
    ("heldout", 510, 299, "router"),
    ("calib", 510, 299, "router"),
    ("oracle", 510, 299, "router"),
    ("calib", 384, 384, "router"),       # static rank-384 = the dead low-rank family
    # global basis: the only rotated form whose rotation is genuinely free
    ("globheldout", 2048, 384, "router"),
    ("globcalib", 2048, 384, "router"),
    ("globcalib", 2048, 512, "router"),
    ("globoracle", 2048, 384, "router"),
]


def arm_form(basis_src, r_prime):
    if basis_src == "raw":
        return "raw"
    if basis_src.startswith("glob"):
        return "rotglob"
    return "rot_full" if r_prime >= H_MODEL else "rot_lr"


def rotation_cost(form, r_prime):
    """Reads of ``U`` per token, in used-parameter units (0 when the basis is shared)."""
    if form in ("raw", "rotglob"):
        return 0.0
    return float(r_prime) / (3.0 * K_MODEL * I_MODEL)


def extra_storage(form, r_prime):
    """Extra *stored* parameters as a fraction of the expert FFN's three matrices."""
    return 2.0 * float(r_prime) / (3.0 * H_MODEL) if form == "rot_lr" else 0.0


def used_params(rho_channel, r0, form, r_prime, bill_rotation=True):
    used = float(rho_channel) + 2.0 * r0 / (3.0 * H_MODEL)
    return used + (rotation_cost(form, r_prime) if bill_rotation else 0.0)


def read_set(x, basis, r0, weights, beta, g):
    """The token's screening input, projected: shared over K slots, or per slot.

    Args:
        x: ``(t, H)`` hidden states.
        basis: ``(H, r')`` orthonormal columns, or ``None`` for the raw basis.
        r0: coordinates read per branch, per expert-equivalent.
        weights: ``(r',)`` ranking weight per coordinate (column norms), or None.
        beta: 0 uniform | 1 router — how the pooled read budget splits over slots.
        g: ``(t, K)`` routing weights, used when ``beta > 0``.

    Returns ``("shared", (t,H))`` or ``("perslot", (t,K,H))``, already projected —
    the caller multiplies by the *unrotated* weights (see module docstring).
    """
    z = x if basis is None else x @ basis
    score = z.abs() if weights is None else z.abs() * weights
    r = score.shape[-1]
    k = min(r0, r)

    if beta == 0:
        keep = score.topk(k, dim=-1).indices
        zs = torch.zeros_like(z).scatter_(-1, keep, z.gather(-1, keep))
        return "shared", (zs if basis is None else zs @ basis.t())

    ranks, sorted_score = descending_abs_ranks(score)
    n_e = allocate_input_reads(sorted_score, g, k / r, float(beta))
    out = [((z * (ranks < n_e[:, j].unsqueeze(-1)).to(z.dtype)) if basis is None
            else (z * (ranks < n_e[:, j].unsqueeze(-1)).to(z.dtype)) @ basis.t())
           for j in range(g.shape[1])]
    return "perslot", torch.stack(out, dim=1)


def eigenbasis_from_C(C):
    """Descending-eigenvalue orthonormal basis of a second-moment matrix."""
    _, U = torch.linalg.eigh(C.double())
    return torch.flip(U, dims=[1]).contiguous().float()


def eigenbasis(X):
    """Descending-eigenvalue orthonormal basis of ``E[x xᵀ]`` for ``(T,H)`` X."""
    Y = X.double()
    C = Y.t() @ Y / max(1, Y.shape[0])
    return eigenbasis_from_C(C), C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,22,38,46")
    ap.add_argument("--tokens", type=int, default=8192, help="capture size")
    ap.add_argument("--eval-tokens", type=int, default=4096,
                    help="tokens scored; the published screens use 4096")
    ap.add_argument("--ratios", default="0.10,0.125,0.15,0.20")
    ap.add_argument("--arms", default="", help="override: basis:r':r0:alloc,...")
    ap.add_argument("--colnorm", action="store_true",
                    help="rank coordinates by |z_i|*rms||W u_i|| (lm_head gate 0g)")
    ap.add_argument("--capture-dir",
                    default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--cov", default=os.path.join(
        _REPO, "docs/results/btt_dynamic/moe_input_cov_t262144.pt"),
        help="probe_cov_collect.py output; supplies the 'calib'/'globcalib' bases")
    ap.add_argument("--out", default=os.path.join(
        _REPO, "docs/results/idea_pilot/rotate_screen.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    ratios = [float(x) for x in args.ratios.split(",")]
    arms_spec = DEFAULT_ARMS if not args.arms else [
        (s.split(":")[0],
         None if s.split(":")[1] in ("-", "None") else int(s.split(":")[1]),
         int(s.split(":")[2]), s.split(":")[3])
        for s in args.arms.split(",") if s]
    dev = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 8)

    def cap_path(L):
        return os.path.join(args.capture_dir, f"capture_L{L}_t{args.tokens}_wd.pt")

    layers = [L for L in layers if os.path.exists(cap_path(L))]
    if not layers:
        print("[abort] no captures found")
        return
    n_ev = args.eval_tokens

    # ---- pass 1: the global bases (pooled over layers), one per fit ----------
    Cg = {"globheldout": None, "globoracle": None}
    for L in layers:
        X = torch.load(cap_path(L), map_location="cpu", weights_only=False)["X"]
        for tag, sl in (("globoracle", slice(0, n_ev)), ("globheldout", slice(n_ev, None))):
            _, C = eigenbasis(X[sl])
            Cn = C / C.diagonal().sum()        # depth-normalize before pooling
            Cg[tag] = Cn if Cg[tag] is None else Cg[tag] + Cn
        del X
    C_calib, n_calib = None, 0
    if os.path.exists(args.cov):
        cov = torch.load(args.cov, map_location="cpu", weights_only=False)
        C_calib, n_calib = cov["C"], cov["n_tokens"]
        # the deployable global basis pools *every* MoE layer, not just the screened
        # ones: rotating the residual stream is one decision for the whole model
        acc = None
        for L, C in C_calib.items():
            Cn = C.double() / C.double().diagonal().sum()
            acc = Cn if acc is None else acc + Cn
        Cg["globcalib"] = acc
        print(f"[bases] calib C from {args.cov}: {n_calib} tokens, "
              f"{len(C_calib)} layers (T/H = {n_calib / H_MODEL:.0f})", flush=True)
    U_glob = {tag: eigenbasis_from_C(C) for tag, C in Cg.items() if C is not None}
    print(f"[bases] oracle fit on X[:{n_ev}] (the scored tokens), heldout on "
          f"X[{n_ev}:{args.tokens}]; global bases pooled over "
          f"{len(C_calib) if C_calib else len(layers)} layers", flush=True)

    rows = []
    for L in layers:
        cap = torch.load(cap_path(L), map_location="cpu", weights_only=False)
        X, gate_w = cap["X"][:n_ev], cap["gate_w"]
        Wu, Wg, Wd = cap["Wu"], cap["Wg"], cap["Wd"]
        K, norm_topk = cap["top_k"], cap["norm_topk"]
        E, I, H = Wu.shape
        T = X.shape[0]
        del cap

        U = {"oracle": eigenbasis(X)[0].to(dev),
             "heldout": eigenbasis(torch.load(
                 cap_path(L), map_location="cpu",
                 weights_only=False)["X"][n_ev:])[0].to(dev),
             **{t: U_glob[t].to(dev) for t in U_glob}}
        if C_calib is not None and L in C_calib:
            U["calib"] = eigenbasis_from_C(C_calib[L]).to(dev)
        Wu_d, Wg_d, Wd_d = Wu.to(dev).float(), Wg.to(dev).float(), Wd.to(dev).float()
        del Wu, Wg, Wd

        colnorm = {}
        if args.colnorm:
            for tag in ("raw",) + tuple(U):
                B = None if tag == "raw" else U[tag]
                acc = torch.zeros(H, device=dev)
                for W in (Wu_d, Wg_d):
                    for e in range(E):
                        A = W[e] if B is None else W[e] @ B
                        acc += (A * A).mean(dim=0)
                colnorm[tag] = (acc / (2 * E)).sqrt()

        logits = X.to(dev) @ gate_w.to(dev).float().t()
        w = F.softmax(logits, dim=-1, dtype=torch.float32)
        g, sel = torch.topk(w, K, dim=-1)
        if norm_topk:
            g = g / g.sum(dim=-1, keepdim=True)
        del logits, w

        budgets = [max(1, min(int(round(r * K * I)), K * I)) for r in ratios]
        print(f"\n[layer {L}] T={T} E={E} I={I} H={H} K={K} B={budgets}", flush=True)

        arms = []
        for src, rp, r0, alloc in arms_spec:
            form = arm_form(src, rp if rp else 0)
            arms.append(dict(
                name=f"{src}_p{rp or 0}_r{r0}_{alloc}", src=src, form=form,
                r_prime=rp, r0=r0, alloc=alloc,
                beta=0 if alloc == "uniform" else 1,
                basis=None if src == "raw" else U[src][:, :rp].contiguous(),
                weights=(colnorm[src][:(rp or H)] if args.colnorm else None)))
        err = {a["name"]: np.zeros(len(ratios)) for a in arms}
        seen = 0

        for s0 in range(0, T, args.chunk):
            x = X[s0:s0 + args.chunk].to(dev).float()
            t = x.shape[0]
            gc_, sc_ = g[s0:s0 + args.chunk], sel[s0:s0 + args.chunk]
            hits = []
            for e in torch.unique(sc_):
                tok, slot = torch.where(sc_ == int(e))
                hits.append((int(e), slot, tok))

            inter = torch.zeros((t, K, I), device=dev)
            for e, slot, tok in hits:
                cur = x[tok]
                inter[tok, slot] = F.silu(cur @ Wg_d[e].t()) * (cur @ Wu_d[e].t())
            y_full = torch.zeros((t, H), device=dev)
            for e, slot, tok in hits:
                y_full.index_add_(0, tok, (inter[tok, slot] @ Wd_d[e].t())
                                  * gc_[tok, slot].unsqueeze(-1))
            y_norm = y_full.norm(dim=1).clamp_min(1e-30)

            for a in arms:
                mode, xt = read_set(x, a["basis"], a["r0"], a["weights"],
                                    a["beta"], gc_)
                score = torch.zeros((t, K, I), device=dev)
                for e, slot, tok in hits:
                    cur = xt[tok] if mode == "shared" else xt[tok, slot]
                    score[tok, slot] = (F.silu(cur @ Wg_d[e].t())
                                        * (cur @ Wu_d[e].t())).abs()
                flat = (score * gc_.unsqueeze(-1)).reshape(t, K * I)
                for bi, B in enumerate(budgets):
                    idx = flat.topk(B, dim=1, sorted=False).indices
                    m = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, idx, True)
                    dropped = inter * (~m.reshape(t, K, I))
                    y_err = torch.zeros((t, H), device=dev)
                    for e, slot, tok in hits:
                        y_err.index_add_(0, tok, (dropped[tok, slot] @ Wd_d[e].t())
                                         * gc_[tok, slot].unsqueeze(-1))
                    err[a["name"]][bi] += float((y_err.norm(dim=1) / y_norm).sum())
                del score, flat, xt
            seen += t
            del inter, y_full

        for a in arms:
            row = {"layer": L, "n_tokens": seen,
                   **{k: a[k] for k in ("name", "src", "form", "alloc", "r0", "r_prime")},
                   "colnorm": bool(args.colnorm)}
            for bi, r in enumerate(ratios):
                row[f"rel_err@rho{r}"] = err[a["name"]][bi] / seen
            rows.append(row)
            print(f"  {a['name']:26s} " + "  ".join(
                f"rho{r}: {row[f'rel_err@rho{r}']:.4f}" for r in ratios), flush=True)

        del Wu_d, Wg_d, Wd_d, U, X, g, sel
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # ---- layer-averaged summary, both cost models ----------------------------
    summary = []
    for nm in dict.fromkeys(r["name"] for r in rows):
        rs = [r for r in rows if r["name"] == nm]
        a = rs[0]
        rp, form = a["r_prime"] or 0, a["form"]
        summary.append({
            **{k: a[k] for k in ("name", "src", "form", "alloc", "r0")},
            "r_prime": rp, "n_layers": len(rs),
            "rel_err": {f"{r}": float(np.mean([x[f"rel_err@rho{r}"] for x in rs]))
                        for r in ratios},
            "used_billed": {f"{r}": used_params(r, a["r0"], form, rp, True)
                            for r in ratios},
            "used_free": {f"{r}": used_params(r, a["r0"], form, rp, False)
                          for r in ratios},
            "rotation_cost": rotation_cost(form, rp),
            "extra_storage": extra_storage(form, rp),
        })

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "summary": summary, "ratios": ratios,
                   "eval_tokens": n_ev, "slope_hs": SLOPE_HS,
                   "slope_mmlu": SLOPE_MMLU}, f, indent=2)
    print(f"\n[done] wrote {args.out}")

    print(f"\n{'arm':26s} {'form':9s} {'r0':>5s} {'r´':>5s} {'rot$':>6s} {'stor':>6s}  "
          + "  ".join(f"{'rel@' + str(r):>9s}" for r in ratios))
    for s in sorted(summary, key=lambda s: s["rel_err"][str(ratios[0])]):
        print(f"{s['name']:26s} {s['form']:9s} {s['r0']:5d} {s['r_prime']:5d} "
              f"{s['rotation_cost']:6.4f} {100 * s['extra_storage']:5.1f}%  "
              + "  ".join(f"{s['rel_err'][str(r)]:9.4f}" for r in ratios))


if __name__ == "__main__":
    main()
