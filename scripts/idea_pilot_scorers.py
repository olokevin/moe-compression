#!/usr/bin/env python
"""Idea-discovery pilot: which *cheap* signal can rank expert channels like `oracle_mag`?

Context. `oracle_mag` (rank the pooled `K*I` channels of a token's active experts
by ``g_e*|inter_{e,j}(x)|``, keep the global top-B) is near-lossless at a 7/8
channel cut, but needs full-width `gate_proj`+`up_proj` just to decide. The
low-rank/BTT proxy family (``lowrank_scorer.py``, ``docs/exps/dynamic_active_param/
btt_dynamic.md``) was measured to be a dead end: recall 0.44 at ρ=0.25 for a cheap
scorer, and ~0.66 only at a cost (0.92) that erases the accounting win.

Diagnosis from that work: truncating the *spectrum of one expert's* ``(I,H)``
weight destroys exactly the fine row structure that decides the top-B. This
script tests three mechanisms that keep each expert's proxy rows **full-rank**
and get their cheapness from somewhere other than rank:

  A. ``basis_m{m}`` — **cross-expert shared basis.** Rank-m truncation across the
     *expert* axis: stack ``W (E, I*H)`` and keep m components, so
     ``W~^(e) = sum_b C[e,b] * B_b`` with each ``B_b`` a full ``(I,H)`` matrix.
     Every expert's proxy is full-rank in ``(I,H)``; the saving comes from
     computing ``B_b x`` **once per layer** and sharing it across all K
     co-activated experts. Cost = ``m/K`` full matmuls (vs K for the exact path)
     — a saving that only exists in an MoE.

  B. ``quant_w{b}`` — **low-precision, full-rank proxy.** Group-wise RTN of
     ``W_up``/``W_gate`` to b bits. Preserves row directions up to a small
     relative perturbation. Byte cost b/16; FLOP cost unchanged (reported
     separately — in memory-bound MoE decode bytes are the binding resource).
     Optionally composed with **input sparsity** (``insp{s1}``): keep only the
     largest-magnitude entries of x (Prox, arXiv:2607.27591, Stage 1).

  C. cascade metrics — **relaxed-budget candidate set + exact verify.** Any proxy
     is used to nominate ``lam*B`` candidates; the exact up/gate are then computed
     *only on those*, and the final top-B is taken by exact score. The final set
     then contains precisely the oracle-top-B members that landed in the candidate
     set (an oracle-top-B channel always outranks a non-member inside the set), so
     ``casc_recall == cover@lam``, and ``casc_mass`` is what accuracy tracks.
     Reported for every variant at lam in ``--lams``.

  D. ``prevtok`` — free candidate generator: score token t by token t-1's realized
     oracle scores (mapped through the global ``(E,I)`` space, since the two tokens
     route to different experts). An **upper bound** on adjacent-token reuse: real
     decode would only retain the previously *kept* channels' values.

Reuses the cached captures written by ``scripts/lowrank_scorer_recall.py``
(``capture_L{layer}_t{tokens}.pt``), so it needs one GPU and no model load.
Metrics (recall/mass/spearman at budget) are computed exactly as that script does,
so numbers are directly comparable to the tables in ``btt_dynamic.md``.
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


# --------------------------------------------------------------------------
# mechanism A — cross-expert shared basis
# --------------------------------------------------------------------------

def fit_expert_basis(W, m):
    """Rank-m truncation across the **expert** axis of ``W (E, I, H)``.

    Returns ``(coef (E,m), bases (m,I,H), energy)`` such that
    ``W[e] ~= sum_b coef[e,b] * bases[b]``, bases orthonormal (as flat vectors).

    Done via the ``E x E`` Gram matrix, so the big ``(E, I*H)`` matrix is touched
    only twice — no ``(I*H, I*H)`` object is ever formed.
    """
    E = W.shape[0]
    A = W.reshape(E, -1)                               # (E, I*H)
    gram = A @ A.t()                                   # (E, E)
    evals, evecs = torch.linalg.eigh(gram.double())
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order][:m], evecs[:, order][:, :m]
    evals = evals.clamp_min(1e-30)
    s = evals.sqrt()                                   # singular values
    U = evecs.to(A.dtype)                              # (E, m)
    bases = (U.t() @ A) / s.to(A.dtype).unsqueeze(-1)  # (m, I*H), orthonormal rows
    coef = U * s.to(A.dtype).unsqueeze(0)              # (E, m)
    total = gram.diagonal().sum().double().clamp_min(1e-30)
    energy = float((evals.sum() / total).clamp(0, 1))
    return coef, bases.reshape(m, *W.shape[1:]), energy


# --------------------------------------------------------------------------
# mechanism B — group-wise RTN quantization (full rank)
# --------------------------------------------------------------------------

def quantize_rtn(W, bits, group=128):
    """Symmetric round-to-nearest quantization of ``W (E, I, H)`` along H.

    Group-wise (default 128 along the input dim), which is the standard recipe
    and what makes 2-3 bit weights usable at all. Returns the dequantized
    tensor — the pilot only needs the *numerics* of the proxy, not packed storage.
    """
    E, I, H = W.shape
    g = group if group and H % group == 0 else H
    Wg_ = W.reshape(E, I, H // g, g)
    qmax = 2 ** (bits - 1) - 1
    scale = Wg_.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    q = torch.clamp(torch.round(Wg_ / scale), -qmax - 1, qmax)
    return (q * scale).reshape(E, I, H)


def sparsify_input(x, keep_frac):
    """Keep the ``keep_frac`` largest-magnitude entries of each token's x, zero rest."""
    if keep_frac >= 1.0:
        return x
    k = max(1, int(round(keep_frac * x.shape[-1])))
    idx = x.abs().topk(k, dim=-1).indices
    out = torch.zeros_like(x)
    return out.scatter_(-1, idx, x.gather(-1, idx))


# --------------------------------------------------------------------------
# shared helpers (mirrored from lowrank_scorer_recall.py)
# --------------------------------------------------------------------------

def _route(X, gate_w, K, norm_topk, dev, chunk=4096):
    gs, ss = [], []
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk].to(dev)
        probs = F.softmax(x @ gate_w.to(dev).t(), dim=1, dtype=torch.float32)
        g, sel = torch.topk(probs, K, dim=-1)
        if norm_topk:
            g = g / g.sum(dim=-1, keepdim=True)
        gs.append(g.cpu())
        ss.append(sel.cpu())
    return torch.cat(gs, 0), torch.cat(ss, 0)


def _spearman(a, b):
    ra = a.argsort(dim=1).argsort(dim=1).float()
    rb = b.argsort(dim=1).argsort(dim=1).float()
    ra = ra - ra.mean(dim=1, keepdim=True)
    rb = rb - rb.mean(dim=1, keepdim=True)
    num = (ra * rb).sum(dim=1)
    den = ra.norm(dim=1) * rb.norm(dim=1)
    return (num / den.clamp_min(1e-12)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,22,38,46")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--ratios", default="0.5,0.25,0.125")
    ap.add_argument("--lams", default="1.25,1.5,2.0",
                    help="candidate-pool multipliers for the cascade metric")
    ap.add_argument("--basis-m", default="2,4,8,16",
                    help="number of shared cross-expert bases")
    ap.add_argument("--bits", default="8,4,3,2", help="proxy weight bit-widths")
    ap.add_argument("--qgroup", type=int, default=128)
    ap.add_argument("--insp", default="0.5,0.25",
                    help="input keep-fractions to compose with the 4-bit proxy")
    ap.add_argument("--capture-dir",
                    default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--out-dir",
                    default=os.path.join(_REPO, "docs/results/idea_pilot"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=1024)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    layers = [int(x) for x in args.layers.split(",")]
    ratios = [float(x) for x in args.ratios.split(",")]
    lams = [float(x) for x in args.lams.split(",")]
    basis_ms = [int(x) for x in args.basis_m.split(",") if x]
    bits_list = [int(x) for x in args.bits.split(",") if x]
    insp_list = [float(x) for x in args.insp.split(",") if x]
    dev = torch.device(args.device)

    all_rows = []
    for layer in layers:
        cap_path = os.path.join(args.capture_dir, f"capture_L{layer}_t{args.tokens}.pt")
        if not os.path.exists(cap_path):
            print(f"[skip] no capture {cap_path}", flush=True)
            continue
        cap = torch.load(cap_path, map_location="cpu")
        X, gate_w, Wg, Wu = cap["X"], cap["gate_w"], cap["Wg"], cap["Wu"]
        K, norm_topk = cap["top_k"], cap["norm_topk"]
        E, I, H = Wu.shape
        T, KI = X.shape[0], K * I
        budgets = [max(1, min(int(round(r * KI)), KI)) for r in ratios]
        print(f"\n[layer {layer}] T={T} E={E} I={I} H={H} K={K} B={budgets}", flush=True)

        g, sel = _route(X, gate_w, K, norm_topk, dev)
        Wu_d, Wg_d = Wu.to(dev), Wg.to(dev)

        # ---- build the proxy weight sets ---------------------------------
        # Each entry: name -> (up_weights_or_basis, gate_..., kind, cost_flops,
        #                     cost_bytes, use_gate, input_keep)
        proxies = {}
        meta = {}

        for m in basis_ms:
            cu, bu, eu = fit_expert_basis(Wu_d, m)
            cg, bg, eg = fit_expert_basis(Wg_d, m)
            # m shared basis-matmuls serve all K active experts -> m/K per matrix.
            c = m / K
            proxies[f"basis_m{m}"] = ("basis", (cu, bu), (cg, bg), 1.0)
            meta[f"basis_m{m}"] = dict(kind="basis", m=m, cost_flops=2 * c,
                                       cost_bytes=2 * c, use_gate=True,
                                       energy_up=eu, energy_gate=eg)
            proxies[f"basis_m{m}_uponly"] = ("basis", (cu, bu), None, 1.0)
            meta[f"basis_m{m}_uponly"] = dict(kind="basis", m=m, cost_flops=c,
                                              cost_bytes=c, use_gate=False,
                                              energy_up=eu, energy_gate=None)
            print(f"[layer {layer}] basis m={m}: energy up={eu:.4f} gate={eg:.4f}",
                  flush=True)

        for b in bits_list:
            qu = quantize_rtn(Wu_d, b, args.qgroup)
            qg = quantize_rtn(Wg_d, b, args.qgroup)
            cb = b / 16.0
            proxies[f"quant_w{b}"] = ("dense", qu, qg, 1.0)
            meta[f"quant_w{b}"] = dict(kind="quant", bits=b, cost_flops=2.0,
                                       cost_bytes=2 * cb, use_gate=True)
            proxies[f"quant_w{b}_uponly"] = ("dense", qu, None, 1.0)
            meta[f"quant_w{b}_uponly"] = dict(kind="quant", bits=b, cost_flops=1.0,
                                              cost_bytes=cb, use_gate=False)

        # 4-bit proxy composed with input sparsity (Prox-style stage 1). Reuses the
        # already-built 4-bit tensors rather than re-quantizing per keep-fraction.
        if 4 in bits_list:
            _, qu4, qg4, _ = proxies["quant_w4"]
            for s_keep in insp_list:
                nm = f"insp{s_keep}_q4"
                proxies[nm] = ("dense", qu4, qg4, s_keep)
                meta[nm] = dict(kind="quant+insp", bits=4, input_keep=s_keep,
                                cost_flops=2 * s_keep,
                                cost_bytes=2 * (4 / 16.0) * s_keep, use_gate=True)

        names = list(proxies) + ["oracle_up_ref", "prevtok"]
        stats = {
            n: {"hit": np.zeros(len(budgets)), "mass": np.zeros(len(budgets)),
                "cover": np.zeros((len(budgets), len(lams))),
                "cmass": np.zeros((len(budgets), len(lams))),
                "spear": 0.0, "n": 0}
            for n in names
        }

        prev_global = None          # (E*I,) oracle scores of the last token seen
        for s0 in range(0, T, args.chunk):
            x = X[s0:s0 + args.chunk].to(dev)
            t = x.shape[0]
            gc_, sc_ = g[s0:s0 + args.chunk].to(dev), sel[s0:s0 + args.chunk].to(dev)
            hits = []
            for e in torch.unique(sc_):
                tokid, slot = torch.where(sc_ == int(e))
                hits.append((int(e), slot, tokid))

            # ---- oracle: g*|SiLU(gate)*up|  (== oracle_mag_noW) -----------
            gate_t = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
            up_t = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
            for e, slot, tokid in hits:
                cur = x[tokid]
                gate_t[tokid, slot] = cur @ Wg_d[e].t()
                up_t[tokid, slot] = cur @ Wu_d[e].t()
            oracle = (gc_.unsqueeze(-1) * (F.silu(gate_t) * up_t).abs()).reshape(t, KI)
            del gate_t
            o_sorted = oracle.sort(dim=1, descending=True).values
            o_cum = o_sorted.cumsum(dim=1)
            o_mask = []
            for B in budgets:
                om = torch.zeros_like(oracle, dtype=torch.bool)
                om.scatter_(1, torch.topk(oracle, B, dim=1, sorted=False).indices, True)
                o_mask.append(om)

            def _score_stats(name, score):
                st = stats[name]
                for bi, B in enumerate(budgets):
                    p_idx = torch.topk(score, B, dim=1, sorted=False).indices
                    st["hit"][bi] += (
                        o_mask[bi].gather(1, p_idx).sum(dim=1).float() / B
                    ).sum().item()
                    st["mass"][bi] += (
                        oracle.gather(1, p_idx).sum(dim=1)
                        / o_cum[:, B - 1].clamp_min(1e-30)
                    ).sum().item()
                    # ---- cascade: nominate lam*B, verify exactly, keep top-B
                    for li, lam in enumerate(lams):
                        C = min(KI, max(B, int(round(lam * B))))
                        c_idx = torch.topk(score, C, dim=1, sorted=False).indices
                        cand_oracle = oracle.gather(1, c_idx)      # exact rescoring
                        f_local = cand_oracle.topk(B, dim=1, sorted=False).indices
                        f_idx = c_idx.gather(1, f_local)
                        st["cover"][bi, li] += (
                            o_mask[bi].gather(1, f_idx).sum(dim=1).float() / B
                        ).sum().item()
                        st["cmass"][bi, li] += (
                            oracle.gather(1, f_idx).sum(dim=1)
                            / o_cum[:, B - 1].clamp_min(1e-30)
                        ).sum().item()
                st["spear"] += _spearman(score, oracle) * t
                st["n"] += t

            _score_stats("oracle_up_ref", (gc_.unsqueeze(-1) * up_t.abs()).reshape(t, KI))
            del up_t

            # ---- D: previous token's realized oracle scores as the signal ----
            # Map through the global (E,I) space, since token t and t-1 route to
            # different experts. Row 0 of each chunk reuses the last row of the
            # previous chunk (prev_global), so the stream stays contiguous.
            flat_pos = (sc_.unsqueeze(-1) * I
                        + torch.arange(I, device=dev).view(1, 1, I)).reshape(t, KI)
            glob = torch.zeros((t, E * I), dtype=torch.float32, device=dev)
            glob.scatter_(1, flat_pos, oracle)
            shifted = torch.empty_like(glob)
            shifted[1:] = glob[:-1]
            shifted[0] = prev_global if prev_global is not None else glob[0]
            _score_stats("prevtok", shifted.gather(1, flat_pos))
            prev_global = glob[-1].clone()
            del glob, shifted

            # ---- proxies -----------------------------------------------------
            for name, (kind, PU, PG, in_keep) in proxies.items():
                xs = sparsify_input(x, in_keep) if in_keep < 1.0 else x
                up_p = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                gate_p = (torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                          if PG is not None else None)
                if kind == "basis":
                    # The point of a shared basis: B_b x is computed ONCE per token
                    # and reused by every co-activated expert. Hoisted out of the
                    # expert loop so the pilot's cost model matches its arithmetic.
                    cu, bu = PU
                    Zu = torch.einsum("th,mih->tmi", xs, bu)
                    Zg = None
                    if PG is not None:
                        cg, bg = PG
                        Zg = torch.einsum("th,mih->tmi", xs, bg)
                for e, slot, tokid in hits:
                    if kind == "basis":
                        up_p[tokid, slot] = torch.einsum("tmi,m->ti", Zu[tokid], cu[e])
                        if PG is not None:
                            gate_p[tokid, slot] = torch.einsum(
                                "tmi,m->ti", Zg[tokid], cg[e])
                    else:
                        cur = xs[tokid]
                        up_p[tokid, slot] = cur @ PU[e].t()
                        if PG is not None:
                            gate_p[tokid, slot] = cur @ PG[e].t()
                if kind == "basis":
                    del Zu, Zg
                if PG is not None:
                    score = gc_.unsqueeze(-1) * (F.silu(gate_p) * up_p).abs()
                else:
                    score = gc_.unsqueeze(-1) * up_p.abs()
                _score_stats(name, score.reshape(t, KI))
                del up_p, gate_p, score

            del oracle, o_cum, o_sorted, o_mask
            print(f"[layer {layer}] {min(s0 + args.chunk, T)}/{T} tokens", flush=True)

        # ---- emit --------------------------------------------------------
        for name in names:
            st = stats[name]
            if st["n"] == 0:
                continue
            if name == "oracle_up_ref":
                md = dict(kind="oracle", cost_flops=1.0, cost_bytes=1.0, use_gate=False)
            elif name == "prevtok":
                md = dict(kind="reuse", cost_flops=0.0, cost_bytes=0.0, use_gate=False)
            else:
                md = meta[name]
            row = {"layer": layer, "name": name, "n_tokens": int(st["n"]),
                   "spearman": st["spear"] / st["n"], **md}
            for rho in ratios:
                # all three matrices gathered to rho, plus the scorer overhead
                row[f"ffn_kept_flops@rho{rho}"] = rho + md["cost_flops"] / 3.0
                row[f"ffn_kept_bytes@rho{rho}"] = rho + md["cost_bytes"] / 3.0
            for bi, (rho, B) in enumerate(zip(ratios, budgets)):
                row[f"recall@rho{rho}"] = st["hit"][bi] / st["n"]
                row[f"mass@rho{rho}"] = st["mass"][bi] / st["n"]
                row[f"random_recall@rho{rho}"] = B / KI
                for li, lam in enumerate(lams):
                    row[f"casc_recall@rho{rho}_lam{lam}"] = st["cover"][bi, li] / st["n"]
                    row[f"casc_mass@rho{rho}_lam{lam}"] = st["cmass"][bi, li] / st["n"]
            all_rows.append(row)
            print(
                f"  {name:22s} cF={md['cost_flops']:.3f} cB={md['cost_bytes']:.3f} "
                f"sp={row['spearman']:.3f} "
                + " ".join(f"rec@{r_}={row[f'recall@rho{r_}']:.3f}" for r_ in ratios)
                + " | casc@1.5: "
                + " ".join(f"{row[f'casc_recall@rho{r_}_lam1.5']:.3f}" for r_ in ratios
                           if 1.5 in lams),
                flush=True)

        del Wu_d, Wg_d, proxies
        torch.cuda.empty_cache()

    out = os.path.join(args.out_dir, "pilot_scorers.json")
    with open(out, "w") as f:
        json.dump({"rows": all_rows, "ratios": ratios, "lams": lams}, f, indent=2)
    print(f"\n[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
