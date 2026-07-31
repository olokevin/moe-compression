#!/usr/bin/env python
"""Level-2 M1 oracle ladder + M3 structure diagnostic (subset, reconstruction).

Realizes the diagnostic half of
``docs/exps/dynamic_active_param/plan/plan_level2_impl.md``. On ~2k C4 tokens for
one MoE layer, measures the **layer-output reconstruction relative error**
``mean_t ‖ŷ_t − y_t‖ / ‖y_t‖`` at several active-param budgets for four selectors:

    level1       router g, block-diagonal pivoted-Cholesky order (current)
    pubsub       router g, offline public-subspace redundancy penalty (Level-2)
    oracle_mag   exact per-token magnitude, block-diagonal (runnable ceiling)
    oracle_exact exact per-token greedy OMP with FULL off-diagonal (Oracle-A)

The gap (oracle_exact − pubsub) and (pubsub − level1) is the M1 decision signal.
Also emits M3: cross-expert coherence bucketed by pivot rank, and the emergent
per-expert prefix-length histogram.

Heavy linalg on CPU (see the cublas-crash memory); capture needs the sharded 30B.
Run via launch-on-a100. Caches the layer capture to <out-dir>/capture_L<layer>.pt.
"""

import argparse
import json
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from src.dynamic_active_param.pivchol import pivoted_cholesky_batched


def _load_capture(args):
    """Capture (or load) one layer's MoE-block input over ~token_cap C4 tokens."""
    cap_path = os.path.join(args.out_dir, f"capture_L{args.layer}.pt")
    if os.path.exists(cap_path):
        print(f"[capture] loading {cap_path}")
        return torch.load(cap_path, map_location="cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.base.datasets import load_datasets

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
        max_memory={i: args.per_gpu_mem for i in range(torch.cuda.device_count())},
    )
    model.eval()
    block = model.model.layers[args.layer].mlp

    rows, state = [], {"n": 0}

    def pre_hook(module, a, kw):
        x = a[0] if a else kw.get("hidden_states")
        if x is None or state["n"] >= args.tokens:
            return
        x = x.detach().reshape(-1, x.shape[-1])
        take = min(args.tokens - state["n"], x.shape[0])
        rows.append(x[:take].to("cpu", torch.float32))
        state["n"] += take

    in_dev = model.get_input_embeddings().weight.device
    h = block.register_forward_pre_hook(pre_hook, with_kwargs=True)
    ds = load_datasets("c4", tok, max_samples=args.tokens // args.seq_len + 8, max_length=args.seq_len)
    try:
        with torch.no_grad():
            for i in range(0, len(ds), 8):
                if state["n"] >= args.tokens:
                    break
                enc = tok([str(x) for x in ds[i:i + 8] if x], max_length=args.seq_len,
                          padding=True, truncation=True, return_tensors="pt")
                model(**{k: v.to(in_dev) for k, v in enc.items()}, use_cache=False)
    finally:
        h.remove()

    X = torch.cat(rows, 0)[:args.tokens].contiguous()
    gate_w = block.gate.weight.data.detach().to("cpu", torch.float32).clone()
    Wg = torch.stack([e.gate_proj.weight.detach().cpu().float() for e in block.experts], 0)
    Wu = torch.stack([e.up_proj.weight.detach().cpu().float() for e in block.experts], 0)
    Wd = torch.stack([e.down_proj.weight.detach().cpu().float() for e in block.experts], 0)
    top_k = block.top_k
    norm_topk = block.norm_topk_prob
    payload = {"X": X, "gate_w": gate_w, "Wg": Wg, "Wu": Wu, "Wd": Wd,
               "top_k": top_k, "norm_topk": norm_topk}
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(payload, cap_path)
    print(f"[capture] saved {cap_path} (X {tuple(X.shape)})")
    del model
    return payload


def _per_expert_pivchol(Wd, cov, lam, dev):
    """Level-1 per-expert coupling Theta=G⊙(Wd^T Wd), pivoted Cholesky -> (pivrank, gains)."""
    E, d, m = Wd.shape
    B = torch.bmm(Wd.transpose(1, 2), Wd)
    G = torch.eye(m).unsqueeze(0).repeat(E, 1, 1)
    for eid, c in cov.items():
        G[eid] = c.float()
    theta = (G * B).to(dev)
    perm, gains = pivoted_cholesky_batched(theta, lambda_r=lam)
    pivrank = torch.argsort(perm, dim=1)
    return pivrank.cpu(), gains.cpu(), G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--layer", type=int, default=46)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--pub-r", type=int, default=8)
    ap.add_argument("--scores-dir", required=True, help="dir holding expert_covariances.pth")
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs/results/level2"))
    ap.add_argument("--per-gpu-mem", default=os.environ.get("PER_GPU_MEM", "36GiB"))
    ap.add_argument("--ratios", default="0.50,0.625,0.75,0.875")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cap = _load_capture(args)
    X, gate_w = cap["X"], cap["gate_w"]
    Wg, Wu, Wd = cap["Wg"], cap["Wu"], cap["Wd"]
    K, norm_topk = cap["top_k"], cap["norm_topk"]
    E, d, m = Wd.shape
    T = X.shape[0]
    dev = torch.device(args.device)

    cov = torch.load(os.path.join(args.scores_dir, "expert_covariances.pth"),
                     map_location="cpu").get(args.layer, {})
    pivrank, gains, G = _per_expert_pivchol(Wd, cov, args.lam, dev)  # (E,m),(E,m)

    # --- pubsub offline: public basis U (top-r of sum_e Wd G Wd^T), deflated gains
    Gsym = 0.5 * (G + G.transpose(1, 2))
    ev, evec = torch.linalg.eigh(Gsym)
    sqrtG = evec @ (ev.clamp_min(0).sqrt().unsqueeze(-1) * evec.transpose(1, 2))
    A = torch.bmm(Wd, sqrtG)
    M = torch.einsum("edm,efm->df", A, A); M = 0.5 * (M + M.t())
    _, Mev = torch.linalg.eigh(M)
    U = Mev[:, -args.pub_r:]                                    # (d, r)
    g_diag = torch.diagonal(G, dim1=1, dim2=2).clamp_min(0).sqrt()  # (E, m)
    coef = torch.einsum("dr,edm->erm", U, Wd) * g_diag.unsqueeze(1)  # (E,r,m)
    carrier_val, carrier_j = coef.abs().max(dim=2)             # (E,r)
    Wt = Wd - torch.einsum("dr,erm->edm", U, torch.einsum("dr,edm->erm", U, Wd))
    Bpriv = torch.bmm(Wt.transpose(1, 2), Wt)
    perm_priv, gains_priv = pivoted_cholesky_batched((G * Bpriv).to(dev), lambda_r=args.lam)
    pivrank_priv = torch.argsort(perm_priv, dim=1).cpu()
    gains_priv = gains_priv.cpu()

    # --- routing over the captured tokens
    logits = X @ gate_w.t()                                    # (T, E)
    probs = torch.softmax(logits, dim=1)
    g_topk, sel = torch.topk(probs, K, dim=1)                  # (T,K)
    if norm_topk:
        g_topk = g_topk / g_topk.sum(1, keepdim=True)

    # per-token, per-slot intermediate (T,K,m) and full output y (T,d)
    inter = torch.empty((T, K, m))
    for s in range(K):
        for eid in range(E):
            msk = sel[:, s] == eid
            if not msk.any():
                continue
            xe = X[msk]
            h = torch.nn.functional.silu(xe @ Wg[eid].t()) * (xe @ Wu[eid].t())
            inter[msk, s] = h
    # y = sum_s g_s * (inter_s @ Wd_sel^T)
    y = torch.zeros((T, d))
    for s in range(K):
        Wd_s = Wd[sel[:, s]]                                   # (T,d,m)
        y += g_topk[:, s:s+1] * torch.einsum("tdm,tm->td", Wd_s, inter[:, s])

    col_norm = Wd.norm(dim=1)                                  # (E,m)

    def recon(keep):
        """keep (T,K,m) bool -> mean rel error of reconstructed y."""
        yh = torch.zeros((T, d))
        for s in range(K):
            Wd_s = Wd[sel[:, s]]
            masked = inter[:, s] * keep[:, s].float()
            yh += g_topk[:, s:s+1] * torch.einsum("tdm,tm->td", Wd_s, masked)
        err = (yh - y).norm(dim=1) / y.norm(dim=1).clamp_min(1e-9)
        return float(err.mean())

    def topB(score, B):
        flat = score.reshape(T, K * m)
        idx = torch.topk(flat, B, dim=1).indices
        kmask = torch.zeros_like(flat, dtype=torch.bool)
        kmask.scatter_(1, idx, True)
        return kmask.reshape(T, K, m)

    ratios = [float(x) for x in args.ratios.split(",")]
    results = {}
    for ratio in ratios:
        B = int(round((1 - ratio) * K * m))
        # level1: g^2 * sigma(channel)
        sig = torch.gather(gains[sel], 2, pivrank[sel])        # (T,K,m) per physical channel
        keep_l1 = topB((g_topk**2).unsqueeze(-1) * sig, B)
        # oracle_mag: g * |inter| * ||Wd[:,j]||
        keep_om = topB(g_topk.unsqueeze(-1) * inter.abs() * col_norm[sel], B)
        # pubsub: private g^2*sigma_priv + forced public carriers (dedup per dir)
        sigp = torch.gather(gains_priv[sel], 2, pivrank_priv[sel])
        score_ps = (g_topk**2).unsqueeze(-1) * sigp
        cval = carrier_val[sel]                                # (T,K,r)
        cidx = carrier_j[sel]                                  # (T,K,r)
        for rr in range(args.pub_r):
            best_slot = cval[:, :, rr].argmax(dim=1)           # (T,)
            tt = torch.arange(T)
            ch = cidx[tt, best_slot, rr]
            score_ps[tt, best_slot, ch] = float("inf")
        keep_ps = topB(score_ps, B)
        # oracle_exact: per-token greedy OMP with full off-diagonal (subset only)
        rel_oe = _omp_recon(inter, Wd, sel, g_topk, y, B)
        results[f"{ratio:.3f}"] = {
            "B": B,
            "level1": recon(keep_l1),
            "oracle_mag": recon(keep_om),
            "pubsub": recon(keep_ps),
            "oracle_exact": rel_oe,
        }
        print(f"ratio {ratio}: {results[f'{ratio:.3f}']}")

    # --- M3: coherence bucketed by pivot rank (mean |Theta_ij|/sqrt(ii*jj))
    m3 = _m3_coherence(G, Wd, pivrank, args.lam)
    out = {"layer": args.layer, "tokens": T, "K": K, "E": E, "m": m,
           "pub_r": args.pub_r, "recon": results, "m3_coherence_by_rank": m3}
    with open(os.path.join(args.out_dir, f"ladder_L{args.layer}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[done] wrote", os.path.join(args.out_dir, f"ladder_L{args.layer}.json"))


def _omp_recon(inter, Wd, sel, g_topk, y, B, max_tokens=512):
    """Per-token greedy OMP over K*m atoms (a_j = g_s * inter_{s,j} * Wd_{s,:,j}),
    selecting B atoms to minimize ‖y − sum a‖. Subset of tokens for tractability."""
    T, K, m = inter.shape
    d = Wd.shape[1]
    n = min(T, max_tokens)
    errs = []
    for t in range(n):
        # build atom matrix A (d, K*m): column (s,j) = g_s * inter[t,s,j] * Wd[sel,: ,j]
        cols = []
        for s in range(K):
            wd = Wd[sel[t, s]]                                  # (d,m)
            a = wd * (g_topk[t, s] * inter[t, s]).unsqueeze(0)  # (d,m)
            cols.append(a)
        Amat = torch.cat(cols, dim=1)                          # (d, K*m)
        target = y[t]
        # greedy OMP
        residual = target.clone()
        chosen = []
        norms = Amat.norm(dim=0).clamp_min(1e-9)
        avail = torch.ones(Amat.shape[1], dtype=torch.bool)
        for _ in range(B):
            proj = (Amat.t() @ residual).abs() / norms
            proj[~avail] = -1
            j = int(proj.argmax())
            chosen.append(j)
            avail[j] = False
            As = Amat[:, chosen]
            # least squares fit
            coef, *_ = torch.linalg.lstsq(As, target.unsqueeze(1))
            residual = target - (As @ coef).squeeze(1)
        errs.append(float(residual.norm() / target.norm().clamp_min(1e-9)))
    return sum(errs) / len(errs)


def _m3_coherence(G, Wd, pivrank, lam, n_pairs=64, n_buckets=8):
    """Cross-expert coherence mu bucketed by pivoted-Cholesky rank.

    For a sample of expert pairs, mu_{ij} = |Theta_cross_ij| / sqrt(Theta_ii Theta_jj)
    where Theta_cross uses cross covariance ~ Wd_e^T Wd_f (H=I) modulated by
    activation scale. Bucketed by min(rank_i, rank_j)."""
    E, d, m = Wd.shape
    g = torch.Generator().manual_seed(0)
    buckets = [[] for _ in range(n_buckets)]
    gdiag = torch.diagonal(G, dim1=1, dim2=2).clamp_min(0).sqrt()  # (E,m)
    for _ in range(n_pairs):
        e = int(torch.randint(0, E, (1,), generator=g))
        f = int(torch.randint(0, E, (1,), generator=g))
        if e == f:
            continue
        # cross output-space coupling of channels: (Wd_e[:,i]·Wd_f[:,j]) scaled by act.
        cross = (Wd[e].t() @ Wd[f])                            # (m,m)
        cross = cross * gdiag[e].unsqueeze(1) * gdiag[f].unsqueeze(0)
        di = (Wd[e].norm(dim=0) * gdiag[e]).clamp_min(1e-9)
        dj = (Wd[f].norm(dim=0) * gdiag[f]).clamp_min(1e-9)
        mu = cross.abs() / (di.unsqueeze(1) * dj.unsqueeze(0))
        ri = pivrank[e].unsqueeze(1).expand(m, m)
        rj = pivrank[f].unsqueeze(0).expand(m, m)
        rmin = torch.minimum(ri, rj)
        for b in range(n_buckets):
            lo = b * m // n_buckets
            hi = (b + 1) * m // n_buckets
            sel = (rmin >= lo) & (rmin < hi)
            if sel.any():
                buckets[b].append(float(mu[sel].mean()))
    return [sum(x) / len(x) if x else None for x in buckets]


if __name__ == "__main__":
    main()
