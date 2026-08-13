#!/usr/bin/env python
"""Early investigation: is a cheap low-rank proxy informative enough to *select*
channels the way `oracle_mag` does?

`oracle_mag` needs the true SwiGLU intermediate to rank channels, so `gate_proj`
and `up_proj` must run at full width — a nominal −87.5% channel cut is only a
−29.2% whole-FFN cut. This script asks whether a **cheap block-low-rank proxy**
of that intermediate picks (nearly) the same channels, which would let all three
expert matrices be gathered to budget.

For one MoE layer, over C4 calibration tokens, we replay the exact per-token
cross-expert selection and compare each proxy's top-B set against the oracle's:

    recall@B  = |proxy top-B ∩ oracle top-B| / B      (set agreement)
    mass@B    = oracle score mass captured by the proxy's top-B, relative to the
                oracle's own top-B mass — the quantity that actually matters,
                since picking a *different* channel of equal magnitude is
                harmless while missing a dominant one is not.

Also reported: Spearman rank correlation over the pooled K·I scores, and a random
baseline (recall = B/(K·I)) so "high recall" can be read against chance.

**Scorer variants swept** (all from ``src/dynamic_active_param/lowrank_scorer.py``):

  * ``svd_r<r>``          global rank-r SVD of W_up and W_gate (m=n=1)
  * ``btt_m<m>n<n>_r<r>`` block grid, per-block rank r (higher effective rank at
                          the same FLOP cost)
  * each in ``up+gate`` mode (proxy ≈ SiLU(ĝ)·û, targets `oracle_mag_noW`) and
    ``up-only`` mode (proxy ≈ |û|, targets `oracle_up`'s cheaper signal)

Every variant is reported next to its **cost** (fraction of one full matmul) so
recall is always read per unit of compute. The oracle reference is
`oracle_mag_noW` (``g·|inter|``), which the Q1 ablation showed is statistically
tied with full `oracle_mag` while needing no weight statistics.

Heavy: loads the sharded 30B to capture one layer's input. Run via
launch-on-a100. The capture is cached, so re-runs of the (cheap) analysis are
fast and need no GPU.
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

from src.dynamic_active_param.lowrank_scorer import (
    factorize_blocks,
    scorer_cost_fraction,
    scorer_proxy,
)


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

def _capture_path(args, layer):
    return os.path.join(args.out_dir, f"capture_L{layer}_t{args.tokens}.pt")


def _ensure_captures(args, layers):
    """Capture every requested layer's input + weights in a **single** model load.

    Mirrors ``scripts/level2_oracle_ladder.py::_load_capture`` so the two
    diagnostics read the same slice of the model, but hooks all layers at once
    (one 30B load instead of one per layer) and frees the model before the
    analysis, so the sharded weights don't compete with it for GPU memory.
    """
    missing = [l for l in layers if not os.path.exists(_capture_path(args, l))]
    if not missing:
        print(f"[capture] all {len(layers)} captures cached", flush=True)
        return
    print(f"[capture] capturing layers {missing} (one model load)", flush=True)

    import gc

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.base.datasets import load_datasets

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
        max_memory={i: args.per_gpu_mem for i in range(torch.cuda.device_count())},
    )
    model.eval()

    rows = {l: [] for l in missing}
    state = {l: 0 for l in missing}
    gstate = {"mask": None}

    def make_hook(layer):
        def pre_hook(module, a, kw):
            x = a[0] if a else kw.get("hidden_states")
            if x is None or state[layer] >= args.tokens:
                return
            x = x.detach().reshape(-1, x.shape[-1])
            m = gstate["mask"]
            if m is not None and m.numel() == x.shape[0]:
                x = x[m.to(x.device)]
            take = min(args.tokens - state[layer], x.shape[0])
            if take > 0:
                rows[layer].append(x[:take].to("cpu", torch.float32))
                state[layer] += take
        return pre_hook

    blocks = {l: model.model.layers[l].mlp for l in missing}
    handles = [blocks[l].register_forward_pre_hook(make_hook(l), with_kwargs=True)
               for l in missing]

    in_dev = model.get_input_embeddings().weight.device
    ds = load_datasets("c4", tok, max_samples=args.tokens // args.seq_len + 16,
                      max_length=args.seq_len)
    try:
        with torch.no_grad():
            for i in range(0, len(ds), 8):
                if min(state.values()) >= args.tokens:
                    break
                chunk = [str(x) for x in ds[i:i + 8] if x]
                if not chunk:
                    continue
                enc = tok(chunk, max_length=args.seq_len, padding=True,
                          truncation=True, return_tensors="pt")
                am = enc.get("attention_mask")
                gstate["mask"] = am.reshape(-1).bool() if am is not None else None
                model(**{k: v.to(in_dev) for k, v in enc.items()}, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    os.makedirs(args.out_dir, exist_ok=True)
    for l in missing:
        block = blocks[l]
        X = torch.cat(rows[l], 0)[:args.tokens].contiguous()
        payload = {
            "X": X,
            "gate_w": block.gate.weight.data.detach().to("cpu", torch.float32).clone(),
            "Wg": torch.stack([e.gate_proj.weight.detach().cpu().float() for e in block.experts], 0),
            "Wu": torch.stack([e.up_proj.weight.detach().cpu().float() for e in block.experts], 0),
            "top_k": block.top_k,
            "norm_topk": block.norm_topk_prob,
        }
        torch.save(payload, _capture_path(args, l))
        print(f"[capture] saved L{l} (X {tuple(X.shape)})", flush=True)
        rows[l] = None

    # Free the sharded model before the analysis phase so it doesn't compete for
    # GPU memory with the (much smaller) scoring tensors.
    del model, blocks, handles
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[capture] model released", flush=True)


# --------------------------------------------------------------------------
# selection replay
# --------------------------------------------------------------------------

def _route(X, gate_w, K, norm_topk, dev, chunk=4096):
    """Router replay -> (g, sel) each (T, K), matching block.py exactly."""
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
    """Mean per-token Spearman correlation between two (t, N) score tensors."""
    ra = a.argsort(dim=1).argsort(dim=1).float()
    rb = b.argsort(dim=1).argsort(dim=1).float()
    ra = ra - ra.mean(dim=1, keepdim=True)
    rb = rb - rb.mean(dim=1, keepdim=True)
    num = (ra * rb).sum(dim=1)
    den = ra.norm(dim=1) * rb.norm(dim=1)
    return (num / den.clamp_min(1e-12)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--layers", default="6,22,38,46",
                    help="comma-separated layer indices to analyze")
    ap.add_argument("--layer", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--ratios", default="0.5,0.25,0.125",
                    help="kept fractions rho; B = round(rho*K*I)")
    ap.add_argument("--ranks", default="4,8,16,32", help="per-block ranks to sweep")
    ap.add_argument("--grids", default="1x1,2x2,4x2",
                    help="block grids mxn; 1x1 is a plain global SVD")
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--per-gpu-mem", default=os.environ.get("PER_GPU_MEM", "36GiB"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=2048, help="token chunk for scoring")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    layers = [int(x) for x in args.layers.split(",")] if args.layer is None else [args.layer]
    ratios = [float(x) for x in args.ratios.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]
    grids = [tuple(int(v) for v in g.split("x")) for g in args.grids.split(",")]
    dev = torch.device(args.device)

    _ensure_captures(args, layers)

    all_rows = []
    for layer in layers:
        cap = torch.load(_capture_path(args, layer), map_location="cpu")
        X, gate_w = cap["X"], cap["gate_w"]
        Wg, Wu = cap["Wg"], cap["Wu"]
        K, norm_topk = cap["top_k"], cap["norm_topk"]
        E, I, H = Wu.shape
        T = X.shape[0]
        KI = K * I
        budgets = [max(1, min(int(round(r * KI)), KI)) for r in ratios]
        print(f"\n[layer {layer}] T={T} E={E} I={I} H={H} K={K} "
              f"ratios={ratios} -> B={budgets}", flush=True)

        g, sel = _route(X, gate_w, K, norm_topk, dev)

        # ---- variants: (name, m, n, rank, use_gate) -------------------------
        variants = []
        for (m, n) in grids:
            if I % m or H % n:
                print(f"[layer {layer}] skip grid {m}x{n} (does not divide I={I}, H={H})")
                continue
            for r in ranks:
                if r > min(I // m, H // n):
                    continue
                tag = "svd" if (m == 1 and n == 1) else f"btt_m{m}n{n}"
                variants.append((f"{tag}_r{r}", m, n, r, True))
                variants.append((f"{tag}_r{r}_uponly", m, n, r, False))

        # ---- factorize per (grid, rank), keeping only the largest rank's cores
        # resident is unnecessary — they are small (E*(m*n)*(a+b)*r) — but we
        # build them all up front so the per-chunk loop stays pure compute.
        cores = {}
        for (m, n) in grids:
            if I % m or H % n:
                continue
            for r in ranks:
                if r > min(I // m, H // n):
                    continue
                cores[(m, n, r, "up")] = factorize_blocks(Wu.to(dev), m, n, r)
                cores[(m, n, r, "gate")] = factorize_blocks(Wg.to(dev), m, n, r)
        n_core_el = sum(c.L_core.numel() + c.R_core.numel() for c in cores.values())
        print(f"[layer {layer}] factorized {len(cores)} core sets "
              f"({4 * n_core_el / 2**30:.2f} GiB fp32)", flush=True)

        # ---- accumulators ---------------------------------------------------
        stats = {
            name: {
                "hit": np.zeros(len(budgets)),
                "mass": np.zeros(len(budgets)),
                "spear": 0.0,
                "n": 0,
            }
            for name, *_ in variants
        }
        # up-only oracle (oracle_up's target) tracked separately for reference
        stats["oracle_up_ref"] = {
            "hit": np.zeros(len(budgets)), "mass": np.zeros(len(budgets)),
            "spear": 0.0, "n": 0,
        }

        # (m,n,r) configs; each serves both the up+gate and up-only variant.
        configs = sorted({(m, n, r) for _, m, n, r, _ in variants})

        Wg_d, Wu_d = Wg.to(dev), Wu.to(dev)
        n_chunks = 0
        for s in range(0, T, args.chunk):
            x = X[s:s + args.chunk].to(dev)
            t = x.shape[0]
            gc, sc_ = g[s:s + args.chunk].to(dev), sel[s:s + args.chunk].to(dev)
            # sc_ is (t, K), so torch.where yields (token_idx, slot_idx) — in that
            # order (unlike block.py, whose expert_mask is transposed to (K, T)).
            hits = []
            for e in torch.unique(sc_):
                tokid, slot = torch.where(sc_ == int(e))
                hits.append((int(e), slot, tokid))

            # --- oracle: true intermediate, g*|inter| (== oracle_mag_noW) ----
            # Expert-major (as block.py does): loop the experts this chunk hits
            # and matmul only the tokens routed to each. Gathering weights per
            # token instead would materialize a (t, I, H) tensor — 12 GB at t=2048.
            gate_t = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
            up_t = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
            for e, slot, tokid in hits:
                cur = x[tokid]                                   # (n_e, H)
                gate_t[tokid, slot] = cur @ Wg_d[e].t()
                up_t[tokid, slot] = cur @ Wu_d[e].t()
            oracle = (gc.unsqueeze(-1) * (F.silu(gate_t) * up_t).abs()).reshape(t, KI)
            del gate_t
            o_cum = oracle.sort(dim=1, descending=True).values.cumsum(dim=1)
            o_mask = []
            for B in budgets:
                om = torch.zeros_like(oracle, dtype=torch.bool)
                om.scatter_(1, torch.topk(oracle, B, dim=1, sorted=False).indices, True)
                o_mask.append(om)

            def _score_stats(name, score):
                st = stats[name]
                for bi, B in enumerate(budgets):
                    p_idx = torch.topk(score, B, dim=1, sorted=False).indices
                    # recall: |proxy top-B ∩ oracle top-B| / B
                    st["hit"][bi] += (
                        o_mask[bi].gather(1, p_idx).sum(dim=1).float() / B
                    ).sum().item()
                    # mass: oracle score captured by proxy top-B, vs oracle's own
                    cap_mass = oracle.gather(1, p_idx).sum(dim=1)
                    st["mass"][bi] += (
                        cap_mass / o_cum[:, B - 1].clamp_min(1e-30)
                    ).sum().item()
                st["spear"] += _spearman(score, oracle) * t
                st["n"] += t

            # --- up-only oracle reference (what oracle_up ranks by) ----------
            _score_stats("oracle_up_ref", (gc.unsqueeze(-1) * up_t.abs()).reshape(t, KI))
            del up_t

            # --- proxies: one (m,n,r) config at a time, so at most two (t,K,I)
            # proxy tensors are live regardless of how many variants are swept.
            for (m, n, r) in configs:
                cu, cg = cores[(m, n, r, "up")], cores[(m, n, r, "gate")]
                up_p = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                gate_p = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                for e, slot, tokid in hits:
                    cur = x[tokid]
                    up_p[tokid, slot] = scorer_proxy(cur, cu.L_core[e], cu.R_core[e]).float()
                    gate_p[tokid, slot] = scorer_proxy(cur, cg.L_core[e], cg.R_core[e]).float()
                tag = "svd" if (m == 1 and n == 1) else f"btt_m{m}n{n}"
                _score_stats(
                    f"{tag}_r{r}",
                    (gc.unsqueeze(-1) * (F.silu(gate_p) * up_p).abs()).reshape(t, KI),
                )
                del gate_p
                _score_stats(
                    f"{tag}_r{r}_uponly",
                    (gc.unsqueeze(-1) * up_p.abs()).reshape(t, KI),
                )
                del up_p

            del oracle, o_cum, o_mask
            n_chunks += 1
            print(f"[layer {layer}] {min(s + args.chunk, T)}/{T} tokens", flush=True)

        # ---- emit ------------------------------------------------------------
        for name in stats:
            st = stats[name]
            if st["n"] == 0:
                continue
            # ``cost`` = compute spent purely to obtain the ranking signal, in
            # units of one full (I,H) matmul. oracle_up pays a full-width up_proj
            # (1.0); oracle_mag_noW would pay gate+up (2.0); a proxy pays
            # n_scorers * c. Reported alongside the whole-FFN kept fraction that
            # results when the decision lets all three matrices be gathered.
            if name == "oracle_up_ref":
                m = n = r = None
                cost = 1.0            # one full-width up_proj
                use_gate = False
            else:
                spec = next(v for v in variants if v[0] == name)
                _, m, n, r, use_gate = spec
                cost = scorer_cost_fraction(I, H, m, n, r) * (2 if use_gate else 1)
            row = {
                "layer": layer, "name": name, "m": m, "n": n, "rank": r,
                "use_gate": bool(use_gate), "n_tokens": int(st["n"]),
                "cost_frac_of_one_matmul": cost,
                "spearman": st["spear"] / st["n"],
            }
            for rho in ratios:
                # all three matrices gathered to rho, plus the scorer overhead
                row[f"ffn_kept@rho{rho}"] = rho + cost / 3.0
            for bi, (rho, B) in enumerate(zip(ratios, budgets)):
                row[f"recall@rho{rho}"] = st["hit"][bi] / st["n"]
                row[f"mass@rho{rho}"] = st["mass"][bi] / st["n"]
                row[f"random_recall@rho{rho}"] = B / KI
            all_rows.append(row)
            print(f"  {name:24s} cost={cost:.4f} spear={row['spearman']:.3f} " +
                  " ".join(f"rec@{r_}={row[f'recall@rho{r_}']:.3f}" for r_ in ratios),
                  flush=True)

    out = os.path.join(args.out_dir, "recall.json")
    with open(out, "w") as f:
        json.dump({"rows": all_rows, "ratios": ratios, "model": args.model}, f, indent=2)
    print(f"\n[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
