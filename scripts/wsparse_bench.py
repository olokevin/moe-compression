#!/usr/bin/env python
"""Wall-clock cost of the scoring pass, at Qwen3-30B-A3B's real MoE dimensions.

Everything in these docs is an *active-parameter* claim; this measures what the
masking-simulation scorers actually cost per MoE block so the eval budget can be
planned and the doc can say something about latency. One synthetic MoE block
(E=128, I=768, H=2048, K=8) with random weights, one GPU.

Note what is and is not being measured: the simulation runs the proxy *and* the
full-width exact intermediate (that is how fake pruning works), so these numbers
are the cost of the **scoring pass** relative to one full-width gate+up, not the
latency of a deployed gathered-expert kernel.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from src.dynamic_active_param.sparse_probe import (
    build_layer_probe, descending_abs_ranks, probe_expert_scores,
    sparsify_input_topk,
)
from src.dynamic_active_param.weight_sparse import (
    build_layer_wsparse, wsparse_expert_scores, wsparse_layer_bands,
)

LADDER = (1.0, 0.7, 0.49, 0.343, 0.24, 0.168, 0.118, 0.082)


class _E(nn.Module):
    def __init__(self, H, I, dtype, dev):
        super().__init__()
        self.up_proj = nn.Linear(H, I, bias=False).to(dev, dtype)
        self.gate_proj = nn.Linear(H, I, bias=False).to(dev, dtype)


def bench(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3          # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--I", type=int, default=768)
    ap.add_argument("--H", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=1600, help="tokens in the batch")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--density", type=float, default=0.1125)
    ap.add_argument("--ladders", default="8,6,4")
    args = ap.parse_args()

    dev = torch.device("cuda")
    dtype = torch.bfloat16
    E, I, H, T, K = args.experts, args.I, args.H, args.tokens, args.topk
    experts = [_E(H, I, dtype, dev) for _ in range(E)]
    x = torch.randn(T, H, device=dev, dtype=dtype)
    g = torch.rand(T, K, device=dev)
    g = g / g.sum(dim=1, keepdim=True)
    sel = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(T)])
    hits = []
    for e in range(E):
        tok, slot = torch.where(sel == e)
        if tok.numel():
            hits.append((e, slot, tok))
    print(f"E={E} I={I} H={H} T={T} K={K} experts_hit={len(hits)} dtype={dtype}")

    # reference: one full-width gate+up over the token's K experts (what
    # oracle_mag_noW pays, and what the exact intermediate in the sim costs anyway)
    def full():
        for e, slot, tok in hits:
            cur = x[tok]
            experts[e].gate_proj(cur)
            experts[e].up_proj(cur)
    t_full = bench(full, args.iters)
    print(f"  {'full-width gate+up (reference)':52s} {t_full:8.1f} ms   1.00x")

    probe = build_layer_probe(experts, bits=16, rho_input=args.density,
                              input_alloc="uniform")

    def col():
        xs = sparsify_input_topk(x, args.density)
        for e, slot, tok in hits:
            probe_expert_scores(xs[tok], probe, e)
    t = bench(col, args.iters)
    print(f"  {f'input_sparse rho_input={args.density} (columns)':52s} "
          f"{t:8.1f} ms   {t / t_full:.2f}x")

    rect = build_layer_wsparse(experts, levels="0.25x0.45")

    def rectf():
        ranks, _ = descending_abs_ranks(x)
        for e, slot, tok in hits:
            wsparse_expert_scores(x[tok], rect, e, ranks=ranks[tok])
    t = bench(rectf, args.iters)
    print(f"  {'weight_sparse rectangle 0.25x0.45 (1 level)':52s} "
          f"{t:8.1f} ms   {t / t_full:.2f}x")

    for nl in [int(v) for v in args.ladders.split(",")]:
        lad = LADDER[:nl]
        p = build_layer_wsparse(experts, levels=tuple((0.0, rf) for rf in lad),
                                alloc_mode="tau", density=args.density,
                                count_reads=True)

        def tau(p=p):
            _, sa = descending_abs_ranks(x)
            lu, lg = wsparse_layer_bands(sa, p, sel)
            for e, slot, tok in hits:
                wsparse_expert_scores(x[tok], p, e, lvl_u=lu[tok, slot],
                                      lvl_g=lg[tok, slot])
        t = bench(tau, args.iters)
        print(f"  {f'weight_sparse tau, {nl}-level ladder (min rf {lad[-1]:g})':52s} "
              f"{t:8.1f} ms   {t / t_full:.2f}x   "
              f"realized density {p.reads_sum / max(p.reads_n, 1):.4f}")

        def levels_only(p=p):
            _, sa = descending_abs_ranks(x)
            wsparse_layer_bands(sa, p, sel)
        tl = bench(levels_only, args.iters)
        print(f"    {f'of which: the batched tau bisection':50s} "
              f"{tl:8.1f} ms   {tl / t_full:.2f}x")


if __name__ == "__main__":
    main()
