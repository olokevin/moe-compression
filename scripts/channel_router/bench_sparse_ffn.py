#!/usr/bin/env python
"""§3.1.3 — measured wall-clock for the sparse FFN, router overhead isolated.

**No FLOP-only claims.** This benchmarks one real MoE block (E=128, I=768, H=2048, top-8)
in bf16 on one GPU and reports tokens/s plus bytes-moved for:

``dense``            the upstream per-expert loop at full width — the baseline.
``router_only``      just the selection: ``Q^T h`` + the ``K`` gathered embedding blocks
                     + top-B. This is the overhead that has to be paid back.
``sparse_gather``    selection, then ``index_select`` the kept rows of ``gate/up`` and
                     columns of ``down`` and matmul the narrow weights. Honest end-to-end
                     number for an implementation with no custom kernel: the gather
                     copies the weights it reads, so bandwidth is paid twice.
``sparse_pregathered`` the same matmuls on already-narrow weights (gather cost excluded).
                     This is the *upper bound* a fused gather-matmul kernel could reach,
                     i.e. what the accounting assumes, so the pair brackets reality.

Decode-regime batch sizes are the interesting ones (batch 1–32): that is where the FFN is
bandwidth-bound and reading 1/8 of the rows can actually pay.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src"))

from src.channel_router.data import load_weights  # noqa: E402
from src.channel_router.metrics import select_topB  # noqa: E402
from src.channel_router.train_utils import load_router_artifact  # noqa: E402


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=46)
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "docs/results/channel_router/data"))
    ap.add_argument("--router-ckpt", default="")
    ap.add_argument("--ratio", type=float, default=0.125)
    ap.add_argument("--batches", default="1,4,16,64")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(
        _REPO, "docs/results/channel_router/eval/bench_sparse_ffn.json"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = torch.device(args.device)

    def log(m):
        print(f"[bench] {m}", flush=True)

    w = load_weights(args.data_dir, args.layer, want_down=True)
    E, I, H = w.Wu.shape
    K = w.top_k
    B = max(1, int(round(args.ratio * K * I)))
    Wg = w.Wg.to(dev, torch.bfloat16)
    Wu = w.Wu.to(dev, torch.bfloat16)
    Wd = w.Wd.to(dev, torch.bfloat16)
    gate_w = w.gate_w.to(dev, torch.bfloat16)
    router = None
    if args.router_ckpt:
        router = load_router_artifact(args.router_ckpt, device=dev)[args.layer]
        router = router.to(torch.float32)
    log(f"layer {args.layer}: E={E} I={I} H={H} K={K} B={B} (rho={B / (K * I):.4f})")

    rows = []
    for bs in [int(v) for v in args.batches.split(",")]:
        torch.manual_seed(0)
        x = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
        probs = F.softmax(x.float() @ gate_w.float().t(), dim=1)
        g, sel = probs.topk(K, dim=-1)
        g = (g / g.sum(-1, keepdim=True)).to(torch.bfloat16)

        def dense():
            out = torch.zeros(bs, H, device=dev, dtype=torch.bfloat16)
            for k in range(K):
                for t in range(bs):
                    e = int(sel[t, k])
                    gp = x[t] @ Wg[e].t()
                    up = x[t] @ Wu[e].t()
                    out[t] += g[t, k] * (F.silu(gp) * up) @ Wd[e].t()
            return out

        # keep-index sets (computed once; the selection itself is timed separately)
        with torch.no_grad():
            if router is not None:
                keep = router.select(x.float(), sel, B, g=g.float())
            else:
                inter = torch.stack([
                    torch.stack([F.silu(x[t] @ Wg[int(sel[t, k])].t())
                                 * (x[t] @ Wu[int(sel[t, k])].t()) for k in range(K)])
                    for t in range(bs)])
                keep = select_topB(inter.abs().float(), B)
        idxs = [[keep[t, k].nonzero(as_tuple=True)[0] for k in range(K)]
                for t in range(bs)]
        pre = [[(Wg[int(sel[t, k])][idxs[t][k]].contiguous(),
                 Wu[int(sel[t, k])][idxs[t][k]].contiguous(),
                 Wd[int(sel[t, k])][:, idxs[t][k]].contiguous())
                for k in range(K)] for t in range(bs)]

        def router_only():
            if router is None:
                return None
            return router.select(x.float(), sel, B, g=g.float())

        # Pre-gathered router: the same arithmetic with the K selected experts'
        # embedding blocks already contiguous, as one bmm + one top-k. This is the
        # analogue of ``sparse_pregathered`` for the scorer — it excludes the gather
        # exactly as that row does, so ``sparse_pregathered + router_pregathered`` is
        # the projected cost of a fused implementation, while ``router_only`` is what
        # today's per-expert Python loop actually costs.
        Cs = Cs2 = beta_s = None
        if router is not None:
            with torch.no_grad():
                Cr = router.C.view(E, I, router.rp)
                Cs = torch.stack([Cr[sel[t]] for t in range(bs)]).to(torch.bfloat16)
                if router.C2 is not None:
                    C2r = router.C2.view(E, I, router.rp)
                    Cs2 = torch.stack([C2r[sel[t]] for t in range(bs)]).to(torch.bfloat16)
                beta_s = (router.beta.view(E, I)[sel].to(torch.bfloat16)
                          if router.beta is not None else None)
                Qb = router.Q.to(torch.bfloat16)
                fscale = router.feat_scale.to(torch.bfloat16)
                oidx = router.outlier_idx

        def router_pregathered():
            if router is None:
                return None
            f = x @ Qb
            if router.m:
                f = torch.cat([f, x[:, oidx]], dim=1)
            f = (f * fscale).unsqueeze(-1)                        # (bs, rp, 1)
            s1 = torch.bmm(Cs.reshape(bs, K * I, -1), f).squeeze(-1)
            sc = s1.abs().clamp_min(1e-8).log()
            if Cs2 is not None:
                s2 = torch.bmm(Cs2.reshape(bs, K * I, -1), f).squeeze(-1)
                sc = sc + s2.abs().clamp_min(1e-8).log()
            if beta_s is not None:
                sc = sc + beta_s.reshape(bs, K * I)
            sc = sc + g.reshape(bs, K, 1).expand(bs, K, I).reshape(bs, K * I).log()
            return sc.topk(B, dim=1).indices

        def sparse_gather():
            out = torch.zeros(bs, H, device=dev, dtype=torch.bfloat16)
            for t in range(bs):
                for k in range(K):
                    e = int(sel[t, k])
                    ii = idxs[t][k]
                    wg = Wg[e].index_select(0, ii)
                    wu = Wu[e].index_select(0, ii)
                    wd = Wd[e].index_select(1, ii)
                    out[t] += g[t, k] * (F.silu(x[t] @ wg.t()) * (x[t] @ wu.t())) @ wd.t()
            return out

        def sparse_pregathered():
            out = torch.zeros(bs, H, device=dev, dtype=torch.bfloat16)
            for t in range(bs):
                for k in range(K):
                    wg, wu, wd = pre[t][k]
                    out[t] += g[t, k] * (F.silu(x[t] @ wg.t()) * (x[t] @ wu.t())) @ wd.t()
            return out

        res = {"batch": bs, "B": B}
        for name, fn in (("dense", dense), ("sparse_gather", sparse_gather),
                         ("sparse_pregathered", sparse_pregathered),
                         ("router_only", router_only),
                         ("router_pregathered", router_pregathered)):
            if fn() is None:
                continue
            dt = timeit(fn, iters=args.iters)
            res[name] = {"ms": 1e3 * dt, "tok_per_s": bs / dt}
        # bytes moved: weights read per token (bf16), the bandwidth-bound quantity
        res["bytes_per_token"] = {
            "dense": 2 * 3 * K * I * H,
            "sparse": 2 * 3 * B * H,
        }
        res["flops_per_token"] = {"dense": 2 * 3 * K * I * H, "sparse": 2 * 3 * B * H}
        rows.append(res)
        log(f"  bs={bs}: " + ", ".join(
            f"{k}={v['ms']:.3f}ms" for k, v in res.items()
            if isinstance(v, dict) and "ms" in v))
        if "sparse_pregathered" in res:
            sp = res["dense"]["ms"] / res["sparse_pregathered"]["ms"]
            spg = res["dense"]["ms"] / res["sparse_gather"]["ms"]
            ro = res.get("router_only", {}).get("ms", 0.0)
            rp_ms = res.get("router_pregathered", {}).get("ms", 0.0)
            net = res["dense"]["ms"] / (res["sparse_pregathered"]["ms"] + ro)
            net_pg = res["dense"]["ms"] / (res["sparse_pregathered"]["ms"] + rp_ms)
            log(f"      speedup pregathered={sp:.2f}x  gather={spg:.2f}x  "
                f"net(loop router)={net:.2f}x  net(pregathered router)={net_pg:.2f}x")
            res["speedup"] = {"pregathered": sp, "gather": spg,
                              "net_with_loop_router": net,
                              "net_with_pregathered_router": net_pg}

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
