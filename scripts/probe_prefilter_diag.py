#!/usr/bin/env python
"""Is there a *static* set of channels the probe never needs to look at?

The probe's cost is linear in how many channel rows it reads. If some
(expert, channel) pairs essentially never reach the oracle's per-token top-B, they
could be excluded offline by a per-expert bitmask (``I`` bits per expert, free) and
the probe would read only the uncertain remainder — a straight multiplier on its
byte cost, spendable on more input coordinates instead.

That is only true if low-frequency channels also carry negligible *mass*. This
script measures exactly that, with an honest split: keep-frequencies are estimated
on the first half of the tokens and the mass loss is evaluated on the second half,
so a channel cannot be excluded using the same tokens that judge it.

Reported per forbidden fraction ``q``:
  * ``mass_kept``   — fraction of the oracle top-B score mass still reachable
  * ``count_kept``  — fraction of the oracle's top-B channels still reachable
  * ``feasible``    — whether the surviving channels still number >= B per token

One GPU, no model load.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts.idea_pilot_scorers import _route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,22,38,46")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--ratios", default="0.25,0.125")
    ap.add_argument("--forbid", default="0.25,0.5,0.625,0.75,0.875")
    ap.add_argument("--capture-dir", default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--out", default=os.path.join(_REPO, "docs/results/idea_pilot/prefilter.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=1024)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    ratios = [float(x) for x in args.ratios.split(",")]
    forbid = [float(x) for x in args.forbid.split(",")]
    dev = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    out = []
    for layer in layers:
        p = os.path.join(args.capture_dir, f"capture_L{layer}_t{args.tokens}.pt")
        if not os.path.exists(p):
            p = os.path.join(args.capture_dir, f"capture_L{layer}_t{args.tokens}_wd.pt")
        if not os.path.exists(p):
            print(f"[skip] no capture for L{layer}", flush=True)
            continue
        cap = torch.load(p, map_location="cpu")
        X, gate_w, Wg, Wu = cap["X"], cap["gate_w"], cap["Wg"], cap["Wu"]
        K, norm_topk = cap["top_k"], cap["norm_topk"]
        E, I, H = Wu.shape
        if args.max_tokens:
            X = X[:args.max_tokens]
        T = X.shape[0]
        half = T // 2
        g, sel = _route(X, gate_w, K, norm_topk, dev)
        Wu_d, Wg_d = Wu.to(dev).float(), Wg.to(dev).float()
        del cap, Wu, Wg
        print(f"\n[layer {layer}] T={T} (fit {half} / eval {T-half}) E={E} I={I} K={K}",
              flush=True)

        for r in ratios:
            B = max(1, min(int(round(r * K * I)), K * I))
            # ---- pass 1: keep-frequency per (expert, channel) on the fit half
            freq = torch.zeros((E, I), dtype=torch.float32, device=dev)
            # ---- pass 2 uses the eval half; run both in one loop
            stats = {q: {"mass": 0.0, "cnt": 0.0} for q in forbid}
            total_mass = 0.0
            total_cnt = 0.0
            allowed = None
            for phase in ("fit", "eval"):
                lo, hi = (0, half) if phase == "fit" else (half, T)
                if phase == "eval":
                    # rank channels by fitted frequency; forbid the lowest q
                    order = freq.argsort(dim=1)                     # ascending
                    rank = order.argsort(dim=1)                     # (E,I) 0=rarest
                    allowed = {q: rank >= int(round(q * I)) for q in forbid}
                for s0 in range(lo, hi, args.chunk):
                    x = X[s0:min(s0 + args.chunk, hi)].to(dev)
                    t = x.shape[0]
                    gc_ = g[s0:s0 + t].to(dev)
                    sc_ = sel[s0:s0 + t].to(dev)
                    inter = torch.zeros((t, K, I), dtype=torch.float32, device=dev)
                    for e in torch.unique(sc_):
                        tok, slot = torch.where(sc_ == int(e))
                        cur = x[tok]
                        inter[tok, slot] = (F.silu(cur @ Wg_d[int(e)].t())
                                            * (cur @ Wu_d[int(e)].t()))
                    score = (gc_.unsqueeze(-1) * inter.abs()).reshape(t, K * I)
                    idx = score.topk(B, dim=1, sorted=False).indices     # (t,B)
                    if phase == "fit":
                        slot = idx // I
                        chan = idx % I
                        eid = sc_.gather(1, slot)                        # (t,B)
                        freq.index_put_((eid.reshape(-1), chan.reshape(-1)),
                                        torch.ones(t * B, device=dev),
                                        accumulate=True)
                    else:
                        val = score.gather(1, idx)                       # (t,B)
                        total_mass += float(val.sum())
                        total_cnt += t * B
                        slot = idx // I
                        chan = idx % I
                        eid = sc_.gather(1, slot)
                        for q in forbid:
                            ok = allowed[q][eid, chan]                   # (t,B) bool
                            stats[q]["mass"] += float((val * ok).sum())
                            stats[q]["cnt"] += float(ok.sum())
                    del inter, score

            for q in forbid:
                surviving = K * (I - int(round(q * I)))
                row = {"layer": layer, "rho": r, "forbid_frac": q,
                       "mass_kept": stats[q]["mass"] / max(total_mass, 1e-30),
                       "count_kept": stats[q]["cnt"] / max(total_cnt, 1e-30),
                       "surviving_channels": surviving, "B": B,
                       "feasible": surviving >= B}
                out.append(row)
                print(f"  rho={r} forbid={q:<5} mass_kept={row['mass_kept']:.4f} "
                      f"count_kept={row['count_kept']:.4f} "
                      f"surviving={surviving} (B={B}) feasible={row['feasible']}",
                      flush=True)

        del Wu_d, Wg_d
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({"rows": out}, f, indent=2)
    print(f"\n[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
