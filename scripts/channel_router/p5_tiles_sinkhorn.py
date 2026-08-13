#!/usr/bin/env python
"""P5 — tile-ability via balanced Sinkhorn clustering.  ★ the DOT-MoE connection

Builds ``n_tiles`` balanced groups of each expert's channels under four constructions
(co-activation spectral + Sinkhorn-balanced k-means, weight-similarity k-means, random
balanced, native single-tile) and asks how concentrated a token's oracle mask is across
tiles:

- touched-tile count per token,
- recall@B when the selection is restricted to the token's top-``n`` tiles, where tiles
  are chosen by their *oracle* mass (the ceiling for a two-level router) and by a static
  frequency-based tile score (what a cheap level-1 scorer could do).

Decision (plan): if the top-10 of 64 tiles cover ≥ 95% of the oracle mass, enable the
level-1 tile scorer — the output space shrinks by ``I/T``. The comparison between tile
constructions is a standalone result regardless of the router.

One GPU, no model load.
"""

import argparse
import json
import os
import sys
import time

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src"))

from src.channel_router.metrics import mass_recall, recall, select_topB  # noqa: E402
from src.channel_router.prep import LayerData  # noqa: E402
from src.channel_router.tiles import balanced_kmeans, spectral_embedding  # noqa: E402


@torch.no_grad()
def accumulate_coactivation(ld, ratio, *, fit_tokens, batch, log):
    """``(E, I, I)`` co-activation counts and ``(E, I)`` keep counts."""
    B = ld.budget(ratio)
    C = torch.zeros((ld.E, ld.I, ld.I), dtype=torch.float32, device=ld.device)
    cnt = torch.zeros((ld.E, ld.I), dtype=torch.float32, device=ld.device)
    nact = torch.zeros(ld.E, dtype=torch.float32, device=ld.device)
    idx_all = ld.take(ld.train_sl, fit_tokens)
    t0 = time.time()
    for s in range(0, len(idx_all), batch):
        x, sel, g, imp = ld.batch(idx_all[s:s + batch])
        keep = select_topB(imp, B)                             # (T,K,I)
        for e in sel.unique().tolist():
            tok, slot = (sel == e).nonzero(as_tuple=True)
            m = keep[tok, slot].float()                        # (n_e, I)
            C[e] += m.t() @ m
            cnt[e] += m.sum(0)
            nact[e] += m.shape[0]
        if (s // batch) % 8 == 0:
            log(f"    coact {s + x.shape[0]}/{len(idx_all)} ({time.time() - t0:.0f}s)")
    C /= nact.view(-1, 1, 1).clamp_min(1)
    return C, cnt / nact.view(-1, 1).clamp_min(1), nact


@torch.no_grad()
def build_all_tiles(ld, C, n_tiles, methods, log):
    """``{method: (E, I) long}`` tile labels, per expert."""
    out = {}
    for meth in methods:
        t0 = time.time()
        lab = torch.zeros((ld.E, ld.I), dtype=torch.long, device=ld.device)
        if meth == "native":
            pass                                               # all zeros: one tile
        else:
            for e in range(ld.E):
                if meth == "coact":
                    Z = spectral_embedding(C[e], dim=16)
                elif meth == "weight":
                    W = ld.w.Wg[e].float()
                    Z = W / W.norm(dim=1, keepdim=True).clamp_min(1e-9)
                elif meth == "random":
                    gen = torch.Generator(device="cpu").manual_seed(e)
                    l = torch.arange(ld.I) % n_tiles
                    lab[e] = l[torch.randperm(ld.I, generator=gen)].to(ld.device)
                    continue
                else:
                    raise ValueError(meth)
                lab[e] = balanced_kmeans(Z, n_tiles, seed=e)
        out[meth] = lab
        log(f"    tiles[{meth}] built in {time.time() - t0:.0f}s")
    return out


@torch.no_grad()
def evaluate_tiles(ld, ratio, lab, n_tiles, freq, ns, *, eval_tokens, batch):
    """Touched-tile stats + recall when restricted to the top-n tiles."""
    B = ld.budget(ratio)
    K, I = ld.K, ld.I
    idx_all = ld.take(ld.val_sl, eval_tokens)
    touched, cover = [], []
    rec = {("oracle", n): [0.0, 0.0] for n in ns}
    rec.update({("static", n): [0.0, 0.0] for n in ns})
    ntok = 0
    tile_freq = torch.zeros((ld.E, n_tiles), device=ld.device)
    tile_freq.scatter_add_(1, lab, freq)
    for s in range(0, len(idx_all), batch):
        x, sel, g, imp = ld.batch(idx_all[s:s + batch])
        T = x.shape[0]
        ref = select_topB(imp, B)
        tl = lab[sel]                                          # (T,K,I) in [0,n_tiles)
        off = (torch.arange(K, device=x.device).view(1, K, 1) * n_tiles + tl).reshape(T, -1)
        mass = torch.zeros((T, K * n_tiles), device=x.device)
        mass.scatter_add_(1, off, (imp * ref).reshape(T, -1))
        cnt = torch.zeros((T, K * n_tiles), device=x.device)
        cnt.scatter_add_(1, off, ref.reshape(T, -1).float())
        touched.append((cnt > 0).sum(1).float().mean())
        srt = mass.sort(dim=1, descending=True).values
        cover.append((srt.cumsum(1) / srt.sum(1, keepdim=True).clamp_min(1e-20)).mean(0))
        # what a *static* level-1 scorer could do: rank tiles by their held-out
        # keep-frequency mass, with no per-token information at all.
        stat_score = tile_freq[sel].reshape(T, K * n_tiles)
        for n in ns:
            for kind, sc in (("oracle", mass), ("static", stat_score)):
                allowed_t = torch.zeros_like(sc, dtype=torch.bool)
                allowed_t.scatter_(1, sc.topk(min(n, sc.shape[1]), dim=1).indices, True)
                allowed = allowed_t.gather(1, off).reshape(T, K, I)
                pred = select_topB(imp.masked_fill(~allowed, float("-inf")), B)
                rec[(kind, n)][0] += recall(pred, ref) * T
                rec[(kind, n)][1] += mass_recall(imp, pred, ref) * T
        ntok += T
    return {
        "touched_mean": float(torch.stack(touched).mean()),
        "coverage_curve": torch.stack(cover).mean(0).cpu().tolist(),
        "restricted": {f"{k[0]}_n{k[1]}": {"recall": v[0] / ntok,
                                           "mass_recall": v[1] / ntok}
                       for k, v in rec.items()},
        "tokens": ntok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="46")
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "docs/results/channel_router/data"))
    ap.add_argument("--tag", default="c4")
    ap.add_argument("--tokens", type=int, default=1 << 20)
    ap.add_argument("--ratios", default="0.125")
    ap.add_argument("--n-tiles", default="8,16,6", help="tiles per expert (I/T groups)")
    ap.add_argument("--methods", default="coact,weight,random,native")
    ap.add_argument("--ns", default="1,2,4,8,10,16,24,32,48,64")
    ap.add_argument("--fit-tokens", type=int, default=32768)
    ap.add_argument("--eval-tokens", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs/results/channel_router/phase0"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    def log(m):
        print(f"[p5] {m}", flush=True)

    out = {"args": vars(args), "layers": {}}
    ns = [int(v) for v in args.ns.split(",")]
    for layer in [int(v) for v in args.layers.split(",")]:
        ld = LayerData(args.data_dir, layer, tag=args.tag, tokens=args.tokens,
                       device=args.device, want_down=False)
        log(f"=== layer {layer}")
        lo = {}
        for ratio in [float(v) for v in args.ratios.split(",")]:
            C, freq, nact = accumulate_coactivation(
                ld, ratio, fit_tokens=args.fit_tokens, batch=args.batch, log=log)
            ro = {}
            for nt in [int(v) for v in args.n_tiles.split(",")]:
                if ld.I % nt:
                    log(f"  skip n_tiles={nt} (I={ld.I} not divisible)")
                    continue
                labs = build_all_tiles(ld, C, nt, args.methods.split(","), log)
                for meth, lab in labs.items():
                    n_eff = 1 if meth == "native" else nt
                    res = evaluate_tiles(ld, ratio, lab, n_eff, freq,
                                         [n for n in ns if n <= ld.K * n_eff],
                                         eval_tokens=args.eval_tokens,
                                         batch=args.batch)
                    ro[f"{meth}_nt{n_eff}"] = res
                    top10 = res["restricted"].get("oracle_n10")
                    log(f"  rho={ratio} {meth} nt={n_eff}: touched="
                        f"{res['touched_mean']:.1f}/{ld.K * n_eff}"
                        + (f" oracle@10 mass={top10['mass_recall']:.4f}" if top10 else "")
                        + f" static@10 mass="
                        + (f"{res['restricted']['static_n10']['mass_recall']:.4f}"
                           if 'static_n10' in res['restricted'] else "n/a"))
            lo[str(ratio)] = ro
            del C
            torch.cuda.empty_cache()
        out["layers"][str(layer)] = lo
        del ld
        torch.cuda.empty_cache()
        with open(os.path.join(args.out_dir, "p5_tiles_sinkhorn.json"), "w") as f:
            json.dump(out, f, indent=2)
    log(f"wrote {os.path.join(args.out_dir, 'p5_tiles_sinkhorn.json')}")


if __name__ == "__main__":
    main()
