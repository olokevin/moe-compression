#!/usr/bin/env python
"""P4 — static / dynamic decomposition: size the free bypass and the true dynamic budget.

Three numbers, all measured on a *held-out* slice so the "static" predictor is a
legitimate predictor rather than an oracle:

1. **Channel marginal frequency** ``f_i = P(i ∈ oracle mask | expert active)`` and the
   hot-set coverage curve ``E[|M ∩ H(q)|]/B`` for ``H(q)`` = top-``q`` channels by ``f``.
   The knee of that curve is the size of the always-keep bypass.
2. **The static floor**: recall of the zero-parameter predictor that ignores ``h``
   entirely. Two variants — rank by frequency, and rank by mean oracle score (the
   stronger of the two in earlier repo measurements). Every learned method must beat
   this by ≥ 10 mass-recall points to justify its parameters.
3. **Hot-set feasibility**: what a router that keeps ``H`` for free and predicts only
   ``B − |M ∩ H|`` slots would have to do — reported as the residual budget.

One GPU, no model load.
"""

import argparse
import json
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src"))

from src.channel_router import data as D  # noqa: E402
from src.channel_router.metrics import mass_recall, recall, select_topB  # noqa: E402
from src.channel_router.prep import LayerData  # noqa: E402


@torch.no_grad()
def evaluate_static(ld, stat, ratio, *, eval_tokens, batch, kind):
    """Recall of the static ranking ``stat`` (``(E*I,)``) on held-out tokens."""
    B = ld.budget(ratio)
    prof = stat.view(ld.E, ld.I)
    recs, mrecs = [], []
    idx_all = ld.take(ld.test_sl, eval_tokens)
    for s in range(0, len(idx_all), batch):
        x, sel, g, imp = ld.batch(idx_all[s:s + batch])
        sc = prof[sel].float()                                  # (T,K,I) static
        if kind.endswith("_g"):
            sc = sc * g.unsqueeze(-1)
        pred = select_topB(sc, B)
        ref = select_topB(imp, B)
        recs.append(recall(pred, ref))
        mrecs.append(mass_recall(imp, pred, ref))
    return {"recall": sum(recs) / len(recs), "mass_recall": sum(mrecs) / len(mrecs)}


@torch.no_grad()
def hot_coverage(ld, freq, ratio, sizes, *, eval_tokens, batch):
    """``E[|M ∩ H(q)|]/B`` for hot sets of the given sizes, on held-out tokens."""
    B = ld.budget(ratio)
    order = freq.argsort(descending=True)
    hot_rank = torch.empty_like(order)
    hot_rank[order] = torch.arange(order.numel(), device=order.device)
    cov = {q: 0.0 for q in sizes}
    n = 0
    idx_all = ld.take(ld.test_sl, eval_tokens)
    for s in range(0, len(idx_all), batch):
        x, sel, g, imp = ld.batch(idx_all[s:s + batch])
        keep = select_topB(imp, B)
        gid = D.global_ids(sel, ld.I)
        hr = hot_rank[gid]                                       # (T,K,I) global rank
        sel_hr = hr[keep].reshape(x.shape[0], B)                 # ranks of kept channels
        for q in sizes:
            cov[q] += float((sel_hr < q).float().sum())
        n += x.shape[0]
    return {q: cov[q] / (n * B) for q in sizes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="22,46")
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "docs/results/channel_router/data"))
    ap.add_argument("--tag", default="c4")
    ap.add_argument("--tokens", type=int, default=1 << 20)
    ap.add_argument("--ratios", default="0.25,0.125")
    ap.add_argument("--fit-tokens", type=int, default=131072)
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--hot-sizes", default="0,768,1536,3072,6144,12288,24576,49152")
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs/results/channel_router/phase0"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def log(m):
        print(f"[p4] {m}", flush=True)

    sizes = [int(v) for v in args.hot_sizes.split(",")]
    out = {"args": vars(args), "layers": {}}
    for layer in [int(v) for v in args.layers.split(",")]:
        ld = LayerData(args.data_dir, layer, tag=args.tag, tokens=args.tokens,
                       device=args.device, want_down=False)
        log(f"=== layer {layer}: N={ld.N}")
        rows, cov_all, stats_all = [], {}, {}
        for ratio in [float(v) for v in args.ratios.split(",")]:
            st = ld.channel_freq(ld.train_sl, ratio, n=args.fit_tokens,
                                 batch=args.batch)
            log(f"  rho={ratio}: fit on {st['tokens']} tokens, B={st['B']}")
            # distribution summary of the frequency prior
            fa = st["freq_active"]
            summary = {
                "frac_hot_gt50": float((fa > 0.5).float().mean()),
                "frac_hot_gt25": float((fa > 0.25).float().mean()),
                "frac_cold_lt01": float((fa < 0.01).float().mean()),
                "freq_mean": float(fa.mean()), "freq_std": float(fa.std()),
            }
            for kind, stat in (("freq", st["freq_active"]),
                               ("mean_score", st["mean_score"]),
                               ("freq_g", st["freq_active"]),
                               ("mean_score_g", st["mean_score"])):
                res = evaluate_static(ld, stat, ratio, eval_tokens=args.eval_tokens,
                                      batch=args.batch, kind=kind)
                rows.append({"ratio": ratio, "predictor": f"static_{kind}", **res})
                log(f"    static_{kind:<14} recall={res['recall']:.4f} "
                    f"mass={res['mass_recall']:.4f}")
            cov = hot_coverage(ld, st["freq_active"], ratio, sizes,
                               eval_tokens=args.eval_tokens, batch=args.batch)
            cov_all[str(ratio)] = {str(k): v for k, v in cov.items()}
            stats_all[str(ratio)] = summary
            log("    hot coverage: " + ", ".join(
                f"|H|={k}:{v:.3f}" for k, v in cov.items()))
            # marginal gain per added channel, for the knee rule
            ks = sorted(cov)
            gains = {}
            for a, b in zip(ks, ks[1:]):
                gains[f"{a}->{b}"] = (cov[b] - cov[a]) * st["B"] / max(b - a, 1)
            cov_all[str(ratio)]["marginal_gain_per_channel"] = gains
        out["layers"][str(layer)] = {"static": rows, "hot_coverage": cov_all,
                                     "freq_summary": stats_all}
        del ld
        torch.cuda.empty_cache()
    p = os.path.join(args.out_dir, "p4_static_dynamic.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {p}")


if __name__ == "__main__":
    main()
