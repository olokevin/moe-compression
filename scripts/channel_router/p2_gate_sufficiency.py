#!/usr/bin/env python
"""P2 — gate sufficiency: what is the right distillation target?

Ranks channels by every cheap surrogate of the oracle that is computable from the same
frozen weights, and scores each against the oracle top-B:

    |silu(W_g h)|      gate only, post-activation      (the plan's candidate)
    |W_g h|            gate only, pre-activation
    |W_u h|            up only
    |silu(W_g h)·W_u h| the bilinear product           (= oracle without ‖W_d‖ and g)

each with the two free multiplicative factors switched on and off (``g_e``, ``‖W_d[:,j]‖``),
because a "target" is only well defined together with them.

Decision rule (plan): gate-only mass-recall ≥ 97% ⇒ Stage-B may distil gate scores (a
linear, smoother signal). Otherwise the target stays the full ``imp`` and the bilinear
structure has to be modelled explicitly — which is what the router's ``bilinear`` head
does.

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

import torch.nn.functional as F  # noqa: E402

from src.channel_router.metrics import mass_recall, recall, select_topB  # noqa: E402
from src.channel_router.prep import LayerData  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="22,46")
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "docs/results/channel_router/data"))
    ap.add_argument("--tag", default="c4")
    ap.add_argument("--tokens", type=int, default=1 << 20)
    ap.add_argument("--ratios", default="0.5,0.25,0.125")
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs/results/channel_router/phase0"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def log(m):
        print(f"[p2] {m}", flush=True)

    out = {"args": vars(args), "layers": {}}
    for layer in [int(v) for v in args.layers.split(",")]:
        ld = LayerData(args.data_dir, layer, tag=args.tag, tokens=args.tokens,
                       device=args.device, want_down=False)
        log(f"=== layer {layer}: N={ld.N}")
        ratios = [float(v) for v in args.ratios.split(",")]
        acc = {}
        idx_all = ld.take(ld.val_sl, args.eval_tokens)
        for s in range(0, len(idx_all), args.batch):
            idx = idx_all[s:s + args.batch]
            x, sel, g, imp, inter, gate_pre, up_out = ld.batch(idx, also_parts=True)
            cn = ld.w.col_norm[sel]
            gg = g.unsqueeze(-1)
            cands = {
                "silu_gate": F.silu(gate_pre).abs(),
                "gate_raw": gate_pre.abs(),
                "up": up_out.abs(),
                "bilinear": inter.abs(),
            }
            for name, base in cands.items():
                for use_cn in (False, True):
                    for use_g in (False, True):
                        sc = base
                        if use_cn:
                            sc = sc * cn
                        if use_g:
                            sc = sc * gg
                        key = f"{name}{'_cn' if use_cn else ''}{'_g' if use_g else ''}"
                        for ratio in ratios:
                            B = ld.budget(ratio)
                            ref = select_topB(imp, B)
                            pred = select_topB(sc, B)
                            k = (key, ratio)
                            r, m = recall(pred, ref), mass_recall(imp, pred, ref)
                            a = acc.setdefault(k, [0.0, 0.0, 0])
                            a[0] += r * x.shape[0]
                            a[1] += m * x.shape[0]
                            a[2] += x.shape[0]
        rows = [{"target": k[0], "ratio": k[1], "recall": v[0] / v[2],
                 "mass_recall": v[1] / v[2], "tokens": v[2]}
                for k, v in sorted(acc.items(), key=lambda kv: (kv[0][1], -kv[1][1]))]
        for r in rows:
            log(f"  rho={r['ratio']:<6} {r['target']:<22} recall={r['recall']:.4f} "
                f"mass={r['mass_recall']:.4f}")
        out["layers"][str(layer)] = rows
        del ld
        torch.cuda.empty_cache()
    p = os.path.join(args.out_dir, "p2_gate_sufficiency.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {p}")


if __name__ == "__main__":
    main()
