#!/usr/bin/env python
"""P6 — temporal coherence: is mask caching / delta prediction worth anything?

IoU of the oracle top-B mask between token positions at distance ``d`` inside the same
sequence (the capture stores exact ``(seq_id, pos)``, so pairs never cross a document
boundary). Also reports the recall a "reuse the previous token's mask" predictor would
get — the zero-parameter *dynamic* baseline, distinct from P4's static one.

Decision (plan): mean adjacent IoU ≥ 70% ⇒ add a reuse-previous-mask + top-up variant to
the ablation list; otherwise drop it.

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
from src.channel_router.metrics import mass_recall, select_topB  # noqa: E402
from src.channel_router.prep import LayerData  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="22,46")
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "docs/results/channel_router/data"))
    ap.add_argument("--tag", default="c4")
    ap.add_argument("--tokens", type=int, default=1 << 20)
    ap.add_argument("--ratios", default="0.25,0.125")
    ap.add_argument("--dists", default="1,2,4,8,16,32")
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs/results/channel_router/phase0"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    def log(m):
        print(f"[p6] {m}", flush=True)

    dists = [int(v) for v in args.dists.split(",")]
    out = {"args": vars(args), "layers": {}}
    for layer in [int(v) for v in args.layers.split(",")]:
        ld = LayerData(args.data_dir, layer, tag=args.tag, tokens=args.tokens,
                       device=args.device, want_down=False)
        log(f"=== layer {layer}")
        res = {}
        for ratio in [float(v) for v in args.ratios.split(",")]:
            B = ld.budget(ratio)
            idx = ld.take(ld.test_sl, args.eval_tokens)
            seq = ld.acts.seq_id[idx]
            # global channel ids of each token's kept set, in one pass
            keeps = []
            for s in range(0, len(idx), args.batch):
                x, sel, g, imp = ld.batch(idx[s:s + args.batch])
                keep = select_topB(imp, B)
                gid = D.global_ids(sel, ld.I)
                keeps.append(gid[keep].reshape(x.shape[0], B).cpu())
            kept = torch.cat(keeps, 0)                          # (n, B) global ids
            n = kept.shape[0]
            row = {}
            for d in dists:
                a = torch.arange(0, n - d)
                same = seq[a] == seq[a + d]
                a = a[same]
                if len(a) == 0:
                    continue
                ious = []
                for s in range(0, len(a), 4096):
                    ia = a[s:s + 4096]
                    A = torch.zeros((len(ia), ld.E * ld.I), dtype=torch.bool)
                    Bm = torch.zeros_like(A)
                    A.scatter_(1, kept[ia].long(), True)
                    Bm.scatter_(1, kept[ia + d].long(), True)
                    inter = (A & Bm).sum(1).float()
                    ious.append((inter / (2 * B - inter)).mean())
                row[d] = {"iou": float(torch.stack(ious).mean()),
                          "recall_reuse": float(torch.stack(ious).mean() * (2 * B) /
                                                (1 + torch.stack(ious).mean())) / B,
                          "pairs": int(len(a))}
                log(f"  rho={ratio} d={d}: IoU={row[d]['iou']:.4f} "
                    f"reuse_recall={row[d]['recall_reuse']:.4f}")
            # mass-recall of the literal "previous token's mask" predictor at d=1
            mr = []
            for s in range(0, len(idx) - 1, args.batch):
                sub = idx[s:s + args.batch]
                x, sel, g, imp = ld.batch(sub)
                ref = select_topB(imp, B)
                gid = D.global_ids(sel, ld.I)
                prev = torch.zeros((len(sub), ld.E * ld.I), dtype=torch.bool,
                                   device=ld.device)
                shifted = torch.cat([kept[s:s + 1], kept[s:s + len(sub) - 1]], 0)
                prev.scatter_(1, shifted.to(ld.device).long(), True)
                pred_flat = torch.gather(prev, 1, gid.reshape(len(sub), -1))
                pred = pred_flat.reshape(imp.shape)
                same = (ld.acts.seq_id[sub] == torch.cat(
                    [ld.acts.seq_id[sub[:1]], ld.acts.seq_id[sub[:-1]]])).to(ld.device)
                if same.any():
                    mr.append(mass_recall(imp[same], pred[same], ref[same]))
            row["prev_mask_mass_recall"] = sum(mr) / max(len(mr), 1)
            log(f"  rho={ratio} prev-mask mass_recall={row['prev_mask_mass_recall']:.4f}")
            res[str(ratio)] = {str(k): v for k, v in row.items()}
        out["layers"][str(layer)] = res
        del ld
        torch.cuda.empty_cache()
    p = os.path.join(args.out_dir, "p6_temporal.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {p}")


if __name__ == "__main__":
    main()
