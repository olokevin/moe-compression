#!/usr/bin/env python
"""P3 — input screening: how many coordinates of ``h`` does the decision need?

Two questions, both answered against the *full-input* oracle mask:

(a) **Truncation recall.** Zero all but ``m`` coordinates of ``h`` and push the truncated
    input through the frozen ``W_g``/``W_u`` to re-derive a selection. Three coordinate
    sets: per-token top-|h| (the upper bound for any fixed-``m`` scheme), a *global*
    fixed set ranked by ``E[h_j²]``, and a global set ranked by anchored-ANOVA main
    sensitivity. The plan's decision rule needs the **global** curve, since the router's
    outlier-passthrough branch is a fixed set of coordinates.

(b) **Anchored-ANOVA main sensitivity.** Anchor ``q = E[h]``; for coordinate ``j`` the
    main effect is the variance, over the empirical marginal of ``h_j``, of the score
    obtained by replacing only coordinate ``j`` of the anchor. Because the pre-activations
    are affine in ``h``, ``W(q + δ_j e_j) = Wq + δ_j W[:,j]`` — the sweep over all 2048
    coordinates needs no extra matmuls, only the two anchor projections. Reported both
    analytically for the linear part (``Var(h_j)·E_i[w_{ij}²]``, exact) and by sampling
    for the true bilinear score.

Decision (plan): if ``m ≤ 64`` global coordinates reach ≥ 90% of the full-input recall,
enable the outlier-passthrough branch with that coordinate set; otherwise the router
input is the low-rank projection only.

One GPU, no model load.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src"))

from src.channel_router import data as D  # noqa: E402
from src.channel_router.metrics import mass_recall, recall, select_topB  # noqa: E402
from src.channel_router.prep import LayerData  # noqa: E402


@torch.no_grad()
def sparsify_topm(x, m):
    """Keep each token's ``m`` largest-|value| coordinates, zero the rest."""
    if m >= x.shape[1]:
        return x
    idx = x.abs().topk(m, dim=1).indices
    out = torch.zeros_like(x)
    return out.scatter_(1, idx, x.gather(1, idx))


@torch.no_grad()
def sparsify_global(x, keep_idx):
    out = torch.zeros_like(x)
    out[:, keep_idx] = x[:, keep_idx]
    return out


@torch.no_grad()
def anova_linear(ld, var_h):
    """Analytic main sensitivity of the *linear* score: ``Var(h_j)·E_{e,i}[w_{e,i,j}²]``."""
    acc = torch.zeros(ld.H, device=ld.device)
    for e in range(ld.E):
        acc += (ld.w.Wg[e].float() ** 2).mean(0) + (ld.w.Wu[e].float() ** 2).mean(0)
    return var_h * acc / 2.0


@torch.no_grad()
def anova_sampled(ld, x, q, *, experts, chunk_j=32):
    """Sampled main sensitivity of the true bilinear score at the anchor ``q``."""
    d = x - q.unsqueeze(0)                                    # (T, H)
    H, I = ld.H, ld.I
    out = torch.zeros(H, device=ld.device)
    for e in experts:
        Wg, Wu = ld.w.Wg[e].float(), ld.w.Wu[e].float()       # (I,H)
        bg, bu = q @ Wg.t(), q @ Wu.t()                       # (I,)
        cn = ld.w.col_norm[e].float()                         # (I,)
        for j0 in range(0, H, chunk_j):
            j1 = min(j0 + chunk_j, H)
            dj = d[:, j0:j1].unsqueeze(-1)                    # (T, c, 1)
            gate = bg.view(1, 1, I) + dj * Wg[:, j0:j1].t().unsqueeze(0)
            up = bu.view(1, 1, I) + dj * Wu[:, j0:j1].t().unsqueeze(0)
            s = (F.silu(gate) * up).abs() * cn.view(1, 1, I)  # (T, c, I)
            out[j0:j1] += s.var(dim=0).mean(dim=-1)
    return out / max(len(experts), 1)


@torch.no_grad()
def truncation_curve(ld, ratio, ms, *, mode, keep_sets, eval_tokens, batch):
    """recall/mass-recall of the selection derived from a truncated input."""
    B = ld.budget(ratio)
    res = {}
    idx_all = ld.take(ld.val_sl, eval_tokens)
    for m in ms:
        recs, mrecs = [], []
        for s in range(0, len(idx_all), batch):
            idx = idx_all[s:s + batch]
            x, sel, g, imp = ld.batch(idx)
            xs = (sparsify_topm(x, m) if mode == "per_token"
                  else sparsify_global(x, keep_sets[m]))
            sc = D.oracle_scores(xs, sel, g, ld.w)             # same oracle, sparse input
            pred = select_topB(sc, B)
            ref = select_topB(imp, B)
            recs.append(recall(pred, ref))
            mrecs.append(mass_recall(imp, pred, ref))
        res[m] = {"recall": sum(recs) / len(recs),
                  "mass_recall": sum(mrecs) / len(mrecs)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="22,46")
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "docs/results/channel_router/data"))
    ap.add_argument("--tag", default="c4")
    ap.add_argument("--tokens", type=int, default=1 << 20)
    ap.add_argument("--ratios", default="0.125")
    ap.add_argument("--ms", default="8,16,32,64,128,512,2048")
    ap.add_argument("--fit-tokens", type=int, default=65536)
    ap.add_argument("--eval-tokens", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--anova-experts", type=int, default=4)
    ap.add_argument("--anova-tokens", type=int, default=512)
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs/results/channel_router/phase0"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    def log(m):
        print(f"[p3] {m}", flush=True)

    ms = [int(v) for v in args.ms.split(",")]
    out = {"args": vars(args), "layers": {}}
    for layer in [int(v) for v in args.layers.split(",")]:
        ld = LayerData(args.data_dir, layer, tag=args.tag, tokens=args.tokens,
                       device=args.device, want_down=False)
        log(f"=== layer {layer}")
        Xf = ld.X[ld.train_sl][:args.fit_tokens].to(ld.device, torch.float32)
        q = Xf.mean(0)
        var_h = Xf.var(0)
        energy = (Xf ** 2).mean(0)
        an_lin = anova_linear(ld, var_h)
        an_smp = anova_sampled(ld, Xf[:args.anova_tokens], q,
                               experts=list(range(0, ld.E,
                                                  max(1, ld.E // args.anova_experts))))
        del Xf
        torch.cuda.empty_cache()
        rank = {
            "energy": energy.argsort(descending=True),
            "anova_linear": an_lin.argsort(descending=True),
            "anova_sampled": an_smp.argsort(descending=True),
        }
        # overlap between the three global rankings (are they the same coordinates?)
        ov = {}
        for a in rank:
            for b in rank:
                if a < b:
                    for m in (16, 64):
                        sa = set(rank[a][:m].tolist())
                        sb = set(rank[b][:m].tolist())
                        ov[f"{a}|{b}|top{m}"] = len(sa & sb) / m
        layer_out = {"anova_overlap": ov, "curves": {}}
        for ratio in [float(v) for v in args.ratios.split(",")]:
            layer_out["curves"][str(ratio)] = {}
            pt = truncation_curve(ld, ratio, ms, mode="per_token", keep_sets=None,
                                  eval_tokens=args.eval_tokens, batch=args.batch)
            layer_out["curves"][str(ratio)]["per_token"] = pt
            log(f"  rho={ratio} per-token: " + ", ".join(
                f"m={k}:{v['mass_recall']:.3f}" for k, v in pt.items()))
            for name, order in rank.items():
                ks = {m: order[:m] for m in ms}
                cur = truncation_curve(ld, ratio, ms, mode="global", keep_sets=ks,
                                       eval_tokens=args.eval_tokens, batch=args.batch)
                layer_out["curves"][str(ratio)][f"global_{name}"] = cur
                log(f"  rho={ratio} global[{name}]: " + ", ".join(
                    f"m={k}:{v['mass_recall']:.3f}" for k, v in cur.items()))
        layer_out["top64_by_anova_sampled"] = rank["anova_sampled"][:64].tolist()
        layer_out["top64_by_energy"] = rank["energy"][:64].tolist()
        out["layers"][str(layer)] = layer_out
        del ld
        torch.cuda.empty_cache()
        with open(os.path.join(args.out_dir, "p3_input_screening.json"), "w") as f:
            json.dump(out, f, indent=2)
    log(f"wrote {os.path.join(args.out_dir, 'p3_input_screening.json')}")


if __name__ == "__main__":
    main()
