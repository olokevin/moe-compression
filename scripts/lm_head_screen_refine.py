"""Screen-and-refine head: two nested sparsities, neither of them static.

Part 1 of the results doc closes "sparse reads" on the basis of a **static frequency
tier**: read the top-T frequent rows, give everything else ``-inf`` (perplexity inf,
HellaSwag at chance) or one shared tail logit (still +117% at 21.6% of reads). Two
things were wrong with that design, and they are separable:

``static``    the read set is the same at every position, so a target the tier omits is
              unrecoverable -- and a length-L continuation needs *every* token in-tier,
              which decays like coverage^L (doc section 1d).
``ungraded``  omitted rows get ``-inf`` or a single shared constant, so the head cannot
              even rank the tail approximately.

This script fixes both, and measures each fix separately so the win is attributable:

    stage 1 (screen)   coarse logits for ALL V rows from ``r0`` per-token-adaptive
                       coordinates          -> r0 * V reads  + D^2 for the rotation
    stage 2 (refine)   exact logits for the top-N rows of the coarse ranking
                                            -> N * (D - r0) additional reads
    tail               keeps its **coarse estimate** (optionally shifted by a scalar
                       calibrated on held-out states), never -inf

Nothing here is a parameter-count reduction: all V*D weights are stored. The axis is
reads/token, i.e. the plan's B1-a axis and the head's share of *active* params.

Ablations reported:
  screen=adaptive / static / freq  x  tail=coarse / -inf / shared
so "dynamic candidates" and "graded tail" are never confounded.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from scripts.lm_head_adarank_basis import make_basis
from scripts.lm_head_adarank_diag import load_head, pick_device, _print


@torch.no_grad()
def run(A, coef, colnorm, dense, r0, N, dev, screen="adaptive", tail="coarse",
        shift=0.0, freq_rank=None, chunk=128):
    """KL / top-1 / target-recall for one (screen, refine, tail) configuration."""
    n, D = coef.shape
    V = A.shape[0]
    Ad = A.to(dev, torch.float32)
    static_sel = (coef.pow(2).mean(0).sqrt() * colnorm).topk(r0).indices

    kl = agree = 0.0
    tail_mass = 0.0
    top1_in_cand = 0
    gold_err = dict(argmax=0.0, sampled=0.0, band_1e6_1e4=0.0, uniform=0.0, n=0)
    for s in range(0, n, chunk):
        cf = coef[s:s + chunk].to(dev, torch.float32)
        b = cf.shape[0]
        if screen == "adaptive":
            sel = (cf.abs() * colnorm.to(dev)).topk(r0, -1).indices
            m = torch.zeros_like(cf, dtype=torch.bool).scatter_(1, sel, True)
            ck = cf * m
        else:                                        # static subspace = low-rank sketch
            ck = torch.zeros_like(cf)
            ck[:, static_sel.to(dev)] = cf[:, static_sel.to(dev)]
        coarse = ck @ Ad.T                            # (b, V) stage-1 logits
        ld = dense[s:s + chunk].to(dev, torch.float32)

        if freq_rank is None:
            cand = coarse.topk(N, -1).indices         # dynamic candidate set
        else:
            cand = freq_rank[:N].to(dev).unsqueeze(0).expand(b, N)

        la = coarse + shift
        la.scatter_(1, cand, ld.gather(1, cand))      # stage 2: exact for candidates
        if tail == "inf":
            keep = torch.zeros_like(la, dtype=torch.bool).scatter_(1, cand, True)
            la = la.masked_fill(~keep, float("-inf"))
        elif tail == "shared":
            keep = torch.zeros_like(la, dtype=torch.bool).scatter_(1, cand, True)
            la = torch.where(keep, la, coarse.mean(-1, keepdim=True).expand_as(la))

        lpd, lpa = F.log_softmax(ld, -1), F.log_softmax(la, -1)
        pd = lpd.exp()
        # log-prob error on GOLD-style targets. HellaSwag / ARC-C do not consume the
        # argmax, they sum log p over a given continuation whose tokens can be
        # low-probability -- exactly the tokens most likely to fall outside the
        # candidate set. KL alone is dense-p-weighted and would hide that.
        dl = (lpd - lpa).abs()
        gold_err["argmax"] += float(dl.gather(1, ld.argmax(-1, keepdim=True)).sum())
        samp = torch.multinomial(pd, 1)                       # ~ p, like a real target
        gold_err["sampled"] += float(dl.gather(1, samp).sum())
        band = ((pd > 1e-6) & (pd < 1e-4)).float()
        gold_err["band_1e6_1e4"] += float((dl * band).sum() / band.sum().clamp_min(1))
        gold_err["uniform"] += float(dl.mean(-1).sum())
        gold_err["n"] += b
        # Do NOT nan_to_num the -inf terms. Zeroing them while the survivors stay
        # renormalized yields a *negative* KL -- results doc bug 3. A masked tail makes
        # KL(dense || approx) genuinely +inf, and dense_mass_outside_cand below is the
        # finite quantity that predicts the damage.
        kl += float("inf") if tail == "inf" else float((pd * (lpd - lpa)).sum())
        agree += int((ld.argmax(-1) == la.argmax(-1)).sum())
        inc = torch.zeros_like(la, dtype=torch.bool).scatter_(1, cand, True)
        top1_in_cand += int(inc.gather(1, ld.argmax(-1, keepdim=True)).sum())
        tail_mass += float(pd.masked_fill(inc, 0.0).sum())
    g = gold_err.pop("n")
    return dict(kl=kl / n, top1=agree / n, top1_in_cand=top1_in_cand / n,
                dense_mass_outside_cand=tail_mass / n,
                **{f"dlogp_{k}": v / g for k, v in gold_err.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--calib", default="calib/lm_head_qwen3_0_6b/sigma_lm_head_c4_128x16x512.pt")
    ap.add_argument("--unigram", default="calib/lm_head_qwen3_0_6b/unigram_c4_5000000.pt")
    ap.add_argument("--out", default="results_eval/lm_head_screen_refine_0_6b.json")
    ap.add_argument("--basis", default="ceig")
    ap.add_argument("--n-states", type=int, default=2048)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    dev = a.device or pick_device()
    W, _ = load_head(a.model)
    V, D = W.shape
    pay = torch.load(a.calib, map_location="cpu")
    C, H = pay["C"].float(), pay["H"].float()[: a.n_states]
    n = H.shape[0]

    dense = torch.empty(n, V, dtype=torch.float32)
    Wd = W.to(dev, torch.float32)
    for s in range(0, n, 256):
        dense[s:s + 256] = (H[s:s + 256].to(dev) @ Wd.T).cpu()
    del Wd
    torch.cuda.empty_cache()

    A, M, rotated = make_basis(a.basis, W, C, dev)
    coef = H @ M.T.float()
    colnorm = torch.zeros(D)
    for s in range(0, V, 16384):
        colnorm += A[s:s + 16384].to(dev).pow(2).sum(0).cpu()
    colnorm = colnorm.sqrt()

    freq_rank = None
    if os.path.exists(a.unigram):
        freq_rank = torch.load(a.unigram, map_location="cpu")["counts"].argsort(descending=True)

    def reads(r0, N):
        return (r0 * V + N * max(D - r0, 0) + (D * D if rotated else 0)) / (V * D)

    rows = []

    def log(tag, cfg, res, rf):
        rows.append(dict(tag=tag, **cfg, read_frac=rf, **res))
        _print(f"{tag:<34} reads={100*rf:5.2f}%  KL={res['kl']:8.4f} "
               f"(pred PPL x{torch.tensor(res['kl']).exp():7.3f})  top1={100*res['top1']:5.2f}%  "
               f"argmax-in-cand={100*res['top1_in_cand']:6.2f}%  "
               f"mass out={100*res['dense_mass_outside_cand']:6.3f}%  "
               f"|dlogp| argmax/sampled/tail={res['dlogp_argmax']:.4f}/"
               f"{res['dlogp_sampled']:.4f}/{res['dlogp_band_1e6_1e4']:.4f}")

    _print(f"\n=== screen-and-refine, basis={a.basis}, {n} states, {V}x{D} ===")
    _print("--- the proposal: adaptive screen + exact refine + coarse tail ---")
    for r0, N in ((64, 8192), (128, 8192), (192, 8192), (128, 16384),
                  (192, 16384), (256, 8192), (256, 16384)):
        rf = reads(r0, N)
        log(f"adaptive r0={r0} N={N}", dict(r0=r0, N=N, screen="adaptive", tail="coarse"),
            run(A, coef, colnorm, dense, r0, N, dev), rf)

    _print("\n--- ablation A: was it the DYNAMIC candidate set? (static-subspace screen) ---")
    for r0, N in ((192, 8192), (192, 16384)):
        log(f"static-screen r0={r0} N={N}",
            dict(r0=r0, N=N, screen="static", tail="coarse"),
            run(A, coef, colnorm, dense, r0, N, dev, screen="static"), reads(r0, N))

    _print("\n--- ablation B: was it the GRADED tail? (same candidates, -inf / shared) ---")
    for tail in ("inf", "shared"):
        log(f"adaptive r0=192 N=8192 tail={tail}",
            dict(r0=192, N=8192, screen="adaptive", tail=tail),
            run(A, coef, colnorm, dense, 192, 8192, dev, tail=tail), reads(192, 8192))

    if freq_rank is not None:
        _print("\n--- ablation C: the doc's B1-a candidate set (static frequency tier) ---")
        for N, tail in ((8192, "coarse"), (8192, "inf"), (32768, "coarse")):
            log(f"freq-tier N={N} tail={tail}",
                dict(r0=192, N=N, screen="adaptive", tail=tail, cand="freq"),
                run(A, coef, colnorm, dense, 192, N, dev, tail=tail, freq_rank=freq_rank),
                reads(192, N))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(model=a.model, V=V, D=D, basis=a.basis, n_states=n, rows=rows), f, indent=2)
    _print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
