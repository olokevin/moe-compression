#!/usr/bin/env python
"""Stage A + Stage B — structural init then supervised set distillation (§2).

Stage A is free: the whitened-SVD basis of the frozen expert projections initializes the
feature map and the channel embeddings so that the router *starts* at the training-free
whitened-SVD baseline (checked and logged — the plan's checkpoint standard). Stage B then
trains ``C``, ``C2``, ``beta``, ``Q`` against the oracle top-B with a boundary-focused
ranking loss.

Stopping rule from the plan: train until val mass-recall plateaus (< 0.1 pt/epoch) and do
not chase recall past the plateau — the oracle mask is sufficient, not necessary, so the
residual errors may be harmless; the ΔPPL ladder decides.

One GPU per layer, minutes to a couple of hours depending on token count.
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

from src.channel_router.baselines import SvdScorer, StaticFreq  # noqa: E402
from src.channel_router.metrics import router_accounting  # noqa: E402
from src.channel_router.model import ChannelRouter, stage_a_init, whiten_stats  # noqa: E402
from src.channel_router.prep import LayerData  # noqa: E402
from src.channel_router.train_utils import (  # noqa: E402
    evaluate_router, ranking_loss, save_router_artifact,
)


def outlier_coords(ld, m, source, p3_json=None):
    """Fixed coordinate set for the passthrough branch (P3's decision)."""
    if m == 0 or source == "none":
        return torch.zeros(0, dtype=torch.long)
    if source == "p3" and p3_json:
        with open(p3_json) as f:
            d = json.load(f)
        key = str(ld.layer)
        idx = d["layers"][key]["top64_by_anova_sampled"][:m]
        return torch.tensor(idx, dtype=torch.long)
    Xf = ld.X[ld.train_sl][:65536].to(ld.device, torch.float32)
    energy = (Xf ** 2).mean(0)
    del Xf
    return energy.argsort(descending=True)[:m].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="46")
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "docs/results/channel_router/data"))
    ap.add_argument("--tag", default="c4")
    ap.add_argument("--tokens", type=int, default=1 << 20)
    ap.add_argument("--ratio", type=float, default=0.125)
    # architecture
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--head", default="swiglu",
                    choices=["swiglu", "bilinear", "abs", "linear"])
    ap.add_argument("--no-bias", action="store_true")
    ap.add_argument("--no-g", action="store_true")
    ap.add_argument("--init", default="stage_a", choices=["stage_a", "random"])
    ap.add_argument("--init-source", default="gate", choices=["gate", "up", "both"])
    ap.add_argument("--bias-init", default="colnorm",
                    choices=["colnorm", "freq", "both", "zero"])
    ap.add_argument("--outlier-source", default="energy",
                    choices=["energy", "p3", "none"])
    ap.add_argument("--p3-json", default=os.path.join(
        _REPO, "docs/results/channel_router/phase0/p3_input_screening.json"))
    ap.add_argument("--hot-size", type=int, default=0)
    # optimization
    ap.add_argument("--loss", default="margin", choices=["margin", "bce", "listwise"])
    ap.add_argument("--delta", type=int, default=256)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--w-fn", type=float, default=3.0)
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr-emb", type=float, default=1e-2)
    ap.add_argument("--lr-proj", type=float, default=3e-4)
    ap.add_argument("--fit-tokens", type=int, default=0, help="0 = all train tokens")
    ap.add_argument("--eval-tokens", type=int, default=8192)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--plateau", type=float, default=0.001,
                    help="stop when val mass-recall gains < this per epoch")
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs/results/channel_router/stage_b"))
    ap.add_argument("--name", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or (f"r{args.r}m{args.m}_{args.head}_{args.loss}"
                         f"_rho{args.ratio}")

    def log(m):
        print(f"[stageB:{name}] {m}", flush=True)

    entries, report = {}, {"args": vars(args), "layers": {}}
    for layer in [int(v) for v in args.layers.split(",")]:
        ld = LayerData(args.data_dir, layer, tag=args.tag, tokens=args.tokens,
                       device=args.device, want_down=False)
        B = ld.budget(args.ratio)
        log(f"=== layer {layer}: N={ld.N} B={B}/{ld.K * ld.I}")
        _, Sh, Sinv, _ = whiten_stats(ld.X[ld.train_sl][:262144], device=args.device)

        router = ChannelRouter(ld.H, ld.E, ld.I, ld.K, r=args.r, m=args.m,
                               head=args.head, use_bias=not args.no_bias,
                               use_g=not args.no_g).to(args.device)
        oi = outlier_coords(ld, args.m, args.outlier_source, args.p3_json)
        st = ld.channel_freq(ld.train_sl, args.ratio, n=65536, batch=2048)
        if args.init == "stage_a":
            info = stage_a_init(router, ld.w, Sh, Sinv, source=args.init_source,
                                outlier_idx=oi, freq=st["freq_active"],
                                bias_init=args.bias_init, device=args.device)
        else:
            info = {"source": "random"}
            for p in (router.C, router.C2):
                if p is not None:
                    torch.nn.init.normal_(p, std=0.02)
            if args.m:
                router.outlier_idx.copy_(oi.to(args.device))
        # function-preserving feature normalization
        with torch.no_grad():
            xs = ld.X[ld.train_sl][:16384].to(args.device, torch.float32)
            router.set_feature_scale(router.features(xs).std(0))
            del xs
        if args.hot_size:
            router.set_hot(st["freq_active"], args.hot_size)

        acct = router_accounting(ld.H, ld.E, ld.I, ld.K, args.r, args.m,
                                 head=args.head, hot_size=args.hot_size,
                                 rho=args.ratio)
        log(f"  params={acct['params'] / 1e6:.3f}M "
            f"({acct['params_pct_stored_ffn']:.3f}% of stored FFN, "
            f"{acct['params_pct_active_ffn']:.2f}% of activated), "
            f"online {acct['online_flops'] / 1e6:.3f} MFLOP/token "
            f"({acct['flops_pct_saved']:.2f}% of saved)")

        # --- Stage A checkpoint standard: init must match the whitened-SVD baseline
        init_ev = evaluate_router(router, ld, ld.val_sl, args.ratio,
                                  tokens=args.eval_tokens)
        wsvd = SvdScorer(ld.w, args.r, whitened=True, Sh=Sh, Sinv=Sinv, bilinear=True,
                         prior="both", device=args.device,
                         source=("gate" if args.init_source != "up" else "up"))
        wsvd_ev = evaluate_router(None, ld, ld.val_sl, args.ratio,
                                  tokens=args.eval_tokens, scorer=wsvd)
        static = StaticFreq(ld.w, st["freq_active"], prior="none")
        static_ev = evaluate_router(None, ld, ld.val_sl, args.ratio,
                                    tokens=args.eval_tokens, scorer=static)
        log(f"  init recall={init_ev['recall']:.4f} mass={init_ev['mass_recall']:.4f} | "
            f"wsvd_r{args.r} recall={wsvd_ev['recall']:.4f} "
            f"mass={wsvd_ev['mass_recall']:.4f} | static recall={static_ev['recall']:.4f} "
            f"mass={static_ev['mass_recall']:.4f}")
        del wsvd

        # --- labels: ranked top-(B+delta) once, reused every epoch
        n_fit = args.fit_tokens or (ld.train_sl.stop - ld.train_sl.start)
        idx_fit = ld.take(ld.train_sl, n_fit)
        ext_ratio = min(1.0, (B + args.delta) / (ld.K * ld.I))
        t0 = time.time()
        topk_idx = ld.cache_topb(idx_fit, ext_ratio, batch=2048, log=None)
        log(f"  cached {topk_idx.shape} labels in {time.time() - t0:.0f}s")

        emb = [p for p in (router.C, router.C2) if p is not None]
        emb += [router.beta] if router.beta is not None else []
        opt = torch.optim.AdamW([
            {"params": emb, "lr": args.lr_emb, "weight_decay": 0.0},
            {"params": [router.Q], "lr": args.lr_proj, "weight_decay": 0.0},
        ])
        total_steps = args.epochs * max(1, len(idx_fit) // args.batch)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        best = {"mass_recall": -1.0}
        hist, step, prev_ep = [], 0, init_ev["mass_recall"]
        for ep in range(args.epochs):
            perm = torch.randperm(len(idx_fit))
            run, nb = 0.0, 0
            for s in range(0, len(idx_fit) - args.batch + 1, args.batch):
                rows = perm[s:s + args.batch]
                idx = idx_fit[rows]
                x = ld.X[idx].to(args.device, torch.float32)
                sel = ld.sel[idx].to(args.device)
                g = ld.g[idx].to(args.device)
                imp = None
                if args.loss == "listwise":
                    _, _, _, imp = ld.batch(idx)
                score = router.score(x, sel, g)
                loss = ranking_loss(score, topk_idx[rows].to(args.device), B,
                                    kind=args.loss, delta=args.delta,
                                    margin=args.margin, w_fn=args.w_fn,
                                    imp=imp, pairs=args.pairs)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for grp in opt.param_groups for p in grp["params"]], 5.0)
                opt.step()
                sched.step()
                run += float(loss.detach())
                nb += 1
                step += 1
                if step % args.eval_every == 0:
                    ev = evaluate_router(router, ld, ld.val_sl, args.ratio,
                                         tokens=args.eval_tokens)
                    hist.append({"step": step, "loss": run / max(nb, 1), **ev})
                    log(f"  ep{ep} step{step} loss={run / max(nb, 1):.4f} "
                        f"val recall={ev['recall']:.4f} mass={ev['mass_recall']:.4f}")
                    run, nb = 0.0, 0
                    if ev["mass_recall"] > best["mass_recall"]:
                        best = {**ev, "step": step,
                                "state": {k: v.detach().clone()
                                          for k, v in router.state_dict().items()}}
            ev = evaluate_router(router, ld, ld.val_sl, args.ratio,
                                 tokens=args.eval_tokens)
            gain = ev["mass_recall"] - prev_ep
            log(f"  end epoch {ep}: val mass={ev['mass_recall']:.4f} (gain {gain:+.4f})")
            prev_ep = ev["mass_recall"]
            if ev["mass_recall"] > best["mass_recall"]:
                best = {**ev, "step": step,
                        "state": {k: v.detach().clone()
                                  for k, v in router.state_dict().items()}}
            if gain < args.plateau and ep >= 1:
                log(f"  plateau reached (gain {gain:+.4f} < {args.plateau}); stopping")
                break

        router.load_state_dict(best.pop("state"))
        test_ev = evaluate_router(router, ld, ld.test_sl, args.ratio,
                                  tokens=args.eval_tokens)
        log(f"  BEST val mass={best['mass_recall']:.4f} | TEST recall="
            f"{test_ev['recall']:.4f} mass={test_ev['mass_recall']:.4f}")
        cfg = {"H": ld.H, "E": ld.E, "I": ld.I, "K": ld.K, "r": args.r, "m": args.m,
               "head": args.head, "use_bias": not args.no_bias,
               "use_g": not args.no_g, "n_tiles_per_expert": 0}
        entries[layer] = (router, cfg)
        report["layers"][str(layer)] = {
            "B": B, "accounting": acct, "init": {**init_ev, "info": info},
            "wsvd": wsvd_ev, "static": static_ev, "best_val": best,
            "test": test_ev, "history": hist,
        }
        del ld
        torch.cuda.empty_cache()

    ck = os.path.join(args.out_dir, f"router_{name}.pt")
    save_router_artifact(ck, entries, meta={"args": vars(args)})
    with open(os.path.join(args.out_dir, f"report_{name}.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"wrote {ck}")


if __name__ == "__main__":
    main()
