#!/usr/bin/env python
"""§0.2 oracle sanity check + §0.3 calibration curve + Phase-3 end metric, one job.

Runs the real model with a channel keep-mask installed and reports perplexity, so
every claim about a mask is made in the currency the plan says decides:

- ``oracle:RHO``          exact-``imp`` top-B at ratio RHO. The §0.2 gate is
                          ΔPPL < 1% at k=768 (RHO=0.125); the ladder over several RHO
                          re-derives k honestly if the gate fails.
- ``degrade:RHO:FRAC``    oracle top-B with FRAC of its channels replaced by the
                          next-ranked ones. Pairs a *measured* mass-recall with a
                          measured ΔPPL → the §0.3 recall→PPL conversion, fitted
                          instead of assumed.
- ``router:RHO[:CKPT]``   a trained router artifact (Phase 3).

Because every spec is evaluated on the *same* windows of the same text, the
comparison is paired: the ΔPPL differences are not limited by PPL's absolute noise.

One model load serves all specs. Run via launch-on-a100 (4 GPUs for the 30B).
"""

import argparse
import json
import math
import os
import sys
import time

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src"))

from src.channel_router.scorers import install_channel_router, moe_layer_indices  # noqa: E402


def uninstall(model):
    """Restore the class forward on every MoE block (drops the instance override)."""
    from src.base.shared_utils.safe_isinstance import _get_moe_block
    for li in moe_layer_indices(model):
        blk = _get_moe_block(model, li)
        if "forward" in blk.__dict__:
            del blk.__dict__["forward"]
        for attr in ("_dyn_router",):
            if hasattr(blk, attr):
                delattr(blk, attr)


@torch.no_grad()
def ppl(model, ids, seqlen, device, max_windows=0, log=print):
    n = ids.shape[1] // seqlen
    if max_windows:
        n = min(n, max_windows)
    nlls = []
    t0 = time.time()
    for i in range(n):
        chunk = ids[:, i * seqlen:(i + 1) * seqlen].to(device)
        loss = model(chunk, labels=chunk).loss
        nlls.append(float(loss))
        if i and i % 32 == 0:
            log(f"    window {i}/{n} running_ppl={math.exp(sum(nlls) / len(nlls)):.4f} "
                f"({(time.time() - t0) / i:.2f}s/win)")
    return math.exp(sum(nlls) / len(nlls)), n


def parse_specs(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        kind = parts[0]
        if kind == "dense":
            out.append({"kind": "dense"})
        elif kind == "oracle":
            out.append({"kind": "oracle", "rho": float(parts[1])})
        elif kind == "degrade":
            out.append({"kind": "degrade", "rho": float(parts[1]),
                        "drop": float(parts[2])})
        elif kind == "router":
            # router:RHO[:SLACK] — the checkpoint comes from --router-ckpt so one job
            # can sweep the budget/slack frontier of a single trained router.
            out.append({"kind": "router", "rho": float(parts[1]),
                        "slack": float(parts[2]) if len(parts) > 2 else None})
        else:
            raise ValueError(f"bad spec {tok}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--specs", default="dense,oracle:0.5,oracle:0.25,oracle:0.125")
    ap.add_argument("--layers", default="all", help="'all' or comma list of layer ids")
    ap.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"])
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--slack", type=float, default=1.0)
    ap.add_argument("--router-ckpt", default="")
    ap.add_argument("--top-tiles", type=int, default=0)
    ap.add_argument("--no-colnorm", action="store_true",
                    help="score without the ||W_d|| factor (oracle_mag_noW variant)")
    ap.add_argument("--out", default=os.path.join(
        _REPO, "docs/results/channel_router/ppl_ladder.json"))
    ap.add_argument("--per-gpu-mem", default=os.environ.get("PER_GPU_MEM", "36GiB"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def log(m):
        print(f"[ladder] {m}", flush=True)

    specs = parse_specs(args.specs)
    log(f"{len(specs)} specs: {specs}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.compress.ppl_eval import _get_c4_valenc, _get_wikitext2_testenc

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
        max_memory={i: args.per_gpu_mem for i in range(torch.cuda.device_count())},
    )
    model.eval()
    device = model.get_input_embeddings().weight.device

    ids = (_get_wikitext2_testenc(tok) if args.dataset == "wikitext2"
           else _get_c4_valenc(tok, seqlen=args.seqlen))
    log(f"{args.dataset}: {ids.shape[1]} tokens -> {ids.shape[1] // args.seqlen} windows")

    layers = None if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    routers = None
    if any(s["kind"] == "router" for s in specs):
        from src.channel_router.train_utils import load_router_artifact
        routers = load_router_artifact(args.router_ckpt, device="cpu")
        log(f"loaded routers for layers {sorted(routers)}")

    results = {"model": args.model, "dataset": args.dataset, "seqlen": args.seqlen,
               "layers": args.layers, "slack": args.slack, "rows": []}
    dense_ppl = None
    for spec in specs:
        t0 = time.time()
        uninstall(model)
        sels = {}
        if spec["kind"] == "dense":
            tag = "dense"
        elif spec["kind"] == "oracle":
            tag = f"oracle rho={spec['rho']}"
            sels = install_channel_router(
                model, prune_ratio=1.0 - spec["rho"], mode="oracle", layers=layers,
                slack=args.slack, use_colnorm=not args.no_colnorm)
        elif spec["kind"] == "degrade":
            tag = f"degrade rho={spec['rho']} drop={spec['drop']}"
            sels = install_channel_router(
                model, prune_ratio=1.0 - spec["rho"], mode="oracle_degrade",
                drop_frac=spec["drop"], layers=layers, slack=args.slack,
                use_colnorm=not args.no_colnorm)
        else:
            slack = spec.get("slack") or args.slack
            tag = f"router rho={spec['rho']} slack={slack}"
            sels = install_channel_router(
                model, prune_ratio=1.0 - spec["rho"], mode="predict", routers=routers,
                layers=layers, slack=slack, top_tiles=args.top_tiles,
                use_colnorm=not args.no_colnorm)
        log(f"--- {tag}")
        p, nwin = ppl(model, ids, args.seqlen, device, args.max_windows, log)
        if spec["kind"] == "dense":
            dense_ppl = p
        stats = {}
        if sels:
            keys = ["mass_recall", "recall", "kept_per_token"]
            summ = [s.summary() for s in sels.values()]
            stats = {k: sum(x[k] for x in summ) / len(summ) for k in keys}
            stats["per_layer_mass_recall"] = {
                str(li): round(s.summary()["mass_recall"], 6) for li, s in sels.items()}
        row = {"spec": spec, "tag": tag, "ppl": p, "windows": nwin,
               "d_ppl_pct": (None if dense_ppl is None else
                             100.0 * (p - dense_ppl) / dense_ppl),
               "secs": round(time.time() - t0, 1), **stats}
        results["rows"].append(row)
        log(f"    PPL={p:.4f}" + (f" (Δ={row['d_ppl_pct']:+.3f}%)"
                                  if row["d_ppl_pct"] is not None else "")
            + (f" mass_recall={stats['mass_recall']:.4f}" if stats else "")
            + f" [{row['secs']}s]")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    uninstall(model)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
