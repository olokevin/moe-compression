"""Pre-study: tune the mix_btt ACTIVATION-space local-fit (lr, iters) on Qwen3-8B.

Per ``docs/results/total_param/methods/local_fit.md`` ("tune lr on a DEEP layer;
a shallow-tuned lr diverges at depth"), we pick a few ``gate_proj`` layers spanning
depth, capture their inputs ONCE (the expensive part — a full calibration sweep),
then from the same SVD warm-start sweep ``lr × iters`` and report the realized
reconstruction error per layer. This amortizes the capture across all candidates,
exactly like ``nystrom_moe``'s ``fit_lr_scan``.

Only the activation-space cells (2 & 3) need this — cell 1's weight-space fit uses
the fixed MoBE default (lr=0.07, iters=30000).

Usage (on the A100 box):
    ATTN_IMPLEMENTATION=sdpa .venv/bin/python scripts/mix_btt_fit_scan.py \
        --model Qwen/Qwen3-8B --ratio 0.67 --cell 3 \
        --out docs/results/mix_btt/fit_scan.json
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from compress.loaders import build_c4_calib_loader  # noqa: E402
from compress.btt.mix_btt import (  # noqa: E402
    capture_linear_inputs,
    decompose_to_mix_btt,
    fit_mix_btt_layer,
)


def _pick_probe_layers(n_layers, requested):
    """Return sorted, in-range probe layer indices spanning depth."""
    if requested:
        idxs = [int(x) for x in requested.split(",") if x.strip() != ""]
    else:
        # shallow / mid / deep
        idxs = [2, n_layers // 2, n_layers - 4]
    idxs = sorted({max(0, min(i, n_layers - 1)) for i in idxs})
    return idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--ratio", type=float, default=0.67, help="retain fraction (float 0<r<1)")
    ap.add_argument("--cell", type=int, default=3, choices=[2, 3],
                    help="2 = nonlin only (α fixed I); 3 = nonlin + learnable mix")
    ap.add_argument("--layers", default=None, help="comma-sep probe layer indices (default: shallow/mid/deep)")
    ap.add_argument("--lrs", default="1e-4,3e-4,1e-3,3e-3")
    ap.add_argument("--iters", default="500,1500,3000")
    ap.add_argument("--decomp_mode", default="output_one_block")
    ap.add_argument("--calib_num_seqs", type=int, default=128)
    ap.add_argument("--calib_max_length", type=int, default=2048)
    ap.add_argument("--calib_batch_size", type=int, default=2)
    ap.add_argument("--cap_tokens", type=int, default=8192)
    ap.add_argument("--snapshot_every", type=int, default=200)
    ap.add_argument("--out", default="docs/results/mix_btt/fit_scan.json")
    args = ap.parse_args()

    use_mix = args.cell == 3
    lrs = [float(x) for x in args.lrs.split(",")]
    iters_list = [int(x) for x in args.iters.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    attn_impl = os.environ.get("ATTN_IMPLEMENTATION") or None
    load_kwargs = dict(torch_dtype=torch.bfloat16)
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl
    print(f"[scan] loading {args.model} (attn={attn_impl}) ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if device == "cuda" and torch.cuda.device_count() == 1:
        model = model.to(device)
    elif device == "cuda":
        # multi-GPU: shard
        model = model.to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    n_layers = model.config.num_hidden_layers
    H = model.config.hidden_size
    I = getattr(model.config, "intermediate_size", None)
    probe = _pick_probe_layers(n_layers, args.layers)
    print(f"[scan] model: H={H} I={I} num_layers={n_layers}; probe layers={probe}")

    target_names = [f"model.layers.{i}.mlp.gate_proj" for i in probe]

    print(f"[scan] capturing inputs for {len(target_names)} gate_proj modules "
          f"(cap={args.cap_tokens}) ...")
    loader = build_c4_calib_loader(
        tok, num_seqs=args.calib_num_seqs, max_length=args.calib_max_length,
        batch_size=args.calib_batch_size, seed=3,
    )
    captured = capture_linear_inputs(model, loader, target_names, args.cap_tokens, device)

    resolved = dict(model.named_modules())
    results = {"model": args.model, "cell": args.cell, "ratio": args.ratio,
               "use_mix": use_mix, "decomp_mode": args.decomp_mode,
               "lrs": lrs, "iters": iters_list, "layers": {}}

    for name in target_names:
        X = captured.get(name)
        if X is None:
            print(f"[scan] {name}: NO captured input; skipping")
            continue
        W = resolved[name].weight.data.detach().float().cpu()
        bias = resolved[name].bias.data.detach().float().cpu() if resolved[name].bias is not None else None
        layer_out = {}
        for it in iters_list:
            for lr in lrs:
                # Fresh SVD warm-start from the SAME seed for every candidate.
                mixbtt = decompose_to_mix_btt(
                    W, rank=args.ratio, bias=bias,
                    mix_space="activation", use_mix=use_mix, use_nonlin=True,
                    decomp_mode=args.decomp_mode, device=device,
                )
                info = fit_mix_btt_layer(
                    mixbtt, X, W, bias, iters=it, lr=lr,
                    snapshot_every=args.snapshot_every, dev=device,
                    tag=f"{name}/lr{lr:g}/it{it}", log_every=max(args.snapshot_every, it),
                )
                key = f"lr{lr:g}_it{it}"
                layer_out[key] = {
                    "lr": lr, "iters": it, "rank": int(mixbtt.rank),
                    "init_mse": info["init_mse"], "final_mse": info["final_mse"],
                    "rel_init": info["rel_init"], "rel_final": info["rel_final"],
                    "rows": info["rows"],
                }
                print(f"[scan] {name} lr={lr:g} it={it} rank={mixbtt.rank}: "
                      f"rel {info['rel_init']:.4f} -> {info['rel_final']:.4f}")
                del mixbtt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        results["layers"][name] = layer_out
        captured.pop(name, None)

    # Pick the best (lr, iters) per layer and the globally-robust choice (best on
    # the DEEPEST probe layer — the binding constraint per local_fit.md).
    best_per_layer = {}
    for name, grid in results["layers"].items():
        best = min(grid.values(), key=lambda r: r["rel_final"])
        best_per_layer[name] = {"lr": best["lr"], "iters": best["iters"],
                                "rel_final": best["rel_final"]}
    results["best_per_layer"] = best_per_layer
    if results["layers"]:
        deepest = target_names[-1]
        if deepest in results["layers"]:
            dgrid = results["layers"][deepest]
            dbest = min(dgrid.values(), key=lambda r: r["rel_final"])
            results["recommended"] = {"basis": deepest, "lr": dbest["lr"],
                                      "iters": dbest["iters"], "rel_final": dbest["rel_final"]}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[scan] wrote {args.out}")
    print(f"[scan] best_per_layer: {json.dumps(best_per_layer, indent=2)}")
    if "recommended" in results:
        print(f"[scan] RECOMMENDED (deepest-layer): {results['recommended']}")


if __name__ == "__main__":
    main()
