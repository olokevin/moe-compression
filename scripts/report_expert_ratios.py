#!/usr/bin/env python
"""Report the actual per-expert channel-keep ratio produced by CBA+AAR mask planning.

Given a scores_dir (Stage-1 ALA output) and the same prune_kwargs used for
train/eval, regenerate the mask and summarize how many intermediate channels
each expert keeps (the "actual ratio allocated to each expert"). Writes a JSON
summary and prints a compact table.

Usage:
  PYTHONPATH="$(pwd):$(pwd)/src" python scripts/report_expert_ratios.py \
      --scores-dir results/Qwen_..._Thinking-2507/c4/scores \
      --prune-ratio 0.25 --min-per-expert 16 --out expert_ratios.json
"""
import argparse
import json
import torch

from src.prune.generate import generate_masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--prune-ratio", type=float, default=0.25)
    p.add_argument("--intra-layer-method", default="attr_coverage")
    p.add_argument("--intra-expert-metric", default="activation")
    p.add_argument("--inter-layer-method", default="loss_coverage")
    p.add_argument("--align-inter", type=int, default=0)
    p.add_argument("--min-per-expert", type=int, default=16)
    p.add_argument("--out", default="expert_ratios.json")
    args = p.parse_args()

    prune_kwargs = {
        "prune_ratio": args.prune_ratio,
        "mask_method_kwargs": {
            "intra_layer_method": args.intra_layer_method,
            "intra_expert_metric": args.intra_expert_metric,
            "inter_layer_method": args.inter_layer_method,
        },
        "adjust_masks_kwargs": {
            "align_inter": args.align_inter,
            "min_per_expert": args.min_per_expert,
        },
    }

    res = generate_masks(
        scores_dir=args.scores_dir,
        mask_dir=None,
        prune_kwargs=prune_kwargs,
        device="cpu",
        verbose=False,
    )
    m = res["intermediate_masks"].bool()  # [L, E, I]
    L, E, I = m.shape

    kept = int(m.sum().item())
    tot = int(m.numel())
    global_keep = kept / tot

    per_expert_kept = m.sum(dim=2).float()          # [L, E]
    per_expert_ratio = per_expert_kept / I          # [L, E]

    # Per-layer summary
    per_layer = []
    for l in range(L):
        pe = per_expert_kept[l]
        per_layer.append({
            "layer": l,
            "keep_ratio": float(pe.mean().item() / I),
            "kept_min": int(pe.min().item()),
            "kept_max": int(pe.max().item()),
            "kept_mean": float(pe.mean().item()),
        })

    summary = {
        "shape": {"layers": L, "experts": E, "intermediate": I},
        "prune_kwargs": prune_kwargs,
        "global_keep_ratio": global_keep,
        "global_prune_ratio": 1.0 - global_keep,
        "per_expert_ratio_stats": {
            "min": float(per_expert_ratio.min().item()),
            "max": float(per_expert_ratio.max().item()),
            "mean": float(per_expert_ratio.mean().item()),
            "std": float(per_expert_ratio.std().item()),
        },
        "per_expert_kept_stats": {
            "min": int(per_expert_kept.min().item()),
            "max": int(per_expert_kept.max().item()),
            "mean": float(per_expert_kept.mean().item()),
        },
        # Full per-(layer,expert) kept counts, for the actual allocation table.
        "per_layer_expert_kept": per_expert_kept.to(torch.int32).tolist(),
        "per_layer_summary": per_layer,
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[report] shape L={L} E={E} I={I}")
    print(f"[report] GLOBAL keep {100*global_keep:.2f}%  prune {100*(1-global_keep):.2f}%")
    s = summary["per_expert_ratio_stats"]
    print(f"[report] per-expert keep ratio: min {100*s['min']:.1f}%  max {100*s['max']:.1f}%  "
          f"mean {100*s['mean']:.1f}%  std {100*s['std']:.1f}%")
    print(f"[report] wrote {args.out}")


if __name__ == "__main__":
    main()
