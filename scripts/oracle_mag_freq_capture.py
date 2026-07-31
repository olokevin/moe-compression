#!/usr/bin/env python
"""Capture oracle_mag channel activation-frequency + per-token score profiles.

Replays the Level-2 ``oracle_mag`` selection over C4 calibration tokens and
records two families of statistics, saved to a single ``.npz`` for offline
plotting (``scripts/oracle_mag_freq_plot.py``). Heavy: loads the full sharded
30B, so run on the A100 via launch-on-a100.

**The formulation (oracle_mag, from ``src/dynamic_active_param/block.py``):** per
token, each active expert's channel ``j`` is scored by its exact per-token
output magnitude

    s_{e,j}(x) = g_e * |inter_{e,j}(x)| * ||W_down[:, j]||

(``g_e`` = norm_topk routing weight, ``inter`` = SwiGLU intermediate), and the
global top-``B`` channels over the token's K active experts are kept; the rest
of the down_proj input is zeroed (masking simulation, ``real_slim=false``).

We run the **unpruned** forward (calibration reference, matching the scoring
stage and the M1 ladder) and, in a forward-pre-hook on every MoE block,
recompute ``inter`` for that block's active experts, form the score, and:

  * **Investigation 1 — activation frequency.** For each budget ``B`` (a set of
    kept-fractions ``rho``), count how often each channel ``(layer, expert, j)``
    survives the per-token global top-B (``kept_count``), and how often its
    expert fires at all (``route_count``). Frequency = kept / routed shows how
    sparsely each expert's down_proj is actually driven under oracle_mag.

  * **Investigation 2 — per-token score profiles.** Per token, sort the pooled
    ``K*I`` scores descending; accumulate the mean sorted curve per layer, keep a
    fixed sample of individual sorted curves, and record per-token concentration
    (participation ratio = effective #channels, and top-B score mass). Reveals
    whether some tokens concentrate magnitude in a few channels while others
    spread it out, and how that changes with depth.

Only tiny per-layer summaries live on device; the only arrays that scale with
#tokens are the two ``(L, T)`` concentration arrays (CPU) and a fixed-size
sample of sorted curves.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from src.base.datasets import load_datasets
from src.base.shared_utils.safe_isinstance import (
    _get_experts,
    _get_moe_block,
    _get_moe_intermediate_size,
    _get_num_hidden_layers,
    _get_topk,
)


def _build_moe_layer_map(model):
    """Ordered list of (layer_idx, moe_block) for every MoE layer (install.py order)."""
    pairs = []
    for layer_idx in range(_get_num_hidden_layers(model)):
        block = _get_moe_block(model, layer_idx)
        if _get_experts(block) is None:
            continue
        pairs.append((layer_idx, block))
    return pairs


def route_and_score(module, x, col_norm, K, I):
    """Reproduce the oracle_mag routing + per-channel score for one MoE block.

    Mirrors ``dynamic_moe_block_forward`` / ``_cross_expert_keep`` exactly (same
    softmax, top-k, norm_topk_prob, SwiGLU intermediate, and score
    ``g * |inter| * ||W_down[:, j]||``). Returns ``(g, sel, inter, flat_score)``
    with ``g,sel`` of shape ``(T,K)``, ``inter`` ``(T,K,I)``, ``flat_score``
    ``(T, K*I)`` float32.
    """
    dev = x.device
    logits = module.gate(x)
    probs = F.softmax(logits, dim=1, dtype=torch.float)
    g, sel = torch.topk(probs, module.top_k, dim=-1)             # (T,K)
    if getattr(module, "norm_topk_prob", False):
        g = g / g.sum(dim=-1, keepdim=True)
    g = g.to(torch.float32)

    T = x.shape[0]
    inter = torch.zeros((T, K, I), dtype=torch.float32, device=dev)
    expert_mask = F.one_hot(sel, num_classes=module.num_experts).permute(2, 1, 0)
    hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for eid_t in hit:
        eid = int(eid_t)
        el = module.experts[eid]
        slot, top_x = torch.where(expert_mask[eid].squeeze(0))
        cur = x[top_x]
        h = el.act_fn(el.gate_proj(cur)) * el.up_proj(cur)
        inter[top_x, slot] = h.to(torch.float32)

    col = col_norm[sel]                                          # (T,K,I)
    score = g.unsqueeze(-1) * inter.abs() * col
    return g, sel, inter, score.reshape(T, K * I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--tokens", type=int, default=131072,
                    help="approx number of calibration tokens to process")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-seqs", type=int, default=16, help="sequences per forward batch")
    ap.add_argument("--ratios", default="0.5,0.25,0.125",
                    help="kept fractions rho; B = round(rho*K*I) per budget")
    ap.add_argument("--sample-curves", type=int, default=256,
                    help="#per-token sorted-score curves to store (first tokens)")
    ap.add_argument("--dataset", default="c4")
    ap.add_argument("--out", default=os.path.join(_REPO, "docs/results/level2/oracle_mag_freq.npz"))
    ap.add_argument("--per-gpu-mem", default=os.environ.get("PER_GPU_MEM", "36GiB"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
        max_memory={i: args.per_gpu_mem for i in range(torch.cuda.device_count())},
    )
    model.eval()

    K = _get_topk(model)
    I = _get_moe_intermediate_size(model)
    KI = K * I
    ratios = [float(x) for x in args.ratios.split(",")]
    budgets = [min(max(1, int(round(r * KI))), KI) for r in ratios]

    moe_pairs = _build_moe_layer_map(model)
    L = len(moe_pairs)
    print(f"[capture] model={args.model} L(MoE)={L} K={K} I={I} K*I={KI}", flush=True)
    print(f"[capture] ratios={ratios} -> budgets B={budgets}", flush=True)

    # --- per-layer accumulators (each on its block's own device) ------------
    acc = {}
    for pos, (layer_idx, block) in enumerate(moe_pairs):
        dev = next(block.parameters()).device
        experts = _get_experts(block)
        col_norm = torch.stack(
            [e.down_proj.weight.detach().float().norm(dim=0) for e in experts], dim=0
        ).to(dev)                                            # (E, I) = ||W_down[:, j]||
        Eloc = col_norm.shape[0]
        acc[pos] = {
            "dev": dev,
            "E": Eloc,
            "col_norm": col_norm,
            "route_count": torch.zeros(Eloc, dtype=torch.float64, device=dev),
            "kept_count": [torch.zeros(Eloc, I, dtype=torch.float64, device=dev) for _ in budgets],
            "sum_sorted": torch.zeros(KI, dtype=torch.float64, device=dev),
            "n_tok": 0,
        }
    E = acc[0]["E"]

    # per-token concentration collected on CPU, one list per layer
    pr_vals = [[] for _ in range(L)]                          # participation ratio
    topB_mass = [[[] for _ in range(L)] for _ in budgets]     # [budget][layer] -> lists
    S = int(args.sample_curves)
    sample_curves = np.zeros((L, S, KI), dtype=np.float16)    # first-S global tokens
    gstate = {"seen": 0, "mask": None}                        # global offset + valid-token mask

    def make_hook(pos):
        a = acc[pos]

        def hook(module, inputs, kwargs):
            x = inputs[0] if inputs else kwargs.get("hidden_states")
            if x is None:
                return
            dev = a["dev"]
            x = x.detach().reshape(-1, x.shape[-1]).to(dev)
            # drop padding tokens so per-token stats reflect real content only
            m = gstate["mask"]
            if m is not None and m.numel() == x.shape[0]:
                x = x[m.to(dev)]
            T = x.shape[0]
            if T == 0:
                return
            g0 = gstate["seen"]

            # routing + oracle_mag score (shared helper, mirrors block.py)
            g, sel, inter, flat = route_and_score(module, x, a["col_norm"], K, I)

            a["route_count"].scatter_add_(
                0, sel.reshape(-1), torch.ones(T * K, dtype=torch.float64, device=dev)
            )

            # Investigation 1: per-budget global top-B keep counts
            for bi, B in enumerate(budgets):
                idx = torch.topk(flat, B, dim=1, sorted=False).indices
                keep = torch.zeros_like(flat, dtype=torch.bool)
                keep.scatter_(1, idx, True)
                keep = keep.reshape(T, K, I)
                kc = a["kept_count"][bi]
                for k in range(K):
                    kc.index_add_(0, sel[:, k], keep[:, k].to(torch.float64))

            # Investigation 2: per-token sorted score profile + concentration
            sorted_desc, _ = torch.sort(flat, dim=1, descending=True)   # (T,KI)
            a["sum_sorted"] += sorted_desc.sum(dim=0).to(torch.float64)
            a["n_tok"] += T
            total = sorted_desc.sum(dim=1).clamp_min(1e-30)
            sq = (sorted_desc * sorted_desc).sum(dim=1).clamp_min(1e-30)
            pr = (total * total) / sq                                   # effective #channels
            pr_vals[pos].append(pr.to("cpu", torch.float32).numpy())
            csum = torch.cumsum(sorted_desc, dim=1)
            for bi, B in enumerate(budgets):
                mass = (csum[:, B - 1] / total).to("cpu", torch.float32).numpy()
                topB_mass[bi][pos].append(mass)

            if g0 < S:
                n = min(T, S - g0)
                sample_curves[pos, g0:g0 + n] = \
                    sorted_desc[:n].to("cpu", torch.float32).numpy().astype(np.float16)

        return hook

    handles = [block.register_forward_pre_hook(make_hook(pos), with_kwargs=True)
               for pos, (_, block) in enumerate(moe_pairs)]

    in_dev = model.get_input_embeddings().weight.device

    n_seq = args.tokens // args.seq_len + 8
    ds = load_datasets(args.dataset, tok, max_samples=n_seq, max_length=args.seq_len)
    processed = 0
    with torch.no_grad():
        for i in range(0, len(ds), args.batch_seqs):
            if processed >= args.tokens:
                break
            chunk = [str(t) for t in ds[i:i + args.batch_seqs] if t]
            if not chunk:
                continue
            enc = tok(chunk, max_length=args.seq_len, padding=True,
                      truncation=True, return_tensors="pt")
            enc = {k: v.to(in_dev) for k, v in enc.items()}
            # valid-token mask (flattened) so hooks skip padding positions
            am = enc.get("attention_mask")
            gstate["mask"] = am.reshape(-1).bool().cpu() if am is not None else None
            batch_T = int(am.sum()) if am is not None else int(enc["input_ids"].numel())
            model(**enc, use_cache=False)
            gstate["seen"] += batch_T
            processed += batch_T
            if (i // args.batch_seqs) % 10 == 0:
                print(f"[capture] processed ~{processed}/{args.tokens} tokens", flush=True)

    for h in handles:
        h.remove()

    # --- assemble + save -----------------------------------------------------
    n_tok = acc[0]["n_tok"]
    print(f"[capture] done, {n_tok} tokens through each layer", flush=True)
    freq = np.zeros((len(budgets), L, E, I), dtype=np.float16)   # conditional kept/routed
    route = np.zeros((L, E), dtype=np.float32)
    mean_curve = np.zeros((L, KI), dtype=np.float32)
    for pos in range(L):
        a = acc[pos]
        rc = a["route_count"].clamp_min(1.0)
        route[pos] = a["route_count"].to("cpu", torch.float32).numpy()
        for bi in range(len(budgets)):
            f = (a["kept_count"][bi] / rc.unsqueeze(1)).to("cpu", torch.float32).numpy()
            freq[bi, pos] = f.astype(np.float16)
        mean_curve[pos] = (a["sum_sorted"] / max(1, a["n_tok"])).to("cpu", torch.float32).numpy()

    pr_arr = np.stack([np.concatenate(pr_vals[pos]) for pos in range(L)], axis=0)   # (L,Ttot)
    mass_arr = np.stack(
        [np.stack([np.concatenate(topB_mass[bi][pos]) for pos in range(L)], 0)
         for bi in range(len(budgets))], axis=0
    )                                                                               # (nb,L,Ttot)

    layer_indices = np.array([li for li, _ in moe_pairs], dtype=np.int32)
    np.savez_compressed(
        args.out,
        freq=freq, route=route, mean_curve=mean_curve,
        pr=pr_arr.astype(np.float32), topB_mass=mass_arr.astype(np.float32),
        sample_curves=sample_curves,
        ratios=np.array(ratios, dtype=np.float32),
        budgets=np.array(budgets, dtype=np.int32),
        layer_indices=layer_indices,
        K=K, I=I, E=E, L=L, KI=KI, n_tokens=n_tok, sample_n=min(S, n_tok),
        model=args.model,
    )
    print(f"[capture] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
