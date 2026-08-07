#!/usr/bin/env python
"""Profile a single MoE expert's SwiGLU activations for the slide 6-8 thesis.

Reproduces (on Qwen3-30B-A3B, WikiText-2) the single-expert neuron-activation
setting from the reference figure, then adds the token-specificity measurement
our thesis needs:

    "A token doesn't need 8 whole experts — it needs a sparse, token-specific
     subset of channels across those experts."

For each target (layer, expert) we replay the forward, and in a forward-pre-hook
run the router + that expert's SwiGLU to record, for every token routed to it,
the raw per-neuron activation output

    h_j(x) = SiLU(gate_j . x) * (up_j . x)          # == `inter` in block.py:81

(the exact quantity `oracle_mag` ranks by, up to the negligible ||W_down[:,j]||
factor — see the Q1 ablation). We keep up to ``--keep`` tokens' full ``(keep, I)``
matrix so all statistics — the pooled magnitude histogram (Fig-4a analog), the
per-neuron survival count at a global sparsity threshold (Fig-4b analog), and the
per-token top-B keep masks + static-vs-per-token capture (token-specificity) —
are computed offline in ``scripts/expert_activation_plot.py``.

Heavy: loads the full sharded 30B, so run on the A100 via launch-on-a100.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from src.base.shared_utils.safe_isinstance import (
    _get_experts,
    _get_moe_block,
    _get_moe_intermediate_size,
    _get_num_hidden_layers,
    _get_topk,
)


def _build_moe_layer_map(model):
    pairs = []
    for layer_idx in range(_get_num_hidden_layers(model)):
        block = _get_moe_block(model, layer_idx)
        if _get_experts(block) is None:
            continue
        pairs.append((layer_idx, block))
    return pairs


def load_wikitext(tok, n_seq, seq_len):
    """WikiText-2 test split, matching the reference; return tokenised chunks."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if t and t.strip()]
    # pack into fixed-length chunks so every "sample" is a full seq_len window
    ids = tok("\n\n".join(texts), return_tensors="pt").input_ids[0]
    n = min(n_seq, ids.numel() // seq_len)
    return ids[: n * seq_len].reshape(n, seq_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    # target (layer, expert) pairs; expert -1 means "most-routed in this batch".
    ap.add_argument("--targets", default="0:0,0:-1,24:-1,46:-1",
                    help="comma-list of layer:expert (expert -1 = most-routed)")
    ap.add_argument("--n-seq", type=int, default=2048)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--keep", type=int, default=8000,
                    help="max #routed tokens whose full (I,) activation is stored")
    ap.add_argument("--batch-seqs", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(
        _REPO, "docs/results/presentation/expert_activation.npz"))
    ap.add_argument("--per-gpu-mem", default=os.environ.get("PER_GPU_MEM", "34GiB"))
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
    moe_pairs = _build_moe_layer_map(model)
    pos_of = {li: p for p, (li, _) in enumerate(moe_pairs)}

    # parse targets; -1 experts resolved after a warmup routing pass
    want = []
    for t in args.targets.split(","):
        li, e = (int(v) for v in t.split(":"))
        want.append((li, e))
    layers = sorted({li for li, _ in want if li in pos_of})
    print(f"[cap] K={K} I={I} targets={want} layers={layers}", flush=True)

    # ---- pass 1: routing counts to resolve most-routed experts -------------
    route_count = {li: torch.zeros(len(_get_experts(moe_pairs[pos_of[li]][1])),
                                   dtype=torch.long) for li in layers}

    def route_hook(li):
        block = moe_pairs[pos_of[li]][1]

        def hook(module, inputs, kwargs):
            x = inputs[0] if inputs else kwargs.get("hidden_states")
            if x is None:
                return
            x = x.reshape(-1, x.shape[-1])
            logits = block.gate(x.to(next(block.parameters()).dtype))
            probs = F.softmax(logits, dim=1, dtype=torch.float)
            _, sel = torch.topk(probs, block.top_k, dim=-1)
            bc = torch.bincount(sel.reshape(-1).cpu(),
                                minlength=route_count[li].numel())
            route_count[li] += bc
        return hook

    data = load_wikitext(tok, args.n_seq, args.seq_len)
    in_dev = model.get_input_embeddings().weight.device
    handles = [moe_pairs[pos_of[li]][1].register_forward_pre_hook(
        route_hook(li), with_kwargs=True) for li in layers]
    with torch.no_grad():
        # a small warmup subset is enough to pick the most-routed expert
        warm = min(len(data), 256)
        for i in range(0, warm, args.batch_seqs):
            enc = data[i:i + args.batch_seqs].to(in_dev)
            model(input_ids=enc, use_cache=False)
    for h in handles:
        h.remove()

    targets = []
    for li, e in want:
        if li not in pos_of:
            continue
        if e < 0:
            e = int(route_count[li].argmax())
        targets.append((li, e))
    print(f"[cap] resolved targets (layer, expert) = {targets}", flush=True)

    # ---- pass 2: collect raw SwiGLU activations for each target ------------
    store = {t: {"h": [], "n": 0} for t in targets}
    by_layer = {}
    for (li, e) in targets:
        by_layer.setdefault(li, []).append(e)

    def act_hook(li):
        block = moe_pairs[pos_of[li]][1]
        experts = _get_experts(block)
        dtype = next(block.parameters()).dtype

        def hook(module, inputs, kwargs):
            x = inputs[0] if inputs else kwargs.get("hidden_states")
            if x is None:
                return
            x = x.reshape(-1, x.shape[-1]).to(dtype)
            logits = block.gate(x)
            probs = F.softmax(logits, dim=1, dtype=torch.float)
            _, sel = torch.topk(probs, block.top_k, dim=-1)   # (T,K)
            for e in by_layer[li]:
                st = store[(li, e)]
                if st["n"] >= args.keep:
                    continue
                mask = (sel == e).any(dim=1)
                idx = mask.nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                cur = x[idx]
                el = experts[e]
                h = (el.act_fn(el.gate_proj(cur)) * el.up_proj(cur))  # SwiGLU out
                take = min(idx.numel(), args.keep - st["n"])
                st["h"].append(h[:take].detach().to("cpu", torch.float16))
                st["n"] += take
        return hook

    handles = [moe_pairs[pos_of[li]][1].register_forward_pre_hook(
        act_hook(li), with_kwargs=True) for li in by_layer]
    with torch.no_grad():
        for i in range(0, len(data), args.batch_seqs):
            if all(store[t]["n"] >= args.keep for t in targets):
                break
            enc = data[i:i + args.batch_seqs].to(in_dev)
            model(input_ids=enc, use_cache=False)
            if (i // args.batch_seqs) % 10 == 0:
                got = {f"L{li}e{e}": store[(li, e)]["n"] for (li, e) in targets}
                print(f"[cap] batch {i}: {got}", flush=True)
    for h in handles:
        h.remove()

    # ---- save ---------------------------------------------------------------
    out = {"K": K, "I": I, "model": args.model,
           "col_norm": {}, "targets": np.array(targets, dtype=np.int32)}
    save = {}
    for (li, e) in targets:
        h = torch.cat(store[(li, e)]["h"], 0).numpy() if store[(li, e)]["h"] \
            else np.zeros((0, I), np.float16)
        save[f"h_L{li}_e{e}"] = h
        # ||W_down[:,j]|| for reference (our full score multiplies by this)
        el = _get_experts(moe_pairs[pos_of[li]][1])[e]
        save[f"colnorm_L{li}_e{e}"] = \
            el.down_proj.weight.detach().float().norm(dim=0).cpu().numpy().astype(np.float32)
        save[f"route_L{li}"] = route_count[li].numpy().astype(np.int64)
        print(f"[cap] L{li} e{e}: stored {h.shape[0]} tokens", flush=True)

    np.savez_compressed(args.out, targets=np.array(targets, dtype=np.int32),
                        K=K, I=I, model=args.model,
                        n_seq=args.n_seq, seq_len=args.seq_len, **save)
    print(f"[cap] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
