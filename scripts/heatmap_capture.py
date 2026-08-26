#!/usr/bin/env python
"""Capture the tensors behind the MoE heatmap figures (Qwen3-30B-A3B).

Two families of statistics are collected in a single model load and dumped to one
``.npz`` (+ a small json of token strings) for offline plotting by
``scripts/heatmap_plot.py``. Heavy: loads the full sharded 30B, so run on the
A100 via launch-on-a100.

**Part 1 — per-expert channel heatmap (selected layers).** Over a batch of
calibration tokens (WikiText-2), for each target layer in ``--p1-layers``
(default 0,11,23,47) and each expert ``e``, accumulate the mean absolute SwiGLU
intermediate per channel

    a_{e,j} = mean_{tokens t routed to e} | SiLU(gate_e . x_t) * (up_e . x_t) |_j

i.e. the average magnitude of the signal that channel ``j`` feeds into that
expert's ``down_proj`` (== ``inter`` in ``block.py``). This yields an ``(E, I)``
map per layer showing which channels each expert actually drives. Also keep the
per-expert routed-token count.

**Part 2 — per-token traces (special tokens).** Run ONE short prompt (built with
the chat template so real special tokens like ``<|im_start|>``/``<|im_end|>``
appear alongside content tokens) and, for EVERY MoE layer, record for every token
position:

  * ``hidden``  ``(L, T, H)``  — the block input ``x`` == the input of ``up_proj``
    (and of ``gate_proj``) for that token.
  * ``inter_top1`` ``(L, T, I)`` — the SwiGLU intermediate of the token's *top-1*
    routed expert == the literal input of that expert's ``down_proj``.
  * ``inter_mag`` ``(L, T, I)`` — routing-weighted magnitude across the K active
    experts, ``sum_k g_k * |inter_{e_k}|`` (a single "how hard is channel j driven
    at this token" profile over the K experts).
  * ``router_probs`` ``(L, T, E)`` — full softmax over all experts; ``sel``/``g``
    ``(L, T, K)`` — the top-k selected experts and their (norm) routing weights.

Only tiny per-layer arrays scale with anything; Part 2 is one short sequence.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from src.base.shared_utils.safe_isinstance import (  # noqa: E402
    _get_experts,
    _get_moe_block,
    _get_moe_intermediate_size,
    _get_num_hidden_layers,
    _get_topk,
)


def _build_moe_layer_map(model):
    """Ordered list of (layer_idx, moe_block) for every MoE layer."""
    pairs = []
    for layer_idx in range(_get_num_hidden_layers(model)):
        block = _get_moe_block(model, layer_idx)
        if _get_experts(block) is None:
            continue
        pairs.append((layer_idx, block))
    return pairs


def load_wikitext(tok, n_seq, seq_len):
    """WikiText-2 test split packed into fixed-length token windows."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if t and t.strip()]
    ids = tok("\n\n".join(texts), return_tensors="pt").input_ids[0]
    n = min(n_seq, ids.numel() // seq_len)
    return ids[: n * seq_len].reshape(n, seq_len)


def route_topk(module, xw):
    """Softmax router + top-k, mirroring the MoE block (norm_topk_prob aware)."""
    logits = module.gate(xw)
    probs = F.softmax(logits, dim=1, dtype=torch.float)          # (T, E)
    g, sel = torch.topk(probs, module.top_k, dim=-1)             # (T, K)
    if getattr(module, "norm_topk_prob", False):
        g = g / g.sum(dim=-1, keepdim=True)
    return probs, g.to(torch.float32), sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--p1-layers", default="0,11,23,47",
                    help="comma-list of layer indices for the per-expert heatmaps")
    ap.add_argument("--n-seq", type=int, default=256, help="Part 1 calibration sequences")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-seqs", type=int, default=16)
    ap.add_argument("--prompt",
                    default="The Eiffel Tower is in Paris. 2 + 2 = 4. Hello, world!",
                    help="Part 2 prompt (special tokens added via chat template)")
    ap.add_argument("--no-chat-template", action="store_true",
                    help="tokenize --prompt raw instead of via the chat template")
    ap.add_argument("--p1-experts", type=int, default=8,
                    help="Part 1: #most-routed experts per layer to plot individually")
    ap.add_argument("--p1-keep", type=int, default=2048,
                    help="Part 1: max #routed tokens stored per (layer, expert)")
    ap.add_argument("--p1-warm", type=int, default=64,
                    help="Part 1: #calibration seqs for the routing-count warmup pass")
    ap.add_argument("--out", default=os.path.join(_REPO, "docs/results/heatmap/heatmap.npz"))
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
    L = len(moe_pairs)
    E = int(moe_pairs[0][1].num_experts)
    H = int(model.config.hidden_size)
    pos_of = {li: p for p, (li, _) in enumerate(moe_pairs)}
    p1_layers = [int(x) for x in args.p1_layers.split(",") if int(x) in pos_of]
    print(f"[cap] model={args.model} L(MoE)={L} K={K} I={I} E={E} H={H}", flush=True)
    print(f"[cap] Part1 layers={p1_layers}", flush=True)

    in_dev = model.get_input_embeddings().weight.device
    gstate = {"mask": None}

    # ======================================================================
    # Part 1 — per-expert RAW intermediate (input of down_proj): one
    # (tokens x channels) heatmap per expert, for the top-N most-routed experts
    # of each target layer.  Pass A counts routing to pick the experts; pass B
    # stores each target's routed-token activations.
    # ======================================================================
    data = load_wikitext(tok, args.n_seq, args.seq_len)

    # --- pass A: routing counts on the target layers ------------------------
    rc = {li: torch.zeros(E, dtype=torch.long) for li in p1_layers}

    def rc_hook(li):
        def hook(module, inputs, kwargs):
            x = inputs[0] if inputs else kwargs.get("hidden_states")
            if x is None:
                return
            xw = x.detach().reshape(-1, x.shape[-1]).to(next(module.parameters()).dtype)
            _, _, sel = route_topk(module, xw)
            rc[li] += torch.bincount(sel.reshape(-1).cpu(), minlength=E)
        return hook

    handles = [moe_pairs[pos_of[li]][1].register_forward_pre_hook(
        rc_hook(li), with_kwargs=True) for li in p1_layers]
    with torch.no_grad():
        warm = min(len(data), args.p1_warm)
        for i in range(0, warm, args.batch_seqs):
            model(input_ids=data[i:i + args.batch_seqs].to(in_dev), use_cache=False)
    for h in handles:
        h.remove()

    targets, by_layer = [], {}
    for li in p1_layers:
        top = torch.topk(rc[li], min(args.p1_experts, E)).indices.tolist()
        for e in top:
            targets.append((li, e))
            by_layer.setdefault(li, []).append(e)
    print(f"[cap-p1] targets (layer, expert) = {targets}", flush=True)

    # --- pass B: store raw |inter| per routed token for each target ---------
    store = {t: {"h": [], "n": 0} for t in targets}

    def p1_hook(li):
        experts = _get_experts(moe_pairs[pos_of[li]][1])

        def hook(module, inputs, kwargs):
            x = inputs[0] if inputs else kwargs.get("hidden_states")
            if x is None:
                return
            xw = x.detach().reshape(-1, x.shape[-1]).to(next(module.parameters()).dtype)
            _, _, sel = route_topk(module, xw)                    # (T,K)
            for e in by_layer[li]:
                st = store[(li, e)]
                if st["n"] >= args.p1_keep:
                    continue
                idx = (sel == e).any(dim=1).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                cur = xw[idx]
                el = experts[e]
                h = (el.act_fn(el.gate_proj(cur)) * el.up_proj(cur)).abs()
                take = min(idx.numel(), args.p1_keep - st["n"])
                st["h"].append(h[:take].to("cpu", torch.float16))
                st["n"] += take
        return hook

    handles = [moe_pairs[pos_of[li]][1].register_forward_pre_hook(
        p1_hook(li), with_kwargs=True) for li in by_layer]
    with torch.no_grad():
        for i in range(0, len(data), args.batch_seqs):
            if all(store[t]["n"] >= args.p1_keep for t in targets):
                break
            model(input_ids=data[i:i + args.batch_seqs].to(in_dev), use_cache=False)
            if (i // args.batch_seqs) % 4 == 0:
                got = min(st["n"] for st in store.values())
                print(f"[cap-p1] {min(i + args.batch_seqs, len(data))}/{len(data)} seqs "
                      f"(min kept={got}/{args.p1_keep})", flush=True)
    for h in handles:
        h.remove()

    p1_arrays = {}
    for (li, e) in targets:
        h = torch.cat(store[(li, e)]["h"], 0).numpy() if store[(li, e)]["h"] \
            else np.zeros((0, I), np.float16)
        p1_arrays[f"p1_L{li}_e{e}"] = h
        print(f"[cap-p1] L{li} e{e}: stored {h.shape[0]} tokens", flush=True)
    p1_route = np.stack([rc[li].numpy().astype(np.float32) for li in p1_layers], 0)
    p1_targets = np.array(targets, dtype=np.int32)
    print("[cap-p1] done", flush=True)

    # ======================================================================
    # Part 2 — per-token traces over one prompt (special tokens)
    # ======================================================================
    if args.no_chat_template:
        enc = tok(args.prompt, return_tensors="pt")
    else:
        text = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(in_dev)
    T = int(input_ids.shape[1])
    tokens = tok.convert_ids_to_tokens(input_ids[0].tolist())
    special_ids = set(tok.all_special_ids)
    is_special = [int(i) in special_ids for i in input_ids[0].tolist()]
    print(f"[cap-p2] T={T} tokens; {sum(is_special)} special", flush=True)

    p2_hidden = np.zeros((L, T, H), dtype=np.float32)
    p2_inter_top1 = np.zeros((L, T, I), dtype=np.float32)
    p2_inter_mag = np.zeros((L, T, I), dtype=np.float32)
    p2_probs = np.zeros((L, T, E), dtype=np.float32)
    p2_sel = np.zeros((L, T, K), dtype=np.int32)
    p2_g = np.zeros((L, T, K), dtype=np.float32)

    def p2_hook(pos):
        def hook(module, inputs, kwargs):
            x = inputs[0] if inputs else kwargs.get("hidden_states")
            if x is None:
                return
            x2 = x.detach().reshape(-1, x.shape[-1])
            xw = x2.to(next(module.parameters()).dtype)
            probs, g, sel = route_topk(module, xw)                # (T,E),(T,K),(T,K)
            n = x2.shape[0]
            inter_mag = torch.zeros(n, I, dtype=torch.float32, device=xw.device)
            inter_top1 = torch.zeros(n, I, dtype=torch.float32, device=xw.device)
            emask = F.one_hot(sel, num_classes=module.num_experts).permute(2, 1, 0)
            hit = torch.greater(emask.sum(dim=(-1, -2)), 0).nonzero()
            for eid_t in hit:
                eid = int(eid_t)
                el = module.experts[eid]
                slot, top_x = torch.where(emask[eid].squeeze(0))
                cur = xw[top_x]
                hf = (el.act_fn(el.gate_proj(cur)) * el.up_proj(cur)).to(torch.float32)
                gw = g[top_x, slot].unsqueeze(1)
                inter_mag[top_x] += gw * hf.abs()
                is1 = slot == 0                                   # top-1 routed expert
                if is1.any():
                    inter_top1[top_x[is1]] = hf[is1]
            p2_hidden[pos] = x2.to("cpu", torch.float32).numpy()
            p2_inter_top1[pos] = inter_top1.cpu().numpy()
            p2_inter_mag[pos] = inter_mag.cpu().numpy()
            p2_probs[pos] = probs.to("cpu", torch.float32).numpy()
            p2_sel[pos] = sel.to("cpu", torch.int32).numpy()
            p2_g[pos] = g.to("cpu", torch.float32).numpy()
        return hook

    handles = [block.register_forward_pre_hook(p2_hook(pos), with_kwargs=True)
               for pos, (_, block) in enumerate(moe_pairs)]
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    for h in handles:
        h.remove()
    print("[cap-p2] done", flush=True)

    # ---- save ---------------------------------------------------------------
    layer_indices = np.array([li for li, _ in moe_pairs], dtype=np.int32)
    np.savez_compressed(
        args.out,
        p1_layers=np.array(p1_layers, dtype=np.int32),
        p1_targets=p1_targets, p1_route=p1_route,
        p2_hidden=p2_hidden, p2_inter_top1=p2_inter_top1, p2_inter_mag=p2_inter_mag,
        p2_probs=p2_probs, p2_sel=p2_sel, p2_g=p2_g,
        input_ids=input_ids[0].to("cpu").numpy().astype(np.int64),
        is_special=np.array(is_special, dtype=bool),
        layer_indices=layer_indices,
        K=K, I=I, E=E, H=H, L=L, T=T, model=args.model,
        **p1_arrays,
    )
    meta = {"model": args.model, "prompt": args.prompt,
            "used_chat_template": not args.no_chat_template,
            "tokens": tokens, "is_special": is_special,
            "K": K, "I": I, "E": E, "H": H, "L": L, "T": T,
            "p1_layers": p1_layers,
            "p1_targets": [[int(li), int(e)] for (li, e) in targets],
            "layer_indices": [int(x) for x in layer_indices]}
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[cap] saved {args.out} (+ _meta.json)", flush=True)


if __name__ == "__main__":
    main()
