#!/usr/bin/env python
"""Capture MoE-layer inputs **and all three expert matrices** for the scorer screens.

The captures written by ``lowrank_scorer_recall.py`` hold ``X``, ``gate_w``, ``Wg``,
``Wu`` — enough to score channel *indices* against the oracle, but not enough to
ask the question that actually matters: how much does a scorer's imperfect
selection change the **block output**? That needs ``W_down``, so this script adds
it and writes to ``capture_L{layer}_t{tokens}_wd.pt`` (a distinct name, so the
existing caches stay valid).

Why output error and not just recall: recall counts *how many* of the oracle's
top-B channels were missed; the accuracy of the model depends on *which*, weighted
by how much output each channel actually moves (``inter_j * W_down[:,j]``).
Missing a channel whose down-projection column is small is free. With ``W_down``
in hand, the screen can report relative output error, which is anchorable to the
two measured downstream numbers (``oracle_up`` 71.30 HS, ``oracle_mag_noW``
77.11 HS at rho=0.125) and therefore predicts accuracy from a 4-minute CPU/GPU
job instead of a 3-hour 4-GPU eval.

One model load for all requested layers. Run via launch-on-a100.
"""

import argparse
import gc
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def capture_path(out_dir, layer, tokens):
    return os.path.join(out_dir, f"capture_L{layer}_t{tokens}_wd.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--layers", default="6,22,38,46")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--per-gpu-mem", default=os.environ.get("PER_GPU_MEM", "36GiB"))
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                    help="storage dtype for the weight stacks (float16 halves disk)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    layers = [int(x) for x in args.layers.split(",")]
    missing = [l for l in layers if not os.path.exists(capture_path(args.out_dir, l, args.tokens))]
    if not missing:
        print(f"[capture] all {len(layers)} captures already present", flush=True)
        return
    print(f"[capture] capturing layers {missing} (one model load)", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.base.datasets import load_datasets

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
        max_memory={i: args.per_gpu_mem for i in range(torch.cuda.device_count())},
    )
    model.eval()

    rows = {l: [] for l in missing}
    state = {l: 0 for l in missing}
    gstate = {"mask": None}

    def make_hook(layer):
        def pre_hook(module, a, kw):
            x = a[0] if a else kw.get("hidden_states")
            if x is None or state[layer] >= args.tokens:
                return
            x = x.detach().reshape(-1, x.shape[-1])
            m = gstate["mask"]
            if m is not None and m.numel() == x.shape[0]:
                x = x[m.to(x.device)]
            take = min(args.tokens - state[layer], x.shape[0])
            if take > 0:
                rows[layer].append(x[:take].to("cpu", torch.float32))
                state[layer] += take
        return pre_hook

    blocks = {l: model.model.layers[l].mlp for l in missing}
    handles = [blocks[l].register_forward_pre_hook(make_hook(l), with_kwargs=True)
               for l in missing]

    in_dev = model.get_input_embeddings().weight.device
    ds = load_datasets("c4", tok, max_samples=args.tokens // args.seq_len + 16,
                       max_length=args.seq_len)
    try:
        with torch.no_grad():
            for i in range(0, len(ds), 8):
                if min(state.values()) >= args.tokens:
                    break
                chunk = [str(x) for x in ds[i:i + 8] if x]
                if not chunk:
                    continue
                enc = tok(chunk, max_length=args.seq_len, padding=True,
                          truncation=True, return_tensors="pt")
                am = enc.get("attention_mask")
                gstate["mask"] = am.reshape(-1).bool() if am is not None else None
                model(**{k: v.to(in_dev) for k, v in enc.items()}, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    wdt = getattr(torch, args.dtype)
    for l in missing:
        block = blocks[l]
        X = torch.cat(rows[l], 0)[:args.tokens].contiguous()
        payload = {
            "X": X,
            "gate_w": block.gate.weight.data.detach().to("cpu", torch.float32).clone(),
            "Wg": torch.stack([e.gate_proj.weight.detach().cpu().to(wdt) for e in block.experts], 0),
            "Wu": torch.stack([e.up_proj.weight.detach().cpu().to(wdt) for e in block.experts], 0),
            "Wd": torch.stack([e.down_proj.weight.detach().cpu().to(wdt) for e in block.experts], 0),
            "top_k": block.top_k,
            "norm_topk": block.norm_topk_prob,
        }
        p = capture_path(args.out_dir, l, args.tokens)
        torch.save(payload, p)
        print(f"[capture] saved {p} (X {tuple(X.shape)}, Wd {tuple(payload['Wd'].shape)})",
              flush=True)
        rows[l] = None
        del payload

    del model, blocks, handles
    gc.collect()
    torch.cuda.empty_cache()
    print("[capture] done", flush=True)


if __name__ == "__main__":
    main()
