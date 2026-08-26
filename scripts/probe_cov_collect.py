#!/usr/bin/env python
"""Collect per-layer MoE-input second moments ``C = E[x xᵀ]`` over many tokens.

The rotated-basis screen (``scripts/probe_rotate_screen.py``) needs an eigenbasis
of the MoE block input. Estimating one from a capture is not good enough: ``C`` is
``H×H = 2048×2048`` and a capture holds 8192 tokens, so the sample eigenbasis
overfits — ``scripts/probe_rotate_diag.py`` measures an in-sample energy gain of
+0.21 that shrinks to +0.02 held out. Whether the rotation is worth anything
therefore depends on a basis fit on enough tokens, and that is all this script
produces: no ``X`` is stored, only the ``H×H`` accumulator per layer, so 100× more
tokens costs no more memory than 8192 did.

Padding is excluded. lm_head's bug 1 was exactly this — 45.3% of a right-padded
calibration grid was padding, and it corrupted the activation metric badly, because
a pad position's hidden state points somewhere the real distribution never goes.

Writes ``{layer: C (H,H) float64, "mu": {layer: (H,)}, "n_tokens": int}``.
One model load; run under ``launch-on-a100``.
"""

import argparse
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--layers", default="all",
                    help="'all' for every MoE layer, or a comma list")
    ap.add_argument("--tokens", type=int, default=262144)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs/results/btt_dynamic"))
    ap.add_argument("--per-gpu-mem", default=os.environ.get("PER_GPU_MEM", "36GiB"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out = args.out or os.path.join(args.out_dir,
                                   f"moe_input_cov_t{args.tokens}.pt")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.base.datasets import load_datasets
    from src.base.shared_utils.safe_isinstance import _is_moe_block

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
        max_memory={i: args.per_gpu_mem for i in range(torch.cuda.device_count())},
    )
    model.eval()

    blocks = {}
    for i, layer in enumerate(model.model.layers):
        if _is_moe_block(layer.mlp):
            blocks[i] = layer.mlp
    if args.layers != "all":
        want = {int(x) for x in args.layers.split(",")}
        blocks = {i: b for i, b in blocks.items() if i in want}
    print(f"[cov] {len(blocks)} MoE layers: {sorted(blocks)[:8]}...", flush=True)

    acc, mu, seen = {}, {}, {"n": 0, "pad": 0}
    gstate = {"mask": None}

    def make_hook(L):
        def pre_hook(module, a, kw):
            x = a[0] if a else kw.get("hidden_states")
            if x is None:
                return
            x = x.detach().reshape(-1, x.shape[-1])
            m = gstate["mask"]
            if m is not None and m.numel() == x.shape[0]:
                x = x[m.to(x.device)]
            xf = x.double()
            if L not in acc:
                acc[L] = torch.zeros((xf.shape[-1], xf.shape[-1]),
                                     dtype=torch.float64, device=xf.device)
                mu[L] = torch.zeros(xf.shape[-1], dtype=torch.float64,
                                    device=xf.device)
            acc[L] += xf.t() @ xf
            mu[L] += xf.sum(dim=0)
            if L == min(blocks):
                seen["n"] += xf.shape[0]
        return pre_hook

    handles = [b.register_forward_pre_hook(make_hook(L), with_kwargs=True)
               for L, b in blocks.items()]

    in_dev = model.get_input_embeddings().weight.device
    ds = load_datasets("c4", tok, max_samples=args.tokens // args.seq_len + 64,
                       max_length=args.seq_len)
    try:
        with torch.no_grad():
            for i in range(0, len(ds), args.batch):
                if seen["n"] >= args.tokens:
                    break
                chunk = [str(x) for x in ds[i:i + args.batch] if x]
                if not chunk:
                    continue
                enc = tok(chunk, max_length=args.seq_len, padding=True,
                          truncation=True, return_tensors="pt")
                am = enc.get("attention_mask")
                if am is not None:
                    flat = am.reshape(-1).bool()
                    seen["pad"] += int((~flat).sum())
                    gstate["mask"] = flat
                else:
                    gstate["mask"] = None
                model(**{k: v.to(in_dev) for k, v in enc.items()}, use_cache=False)
                if seen["n"] % (args.batch * args.seq_len * 8) < args.batch * args.seq_len:
                    print(f"  {seen['n']}/{args.tokens} tokens "
                          f"({seen['pad']} pad positions skipped)", flush=True)
    finally:
        for h in handles:
            h.remove()

    n = max(1, seen["n"])
    payload = {"n_tokens": seen["n"], "pad_skipped": seen["pad"],
               "seq_len": args.seq_len, "model": args.model,
               "C": {L: (acc[L] / n).cpu() for L in acc},
               "mu": {L: (mu[L] / n).cpu() for L in mu}}
    torch.save(payload, out)
    print(f"[cov] {seen['n']} tokens, {seen['pad']} pad skipped -> {out}", flush=True)


if __name__ == "__main__":
    main()
