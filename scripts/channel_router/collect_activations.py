#!/usr/bin/env python
"""§0.2 data collection for the channel-level router.

Captures, for one or more MoE layers, the exact tensor fed to the expert
``gate_proj``/``up_proj`` (i.e. the MLP input after the post-attention LayerNorm)
together with that layer's routing gate and all three expert weight stacks.

**Deviation from the plan's §0.2 spec, on purpose.** The spec asks for `imp`
`[N, 6144]` and `mask_idx` `[N, 768]` on disk. Storing `imp` for 1M tokens is
24 GB *per layer* in fp16, and `mask_idx` another 1.5 GB, while the oracle is a
deterministic function of `(h, W_g, W_u, W_d, gate)` that costs ~200 GFLOP per
4096-token chunk to recompute — under a second on an A100. So we store `h` plus
the weights and recompute `imp`/`mask` on the fly
(`src.channel_router.data.oracle_scores`). That keeps every study reading the
*same* oracle definition instead of a stale snapshot of it, and makes the
importance definition (§0.1: ``imp = g_e · |silu(W_g h) ⊙ (W_u h)| ·
‖W_d[:,i]‖``) a single code path.

Token blocks are built by concatenating tokenized documents and chopping into
exact ``seq_len`` blocks, so there is **no padding** and every row carries an
exact ``(seq_id, pos)`` — which P6 (temporal coherence) needs and which the
existing `probe_capture.py` captures cannot provide. Position 0 of each block is
flagged via ``pos``, so the BOS-like-token question of §0.2 can be answered by
filtering rather than by a separate dedup pass.

Run via launch-on-a100 (one model load serves all requested layers).
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from src.channel_router.data import acts_path, weights_path  # noqa: E402


def build_blocks(tok, texts, seq_len, n_tokens, log):
    """Concatenate tokenized docs (eos-separated) and chop into exact-length blocks."""
    ids = []
    total = 0
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    for i in range(0, len(texts), 256):
        batch = tok(texts[i:i + 256], add_special_tokens=False)["input_ids"]
        for seq in batch:
            ids.extend(seq)
            if eos is not None:
                ids.append(eos)
            total = len(ids)
        if total >= n_tokens + seq_len:
            break
        if i % 2048 == 0:
            log(f"  tokenized {i}/{len(texts)} docs -> {total} tokens")
    n_blocks = min(len(ids) // seq_len, -(-n_tokens // seq_len))
    if n_blocks * seq_len < n_tokens:
        log(f"[warn] only {n_blocks * seq_len} tokens available (< requested {n_tokens})")
    blocks = torch.tensor(ids[:n_blocks * seq_len], dtype=torch.long).view(n_blocks, seq_len)
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--layers", default="22,46")
    ap.add_argument("--tokens", type=int, default=1 << 20)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-seqs", type=int, default=8)
    ap.add_argument("--dataset", default="c4")
    ap.add_argument("--tag", default="", help="dataset tag in the filename (default: dataset)")
    ap.add_argument("--max-docs", type=int, default=12000)
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs/results/channel_router/data"))
    ap.add_argument("--per-gpu-mem", default=os.environ.get("PER_GPU_MEM", "36GiB"))
    ap.add_argument("--weights-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tag = args.tag or args.dataset.replace("/", "_")
    os.makedirs(args.out_dir, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x != ""]

    def log(m):
        print(f"[collect] {m}", flush=True)

    need_acts = [l for l in layers
                 if args.overwrite or not os.path.exists(acts_path(args.out_dir, l, tag, args.tokens))]
    need_w = [l for l in layers
              if args.overwrite or not os.path.exists(weights_path(args.out_dir, l))]
    if not need_acts and not need_w:
        log("nothing to do (all artifacts present)")
        return
    log(f"layers needing activations: {need_acts}; weights: {need_w}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.base.datasets import load_datasets

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = load_datasets(args.dataset, tok, max_samples=args.max_docs)
    if isinstance(texts, tuple):                     # alpaca-style loaders
        texts = [str(x["text"]) for x in texts[0]]
    texts = [str(t) for t in texts if t]
    log(f"loaded {len(texts)} docs from {args.dataset}")
    blocks = build_blocks(tok, texts, args.seq_len, args.tokens, log)
    log(f"built {blocks.shape[0]} blocks x {args.seq_len} = {blocks.numel()} tokens")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
        max_memory={i: args.per_gpu_mem for i in range(torch.cuda.device_count())},
    )
    model.eval()

    blks = {l: model.model.layers[l].mlp for l in layers}

    # ---- weights (independent of the token stream; write once per layer) ----
    wdt = getattr(torch, args.weights_dtype)
    for l in need_w:
        block = blks[l]
        payload = {
            "gate_w": block.gate.weight.data.detach().to("cpu", torch.float32).clone(),
            "Wg": torch.stack([e.gate_proj.weight.detach().cpu().to(wdt) for e in block.experts], 0),
            "Wu": torch.stack([e.up_proj.weight.detach().cpu().to(wdt) for e in block.experts], 0),
            "Wd": torch.stack([e.down_proj.weight.detach().cpu().to(wdt) for e in block.experts], 0),
            "top_k": int(block.top_k),
            "norm_topk": bool(block.norm_topk_prob),
            "model": args.model,
            "layer": l,
        }
        p = weights_path(args.out_dir, l)
        torch.save(payload, p)
        log(f"saved {p} (Wu {tuple(payload['Wu'].shape)}, Wd {tuple(payload['Wd'].shape)})")
        del payload
        gc.collect()

    if not need_acts:
        log("done (weights only)")
        return

    # ---- activations ----
    rows = {l: [] for l in need_acts}
    state = {l: 0 for l in need_acts}

    def make_hook(layer):
        def pre_hook(module, a, kw):
            x = a[0] if a else kw.get("hidden_states")
            if x is None or state[layer] >= args.tokens:
                return
            x = x.detach().reshape(-1, x.shape[-1])
            take = min(args.tokens - state[layer], x.shape[0])
            rows[layer].append(x[:take].to("cpu", torch.float16))
            state[layer] += take
        return pre_hook

    handles = [blks[l].register_forward_pre_hook(make_hook(l), with_kwargs=True)
               for l in need_acts]
    in_dev = model.get_input_embeddings().weight.device
    bs = args.batch_seqs
    t0 = time.time()
    try:
        with torch.no_grad():
            for i in range(0, blocks.shape[0], bs):
                if min(state.values()) >= args.tokens:
                    break
                ids = blocks[i:i + bs].to(in_dev)
                model(input_ids=ids, use_cache=False)
                if (i // bs) % 25 == 0:
                    done = min(state.values())
                    rate = done / max(time.time() - t0, 1e-6)
                    log(f"  {done}/{args.tokens} tokens ({rate:.0f} tok/s)")
    finally:
        for h in handles:
            h.remove()

    n_seq_tokens = args.seq_len
    for l in need_acts:
        X = torch.cat(rows[l], 0)[:args.tokens].contiguous()
        N = X.shape[0]
        pos = torch.arange(N, dtype=torch.long) % n_seq_tokens
        seq_id = torch.arange(N, dtype=torch.long) // n_seq_tokens
        payload = {
            "X": X,
            "pos": pos.to(torch.int16),
            "seq_id": seq_id.to(torch.int32),
            "meta": {
                "model": args.model, "layer": l, "dataset": args.dataset, "tag": tag,
                "seq_len": args.seq_len, "tokens": int(N), "dtype": "float16",
                "padding": "none (concatenated docs chopped to exact blocks)",
            },
        }
        p = acts_path(args.out_dir, l, tag, args.tokens)
        torch.save(payload, p)
        log(f"saved {p} (X {tuple(X.shape)})")
        rows[l] = None
        del payload, X
        gc.collect()

    with open(os.path.join(args.out_dir, f"collect_meta_{tag}.json"), "w") as f:
        json.dump({"model": args.model, "dataset": args.dataset, "tag": tag,
                   "layers": layers, "tokens": args.tokens, "seq_len": args.seq_len,
                   "n_docs": len(texts)}, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
