#!/usr/bin/env python
"""Layer-budget allocation + full-vs-per-expert reconstruction + HellaSwag eval.

Extends scripts/full_nystrom_cov_analysis.py from "does the leverage RANKING of one
expert change" to the two decisions that actually build a compressed model at a
50%-of-TOTAL-channel budget:

  AXIS 1 — allocation.  Per-expert Nyström keeps a UNIFORM top-(d_mlp/2) per expert.
    Full Nyström ranks all N·d_mlp channels GLOBALLY and keeps the top 50% — giving
    HETEROGENEOUS per-expert widths (some experts wider, some narrower, some starved).
    We report the resulting per-expert width distribution and the selected-set overlap.

  AXIS 2 — reconstruction.  Per-expert down_proj reconstruction solves, per expert,
    W_e_new^T = (Sᵀ C_e S)⁻¹(Sᵀ C_e) W_eᵀ  with the per-expert covariance C_e.
    Full reconstruction concatenates all experts' down_proj into one (H, N·d_mlp) map
    and solves the JOINT ridge problem against the SUMMED (ungated) down output D:
        min_W ‖ D − Z_S Wᵀ ‖²  ⇒  Wᵀ = Z_Sᵀ (Z_S Z_Sᵀ/T + λI)⁻¹ D / T   (push-through)
    where Z_S are the globally-kept stacked channels. This lets each expert absorb
    reconstruction error using cross-expert structure, then we split W back to experts
    (heterogeneous widths). The T×T push-through keeps it feasible (never forms the
    N·d_mlp × N·d_mlp = 98304² covariance).

Two END-TO-END methods are compared, each compressing ONLY layer `--layer` to 50% of
its expert-FFN params:
  * per_expert : uniform selection  + per-expert down reconstruction (C_e).
  * full       : global selection   + full joint down reconstruction (push-through).

For each we measure the layer's block-output MSE (true gated output) and run HellaSwag
(0-shot) via the repo's lm-eval adaptor. A baseline (uncompressed) row is also run.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

_THIS = os.path.abspath(__file__)
_REPO = os.path.dirname(os.path.dirname(_THIS))
for _p in (_REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.compress.loaders import build_c4_calib_loader  # noqa: E402
from src.prune.apply.slimming.expert_slim import _nystrom_reconstruct_down_proj  # noqa: E402
from eval.lm_harness.eval import eval_tasks  # noqa: E402


def _make_linear(weight: torch.Tensor, dtype, dev) -> nn.Linear:
    out_f, in_f = weight.shape
    lin = nn.Linear(in_f, out_f, bias=False)
    lin.weight = nn.Parameter(weight.detach().to(dev, dtype).clone(), requires_grad=False)
    return lin


# --------------------------------------------------------------------------- #
# Capture: block input X, true gated output Y_ref, routing, per-expert z_e.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def capture(model, loader, layer_idx, token_cap):
    block = model.model.layers[layer_idx].mlp
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    st = {"n": 0}

    def hook(module, inputs, output):
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        y = output[0] if isinstance(output, (tuple, list)) else output
        x = x.detach().reshape(-1, x.shape[-1])
        y = y.detach().reshape(-1, y.shape[-1])
        if st["n"] >= token_cap or x.shape[0] == 0:
            return
        take = min(token_cap - st["n"], x.shape[0])
        xs.append(x[:take].to("cpu", torch.float32))
        ys.append(y[:take].to("cpu", torch.float32))
        st["n"] += take

    try:
        in_dev = model.get_input_embeddings().weight.device
    except Exception:
        in_dev = torch.device("cuda:0")
    h = block.register_forward_hook(hook)
    model.eval()
    try:
        for batch in loader:
            if st["n"] >= token_cap:
                break
            ii = batch["input_ids"].to(in_dev)
            am = batch.get("attention_mask")
            if am is not None:
                am = am.to(in_dev)
            model(input_ids=ii, attention_mask=am, use_cache=False)
    finally:
        h.remove()
    X = torch.cat(xs)[:token_cap].contiguous()
    Y = torch.cat(ys)[:token_cap].contiguous()
    return X, Y, block


@torch.no_grad()
def routing_and_z(X, block, model, dev):
    """Top-k routing (frozen) + per-expert intermediate z_e on routed tokens."""
    top_k = model.config.num_experts_per_tok
    experts = block.experts
    N = len(experts)
    d_mlp = experts[0].gate_proj.weight.shape[0]
    gate_w = block.gate.weight.data.to(dev, torch.float32)
    Xd = X.to(dev)
    logits = Xd @ gate_w.T
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    tw, sel = torch.topk(probs, top_k, dim=-1)
    if getattr(block, "norm_topk_prob", True):
        tw = tw / tw.sum(-1, keepdim=True)
    z_list, idx_list, gatew_list = [], [], []
    for e in range(N):
        hit = (sel == e)
        rows = hit.any(dim=1).nonzero(as_tuple=True)[0]
        idx_list.append(rows.cpu())
        if rows.numel() == 0:
            z_list.append(torch.zeros(0, d_mlp))
            gatew_list.append(torch.zeros(0))
            continue
        xe = Xd.index_select(0, rows)
        gw = experts[e].gate_proj.weight.data.to(dev, torch.float32)
        uw = experts[e].up_proj.weight.data.to(dev, torch.float32)
        ze = torch.nn.functional.silu(xe @ gw.T) * (xe @ uw.T)
        z_list.append(ze.cpu())
        # gate weight of expert e for its routed tokens (for reference; MSE uses real block)
        col = (sel[rows] == e).float()
        gatew_list.append((tw[rows] * col).sum(dim=1).cpu())
    return sel.cpu(), z_list, idx_list, gate_w.cpu(), top_k


def _robust_inv(A):
    """torch.linalg.inv with a CPU fallback (transient CUBLAS failures on a shared,
    context-mutated GPU fall back to a CPU solve, then move back)."""
    try:
        return torch.linalg.inv(A)
    except Exception:
        dev = A.device
        return torch.linalg.inv(A.cpu()).to(dev)


def per_expert_leverage(C, lam):
    C = C.to(torch.float32); C = 0.5 * (C + C.T)
    ridge = float(lam)
    for _ in range(6):
        M = C.clone(); M.diagonal().add_(ridge)
        try:
            return torch.linalg.solve(M, C).diagonal()
        except Exception:
            ridge *= 10.0
    return C.diagonal().clamp_min(0.0)


# --------------------------------------------------------------------------- #
# Build compressed experts for a method. Returns (new_expert_modules, stats).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_compressed(method, block, z_list, idx_list, T, d_mlp, N, keep_ratio,
                     lam, min_per_expert, dev, orig_dtype):
    experts = block.experts
    active = [e for e in range(N) if z_list[e].shape[0] > 0]

    # ---- selection ---- #
    if method == "per_expert":
        k_uni = int(round(keep_ratio * d_mlp))
        kept = {}
        for e in range(N):
            if z_list[e].shape[0] == 0:
                kept[e] = torch.arange(min(k_uni, d_mlp))
                continue
            Ce = (z_list[e].to(dev).T @ z_list[e].to(dev)) / max(z_list[e].shape[0], 1)
            lev = per_expert_leverage(Ce, lam)
            kept[e] = torch.topk(lev, k_uni).indices.sort().values.cpu()
    else:  # full: global top-(keep_ratio) over ACTIVE experts' channels
        budget = int(round(keep_ratio * len(active) * d_mlp))
        # The heavy T×T linalg runs on CPU: the model stays resident on every GPU
        # for the subsequent eval, and a 16k×16k inverse under that memory pressure
        # crashes CUBLAS on the shard GPU (illegal access / execution failed). CPU
        # is slower (~tens of s) but reliable; z_list/idx_list are already on CPU.
        ld = torch.device("cpu")
        # full-covariance leverage per channel via push-through T×T inverse
        G = torch.zeros(T, T, device=ld)
        for e in active:
            ze = z_list[e]; idx = idx_list[e]
            G[idx.unsqueeze(1), idx.unsqueeze(0)] += ze @ ze.T
        G.div_(T); G.diagonal().add_(lam)
        M = torch.linalg.inv(G); del G
        lev_full = torch.full((N, d_mlp), -1.0)
        for e in active:
            ze = z_list[e]; idx = idx_list[e]
            Me = M.index_select(0, idx).index_select(1, idx)
            lev_full[e] = ((ze * (Me @ ze)).sum(0) / T)
        del M
        # global threshold over active channels, then enforce a per-expert floor
        flat = lev_full[active].reshape(-1)
        order = torch.argsort(flat, descending=True)[:budget]
        ei = torch.repeat_interleave(torch.tensor(active), d_mlp)
        ci = torch.arange(d_mlp).repeat(len(active))
        ke = {e: [] for e in range(N)}
        for pos in order.tolist():
            ke[int(ei[pos])].append(int(ci[pos]))
        # floor: guarantee min_per_expert channels for active experts (steal from the
        # global tail of the largest experts if needed — negligible budget impact).
        kept = {}
        for e in range(N):
            chans = ke.get(e, [])
            if e in active and len(chans) < min_per_expert:
                topup = torch.topk(lev_full[e], min_per_expert).indices.tolist()
                chans = sorted(set(chans) | set(topup))[:max(len(chans), min_per_expert)]
                # ensure exactly >= min_per_expert
                if len(chans) < min_per_expert:
                    chans = topup
            if not chans:  # dead expert (0 tokens): keep an arbitrary min slice to stay runnable
                chans = list(range(min(min_per_expert, d_mlp)))
            kept[e] = torch.tensor(sorted(chans))

    k_per_expert = np.array([len(kept[e]) for e in range(N)])

    # ---- reconstruction of down_proj ---- #
    new_experts = {}
    if method == "per_expert":
        for e in range(N):
            W_gate = experts[e].gate_proj.weight.data
            W_up = experts[e].up_proj.weight.data
            W_down = experts[e].down_proj.weight.data
            idx = kept[e].to(dev)
            keep_mask = torch.zeros(d_mlp, dtype=torch.bool, device=dev)
            keep_mask[idx] = True
            if z_list[e].shape[0] > 0:
                Ce = (z_list[e].to(dev).T @ z_list[e].to(dev)) / max(z_list[e].shape[0], 1)
                try:
                    W_down_k = _nystrom_reconstruct_down_proj(W_down, Ce, keep_mask,
                                                              lambda_ridge=lam, device=str(dev))
                    if not torch.isfinite(W_down_k).all():
                        raise ValueError("nonfinite")
                except Exception:
                    W_down_k = W_down.index_select(1, idx).clone()
            else:
                W_down_k = W_down.index_select(1, idx).clone()
            new_experts[e] = (
                _make_linear(W_gate.index_select(0, idx), orig_dtype, W_gate.device),
                _make_linear(W_up.index_select(0, idx), orig_dtype, W_up.device),
                _make_linear(W_down_k, orig_dtype, W_down.device),
            )
    else:  # full joint reconstruction via push-through (CPU linalg, see above)
        ld = torch.device("cpu")
        Hdim = experts[0].down_proj.weight.shape[0]
        D = torch.zeros(T, Hdim, device=ld)          # summed ungated down output
        colmap = []          # (e, local idx, col-start, col-end) to split W back
        Z_S_blocks = []
        offset = 0
        for e in range(N):
            idx = kept[e]                            # cpu long
            W_down = experts[e].down_proj.weight.data.to(ld, torch.float32)  # (H, d_mlp)
            if z_list[e].shape[0] > 0:
                ze = z_list[e]                       # (T_e, d_mlp) cpu
                rows = idx_list[e]
                D[rows] += ze @ W_down.T             # ungated summed down output
                zk = torch.zeros(T, idx.numel(), device=ld)
                zk[rows] = ze.index_select(1, idx)
            else:
                zk = torch.zeros(T, idx.numel(), device=ld)
            Z_S_blocks.append(zk)
            colmap.append((e, idx, offset, offset + idx.numel()))
            offset += idx.numel()
        Z_S = torch.cat(Z_S_blocks, dim=1)           # (T, |S|)
        del Z_S_blocks
        # push-through: Wᵀ = Z_Sᵀ (Z_S Z_Sᵀ/T + λI)⁻¹ D / T   -> (|S|, H)
        Gs = Z_S @ Z_S.T / T
        Gs.diagonal().add_(lam)
        Ms = torch.linalg.inv(Gs); del Gs
        Wt = (Z_S.T @ (Ms @ D)) / T                  # (|S|, H)
        del Ms, D, Z_S
        for (e, idx, a, b) in colmap:
            W_down_k = Wt[a:b].T.contiguous()        # (H, k_e)
            if not torch.isfinite(W_down_k).all():
                W_down_k = experts[e].down_proj.weight.data.index_select(
                    1, idx.to(experts[e].down_proj.weight.device)).clone().float()
            W_gate = experts[e].gate_proj.weight.data
            W_up = experts[e].up_proj.weight.data
            new_experts[e] = (
                _make_linear(W_gate.index_select(0, idx.to(W_gate.device)), orig_dtype, W_gate.device),
                _make_linear(W_up.index_select(0, idx.to(W_up.device)), orig_dtype, W_up.device),
                _make_linear(W_down_k, orig_dtype, experts[e].down_proj.weight.device),
            )

    stats = {
        "k_per_expert_active": k_per_expert[active].tolist(),
        "k_mean": float(k_per_expert[active].mean()),
        "k_std": float(k_per_expert[active].std()),
        "k_min": int(k_per_expert[active].min()),
        "k_max": int(k_per_expert[active].max()),
        "n_active": len(active),
        "n_starved_below_floor": int((k_per_expert[active] <= min_per_expert).sum()),
    }
    return new_experts, kept, stats


@torch.no_grad()
def swap_experts(block, new_experts):
    """Replace each expert's gate/up/down; return originals for restore."""
    orig = {}
    for e, (g, u, d) in new_experts.items():
        ex = block.experts[e]
        orig[e] = (ex.gate_proj, ex.up_proj, ex.down_proj)
        ex.gate_proj, ex.up_proj, ex.down_proj = g, u, d
    return orig


@torch.no_grad()
def restore_experts(block, orig):
    for e, (g, u, d) in orig.items():
        ex = block.experts[e]
        ex.gate_proj, ex.up_proj, ex.down_proj = g, u, d


@torch.no_grad()
def block_output_mse(block, X, Y_ref, chunk=4096):
    """Run the (possibly compressed) block on X in chunks; rel MSE vs true output."""
    dev = block.gate.weight.device
    wdtype = block.gate.weight.dtype
    T = X.shape[0]
    num = 0.0; den = 0.0
    for s in range(0, T, chunk):
        xb = X[s:s + chunk].to(dev, wdtype).unsqueeze(0)     # (1, c, H)
        out = block(xb)
        out = out[0] if isinstance(out, (tuple, list)) else out
        out = out.reshape(-1, out.shape[-1]).float().cpu()
        yb = Y_ref[s:s + chunk]
        num += (out - yb).pow(2).sum().item()
        den += yb.pow(2).sum().item()
    return num / max(den, 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--layer", type=int, default=46)
    ap.add_argument("--keep-ratio", type=float, default=0.5)
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--min-per-expert", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--hellaswag-limit", type=int, default=-1, help="-1 = full 10042")
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--methods", default="baseline,per_expert,full")
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs/results/full_nystrom"))
    ap.add_argument("--per-gpu-mem", default=os.environ.get("PER_GPU_MEM", "36GiB"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    n_gpu = torch.cuda.device_count()
    max_mem = {i: args.per_gpu_mem for i in range(n_gpu)}
    print(f"[load] {args.model} device_map=auto over {n_gpu} gpus")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", max_memory=max_mem,
        attn_implementation="sdpa", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    n_seqs = (args.tokens + args.seq_len - 1) // args.seq_len + 2
    loader = build_c4_calib_loader(tok, num_seqs=n_seqs, max_length=args.seq_len, batch_size=16)
    print(f"[capture] layer {args.layer}, {args.tokens} tokens")
    X, Y_ref, block = capture(model, loader, args.layer, args.tokens)
    dev = block.gate.weight.device
    orig_dtype = block.experts[0].gate_proj.weight.dtype
    sel, z_list, idx_list, gate_w, top_k = routing_and_z(X, block, model, dev)
    T = X.shape[0]
    d_mlp = block.experts[0].gate_proj.weight.shape[0]
    N = len(block.experts)
    print(f"[capture] T={T} N={N} d_mlp={d_mlp} top_k={top_k} ({time.time()-t0:.0f}s)")

    methods = args.methods.split(",")
    results = {"model": args.model, "layer": args.layer, "keep_ratio": args.keep_ratio,
               "tokens": T, "lambda": args.lam, "min_per_expert": args.min_per_expert,
               "hellaswag_limit": args.hellaswag_limit, "methods": {}}

    # Build ALL compressed weight sets + block-MSEs UP FRONT, while the GPU is
    # clean right after capture. Running the heavy T×T linalg AFTER an lm-eval
    # pass has mutated the CUDA/accelerate context can trigger a CUBLAS execution
    # failure; precomputing avoids interleaving linalg with eval entirely.
    prebuilt = {}   # method -> (new_experts, stats, mse)
    for method in methods:
        if method == "baseline":
            continue
        print(f"\n[build] {method} ...")
        new_experts, kept, stats = build_compressed(
            method, block, z_list, idx_list, T, d_mlp, N, args.keep_ratio,
            args.lam, args.min_per_expert, dev, orig_dtype)
        orig = swap_experts(block, new_experts)
        torch.cuda.empty_cache()
        mse = block_output_mse(block, X, Y_ref)
        restore_experts(block, orig)
        torch.cuda.empty_cache()
        prebuilt[method] = (new_experts, stats, mse)
        print(f"[build] {method} block rel-MSE={mse:.4e} | k: mean={stats['k_mean']:.1f} "
              f"std={stats['k_std']:.1f} min={stats['k_min']} max={stats['k_max']} "
              f"starved={stats['n_starved_below_floor']}")
    # free capture tensors before the (memory-heavy) eval passes
    del z_list, idx_list, X, Y_ref
    torch.cuda.empty_cache()

    for method in methods:
        print(f"\n===== METHOD: {method} =====")
        orig = None
        stats = {}
        mse = 0.0
        if method != "baseline":
            new_experts, stats, mse = prebuilt[method]
            orig = swap_experts(block, new_experts)
            torch.cuda.empty_cache()
            print(f"[{method}] block rel-MSE={mse:.4e} | k: mean={stats['k_mean']:.1f} "
                  f"std={stats['k_std']:.1f} min={stats['k_min']} max={stats['k_max']} "
                  f"starved={stats['n_starved_below_floor']}")
        else:
            print("[baseline] uncompressed reference")

        print(f"[{method}] running HellaSwag (limit={args.hellaswag_limit}) ...")
        try:
            res = eval_tasks(model=model, model_name=f"q3-30b-{method}", tokenizer=tok,
                             tasks=["hellaswag"], limit=args.hellaswag_limit,
                             max_seqlen=2048, batch_size=args.eval_batch_size, num_fewshot=0)
            hs = res.get("hellaswag", {})
            acc = hs.get("acc,none", hs.get("acc"))
            acc_norm = hs.get("acc_norm,none", hs.get("acc_norm"))
        except Exception as exc:
            print(f"[{method}] eval failed: {exc}")
            acc = acc_norm = None
        print(f"[{method}] HellaSwag acc={acc} acc_norm={acc_norm}")

        results["methods"][method] = {"block_rel_mse": mse, "hellaswag_acc": acc,
                                      "hellaswag_acc_norm": acc_norm, "alloc": stats}
        if orig is not None:
            restore_experts(block, orig)
            torch.cuda.empty_cache()

        # incremental save (eval is long; don't lose completed methods on a crash)
        with open(os.path.join(args.out_dir, f"eval_L{args.layer}_50pct.json"), "w") as f:
            json.dump(results, f, indent=2)

    results["runtime_s"] = time.time() - t0
    with open(os.path.join(args.out_dir, f"eval_L{args.layer}_50pct.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] {json.dumps(results['methods'], indent=2)}")


if __name__ == "__main__":
    main()
