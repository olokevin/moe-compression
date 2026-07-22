#!/usr/bin/env python
"""Full cross-expert Nyström covariance analysis for one MoE layer.

MOTIVATION
----------
The current `nystrom_moe` compressor treats every routed expert INDEPENDENTLY:
it collects a per-expert down_proj-input covariance ``C_e = z_eᵀz_e / T_e``
(``d_mlp x d_mlp`` = 768x768) and ranks that expert's channels by the ridge
leverage ``diag((C_e+λI)⁻¹ C_e)``.

An alternative view stacks ALL experts into one big layer: the down_proj input is
a vector of dimension ``N·d_mlp = 128·768 = 98304``, and each token contributes a
SPARSE outer product (only its top-8 experts are non-zero). The full covariance
``C_full = Zᵀ Z / T`` is ``98304 x 98304`` and contains cross-expert (off-diagonal)
blocks that the per-expert method throws away.

This script, for a single layer (default 46 of Qwen3-30B-A3B):

  1. Captures the MoE block input X on calibration data (c4), recomputes the
     router's top-8 routing exactly, and reconstructs each expert's intermediate
     activation ``z_e = silu(gate_e X) ⊙ (up_e X)`` for its routed tokens — giving
     exact global token indexing (avoids reverse-engineering HF's expert dispatch
     permutation).

  2. ENERGY: measures how much of ``‖C_full‖_F²`` lives OFF the diagonal blocks.
     Uses the trace identity ``‖ZᵀZ‖_F² = ‖ZZᵀ‖_F² = ‖G‖_F²`` (G = TxT token Gram)
     so the 38 GB matrix is never formed. Also builds the 128x128 block-energy
     matrix ``B[e,f] = ‖Z_eᵀ Z_f‖_F²`` for a heatmap.

  3. LEVERAGE: computes the TRUE full-covariance ridge leverage for every channel
     via the push-through identity
         diag((C_full+λI)⁻¹ C_full)_i = a_iᵀ (AAᵀ+λI)⁻¹ a_i ,   A = Z/√T,
     which collapses the ``98304³`` solve into ONE ``TxT`` inverse M=(AAᵀ+λI)⁻¹.
     Because channel i of expert e is supported only on e's routed tokens, the
     per-channel score reduces to a slice ``M_e = M[idx_e, idx_e]``:
         τ_{e,i} = (1/T) · z_{e,·i}ᵀ M_e z_{e,·i}.

  4. VALIDATION: on an 8-expert sub-system it forms the dense
     ``C_sub`` (6144x6144) and compares the direct ``diag((C+λI)⁻¹C)`` against the
     push-through τ — confirming the identity/implementation numerically.

  5. COMPARISON: full-covariance leverage vs the current per-expert (block-diag)
     leverage — per-expert top-k overlap, Spearman rank correlation, and how much
     the selected channel set would change.

All heavy linear algebra is done on ONE GPU after the model is freed, so only the
capture step needs the sharded 30B model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

# Repo imports: repo root + src on path (matches the rest of the codebase).
_THIS = os.path.abspath(__file__)
_REPO = os.path.dirname(os.path.dirname(_THIS))
for _p in (_REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.compress.loaders import build_c4_calib_loader  # noqa: E402


# --------------------------------------------------------------------------- #
# Leverage helpers (self-contained; mirror src/calibration/.../leverage.py and
# the robust escalating-ridge variant in src/compress/moe_basis/nystrom_moe.py).
# --------------------------------------------------------------------------- #
def ridge_leverage_direct(C: torch.Tensor, lam: float) -> torch.Tensor:
    """diag((C+λI)⁻¹ C) with escalating-ridge fallback on a singular solve."""
    C = C.to(torch.float32)
    C = 0.5 * (C + C.T)
    ridge = float(lam)
    for _ in range(6):
        M = C.clone()
        M.diagonal().add_(ridge)
        try:
            return torch.linalg.solve(M, C).diagonal()
        except Exception:
            ridge *= 10.0
    return C.diagonal().clamp_min(0.0)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation (Pearson on ranks), no scipy dependency."""
    a = a.double()
    b = b.double()
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return float((ra @ rb) / denom)


# --------------------------------------------------------------------------- #
# Step 1 — capture MoE block input X at the target layer.
# --------------------------------------------------------------------------- #
def capture_layer_input(model, loader, layer_idx: int, token_cap: int):
    """Run the (sharded) model over the loader, capturing layer `layer_idx`'s MoE
    block input hidden states, capped at `token_cap` rows. Returns (X cpu fp32,
    gate_w cpu fp32, experts_ref) where experts_ref is the live expert ModuleList."""
    block = model.model.layers[layer_idx].mlp
    rows: List[torch.Tensor] = []
    state = {"n": 0}

    def pre_hook(module, args, kwargs):
        x = args[0] if args else kwargs.get("hidden_states")
        if x is None:
            return
        x = x.detach()
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        if state["n"] >= token_cap:
            return
        take = min(token_cap - state["n"], x.shape[0])
        rows.append(x[:take].to("cpu", dtype=torch.float32))
        state["n"] += take

    # Sharded (device_map="auto") models: inputs must land on the input-embedding's
    # device; `model.device` is unreliable when hf_device_map is set.
    try:
        in_dev = model.get_input_embeddings().weight.device
    except Exception:
        in_dev = torch.device("cuda:0")

    h = block.register_forward_pre_hook(pre_hook, with_kwargs=True)
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                if state["n"] >= token_cap:
                    break
                input_ids = batch["input_ids"].to(in_dev)
                am = batch.get("attention_mask")
                if am is not None:
                    am = am.to(in_dev)
                model(input_ids=input_ids, attention_mask=am, use_cache=False)
    finally:
        h.remove()

    X = torch.cat(rows, dim=0)[:token_cap].contiguous()
    gate_w = block.gate.weight.data.detach().to("cpu", torch.float32).clone()
    return X, gate_w, block.experts


def compute_expert_z(X_dev, experts, idx_e, eid, dev):
    """z_e = silu(X_e @ gate_eᵀ) ⊙ (X_e @ up_eᵀ) for expert eid's routed tokens."""
    xe = X_dev.index_select(0, idx_e)
    gw = experts[eid].gate_proj.weight.data.to(dev, torch.float32)
    uw = experts[eid].up_proj.weight.data.to(dev, torch.float32)
    h = torch.nn.functional.silu(xe @ gw.T) * (xe @ uw.T)  # (T_e, d_mlp)
    return h


# --------------------------------------------------------------------------- #
# Main analysis.
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--layer", type=int, default=46)
    ap.add_argument("--tokens", type=int, default=32768, help="TxT Gram token budget")
    ap.add_argument("--lam", type=float, default=1.0, help="ridge lambda (pipeline default)")
    ap.add_argument("--keep", type=int, default=512, help="kept channels/expert (k) for overlap")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "docs/results/full_nystrom"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-gpu-mem", default=os.environ.get("PER_GPU_MEM", "36GiB"))
    ap.add_argument("--no-model", action="store_true",
                    help="skip model load; reuse cached capture.pt in out-dir")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)
    t0 = time.time()
    cap_path = os.path.join(args.out_dir, f"capture_L{args.layer}.pt")

    # ---- Step 1: capture (or reuse) ---------------------------------------- #
    if args.no_model and os.path.exists(cap_path):
        print(f"[capture] reusing {cap_path}")
        blob = torch.load(cap_path, map_location="cpu")
        X, gate_w = blob["X"], blob["gate_w"]
        z_list = blob["z_list"]          # list of (T_e, d_mlp) fp32 cpu
        idx_list = blob["idx_list"]      # list of long cpu
        d_mlp = blob["d_mlp"]
        N = blob["N"]
    else:
        from transformers import AutoModelForCausalLM
        n_gpu = torch.cuda.device_count()
        max_mem = {i: args.per_gpu_mem for i in range(n_gpu)}
        print(f"[load] {args.model} device_map=auto over {n_gpu} gpus (cap {args.per_gpu_mem})")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            max_memory=max_mem, attn_implementation="sdpa", trust_remote_code=True,
        )
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        n_seqs = (args.tokens + args.seq_len - 1) // args.seq_len + 2
        loader = build_c4_calib_loader(tok, num_seqs=n_seqs, max_length=args.seq_len, batch_size=16)
        print(f"[capture] layer {args.layer}, target {args.tokens} tokens ({n_seqs} c4 windows)")
        X, gate_w, experts = capture_layer_input(model, loader, args.layer, args.tokens)
        d_mlp = experts[0].gate_proj.weight.shape[0]
        N = len(experts)
        print(f"[capture] X={tuple(X.shape)} d_mlp={d_mlp} N={N} ({time.time()-t0:.0f}s)")

        # Routing: top-8 experts per token (softmax monotonic -> topk of logits).
        top_k = model.config.num_experts_per_tok
        Xd = X.to(dev)
        logits = Xd @ gate_w.to(dev).T
        sel = torch.topk(logits, top_k, dim=-1).indices  # (T, top_k)
        idx_list, z_list = [], []
        for e in range(N):
            idx_e = (sel == e).any(dim=1).nonzero(as_tuple=True)[0].to(dev)
            idx_list.append(idx_e.cpu())
            if idx_e.numel() == 0:
                z_list.append(torch.zeros(0, d_mlp))
                continue
            z_e = compute_expert_z(Xd, experts, idx_e, e, dev)
            z_list.append(z_e.cpu())
        del Xd, model
        torch.cuda.empty_cache()
        torch.save({"X": X, "gate_w": gate_w, "z_list": z_list, "idx_list": idx_list,
                    "d_mlp": d_mlp, "N": N}, cap_path)
        print(f"[capture] saved {cap_path}")

    T = X.shape[0]
    Te = torch.tensor([z.shape[0] for z in z_list])
    print(f"[dims] T={T} N={N} d_mlp={d_mlp} | tokens/expert min={Te.min()} "
          f"mean={Te.float().mean():.0f} max={Te.max()}")

    # ---- Step 2: energy — build TxT token Gram G = sum_e Z_e Z_eᵀ ---------- #
    G = torch.zeros(T, T, dtype=torch.float32, device=dev)
    diag_energy = 0.0
    for e in range(N):
        if z_list[e].shape[0] == 0:
            continue
        ze = z_list[e].to(dev)
        idx = idx_list[e].to(dev)
        # scatter Z_e Z_eᵀ (T_e x T_e) into G[idx, idx]
        Ge = ze @ ze.T
        G[idx.unsqueeze(1), idx.unsqueeze(0)] += Ge
        # S_ee = Z_eᵀ Z_e ; diagonal-block energy
        See = ze.T @ ze
        diag_energy += float((See * See).sum())
        del ze, Ge, See
    total_energy = float((G * G).sum())          # ‖S‖_F² = ‖G‖_F²
    off_energy = total_energy - diag_energy
    off_frac = off_energy / max(total_energy, 1e-30)
    print(f"[energy] total(‖C_full‖_F²)={total_energy:.4e} diag-block={diag_energy:.4e} "
          f"off-diag={off_energy:.4e}  OFF-DIAGONAL FRACTION={off_frac:.4f}")

    # ---- Step 3: full-covariance ridge leverage via push-through ----------- #
    # M = (AAᵀ + λI)⁻¹ = (G/T + λI)⁻¹.  τ_{e,i} = (1/T) z_{e,·i}ᵀ M_e z_{e,·i}.
    # In-place G -> (G/T + λI) so we never hold two TxT copies (halves peak mem);
    # torch.linalg.inv reuses the buffer where it can.
    print(f"[leverage] inverting (G/T+λI), T={T} ...")
    G.div_(T)
    G.diagonal().add_(args.lam)
    M = torch.linalg.inv(G)
    del G
    torch.cuda.empty_cache()

    # Three leverage variants per channel:
    #   lev_full   : full stacked covariance C_full=Zᵀ Z/T  (includes off-diag)
    #   lev_matched: block-diagonal of the SAME C_full (z_eᵀz_e/T) — isolates the
    #                cross-expert effect from normalization (only diff vs lev_full
    #                is the off-diagonal blocks; same /T scale, same λ)
    #   lev_pipe   : pipeline-faithful per-expert (z_eᵀz_e/T_e) — what the code does today
    lev_full = torch.zeros(N, d_mlp)
    lev_matched = torch.zeros(N, d_mlp)
    lev_pipe = torch.zeros(N, d_mlp)
    for e in range(N):
        if z_list[e].shape[0] == 0:
            continue
        ze = z_list[e].to(dev)               # (T_e, d_mlp)
        idx = idx_list[e].to(dev)
        Me = M.index_select(0, idx).index_select(1, idx)   # (T_e, T_e)
        tau = (ze * (Me @ ze)).sum(dim=0) / T              # (d_mlp,)  full-cov leverage
        lev_full[e] = tau.cpu()
        gram = ze.T @ ze
        lev_matched[e] = ridge_leverage_direct(gram / T, args.lam).cpu()
        lev_pipe[e] = ridge_leverage_direct(gram / max(ze.shape[0], 1), args.lam).cpu()
        del ze, Me, gram
    del M
    torch.cuda.empty_cache()

    # ---- Step 4: validate push-through on an 8-expert sub-system ----------- #
    subset = torch.topk(Te, min(8, N)).indices.tolist()   # 8 busiest experts
    zsub = torch.zeros(T, len(subset) * d_mlp, device=dev)
    for j, e in enumerate(subset):
        if z_list[e].shape[0] == 0:
            continue
        zsub[idx_list[e].to(dev), j * d_mlp:(j + 1) * d_mlp] = z_list[e].to(dev)
    Csub = (zsub.T @ zsub) / T                                     # (8d, 8d) dense
    lev_direct = ridge_leverage_direct(Csub, args.lam)             # 98304-free ground truth
    Gsub = zsub @ zsub.T / T
    Gsub.diagonal().add_(args.lam)
    Msub = torch.linalg.inv(Gsub)
    max_err = 0.0
    for j, e in enumerate(subset):
        if z_list[e].shape[0] == 0:
            continue
        ze = z_list[e].to(dev)
        idx = idx_list[e].to(dev)
        Me = Msub.index_select(0, idx).index_select(1, idx)
        tau = (ze * (Me @ ze)).sum(dim=0) / T
        direct = lev_direct[j * d_mlp:(j + 1) * d_mlp]
        err = float((tau - direct).abs().max())
        rel = err / float(direct.abs().max().clamp_min(1e-30))
        max_err = max(max_err, rel)
    del zsub, Csub, Gsub, Msub
    torch.cuda.empty_cache()
    print(f"[validate] push-through vs direct dense (8 experts): max rel err = {max_err:.2e}")

    # ---- Step 5: 128x128 block-energy heatmap  B[e,f]=‖Z_eᵀ Z_f‖_F² -------- #
    # Build sparse-dense Z_full (T x N·d_mlp) in fp32 — fp16 overflows here
    # (activation energies ~1e14 exceed fp16's 65504 max -> inf -> NaN sums).
    # M is already freed, so the ~12.9 GB fp32 Z_full fits on one 40 GB GPU.
    Zf = torch.zeros(T, N * d_mlp, dtype=torch.float32, device=dev)
    for e in range(N):
        if z_list[e].shape[0] == 0:
            continue
        Zf[idx_list[e].to(dev), e * d_mlp:(e + 1) * d_mlp] = z_list[e].to(dev)
    B = torch.zeros(N, N)
    for e in range(N):
        if z_list[e].shape[0] == 0:
            continue
        ze = z_list[e].to(dev)
        idx = idx_list[e].to(dev)
        Se = ze.T @ Zf.index_select(0, idx)            # (d_mlp, N·d_mlp)
        Se = Se.reshape(d_mlp, N, d_mlp)
        B[e] = (Se * Se).sum(dim=(0, 2)).cpu()         # ‖Z_eᵀ Z_f‖² for each f
        del ze, Se
    del Zf
    torch.cuda.empty_cache()
    # cross-check total energy
    total_from_B = float(B.sum())
    print(f"[heatmap] total energy from B={total_from_B:.4e} vs G={total_energy:.4e} "
          f"(rel diff {abs(total_from_B-total_energy)/total_energy:.2e})")

    # ---- Step 6: compare full-cov vs per-expert leverage ------------------- #
    # Against BOTH the matched-normalization block-diagonal (isolates the pure
    # cross-expert / off-diagonal effect) and the pipeline-faithful per-expert
    # score (what channel selection uses today).
    k = min(args.keep, d_mlp)

    def compare(lev_a, lev_b):
        ovl, sp = [], []
        for e in range(N):
            if z_list[e].shape[0] == 0:
                continue
            sa = set(torch.topk(lev_a[e], k).indices.tolist())
            sb = set(torch.topk(lev_b[e], k).indices.tolist())
            ovl.append(len(sa & sb) / k)
            sp.append(spearman(lev_a[e], lev_b[e]))
        return torch.tensor(ovl), torch.tensor(sp)

    overlaps, spears = compare(lev_full, lev_matched)      # cross-expert effect only
    overlaps_p, spears_p = compare(lev_full, lev_pipe)      # vs today's pipeline score
    gl_spear = spearman(lev_full.reshape(-1), lev_matched.reshape(-1))
    print(f"[compare vs matched block-diag] top-{k}/{d_mlp} overlap: mean={overlaps.mean():.4f} "
          f"std={overlaps.std():.4f} min={overlaps.min():.4f} | Spearman mean={spears.mean():.4f}")
    print(f"[compare vs pipeline per-expert] top-{k}/{d_mlp} overlap: mean={overlaps_p.mean():.4f} "
          f"std={overlaps_p.std():.4f} min={overlaps_p.min():.4f} | Spearman mean={spears_p.mean():.4f}")
    print(f"[compare] global Spearman full-vs-matched (all channels): {gl_spear:.4f}")

    # per-expert diagonal block energy fraction (how self-dominated is each expert)
    diag_block = torch.tensor([float(B[e, e]) for e in range(N)])
    row_tot = B.sum(dim=1).clamp_min(1e-30)
    per_expert_offdiag = (1.0 - diag_block / row_tot)

    # ---- Save artifacts ---------------------------------------------------- #
    summary = {
        "model": args.model, "layer": args.layer, "tokens_T": int(T),
        "num_experts": int(N), "d_mlp": int(d_mlp), "top_k": int((Te > 0).sum() and 8),
        "lambda_ridge": args.lam, "keep_k": int(k),
        "tokens_per_expert": {"min": int(Te.min()), "mean": float(Te.float().mean()),
                              "max": int(Te.max())},
        "energy": {
            "total": total_energy, "diag_block": diag_energy, "off_diag": off_energy,
            "off_diag_fraction": off_frac,
            "per_expert_offdiag_fraction_mean": float(per_expert_offdiag.mean()),
            "per_expert_offdiag_fraction_std": float(per_expert_offdiag.std()),
        },
        "pushthrough_validation_max_rel_err": max_err,
        "leverage_compare": {
            "top_k": int(k),
            "full_vs_matched_blockdiag": {
                "note": "isolates pure cross-expert/off-diagonal effect (same /T norm & λ)",
                "per_expert_overlap_mean": float(overlaps.mean()),
                "per_expert_overlap_std": float(overlaps.std()),
                "per_expert_overlap_min": float(overlaps.min()),
                "per_expert_spearman_mean": float(spears.mean()),
                "global_spearman": gl_spear,
            },
            "full_vs_pipeline_perexpert": {
                "note": "vs today's channel-selection score (z_eᵀz_e/T_e)",
                "per_expert_overlap_mean": float(overlaps_p.mean()),
                "per_expert_overlap_std": float(overlaps_p.std()),
                "per_expert_overlap_min": float(overlaps_p.min()),
                "per_expert_spearman_mean": float(spears_p.mean()),
            },
        },
        "runtime_s": time.time() - t0,
    }
    with open(os.path.join(args.out_dir, f"summary_L{args.layer}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(
        os.path.join(args.out_dir, f"arrays_L{args.layer}.npz"),
        block_energy=B.numpy(), lev_full=lev_full.numpy(),
        lev_matched=lev_matched.numpy(), lev_pipe=lev_pipe.numpy(),
        overlaps=overlaps.numpy(), spearman=spears.numpy(),
        overlaps_pipe=overlaps_p.numpy(), spearman_pipe=spears_p.numpy(),
        tokens_per_expert=Te.numpy(), per_expert_offdiag=per_expert_offdiag.numpy(),
    )
    print(f"[done] {json.dumps(summary, indent=2)}")

    # ---- Optional figures -------------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        Blog = np.log10(B.numpy() + B.numpy()[B.numpy() > 0].min() * 1e-3)
        im = ax[0].imshow(Blog, cmap="viridis")
        ax[0].set_title(f"L{args.layer} block-energy log10 ‖Z_eᵀZ_f‖_F²")
        ax[0].set_xlabel("expert f"); ax[0].set_ylabel("expert e")
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        ax[1].hist(per_expert_offdiag.numpy(), bins=30, color="#4463b0")
        ax[1].axvline(float(off_frac), color="r", ls="--",
                      label=f"global off-diag={off_frac:.3f}")
        ax[1].set_title("per-expert off-diagonal energy fraction")
        ax[1].set_xlabel("off-diag fraction"); ax[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"figures_L{args.layer}.png"), dpi=130)
        print(f"[fig] saved figures_L{args.layer}.png")
    except Exception as exc:
        print(f"[fig] skipped ({exc})")


if __name__ == "__main__":
    main()
