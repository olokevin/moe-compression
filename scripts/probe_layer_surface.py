#!/usr/bin/env python
"""Per-layer ``rel_err(p, rho)`` surface over **all** MoE layers, then solve the
budget allocation that minimizes total error at a fixed used-parameter cost.

Why this exists. Every probe result so far fixes one ``(p, rho)`` pair and applies
it to all 48 MoE layers, because that is what a hand-picked config can express.
But the layers are not interchangeable: ``probe_output_error.py`` already shows
L46 tolerating a 25% input read at rel_err 0.367 while L22 needs 0.511 for the
same setting. If some layers are cheap to score and others are not, a *uniform*
schedule is leaving accuracy on the table at constant cost.

So: measure the whole surface, then allocate.

**The measurement.** One forward pass per input-sparsity level ``p`` over
calibration data. Each MoE block is hooked to compute, locally:

    y_full  = sum_e g_e W_down^e (SiLU(gate) * up)          (the true block output)
    y_kept  = same, with all but the pooled top-B channels zeroed
    rel_err = mean_t || y_full - y_kept || / || y_full ||

for every ``rho`` in the grid, where the top-B set comes from the probe scored on
the token's top-``p`` coordinates. The block **returns ``y_full``**, so the error
introduced at layer L never contaminates layer L+1: every layer is measured
against the same unperturbed input distribution. That independence is what makes
the numbers additive enough to optimize over, and it is deliberate — the whole
point is a *separable* per-layer cost curve.

**The allocation.** With the surface in hand, choose ``(p_L, rho_L)`` per layer to

    minimize  sum_L rel_err_L(p_L, rho_L)
    s.t.      mean_L kept(p_L, rho_L) <= target,
              kept(p, rho) = rho + 2 p (1 - rho)/3        (the reuse frame)

which is a classic separable resource-allocation problem: a single Lagrange
multiplier ``lambda`` on the budget, bisected until the mean cost hits the target.
Each layer independently picks the grid point minimizing
``rel_err_L + lambda * kept_L``. Because the constraint is a mean over layers and
the objective is a sum, this is exactly optimal on the grid (no greedy or
heuristic step).

Reported alongside: the **mean input and channel thresholds** the solution
induces, i.e. the average ``|x|`` cut and the average per-token score cut, so the
schedule can also be expressed as two global magnitude thresholds instead of 48
pairs of integers -- a much more implementable object. ``--emit-config`` writes
the per-layer schedule to JSON for the eval path to consume.

Run on 1-4 GPUs via launch-on-a100 (needs the model; ~30 min for the default grid).
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

from src.base.shared_utils.safe_isinstance import (
    _get_experts,
    _get_moe_block,
    _get_moe_intermediate_size,
    _get_num_hidden_layers,
    _get_topk,
)
from src.dynamic_active_param.sparse_probe import (
    quantize_rtn_dequant,
    sparsify_input_topk,
    used_param_fraction,
)


# ==========================================================================
# the per-layer measurement hook
# ==========================================================================

class LayerSurfaceProbe:
    """Replacement forward for one MoE block: measures, never perturbs.

    Accumulates ``sum_t rel_err`` and the token count for each ``rho`` in the
    grid, at the single input-sparsity ``p`` this pass is running. Also
    accumulates the score/|x| thresholds so the schedule can be re-expressed as
    global magnitude cuts.
    """

    def __init__(self, block, rhos, p, bits, group, max_tokens):
        self.block, self.rhos, self.p = block, rhos, float(p)
        self.bits, self.group = int(bits), int(group)
        self.max_tokens = int(max_tokens)
        self.I = block._surf_I
        self.K = block._surf_K
        self.err = np.zeros(len(rhos))
        self.n = 0
        # thresholds: the |x| cut and the score cut actually used, per rho
        self.xthr_sum = 0.0
        self.sthr_sum = np.zeros(len(rhos))

    @torch.no_grad()
    def __call__(self, hidden_states):
        blk = self.block
        bs, sl, H = hidden_states.shape
        x = hidden_states.view(-1, H)
        router_logits = blk.gate(x)
        rw = F.softmax(router_logits, dim=1, dtype=torch.float)
        rw, sel = torch.topk(rw, blk.top_k, dim=-1)
        if blk.norm_topk_prob:
            rw = rw / rw.sum(dim=-1, keepdim=True)
        rw = rw.to(x.dtype)

        T, K, I = x.shape[0], self.K, self.I
        take = min(T, self.max_tokens - self.n) if self.max_tokens else T
        if take > 0:
            self._measure(x[:take], rw[:take], sel[:take], H)
            self.n += take

        # --- the true block output: this is what we return, unperturbed -----
        out = torch.zeros((T, H), dtype=x.dtype, device=x.device)
        em = F.one_hot(sel, num_classes=blk.num_experts).permute(2, 1, 0)
        for eid in torch.greater(em.sum(dim=(-1, -2)), 0).nonzero():
            e = int(eid)
            ex = blk.experts[e]
            idx, top_x = torch.where(em[e].squeeze(0))
            cur = x[top_x]
            inter = ex.act_fn(ex.gate_proj(cur)) * ex.up_proj(cur)
            out.index_add_(0, top_x,
                           (ex.down_proj(inter) * rw[top_x, idx, None]).to(x.dtype))
        if getattr(blk, "shared_expert", None) is not None:
            se = blk.shared_expert(x)
            out = out + F.sigmoid(blk.shared_expert_gate(x)) * se
        return out.view(bs, sl, H), router_logits

    @torch.no_grad()
    def _measure(self, x, rw, sel, H):
        blk = self.block
        T, K, I = x.shape[0], self.K, self.I
        dev = x.device
        g = rw.to(torch.float32)

        # probe input: the token's top-p coordinates by |x| (uniform alloc)
        x_sp = sparsify_input_topk(x, self.p)
        if self.p < 1.0:
            k = max(1, int(round(self.p * H)))
            self.xthr_sum += float(x.abs().topk(k, dim=-1).values[:, -1].sum())

        inter = torch.zeros((T, K, I), dtype=torch.float32, device=dev)
        proxy = torch.zeros((T, K, I), dtype=torch.float32, device=dev)
        em = F.one_hot(sel, num_classes=blk.num_experts).permute(2, 1, 0)
        hits = []
        for eid in torch.greater(em.sum(dim=(-1, -2)), 0).nonzero():
            e = int(eid)
            ex = blk.experts[e]
            idx, top_x = torch.where(em[e].squeeze(0))
            hits.append((e, idx, top_x))
            cur = x[top_x]
            inter[top_x, idx] = (ex.act_fn(ex.gate_proj(cur))
                                 * ex.up_proj(cur)).float()
            # probe: reuse regime reads the served weights on the sparse input
            cs = x_sp[top_x]
            Wu, Wg = ex.up_proj.weight, ex.gate_proj.weight
            if self.bits < 16:
                Wu = quantize_rtn_dequant(Wu, self.bits, self.group)
                Wg = quantize_rtn_dequant(Wg, self.bits, self.group)
            uh = (cs @ Wu.t()).float()
            gh = (cs @ Wg.t()).float()
            proxy[top_x, idx] = (F.silu(gh) * uh).abs()
        score = g.unsqueeze(-1) * proxy

        y_full = torch.zeros((T, H), dtype=torch.float32, device=dev)
        for e, idx, top_x in hits:
            Wd = blk.experts[e].down_proj.weight.float()
            y_full.index_add_(0, top_x, (inter[top_x, idx] @ Wd.t())
                              * g[top_x, idx].unsqueeze(-1))
        fnorm = y_full.norm(dim=1).clamp_min(1e-30)

        flat = score.reshape(T, K * I)
        for bi, rho in enumerate(self.rhos):
            B = max(1, min(int(round(rho * K * I)), K * I))
            vals, idxs = flat.topk(B, dim=1, sorted=True)
            self.sthr_sum[bi] += float(vals[:, -1].sum())
            m = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, idxs, True)
            dropped = inter * (~m.reshape(T, K, I))
            y_err = torch.zeros((T, H), dtype=torch.float32, device=dev)
            for e, idx, top_x in hits:
                Wd = blk.experts[e].down_proj.weight.float()
                y_err.index_add_(0, top_x, (dropped[top_x, idx] @ Wd.t())
                                 * g[top_x, idx].unsqueeze(-1))
            self.err[bi] += float((y_err.norm(dim=1) / fnorm).sum())


# ==========================================================================
# the allocation solver
# ==========================================================================

def layer_slope_weights(layers, ref_depth=None):
    """Per-layer accuracy sensitivity ``w_L``, normalized to mean 1.

    ``scripts/probe_relerr_linearity.py`` fits, for each captured layer, how many
    HellaSwag points a unit of *that layer's* rel_err costs at fixed budget:

        L6 −36.07,  L22 −29.77,  L38 −24.99,  L46 −15.06   (pt per unit rel_err)

    which is a strikingly clean linear trend in depth (r = 0.957, +0.478 pt/unit
    per layer): **early layers are ~2.4x more sensitive than late ones.** So
    minimizing the *unweighted* sum of rel_err is the wrong program — it happily
    moves error onto the layers where error is most expensive. (Measured: the
    unweighted schedule lost 0.16pt against a matched uniform baseline, because it
    handed L0-5 the cheapest budget of any band.)

    Weighting each layer's error by its own slope fixes the objective. Only four
    layers were fitted, so we interpolate the depth trend rather than pretend to
    per-layer precision; the sign and magnitude of the trend are what matter.
    """
    lay = np.asarray(layers, dtype=float)
    # slopes fitted in probe_relerr_linearity.py (fixed-budget family)
    fit_x = np.array([6.0, 22.0, 38.0, 46.0])
    fit_y = np.array([36.07, 29.77, 24.99, 15.06])          # |pt per unit rel_err|
    a, b = np.polyfit(fit_x, fit_y, 1)
    w = a * lay + b
    w = np.clip(w, 1e-3, None)
    return w / w.mean()


def allocate(surface, ps, rhos, target, n_matrices=2, weights=None):
    """Minimize ``sum_L w_L * rel_err_L`` s.t. ``mean_L kept_L <= target``.

    Separable, so one Lagrange multiplier suffices: each layer independently
    minimizes ``w_L*rel_err + lambda*kept``, and ``lambda`` is bisected until the
    mean kept fraction meets the target. Exactly optimal on the grid.

    Args:
        surface: ``(L, n_p, n_rho)`` array of rel_err.
        ps, rhos: the grid axes.
        target: mean kept fraction allowed (e.g. 0.25 for a -75% cut).
        n_matrices: scored branches (2 = up+gate).
        weights: ``(L,)`` per-layer accuracy sensitivity (mean 1). ``None`` gives
            the unweighted objective, which is **measured to be wrong** — see
            :func:`layer_slope_weights`.

    Returns dict with the per-layer choice and the achieved cost.
    """
    L = surface.shape[0]
    cost = np.array([[used_param_fraction(p, r, n_matrices)
                      for r in rhos] for p in ps])          # (n_p, n_rho)
    w = np.ones(L) if weights is None else np.asarray(weights, dtype=float)
    wsurf = surface * w[:, None, None]

    def solve(lam):
        obj = wsurf + lam * cost[None, :, :]
        flat = obj.reshape(L, -1).argmin(axis=1)
        pi, ri = np.unravel_index(flat, cost.shape)
        return pi, ri, cost[pi, ri], surface[np.arange(L), pi, ri]

    lo, hi = 0.0, 1e6
    for _ in range(200):
        lam = 0.5 * (lo + hi)
        _, _, c, _ = solve(lam)
        if c.mean() > target:
            lo = lam                      # too expensive -> price the budget up
        else:
            hi = lam
    pi, ri, c, e = solve(hi)
    return {"p_idx": pi, "rho_idx": ri, "cost": c, "err": e, "lam": hi,
            "mean_cost": float(c.mean()), "total_err": float(e.sum()),
            "mean_err": float(e.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--tokens", type=int, default=2048,
                    help="calibration tokens measured per layer")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    # The two sparsities (both keep fractions). --ps/--rhos are kept as aliases
    # because the reproduce commands in the docs use them.
    ap.add_argument("--rho-inputs", "--ps", dest="ps",
                    default="0.125,0.1875,0.25,0.375,0.5,1.0",
                    help="rho_input grid: input coordinates read for scoring")
    ap.add_argument("--rho-channels", "--rhos", dest="rhos",
                    default="0.0625,0.09375,0.125,0.15625,0.1875,0.25",
                    help="rho_channel grid: channels kept for compute")
    ap.add_argument("--bits", type=int, default=16,
                    help="16 = reuse the served weights (no extra storage)")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--targets", default="0.2667,0.2917,0.3167",
                    help="mean used-param fractions to solve for; defaults are the "
                         "-73.3 / -70.8 / -68.3%% cuts")
    ap.add_argument("--per-gpu-mem", default=os.environ.get("PER_GPU_MEM", "36GiB"))
    ap.add_argument("--out", default=os.path.join(
        _REPO, "docs/results/idea_pilot/layer_surface.json"))
    ap.add_argument("--emit-config", default="")
    ap.add_argument("--no-slope-weight", action="store_true",
                    help="minimize the UNWEIGHTED sum of rel_err. Measured to be "
                         "worse than the slope-weighted objective (it moves error "
                         "onto the sensitive early layers); kept for reproducing "
                         "the original negative result.")
    ap.add_argument("--reuse-surface", default="",
                    help="skip the GPU measurement and re-solve from an existing "
                         "layer_surface.json (no model load).")
    args = ap.parse_args()

    ps = [float(v) for v in args.ps.split(",")]
    rhos = [float(v) for v in args.rhos.split(",")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Re-solve from a cached surface: the measurement is the expensive part, and
    # the objective is the thing being iterated on. No model, no GPU.
    if args.reuse_surface:
        prev = json.load(open(args.reuse_surface))
        surface = np.array(prev["surface"])
        ps, rhos = prev["ps"], prev["rhos"]
        xthr = np.array(prev["x_threshold"])
        sthr = np.array(prev["score_threshold"])
        moe_layers = [(li, None) for li in prev["layers"]]
        out = dict(prev)
        out["solutions"] = {}
        out["slope_weighted"] = not args.no_slope_weight
        print(f"[surface] re-solving from {args.reuse_surface} "
              f"({surface.shape[0]} layers, no GPU)", flush=True)
        _solve_and_report(args, out, surface, ps, rhos, moe_layers, xthr, sthr)
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.base.datasets import load_datasets

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("[surface] loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
        max_memory={i: args.per_gpu_mem for i in range(torch.cuda.device_count())},
    )
    model.eval()

    I, K = _get_moe_intermediate_size(model), _get_topk(model)
    moe_layers = []
    for li in range(_get_num_hidden_layers(model)):
        blk = _get_moe_block(model, li)
        if _get_experts(blk) is None:
            continue
        blk._surf_I, blk._surf_K = int(I), int(K)
        moe_layers.append((li, blk))
    print(f"[surface] {len(moe_layers)} MoE layers, I={I} K={K}", flush=True)

    # calibration text -> a fixed list of tokenized batches, reused verbatim for
    # every p so the surface is measured on identical tokens (the whole point is
    # comparing layers and settings, not sampling noise).
    print("[surface] loading calibration data...", flush=True)
    n_seq = args.tokens // args.seq_len + args.batch_size
    ds = load_datasets("c4", tok, max_samples=n_seq + 16, max_length=args.seq_len)
    batches, total = [], 0
    for i in range(0, len(ds), args.batch_size):
        if total >= args.tokens:
            break
        chunk = [str(t) for t in ds[i:i + args.batch_size] if t]
        if not chunk:
            continue
        enc = tok(chunk, max_length=args.seq_len, padding="max_length",
                  truncation=True, return_tensors="pt")
        batches.append(enc["input_ids"])
        total += int(enc["input_ids"].numel())
    print(f"[surface] {len(batches)} batches ({total} tokens)", flush=True)

    import types
    surface = np.zeros((len(moe_layers), len(ps), len(rhos)))
    xthr = np.zeros((len(moe_layers), len(ps)))
    sthr = np.zeros((len(moe_layers), len(ps), len(rhos)))
    originals = {li: blk.forward for li, blk in moe_layers}

    for pi, p in enumerate(ps):
        probes = {}
        for li, blk in moe_layers:
            pr = LayerSurfaceProbe(blk, rhos, p, args.bits, args.group, args.tokens)
            probes[li] = pr
            blk.forward = types.MethodType(lambda self, hs, _pr=pr: _pr(hs), blk)
        with torch.no_grad():
            for bi, ids in enumerate(batches):
                model(ids.to(model.device))
                print(f"[surface] p={p} batch {bi+1}/{len(batches)}", flush=True)
        for i, (li, blk) in enumerate(moe_layers):
            pr = probes[li]
            surface[i, pi] = pr.err / max(pr.n, 1)
            xthr[i, pi] = pr.xthr_sum / max(pr.n, 1)
            sthr[i, pi] = pr.sthr_sum / max(pr.n, 1)
            blk.forward = originals[li]
        print(f"[surface] p={p} done; layer-avg rel_err per rho: "
              + " ".join(f"{r}:{surface[:, pi, j].mean():.4f}"
                         for j, r in enumerate(rhos)), flush=True)

    out = {"model": args.model, "ps": ps, "rhos": rhos, "bits": args.bits,
           "layers": [li for li, _ in moe_layers], "tokens": args.tokens,
           "surface": surface.tolist(), "x_threshold": xthr.tolist(),
           "score_threshold": sthr.tolist(), "solutions": {},
           "slope_weighted": not args.no_slope_weight}
    _solve_and_report(args, out, surface, ps, rhos, moe_layers, xthr, sthr)


def _solve_and_report(args, out, surface, ps, rhos, moe_layers, xthr, sthr):
    """Solve the allocation at each target, print it, and write ``args.out``."""
    # ---- solve the allocation at each target -----------------------------
    cost_grid = np.array([[used_param_fraction(p, r) for r in rhos]
                          for p in ps])
    print("\n[alloc] cost grid (used-param fraction = rho_channel + 2*rho_input/3):")
    print("  rho_channel:  " + "  ".join(f"{r:6.4f}" for r in rhos))
    for i, p in enumerate(ps):
        print(f"  rho_in={p:6.4f}: " + "  ".join(f"{c:6.4f}" for c in cost_grid[i]))

    W = (None if args.no_slope_weight
         else layer_slope_weights([li for li, _ in moe_layers]))
    if W is not None:
        print(f"\n[alloc] slope weights (mean 1): L{moe_layers[0][0]}={W[0]:.3f} ... "
              f"L{moe_layers[-1][0]}={W[-1]:.3f} "
              f"(early layers are more accuracy-sensitive)")
    for tgt in [float(t) for t in args.targets.split(",")]:
        sol = allocate(surface, ps, rhos, tgt, weights=W)
        # the matched uniform baseline: cheapest single (p,rho) meeting the target
        feas = cost_grid <= tgt + 1e-9
        umean = np.where(feas[None, :, :], surface, np.inf).mean(axis=0)
        ui = np.unravel_index(np.argmin(umean), umean.shape)
        uni_err = float(umean[ui])
        sched = [{"layer": int(out["layers"][i]),
                  "p": ps[sol["p_idx"][i]], "rho": rhos[sol["rho_idx"][i]],
                  "kept": float(sol["cost"][i]), "rel_err": float(sol["err"][i])}
                 for i in range(len(moe_layers))]
        # mean induced thresholds
        mx = float(np.mean([xthr[i, sol["p_idx"][i]] for i in range(len(moe_layers))]))
        ms = float(np.mean([sthr[i, sol["p_idx"][i], sol["rho_idx"][i]]
                            for i in range(len(moe_layers))]))
        out["solutions"][f"{tgt:.4f}"] = {
            "target_kept": tgt, "achieved_mean_kept": sol["mean_cost"],
            "lambda": sol["lam"], "mean_rel_err": sol["mean_err"],
            "uniform_best": {"p": ps[ui[0]], "rho": rhos[ui[1]],
                             "mean_rel_err": uni_err,
                             "kept": float(cost_grid[ui])},
            "gain_rel_err": uni_err - sol["mean_err"],
            "mean_x_threshold": mx, "mean_score_threshold": ms,
            "schedule": sched,
        }
        print(f"\n[alloc] target used={tgt:.4f} (cut {100*(1-tgt):.1f}%) "
              f"lambda={sol['lam']:.3f}")
        print(f"  per-layer optimum : mean rel_err {sol['mean_err']:.4f} "
              f"(mean used {sol['mean_cost']:.4f})")
        print(f"  best uniform      : rho_input={ps[ui[0]]} "
              f"rho_channel={rhos[ui[1]]} mean rel_err {uni_err:.4f}")
        print(f"  gain              : {uni_err - sol['mean_err']:+.4f} rel_err "
              f"=> ~{26.4 * (uni_err - sol['mean_err']):+.2f} HellaSwag pt "
              f"(fixed-budget slope)")
        print(f"  induced mean thresholds: |x| cut {mx:.4f}, score cut {ms:.6g}")
        pc = {}
        for s in sched:
            pc.setdefault((s["p"], s["rho"]), []).append(s["layer"])
        print("  schedule (rho_input, rho_channel) -> layers:")
        for (p, r), ls in sorted(pc.items()):
            print(f"    rho_input={p:6.4f} rho_channel={r:7.5f} : "
                  f"{len(ls):2d} layers {ls}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {args.out}")

    if args.emit_config:
        tgt = sorted(out["solutions"])[0]
        with open(args.emit_config, "w") as f:
            json.dump(out["solutions"][tgt]["schedule"], f, indent=2)
        print(f"[done] wrote schedule for target {tgt} -> {args.emit_config}")


if __name__ == "__main__":
    main()
