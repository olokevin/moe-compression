"""Phase 0 model-level gates + the fast C4-PPL ladder for the lm_head baselines.

Gates 0a-0d of ``docs/exps/lm_head/plan/baselines.md`` on a real checkpoint, then an
optional held-out C4 perplexity sweep over head treatments. The PPL loop is a
direct token-level metric on a fixed token budget, which makes it far cheaper than
an lm-eval pass while being *the* metric that is actually sensitive to head
approximation (plan section 3).

Usage::

    python scripts/lm_head_gates.py --model Qwen/Qwen3-0.6B --gates
    python scripts/lm_head_gates.py --model Qwen/Qwen3-0.6B --ladder \
        --calib-dir ./calib/qwen3_0_6b --ppl-tokens 262144
"""

import argparse
import copy
import json
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.base.datasets import load_datasets
from src.base.shared_utils import _print
from src.lm_head import bind_head_forward, install_lm_head, unbind_head_forward
from src.lm_head.calib import ensure_sigma, ensure_unigram, get_lm_head
from src.lm_head.tiering import build_tiers, tier_stats


def load(model_id, dtype=torch.bfloat16, device_map=None):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kw = dict(dtype=dtype, trust_remote_code=True, attn_implementation="sdpa")
    if device_map:
        kw["device_map"] = device_map
    m = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if not device_map:
        m = m.to("cuda" if torch.cuda.is_available() else "cpu")
    m.eval()
    return m, tok


# --------------------------------------------------------------------------- #
# C4 perplexity on a fixed held-out token budget
# --------------------------------------------------------------------------- #

@torch.no_grad()
def c4_ppl(model, tok, n_tokens=262144, seq=1024, skip_texts=20000, batch=4, verbose=True):
    """Held-out C4 perplexity over a fixed token budget.

    ``skip_texts`` steps past the slice used for calibration so the measurement is
    genuinely held out. Returns ``(ppl, oov_rate)`` where ``oov_rate`` is the
    fraction of target positions that received a non-finite log-probability -- the
    honest way to report a strict-mode sparse head, which formally has infinite
    perplexity the moment a target token falls outside its read set.
    """
    texts = load_datasets("c4", tok, max_samples=skip_texts + 8000, max_length=None)
    texts = [t for t in texts[skip_texts:] if isinstance(t, str) and len(t) > 200]
    device = next(model.parameters()).device
    if getattr(model, "hf_device_map", None) is not None:
        device = "cuda"

    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        if len(ids) >= n_tokens + seq:
            break
    ids = torch.tensor(ids[: n_tokens + 1], dtype=torch.long)
    n_win = max(1, (ids.numel() - 1) // seq)

    nll_sum, n_pos, n_bad = 0.0, 0, 0
    for s in range(0, n_win, batch):
        wins = []
        for w in range(s, min(s + batch, n_win)):
            wins.append(ids[w * seq: w * seq + seq + 1])
        wins = [w for w in wins if w.numel() == seq + 1]
        if not wins:
            continue
        chunk = torch.stack(wins).to(device)
        logits = model(chunk[:, :-1]).logits.float()
        lsm = F.log_softmax(logits, dim=-1)
        tgt = chunk[:, 1:]
        lp = lsm.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        bad = ~torch.isfinite(lp)
        n_bad += int(bad.sum())
        nll_sum += float((-lp[~bad]).sum())
        n_pos += int(lp.numel())

    covered = n_pos - n_bad
    ppl_cov = float(torch.exp(torch.tensor(nll_sum / max(covered, 1))))
    oov = n_bad / max(n_pos, 1)
    if verbose:
        _print(
            f"    C4 PPL over {n_pos:,} positions: {ppl_cov:.4f} on covered tokens"
            + (f"; {100 * oov:.4f}% of targets were OUT OF READ SET "
               f"(true PPL = inf)" if n_bad else "")
        )
    return ppl_cov, oov


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

@torch.no_grad()
def gate_0a(model_id, calib_dir):
    """freq_tier with tier_size=V and 16/16 bits must be bit-exact vs dense logits."""
    _print("\n=== Gate 0a: identity install is bit-exact ===")
    m, tok = load(model_id)
    V = get_lm_head(m).weight.shape[0]
    x = tok(["The capital of France is Paris, and the capital of Germany is"],
            return_tensors="pt").to(next(m.parameters()).device)
    ref = m(**x).logits.detach().clone()
    emb_ref = None
    e = m.model.embed_tokens.weight
    emb_ref = e.detach().clone()

    m = install_lm_head(
        m, {"method": "freq_tier", "tier_size": V, "head_bits": 16, "tail_bits": 16,
            "calib_dir": calib_dir, "diagnostics": False}, tokenizer=tok,
    )
    out = m(**x).logits
    same = torch.equal(out, ref)
    max_abs = float((out - ref).abs().max())
    _print(f"  logits bit-identical: {same} (max |Δ| = {max_abs:.3e})")
    emb_same = torch.equal(m.model.embed_tokens.weight, emb_ref)
    _print(f"  input embedding untouched: {emb_same}")
    assert same, f"gate 0a FAILED: identity install perturbed the logits by {max_abs}"
    assert emb_same, "gate 0a FAILED: the input embedding was modified"
    _print("  ✅ gate 0a PASSED")
    del m
    torch.cuda.empty_cache()


@torch.no_grad()
def gate_0c(model_id, calib_dir, T=4096):
    """Strict mode: every out-of-tier logit is exactly -inf, in-tier ones unchanged."""
    _print("\n=== Gate 0c: strict-mode masking on a real head ===")
    m, tok = load(model_id)
    x = tok(["Perplexity is a measurement of how well a probability model predicts"],
            return_tensors="pt").to(next(m.parameters()).device)
    ref = m(**x).logits.detach().clone()

    m = install_lm_head(
        m, {"method": "freq_tier", "tier_size": T, "head_bits": 16, "tail_bits": 0,
            "strict": True, "calib_dir": calib_dir, "diagnostics": False},
        tokenizer=tok,
    )
    mask = get_lm_head(m)._lmh_keep_mask
    out = m(**x).logits
    n_in = int(mask.sum())
    ok_inf = bool(torch.isinf(out[..., ~mask]).all() and (out[..., ~mask] < 0).all())
    ok_exact = torch.equal(out[..., mask], ref[..., mask])
    lsm = F.log_softmax(out.float(), -1)
    ok_finite = bool(torch.isfinite(lsm[..., mask]).all())
    _print(f"  tier rows={n_in:,}; out-of-tier all -inf: {ok_inf}; "
           f"in-tier bit-exact vs dense: {ok_exact}; log_softmax finite in-tier: {ok_finite}")
    assert ok_inf and ok_exact and ok_finite, "gate 0c FAILED"
    _print("  ✅ gate 0c PASSED")
    del m
    torch.cuda.empty_cache()


@torch.no_grad()
def gate_0d(model_id, calib_dir, T=4096):
    """device_map='auto' sharding: no cross-device tensor errors."""
    _print("\n=== Gate 0d: sharded device_map='auto' ===")
    if torch.cuda.device_count() < 2:
        _print("  ⚠️  fewer than 2 GPUs visible; running anyway to exercise the path")
    m, tok = load(model_id, device_map="auto")
    _print(f"  hf_device_map spans: {sorted(set(map(str, (m.hf_device_map or {}).values())))}")
    m = install_lm_head(
        m, {"method": "freq_tier", "tier_size": T, "head_bits": 16, "tail_bits": 4,
            "sparse_activate": True, "strict": True, "calib_dir": calib_dir,
            "diagnostics": False},
        tokenizer=tok,
    )
    x = tok(["Sharded forward pass smoke test"], return_tensors="pt")
    x = {k: v.to("cuda:0") for k, v in x.items()}
    out = m(**x).logits
    head = get_lm_head(m)
    _print(f"  logits {tuple(out.shape)} on {out.device}; "
           f"mask on {head._lmh_keep_mask.device}, weight on {head.weight.device}")
    assert torch.isfinite(out[..., head._lmh_keep_mask]).all()
    _print("  ✅ gate 0d PASSED (mask followed the head module's device)")
    del m
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# the C4-PPL ladder: F2 / F3 / B1 / B2 / B3
# --------------------------------------------------------------------------- #

def _lowrank_head(W, C, rank, whiten=True, compute_device="cpu"):
    """F2: the low-rank ladder, with and without activation whitening."""
    from src.lm_head.quant import metric_transform, randomized_svd
    Wf = W.detach().to(compute_device, torch.float32)
    if whiten:
        Tp, Tp_inv, _ = metric_transform(C, p=0.5, ridge=1e-3, compute_device=compute_device)
        U, S, Vh = randomized_svd(Wf @ Tp, rank=rank)
        return ((U * S.unsqueeze(0)) @ (Vh @ Tp_inv)).to(W.dtype)
    U, S, Vh = randomized_svd(Wf, rank=rank)
    return ((U * S.unsqueeze(0)) @ Vh).to(W.dtype)


def run_ladder(model_id, calib_dir, n_tokens, out_json, runs=None):
    """Measure held-out C4 PPL for every head treatment on one loaded model."""
    os.makedirs(calib_dir, exist_ok=True)
    m, tok = load(model_id)
    head = get_lm_head(m)
    V, D = head.weight.shape

    # untie up front so every row of the ladder shares one baseline
    from src.lm_head.install import _untie_if_needed
    was_tied, untie_cost = _untie_if_needed(m, head)

    counts = ensure_unigram(tok, {}, calib_dir, V)
    C, H = ensure_sigma(m, tok, {}, calib_dir)
    _print("[ladder] corpus mass by tier size: " + ", ".join(
        f"T={t}: {100 * v:.2f}%" for t, v in tier_stats(counts).items()))

    W0 = head.weight.detach().clone()
    from src.lm_head.quant import bits_per_weight, quantize_rows_mixed, quantize_rtn_dequant
    from src.lm_head.archead import build_archead
    from src.lm_head.vq import build_rvq, build_vq_logits
    from src.lm_head.accounting import count_active_params, head_cost
    ctx = count_active_params(m)

    def restore():
        """Put the head back to dense BF16 so every ladder row starts from the same state."""
        head.weight.data.copy_(W0)
        unbind_head_forward(head)

    results = []

    def measure(name, bpw, read_rows=None, read_bpw=None, extra=None):
        t0 = time.time()
        ppl, oov = c4_ppl(m, tok, n_tokens=n_tokens)
        cost = head_cost(V, D, bpw, read_rows=read_rows, read_bits_per_weight=read_bpw)
        row = {
            "run": name, "ppl": ppl, "oov_rate": oov,
            "storage_frac": cost["storage_frac_of_bf16"],
            "read_frac": cost["read_frac_of_bf16"],
            "used_head_params_M": cost["used_head_params_bf16eq"] / 1e6,
            "delta_active_pct": -100 * (cost["dense_params"] - cost["used_head_params_bf16eq"])
                                / max(ctx.active_params, 1),
            "bits_per_weight": bpw, "secs": round(time.time() - t0, 1),
        }
        row.update(extra or {})
        results.append(row)
        _print(f"  -> {name}: PPL {ppl:.4f}, storage {100 * row['storage_frac']:.2f}%, "
               f"reads {100 * row['read_frac']:.2f}%, Δactive {row['delta_active_pct']:+.2f}%")
        with open(out_json, "w") as f:
            json.dump({"model": model_id, "V": V, "D": D, "was_tied": was_tied,
                       "untie_cost_params": untie_cost,
                       "active_params": ctx.active_params,
                       "total_params": ctx.total_params,
                       "n_ppl_tokens": n_tokens, "rows": results}, f, indent=2)
        return row

    want = (lambda k: True) if not runs else (lambda k: any(r in k for r in runs))

    if want("dense"):
        _print("\n--- dense reference ---")
        measure("dense-bf16", 16.0)

    # F3 -- plain group RTN, the honest naive floor
    for b in (8, 4, 3, 2):
        if not want(f"rtn{b}"):
            continue
        _print(f"\n--- F3: group RTN {b}-bit g128 ---")
        head.weight.data.copy_(quantize_rtn_dequant(W0.float().cpu(), bits=b, group=128).to(W0.dtype))
        measure(f"F3-rtn{b}-g128", bits_per_weight(b, 128))
        restore()

    # F2 -- the low-rank ladder (the exclusion in plan section 2 rests on a TIED 0.6B)
    for rank, tag in ((D // 4, "25pct"), (D // 2, "50pct"), (int(D * 0.75), "75pct")):
        for wh in (True, False):
            key = f"F2-lowrank-r{rank}-{'whitened' if wh else 'plain'}"
            if not want(f"lowrank"):
                continue
            _print(f"\n--- F2: low-rank r={rank} ({tag} storage) "
                   f"{'whitened' if wh else 'plain'} ---")
            head.weight.data.copy_(_lowrank_head(W0, C, rank, whiten=wh))
            # storage = (V + D) * rank fp16 values, expressed per dense element
            bpw = 16.0 * (V + D) * rank / (V * D)
            measure(key, bpw, extra={"rank": rank, "whitened": wh})
            restore()

    # B1-s -- frequency-tiered storage
    for T in (4096, 16384):
        for tb in (4, 2):
            if not want("B1-s"):
                continue
            _print(f"\n--- B1-s: T={T}, tail {tb}-bit ---")
            tiers = build_tiers(counts, T)
            head.weight.data.copy_(quantize_rows_mixed(
                W0.float().cpu(), tiers.keep_mask, 16, tb, 128).to(W0.dtype))
            bpw = (16.0 * T + bits_per_weight(tb, 128) * (V - T)) / V
            measure(f"B1-s-T{T}-tail{tb}", bpw,
                    extra={"tier_size": T, "tail_bits": tb, "head_mass": tiers.head_mass})
            restore()

    # B1-a / B1-p -- sparse activation / pruned tail, strict
    for T in (4096, 8192, 16384, 32768):
        if not want("B1-a"):
            continue
        _print(f"\n--- B1-a: T={T}, strict (reads only the tier) ---")
        tiers = build_tiers(counts, T)
        from src.lm_head.install import _tiered_lm_head_forward
        head._lmh_keep_mask = tiers.keep_mask.to(head.weight.device)
        head._lmh_tail_logit = None
        head._lmh_stats = None
        bind_head_forward(head, _tiered_lm_head_forward)
        measure(f"B1-a-T{T}-strict", 16.0, read_rows=T, read_bpw=16.0,
                extra={"tier_size": T, "head_mass": tiers.head_mass, "strict": True})
        # and the uniform fallback, which keeps PPL finite
        head._lmh_tail_logit = tiers.tail_logit_offset
        measure(f"B1-a-T{T}-fallback", 16.0, read_rows=T, read_bpw=16.0,
                extra={"tier_size": T, "head_mass": tiers.head_mass, "strict": False})
        restore()

    # B2 -- ARCHead, at the paper's hyperparameters and one aggressive point
    for rc, rr, rb, tag in ((10, 6, 4, "25"), (10, 6, 2, "15")):
        for am in (True, False):
            if not want("B2"):
                continue
            _print(f"\n--- B2: ARCHead rc={rc} rr={rr} resid={rb}b "
                   f"activation_metric={am} ---")
            Wh, s = build_archead(W0.float().cpu(), C, rc=rc, rr=rr, group=64,
                                 residual_bits=rb, activation_metric=am)
            head.weight.data.copy_(Wh.to(W0.dtype))
            measure(f"B2-{tag}{'' if am else '-nometric'}", s["bits_per_weight"],
                    extra={"rc": rc, "rr": rr, "residual_bits": rb,
                           "activation_metric": am,
                           "rel_metric_err": s["rel_metric_err"]})
            restore()

    # B3 -- residual VQ + the VQ-Logits extreme
    for vd, cb, st, tag in ((16, 8, 3, "1.5b"), (8, 8, 1, "1.0b")):
        if not want("B3-rvq"):
            continue
        _print(f"\n--- B3: RVQ vq_dim={vd} K=2^{cb} stages={st} ---")
        Wh, s = build_rvq(W0.float().cpu(), C, vq_dim=vd, codebook_bits=cb, stages=st,
                          iters=15, compute_device="cuda" if torch.cuda.is_available() else "cpu")
        head.weight.data.copy_(Wh.to(W0.dtype))
        measure(f"B3-rvq-{tag}", s["bits_per_weight"], extra=dict(s))
        restore()

    if want("B3-vql"):
        _print("\n--- B3: VQ-Logits K=1024 (the extreme) ---")
        Wh, s = build_vq_logits(W0.float().cpu(), K=1024, iters=15,
                                compute_device="cuda" if torch.cuda.is_available() else "cpu")
        head.weight.data.copy_(Wh.to(W0.dtype))
        measure("B3-vql-K1024", s["bits_per_weight"], extra=dict(s))
        restore()

    _print(f"\n[ladder] wrote {len(results)} rows to {out_json}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--calib-dir", default="./calib/lm_head_qwen3_0_6b")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--ppl-tokens", type=int, default=262144)
    ap.add_argument("--out", default="./results_eval/lm_head_ladder.json")
    ap.add_argument("--only", nargs="*", default=None,
                    help="substring filter over run names, e.g. --only B1-a B2")
    a = ap.parse_args()

    os.makedirs(a.calib_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.gates:
        gate_0a(a.model, a.calib_dir)
        gate_0c(a.model, a.calib_dir)
        gate_0d(a.model, a.calib_dir)
        _print("\n✅ ALL MODEL-LEVEL GATES PASSED")
    if a.ladder:
        run_ladder(a.model, a.calib_dir, a.ppl_tokens, a.out, runs=a.only)


if __name__ == "__main__":
    main()
