"""Phase 0 gates for S1 (screen-and-refine).

Run: ``.venv/bin/python -m src.lm_head.tests.test_screen_refine`` from the repo root.

The gates that matter for this method specifically:

0a  ``r0 = D`` and ``N = V`` must reproduce the dense logits *exactly* -- if the
    identity configuration drifts, every measured number is suspect.
0b  the accounting must equal a hand-computed read/storage count, including the
    ``D^2`` the rotation costs and the fact that storage goes *up*, not down.
0e  the screen must be the thing that changes: refined (candidate) logits are
    bit-identical to dense, and only non-candidates carry error.
0f  a per-token screen must beat a static one at matched reads on data with
    per-token structure -- the property the whole method rests on.
"""

import torch
import torch.nn.functional as F

from src.lm_head.accounting import ActiveParamContext, head_cost, print_lm_head_accounting
from src.lm_head.screen_refine import build_screen_refine, screen_refine_cost

V, D = 2048, 64


def _W():
    torch.manual_seed(0)
    return torch.randn(V, D) * 0.02


def _H(n=256, spiky=True):
    """Hidden states with a per-token *rotating* set of large coordinates.

    ``spiky=True`` gives each state its own random subset of dominant coordinates, so
    the average energy ordering is flat while the per-token one is sharp -- exactly the
    structure a per-token screen exploits and a static subspace cannot.
    """
    torch.manual_seed(1)
    H = torch.randn(n, D) * 0.1
    if spiky:
        for i in range(n):
            idx = torch.randperm(D)[: D // 8]
            H[i, idx] *= 12.0
    return H


def _apply(W, H, U, col, r0, N, fixed_sel=None):
    coef = H @ U
    ck = torch.zeros_like(coef)
    if fixed_sel is None:
        idx = (coef.abs() * col).topk(r0, -1).indices
        ck.scatter_(1, idx, coef.gather(1, idx))
    else:
        ck[:, fixed_sel] = coef[:, fixed_sel]
    coarse = (ck @ U.T) @ W.T
    full = H @ W.T
    cand = coarse.topk(N, -1).indices
    out = coarse.clone()
    out.scatter_(1, cand, full.gather(1, cand))
    return out, full, cand


def test_identity_reproduces_dense():
    """Gate 0a: r0 = D, N = V is the dense head."""
    W, H = _W(), _H()
    U, col, _, _ = build_screen_refine(W, H.T @ H / H.shape[0], basis="ceig", verbose=False)
    out, full, _ = _apply(W, H, U, col, D, V)
    assert torch.allclose(out, full, atol=1e-4), (out - full).abs().max()
    # and with no rotation at all
    U2, col2, _, _ = build_screen_refine(W, None, basis="raw", verbose=False)
    assert torch.equal(U2, torch.eye(D))
    out2, full2, _ = _apply(W, H, U2, col2, D, V)
    assert torch.allclose(out2, full2, atol=1e-5)
    print("0a  identity config reproduces the dense head                     OK")


def test_refined_rows_are_bit_identical():
    """Gate 0e: candidates carry zero error; only the tail is approximate."""
    W, H = _W(), _H()
    U, col, _, _ = build_screen_refine(W, H.T @ H / H.shape[0], basis="ceig", verbose=False)
    out, full, cand = _apply(W, H, U, col, D // 4, 128)
    assert torch.equal(out.gather(1, cand), full.gather(1, cand))
    keep = torch.zeros_like(out, dtype=torch.bool).scatter_(1, cand, True)
    assert (out - full).masked_fill(keep, 0.0).abs().max() > 0, "the tail should differ"
    # every logit is finite -- the failure mode that made B1-a's perplexity infinite
    assert torch.isfinite(out).all()
    print("0e  refined logits bit-identical to dense, tail finite            OK")


def test_per_token_beats_static_at_matched_reads():
    """Gate 0f: the load-bearing claim, on data built to have per-token structure."""
    W, H = _W(), _H()
    C = H.T @ H / H.shape[0]
    U, col, static_score, _ = build_screen_refine(W, C, basis="ceig", verbose=False)
    r0, N = D // 4, 64
    sel = static_score.topk(r0).indices
    full = H @ W.T
    lpd = F.log_softmax(full, -1)

    def kl(out):
        return float((lpd.exp() * (lpd - F.log_softmax(out, -1))).sum(-1).mean())

    kl_ad = kl(_apply(W, H, U, col, r0, N)[0])
    kl_st = kl(_apply(W, H, U, col, r0, N, fixed_sel=sel)[0])
    assert kl_ad < kl_st, (kl_ad, kl_st)
    print(f"0f  per-token screen KL {kl_ad:.5f} < static screen KL {kl_st:.5f}   OK")


def test_accounting_matches_hand_count():
    """Gate 0b: reads and storage, including the rotation, against hand arithmetic."""
    r0, N = 16, 256
    sc = screen_refine_cost(V, D, r0, N)
    assert sc["read_params"] == r0 * V + N * (D - r0) + D * D
    assert sc["stored_params"] == V * D + D * D
    # storage goes UP: this is a read method, and claiming otherwise would be the
    # exact category error the results doc keeps quantization out of Part 1 for.
    assert sc["stored_param_frac"] > 1.0
    assert abs(sc["bits_per_weight"] - 16.0 * (1 + D / V)) < 1e-9

    cost = head_cost(V, D, sc["bits_per_weight"], read_rows=round(sc["read_params"] / D),
                     read_bits_per_weight=16.0, stored_params=sc["stored_params"],
                     read_params=sc["read_params"])
    assert cost["stored_params"] == V * D + D * D
    assert cost["read_params_per_token"] == sc["read_params"]
    assert abs(cost["read_param_frac"] - sc["read_params"] / (V * D)) < 1e-12
    # bytes: nothing is quantized, so read bytes are 2 per read parameter
    assert abs(cost["read_bytes_per_token"] - 2.0 * sc["read_params"]) < 2.0 * D

    ctx = ActiveParamContext(total_params=10 * V * D, active_params=4 * V * D,
                             head_params=V * D)
    out = print_lm_head_accounting(cost, ctx, label="screen_refine")
    # read-axis saving against the active budget, hand-checked
    exp = -(1 - sc["read_params"] / (V * D)) * (V * D) / (4 * V * D)
    assert abs(out["delta_active_used"] - exp) < 1e-3, (out["delta_active_used"], exp)
    print("0b  read / storage / active accounting matches hand arithmetic    OK")


def test_col_norm_is_needed():
    """The screen score needs the column norm; ranking |coef| alone is worse."""
    W, H = _W(), _H()
    C = H.T @ H / H.shape[0]
    # make the columns of W U differ a lot in scale so the factor matters
    W = W * torch.linspace(0.2, 5.0, D).unsqueeze(0)
    U, col, _, _ = build_screen_refine(W, C, basis="ceig", verbose=False)
    full = H @ W.T
    lpd = F.log_softmax(full, -1)

    def kl(out):
        return float((lpd.exp() * (lpd - F.log_softmax(out, -1))).sum(-1).mean())

    kl_with = kl(_apply(W, H, U, col, D // 4, 64)[0])
    kl_without = kl(_apply(W, H, U, torch.ones_like(col), D // 4, 64)[0])
    assert kl_with < kl_without, (kl_with, kl_without)
    print(f"0g  |coef|*||W u_i|| KL {kl_with:.5f} < |coef| alone {kl_without:.5f}      OK")


if __name__ == "__main__":
    test_identity_reproduces_dense()
    test_refined_rows_are_bit_identical()
    test_per_token_beats_static_at_matched_reads()
    test_accounting_matches_hand_count()
    test_col_norm_is_needed()
    print("\nall S1 gates pass")
