"""Phase 0 gates 0a / 0b: dequant round-trip and accounting vs the analytic byte count.

Run: ``.venv/bin/python -m src.lm_head.tests.test_quant`` from the repo root.
"""

import torch

from src.lm_head.accounting import ActiveParamContext, head_cost, print_lm_head_accounting
from src.lm_head.archead import archead_bits_per_weight, build_archead
from src.lm_head.quant import (
    bits_per_weight,
    metric_transform,
    quantize_int8_rowwise,
    quantize_rows_mixed,
    quantize_rtn_dequant,
    randomized_svd,
)
from src.lm_head.vq import build_rvq, build_vq_logits, rvq_bits_per_weight

V, D = 1024, 128


def _W():
    torch.manual_seed(0)
    return torch.randn(V, D) * 0.02


def _C():
    torch.manual_seed(1)
    H = torch.randn(4096, D) @ torch.diag(torch.linspace(0.1, 3.0, D))
    return H.T @ H / H.shape[0]


def test_bits_per_weight():
    assert bits_per_weight(16, 128) == 16.0
    assert bits_per_weight(17, 128) == 16.0          # >=16 means unquantized
    assert bits_per_weight(4, 128) == 4.125
    assert bits_per_weight(4, 64) == 4.25
    assert bits_per_weight(2, 128) == 2.125
    assert bits_per_weight(8, 0) == 8.0
    print("  ✅ bits/weight = bits + scale_bits/group, with >=16 meaning BF16")


def test_identity_is_bit_exact():
    """Gate 0a -- 16-bit 'quantization' must be a no-op, so freq_tier at
    tier_size=V / tail_bits=16 reproduces the dense logits bit-for-bit."""
    W = _W()
    assert torch.equal(quantize_rtn_dequant(W, bits=16, group=128), W)
    all_head = torch.ones(V, dtype=torch.bool)
    assert torch.equal(quantize_rows_mixed(W, all_head, 16, 16, 128), W)
    # even with a partial tier, 16/16 bits is the identity
    m = torch.zeros(V, dtype=torch.bool); m[: V // 3] = True
    assert torch.equal(quantize_rows_mixed(W, m, 16, 16, 128), W)
    print("  ✅ gate 0a: head_bits=tail_bits=16 is bit-exact on the weight")


def test_mixed_rows_touch_only_their_tier():
    W = _W()
    m = torch.zeros(V, dtype=torch.bool); m[:100] = True
    out = quantize_rows_mixed(W, m, 16, 4, 64)
    assert torch.equal(out[m], W[m]), "head tier at 16 bits must be untouched"
    assert not torch.equal(out[~m], W[~m]), "tail at 4 bits must actually change"
    # Tail error must sit at the analytic symmetric-RTN noise floor, not above it:
    # with qmax = 2^(b-1)-1 and a group max of ~3 sigma, the step is 3/7 sigma and
    # the round-off is uniform over +-step/2, i.e. rel err ~ 3/(7*sqrt(12)) = 0.124.
    rel = float((out[~m] - W[~m]).norm() / W[~m].norm())
    assert rel < 0.15, rel
    # groups never straddle the tier boundary: re-quantizing the tail alone agrees
    alone = quantize_rtn_dequant(W[~m], bits=4, group=64)
    assert torch.equal(out[~m], alone)
    print(f"  ✅ row-disjoint tiers: head exact, tail rel err {rel:.4f}, "
          "no group straddles the boundary")


def test_rtn_roundtrip_error_shrinks_with_bits():
    W = _W()
    errs = [float((quantize_rtn_dequant(W, bits=b, group=128) - W).norm() / W.norm())
            for b in (2, 3, 4, 8)]
    assert all(b < a for a, b in zip(errs, errs[1:])), errs
    # 8-bit floor: step = 3 sigma/127, rel err ~ 3/(127*sqrt(12)) = 0.0068
    assert errs[-1] < 0.01, errs
    # each *consecutive* extra bit should roughly halve the error (2->3, 3->4)
    for a, b in zip(errs[:2], errs[1:3]):
        assert 0.3 < b / a < 0.7, errs
    print(f"  ✅ RTN round-trip error monotone in bits: "
          + ", ".join(f"{b}b={e:.5f}" for b, e in zip((2, 3, 4, 8), errs)))


def test_int8_rowwise():
    W = _W()
    q = quantize_int8_rowwise(W)
    rel = float((q - W).norm() / W.norm())
    assert rel < 0.01, rel
    # per-row: a row scaled up by 1000 must quantize with the same *relative* error
    W2 = W.clone(); W2[0] *= 1000.0
    q2 = quantize_int8_rowwise(W2)
    r0 = float((q2[0] - W2[0]).norm() / W2[0].norm())
    r1 = float((q2[1] - W2[1]).norm() / W2[1].norm())
    assert abs(r0 - r1) < 0.01, (r0, r1)
    print(f"  ✅ row-wise INT8: rel err {rel:.5f}, scale-invariant across rows")


def test_metric_transform_is_a_square_root():
    C = _C()
    Tp, Tp_inv, info = metric_transform(C, p=0.5, ridge=1e-6)
    # Tp @ Tp should reproduce the damped C
    cbar = torch.diagonal(C).mean()
    Cl = C + 1e-6 * cbar * torch.eye(D)
    assert torch.allclose(Tp @ Tp, Cl, atol=1e-4 * float(Cl.abs().max())), \
        float((Tp @ Tp - Cl).abs().max())
    assert torch.allclose(Tp @ Tp_inv, torch.eye(D), atol=1e-3)
    # p=1 gives exactly C
    Tp1, _, _ = metric_transform(C, p=1.0, ridge=1e-6)
    assert torch.allclose(Tp1, Cl, atol=1e-4 * float(Cl.abs().max()))
    print(f"  ✅ metric transform: T_(1/2)^2 == C_lambda, T_p T_p^-1 == I, "
          f"cond={info['cond']:.3e}")


def test_randomized_svd_matches_full():
    torch.manual_seed(0)
    A = torch.randn(400, 64) @ torch.randn(64, 64)
    U, S, Vh = randomized_svd(A, rank=8, n_iter=8)
    S_full = torch.linalg.svdvals(A)[:8]
    assert torch.allclose(S, S_full, rtol=2e-2), (S, S_full)
    # rank-8 reconstruction error matches the analytic tail
    rec = (U * S.unsqueeze(0)) @ Vh
    tail = torch.linalg.svdvals(A)[8:].pow(2).sum().sqrt()
    assert abs(float((A - rec).norm()) - float(tail)) < 0.05 * float(tail)
    print("  ✅ randomized SVD reproduces the top singular values and tail energy")


def test_archead_beats_matched_rtn_in_the_activation_metric():
    """The paper's whole claim, at toy scale: at matched storage ARCHead should have
    lower error *in the activation metric* than plain group INT4."""
    W, C = _W(), _C()
    W_arc, s = build_archead(W, C, rc=4, rr=4, group=64, verbose=False)
    W_rtn = quantize_rtn_dequant(W, bits=4, group=64)

    def cerr(Dm):
        return float(torch.einsum("vd,de,ve->", Dm, C, Dm).clamp_min(0).sqrt())

    e_arc, e_rtn = cerr(W_arc - W), cerr(W_rtn - W)
    assert e_arc < e_rtn, (e_arc, e_rtn)
    # and the storage must be in the same ballpark as INT4
    assert 0.20 < s["storage_frac_of_bf16"] < 0.45, s["storage_frac_of_bf16"]
    print(f"  ✅ ARCHead metric err {e_arc:.4f} < matched INT4 {e_rtn:.4f} "
          f"({100 * e_arc / e_rtn:.1f}%), storage {100 * s['storage_frac_of_bf16']:.1f}% of BF16")


def test_archead_activation_metric_ablation():
    """Fitting the correction in C must beat fitting it in plain Frobenius, measured
    in C -- that is the mechanism the pilots independently confirmed dominates."""
    W, C = _W(), _C()
    Wa, _ = build_archead(W, C, rc=4, rr=8, group=64, activation_metric=True, verbose=False)
    Wf, _ = build_archead(W, C, rc=4, rr=8, group=64, activation_metric=False, verbose=False)

    def cerr(Dm):
        return float(torch.einsum("vd,de,ve->", Dm, C, Dm).clamp_min(0).sqrt())

    assert cerr(Wa - W) < cerr(Wf - W), (cerr(Wa - W), cerr(Wf - W))
    print(f"  ✅ activation-metric fit {cerr(Wa - W):.4f} < Frobenius fit "
          f"{cerr(Wf - W):.4f} when measured in C")


def test_archead_analytic_accounting_matches_paper_band():
    """Gate 0b -- our analytic bit count must land in ARCHead's measured 25-27% band
    on the exact head shape the paper reports for VibeThinker-3B (151936 x 2048)."""
    a = archead_bits_per_weight(151936, 2048, rc=10, rr=6, group=64)
    assert 0.25 <= a["storage_frac_of_bf16"] <= 0.28, a
    # components must sum to the total
    assert abs(a["residual_bpw"] + a["core_bpw"] + a["correction_bpw"]
               - a["bits_per_weight"]) < 1e-9
    # the residual is the dominant term, as the paper's packing implies
    assert a["residual_bpw"] / a["bits_per_weight"] > 0.95
    print(f"  ✅ gate 0b: ARCHead on 151936x2048 = {a['bits_per_weight']:.3f} bits/weight "
          f"({100 * a['storage_frac_of_bf16']:.1f}% of BF16; paper measures 25.8%)")


def test_vq_bits_and_reconstruction():
    W, C = _W(), _C()
    W_hat, s = build_rvq(W, C, vq_dim=16, codebook_bits=6, stages=2, iters=8, verbose=False)
    # analytic bits/weight must equal stages*bits/vq_dim up to the codebook term
    assert abs(s["codes_bpw"] - 2 * 6 / 16) < 1e-9, s
    assert s["rel_fro_err"] < 1.0, s
    # more stages must reduce the error
    _, s3 = build_rvq(W, C, vq_dim=16, codebook_bits=6, stages=3, iters=8, verbose=False)
    assert s3["rel_fro_err"] < s["rel_fro_err"], (s["rel_fro_err"], s3["rel_fro_err"])
    print(f"  ✅ RVQ: codes_bpw exact, rel err {s['rel_fro_err']:.4f} (2 stages) -> "
          f"{s3['rel_fro_err']:.4f} (3 stages)")

    a = rvq_bits_per_weight(151936, 2048, vq_dim=16, codebook_bits=8, stages=3)
    assert abs(a["codes_bpw"] - 1.5) < 1e-9
    # The per-position codebooks are small but NOT free: 3 stages x 128 positions x
    # 256 codes x 16 dims x fp16 = 25.2 Mbit = 0.081 bits/weight, i.e. ~5% of the
    # budget. Counting it is what keeps the "~1.6 bits" claim honest.
    assert 0.05 < a["codebook_bpw"] < 0.12, a
    assert a["codebook_bpw"] / a["bits_per_weight"] < 0.08, a
    print(f"  ✅ RVQ at V=151936: {a['bits_per_weight']:.4f} bits/weight "
          f"(codes {a['codes_bpw']:.3f} + books {a['codebook_bpw']:.4f} = "
          f"{100 * a['codebook_bpw'] / a['bits_per_weight']:.1f}% of the budget)")


def test_vq_logits_is_extreme_and_lossy():
    W = _W()
    W_hat, s = build_vq_logits(W, K=32, iters=10, verbose=False)
    assert s["storage_frac_of_bf16"] < 0.10, s
    assert s["rel_fro_err"] > 0.5, s          # one code per 32 rows is brutal, as expected
    assert s["n_used_codes"] <= 32
    # every row must be exactly one of the codes
    assert torch.unique(W_hat, dim=0).shape[0] <= 32
    print(f"  ✅ VQ-Logits: {100 * s['storage_frac_of_bf16']:.2f}% of BF16, "
          f"rel err {s['rel_fro_err']:.3f}, {s['n_used_codes']} codes used")


def test_accounting_analytic_byte_count():
    """Gate 0b -- head_cost must match a hand-computed byte count to <0.1%."""
    V_, D_ = 151936, 2048
    # dense BF16
    c = head_cost(V_, D_, 16.0)
    assert c["dense_bytes"] == V_ * D_ * 2
    assert abs(c["storage_frac_of_bf16"] - 1.0) < 1e-12
    assert abs(c["used_head_params_bf16eq"] - V_ * D_) < 1e-6

    # INT4 g128: 4.125 bits -> 25.78% of BF16
    c4 = head_cost(V_, D_, bits_per_weight(4, 128))
    assert abs(c4["storage_frac_of_bf16"] - 4.125 / 16) < 1e-12
    hand = V_ * D_ * 4.125 / 8
    assert abs(c4["storage_bytes"] - hand) / hand < 1e-9

    # B1-a: stored dense BF16, only 4096 rows read
    ca = head_cost(V_, D_, 16.0, read_rows=4096, read_bits_per_weight=16.0)
    assert ca["storage_frac_of_bf16"] == 1.0
    assert abs(ca["used_head_params_bf16eq"] - 4096 * D_) < 1e-6
    assert abs(ca["read_frac_of_bf16"] - 4096 / V_) < 1e-12

    # read_rows is clamped to V
    assert head_cost(V_, D_, 16.0, read_rows=10 * V_)["read_rows"] == V_
    print("  ✅ gate 0b: head_cost matches hand-computed bytes for BF16 / INT4 / B1-a")


def test_accounting_separates_params_from_bytes():
    """Parameter count and precision are independent axes. Quantization must move
    ONLY the byte axis; structural methods must move the parameter axis."""
    V_, D_ = 151936, 2048
    dense = V_ * D_

    # pure quantization: same parameter count, fewer bits
    q = head_cost(V_, D_, bits_per_weight(4, 128))
    assert q["stored_params"] == dense and q["read_params_per_token"] == dense
    assert q["stored_param_frac"] == 1.0 and q["read_param_frac"] == 1.0
    assert abs(q["storage_frac_of_bf16"] - 4.125 / 16) < 1e-12

    # row pruning (B1-p): T*D parameters, BF16
    T = 8192
    p = head_cost(V_, D_, 16.0 * T / V_, read_rows=T, read_bits_per_weight=16.0,
                  stored_params=T * D_)
    assert p["stored_params"] == T * D_
    assert abs(p["stored_param_frac"] - T / V_) < 1e-12
    assert abs(p["read_param_frac"] - T / V_) < 1e-12

    # sparse activation (B1-a): all params stored, T*D read
    a = head_cost(V_, D_, 16.0, read_rows=T, read_bits_per_weight=16.0)
    assert a["stored_param_frac"] == 1.0
    assert abs(a["read_param_frac"] - T / V_) < 1e-12

    # low-rank: (V+D)*r parameters, none of them a row subset
    r = 512
    lr = head_cost(V_, D_, 16.0 * (V_ + D_) * r / dense,
                   stored_params=(V_ + D_) * r, read_params=(V_ + D_) * r)
    assert lr["stored_params"] == (V_ + D_) * r
    assert abs(lr["stored_param_frac"] - (V_ + D_) * r / dense) < 1e-12
    # for BF16 factors the parameter fraction and the byte fraction coincide
    assert abs(lr["stored_param_frac"] - lr["storage_frac_of_bf16"]) < 1e-9
    print(f"  ✅ params and bytes are separate axes: INT4 = 100.00% of params / "
          f"{100 * q['storage_frac_of_bf16']:.2f}% of bytes; "
          f"B1-p T=8192 = {100 * p['stored_param_frac']:.2f}% of params; "
          f"low-rank r=512 = {100 * lr['stored_param_frac']:.2f}% of params")


def test_accounting_denominators():
    """The two denominators of plan section 1 must come out at 1.02% / 9.28%."""
    V_, D_ = 151936, 2048
    ctx = ActiveParamContext(
        total_params=30_532_122_624, active_params=3_353_000_000,
        head_params=V_ * D_, is_moe=True, active_params_pruned=2_019_000_000,
    )
    out = print_lm_head_accounting(head_cost(V_, D_, bits_per_weight(4, 128)), ctx, "int4")
    assert abs(100 * out["head_share_of_total"] - 1.02) < 0.02, out["head_share_of_total"]
    assert abs(100 * out["head_share_of_active"] - 9.28) < 0.02, out["head_share_of_active"]
    # INT4 -> -6.9% of active (plan says -6.96% at exactly 25% storage)
    assert -7.1 < 100 * out["delta_active_used"] < -6.8, out["delta_active_used"]
    # a free head is the ceiling: -9.28%
    free = print_lm_head_accounting(head_cost(V_, D_, 0.0, read_rows=0), ctx, "free")
    assert abs(100 * free["delta_active_used"] + 9.28) < 0.02, free["delta_active_used"]
    print("  ✅ denominators reproduce plan section 1: head = 1.02% of total / "
          "9.28% of active; INT4 = -6.9% active; free head = -9.28% (the ceiling)")


if __name__ == "__main__":
    print("Phase 0 gates 0a/0b -- numerics round-trip + analytic accounting")
    test_bits_per_weight()
    test_identity_is_bit_exact()
    test_mixed_rows_touch_only_their_tier()
    test_rtn_roundtrip_error_shrinks_with_bits()
    test_int8_rowwise()
    test_metric_transform_is_a_square_root()
    test_randomized_svd_matches_full()
    test_archead_beats_matched_rtn_in_the_activation_metric()
    test_archead_activation_metric_ablation()
    test_archead_analytic_accounting_matches_paper_band()
    test_vq_bits_and_reconstruction()
    test_vq_logits_is_extreme_and_lossy()
    test_accounting_analytic_byte_count()
    test_accounting_separates_params_from_bytes()
    test_accounting_denominators()
    print("ALL QUANT/ACCOUNTING TESTS PASSED")
