import torch

from src.dynamic_active_param.pivchol import pivoted_cholesky_batched


def _psd(m, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(m, m, generator=g)
    return A @ A.t()


def test_perm_is_permutation():
    E, m = 4, 20
    theta = torch.stack([_psd(m, s) for s in range(E)], dim=0)
    perm, gains = pivoted_cholesky_batched(theta, lambda_r=1.0)
    assert perm.shape == (E, m) and gains.shape == (E, m)
    for e in range(E):
        assert torch.equal(torch.sort(perm[e]).values, torch.arange(m))


def test_gains_monotone_non_increasing():
    E, m = 3, 24
    theta = torch.stack([_psd(m, s + 10) for s in range(E)], dim=0)
    _, gains = pivoted_cholesky_batched(theta, lambda_r=1.0)
    # marginal gain must not increase along the pivot order (needed for the
    # online global threshold to cut a prefix).
    assert torch.all(gains[:, 1:] <= gains[:, :-1] + 1e-4), gains


def test_factor_reconstructs_theta_plus_ridge():
    # L L^T (with L the built factor, columns in pivot order) reconstructs
    # Theta + lambda*I. We re-run the reference (non-batched) to get L.
    m, lam = 16, 0.5
    theta = _psd(m, 3).unsqueeze(0)  # (1, m, m)
    perm, gains = pivoted_cholesky_batched(theta, lambda_r=lam)
    # rebuild via a plain reference pivoted Cholesky to validate the gains match
    T = (theta[0] + lam * torch.eye(m)).clone()
    diag = torch.diag(T).clone()
    chosen = torch.zeros(m, dtype=torch.bool)
    Lref = torch.zeros(m, m)
    ref_gain = torch.zeros(m)
    for t in range(m):
        d = diag.masked_fill(chosen, float("-inf"))
        p = int(d.argmax())
        ref_gain[t] = diag[p].clamp_min(0.0)
        chosen[p] = True
        s = ref_gain[t].clamp_min(1e-12).sqrt()
        col = T[:, p].clone()
        if t > 0:
            col = col - Lref[:, :t] @ Lref[p, :t]
        Lt = col / s
        Lt[p] = s
        Lref[:, t] = Lt
        diag = (diag - Lt * Lt).clamp_min(0.0)
        diag[p] = 0.0
    assert torch.allclose(gains[0], ref_gain, atol=1e-3), (gains[0], ref_gain)
    recon = Lref @ Lref.t()
    assert torch.allclose(recon, T, atol=1e-2), (recon - T).abs().max()


def test_batched_matches_loop():
    E, m = 5, 18
    theta = torch.stack([_psd(m, s + 100) for s in range(E)], dim=0)
    perm_b, gains_b = pivoted_cholesky_batched(theta, lambda_r=1.0)
    for e in range(E):
        perm_e, gains_e = pivoted_cholesky_batched(theta[e : e + 1], lambda_r=1.0)
        assert torch.equal(perm_b[e], perm_e[0])
        assert torch.allclose(gains_b[e], gains_e[0], atol=1e-4)


def test_redundancy_second_duplicate_gain_collapses():
    # Two channels sharing an identical direction: after the first is picked, the
    # duplicate's residual (hence marginal gain) collapses toward the ridge floor.
    # Build theta = V V^T with row 1 an exact copy of row 0 (genuine duplicate),
    # and use a tiny ridge so the collapse is unambiguous.
    m = 8
    V = torch.randn(m, m, generator=torch.Generator().manual_seed(7))
    V[1] = V[0]  # channel 1 has the identical output/activation direction as 0
    theta = V @ V.t()
    lam = 1e-3
    perm, gains = pivoted_cholesky_batched(theta.unsqueeze(0), lambda_r=lam)
    perm, gains = perm[0], gains[0]
    pos0 = int((perm == 0).nonzero())
    pos1 = int((perm == 1).nonzero())
    first, second = sorted([pos0, pos1])
    # the later duplicate collapses to ~ridge level, far below the first's gain.
    assert gains[second] < 0.1 * gains[first], (gains[first], gains[second])
    assert gains[second] < 10 * lam, gains[second]
