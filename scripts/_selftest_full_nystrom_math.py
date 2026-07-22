"""Synthetic self-test of the identities used by full_nystrom_cov_analysis.py.

Builds a tiny stacked sparse MoE covariance from random data and checks:
  (1) energy identity: ‖ZᵀZ‖_F² == ‖ZZᵀ‖_F², and diag-block accounting closes.
  (2) push-through identity: diag((C+λI)⁻¹C) == a_iᵀ(AAᵀ+λI)⁻¹a_i (A=Z/√T).
  (3) block scatter G[idx,idx]+=Ge reconstructs the same Gram as dense Z.
Runs on CPU in <1s. No model, no repo deps.
"""
import torch

torch.manual_seed(0)
T, N, d, topk, lam = 200, 6, 5, 2, 0.7   # tokens, experts, d_mlp, per-token experts
# random routing: each token picks `topk` experts
logits = torch.randn(T, N)
sel = torch.topk(logits, topk, dim=-1).indices
idx_list, z_list = [], []
for e in range(N):
    idx_e = (sel == e).any(dim=1).nonzero(as_tuple=True)[0]
    idx_list.append(idx_e)
    z_list.append(torch.randn(idx_e.numel(), d))

# dense stacked Z (T x N*d) with each expert block populated on its routed rows
Z = torch.zeros(T, N * d)
for e in range(N):
    Z[idx_list[e], e * d:(e + 1) * d] = z_list[e]

C_full = Z.T @ Z / T                     # (Nd x Nd) full covariance

# (1) energy identity + diag-block accounting
G_dense = Z @ Z.T
total_S = float((C_full * C_full).sum())
total_G = float((G_dense @ G_dense.T).diagonal().sum()) / T**2  # ‖G/T‖_F² == ‖C‖_F²
diag_block = 0.0
for e in range(N):
    See = z_list[e].T @ z_list[e] / T
    diag_block += float((See * See).sum())
off = total_S - diag_block
print(f"(1) ‖C‖²={total_S:.6f}  ‖G/T‖²={total_G:.6f}  rel={abs(total_S-total_G)/total_S:.2e}")
print(f"    diag-block={diag_block:.6f} off-diag={off:.6f} off-frac={off/total_S:.4f}")
assert abs(total_S - total_G) / total_S < 1e-5

# (3) block scatter reconstructs G
G = torch.zeros(T, T)
for e in range(N):
    Ge = z_list[e] @ z_list[e].T
    idx = idx_list[e]
    G[idx.unsqueeze(1), idx.unsqueeze(0)] += Ge
print(f"(3) block-scatter G vs dense: max abs err = {float((G - G_dense).abs().max()):.2e}")
assert torch.allclose(G, G_dense, atol=1e-4)

# (2) push-through identity, full and per-expert-sliced
lev_direct = torch.linalg.solve(C_full + lam * torch.eye(N * d), C_full).diagonal()
M = torch.linalg.inv(G / T + lam * torch.eye(T))
lev_push = torch.zeros(N * d)
for e in range(N):
    idx = idx_list[e]
    Me = M.index_select(0, idx).index_select(1, idx)
    ze = z_list[e]
    tau = (ze * (Me @ ze)).sum(dim=0) / T
    lev_push[e * d:(e + 1) * d] = tau
err = float((lev_direct - lev_push).abs().max())
print(f"(2) push-through vs direct diag((C+λI)⁻¹C): max abs err = {err:.2e}")
assert err < 1e-4
print("ALL SELF-TESTS PASSED")
