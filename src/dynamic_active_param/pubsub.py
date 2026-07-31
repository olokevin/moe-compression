"""Level-2 offline artifact: shared-public-subspace redundancy penalty.

Realizes the runnable Level-2 selector in
``docs/exps/dynamic_active_param/plan/plan_level2_impl.md`` (§A2 ``pubsub``).

The idea: some output directions are written into by *many* experts ("public"
knowledge that should be loaded **once**), while the rest are "private" to a
single expert. Level-1's block-diagonal pivoted Cholesky cannot see this — two
experts that both carry a public direction each spend budget re-loading it. We

1. build the layer's aggregated output second moment
   ``M = sum_e W_down_e G_e W_down_e^T`` (d x d, PSD) and take its top-``r``
   eigenvectors ``U`` (the shared public basis);
2. deflate every expert's ``down_proj`` by ``U`` (``W_tilde = (I - U U^T) W``)
   and run the Level-1 batched pivoted Cholesky on the *private* coupling
   ``Theta_priv = G ⊙ (W_tilde^T W_tilde)`` -> private gains ``sigma_priv``;
3. store, per public direction and expert, the carrier coefficient
   ``c_{e,dir,j} = (U[:,dir]^T v_{e,j})`` where ``v_{e,j} = sqrt(G_e[j,j]) W[:,j]``
   is the activation-scaled channel output. Online we keep, for each public
   direction, the single co-activated channel with the largest ``|c|`` (dedup),
   then fill the rest with the Level-1 rule on ``sigma_priv``.

Everything is router-only online (touches no expert weights beyond this
artifact + the free router ``g``), and preserves prefix-contiguity for the
private tail.
"""

import os

import torch

from src.base.shared_utils import _print
from src.base.shared_utils.safe_isinstance import (
    _get_moe_block,
    _get_experts,
    _get_moe_intermediate_size,
    _get_num_hidden_layers,
)
from src.dynamic_active_param.pivchol import pivoted_cholesky_batched

__all__ = ["build_pubsub_artifact", "PubSubArtifact"]


class PubSubArtifact:
    """Per-layer Level-2 public/private tensors, keyed by MoE-layer index.

    Attributes (all stacked over L MoE layers):
        pivrank_priv: ``(L, E, I)`` long — private pivot rank of each channel.
        gains_priv:   ``(L, E, I)`` float — private marginal gains (rank order).
        carrier_idx:  ``(L, r, E)`` long — for each public direction, the
            *physical channel index* of the best-carrying channel in each expert.
        carrier_val:  ``(L, r, E)`` float — the |coefficient| of that channel.
        r:            number of public directions.
    """

    def __init__(self, pivrank_priv, gains_priv, carrier_idx, carrier_val, r, L, E, I):
        self.pivrank_priv = pivrank_priv
        self.gains_priv = gains_priv
        self.carrier_idx = carrier_idx
        self.carrier_val = carrier_val
        self.r = int(r)
        self.L = int(L)
        self.E = int(E)
        self.I = int(I)


def build_pubsub_artifact(
    model,
    scores_dir: str,
    r: int = 8,
    lambda_r: float = 1.0,
    device: str = "cuda",
    save: bool = True,
    verbose: bool = True,
    compute_device: str = None,
) -> PubSubArtifact:
    """Build (or load) the Level-2 pub/sub artifact.

    Needs the loaded model (``down_proj`` weights) and the cached
    ``expert_covariances.pth`` (Phase-A activation Gram) in ``scores_dir``.

    ``compute_device`` selects where the linalg runs; defaults to CPU to avoid
    poisoning a CUDA context that still holds a ``device_map='auto'`` shard (see
    the cublas-crash memory).
    """
    cache_path = os.path.join(scores_dir, f"pubsub_artifact_r{r}.pth")
    if os.path.exists(cache_path):
        if verbose:
            _print(f"[PubSub] Loading cached artifact from {cache_path}")
        p = torch.load(cache_path, map_location=device)
        return PubSubArtifact(
            pivrank_priv=p["pivrank_priv"].to(device),
            gains_priv=p["gains_priv"].to(device),
            carrier_idx=p["carrier_idx"].to(device),
            carrier_val=p["carrier_val"].to(device),
            r=int(p["r"]), L=int(p["L"]), E=int(p["E"]), I=int(p["I"]),
        )

    cov_path = os.path.join(scores_dir, "expert_covariances.pth")
    if not os.path.exists(cov_path):
        raise FileNotFoundError(
            f"{cov_path} missing — pubsub needs the activation Gram (covariances). "
            "Run an eval with channel_metric=leverage once to collect them."
        )
    if verbose:
        _print(f"[PubSub] Loading covariances from {cov_path}")
    expert_covariances = torch.load(cov_path, map_location="cpu")

    comp_dev = torch.device(compute_device) if compute_device is not None else torch.device("cpu")
    m = _get_moe_intermediate_size(model)
    num_layers = _get_num_hidden_layers(model)

    pivrank_layers, gains_layers = [], []
    cidx_layers, cval_layers = [], []
    mask_idx = 0
    for layer_idx in range(num_layers):
        moe_block = _get_moe_block(model, layer_idx)
        experts = _get_experts(moe_block)
        if experts is None:
            continue
        E = len(experts)

        # (E, d, m) stacked down_proj weights.
        Wd = torch.stack(
            [e.down_proj.weight.detach().float().cpu() for e in experts], dim=0
        ).to(comp_dev)                                          # (E, d, m)
        d = Wd.shape[1]

        # activation Gram (identity fallback for missing/dead experts).
        layer_covs = expert_covariances.get(layer_idx, {})
        G = torch.eye(m, device=comp_dev).unsqueeze(0).repeat(E, 1, 1)
        for eid, cov in layer_covs.items():
            G[eid] = cov.to(comp_dev).float()
        g_diag = torch.diagonal(G, dim1=1, dim2=2).clamp_min(0.0)  # (E, m)

        # ---- public basis U: top-r eigvecs of M = sum_e Wd_e G_e Wd_e^T (d,d) --
        # Wd_e G_e Wd_e^T = (Wd_e sqrt(G_e)) (.)^T; build via a symmetric product.
        Gsym = 0.5 * (G + G.transpose(1, 2))
        evals, evecs = torch.linalg.eigh(Gsym)                  # (E,m),(E,m,m)
        sqrtG = evecs @ (evals.clamp_min(0.0).sqrt().unsqueeze(-1) * evecs.transpose(1, 2))
        A = torch.bmm(Wd, sqrtG)                                # (E, d, m)
        M = torch.einsum("edm,efm->df", A, A)                   # (d, d) = sum_e Wd G Wd^T
        M = 0.5 * (M + M.t())
        _, Mevecs = torch.linalg.eigh(M)                        # ascending
        U = Mevecs[:, -r:]                                      # (d, r) top-r
        del Gsym, evals, evecs, sqrtG, A, M, Mevecs

        # ---- carrier coefficients c_{e,dir,j} = U[:,dir]^T v_{e,j} -------------
        # v_{e,j} = sqrt(G_e[j,j]) * Wd[:, j]; project onto each public dir.
        # (E, r, m): coefficient of channel j of expert e onto public dir.
        UtW = torch.einsum("dr,edm->erm", U, Wd)               # (E, r, m)
        coef = UtW * g_diag.sqrt().unsqueeze(1)                 # scale by sqrt(G_jj)
        coef_abs = coef.abs()

        # ---- private coupling: deflate Wd by U, then Level-1 pivoted Cholesky --
        # W_tilde = (I - U U^T) Wd  -> (E, d, m)
        UtW_full = torch.einsum("dr,edm->erm", U, Wd)          # (E, r, m)
        Wt = Wd - torch.einsum("dr,erm->edm", U, UtW_full)     # (E, d, m)
        del Wd, UtW_full
        Bpriv = torch.bmm(Wt.transpose(1, 2), Wt)              # (E, m, m) weight Gram
        del Wt
        theta_priv = G * Bpriv                                 # Hadamard (E, m, m)
        del G, Bpriv
        perm, gains = pivoted_cholesky_batched(theta_priv, lambda_r=lambda_r)
        del theta_priv
        pivrank = torch.argsort(perm, dim=1).to(torch.long)    # (E, m) channel->rank

        # best carrier per (dir, expert): argmax_j |coef| -> physical channel idx.
        # Stored as physical channel (the online score tensor is channel-indexed).
        best_val, best_j = coef_abs.max(dim=2)                 # (E, r) over channels
        cidx_layers.append(best_j.t().contiguous().cpu())      # (r, E) channel idx
        cval_layers.append(best_val.t().contiguous().cpu())    # (r, E)

        pivrank_layers.append(pivrank.cpu())
        gains_layers.append(gains.cpu())
        del coef, coef_abs, UtW, U
        mask_idx += 1
        if verbose and mask_idx % 8 == 0:
            _print(f"[PubSub] processed {mask_idx} MoE layers")

    pivrank_priv = torch.stack(pivrank_layers, dim=0)          # (L, E, m)
    gains_priv = torch.stack(gains_layers, dim=0)              # (L, E, m)
    carrier_idx = torch.stack(cidx_layers, dim=0)              # (L, r, E)
    carrier_val = torch.stack(cval_layers, dim=0)              # (L, r, E)
    L, E, I = pivrank_priv.shape

    if verbose:
        _print(
            f"[PubSub] pivrank_priv {tuple(pivrank_priv.shape)}, "
            f"carrier_idx {tuple(carrier_idx.shape)}, r={r}"
        )

    if save:
        torch.save(
            {
                "pivrank_priv": pivrank_priv, "gains_priv": gains_priv,
                "carrier_idx": carrier_idx, "carrier_val": carrier_val,
                "r": r, "L": L, "E": E, "I": I,
            },
            cache_path,
        )
        if verbose:
            _print(f"[PubSub] Cached artifact to {cache_path}")

    return PubSubArtifact(
        pivrank_priv=pivrank_priv.to(device),
        gains_priv=gains_priv.to(device),
        carrier_idx=carrier_idx.to(device),
        carrier_val=carrier_val.to(device),
        r=r, L=L, E=E, I=I,
    )
