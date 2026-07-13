"""Ridge leverage score computation for Nyström-based channel selection."""

import torch


def compute_ridge_leverage_scores(
    C: torch.Tensor,
    lambda_ridge: float = 1.0,
) -> torch.Tensor:
    """Compute ridge leverage scores: diag((C + lambda*I)^{-1} C).

    These scores rank the importance of each intermediate channel (neuron)
    for the Nyström approximation. Higher leverage = more important to keep.

    Mirrors the formula at src/compress/structured/nystrom.py:87-95.

    Args:
        C: (I, I) symmetric positive semi-definite covariance matrix.
           Typically the per-expert down_proj input covariance (z^T z / N).
        lambda_ridge: Ridge regularization parameter (must be > 0).

    Returns:
        scores: (I,) tensor of ridge leverage scores, one per channel.
    """
    if lambda_ridge <= 0.0:
        raise ValueError(f"lambda_ridge must be positive, got {lambda_ridge}")

    C = C.to(dtype=torch.float32)
    C = 0.5 * (C + C.T)  # symmetrize

    ridge_mat = C.clone()
    ridge_mat.diagonal().add_(lambda_ridge)

    # scores_i = [(C + lambda*I)^{-1} C]_{ii}
    scores = torch.linalg.solve(ridge_mat, C).diagonal()
    return scores
