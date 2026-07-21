"""One-time warm-up: populate the leverage metric + v2 dynamic-alloc artifact
cache from already-collected expert covariances, WITHOUT a model forward sweep.

Rationale: ``expert_scores.pth`` may have an empty ``leverage`` sub-dict even
when ``expert_covariances.pth`` is present. Launching several leverage eval jobs
in parallel would each trigger a full 128-batch forward recompute and race on
the shared cache writes. This script computes ridge-leverage directly from the
cached covariances (cheap, CPU) and writes both caches once, so subsequent jobs
short-circuit via ``have_leverage_and_covariances`` and load the v2 artifact.

Usage:
    .venv/bin/python scripts/warm_leverage_cache.py --scores_dir <dir> [--lambda_ridge 1.0]
"""

import argparse
import os

import torch

from src.base.shared_utils import _layer_norm, _print
from src.calibration.channel_scoring.leverage import compute_ridge_leverage_scores
from src.calibration.channel_scoring.collect_covariance import _leverage_is_populated
from src.dynamic_active_param.precompute import build_alloc_artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--lambda_ridge", type=float, default=1.0)
    args = ap.parse_args()

    scores_path = os.path.join(args.scores_dir, "expert_scores.pth")
    cov_path = os.path.join(args.scores_dir, "expert_covariances.pth")

    expert_scores = torch.load(scores_path, map_location="cpu")
    if _leverage_is_populated(expert_scores):
        _print(f"[warm] leverage already populated in {scores_path}; skipping recompute")
    else:
        if not os.path.exists(cov_path):
            raise FileNotFoundError(
                f"{cov_path} missing — cannot warm leverage without covariances "
                "(run an eval with channel_metric=leverage to collect them first)."
            )
        _print(f"[warm] loading covariances from {cov_path} (large file, be patient)")
        expert_covariances = torch.load(cov_path, map_location="cpu")

        if not isinstance(expert_scores.get("leverage"), dict):
            expert_scores["leverage"] = {}

        n_layers = 0
        for layer_idx, layer_covs in expert_covariances.items():
            if not layer_covs:
                continue
            # infer (E, I) from the covariance entries
            eids = sorted(layer_covs.keys())
            I = layer_covs[eids[0]].shape[0]
            E = max(eids) + 1
            leverage_layer = torch.zeros((E, I), dtype=torch.float32)
            for eid, cov in layer_covs.items():
                leverage_layer[eid] = compute_ridge_leverage_scores(
                    cov.float(), lambda_ridge=args.lambda_ridge
                )
            expert_scores["leverage"][layer_idx] = _layer_norm(leverage_layer)
            n_layers += 1
        _print(f"[warm] computed leverage for {n_layers} MoE layers from covariances")
        torch.save(expert_scores, scores_path)
        _print(f"[warm] ✅ appended leverage to {scores_path}")

    # Build the v2 artifact cache once (channel_rank + prefix_sums + contrib).
    _print("[warm] building v2 dynamic-alloc artifact (leverage)")
    build_alloc_artifact(args.scores_dir, channel_metric="leverage", device="cpu", save=True)
    _print("[warm] ✅ done — jobs will short-circuit and load the cached artifact")


if __name__ == "__main__":
    main()
