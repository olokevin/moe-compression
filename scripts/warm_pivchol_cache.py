"""One-time warm-up: build the Level-1 pivoted-Cholesky artifact and cache it to
``scores_dir/pivchol_artifact.pth``.

Loads the model (for ``down_proj`` weights) and the cached
``expert_covariances.pth`` (Phase-A activation Gram), builds ``Theta_k = G_k⊙B_k``
per expert, runs batched ridge-pivoted Cholesky, and saves the small artifact
(pivot ranks + marginal gains, tens of MB). Run this once on the box that holds
the covariances; the resulting artifact can be copied to any eval box.

Usage:
    .venv/bin/python scripts/warm_pivchol_cache.py --config <eval-config.yaml>
"""

import argparse

import yaml

from src.base.argparser.e2e_args import E2EArguments
from src.base.models import get_model
from src.base.shared_utils import _print
from src.dynamic_active_param.pivchol import build_pivchol_artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lambda_r", type=float, default=1.0)
    cli = ap.parse_args()

    with open(cli.config, "r") as f:
        cfg = yaml.safe_load(f)
    args = E2EArguments(**cfg)
    args.post_init()
    lambda_r = args.prune_kwargs.get("dynamic_alloc", {}).get("lambda_r", cli.lambda_r)

    _print(f"[warm-pivchol] loading model {args.model_name_or_path}")
    model, tokenizer = get_model(args)
    model.eval()

    _print(f"[warm-pivchol] building pivchol artifact (lambda_r={lambda_r})")
    build_pivchol_artifact(
        model,
        scores_dir=args.scores_dir,
        lambda_r=lambda_r,
        device=args.device,
        save=True,
        verbose=True,
    )
    _print("[warm-pivchol] ✅ done — pivchol_artifact.pth cached in scores_dir")


if __name__ == "__main__":
    main()
