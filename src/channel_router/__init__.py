"""Channel-level router for sparse MLP activation.

Implements ``docs/exps/dynamic_active_param/plan/channel_router.md``: learn a
parameter-efficient map from a token's hidden state to the set of expert-FFN
intermediate channels that must be computed, with the FFN weights frozen.

Submodules
----------
``data``          capture I/O + the single oracle-importance definition (§0.1/§0.2)
``metrics``       recall / mass-recall / output-error / budget accounting (§0.3)
``model``         the router architecture (§1.1), every component flag-gated
``sinkhorn_topk`` differentiable exact-budget top-k (Stage C)
``tiles``         balanced Sinkhorn co-activation clustering (P5)
``baselines``     the §1.3 mandatory baselines
``scorers``       adapters that plug any of the above into the real forward pass
"""

__all__ = ["data", "metrics", "model", "sinkhorn_topk", "tiles", "baselines", "scorers"]
