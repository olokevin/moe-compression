"""Dynamic per-token, per-expert active-parameter allocation.

Distributes a fixed channel budget unevenly across a token's top-K experts —
more channels to experts that matter more for that token — while keeping the
total activated expert-FFN params at a preset budget (masking simulation, so
exact accuracy at budget with no variable-width matmuls).

See docs/results/dynamic_active_param/plan/plan_initial.md.
"""

from src.dynamic_active_param.allocate import allocate_budgets
from src.dynamic_active_param.precompute import AllocArtifact, build_alloc_artifact
from src.dynamic_active_param.install import install_dynamic_alloc

__all__ = [
    "allocate_budgets",
    "AllocArtifact",
    "build_alloc_artifact",
    "install_dynamic_alloc",
]
