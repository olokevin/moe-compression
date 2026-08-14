"""lm_head compression / sparse-activation baselines.

Implements the shortlist of ``docs/exps/lm_head/plan/baselines.md``:

===========  ====================================================================
``B1``       frequency-tiered head (storage / prune / sparse-activate) -- the free
             static prior, and the floor every other method must clear
``B2``       ARCHead (arXiv:2608.02703) -- the SOTA storage/quality point
``B3``       group residual VQ (CARVQ) and VQ-Logits -- the highest param-reduction
             ceiling
``F3``       plain group RTN -- the honest naive floor
===========  ====================================================================

Driven entirely from the ``prune_kwargs.lm_head`` block of an eval YAML; see
:func:`src.lm_head.install.install_lm_head`.
"""

from src.lm_head.install import (
    bind_head_forward,
    install_lm_head,
    lm_head_eval_stats,
    unbind_head_forward,
)

__all__ = [
    "install_lm_head",
    "lm_head_eval_stats",
    "bind_head_forward",
    "unbind_head_forward",
]
