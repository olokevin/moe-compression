"""**`input_only`** — one-pass input sparsity: the sparse read *is* the compute.

Every other scorer in this package is **two-pass**: a cheap proxy decides which
channels to keep, and then the *true* ``up``/``gate``/``down`` are re-read at full
width on those channels. ``input_sparse`` (:mod:`.sparse_probe`) is the best of
them, but that structure is exactly what its cost model charges for::

    input_sparse   used = rho_channel + 2·rho_input/3      (scoring + compute)
                          \\_________/   \\____________/
                           3·rho_ch/3      the discarded pass

The scoring pass reads ``2·rho_input`` of the expert and then throws the numbers
away; the compute pass re-reads all three matrices at ``rho_channel``. The kept
rows are billed twice, deliberately (see
:func:`sparse_probe.used_param_fraction`).

``input_only`` deletes the second pass. Per token:

1. rank the token's input coordinates by ``|x_i|`` and keep the top ``rho_input``
   fraction (optionally split across the token's K experts by ``g_e·|x_i|`` —
   ``input_alloc="router"``, the same allocator :mod:`.sparse_probe` uses);
2. run ``gate``/``up`` **on those coordinates only**, and treat the result as the
   FFN's actual intermediate::

       h̃_e = SiLU(W_gate^(e) x_sp) ⊙ (W_up^(e) x_sp)

3. rank all ``K·I`` of the token's channels by ``g_e·|h̃_{e,j}|`` on one global
   scale and keep the top ``B = rho_channel·K·I``;
4. run ``down`` on the kept channels only.

So there is no proxy and no scorer: **the same reads that decide are the reads
that compute**. Step 3 is ``oracle_mag_noW`` — the best selector in this repo —
applied to the intermediate that is actually being used, which makes the
selection *free*. The price is that ``h̃`` is a sparse-input approximation of the
true intermediate, so the error is no longer confined to the selection: every
kept channel carries a value error too. That trade is the whole experiment.

Read literally, the method applies **one** rule uniformly to all three matrices
of the expert FFN: *read only the largest-magnitude coordinates of your own
input*. For ``gate``/``up`` that input is ``x``; for ``down`` it is the
intermediate, whose top-magnitude coordinates are precisely the pooled top-B.
Hence ``rho_input = rho_channel`` is the natural symmetric operating point, and
``used`` then equals that single sparsity exactly (see below).

Cost model
----------
Units of one expert ``(I, H)`` matrix; a dense expert FFN is 3::

    gate : all I rows, rho_input of the H columns   -> rho_input
    up   : same                                     -> rho_input
    down : (H, I), only the kept channels' columns   -> rho_channel

    used = (2·rho_input + rho_channel) / 3

Three consequences worth stating, because each one differs from the two-pass frame:

* **No ``2/3`` floor.** ``oracle_mag``'s realized cut floors at
  ``(1+1+rho_channel)/3 ≥ 2/3`` because it must run gate+up at full width to
  decide. Here gate+up are *also* sparse, so ``used → 0`` as both knobs shrink.
* **No double billing.** Nothing is read twice, so this frame is not
  conservative in the way ``sparse_probe``'s is — it is just the read count.
* **The cheap axis flips.** ``∂used/∂rho_input = 2/3`` but
  ``∂used/∂rho_channel = 1/3``, so a unit of ``rho_channel`` now costs **half** a
  unit of ``rho_input``. Under ``input_sparse`` the ordering was the other way
  around (``rho_channel`` cost 1, ``rho_input`` cost 2/3), and "cut ``rho_input``
  first" was the standing advice. It does **not** carry over.

Anchors (both pinned in ``tests/test_input_only.py``):

* ``rho_input = 1`` reproduces ``oracle_mag_noW`` bit-for-bit, and its cost
  ``(2 + rho_channel)/3`` equals ``oracle_mag_noW``'s ``(1 + 1 + rho_channel)/3``
  exactly — so this frame agrees with the two-pass frame precisely where the two
  methods coincide.
* ``rho_channel = 1`` is pure input sparsity with no channel selection at all:
  ``used = (2·rho_input + 1)/3``, floored at ``1/3`` by the full-width ``down``.

There is no ``use_gate`` knob: SwiGLU needs both branches to *compute*, not just
to rank, so ``n = 2`` always. (``input_sparse``'s up-only variant is a scorer
ablation, which has no analogue once the scorer is the computation.)

Like the rest of the package the eval path is a **masking simulation**: ``down``
runs at full width on a zero-filled ``h̃`` rather than on a gathered ``(H, B)``
slice. The arithmetic is identical to a real gather; only the accounting changes.
The ``gate``/``up`` sparsity, by contrast, is *not* simulated — the sparse input
genuinely changes the numbers, which is why this method can lose accuracy that
``input_sparse`` at the same ``rho_input`` does not.
"""

from dataclasses import dataclass

from src.base.shared_utils import _print

__all__ = [
    "InputOnlyCfg",
    "used_param_fraction",
    "report_input_only_accounting",
    "print_input_only_accounting",
    "solve_symmetric",
]

# gate + up. Not a knob: both branches are needed to compute the SwiGLU
# intermediate, so there is no up-only variant of this method.
N_BRANCHES = 2


@dataclass
class InputOnlyCfg:
    """Per-layer state for the ``input_only`` forward.

    ``rho_input`` is the fraction of each token's input coordinates that
    ``gate``/``up`` read; ``input_alloc`` decides how that pooled read budget is
    split across the token's K experts (``uniform`` | ``router`` | ``router2``),
    reusing :func:`sparse_probe.allocate_input_reads`. The channel budget lives
    on the block as ``_dyn_B``, like every other cross-expert criterion.

    Deliberately holds no weights: unlike :class:`sparse_probe.SparseProbe` there
    is nothing to build, because the method reads the served ``gate``/``up``
    modules directly in the forward.
    """

    rho_input: float
    input_alloc: str = "uniform"


def used_param_fraction(rho_input: float, rho_channel: float) -> float:
    """Whole-FFN used-parameter fraction ``(2·rho_input + rho_channel)/3``.

    Both arguments are *keep* fractions in ``[0, 1]``. See the module docstring
    for the derivation and for why this is a plain read count rather than the
    conservative two-pass frame in :func:`sparse_probe.used_param_fraction`.
    """
    return (N_BRANCHES * float(rho_input) + float(rho_channel)) / 3.0


def solve_symmetric(used: float) -> float:
    """The sparsity that puts the symmetric point ``rho_input = rho_channel`` at ``used``.

    Trivially ``used`` itself — with ``2p + r = 3·used`` and ``p = r``, ``p = used``
    — which is why the symmetric operating point is the natural one to quote: a
    −75% used-parameter cut *is* "read a quarter of every input".
    """
    return float(used)


def report_input_only_accounting(rho_input: float, rho_channel: float) -> dict:
    """Accounting dict for one ``(rho_input, rho_channel)`` pair.

    ``input_sparse_used`` is the same pair costed in the two-pass frame, for a
    like-for-like read of how much the deleted second pass was worth.
    """
    p, r = float(rho_input), float(rho_channel)
    used = used_param_fraction(p, r)
    return {
        "used_param_fraction": used,
        "used_param_cut": 1.0 - used,
        "rho_input": p,
        "rho_channel": r,
        "gate_up_reads": N_BRANCHES * p / 3.0,
        "down_reads": r / 3.0,
        "gate_up_share": (N_BRANCHES * p / 3.0) / used if used > 0 else 0.0,
        # the same knobs under the two-pass frame, and what oracle_mag_noW would
        # pay for this channel budget: the two references this method is placed
        # against in the docs.
        "input_sparse_used": r + N_BRANCHES * p / 3.0,
        "oracle_mag_noW_used": (1.0 + 1.0 + r) / 3.0,
        # zero: nothing is stored beyond the served weights, and unlike the
        # unstructured `weight_sparse` family the read set is a contiguous
        # coordinate list, so there is no positional metadata either.
        "extra_storage_frac_of_experts": 0.0,
    }


def print_input_only_accounting(rho_input, rho_channel, input_alloc="uniform"):
    a = report_input_only_accounting(rho_input, rho_channel)
    _print(
        f"[input_only] rho_input={a['rho_input']:g} rho_channel={a['rho_channel']:g} "
        f"(gate+up {a['gate_up_reads']:.4f} + down {a['down_reads']:.4f}) "
        f"-> USED PARAMS={a['used_param_fraction']:.4f}, cut "
        f"{100 * a['used_param_cut']:.1f}%; input_alloc={input_alloc}; "
        f"one-pass (the sparse read IS the compute, no proxy, no re-read); "
        f"same knobs cost {a['input_sparse_used']:.4f} under two-pass input_sparse, "
        f"oracle_mag_noW would keep {a['oracle_mag_noW_used']:.4f}; "
        f"extra storage = 0.0% of expert weights"
    )
    return a
