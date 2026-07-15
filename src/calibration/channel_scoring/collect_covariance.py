"""Eval-time on-the-fly collection of per-expert down_proj-input covariances
and ridge-leverage scores for Nyström reconstruction.

This restores the "Step 2.5" pass used by the one-shot leverage+Nyström runs
(see docs/results/attribution_guided/nystrom.md): the base scoring stage may
predate the Nyström feature, so its ``expert_scores.pth`` lacks the ``leverage``
metric and there is no ``expert_covariances.pth``. Rather than re-run the full
per-layer distillation scorer, we do a single hooked forward pass over a
calibration set, accumulating each expert's ``down_proj`` input covariance
``zᵀz / N`` (averaged over batches, mirroring
``collector/attn_mlp.py`` + ``main.py``), then compute the ridge leverage
``diag((C+λI)⁻¹C)`` per channel.

Collected artifacts are written back into ``scores_dir`` (leverage appended to
``expert_scores.pth``; covariances saved to ``expert_covariances.pth``) so that
subsequent eval runs load them from disk instead of recollecting.
"""

import os
import torch
from tqdm import tqdm

from src.base.datasets import load_datasets
from src.base.shared_utils import _layer_norm, _print
from src.base.shared_utils.safe_isinstance import (
    _get_num_experts,
    _get_moe_intermediate_size,
    _get_num_hidden_layers,
    _is_moe_block,
)
from src.calibration.channel_scoring.leverage import compute_ridge_leverage_scores

__all__ = [
    "have_leverage_and_covariances",
    "ensure_leverage_and_covariances",
]


def _leverage_is_populated(expert_scores) -> bool:
    lev = expert_scores.get("leverage") if isinstance(expert_scores, dict) else None
    return isinstance(lev, dict) and len(lev) > 0


def have_leverage_and_covariances(scores_dir: str) -> bool:
    """True iff scores_dir already holds a populated leverage metric AND covariances."""
    cov_path = os.path.join(scores_dir, "expert_covariances.pth")
    scores_path = os.path.join(scores_dir, "expert_scores.pth")
    if not (os.path.exists(cov_path) and os.path.exists(scores_path)):
        return False
    try:
        expert_scores = torch.load(scores_path, map_location="cpu")
    except Exception:
        return False
    return _leverage_is_populated(expert_scores)


def _make_cov_hook(store, key, device):
    """Forward hook accumulating per-batch zᵀz/N covariance for one expert."""

    def hook(module, inp, out):
        z = inp[0] if isinstance(inp, (tuple, list)) else inp
        z = z.detach()
        if z.ndim == 3:
            z = z.reshape(-1, z.shape[-1])
        if z.shape[0] == 0:
            return
        z = z.float()
        cov_batch = (z.t() @ z) / max(z.shape[0], 1)  # (I, I), per-batch normalized
        cov_batch = cov_batch.cpu()
        entry = store.get(key)
        if entry is None:
            store[key] = [cov_batch, 1]
        else:
            entry[0].add_(cov_batch)
            entry[1] += 1

    return hook


# Covariance-collection recipe used by the one-shot leverage+Nyström runs
# (docs/results/attribution_guided/nystrom.md): c4, 128 batches × bs16, seq 512.
# Kept independent of args.max_seq_length, which also drives the eval harness
# context length and must not be shrunk to the calibration length.
_DEFAULT_COV_RECIPE = {
    "calib_datasets": ["c4"],
    "calib_batches": 128,
    "batch_size": 16,
    "max_seq_length": 512,
}


@torch.no_grad()
def ensure_leverage_and_covariances(
    model,
    tokenizer,
    args,
    lambda_ridge: float = 1.0,
    verbose: bool = True,
):
    """Collect (or load) per-expert covariances + ridge-leverage into scores_dir.

    Returns the ``expert_covariances`` dict ``{layer_idx: {eid: (I,I) cpu tensor}}``.
    If the artifacts already exist in ``scores_dir`` they are simply loaded.

    The collection recipe (dataset / #batches / batch size / seq len) defaults to
    the doc's values and can be overridden per-run via
    ``prune_kwargs.cov_collect_kwargs`` in the YAML.
    """
    scores_dir = args.scores_dir
    cov_path = os.path.join(scores_dir, "expert_covariances.pth")
    scores_path = os.path.join(scores_dir, "expert_scores.pth")

    if have_leverage_and_covariances(scores_dir):
        _print(f"[Step 2.5] Leverage + covariances already present in {scores_dir}; loading covariances")
        return torch.load(cov_path, map_location="cpu")

    device = args.device
    E = _get_num_experts(model)
    I = _get_moe_intermediate_size(model)
    L = _get_num_hidden_layers(model)

    recipe = dict(_DEFAULT_COV_RECIPE)
    recipe.update(args.prune_kwargs.get("cov_collect_kwargs", {}) or {})
    calib_datasets = recipe["calib_datasets"]
    calib_batches = recipe["calib_batches"]
    batch_size = recipe["batch_size"]
    max_seq_length = recipe["max_seq_length"]

    _print(
        f"[Step 2.5] Collecting ridge-leverage scores + covariances "
        f"(nystrom_reconstruct=True, metric=leverage): "
        f"datasets={calib_datasets}, calib_batches={calib_batches}, "
        f"batch_size={batch_size}, max_seq_length={max_seq_length}, lambda_ridge={lambda_ridge}"
    )

    calib_dataset = load_datasets(
        calib_datasets[0], tokenizer, max_samples=calib_batches * batch_size, max_length=max_seq_length
    )

    model.eval()
    if getattr(model, "hf_device_map", None) is None:
        model.to(device)

    # Register a covariance-accumulating hook on every expert down_proj across
    # all MoE layers, then run a single forward sweep over the calib set.
    store = {}  # {(layer_idx, eid): [cov_sum (I,I) cpu, count]}
    hooks = []
    layers = model.model.layers
    moe_layer_indices = []
    for layer_idx in range(L):
        mlp = layers[layer_idx].mlp
        if not _is_moe_block(mlp):
            continue
        moe_layer_indices.append(layer_idx)
        for eid, expert in enumerate(mlp.experts):
            h = expert.down_proj.register_forward_hook(
                _make_cov_hook(store, (layer_idx, eid), device)
            )
            hooks.append(h)

    try:
        for n_batches, batch_start in enumerate(
            tqdm(
                range(0, len(calib_dataset), batch_size),
                desc="Collecting expert covariances",
                total=calib_batches,
                disable=not verbose,
            ),
            start=1,
        ):
            if n_batches > calib_batches:
                break

            raw = calib_dataset[batch_start : batch_start + batch_size]
            batch_texts = []
            for x in raw:
                if x is None:
                    continue
                if not isinstance(x, str):
                    x = str(x)
                x = x.strip()
                if x:
                    batch_texts.append(x)
            if not batch_texts:
                continue

            enc = tokenizer(
                batch_texts,
                max_length=max_seq_length,
                padding=True,
                pad_to_multiple_of=8,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            model(**inputs, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    _print(f"[Covariance] collected covariances for {len(moe_layer_indices)} layers")

    # Assemble covariances + leverage per layer, matching main.py's layout.
    expert_scores = torch.load(scores_path, map_location=device)
    if not isinstance(expert_scores.get("leverage"), dict):
        expert_scores["leverage"] = {}

    expert_covariances = {}
    for layer_idx in moe_layer_indices:
        layer_covs = {}
        leverage_layer = torch.zeros((E, I), dtype=torch.float32, device=device)
        for eid in range(E):
            entry = store.get((layer_idx, eid))
            if entry is None or entry[1] == 0:
                continue
            cov = entry[0] / entry[1]  # average over batches
            layer_covs[eid] = cov
            leverage = compute_ridge_leverage_scores(cov.to(device), lambda_ridge=lambda_ridge)
            leverage_layer[eid] = leverage
        expert_covariances[layer_idx] = layer_covs
        expert_scores["leverage"][layer_idx] = _layer_norm(leverage_layer)

    torch.save(expert_scores, scores_path)
    torch.save(expert_covariances, cov_path)
    _print(
        f"[Step 2.5] ✅ Leverage appended to {scores_path}; "
        f"covariances saved to {cov_path} for {len(expert_covariances)} layers"
    )

    return expert_covariances
