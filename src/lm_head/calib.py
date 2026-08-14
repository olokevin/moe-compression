"""Calibration artifacts for the lm_head baselines.

Two artifacts, both cached next to ``scores_dir`` so a sweep pays for them once:

``unigram.pt``
    Token counts over a C4 sample. This is the **only** usable tiering axis for
    the head: the pilots measured ``corr(log freq, ||w||) = -0.13`` and row norms
    near-uniform (p99/p50 = 1.19-1.33), so magnitude carries no signal. Counting
    needs no forward pass at all -- it is a pure tokenizer sweep -- which is why
    the plan's ">= 5M tokens" requirement is cheap to honour (the pilots used
    ~25k tokens and saw only 5945 distinct types, badly underestimating the tail).

``sigma_lm_head.pt``
    ``C = H^T H / N`` for the post-final-norm hidden state, plus a subsample of
    ``H`` itself. ``C`` is the activation metric that ARCHead (and the optional
    RVQ adaptor) fit in; the ``H`` subsample drives the cheap top-1 agreement /
    KL diagnostics, which have to be computed against the *dense* head before it
    is overwritten.

Both are keyed by model + recipe in the filename, so a 0.6B run and a 30B run can
share one ``calib_dir`` without colliding.
"""

import os
import re

import torch
from tqdm import tqdm

from src.base.datasets import load_datasets
from src.base.shared_utils import _print

__all__ = [
    "DEFAULT_CALIB_RECIPE",
    "find_final_norm",
    "get_lm_head",
    "ensure_unigram",
    "ensure_sigma",
]

DEFAULT_CALIB_RECIPE = {
    "dataset": "c4",
    # Unigram counting: pure tokenizer sweep, so we can afford the >= 5M tokens
    # the plan asks for. C4 validation docs average ~450 tokens.
    "unigram_min_tokens": 5_000_000,
    "unigram_max_texts": 40_000,
    # Sigma / H collection: one hooked forward sweep. 128 x 512 tokens = 65k
    # hidden states, which over-determines a 2048x2048 second moment.
    "sigma_batches": 128,
    "sigma_batch_size": 16,
    "sigma_seq_len": 512,
    # Rows of H kept for the agreement/KL diagnostics (V x n logits materialized
    # twice, so keep this small).
    "h_sample": 2048,
}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")


def get_lm_head(model):
    """Return the output-projection ``nn.Linear`` of a causal LM."""
    head = getattr(model, "lm_head", None)
    if head is None:  # PEFT / wrapped models
        base = getattr(model, "base_model", None)
        head = getattr(base, "lm_head", None) if base is not None else None
    if head is None or not hasattr(head, "weight"):
        raise AttributeError(
            "Could not locate an lm_head with a .weight on this model; "
            "the lm_head baselines need a dense output projection."
        )
    return head


def find_final_norm(model):
    """Return the final pre-head norm module (``model.model.norm`` on Qwen/Llama).

    The head's input distribution -- and therefore the activation metric -- is the
    *post*-norm hidden state, so the hook has to sit on the norm's output, not on
    the last decoder layer.
    """
    for path in ("model.model.norm", "model.norm", "model.model.final_layernorm",
                 "transformer.ln_f", "model.transformer.ln_f"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError("Could not locate the final norm module before lm_head.")


# --------------------------------------------------------------------------- #
# unigram frequency
# --------------------------------------------------------------------------- #

@torch.no_grad()
def ensure_unigram(tokenizer, cfg, calib_dir, vocab_size, verbose=True):
    """Token counts over a C4 sample, cached to ``<calib_dir>/unigram_<n>.pt``.

    Returns a ``(V,) int64`` count tensor on CPU. Tokens never seen get count 0 and
    therefore land in the tail of every tiering -- which is the intended behaviour,
    but it is also why the token budget matters: a small sample pushes genuinely
    common tokens into the tail purely by sampling noise.
    """
    recipe = dict(DEFAULT_CALIB_RECIPE)
    recipe.update(cfg.get("calib_kwargs", {}) or {})
    min_tokens = int(recipe["unigram_min_tokens"])
    path = os.path.join(calib_dir, f"unigram_{_slug(recipe['dataset'])}_{min_tokens}.pt")

    if os.path.exists(path):
        payload = torch.load(path, map_location="cpu")
        if verbose:
            _print(
                f"[lm_head/calib] unigram loaded from {path}: "
                f"{int(payload['total']):,} tokens, "
                f"{int((payload['counts'] > 0).sum()):,} distinct types"
            )
        return payload["counts"]

    os.makedirs(calib_dir, exist_ok=True)
    texts = load_datasets(
        recipe["dataset"], tokenizer, max_samples=int(recipe["unigram_max_texts"]),
        max_length=None,
    )
    counts = torch.zeros(vocab_size, dtype=torch.int64)
    total = 0
    CHUNK = 256
    pbar = tqdm(range(0, len(texts), CHUNK), desc="lm_head unigram", disable=not verbose)
    for start in pbar:
        batch = [t for t in texts[start:start + CHUNK] if isinstance(t, str) and t.strip()]
        if not batch:
            continue
        ids = tokenizer(batch, add_special_tokens=False)["input_ids"]
        flat = torch.tensor([i for seq in ids for i in seq], dtype=torch.int64)
        if flat.numel():
            counts += torch.bincount(flat, minlength=vocab_size)
            total += int(flat.numel())
        pbar.set_postfix(tokens=f"{total/1e6:.2f}M")
        if total >= min_tokens:
            break

    if total < min_tokens and verbose:
        _print(
            f"[lm_head/calib] ⚠️  only {total:,} tokens available "
            f"(< requested {min_tokens:,}); tail coverage will be underestimated"
        )
    torch.save({"counts": counts, "total": total, "recipe": recipe}, path)
    if verbose:
        _print(
            f"[lm_head/calib] unigram: {total:,} tokens, "
            f"{int((counts > 0).sum()):,}/{vocab_size} distinct types -> {path}"
        )
    return counts


# --------------------------------------------------------------------------- #
# activation second moment
# --------------------------------------------------------------------------- #

@torch.no_grad()
def ensure_sigma(model, tokenizer, cfg, calib_dir, verbose=True):
    """Collect (or load) ``C = H^T H / N`` and an ``H`` subsample.

    ``C`` is accumulated in fp64 on CPU: it is a sum over ~65k outer products of a
    2048-vector whose entries differ by orders of magnitude across dimensions, and
    the downstream eigendecomposition of it is what the whole activation-metric
    story rests on.

    Returns ``(C, H_sample)`` -- ``C`` is ``(D, D) float32`` and ``H_sample`` is
    ``(n, D) float32``, both on CPU.
    """
    recipe = dict(DEFAULT_CALIB_RECIPE)
    recipe.update(cfg.get("calib_kwargs", {}) or {})
    n_batches = int(recipe["sigma_batches"])
    bs = int(recipe["sigma_batch_size"])
    seq = int(recipe["sigma_seq_len"])
    n_h = int(recipe["h_sample"])
    tag = f"{_slug(recipe['dataset'])}_{n_batches}x{bs}x{seq}"
    path = os.path.join(calib_dir, f"sigma_lm_head_{tag}.pt")

    if os.path.exists(path):
        payload = torch.load(path, map_location="cpu")
        if verbose:
            _print(
                f"[lm_head/calib] sigma loaded from {path}: "
                f"C {tuple(payload['C'].shape)} over {int(payload['n']):,} states"
            )
        return payload["C"], payload["H"]

    os.makedirs(calib_dir, exist_ok=True)
    norm = find_final_norm(model)
    D = get_lm_head(model).weight.shape[1]

    acc = torch.zeros(D, D, dtype=torch.float64)
    n_seen = 0
    n_pad_skipped = 0
    h_keep = []
    cur_attn = {"mask": None}     # set just before each forward, read by the hook

    def hook(module, inp, out):
        nonlocal acc, n_seen, n_pad_skipped
        h = out[0] if isinstance(out, (tuple, list)) else out
        h = h.detach()
        # Drop PADDING positions. The batches are right-padded to a common length,
        # and a pad position's post-norm hidden state is not a state the head is
        # ever asked about -- but it lands in C and in the H sample all the same.
        # Left in, it both skews the activation metric ARCHead fits in and wrecks the
        # top-1-agreement diagnostic (measured 66% instead of the true 88% at
        # T=4096, because pad states argmax onto rare tokens).
        am = cur_attn["mask"]
        if h.ndim == 3 and am is not None and am.shape == h.shape[:2]:
            keep = am.to(h.device).bool()
            n_pad_skipped += int((~keep).sum())
            h = h[keep]
        elif h.ndim == 3:
            h = h.reshape(-1, h.shape[-1])
        if h.shape[0] == 0:
            return
        hf = h.float().cpu()
        acc += (hf.T.double() @ hf.double())
        n_seen += hf.shape[0]
        if sum(t.shape[0] for t in h_keep) < n_h:
            # stride the kept rows so the diagnostic sample spans the sweep
            h_keep.append(hf[:: max(1, hf.shape[0] // 64)].clone())

    handle = norm.register_forward_hook(hook)
    try:
        texts = load_datasets(recipe["dataset"], tokenizer,
                              max_samples=n_batches * bs, max_length=seq)
        device = next(model.parameters()).device
        if getattr(model, "hf_device_map", None) is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        for i in tqdm(range(0, min(len(texts), n_batches * bs), bs),
                      desc="lm_head sigma", total=n_batches, disable=not verbose):
            batch = [t for t in texts[i:i + bs] if isinstance(t, str) and t.strip()]
            if not batch:
                continue
            enc = tokenizer(batch, max_length=seq, padding=True,
                            pad_to_multiple_of=8, truncation=True, return_tensors="pt")
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            cur_attn["mask"] = enc.get("attention_mask")
            model(**enc, use_cache=False)
    finally:
        handle.remove()
        cur_attn["mask"] = None

    if n_seen == 0:
        raise RuntimeError("lm_head sigma collection saw no hidden states.")
    C = (acc / n_seen).float()
    H = torch.cat(h_keep, dim=0)[:n_h].contiguous() if h_keep else torch.zeros(0, D)
    torch.save({"C": C, "H": H, "n": n_seen, "n_pad_skipped": n_pad_skipped,
                "recipe": recipe}, path)
    if verbose:
        _print(
            f"[lm_head/calib] skipped {n_pad_skipped:,} padding positions "
            f"({100 * n_pad_skipped / max(n_seen + n_pad_skipped, 1):.1f}% of the "
            f"batch grid) -- they are not states the head is ever asked about"
        )
        d = torch.linalg.eigvalsh(C.double())
        d = d.clamp_min(0)
        _print(
            f"[lm_head/calib] sigma: C {tuple(C.shape)} over {n_seen:,} states, "
            f"H sample {tuple(H.shape)}; eig max/mean={d.max().item():.3e}/"
            f"{d.mean().item():.3e}, top-1 energy share="
            f"{100 * d.max().item() / d.sum().clamp_min(1e-30).item():.2f}% -> {path}"
        )
    return C, H
