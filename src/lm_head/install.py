"""``install_lm_head(model, cfg)`` -- bind a compressed / tiered output head.

Follows ``install_dynamic_alloc``'s pattern: build the artifacts on CPU, move the
per-head tensors onto the head module's **own** device (required -- 30B-A3B is
sharded by ``device_map='auto'``), and rebind ``lm_head.forward`` via
``types.MethodType``.

Like the rest of the repo, this is a **masking simulation**: pruned/unread rows are
not physically removed, their logits are masked. The arithmetic is identical to
removing them, while the *accounting* changes -- which is the same fake-vs-real
split the expert path uses, and it is what lets lm-eval score arbitrary target
tokens against a head that nominally cannot emit them.
"""

import types
from typing import Optional

import torch
import torch.nn.functional as F

from src.base.shared_utils import _print
from src.lm_head.accounting import count_active_params, head_cost, print_lm_head_accounting
from src.lm_head.calib import ensure_sigma, ensure_unigram, get_lm_head
from src.lm_head.quant import bits_per_weight, quantize_rows_mixed
from src.lm_head.tiering import build_tiers, tier_stats

__all__ = ["install_lm_head"]

_METHODS = ("freq_tier", "archead", "rvq", "vq_logits", "rtn")


def _get_input_embedding(model):
    for path in ("model.model.embed_tokens", "model.embed_tokens",
                 "model.transformer.wte", "transformer.wte"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "weight"):
            return obj
    return None


def _untie_if_needed(model, head, verbose=True):
    """Give ``lm_head`` its own weight storage if it is tied to the input embedding.

    Non-negotiable for the SLM arm: Qwen3-0.6B ships ``tie_word_embeddings: true``, so
    ``lm_head.weight`` and ``embed_tokens.weight`` are the *same tensor*. Compressing
    the head in place would silently also compress the input embedding, and the
    result would be a different (worse, and differently-motivated) experiment.

    Returns ``(was_tied, untie_cost_params)``. The untie itself adds ``V*D``
    parameters to the checkpoint, which has to be reported or the accounting is
    wrong -- 155.6M params on Qwen3-0.6B, as large as the head we are compressing.
    """
    emb = _get_input_embedding(model)
    tied = (
        emb is not None
        and emb.weight.shape == head.weight.shape
        and emb.weight.data_ptr() == head.weight.data_ptr()
    )
    if not tied:
        return False, 0
    head.weight = torch.nn.Parameter(
        head.weight.detach().clone(), requires_grad=head.weight.requires_grad
    )
    if hasattr(model, "config"):
        model.config.tie_word_embeddings = False
    cost = head.weight.numel()
    if verbose:
        _print(
            f"[lm_head] ⚠️  head was TIED to the input embedding -- untied it before "
            f"compressing (the untie itself costs {cost / 1e6:.1f}M params; the input "
            f"embedding is left untouched at BF16)"
        )
    return True, cost


def bind_head_forward(head, fn):
    """Install ``fn`` as ``head``'s forward, surviving ``device_map='auto'``.

    Under accelerate's dispatch, ``head.forward`` is already a wrapper that moves
    incoming activations onto the module's device and calls ``head._old_forward``.
    Overwriting ``head.forward`` -- which is what ``install_dynamic_alloc`` does for
    MoE blocks, where it is safe because their inputs already arrive on the right
    device -- would *replace* that wrapper and the head would receive a hidden state
    still sitting on whichever GPU holds the final norm. Gate 0d exists to catch
    exactly this; on a 3-way shard it raised
    ``mat2 is on cuda:0, different from other tensors on cuda:2``.

    So: hook ``_old_forward`` when accelerate owns ``forward``, else ``forward``.
    """
    if hasattr(head, "_old_forward"):
        head._old_forward = types.MethodType(fn, head)
        head._lmh_hooked_attr = "_old_forward"
    else:
        head.forward = types.MethodType(fn, head)
        head._lmh_hooked_attr = "forward"


def unbind_head_forward(head):
    """Undo :func:`bind_head_forward`, restoring the original bound method."""
    attr = getattr(head, "_lmh_hooked_attr", None)
    if attr == "_old_forward":
        head._old_forward = types.MethodType(type(head).forward, head)
    elif attr == "forward":
        head.__dict__.pop("forward", None)
    for a in ("_lmh_keep_mask", "_lmh_tail_logit", "_lmh_stats", "_lmh_hooked_attr"):
        head.__dict__.pop(a, None)


def _tiered_lm_head_forward(self, x):
    """``lm_head.forward`` with a vocabulary mask applied to the logits.

    ``_lmh_tail_logit`` is None in strict mode (masked rows get exactly ``-inf``,
    so a tail target token is an unmissable failure) or a finite constant for the
    uniform tail fallback, which is the classic tiered-softmax construction and
    keeps perplexity finite.
    """
    logits = F.linear(x, self.weight, self.bias)
    mask = self._lmh_keep_mask
    if mask.device != logits.device:
        mask = mask.to(logits.device)
        self._lmh_keep_mask = mask
    if self._lmh_stats is not None:
        # Measure the tier hit-rate on the PRE-mask logits. Taking the argmax after
        # masking makes it in-tier by construction and reports a useless 100%; the
        # quantity that matters is whether the head *would* have chosen a tier row,
        # i.e. the block-accept rate an exact-decode scheme would live or die on.
        st = self._lmh_stats
        with torch.no_grad():
            flat = logits.reshape(-1, logits.shape[-1])
            st["tokens"] += flat.shape[0]
            st["argmax_in_tier"] += int(mask[flat.argmax(-1)].sum())
    fill = self._lmh_tail_logit
    logits = torch.where(
        mask,
        logits,
        logits.new_full((), float("-inf") if fill is None else float(fill)),
    )
    return logits


@torch.no_grad()
def _diagnostics(W_dense, W_hat, H, keep_mask=None, chunk=256):
    """Top-1 agreement / KL / logit MSE of the approximate head vs the dense one.

    Computed on the calibration hidden states -- the only place it *can* be
    computed, since the dense weight is gone once we overwrite it.

    Masked (strict) heads need care. ``KL(P_dense || P_approx)`` is *genuinely*
    ``+inf`` the moment the mask zeroes a token the dense head gives positive
    probability to, and that is the honest number, so we report it as inf rather
    than sweeping the ``-inf`` logits under a ``nan_to_num``. (Doing so silently
    produced a **negative** KL, because zeroing the tail terms while ``P_approx``
    stays renormalized over the tier makes every surviving term negative.)

    So for a masked head we report two finite, interpretable quantities instead:

    ``dense_mass_outside_tier``
        the dense probability mass the mask throws away -- the quantity that
        actually predicts the downstream damage.
    ``kl_in_tier``
        ``KL`` between the dense and approximate distributions *renormalized over
        the tier*, i.e. the distortion among the tokens the head can still emit.

    ``logit_mse`` is averaged over finite entries only; comparing ``-inf`` against a
    finite logit is not a number.
    """
    if H is None or H.numel() == 0:
        return {}
    dev = W_dense.device
    n = H.shape[0]
    Wd = W_dense.float()
    Wa = W_hat.float()
    mask = None if keep_mask is None else keep_mask.to(dev)

    agree = 0
    kl_sum = 0.0          # exact KL(P_dense || P_approx) over the full vocabulary
    kl_tier_sum = 0.0     # KL restricted+renormalized to the tier
    mse_sum = 0.0
    out_mass_sum = 0.0
    for s in range(0, n, chunk):
        h = H[s:s + chunk].to(device=dev, dtype=torch.float32)
        ld = h @ Wd.T
        la = h @ Wa.T
        lpd = F.log_softmax(ld, dim=-1)
        pdn = lpd.exp()
        if mask is None:
            lpa = F.log_softmax(la, dim=-1)
            agree += int((ld.argmax(-1) == la.argmax(-1)).sum())
            kl_sum += float((pdn * (lpd - lpa)).sum())
            kl_tier_sum = kl_sum
            mse_sum += float((ld - la).pow(2).mean(-1).sum())
        else:
            m = mask.unsqueeze(0)
            la_m = la.masked_fill(~m, float("-inf"))
            agree += int((ld.argmax(-1) == la_m.argmax(-1)).sum())
            # mass the mask discards -- finite, and the number that matters
            out_mass_sum += float(pdn.masked_fill(m, 0.0).sum())
            # renormalize BOTH sides over the tier and compare there
            lpd_t = F.log_softmax(ld.masked_fill(~m, float("-inf")), dim=-1)
            lpa_t = F.log_softmax(la_m, dim=-1)
            pdt = lpd_t.exp()
            kl_tier_sum += float((pdt * (lpd_t - lpa_t)).masked_fill(~m, 0.0).sum())
            kl_sum = float("inf")
            mse_sum += float((ld - la).pow(2).masked_fill(~m, 0.0).sum(-1).sum()
                             / max(int(mask.sum()), 1))

    out = {
        "top1_agreement": agree / max(n, 1),
        "kl_vs_dense": kl_sum if mask is None else float("inf"),
        "kl_in_tier": kl_tier_sum / max(n, 1),
        "logit_mse": mse_sum / max(n, 1),
        "n_diag_states": int(n),
    }
    if mask is not None:
        out["dense_mass_outside_tier"] = out_mass_sum / max(n, 1)
    if mask is None:
        out["kl_vs_dense"] = kl_sum / max(n, 1)
    return out


@torch.no_grad()
def install_lm_head(model, cfg: dict, tokenizer=None, args=None, verbose: bool = True):
    """Install an lm_head baseline described by ``cfg`` (the ``prune_kwargs.lm_head`` block).

    Recognized ``method`` values: ``freq_tier`` (B1), ``archead`` (B2), ``rvq`` /
    ``vq_logits`` (B3), ``rtn`` (F3, the honest naive floor). Returns ``model``.
    """
    method = str(cfg.get("method", "freq_tier"))
    if method not in _METHODS:
        raise ValueError(f"unknown lm_head method {method!r}; expected one of {_METHODS}")

    head = get_lm_head(model)
    was_tied, untie_cost = _untie_if_needed(model, head, verbose=verbose)
    W = head.weight
    V, D = W.shape
    dev = W.device
    compute_device = str(cfg.get("compute_device", "cpu"))
    group = int(cfg.get("group", 128 if method in ("freq_tier", "rtn") else 64))

    calib_dir = cfg.get("calib_dir") or (getattr(args, "scores_dir", "") or ".")
    ctx = count_active_params(model, expert_keep_frac=float(cfg.get("expert_keep_frac", 0.27)))
    if verbose:
        _print(
            f"[lm_head] method={method}, head {V}x{D} on {dev}, "
            f"total={ctx.total_params / 1e9:.3f}B active={ctx.active_params / 1e9:.3f}B "
            f"(head = {100 * ctx.head_params / ctx.active_params:.2f}% of active)"
        )

    # ---- calibration ------------------------------------------------------- #
    counts = None
    if method == "freq_tier" or cfg.get("tier_size"):
        if tokenizer is None:
            raise ValueError("freq_tier needs a tokenizer to count unigram frequencies")
        counts = ensure_unigram(tokenizer, cfg, calib_dir, V, verbose=verbose)
        if verbose:
            cov = tier_stats(counts)
            _print("[lm_head/B1] corpus mass by tier size: " +
                   ", ".join(f"T={t}: {100 * m:.2f}%" for t, m in cov.items()))
    C = H = None
    needs_sigma = method in ("archead",) or (
        method in ("rvq", "vq_logits") and cfg.get("activation_metric", True)
    )
    if needs_sigma or cfg.get("diagnostics", True):
        if tokenizer is None:
            raise ValueError(f"method={method} needs a tokenizer to collect the activation metric")
        C, H = ensure_sigma(model, tokenizer, cfg, calib_dir, verbose=verbose)

    # ---- build the approximate weight and/or the tier mask ------------------ #
    W_cpu = W.detach().to(device=compute_device, dtype=torch.float32)
    W_hat = None
    keep_mask = None
    tiers = None
    stats = {}
    read_rows = None
    read_bpw = None

    if method == "freq_tier":
        tiers = build_tiers(counts, int(cfg.get("tier_size", 4096)), verbose=verbose)
        head_bits = int(cfg.get("head_bits", 16))
        tail_bits = int(cfg.get("tail_bits", 4))
        sparse = bool(cfg.get("sparse_activate", False))
        # tail_bits == 0 is B1-p: the tail rows are gone, not quantized.
        pruned = (tail_bits == 0)

        if pruned or sparse:
            keep_mask = tiers.keep_mask.clone()
        if not pruned and (head_bits < 16 or tail_bits < 16):
            W_hat = quantize_rows_mixed(
                W_cpu, tiers.keep_mask.to(compute_device), head_bits, tail_bits, group
            )

        T = tiers.tier_size
        if pruned:
            # B1-p: only the tier is stored, at head_bits.
            store_bpw = bits_per_weight(head_bits, group) * T / V
            read_rows, read_bpw = T, bits_per_weight(head_bits, group)
        elif sparse:
            # B1-a: everything stored (head_bits/tail_bits), only the tier read.
            store_bpw = (bits_per_weight(head_bits, group) * T
                         + bits_per_weight(tail_bits, group) * (V - T)) / V
            read_rows, read_bpw = T, bits_per_weight(head_bits, group)
        else:
            # B1-s: everything stored and read; the tail is just cheaper.
            store_bpw = (bits_per_weight(head_bits, group) * T
                         + bits_per_weight(tail_bits, group) * (V - T)) / V
        stats.update({
            "tier_size": T, "head_bits": head_bits, "tail_bits": tail_bits,
            "sparse_activate": sparse, "pruned": pruned,
            "calib_head_mass": tiers.head_mass, "calib_tail_mass": tiers.tail_mass,
        })

    elif method == "rtn":  # F3 -- plain group RTN, the honest naive floor
        bits = int(cfg.get("bits", 4))
        W_hat = quantize_rows_mixed(
            W_cpu, torch.zeros(V, dtype=torch.bool, device=compute_device), 16, bits, group
        )
        store_bpw = bits_per_weight(bits, group)
        stats.update({"bits": bits, "group": group})

    elif method == "archead":
        from src.lm_head.archead import build_archead
        W_hat, s = build_archead(
            W_cpu, C, rc=int(cfg.get("rank", 10)), rr=int(cfg.get("correction_rank", 6)),
            group=group, p=float(cfg.get("metric_power", 0.75)),
            ridge=float(cfg.get("ridge", 1e-3)),
            residual_bits=int(cfg.get("residual_bits", 4)),
            activation_metric=bool(cfg.get("activation_metric", True)),
            compute_device=compute_device, verbose=verbose,
        )
        store_bpw = s["bits_per_weight"]
        stats.update(s)

    elif method == "rvq":
        from src.lm_head.vq import build_rvq
        W_hat, s = build_rvq(
            W_cpu, C, vq_dim=int(cfg.get("vq_dim", 16)),
            codebook_bits=int(cfg.get("vq_bits", 8)),
            stages=int(cfg.get("vq_stages", 3)),
            iters=int(cfg.get("vq_iters", 20)),
            adaptor_rank=int(cfg.get("adaptor_rank", 0)),
            p=float(cfg.get("metric_power", 0.75)), ridge=float(cfg.get("ridge", 1e-3)),
            activation_metric=bool(cfg.get("activation_metric", True)),
            compute_device=str(cfg.get("vq_device", compute_device)), verbose=verbose,
        )
        store_bpw = s["bits_per_weight"]
        stats.update(s)

    else:  # vq_logits
        from src.lm_head.vq import build_vq_logits
        W_hat, s = build_vq_logits(
            W_cpu, K=int(cfg.get("vq_codes", 1024)), iters=int(cfg.get("vq_iters", 25)),
            C=C, compute_device=str(cfg.get("vq_device", compute_device)), verbose=verbose,
        )
        store_bpw = s["bits_per_weight"]
        stats.update(s)

    # ---- diagnostics against the dense head, before we overwrite it -------- #
    if cfg.get("diagnostics", True) and H is not None and H.numel():
        # Always pass the mask when one exists, including for tail_fallback="uniform".
        # Withholding it made the fallback variants compare an unmodified weight
        # against itself and report a meaningless 100% agreement. The strict (-inf)
        # semantics are the right proxy for the fallback too: the shared tail logit is
        # log(tail_mass / n_tail) ~ -13.5, far below any plausible in-tier max, so the
        # argmax is the same either way.
        diag = _diagnostics(
            W_cpu, W_cpu if W_hat is None else W_hat, H, keep_mask=keep_mask,
        )
        stats.update(diag)
        if verbose and diag:
            msg = (
                f"[lm_head] diagnostics on {diag['n_diag_states']} calibration states: "
                f"top-1 agreement vs dense={100 * diag['top1_agreement']:.2f}%, "
                f"logit MSE={diag['logit_mse']:.5f}"
            )
            if "dense_mass_outside_tier" in diag:
                msg += (
                    f", KL vs dense=inf (the mask zeroes tokens the dense head wants), "
                    f"dense mass thrown away={100 * diag['dense_mass_outside_tier']:.3f}%, "
                    f"KL within the tier={diag['kl_in_tier']:.5f} nats"
                )
            else:
                msg += f", KL vs dense={diag['kl_vs_dense']:.5f} nats"
            _print(msg)

    # ---- apply -------------------------------------------------------------- #
    if W_hat is not None:
        W.data.copy_(W_hat.to(device=dev, dtype=W.dtype))
        if verbose:
            _print(f"[lm_head] ✅ wrote the approximated head into lm_head.weight ({W.dtype})")
    if keep_mask is not None:
        head._lmh_keep_mask = keep_mask.to(dev)
        strict = bool(cfg.get("strict", True))
        fallback = str(cfg.get("tail_fallback", "none"))
        if strict or fallback == "none":
            head._lmh_tail_logit = None
        elif fallback == "uniform":
            head._lmh_tail_logit = tiers.tail_logit_offset
        else:
            raise ValueError(f"unknown tail_fallback {fallback!r} (none|uniform)")
        head._lmh_stats = {"tokens": 0, "argmax_in_tier": 0} if cfg.get("collect_stats", True) else None
        bind_head_forward(head, _tiered_lm_head_forward)
        model._lm_head_module = head
        if verbose:
            _print(
                f"[lm_head] ✅ tier mask installed on {int(keep_mask.sum()):,}/{V:,} rows; "
                + ("strict (tail logits = -inf)" if head._lmh_tail_logit is None
                   else f"uniform tail fallback at logit offset {head._lmh_tail_logit:.3f}")
            )

    cost = head_cost(V, D, store_bpw, read_rows=read_rows, read_bits_per_weight=read_bpw)
    acct = print_lm_head_accounting(cost, ctx, label=method)
    if was_tied and verbose:
        _print(
            f"[lm_head/accounting]   NOTE: head was tied. Against the ORIGINAL "
            f"(tied) checkpoint the net change is "
            f"{100 * (cost['storage_bytes'] / 2.0) / max(ctx.total_params, 1):+.2f}% of total "
            f"params, because the untie adds a second copy of the head. The Δ above "
            f"is measured against the untied model, which is the right baseline for "
            f"a head-compression question but is NOT a saving over the shipped model."
        )
    model._lm_head_report = {
        "method": method, "cfg": dict(cfg), "was_tied": was_tied,
        "untie_cost_params": untie_cost, **stats, **acct,
    }
    return model


def lm_head_eval_stats(model) -> Optional[dict]:
    """Tier hit-rate accumulated over the eval stream (None if no mask was installed).

    This is the honesty check plan section 7 asks for: if a sparse-activation head
    scores at dense level, the argmax-in-tier rate says whether the mask was
    actually doing anything.
    """
    head = getattr(model, "_lm_head_module", None)
    st = getattr(head, "_lmh_stats", None) if head is not None else None
    if not st or not st["tokens"]:
        return None
    return {
        "eval_tokens": st["tokens"],
        "argmax_in_tier": st["argmax_in_tier"] / st["tokens"],
    }
