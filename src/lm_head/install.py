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

_METHODS = ("freq_tier", "archead", "rvq", "vq_logits", "rtn", "lowrank", "screen_refine")


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


def _require_materialized_head(model, head):
    """Fail early and legibly if ``lm_head.weight`` is a meta/offloaded placeholder.

    Every method here reads the head's weight *outside* a forward pass, so an offloaded
    head is fatal. But it fails deep inside a build with

        NotImplementedError: Cannot copy out of meta tensor; no data!

    which says nothing about the cause. And the *dense* rows of a sweep still work,
    because accelerate materializes offloaded weights on the fly during ``forward`` --
    so a contended box produces a run where the reference passes and every treatment
    dies 40 minutes in.

    The cause is always the same: ``device_map='auto'`` could not fit the model in the
    visible GPU memory (usually because another job is resident on one of the picked
    GPUs) and pushed layers to ``meta``/disk.
    """
    w = getattr(head, "weight", None)
    offloaded = w is not None and (w.is_meta or w.device.type == "meta")
    dm = getattr(model, "hf_device_map", None) or {}
    bad = sorted({str(v) for v in dm.values()} & {"meta", "disk", "cpu"})
    if not offloaded:
        if bad:
            _print(f"[lm_head] ⚠️  device_map offloads some modules to {bad} -- the head "
                   f"itself is materialized on {w.device}, so this run is fine, but the "
                   f"box is tight on GPU memory")
        return
    raise RuntimeError(
        "lm_head.weight is a meta tensor: accelerate offloaded the head instead of "
        "placing it on a GPU, so its values are not available to read.\n"
        f"  hf_device_map placements in use: {bad or 'unknown'}\n"
        "  Cause: device_map='auto' could not fit the model in the visible GPU memory "
        "-- most often another job is already resident on one of the selected GPUs.\n"
        "  Fix: give the run more/emptier GPUs (check nvidia-smi first), or cap "
        "accelerate's budget with PER_GPU_MEM so it shards evenly instead of "
        "offloading. Note the DENSE rows of a sweep will run fine either way, so a "
        "passing dense row is not evidence the placement is healthy."
    )


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
    for a in ("_lmh_keep_mask", "_lmh_tail_logit", "_lmh_stats", "_lmh_hooked_attr",
              "_sr_U", "_sr_col", "_sr_rank", "_sr_cand", "_sr_shift", "_sr_chunk",
              "_sr_stats", "_sr_sel", "_sr_cand_idx", "_sr_tail"):
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


def _screen_refine_forward(self, x):
    """S1 -- screen every row with a projected hidden state, refine the top-``N``.

    Chunked over flattened positions: an lm-eval batch is ``[bs, 2048, 151936]`` and this
    forward materializes *two* such logit tensors (the screen scores and the refined
    ones), so an unchunked version OOMs a 30B before it computes anything.

    ``coarse = h~ @ W.T`` with ``h~ = U_S U_S^T h`` is exactly ``A[:, S] coef_S`` for the
    rotated head ``A = W U`` -- see the module docstring of ``screen_refine``. The refine
    stage reuses the dense ``W @ h``, so refined logits are **bit-identical** to the dense
    head; only the tail carries approximation error.
    """
    U = self._sr_U
    if U.device != x.device:
        U = self._sr_U = U.to(x.device)
        self._sr_col = self._sr_col.to(x.device)
        if self._sr_sel is not None:
            self._sr_sel = self._sr_sel.to(x.device)
        if self._sr_cand_idx is not None:
            self._sr_cand_idx = self._sr_cand_idx.to(x.device)
    col = self._sr_col
    shp = x.shape[:-1]
    h = x.reshape(-1, x.shape[-1])
    n, D = h.shape
    V = self.weight.shape[0]
    r0, N = self._sr_rank, self._sr_cand
    out = torch.empty(n, V, dtype=self.weight.dtype, device=x.device)
    st = self._sr_stats
    for s in range(0, n, self._sr_chunk):
        hb = h[s:s + self._sr_chunk]
        coef = hb @ U                                        # (b, D) rotated coordinates
        ck = torch.zeros_like(coef)
        if self._sr_sel is None:                             # per-token screen
            idx = (coef.abs() * col).topk(r0, dim=-1).indices
            ck.scatter_(1, idx, coef.gather(1, idx))
        else:                                                # static-screen ablation
            ck[:, self._sr_sel] = coef[:, self._sr_sel]
        coarse = torch.nn.functional.linear(ck @ U.T, self.weight, self.bias)
        full = torch.nn.functional.linear(hb, self.weight, self.bias)
        if self._sr_cand_idx is None:
            cand = coarse.topk(N, dim=-1).indices             # dynamic candidates
        else:
            cand = self._sr_cand_idx.unsqueeze(0).expand(hb.shape[0], -1)
        ob = coarse + self._sr_shift
        ob.scatter_(1, cand, full.gather(1, cand))
        if self._sr_tail == "inf":
            # ablation reproducing B1-a's semantics: non-candidates cannot be emitted.
            keep = torch.zeros_like(ob, dtype=torch.bool).scatter_(1, cand, True)
            ob = ob.masked_fill(~keep, float("-inf"))
        if st is not None:
            with torch.no_grad():
                st["tokens"] += hb.shape[0]
                inc = torch.zeros_like(ob, dtype=torch.bool).scatter_(1, cand, True)
                st["argmax_in_cand"] += int(
                    inc.gather(1, full.argmax(-1, keepdim=True)).sum())
                pd = full.float().softmax(-1)
                st["mass_outside_cand"] += float(pd.masked_fill(inc, 0.0).sum())
        out[s:s + self._sr_chunk] = ob.to(out.dtype)
    return out.reshape(*shp, V)


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
def _diagnostics_screen_refine(W, H, U, col, r0, N, shift=0.0, chunk=128,
                               fixed_sel=None, cand_idx=None, tail="coarse"):
    """Top-1 / KL / discarded-mass for S1, whose approximation is **per token**.

    ``_diagnostics`` compares one fixed ``W_hat`` against ``W`` and cannot express a head
    whose effective weight changes every position, so S1 gets its own. The quantities are
    the same ones, and KL is finite here: the tail keeps a graded score, so no token the
    dense head wants is ever assigned zero probability.
    """
    if H is None or H.numel() == 0:
        return {}
    dev = W.device
    Wf = W.float()
    Ud, cold = U.to(dev).float(), col.to(dev).float()
    n = H.shape[0]
    kl = agree = mass = 0.0
    dlogp = 0.0
    sel = None if fixed_sel is None else fixed_sel.to(dev)
    cidx = None if cand_idx is None else cand_idx.to(dev)
    for s in range(0, n, chunk):
        hb = H[s:s + chunk].to(device=dev, dtype=torch.float32)
        coef = hb @ Ud
        ck = torch.zeros_like(coef)
        if sel is None:
            idx = (coef.abs() * cold).topk(r0, dim=-1).indices
            ck.scatter_(1, idx, coef.gather(1, idx))
        else:
            ck[:, sel] = coef[:, sel]
        coarse = (ck @ Ud.T) @ Wf.T
        full = hb @ Wf.T
        cand = (coarse.topk(N, dim=-1).indices if cidx is None
                else cidx.unsqueeze(0).expand(hb.shape[0], -1))
        la = coarse + float(shift)
        la.scatter_(1, cand, full.gather(1, cand))
        if tail == "inf":
            keep = torch.zeros_like(la, dtype=torch.bool).scatter_(1, cand, True)
            la = la.masked_fill(~keep, float("-inf"))
        lpd = F.log_softmax(full, -1)
        lpa = F.log_softmax(la, -1)
        pd = lpd.exp()
        # A masked tail makes KL(dense || approx) genuinely +inf; zeroing those terms
        # while the survivors stay renormalized reports a negative KL (doc bug 3).
        kl += float("inf") if tail == "inf" else float((pd * (lpd - lpa)).sum())
        agree += int((full.argmax(-1) == la.argmax(-1)).sum())
        inc = torch.zeros_like(la, dtype=torch.bool).scatter_(1, cand, True)
        mass += float(pd.masked_fill(inc, 0.0).sum())
        # |Δ log p| on a target drawn from the dense distribution: HellaSwag / ARC-C
        # score a *given* continuation, not the argmax, so this is the quantity that
        # predicts them. KL is dense-p-weighted and would hide a bad tail.
        dlogp += float((lpd - lpa).abs().gather(1, torch.multinomial(pd, 1)).sum())
    return {
        "top1_agreement": agree / n, "kl_vs_dense": kl / n,
        "dense_mass_outside_cand": mass / n, "dlogp_sampled_target": dlogp / n,
        "n_diag_states": int(n),
    }


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
    _require_materialized_head(model, head)
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
    if (method == "freq_tier" or cfg.get("tier_size")
            or str(cfg.get("cand_source", "")) == "freq"):
        if tokenizer is None:
            raise ValueError("freq_tier needs a tokenizer to count unigram frequencies")
        counts = ensure_unigram(tokenizer, cfg, calib_dir, V, verbose=verbose)
        if verbose:
            cov = tier_stats(counts)
            _print("[lm_head/B1] corpus mass by tier size: " +
                   ", ".join(f"T={t}: {100 * m:.2f}%" for t, m in cov.items()))
    C = H = None
    needs_sigma = method in ("archead",) or (
        method == "lowrank" and cfg.get("whiten", True)
    ) or (
        method in ("rvq", "vq_logits") and cfg.get("activation_metric", True)
    ) or (
        method == "screen_refine" and str(cfg.get("basis", "ceig")) != "raw"
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
    # True parameter counts. None => V*D stored and read_rows*D read, which is right
    # for row-subset methods and for every pure-precision method. Overridden only by
    # representations that are not a row subset (low-rank factors, codebooks).
    n_stored = None
    n_read = None
    screen_refine_artifacts = None

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
            # B1-p: only the tier is stored, at head_bits. A genuine parameter-count
            # reduction: T*D numbers instead of V*D.
            store_bpw = bits_per_weight(head_bits, group) * T / V
            read_rows, read_bpw = T, bits_per_weight(head_bits, group)
            n_stored = T * D
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

    elif method == "lowrank":  # F2 -- the low-rank ladder
        from src.lm_head.quant import build_lowrank
        # rank_frac is a fraction of D, so one variant means the same storage point on
        # every model. An absolute `rank` from a d=2048 config silently becomes a
        # *different* storage point on a d=1024 head, which is how "lr25" ended up
        # meaning 50% there.
        if cfg.get("rank_frac") is not None:
            _rank = max(1, int(round(float(cfg["rank_frac"]) * D)))
        else:
            _rank = int(cfg.get("rank", D // 2))
        W_hat, s = build_lowrank(
            W_cpu, C, rank=_rank,
            whiten=bool(cfg.get("whiten", True)),
            p=float(cfg.get("metric_power", 0.5)), ridge=float(cfg.get("ridge", 1e-3)),
            compute_device=compute_device,
        )
        store_bpw = s["bits_per_weight"]
        # Low-rank is the one full-vocabulary method that truly shrinks the parameter
        # COUNT: (V + D) * r numbers instead of V * D, all of them read every token.
        n_stored = n_read = (V + D) * _rank
        stats.update(s)

    elif method == "screen_refine":  # S1 -- dynamic reads, graded tail
        from src.lm_head.screen_refine import build_screen_refine, screen_refine_cost
        U, col_norm, static_score, s = build_screen_refine(
            W_cpu, C, basis=str(cfg.get("basis", "ceig")),
            ridge=float(cfg.get("ridge", 1e-3)), compute_device=compute_device,
            verbose=verbose,
        )
        # Ablation: rank the screen by |coef_i| alone, dropping the ||W u_i|| factor.
        # With basis="raw" (U=I) coef=h, so this selects the top-r0 *hidden-state
        # entries by magnitude* directly -- gate 0g's |coef|-only score promoted to a
        # full sweep variant instead of a synthetic-W diagnostic. The read/storage
        # accounting is unchanged, so it slots in at the same budget as its basis="raw"
        # (col-norm-weighted) and basis="ceig" siblings.
        if not bool(cfg.get("screen_use_col_norm", True)):
            col_norm = torch.ones_like(col_norm)  # affects only the adaptive screen
            s["screen_use_col_norm"] = False
        # screen_rank_frac is a fraction of D so one variant name means the same read
        # budget on a d=1024 and a d=2048 head -- the mistake f2_lr25 made (bug 6).
        if cfg.get("screen_rank_frac") is not None:
            _r0 = max(1, int(round(float(cfg["screen_rank_frac"]) * D)))
        else:
            _r0 = int(cfg.get("screen_rank", max(1, D // 5)))
        sc = screen_refine_cost(V, D, _r0, int(cfg.get("cand_size", 8192)))
        fixed_sel = None
        if str(cfg.get("screen", "adaptive")) == "static":
            # ablation: one fixed coordinate set for every position -- which is exactly
            # activation-aware low-rank used as a screen.
            fixed_sel = static_score.topk(sc["screen_rank"]).indices.contiguous()
            s["screen"] = "static"
        cand_rank = None
        if str(cfg.get("cand_source", "screen")) == "freq":
            if counts is None:
                raise ValueError('screen_refine cand_source="freq" needs the unigram counts')
            cand_rank = counts.argsort(descending=True)[: sc["cand_size"]].contiguous()
            s["cand_source"] = "freq"
        store_bpw = sc["bits_per_weight"]
        read_bpw = sc["read_bits_per_weight"]
        n_stored, n_read = sc["stored_params"], sc["read_params"]
        # head_cost derives read BYTES from read_rows * D, so express the read count in
        # row-equivalents. The parameter count itself is passed exactly via n_read.
        read_rows = max(1, int(round(n_read / D)))
        stats.update(s)
        stats.update({k: v for k, v in sc.items() if k not in ("bits_per_weight",)})
        screen_refine_artifacts = (U, col_norm, fixed_sel, cand_rank)

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
    if method == "screen_refine":
        if cfg.get("diagnostics", True) and H is not None and H.numel():
            diag = _diagnostics_screen_refine(
                W_cpu, H, screen_refine_artifacts[0], screen_refine_artifacts[1],
                sc["screen_rank"], sc["cand_size"],
                shift=float(cfg.get("tail_shift", 0.0)),
                fixed_sel=screen_refine_artifacts[2],
                cand_idx=screen_refine_artifacts[3],
                tail=str(cfg.get("tail", "coarse")),
            )
            stats.update(diag)
            if verbose and diag:
                _print(
                    f"[lm_head/S1] diagnostics on {diag['n_diag_states']} states: "
                    f"top-1 agreement={100 * diag['top1_agreement']:.2f}%, "
                    f"KL vs dense={diag['kl_vs_dense']:.5f} nats, dense mass outside the "
                    f"candidate set={100 * diag['dense_mass_outside_cand']:.3f}%, "
                    f"|Δlog p| on a sampled target={diag['dlogp_sampled_target']:.5f}"
                )
    elif cfg.get("diagnostics", True) and H is not None and H.numel():
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
    if screen_refine_artifacts is not None:
        U, col_norm, fixed_sel, cand_rank = screen_refine_artifacts
        head._sr_U = U.to(device=dev, dtype=W.dtype)
        head._sr_col = col_norm.to(device=dev, dtype=W.dtype)
        head._sr_sel = None if fixed_sel is None else fixed_sel.to(dev)
        head._sr_cand_idx = None if cand_rank is None else cand_rank.to(dev)
        head._sr_tail = str(cfg.get("tail", "coarse"))
        head._sr_rank = sc["screen_rank"]
        head._sr_cand = sc["cand_size"]
        head._sr_shift = float(cfg.get("tail_shift", 0.0))
        head._sr_chunk = int(cfg.get("forward_chunk", 512))
        head._sr_stats = ({"tokens": 0, "argmax_in_cand": 0, "mass_outside_cand": 0.0}
                          if cfg.get("collect_stats", True) else None)
        bind_head_forward(head, _screen_refine_forward)
        model._lm_head_module = head
        if verbose:
            _print(
                f"[lm_head/S1] ✅ installed: screen with {sc['screen_rank']}/{D} "
                f"coordinates over all {V:,} rows, then refine the top "
                f"{sc['cand_size']:,} exactly -> reads "
                f"{100 * sc['read_param_frac']:.2f}% of V*D, storage "
                f"{100 * sc['stored_param_frac']:.2f}%"
            )
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

    cost = head_cost(V, D, store_bpw, read_rows=read_rows, read_bits_per_weight=read_bpw,
                     stored_params=n_stored, read_params=n_read)
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
    if head is None:
        return None
    st = getattr(head, "_lmh_stats", None)
    if st and st["tokens"]:
        return {
            "eval_tokens": st["tokens"],
            "argmax_in_tier": st["argmax_in_tier"] / st["tokens"],
        }
    st = getattr(head, "_sr_stats", None)
    if st and st["tokens"]:
        # S1's analogue: the dense argmax has to survive the *screen*, or the refine
        # stage never sees it. Measured over the eval stream, not the calibration set.
        return {
            "eval_tokens": st["tokens"],
            "argmax_in_tier": st["argmax_in_cand"] / st["tokens"],
            "eval_mass_outside_cand": st["mass_outside_cand"] / st["tokens"],
        }
    return None
