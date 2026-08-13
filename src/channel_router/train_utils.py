"""Shared training machinery for Stage B (set distillation) and Stage C (task KL).

Holds the three loss families of §2 Stage B, the label cache, the evaluation loop, and
the router artifact format that ``ppl_ladder.py`` and the eval protocol both read.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from src.channel_router.metrics import mass_recall, recall, select_topB
from src.channel_router.model import ChannelRouter

__all__ = ["ranking_loss", "evaluate_router", "save_router_artifact",
           "load_router_artifact", "build_router"]


# --------------------------------------------------------------------- losses
def ranking_loss(score: torch.Tensor, topk_idx: torch.Tensor, B: int, *,
                 kind: str = "margin", delta: int = 256, margin: float = 1.0,
                 w_fn: float = 3.0, imp: torch.Tensor | None = None,
                 pairs: int = 0, generator=None) -> torch.Tensor:
    """Loss for one batch.

    Args:
        score: ``(T, K, I)`` router scores.
        topk_idx: ``(T, B+delta)`` int slot-space indices of the oracle's ranked
            top-``B+delta`` channels (descending importance).
        B: budget; the decision boundary sits between rank ``B-1`` and ``B``.
        kind: ``margin`` (boundary-focused pairwise hinge, the plan's default),
            ``bce`` (all activated channels, membership labels),
            ``listwise`` (softmax cross-entropy against the importance mass inside the
            boundary window).
        delta: half-width of the boundary window in ranks.
        w_fn: weight on the positive side — a false negative (dropping a channel the
            oracle keeps) costs more than a false positive, which only wastes budget.
        pairs: if > 0, subsample this many (pos, neg) pairs per token instead of all
            ``delta²`` (the full set is affordable at delta ≤ 256, but the ablation
            "boundary window vs all-pairs" needs the knob).
    """
    T = score.shape[0]
    flat = score.reshape(T, -1)
    if kind == "bce":
        y = torch.zeros_like(flat)
        y.scatter_(1, topk_idx[:, :B].long(), 1.0)
        pos_w = torch.tensor(w_fn, device=flat.device)
        return F.binary_cross_entropy_with_logits(flat, y, pos_weight=pos_w)

    lo = max(0, B - delta)
    hi = min(topk_idx.shape[1], B + delta)
    pos_idx = topk_idx[:, lo:B].long()
    neg_idx = topk_idx[:, B:hi].long()
    s_pos = torch.gather(flat, 1, pos_idx)                      # (T, n_pos)
    s_neg = torch.gather(flat, 1, neg_idx)                      # (T, n_neg)

    if kind == "listwise":
        win = torch.cat([s_pos, s_neg], dim=1)
        if imp is None:
            raise ValueError("listwise needs imp")
        tgt = torch.gather(imp.reshape(T, -1), 1,
                           torch.cat([pos_idx, neg_idx], dim=1))
        tgt = tgt / tgt.sum(1, keepdim=True).clamp_min(1e-20)
        return -(tgt * F.log_softmax(win, dim=1)).sum(1).mean()

    if kind != "margin":
        raise ValueError(kind)
    if pairs:
        n_pos, n_neg = s_pos.shape[1], s_neg.shape[1]
        pi = torch.randint(0, n_pos, (T, pairs), device=flat.device, generator=generator)
        ni = torch.randint(0, n_neg, (T, pairs), device=flat.device, generator=generator)
        d = torch.gather(s_pos, 1, pi) - torch.gather(s_neg, 1, ni)
        return (w_fn * F.relu(margin - d)).mean()
    d = s_pos.unsqueeze(2) - s_neg.unsqueeze(1)                 # (T, n_pos, n_neg)
    return (w_fn * F.relu(margin - d)).mean()


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate_router(router, ld, sl, ratio, *, tokens=8192, batch=2048, slack=1.0,
                    top_tiles=0, scorer=None):
    """recall / mass-recall of ``router`` (or any scorer) on a held-out slice."""
    B = ld.budget(ratio)
    obj = scorer if scorer is not None else router
    recs, mrecs = [], []
    idx_all = ld.take(sl, tokens)
    for s in range(0, len(idx_all), batch):
        x, sel, g, imp = ld.batch(idx_all[s:s + batch])
        score = obj.score(x, sel, g)
        if hasattr(obj, "select_from_score"):
            pred = obj.select_from_score(score, sel, B, top_tiles=top_tiles, slack=slack)
        else:
            pred = select_topB(score, min(int(round(B * slack)), sel.shape[1] * ld.I))
        ref = select_topB(imp, B)
        recs.append(recall(pred, ref))
        mrecs.append(mass_recall(imp, pred, ref))
    return {"recall": sum(recs) / len(recs), "mass_recall": sum(mrecs) / len(mrecs),
            "tokens": int(len(idx_all)), "B": B}


# -------------------------------------------------------------------- artifact
def build_router(cfg: dict) -> ChannelRouter:
    return ChannelRouter(
        cfg["H"], cfg["E"], cfg["I"], cfg["K"], r=cfg["r"], m=cfg["m"],
        head=cfg.get("head", "swiglu"), use_bias=cfg.get("use_bias", True),
        use_g=cfg.get("use_g", True),
        n_tiles_per_expert=cfg.get("n_tiles_per_expert", 0))


def save_router_artifact(path: str, entries: dict, meta: dict | None = None):
    """``entries = {layer: (router, cfg)}`` -> one file with state dicts and configs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"meta": meta or {}, "layers": {}}
    for li, (router, cfg) in entries.items():
        payload["layers"][int(li)] = {
            "cfg": cfg,
            "state": {k: v.detach().to("cpu", torch.float32 if v.is_floating_point()
                                       else v.dtype)
                      for k, v in router.state_dict().items()},
        }
    torch.save(payload, path)
    return path


def load_router_artifact(path: str, device="cpu") -> dict:
    d = torch.load(path, map_location="cpu")
    out = {}
    for li, ent in d["layers"].items():
        r = build_router(ent["cfg"])
        r.load_state_dict(ent["state"])
        out[int(li)] = r.to(device).eval()
    return out
