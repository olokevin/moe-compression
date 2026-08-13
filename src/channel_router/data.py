"""Capture I/O and the single definition of the oracle channel importance.

Everything in the channel-router study reads its supervision from here, so the
importance definition of the plan's §0.1 exists exactly once:

    imp_{t,e,j} = g_{t,e} · |silu(W_g^{(e)} h_t)_j · (W_u^{(e)} h_t)_j| · ‖W_d^{(e)}[:,j]‖

This is bit-for-bit the score that the deployed ``oracle_mag`` criterion computes
in ``src/dynamic_active_param/block.py`` (verified by
``tests/test_data_oracle.py``), which is what makes the router's recall numbers
comparable to the already-measured ``oracle_mag`` downstream accuracies.

Channel indexing. The MoE router picks ``K`` of ``E`` experts per token, so the
"D = K·I activated channels" of the plan is a *token-dependent* subset of the
``E·I`` physical channels. Two index spaces are therefore used throughout:

- **slot space** ``(t, k, j)`` with ``k ∈ [0,K)`` — what the forward pass works in.
- **global space** ``c = e·I + j ∈ [0, E·I)`` — what a learned per-channel
  parameter must be keyed by, since slot ``k`` means a different expert for
  every token. ``global_ids(sel, I)`` converts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = [
    "acts_path", "weights_path", "LayerWeights", "LayerActs",
    "load_weights", "load_acts", "route", "oracle_scores", "global_ids",
    "split_slices", "iter_chunks",
]


def acts_path(out_dir: str, layer: int, tag: str, tokens: int) -> str:
    return os.path.join(out_dir, f"acts_L{layer}_{tag}_t{tokens}.pt")


def weights_path(out_dir: str, layer: int) -> str:
    return os.path.join(out_dir, f"weights_L{layer}.pt")


@dataclass
class LayerWeights:
    """One MoE layer's routing gate and expert weight stacks.

    ``Wg``/``Wu`` are ``(E, I, H)``; ``Wd`` is ``(E, H, I)`` (the raw
    ``down_proj.weight`` layout), so ``col_norm = Wd.norm(dim=1)`` is ``(E, I)``.
    """

    Wg: torch.Tensor
    Wu: torch.Tensor
    Wd: torch.Tensor | None
    gate_w: torch.Tensor
    col_norm: torch.Tensor
    top_k: int
    norm_topk: bool
    layer: int

    @property
    def E(self) -> int:
        return self.Wu.shape[0]

    @property
    def I(self) -> int:
        return self.Wu.shape[1]

    @property
    def H(self) -> int:
        return self.Wu.shape[2]

    def to(self, device, dtype=None) -> "LayerWeights":
        f = lambda t: None if t is None else t.to(device=device, dtype=dtype or t.dtype)
        return LayerWeights(
            Wg=f(self.Wg), Wu=f(self.Wu), Wd=f(self.Wd),
            gate_w=self.gate_w.to(device), col_norm=self.col_norm.to(device),
            top_k=self.top_k, norm_topk=self.norm_topk, layer=self.layer,
        )


@dataclass
class LayerActs:
    X: torch.Tensor          # (N, H) float16, MLP input
    pos: torch.Tensor        # (N,) position inside its block
    seq_id: torch.Tensor     # (N,) block id
    meta: dict

    @property
    def N(self) -> int:
        return self.X.shape[0]


def load_weights(data_dir: str, layer: int, *, want_down: bool = True,
                 device="cpu") -> LayerWeights:
    """Load ``weights_L{layer}.pt``, falling back to a legacy ``_wd`` capture."""
    p = weights_path(data_dir, layer)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} missing — run scripts/channel_router/collect_activations.py")
    d = torch.load(p, map_location="cpu")
    Wd = d["Wd"] if want_down else None
    col_norm = d["Wd"].float().norm(dim=1)                     # (E, I)
    w = LayerWeights(Wg=d["Wg"], Wu=d["Wu"], Wd=Wd, gate_w=d["gate_w"],
                     col_norm=col_norm, top_k=int(d["top_k"]),
                     norm_topk=bool(d["norm_topk"]), layer=layer)
    return w.to(device) if device != "cpu" else w


def load_acts(data_dir: str, layer: int, tag: str, tokens: int) -> LayerActs:
    p = acts_path(data_dir, layer, tag, tokens)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} missing — run scripts/channel_router/collect_activations.py")
    d = torch.load(p, map_location="cpu")
    return LayerActs(X=d["X"], pos=d["pos"], seq_id=d["seq_id"], meta=d.get("meta", {}))


def split_slices(N: int, val: int = 65536, test: int = 65536):
    """Contiguous train/val/test split.

    Contiguous (not random) because the token stream is document-ordered: a random
    split would put neighbouring positions of the same document on both sides and
    leak, given the ~0.7 adjacent-token mask IoU that P6 measures.
    """
    val = min(val, N // 4)
    test = min(test, N // 4)
    n_train = N - val - test
    if n_train <= 0:
        raise ValueError(f"N={N} too small for val={val} test={test}")
    return (slice(0, n_train), slice(n_train, n_train + val),
            slice(n_train + val, N))


@torch.no_grad()
def route(X: torch.Tensor, w: LayerWeights, device="cuda", chunk: int = 16384):
    """Reproduce the block's top-k routing. Returns ``g (N,K) float32``, ``sel (N,K) int64``."""
    gw = w.gate_w.to(device=device, dtype=torch.float32)
    gs, ss = [], []
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk].to(device=device, dtype=torch.float32)
        probs = F.softmax(x @ gw.t(), dim=1, dtype=torch.float32)
        g, sel = torch.topk(probs, w.top_k, dim=-1)
        if w.norm_topk:
            g = g / g.sum(dim=-1, keepdim=True)
        gs.append(g.cpu())
        ss.append(sel.cpu())
    return torch.cat(gs, 0), torch.cat(ss, 0)


def global_ids(sel: torch.Tensor, I: int) -> torch.Tensor:
    """``(T, K)`` expert ids -> ``(T, K, I)`` global channel ids ``e*I + j``."""
    return sel.unsqueeze(-1) * I + torch.arange(I, device=sel.device).view(1, 1, I)


@torch.no_grad()
def oracle_scores(x: torch.Tensor, sel: torch.Tensor, g: torch.Tensor,
                  w: LayerWeights, *, use_g: bool = True, use_colnorm: bool = True,
                  target: str = "mag", also_parts: bool = False):
    """Per-(token, slot, channel) oracle importance, ``(T, K, I)`` float32.

    Args:
        x: ``(T, H)`` MLP inputs, already on the weights' device.
        sel: ``(T, K)`` expert ids; ``g``: ``(T, K)`` routing weights.
        w: layer weights on the same device.
        use_g / use_colnorm: ablation switches (``use_colnorm=False`` is the
            ``oracle_mag_noW`` variant of the plan's P2 discussion).
        target: ``mag`` = ``|silu(gate)·up|`` (the deployed oracle),
            ``up`` = ``|up|``, ``gate`` = ``|silu(gate)|``, ``gate_raw`` = ``|gate|``.
        also_parts: additionally return ``(gate_pre, up)`` raw projections, which
            P2 needs to score several targets from one pass.
    """
    T, K = sel.shape
    I = w.I
    dev = x.device
    xf = x.to(torch.float32)
    inter = torch.zeros((T, K, I), dtype=torch.float32, device=dev)
    gate_pre = torch.zeros_like(inter) if (also_parts or target in ("gate", "gate_raw")) else None
    up_out = torch.zeros_like(inter) if (also_parts or target == "up") else None
    for e in sel.unique().tolist():
        tok, slot = (sel == e).nonzero(as_tuple=True)
        cur = xf[tok]
        gp = cur @ w.Wg[e].to(torch.float32).t()
        up = cur @ w.Wu[e].to(torch.float32).t()
        inter[tok, slot] = F.silu(gp) * up
        if gate_pre is not None:
            gate_pre[tok, slot] = gp
        if up_out is not None:
            up_out[tok, slot] = up

    if target == "mag":
        score = inter.abs()
    elif target == "up":
        score = up_out.abs()
    elif target == "gate":
        score = F.silu(gate_pre).abs()
    elif target == "gate_raw":
        score = gate_pre.abs()
    else:
        raise ValueError(f"unknown target {target}")
    if use_colnorm:
        score = score * w.col_norm[sel]                    # (T,K,I)
    if use_g:
        score = score * g.to(dev).unsqueeze(-1)
    if also_parts:
        return score, inter, gate_pre, up_out
    return score


def iter_chunks(n: int, chunk: int):
    for s in range(0, n, chunk):
        yield s, min(s + chunk, n)
