"""One-line setup shared by every Phase-0 study and by Stage B/C training.

``LayerData`` loads a layer's activations and weights, reproduces the routing once
(top-k expert choice is fixed by the frozen gate, so it never needs recomputing), and
serves oracle batches on demand. Splits are contiguous — the token stream is
document-ordered, so a random split would leak across the ~0.7 adjacent-position mask
IoU that P6 measures.
"""

from __future__ import annotations

import torch

from src.channel_router import data as D
from src.channel_router.metrics import select_topB

__all__ = ["LayerData"]


class LayerData:
    def __init__(self, data_dir: str, layer: int, *, tag: str = "c4",
                 tokens: int = 1 << 20, device: str = "cuda",
                 want_down: bool = True, max_tokens: int = 0,
                 val: int = 65536, test: int = 65536, weights_dtype=torch.float16):
        self.device = torch.device(device)
        self.layer = layer
        self.acts = D.load_acts(data_dir, layer, tag, tokens)
        self.w = D.load_weights(data_dir, layer, want_down=want_down).to(
            self.device, weights_dtype)
        self.w.col_norm = self.w.col_norm.float()
        self.X = self.acts.X
        if max_tokens and max_tokens < self.X.shape[0]:
            self.X = self.X[:max_tokens]
        self.N = self.X.shape[0]
        self.g, self.sel = D.route(self.X, self.w, device=device)
        self.train_sl, self.val_sl, self.test_sl = D.split_slices(
            self.N, val=val, test=test)

    # ------------------------------------------------------------------ shapes
    @property
    def E(self):
        return self.w.E

    @property
    def I(self):
        return self.w.I

    @property
    def K(self):
        return self.w.top_k

    @property
    def H(self):
        return self.w.H

    def budget(self, ratio: float) -> int:
        return max(1, min(int(round(ratio * self.K * self.I)), self.K * self.I))

    # ----------------------------------------------------------------- batches
    def take(self, sl, n: int = 0, offset: int = 0):
        """Index tensor for a contiguous slice, optionally truncated to ``n`` tokens."""
        idx = torch.arange(sl.start + offset, sl.stop)
        return idx[:n] if n else idx

    def batch(self, idx: torch.Tensor, *, target: str = "mag", use_g: bool = True,
              use_colnorm: bool = True, also_parts: bool = False):
        """``(x, sel, g, imp)`` on device for the given token indices."""
        x = self.X[idx].to(self.device, torch.float32)
        sel = self.sel[idx].to(self.device)
        g = self.g[idx].to(self.device)
        out = D.oracle_scores(x, sel, g, self.w, use_g=use_g, use_colnorm=use_colnorm,
                              target=target, also_parts=also_parts)
        if also_parts:
            imp, inter, gate_pre, up_out = out
            return x, sel, g, imp, inter, gate_pre, up_out
        return x, sel, g, out

    def iter_batches(self, sl, batch: int = 4096, n: int = 0, **kw):
        idx_all = self.take(sl, n)
        for s in range(0, len(idx_all), batch):
            yield self.batch(idx_all[s:s + batch], **kw)

    @torch.no_grad()
    def cache_topb(self, idx: torch.Tensor, ratio: float, *, batch: int = 4096,
                   log=None):
        """``(n, B)`` int16 slot-space indices of each token's oracle top-B.

        Training reads the same labels for many epochs, and recomputing the oracle is
        ~30x the cost of the model step it supervises, so the labels are materialized
        once. Slot space (``K*I = 6144 < 2^15``) fits in int16: 1.5 KB/token.
        """
        B = self.budget(ratio)
        out = torch.empty((len(idx), B), dtype=torch.int16)
        for s in range(0, len(idx), batch):
            _, _, _, imp = self.batch(idx[s:s + batch])
            flat = imp.reshape(imp.shape[0], -1)
            out[s:s + flat.shape[0]] = flat.topk(B, dim=1).indices.to(torch.int16).cpu()
            if log and (s // batch) % 8 == 0:
                log(f"    cached labels {s + flat.shape[0]}/{len(idx)}")
        return out

    @staticmethod
    def labels_from_topb(topb: torch.Tensor, KI: int, device) -> torch.Tensor:
        """``(T, KI)`` float 0/1 labels from cached top-B indices."""
        y = torch.zeros((topb.shape[0], KI), device=device)
        return y.scatter_(1, topb.to(device).long(), 1.0)

    # -------------------------------------------------------------- statistics
    @torch.no_grad()
    def channel_freq(self, sl, ratio: float, *, batch: int = 4096, n: int = 0):
        """``(E*I,)`` P(channel in oracle top-B) and ``(E*I,)`` mean oracle score.

        Both are the *static* summaries P4 and the static baseline need: the frequency
        prior and the "rank by mean oracle score, read no x at all" free floor.
        """
        B = self.budget(ratio)
        cnt = torch.zeros(self.E * self.I, device=self.device)
        ssum = torch.zeros(self.E * self.I, device=self.device)
        act = torch.zeros(self.E * self.I, device=self.device)
        ntok = 0
        for x, sel, g, imp in self.iter_batches(sl, batch=batch, n=n):
            keep = select_topB(imp, B)
            gid = D.global_ids(sel, self.I).reshape(-1)
            cnt.scatter_add_(0, gid, keep.reshape(-1).float())
            ssum.scatter_add_(0, gid, imp.reshape(-1))
            act.scatter_add_(0, gid, torch.ones_like(imp.reshape(-1)))
            ntok += x.shape[0]
        freq_global = cnt / max(ntok, 1)                    # P(kept | any token)
        freq_active = cnt / act.clamp_min(1)                # P(kept | expert active)
        mean_score = ssum / act.clamp_min(1)
        return {"freq_global": freq_global, "freq_active": freq_active,
                "mean_score": mean_score, "activation": act / max(ntok, 1),
                "tokens": ntok, "B": B}
