"""Is the output head a linear map of the input embedding the model already stores?

The one idea that could cut the head's *incremental* parameter count by ~99% instead of
75%. On an untied model ``embed_tokens`` (V x D) has to be stored anyway, so if

    W_out  ~=  E_in @ M          M: (D, D)

then the head costs ``D^2`` extra parameters -- 4.2M on the 30B against 311.16M, i.e.
1.3%. Tying is the ``M = I`` special case, and plenty of models ship tied, so the
question is how much of the head a *learned* linear map recovers on a model whose
authors chose not to tie.

Measured here as the exact least-squares optimum, so it is an upper bound on any
such method: ``M* = (E^T E + lam I)^-1 E^T W``, relative error ``||W - E M*||_F/||W||_F``.
Also reported per-row for the frequent vs rare vocabulary, and against a rank-``r``
map ``M = M_1 M_2``, which is what a smaller budget would buy.

Note this test cannot be run on Qwen3-0.6B: it ships tied, so ``M = I`` is exact by
construction and the answer is a tautology. Untied models only.

    python scripts/lm_head_embed_reuse.py --model Qwen/Qwen3-30B-A3B-Thinking-2507
"""

import argparse
import json
import os

import torch

from scripts.lm_head_adarank_diag import pick_device, _print


@torch.no_grad()
def load_pair(model_id):
    """Return (W_out, E_in) as CPU float32, loading only those two tensors."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download(model_id, allow_patterns=["*.safetensors*", "*.json"])
    idx = os.path.join(path, "model.safetensors.index.json")
    wmap = {}
    if os.path.exists(idx):
        with open(idx) as f:
            wmap = json.load(f)["weight_map"]
    got = {}
    for name in ("lm_head.weight", "model.embed_tokens.weight"):
        fn = wmap.get(name, "model.safetensors")
        with safe_open(os.path.join(path, fn), framework="pt", device="cpu") as f:
            if name not in f.keys():
                raise RuntimeError(f"{model_id} has no {name} (tied? then this test is vacuous)")
            got[name] = f.get_tensor(name).float()
    W, E = got["lm_head.weight"], got["model.embed_tokens.weight"]
    if W.data_ptr() == E.data_ptr() or torch.equal(W[:64], E[:64]):
        raise SystemExit(f"{model_id} appears TIED -- M = I is exact and the test is vacuous.")
    return W, E


@torch.no_grad()
def grams(A, B, dev, chunk=16384):
    """(A^T A, A^T B) accumulated in float64."""
    D1, D2 = A.shape[1], B.shape[1]
    G = torch.zeros(D1, D1, dtype=torch.float64)
    X = torch.zeros(D1, D2, dtype=torch.float64)
    for s in range(0, A.shape[0], chunk):
        a = A[s:s + chunk].to(dev, torch.float32)
        b = B[s:s + chunk].to(dev, torch.float32)
        G += (a.T @ a).double().cpu()
        X += (a.T @ b).double().cpu()
    return G, X


@torch.no_grad()
def rel_err_rows(W, E, M, dev, chunk=16384, subset=None):
    num = den = 0.0
    rows = range(0, W.shape[0], chunk) if subset is None else [None]
    if subset is not None:
        w = W[subset].to(dev, torch.float32)
        e = E[subset].to(dev, torch.float32)
        return float(((w - e @ M.to(dev)).pow(2).sum() / w.pow(2).sum().clamp_min(1e-30)).sqrt())
    for s in rows:
        w = W[s:s + chunk].to(dev, torch.float32)
        e = E[s:s + chunk].to(dev, torch.float32)
        num += float((w - e @ M.to(dev)).pow(2).sum())
        den += float(w.pow(2).sum())
    return (num / max(den, 1e-30)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--unigram", default="calib/lm_head_qwen3_0_6b/unigram_c4_5000000.pt")
    ap.add_argument("--out", default="results_eval/lm_head_embed_reuse.json")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    dev = a.device or pick_device()
    W, E = load_pair(a.model)
    V, D = W.shape
    _print(f"[embed-reuse] {a.model}: W_out {tuple(W.shape)}, E_in {tuple(E.shape)}")

    G, X = grams(E, W, dev)
    lam = 1e-6 * torch.diagonal(G).mean()
    M = torch.linalg.solve(G + lam * torch.eye(D, dtype=torch.float64), X).float()

    rows = []
    full = rel_err_rows(W, E, M, dev)
    # the reference every alternative has to beat: the norm-matched zero map, i.e.
    # "the head carries no information transferable from the input embedding at all"
    zero = 1.0
    ident = rel_err_rows(W, E, torch.eye(D), dev)
    _print(f"[embed-reuse] full-rank M (D^2 = {D * D / 1e6:.1f}M params, "
           f"{100 * D * D / (V * D):.2f}% of the head):")
    _print(f"    rel err = {100 * full:.2f}%   (tied, M = I: {100 * ident:.2f}%;  "
           f"predict-zero: {100 * zero:.2f}%)")
    rows += [dict(map="lstsq_full", params=D * D, rel_err=full),
             dict(map="identity_tied", params=0, rel_err=ident),
             dict(map="zero", params=0, rel_err=zero)]

    if os.path.exists(a.unigram):
        cnt = torch.load(a.unigram, map_location="cpu")["counts"]
        if cnt.numel() == V:
            order = cnt.argsort(descending=True)
            for tag, sub in (("top-4096 frequent", order[:4096]),
                             ("rank 4k-32k", order[4096:32768]),
                             ("tail (rest)", order[32768:])):
                e = rel_err_rows(W, E, M, dev, subset=sub)
                _print(f"    rel err on {tag:<18} = {100 * e:.2f}%")
                rows.append(dict(map="lstsq_full", subset=tag, rel_err=e))

    # what a cheaper, low-rank map buys
    U, S, Vh = torch.linalg.svd(M.double(), full_matrices=False)
    for r in (64, 256, 512):
        Mr = ((U[:, :r] * S[:r]) @ Vh[:r]).float()
        e = rel_err_rows(W, E, Mr, dev)
        _print(f"    rank-{r:<4} M ({2 * D * r / 1e6:5.1f}M params): rel err = {100 * e:.2f}%")
        rows.append(dict(map=f"lstsq_rank{r}", params=2 * D * r, rel_err=e))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(model=a.model, V=V, D=D, rows=rows), f, indent=2)
    _print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
