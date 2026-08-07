#!/usr/bin/env python3
"""Profile the three data-movement / compute stages of ONE MoE layer of
Qwen3-30B-A3B, at decode (batch=1, top-K=8 experts active).

Stages (the memory hierarchy of the intro slide):
  1. DRAM  -> GPU memory   : host (pinned CPU) -> device HBM copy over PCIe
  2. GPU memory -> cores   : reading the resident weights from HBM into the
                             tensor cores during the decode GEMV (batch=1) --
                             this kernel is memory-bandwidth bound, so its time
                             is essentially the HBM read time.
  3. Compute in cores      : the raw tensor-core arithmetic, isolated by running
                             the same expert matmuls at a large batch so the
                             kernel is compute bound; per-token compute time is
                             (achieved TFLOPS) applied to the 75.5 MFLOP/token.

Model dims (Qwen/Qwen3-30B-A3B config.json):
  hidden d = 2048, moe_intermediate I = 768, experts N = 128, top-K = 8, bf16.

Run:  uv run python scripts/moe_layer_profile.py
"""
from __future__ import annotations

import argparse
import statistics as st

import torch

# ---- Qwen3-30B-A3B MoE dims ------------------------------------------------
D = 2048            # hidden_size
I = 768             # moe_intermediate_size
K = 8               # num_experts_per_tok (active at decode)
BYTES = 2           # bf16
DTYPE = torch.bfloat16

# One active expert has gate[I,d] + up[I,d] + down[d,I]
PARAMS_PER_EXPERT = 3 * I * D
ACTIVE_PARAMS = K * PARAMS_PER_EXPERT          # 37.75 M
ACTIVE_BYTES = ACTIVE_PARAMS * BYTES           # 75.5 MB
# GEMV: 1 multiply + 1 add per weight = 2 FLOP/param
FLOP_PER_TOKEN = 2 * ACTIVE_PARAMS             # 75.5 MFLOP


def _sync():
    torch.cuda.synchronize()


def _time_ms(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        ts.append(start.elapsed_time(end))  # ms
    return ts


def stage1_pcie(dev, iters, nbytes=None):
    """Host(pinned) -> device copy. Default = one layer's K active experts."""
    n = (nbytes // BYTES) if nbytes else ACTIVE_PARAMS
    host = torch.empty(n, dtype=DTYPE, pin_memory=True)
    dst = torch.empty(n, dtype=DTYPE, device=dev)

    def do():
        dst.copy_(host, non_blocking=True)

    ts = _time_ms(do, iters)
    med = st.median(ts)
    gbps = (n * BYTES) / (med / 1e3) / 1e9
    return med, gbps, ts


def peak_hbm_bw(dev, iters, nbytes=512 * 1024 * 1024):
    """Large device-to-device copy to gauge achievable HBM bandwidth."""
    n = nbytes // BYTES
    a = torch.empty(n, dtype=DTYPE, device=dev)
    b = torch.empty(n, dtype=DTYPE, device=dev)

    def do():
        b.copy_(a)

    ts = _time_ms(do, iters // 4 or 1)
    med = st.median(ts)
    # copy reads n and writes n -> 2*nbytes moved
    gbps = (2 * n * BYTES) / (med / 1e3) / 1e9
    return gbps


def _make_experts(dev):
    gate = torch.randn(K, I, D, dtype=DTYPE, device=dev) * 0.02
    up = torch.randn(K, I, D, dtype=DTYPE, device=dev) * 0.02
    down = torch.randn(K, D, I, dtype=DTYPE, device=dev) * 0.02
    return gate, up, down


def _expert_ffn(gate, up, down, x):
    """x: [K, T, D] token(s) fed to each of the K active experts.
    Returns [K, T, D]. bmm keeps weights resident in HBM."""
    # gate/up: [K,I,D] @ [K,D,T] -> [K,I,T]
    g = torch.bmm(gate, x.transpose(1, 2))
    u = torch.bmm(up, x.transpose(1, 2))
    h = torch.nn.functional.silu(g) * u           # [K,I,T]
    # down: [K,D,I] @ [K,I,T] -> [K,D,T]
    y = torch.bmm(down, h)
    return y.transpose(1, 2)


def stage2_decode(dev, iters):
    """Decode: 1 token through K experts, weights resident in HBM.
    Memory-bound; time ~ HBM read of the active weights."""
    gate, up, down = _make_experts(dev)
    x = torch.randn(K, 1, D, dtype=DTYPE, device=dev) * 0.02

    def do():
        _expert_ffn(gate, up, down, x)

    ts = _time_ms(do, iters)
    med = st.median(ts)
    # effective HBM read BW = weights read once / time
    eff_gbps = ACTIVE_BYTES / (med / 1e3) / 1e9
    return med, eff_gbps, ts


def stage3_compute(dev, iters, T):
    """Compute-bound: same experts, T tokens each -> saturates tensor cores.
    Reports achieved TFLOPS and the implied per-token compute time."""
    gate, up, down = _make_experts(dev)
    x = torch.randn(K, T, D, dtype=DTYPE, device=dev) * 0.02

    def do():
        _expert_ffn(gate, up, down, x)

    ts = _time_ms(do, iters)
    med = st.median(ts)
    total_flop = FLOP_PER_TOKEN * T
    tflops = total_flop / (med / 1e3) / 1e12
    per_tok_us = (FLOP_PER_TOKEN / (tflops * 1e12)) * 1e6
    return med, tflops, per_tok_us, T, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4096,
                    help="tokens per expert for the compute-bound stage 3")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    dev = torch.device(args.device)
    name = torch.cuda.get_device_name(dev)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Device            : {name}")
    print(f"Model             : Qwen3-30B-A3B  (d={D}, I={I}, K={K}, bf16)")
    print(f"Active weights/layer/token : {ACTIVE_PARAMS/1e6:.2f} M params  = "
          f"{ACTIVE_BYTES/1e6:.1f} MB")
    print(f"Compute/layer/token        : {FLOP_PER_TOKEN/1e6:.1f} MFLOP")
    print("-" * 68)

    # reference: achievable PCIe and HBM bandwidths on this box
    _, bw_pcie_peak, _ = stage1_pcie(dev, args.iters, nbytes=256 * 1024 * 1024)
    bw_hbm_peak = peak_hbm_bw(dev, args.iters)
    print(f"Reference bandwidths (this GPU): PCIe H2D peak ~{bw_pcie_peak:.1f} GB/s, "
          f"HBM copy peak ~{bw_hbm_peak:.0f} GB/s")
    print("-" * 68)

    m1, bw1, _ = stage1_pcie(dev, args.iters)
    print(f"[1] DRAM(host) -> GPU HBM  (PCIe H2D, 75.5 MB) : {m1*1e3:8.1f} us   "
          f"({bw1:6.1f} GB/s effective)")

    m2, bw2, _ = stage2_decode(dev, args.iters)
    us2 = m2 * 1e3
    print(f"[2] GPU HBM -> tensor cores (decode, batch=1)  : {us2:8.1f} us   "
          f"({bw2:6.1f} GB/s effective HBM read)")

    m3, tf, ptus, T, _ = stage3_compute(dev, args.iters, args.batch)
    print(f"[3] Tensor-core compute (per token, from batch={T}) : {ptus:8.3f} us   "
          f"({tf:6.1f} TFLOPS achieved)")
    print("-" * 68)
    ratio = us2 / ptus if ptus > 0 else 0
    print(f"On-device decode is ~{ratio:.0f}x memory-bound "
          f"(stage 2 HBM read {us2:.1f} us  vs  stage 3 compute {ptus:.3f} us/token)")
    print(f"Offload path (experts in CPU DRAM): stage 1 dominates at "
          f"{m1*1e3/1e3:.2f} ms/layer/token")


if __name__ == "__main__":
    main()
