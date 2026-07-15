"""
Measure active-parameter ratio (compressed vs uncompressed) for a one-shot
attribution-guided pruned Qwen3-30B-A3B, using real C4 routing.

Total params drop by ~prune_ratio uniformly, but *active* params per token do
NOT: pruning is non-uniform per expert (attr_coverage), and each token routes
to only top_k experts. So the active ratio is data-dependent -> we capture the
router's top_k choices on real C4 tokens and weight by each expert's kept count.

Ratio conventions reported:
  - expert-FFN-only: active kept expert channels / active full expert channels
  - full-model active: includes the (unchanged) attention/embed/router/norm params
"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.base.datasets import load_datasets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--mask", required=True)
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=1024)
    args = ap.parse_args()

    # ---- load mask -> per-(layer,expert) kept channel counts ----
    mk = torch.load(args.mask, map_location="cpu")
    im = mk["intermediate_masks"]            # [L, E, I] bool
    L, E, I = im.shape
    K = im.sum(-1).float()                   # [L, E] kept channels per expert
    print(f"[mask] layers={L} experts={E} intermediate={I} "
          f"total-kept-frac={K.sum().item()/(L*E*I):.4f}")

    # ---- load model (4-bit) ----
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto",
        attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    model.eval()
    cfg = model.config
    top_k = cfg.num_experts_per_tok
    H = cfg.hidden_size
    print(f"[model] hidden={H} top_k={top_k} num_experts={cfg.num_experts} "
          f"n_layers={cfg.num_hidden_layers} moe_intermediate={cfg.moe_intermediate_size}")
    assert E == cfg.num_experts and I == cfg.moe_intermediate_size

    # ---- fixed (non-expert) active params, counted analytically from module ----
    # shapes (dtype-independent: bnb 4-bit packs weights so .numel() is unreliable).
    # Everything that is NOT an expert FFN is active on every token: attention,
    # router gates, embeddings, norms, lm_head.
    import torch.nn as nn
    fixed_active = 0
    for name, mod in model.named_modules():
        if ".experts." in name:
            continue
        if hasattr(mod, "in_features") and hasattr(mod, "out_features"):
            fixed_active += int(mod.in_features) * int(mod.out_features)
            if getattr(mod, "bias", None) is not None:
                fixed_active += int(mod.out_features)
        elif isinstance(mod, nn.Embedding):
            fixed_active += int(mod.num_embeddings) * int(mod.embedding_dim)
    # tie check: if lm_head shares embed weights, we may double count once (minor).

    # analytic param accounting (dtype-independent) ----
    full_expert_ch = top_k * I                    # active expert channels per layer (uncompressed)
    K_dev = K                                     # [L,E] on cpu

    # capture routing: hook each gate (Linear) -> logits -> topk indices
    per_layer_active_ch = [None] * L              # accumulate summed kept-ch of routed experts
    layer_of = {}
    gates = []
    li = 0
    for name, mod in model.named_modules():
        if name.endswith(".mlp.gate") and hasattr(mod, "out_features") and mod.out_features == E:
            layer_of[id(mod)] = li
            gates.append((li, mod))
            li += 1
    assert li == L, f"found {li} gates, expected {L}"

    # accumulators (per token) collected across batches
    stats = {"tok_active_ch": [], "tok_full_ch": L * full_expert_ch}
    buf = {"per_tok": None}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            logits = out if torch.is_tensor(out) else out[0]
            logits = logits.float()               # [tokens, E]
            idx = logits.topk(top_k, dim=-1).indices   # [tokens, top_k]
            kept = K_dev.to(logits.device)[layer_idx]  # [E]
            sel = kept[idx].sum(-1).cpu()         # [tokens] kept ch summed over routed experts
            if buf["per_tok"] is None:            # accumulate on CPU (model may be sharded
                buf["per_tok"] = sel.clone()      # across GPUs -> layers on different devices)
            else:
                buf["per_tok"] += sel
        return hook

    handles = [m.register_forward_hook(make_hook(idx)) for idx, m in gates]

    texts = load_datasets("c4", tok, max_samples=args.n_samples)
    with torch.no_grad():
        for i, t in enumerate(texts):
            enc = tok(t, return_tensors="pt", truncation=True, max_length=args.max_length)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            buf["per_tok"] = None
            model(**enc)
            stats["tok_active_ch"].append(buf["per_tok"].cpu())
            if (i + 1) % 8 == 0:
                print(f"  processed {i+1}/{len(texts)}")

    for h in handles:
        h.remove()

    active_ch = torch.cat(stats["tok_active_ch"])     # [num_tokens], summed over L layers
    full_ch = stats["tok_full_ch"]                    # scalar: L*top_k*I
    ffn_ratio = active_ch / full_ch                   # per-token expert-FFN active ratio

    # full-model active ratio: (fixed + 3*H*active_ch) / (fixed + 3*H*full_ch)
    fixed = fixed_active
    full_active_params = fixed + 3 * H * full_ch
    comp_active_params = fixed + 3 * H * active_ch.double()
    full_ratio = comp_active_params / full_active_params

    def stat(x):
        return x.mean().item(), x.min().item(), x.max().item(), x.std().item()

    print("\n================ ACTIVE PARAM RATIO (compressed / uncompressed) ================")
    print(f"tokens measured: {active_ch.numel()}  (over {len(texts)} C4 samples)")
    fm = stat(ffn_ratio)
    print(f"[expert-FFN only]  avg={fm[0]:.4f}  min={fm[1]:.4f}  max={fm[2]:.4f}  std={fm[3]:.4f}")
    tm = stat(full_ratio.float())
    print(f"[full-model active] avg={tm[0]:.4f}  min={tm[1]:.4f}  max={tm[2]:.4f}  std={tm[3]:.4f}")
    print(f"[reference] uniform total-param kept-frac = {K.sum().item()/(L*E*I):.4f} "
          f"(i.e. {100*(1-K.sum().item()/(L*E*I)):.1f}% total expert params removed)")
    print(f"[reference] fixed(non-expert) active params = {fixed/1e9:.3f}B ; "
          f"uncompressed active = {full_active_params/1e9:.3f}B")

if __name__ == "__main__":
    main()
