# Attribution-Guided and Coverage-Maximized Pruning for Structural MoE Compression

> **[Attribution-Guided and Coverage-Maximized Pruning for Structural MoE Compression](https://openreview.net/pdf?id=oreET6Wz52)**  
> Yifu Ding, Jiacheng Wang, Ge Yang, Yongcheng Jing, Jinyang Guo, Xianglong Liu, Dacheng Tao  
> *Forty-Third International Conference on Machine Learning (ICML 2026) — **Spotlight***

---

## Overview

This repository contains the official implementation of our ICML 2026 Spotlight paper. We propose a structured pruning framework for **Mixture-of-Experts (MoE) Large Language Models** that achieves significant parameter reduction with minimal performance degradation.

Our method operates in four stages:

1. **Channel Scoring** — Computes attribution-based importance scores for each expert's intermediate dimensions using first-order Taylor approximations over calibration data.
2. **Mask Generation** — Plans pruning budgets across layers (*inter-layer*) and across experts within each layer (*intra-layer*) via coverage-maximized allocation.
3. **Structural Pruning** — Physically removes pruned channels from expert feed-forward networks, producing a smaller dense model checkpoint.
4. **Fine-tuning** — Recovers accuracy through LoRA-based fine-tuning with optional gradual mask annealing.

### Supported Models

| Model | Parameters (Total / Active) |
|---|---|
| Qwen1.5-MoE-A2.7B | 14.3B / 2.7B |
| DeepSeek-MoE-16B | 16.4B / 2.8B |
| DeepSeek-V2-Lite | 15.7B / 2.4B |
| Qwen3-30B-A3B | 30.5B / 3.3B |

---

## Installation

```bash
pip install -r requirements.txt        # minimal set
# or, to reproduce the full paper environment:
pip install -r requirements_full.txt   # pins lm_eval==0.4.9.1 for evaluation
```

All commands below are run from the repository root. The scripts add the repo
root to `PYTHONPATH` themselves (modules are imported as `src.*`); if you invoke
Python directly, export it first:

```bash
export PYTHONPATH="$(pwd)"
```

---

## Usage

The framework runs in three stages that map directly onto the paper's method
(Section 4). Each stage writes artifacts that the next one consumes:

```
 Stage 1: Scoring            Stage 2: Prune + Train         Stage 3: Evaluate
 ─────────────────           ──────────────────────         ────────────────
 ALA (§4.1)          ──▶     CBA (§4.2) + AAR (§4.3)   ──▶   real slimming
 calibration         scores/  mask generation + LoRA   ckpt  + benchmarks
```

| Paper component | Where it runs | Config knob |
|---|---|---|
| **Attribution-Guided Loss Approximation** (§4.1) | `src/calibration/channel_scoring/main.py` | `--calib-datasets`, `--calib-batches` |
| **Coverage-Maximized Budget Allocation** (§4.2) | `generate_masks()` during train/eval | `prune_kwargs.mask_method_kwargs` |
| **Alignment-Aware Redistribution** (§4.3) | `generate_masks()` during train/eval | `prune_kwargs.adjust_masks_kwargs` |

### 1. Attribution-Guided Loss Approximation (ALA)

Runs calibration data through the model one layer at a time, perturbs each
expert, and collects the first-order Taylor loss proxy `−(∂L/∂z)ᵀz` plus
per-channel importance statistics in a single backward pass (Eq. 4).

```bash
bash scripts/scoring.sh
# equivalent to (all knobs are overridable env vars in the script):
#   python src/calibration/channel_scoring/main.py \
#       --model-name-or-path "Qwen/Qwen1.5-MoE-A2.7B" \
#       --calib-datasets c4 --calib-batches 200 --batch-size 8 \
#       --max-seq-length 512 --dtype bfloat16 --trust-remote-code \
#       --output-dir ./results/ --verbose
```

Outputs are written to `<output-dir>/<model>/<dataset>/scores/`
(e.g. `./results/Qwen_Qwen1.5-MoE-A2.7B/c4/scores/`):

- `expert_scores.pth` — per-channel metrics; `expert_out_token_contrib` is the ALA expert-wise loss proxy `ϕ`.
- `layerwise_loss.pth` — per-layer loss, used to seed **inter-layer** allocation.
- `gate_scores.pth` — router usage/output statistics.

Use `CALIB_BATCHES=200` for full calibration (~3M tokens as in the paper);
a smaller value gives a fast smoke test. The path to this `scores/` directory
is what you set as `scores_dir` in the configs below.

### 2. Coverage-Maximized Budget Allocation (CBA) + Alignment-Aware Redistribution (AAR)

CBA and AAR are not separate scripts — they run inside `generate_masks()` and
are controlled entirely by `prune_kwargs` in the YAML config. **CBA** (Algorithm 1)
turns the ALA scores into per-layer and per-expert channel budgets by maximizing
score coverage; **AAR** then rounds those budgets to hardware-friendly multiples
via Hamilton's largest-remainder rule.

```yaml
prune_kwargs:
  prune_ratio: 0.5                 # global fraction of channels to remove
  mask_method_kwargs:              # === CBA (§4.2) ===
    inter_layer_method: "loss_coverage"   # per-layer budget, seeded by layerwise_loss
    intra_layer_method: "attr_coverage"   # per-expert budget, seeded by ALA proxy ϕ
    intra_expert_metric: "activation"     # score that ranks channels within an expert
  adjust_masks_kwargs:             # === AAR (§4.3) ===
    align_inter: 64                # align kept channels to multiples of 64/128 (0 = off)
    min_per_expert: 128            # drop experts below this channel floor (0 = off)
scores_dir: ./results/Qwen_Qwen1.5-MoE-A2.7B/c4/scores
```

Set `align_inter: 0` and `min_per_expert: 0` to disable AAR (unaligned 50%
pruning, `Ours (P50%)` in the paper); set `align_inter: 128` for the aligned,
4-bit-friendly variant (`Ours Q(P25% Q4b)`).

### 3. One-shot pruned MoE model (no fine-tuning)

To produce and evaluate the structurally slimmed model directly from the scores
— running CBA + AAR + real slimming with no LoRA recovery — use the eval entry
point with `resume_path` left empty. `real_slim: true` physically removes the
pruned channels (not just masking) and `shrink_gate: true` prunes the router.

```bash
bash scripts/eval.sh
# equivalent to:
#   python src/train/merge_slim_eval.py --config configs/eval/qwen1_5_moe_a2_7b.yaml
```

In `configs/eval/qwen1_5_moe_a2_7b.yaml`, set `scores_dir` to your Stage 1
output, leave `resume_path` empty (or unset), and keep `real_slim: true`. This
reports the parameter reduction and, if `eval_task_names` is set (see Stage 5),
the downstream accuracy of the one-shot model.

### 4. LoRA fine-tuning (accuracy recovery)

Applies the CBA/AAR mask as a differentiable **fake-prune mask** (channels are
zeroed, gradients still flow), wraps the experts in LoRA/DoRA, and fine-tunes on
Alpaca to recover accuracy. Runs multi-GPU via `torchrun`.

```bash
# edit configs/train/<model>.yaml first: set scores_dir and output_dir
bash scripts/train.sh
# equivalent to:
#   torchrun --nproc_per_node=8 src/train/train.py \
#       --config configs/train/qwen1_5_moe_a2_7b_e2e_alpaca.yaml
```

Staged LoRA targets are toggled in the config: expert LoRA is always on;
`enable_gate_lora` and `enable_attn_lora` add router and attention adapters.
The mask used for training is saved next to each checkpoint as `masks.pth`.
When training finishes, the console prints the exact `resume_path` and
`mask_dir` values to copy into your eval config.

### 5. Evaluation

Evaluation reuses the same entry point (`src/train/merge_slim_eval.py`), but with
`resume_path` pointing at a fine-tuned checkpoint: it loads and merges the LoRA
adapter, applies the saved mask (`mask_dir`), builds the real slim model, and
benchmarks on
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

```bash
bash scripts/eval.sh
# equivalent to:
#   python src/train/merge_slim_eval.py --config configs/eval/qwen1_5_moe_a2_7b.yaml
```

Set these fields in the eval config:

- `resume_path` — the fine-tuned checkpoint directory from Stage 4.
- `mask_dir` — path to that checkpoint's `masks.pth` (skips mask re-generation).
- `eval_task_names` — comma-separated harness tasks, e.g.
  `"wikitext2,arc_easy,arc_challenge,hellaswag,piqa,boolq,winogrande,mmlu,gsm8k"`.
  (Leave empty to only report the parameter-reduction statistics.)

Set `test_speed: true` to also measure throughput/latency (Table 2 in the paper).

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{ding2026attribution,
  title     = {Attribution-Guided and Coverage-Maximized Pruning for Structural MoE Compression},
  author    = {Ding, Yifu and Wang, Jiacheng and Yang, Ge and Jing, Yongcheng and Guo, Jinyang and Liu, Xianglong and Tao, Dacheng},
  booktitle = {Proceedings of the Forty-Third International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026},
  note      = {Proceedings URL will be updated once available on PMLR}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
