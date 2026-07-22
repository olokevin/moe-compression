# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

This is a research planning repository for an internship project on **elastic training for MoE (Mixture-of-Experts) model compression**. It contains no source code — only research notes, mathematical derivations, reference documents, and PDF papers.

## Project Goal

Train a large MoE model once and automatically discover nested smaller models inside it (at various compression ratios) that perform optimally — replacing manual compression config search with learned elastic training.

## Document Structure

- `outline.md` — The intern plan: two-stage research outline with timelines and deliverables
- `training_aware_compression.md` — Mathematical summary of the two-sided whitening method (core prior work)
- `ref/macro_plan.md` — The broader Macro team plan (MoE compression for Cor3 hardware, Q3-Q4 2026)
- `ref/nystrom_combined.md` — Full derivation of trainability-aware MLP compression via joint forward+backward kernel
- `ref/ace.md` — ACE proposal: automated compression pathway search framework
- `paper/` — Reference PDFs (FuRA, Nemotron Elastic, Star Elastic)

## Key Concepts and Terminology

- **Macro compression** — Structural parameter reduction (pruning layers, width, MLP dimensions, experts) as opposed to Micro (quantization/precision) or Hologram (distillation)
- **Training-aware compression** — Uses both forward activation covariance (C_f) and backward gradient covariance (C_b) to select parameters that matter for continued training, not just inference
- **Compression-aware training** — Freeze compressed backbone + LoRA/FuRA fine-tuning
- **Elastic training** — Joint optimization that produces multiple nested model sizes from one training run (following Nemotron/Star Elastic papers)
- **Nyström selection** — Column-selection approximation used to pick which MLP hidden neurons to keep
- **K_joint** — The joint forward+backward kernel: C_f^{1/2} C_b C_f^{1/2}, central to the trainability-aware method
- **Cor3/P1/Tehama** — Target hardware platforms with specific memory/compute constraints
- **Percipio** — Internal compression platform where these methods are integrated
- **Compression factor (c.f.)** — Ratio of original to compressed parameter count (e.g., c.f.=1.5 means 1.5x reduction)

## How Documents Relate

The `outline.md` is the master plan. Stage 1 implements elastic training following the Nemotron/Star Elastic papers, incorporating training-aware compression (documented in `training_aware_compression.md` and `ref/nystrom_combined.md`) as an improvement. Stage 2 targets the research question of training one large model that directly contains optimal smaller models at continuous sizes. The broader team context is in `ref/macro_plan.md`, and `ref/ace.md` describes the longer-term vision for automated compression search.
