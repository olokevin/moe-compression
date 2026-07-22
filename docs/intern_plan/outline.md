### Introduction

- ultimate goal: heterogeneous budget across experts
  - reduce total parameters (compression): allocate different compression budget across experts
  - reduce active parameters: load a different set of parameters in experts for each token

### Background

expert statistics

[summarize the findings of docs/results/stats/expert_stats.md, and some example figures]

showing the contribution and compressibility of each expert can greatly vary

expert compression methods:

- per-expert nystrom ()
- MoLAE / MoBE:
  - group multiple experts and do SVD together
  - what is the resulting expert sizes

heterogeneous budget allocation:

present brief summary of the following two papers, how they achieved heterogeneous ratio across experts, what’s their metric and rationale behind it, and a walk through of the whole algorithm

- **Attribution-Guided and Coverage-Maximized Pruning for Structural MoE Compression,**
  - combine expert contribution and compressibility
- RFID-MoE
  - Expert routing frequency & compressibility

### Stage 1: heterogeneous budget allocation across experts for total parameter reduction

actions:

- implement existing methods, and ther combinatons, report

preliminary results:

- 只考虑减少total param：可以压缩很多而不掉性能
- heterogeneous c.f.: **只减total，不减active**

Qwen 30B-A3B

benchmark: Hellasag 80.7, MMLU 81.7

|                              |                   |                    | Hellaswag    |          | MMLU         |          |
| ---------------------------- | ----------------- | ------------------ | ------------ | -------- | ------------ | -------- |
|                              | total param ratio | active param ratio | post-comress | training | post-comress | training |
| Benchmark                    |                   |                    | 80.7         |          | 81.7         |          |
| Nystrom uniform              | 1.5               | 1.5                | 63.9         | 80.2     | 70.3         | 77.2     |
| Attribution-guided           | 1.5               | 1.1                | 77.90        |          | 71.22        |          |
| Attribution-guided + Nystrom | 1.5               | 1.1                | 78.40        |          | 73.00        |          |
| MoBE                         |                   |                    |              |          |              |          |
| MoBE + ratio                 |                   |                    |              |          |              |          |

### Stage 2: adaptive budget allocation at inference time: reduce active parameters

Heads up:

- heterogeous budget allocation greatly help preserving performance, but it does not reduce much active parameters

To guarantee reduction on active parameters:

- setting a ratio for activated parameters
- on-the-fly determine the budget allocation for each token during inference

Why this is different

- orthogonal to current stream of top-p or adaptive top-k (different number of activated experts for each token)

objective:

- given the
  - router probability output
  - the pre-calibrated error of removing each expert / channel
- determine the set of experts / channels that minimize the error for current token, following the global active parameter budget

Idea 1: first determine budget of each activated expert, then determine the parameters to be activated within each expert

[summarize proposal/per_token_budget.md]

Idea 2:  rank all channels in a MoE layer, then select the channels according to the cols/rows

[summarize proposal/per_token_channel_allocation.md]
