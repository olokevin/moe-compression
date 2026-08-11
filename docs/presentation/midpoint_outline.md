# Midpoint Presentation Outline

**Duration:** 35–40 minutes (~28 slides, ~1.5 min/slide avg)

### Background

- Introduce MoE model
- introduce what is in a MoE FFN expert
- why MoE prevails: outperform dense model at the same inference computation

what needs to be improced

- reduce total parameters
  - benefit both single batch decode on edge (fit in DRAM) and cloude serving (reduced computation and memory access)
- reduce active parameters
  - benefit single batch decode (memory-bounded)

### overview of current design and results

reduce active parameters

- a brief summary of the components
  - activate 25% of parameters in up_proj
- 

### Motivation:

- Limitation of MoE pretraining
  - router has no information of the experts
  - each (activated) expert receive the same gradient for update
  - no load balancing
- Redundancy it introduces:
  - experts learn overlap features
  - expert could be compressible
- conclusion: there effective “experts” should live in a finer granularity → certain channel at work

High-level framework

- For each token, only activate the channels (neurons) that contributes most are activated
- [have a figure: on the left is a sample sequence, like “where is”. on the right is a sample moe layer (4 layers, 2 activated, use mesh so that each square denote a parameter. have three rectangles to denote up, gate, down). have two stream to indicate the forward on these two token: upper stream for “where”, highlight this token, two experts activated (with thicker border), and more importantly only a few channels (corresponding position in up, gate, down) is highlighted, denote activated. lower stream for “is”: 2 experts activate, 1 expert is the same as “where”, but differnet channels get activated (highlighted)]
- design challenges:
  - How to find out the channels that contribute most
  - how does this bring real throughput acceleration

our framework:

[summarize the `Online — token-specific channel selection via up_pro` section in docs/report/channel_experts.md]

note that our current best formulation uses full gate_proj, and partial activate up and down

- gate_proj as the “key”: use the activation magnitude
- activate the corresponding rows in up_proj, and columns in down_proj
- the gate_proj it self can undergoes low-rank compression to further boost efficiency
- a channel expert predictor to pre-fetch the activated parameters in the next layer → no waiting

etails of design

- gate_proj as channel router
  - predict what channel experts to activate for each token
  - problem: the router itself has no information of the experts
  - gate_proj produces (approximately) sparse activation →
  - use a compressed up_proj as a built-in predictor of channel experts for each token
- limitation:
  - need full
  - need to wait gate_proj
- Low-rank compression on gate_proj by MoBE
  - [introduce]
- Expert predictor
  - []

### Failing experiments

- summarize the design of level-1 (fixed ranking of channels, so as to reduce active parameter in gate_proj as well) in
- what potential benefits it brings
- explain why:
  - the channel contribution differs across tokens → a pre-calibrated channel importance ranking is less favored

### Efficiency Benchmarks

cloud, large batch size → almost load whole parameters

edge, single batch size → parameter offload, reduce active parameters

- first introduce the edge and cloud settings
- then present the results
- for the edge stream, a break down of the contribution of each stream
  - mobe compressed up/gate
  - dynamic load
  - expert predictor
  - other kernel / system design

### Future steps:

- System design throughput:
  - on edge device: predict the experts in next layer and pre-fetch
  - memory fetching pattern
  - on GPU:
- reduce total parameters:
  - mobe:
    - better than nystrom on initial 1.5x compression ratio
    - orthogonal to: another 1.5x
  - build the current reduce active parameter on it
- learn how many experts for each token
