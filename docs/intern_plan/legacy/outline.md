# Outline

## Introduction:

* key of Macro (reference ref/macro_plan.md)

  * Compression composition: how to find the best config of compression ratio allocated for each axes
    * MLP/MOE intermediate dimension
    * depth (layers)
    * width (embedding/hidden dimension)
  * What really matters for compress-then-train?
    * Criteria: better final convergence
    * training-aware compression: find the parameters / subspaces that training favors
    * compress-aware training

Goal:

* an automatic learning-based way to discover the best compression composition config
* incorporate the practice we found that really improves compress-then-train
* Research/paper target: what is the capacity limit between smaller/larger model? can we train a large model once and nest best-performing smaller models inside?


## Preliminaries: 

### training-aware compression

Developed during the previous collaboration

[Math summarize of training_aware_compression.md]


### compression-aware training

First developed by Evegeny

in summary: freeze the compressed backbone and use LoRA training is favored in many sense (see original post). My additional comments:

* increase trainable parameters for additional degrees of freedom



### FuRA

[summarize the of paper/26_FuRA.pdf]


## First stage: 

Implement the automatic compression config learning engine

[follow paper/25_Nemotron Elastic- Towards Efficient Many-in-One Reasoning LLMs.pdf and paper/26_Star Elastic- Many-in-One Reasoning LLMs with Efficient Budget Control.pdf]

[summarize key ideas of elastic model]

* A large model that nested smaller models
* training that jointly optimize the config and performance of the nested small models

  * both the original large model and the nested small model co-exist and co-optimize -> better than prune and train the small model standalone
* general framework:

  * dynamic config: allocate c.f. per-axes (depth, width, moe) and per-layer and per-expert
  * budget loss: use parameter count as starting point, later can switch to Cor3 metric-of-interest (i.e., actual inference time of the config)



Method overview

[summarize the whole flow of paper/25_Nemotron Elastic- Towards Efficient Many-in-One Reasoning LLMs.pdf and paper]



Results

[Table from 26_Star Elastic- Many-in-One Reasoning LLMs with Efficient Budget Control.pdf]


Improvements:

* incorporate training-aware compression at  the initial importance ranking
  * prioritize more important and more trainable parametrs
* incorportate compression-aware training: LoRA/FuRA training with backbone frozen
  * Does not alter the importance ranking



Plans

* Week 1: FInalize the intern plan
* Week 2: prepare the training pipeline: Start with small MoE models (OLMoE-7B-1B, ) in standalone training experiments (not Percipio 2)
* Week 3: Implement the training-aware compression and compression-aware training, launch baseline training
* Week 4: launch the elastic training, get initial results, and determine the improvement steps


Expected deliverables:

* Elastic training for MoE models of interest ()Qwen3-30B-A3B) at c.f.=1.5 / 2.0: see how it compares with the existing baseline of manually picked compression config
* Quantify the improvement of training-aware compression and compression-aware training


## Second stage (paper target)

The first stage is mostly following existing literatures, and serve as a preliminary/baseline for the second stage

in the second stage, we target on the following research question:

*Can we only train a large model (8B), and after training in directly contains smaller models (6B, 4B, 2B, etc) all with performances with the highest capacity*


Current pipeline

To get a series of model

* Qwen series:
  * pre-train + post-train for each model size
  * each model size has differnet parameters
  * discrete model sizes
* Elastic models
  * Pre-train + Post-train for the largest model size
  * then elastic training
  * each model size share parameters
  * discrete model sizes
* All-in-one model (our proposed)
  * only pre-train + post-train for the largest model only
  * 
  * continuous model sizes

[A table to compare the different pipelines]

[A bar chart to compare the training resources (better to use GPU hours, tokens OK)]


method:

* at each layer, have a "importance" ranking of model parameters
* during training, let the more important information go to the more important parameters
* The (calibrated) spectrum of pretrained model could be a good guideline



How to achieve:

* FuRA: full-rank decomposition of the pre-train model, "importance" ranking of parameters
* use the spectrum as the preconditioning: more important information go to the more important parameters


Risks (and challenges to resolve)

* parameter importance ranking **across** layers
*
