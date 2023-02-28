# MORL/D

Multi-Objective Reinforcement Learning based on Decomposition. The idea of this framework is to decompose the multi-objective problem into a set of single-objective problems. The single-objective problems are then solved by a single-objective RL algorithm (or something close). There are multiple tricks which can be applied to improve the sample efficiency when compared to just sequentially solving each single-objective RL problem.

The paper presents a taxonomy to classify the different approaches to multi-objective RL. The framework presented in this repository is supposed to be modular, so that it can be used to easily implement the approaches presented in the paper.

## Performances
<iframe src="https://wandb.ai/florianfelten/MORL-Baselines/reports/MORL-D-experimental-results--VmlldzozNDYzMzg5" style="border:none;height:1024px;width:100%">


```{eval-rst}
.. autoclass:: morl_baselines.multi_policy.morld.morld.MORLD
    :members:
```
