# MORL/D

Multi-Objective Reinforcement Learning based on Decomposition. The idea of this framework is to decompose the multi-objective problem into a set of single-objective problems. The single-objective problems are then solved by a single-objective RL algorithm (or something close). There are multiple tricks which can be applied to improve the sample efficiency when compared to just sequentially solving each single-objective RL problem.

See the paper [Multi-Objective Reinforcement Learning based on Decomposition](https://arxiv.org/abs/2311.12495) for more details.


```{eval-rst}
.. autoclass:: morl_baselines.multi_policy.morld.morld.MORLD
    :members:
```
