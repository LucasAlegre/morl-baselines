# Performance assessments

:warning: This document is a work in progress.

## Introduction
To ensure the implementation of the algorithms are correct, we want to test them on various environments. For the sake of reproducibility, we want to run for 10 seeds on various environments. For maintenance purposes and long-term support, these tests will be conducted on environments available in [MO-Gymnasium](www.github.com/farama-foundation/mo-gymnasium). Hence, we will not be able to test on some environments that were presented in original papers.


## Metrics
### Single-policy algorithms
For single-policy algorithms, the metric used will be the scalarized return of the policy on the evaluation env (utility). Keywords: `eval/scalarized_return` and `eval/scalarized_discounted_return`.

### Multi-policy algorithms
For multi-policy algorithms, we propose to rely on various metrics to assess the quality of the **discounted** Pareto Fronts (PF) or Convex Coverage Set (CCS). In general, we want to have a metric that is able to assess the convergence of the PF, a metric that is able to assess the diversity of the PF, and a hybrid metric assessing both. The metrics are implemented in `common/performance_indicators`. We propose to use the following metrics:
* **[Do not use]** (Diversity) Sparsity: average distance between each consecutive point in the PF. From the PGMORL paper [1]. Keyword: `eval/sparsity`.
* (Diversity) Cardinality: number of points in the PF. Keyword: `eval/cardinality`.
* (Convergence) IGD: a SOTA metric from Multi-Objective Optimization (MOO) literature. It requires a reference PF that we can compute a posteriori. That is, we do a merge of all the PFs found by the method and compute the IGD with respect to this reference PF. Keyword: `eval/igd`.
* (Hybrid) Hypervolume: a SOTA metric from MOO and MORL literature. Keyword: `eval/hypervolume`.

Moreover, some metrics relying on assumptions on the utility function of the user are proposed in the literature. These metric allow to have an idea on the true value on the user utility, whereas others such as hypervolume do not [2]. We propose to use the following metrics:
* EUM: Expected Utility Metric. From [3]. Keyword: `eval/eum`.
* MUL: Maximum Utility Loss for the problems we know the true CCS/PF. From [3]. Keyword: `eval/mul`.
For both these metrics, we propose to generate a number of equally spaced weights on the objective simplex. The number of weights is 50 by default, can be changed.

Finally, the PF can also be logged as a wandb table for a posteriori analysis. Keyword: `eval/front`.

Here is the function that logs all the metrics:
```{eval-rst}
.. autofunction:: morl_baselines.common.utils.log_all_multi_policy_metrics
```

## Storage

Our official performance metrics are sent to [openrlbenchmark](https://wandb.ai/openrlbenchmark/MORL-Baselines) on wandb. From there, it is possible to use [openrlbenchmark API](https://github.com/openrlbenchmark/openrlbenchmark) to query and plot the wanted results for paper format. Life is good when the full flow is automated 🥸.

## Benchmarking script
It is possible to run algorithms from a CLI and configure the parameters accordingly. The script is located in `benchmark/launch_experiment.py`.

## Algorithms

Below are the algorithms that we want to test along with the environments from their original papers which are supported in MO-Gymnasium.

This [issue](https://github.com/LucasAlegre/morl-baselines/issues/43) tracks the current performance assessment of the algorithms.

## References
[1]  J. Xu, Y. Tian, P. Ma, D. Rus, S. Sueda, and W. Matusik, “Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control,” in Proceedings of the 37th International Conference on Machine Learning, Nov. 2020, pp. 10607–10616. Available: https://proceedings.mlr.press/v119/xu20h.html

[2] C. Hayes et al., “A practical guide to multi-objective reinforcement learning and planning,” Autonomous Agents and Multi-Agent Systems, vol. 36, Apr. 2022, doi: 10.1007/s10458-022-09552-y.

[3] L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.
