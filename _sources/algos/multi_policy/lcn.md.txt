# Lorenz Conditioned Networks

Lorenz Conditioned Networks (LCN) extends [Pareto Conditioned Networks (PCN)](pcn) by replacing the
dominance criterion used to select and condition on past episodes. Where PCN uses
Pareto dominance, LCN instead uses **Lorenz dominance**: each return's
objective values are sorted in increasing order and cumulatively summed (its *Lorenz vector*), and one
return Lorenz-dominates another if its Lorenz vector is at least as high in every position. Lorenz dominance favors returns that
distribute reward more *equitably* across objectives.

The `lcn_lambda` parameter controls **λ-Lorenz dominance**, an interpolation between strict Lorenz
dominance and ordinary dominance computed on the sorted returns:

```
fv = lcn_lambda * sorted(returns) + (1 - lcn_lambda) * lorenz_vector(returns)
```

`lcn_lambda=0` recovers strict Lorenz dominance (maximizes fairness), while values closer to `1` relax
the fairness bias towards rewarding raw performance on the best-performing sorted objectives. Pick one of
the two `distance_ref` options to choose between the two:

- `"nondominated"` (default): strict Lorenz dominance.
- `"lambda_lorenz"`: λ-Lorenz dominance, requires `lcn_lambda` to be set.

For more details, see:

Michailidis, D., Röpke, W., Roijers, D. M., Ghebreab, S., & Santos, F. P. (2026). Scalable
Multi-Objective Reinforcement Learning with Fairness Guarantees using Lorenz Dominance. Journal of
Artificial Intelligence Research, 85. https://jair.org/index.php/jair/article/view/19862

```{eval-rst}
.. autoclass:: morl_baselines.multi_policy.lcn.lcn.LCN
    :members:
```
