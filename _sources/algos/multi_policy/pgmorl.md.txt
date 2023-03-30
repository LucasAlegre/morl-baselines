# PGMORL

Some code for this algorithm has been adapted from the original code provided by the authors of the paper [GitHub](https://github.com/mit-gfx/PGMORL).

## Applicability and limitations
* Supports continuous observation and continuous action spaces.
* Limited to 2 objectives for now.
* The post-processing phase (Pareto analysis stage) has not been implemented yet.

## Principle

```{figure} ../../_static/_images/pgmorl.png
   :alt: PGMORL
```

The principle of this algorithm is to rely on multiple PPO agents to look for various tradeoffs. This algorithm keeps a population of PPO agents along with their current performances. At each iteration, the algorithm selects a few best agents in the population and assigns to each of these a weight vector that is used to train further. The weight vector are generated based on a prediction model computed from historical data gathered during the learning process.

### MOPPO
Our implementation of multi-objective PPO is essentially a refactor of [cleanRL](https://github.com/vwxyzjn/cleanrl). The main difference is that the value network returns a multi-objective value and this value is then scalarized using a weighted sum and the given weight vector.

Note: it might be possible to enhance this algorithm by relying on something else than PPO.

```{eval-rst}
.. autoclass:: morl_baselines.single_policy.ser.mo_ppo.MOPPO
    :members:
```

### Weight generator - prediction model
See section 3.3 of the paper for more details.

```{eval-rst}
.. autoclass:: morl_baselines.multi_policy.pgmorl.pgmorl.PerformancePredictor
    :members:
```

### PGMORL
```{eval-rst}
.. autoclass:: morl_baselines.multi_policy.pgmorl.pgmorl.PGMORL
    :members:
```
