# MORL Baselines

This repository aims at containing standard MORL algorithms, compatible with the [mo-gym](https://github.com/LucasAlegre/mo-gym) API.

## TODO

### Multi-policy
| Person          | Algo                             | Paper                                                                                                                        | Existing implem                                                                  | Done?              |
|-----------------|----------------------------------|------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|--------------------|
| Lucas | Envelope Q-Learning              | https://arxiv.org/pdf/1908.08342.pdf                                                                                         | https://github.com/RunzheYang/MORL                                               |                    |
| Florian         | PGMORL                           | https://people.csail.mit.edu/jiex/papers/PGMORL/paper.pdf / https://people.csail.mit.edu/jiex/papers/PGMORL/supp.pdf         | https://github.com/mit-gfx/PGMORL                                                | :heavy_check_mark: |
| Willem          | Pareto Q-Learning                | https://jmlr.org/papers/volume15/vanmoffaert14a/vanmoffaert14a.pdf                                                           | https://gitlab.ai.vub.ac.be/mreymond/deep-sea-treasure/-/blob/master/pareto_q.py |                    |
| Florian         | MPMOQLearning  (outer loop MOQL) | https://www.researchgate.net/publication/235698665_Scalarized_Multi-Objective_Reinforcement_Learning_Novel_Design_Techniques |                                                                                  | :heavy_check_mark: |
| Lucas | Optimistic Linear Support (OLS) | Section 3.3 of http://roijers.info/pub/thesis.pdf | :heavy_check_mark: |

### Single-policy
| Person  | Algo        | Paper                                                                                                                        | Existing implem | Done?              |
|---------|-------------|------------------------------------------------------------------------------------------------------------------------------|-----------------|--------------------|
| Florian / Lucas | MOQLearning | https://www.researchgate.net/publication/235698665_Scalarized_Multi-Objective_Reinforcement_Learning_Novel_Design_Techniques | -               | :heavy_check_mark: |

## Misc/utils ideas
- [ ] Dump Pareto front every x timesteps into a file (reporting)
- [ ] Standardized API for training (parser)
- [ ] Plotting helpers
- [ ] Perf Indicators helpers (e.g. hypervolume from pymoo assumes minimization whilst most problems here are maximization) 
- [ ] Statistics (avg, std, confidence interval)

## Contributions
Just add entries in the table above so we know who is on what. Then a proper PR with the implementation.

## Citing the Project

```bibtex
@misc{morl_baselines,
    ...
}
```

## Maintainers

MORL-Baselines is currently maintained by [Lucas N. Alegre](https://www.inf.ufrgs.br/~lnalegre/) (@LucasAlegre), Florian Felten (@ffelten), ...