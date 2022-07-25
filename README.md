# MORL Baselines

This repository aims at containing standard MORL algorithms, compatible with the [mo-gym](https://github.com/LucasAlegre/mo-gym) API.

## TODO

| Person  | Algo                | Paper                                | Existing implem                    | Done?   |
|---------|---------------------|--------------------------------------|------------------------------------|---------|
| Florian / Lucas | Envelope Q-Learning | https://arxiv.org/pdf/1908.08342.pdf | https://github.com/RunzheYang/MORL |         |

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