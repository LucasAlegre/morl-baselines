# Hyperparameter optimization

MORL-Baselines contains an early solution to the problem of hyperparameter optimization for MORL.
The problem and solution are introduced and discussed in the following paper:
[F. Felten, D. Gareev, E.-G. Talbi, and G. Danoy, “Hyperparameter Optimization for Multi-Objective Reinforcement Learning.” arXiv, Oct. 25, 2023. doi: 10.48550/arXiv.2310.16487.](https://arxiv.org/abs/2310.16487)


A script to launch the hyperparameter sweep is available in [`benchmark/launch_experiment.py`](https://github.com/LucasAlegre/morl-baselines/experiments/hyperparameter_search/launch_sweep.py).

An example usage of such script is the following:

```bash
python experiments/hyperparameter_search/launch_sweep.py \
--algo envelope \
--env-id minecart-v0 \
--ref-point 0 0 -200 \
--sweep-count 100 \
--seed 10 \
--num-seeds 3 \
--config-name envelope \
--train-hyperparams num_eval_weights_for_front:100 reset_num_timesteps:False eval_freq:10000 total_timesteps:10000
```

It will launch a HP search for Envelope Q-Leaning on minecart, using `[0, 0, -200]` as reference point for hypervolume. It will try 100 values of hyperparameters. The parameters distributions are specified in a yaml file specified by `config-name` (by default same name as the algorithm), it has to be in the directory. For each set of HP values, the algorithm will be trained on 3 different seeds, starting from 10 (so 10, 11, 12).
