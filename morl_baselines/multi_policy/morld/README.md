# Multi-Objective Reinforcement Learning based on Decomposition (MORL/D)

## Performance
Link to report: https://api.wandb.ai/links/florianfelten/pskjmvod.



python benchmark/launch_experiment.py --algo MORLD --env-id deep-sea-treasure-concave-v0 --num-timesteps 1000000 --gamma 0.99 --ref-point 0 -50 --auto-tag True --wandb-entity openrlbenchmark --seed 0 --init-hyperparams "scalarization_method:'tch'" "evaluation_mode:'esr'" "policy_name:'EUPG'" "weight_adaptation_method:'PSA'" "policy_args:{net_arch:np.array([32, 32])}"

--train-hyperparams ...
