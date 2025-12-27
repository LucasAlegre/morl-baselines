import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.morld.morld import MORLD


def main():
    gamma = 0.99

    env = mo_gym.make("mo-lunar-lander-v3")
    eval_env = mo_gym.make("mo-lunar-lander-v3")

    # CLI equivalent:
    # python experiments/benchmark/launch_experiment.py \
    # --algo morld \
    # --env-id mo-lunar-lander-v3 \
    # --seed 0 \
    # --num-timesteps 200000 \
    # --gamma 0.99 \
    # --ref-point -101 -1001 -101 -101 \
    # --auto-tag True \
    # --init-hyperparams "policy_name:'MOSACDiscrete'" "shared_buffer:True" "exchange_every:5000" "pop_size:6" "weight_adaptation_method:'PSA'" "policy_args:{'target_net_freq':200, 'batch_size':128, 'buffer_size':1000000, 'net_arch':[256, 256, 256, 256], 'update_frequency': 1, 'target_entropy_scale':0.3}"
    algo = MORLD(
        env=env,
        exchange_every=5000,
        pop_size=6,
        policy_name="MOSACDiscrete",
        scalarization_method="ws",
        evaluation_mode="ser",
        gamma=gamma,
        log=False,
        neighborhood_size=1,
        update_passes=10,
        shared_buffer=True,
        sharing_mechanism=[],
        weight_adaptation_method="PSA",
        seed=0,
        policy_args={
            "target_net_freq": 200,
            "batch_size": 128,
            "buffer_size": 1000000,
            "net_arch": [256, 256, 256, 256],
            "update_frequency": 1,
            "target_entropy_scale": 0.3,
        },
    )

    algo.train(
        eval_env=eval_env,
        total_timesteps=200_000,
        ref_point=np.array([-101, -1001, -101, -101]),
        known_pareto_front=None,
    )


if __name__ == "__main__":
    main()
