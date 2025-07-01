import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.morld.morld import MORLD


def main():
    gamma = 0.99

    env = mo_gym.make("mo-lunar-lander-v3")
    eval_env = mo_gym.make("mo-lunar-lander-v3")

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

    # Restore weights from checkpoint
    checkpoint_path = "weights/MORL-D(MOSACDiscrete)-SB+PSA step=30000.tar"
    algo.load(checkpoint_path, load_replay_buffer=False)

    # Evaluate all policies in the population
    print("Evaluating restored policies...")
    for i, policy in enumerate(algo.population):
        returns = []
        for ep in range(3):  # Run 3 episodes per policy
            _, _, _, discounted_reward = policy.wrapped.policy_eval(
                eval_env,
                weights=policy.weights,
                scalarization=algo.scalarization,
                log=False,
            )
            returns.append(discounted_reward)
        mean_return = np.mean(returns, axis=0)
        print(f"Policy {i} mean discounted return over 3 episodes: {mean_return}")

    # Evaluate all policies in the archive
    print("Evaluating restored archive...")
    print("restored archive size:", len(algo.archive.individuals))
    for i, policy in enumerate(algo.archive.individuals):
        returns = []
        for ep in range(5):  # Run 3 episodes per policy
            _, _, _, discounted_reward = policy.wrapped.policy_eval(
                eval_env,
                weights=policy.weights,
                scalarization=algo.scalarization,
                log=False,
            )
            returns.append(discounted_reward)
        mean_return = np.mean(returns, axis=0)
        print(
            f"Policy {i} mean discounted return over 3 episodes: {mean_return}, was {algo.archive.evaluations[i]} in the archive"
        )


if __name__ == "__main__":
    main()
