import fire
import mo_gymnasium as mo_gym
import numpy as np
import wandb as wb
from mo_gymnasium.evaluation import policy_evaluation_mo

from morl_baselines.common.performance_indicators import expected_utility
from morl_baselines.common.utils import equally_spaced_weights
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import (
    GPIPDContinuousAction,
)
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport


# from gymnasium.wrappers.record_video import RecordVideo


def main(algo: str, gpi_pd: bool, g: int, timesteps_per_iter: int = 10000, seed: int = 0):
    def make_env():
        env = mo_gym.make("mo-hopper-v4", cost_objective=False, max_episode_steps=500)
        env = mo_gym.MORecordEpisodeStatistics(env, gamma=0.99)
        return env

    env = make_env()
    eval_env = make_env()  # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)
    reward_dim = env.reward_space.shape[0]

    agent = GPIPDContinuousAction(
        env,
        gradient_updates=g,
        min_priority=0.1,
        batch_size=128,
        buffer_size=int(4e5),
        dynamics_rollout_starts=1000,
        dynamics_rollout_len=5,
        dynamics_rollout_freq=250,
        dynamics_rollout_batch_size=50000,
        dynamics_train_freq=250,
        dynamics_buffer_size=200000,
        dynamics_real_ratio=0.1,
        dynamics_min_uncertainty=2.0,
        dyna=gpi_pd,
        per=gpi_pd,
        project_name="MORL Baselines",
        experiment_name="GPI-PD - Hopper",
        log=True,
    )

    if algo == "ols" or algo == "gpi-ls":
        linear_support = LinearSupport(num_objectives=reward_dim, epsilon=0.0 if algo == "ols" else None)

    test_tasks = equally_spaced_weights(reward_dim, n=100, seed=42)
    max_iter = 10
    for iter in range(1, max_iter + 1):
        if algo == "ols" or algo == "gpi-ls":
            if algo == "gpi-ls":
                agent.set_weight_support(linear_support.get_weight_support())
                agent.use_gpi = True
                w = linear_support.next_weight(algo="gpi-ls", gpi_agent=agent, env=eval_env)
                agent.use_gpi = False
            else:
                w = linear_support.next_weight(algo="ols")

            if w is None:
                break
        else:
            raise ValueError(f"Unknown algorithm {algo}.")

        print("Next weight vector:", w)
        if algo == "gpi-ls":
            M = linear_support.get_weight_support() + linear_support.get_corner_weights(top_k=4) + [w]
        elif algo == "ols":
            M = linear_support.get_weight_support() + [w]
        else:
            M = None

        agent.train(
            total_timesteps=timesteps_per_iter,
            weight=w,
            weight_support=M,
            change_weight_every_episode=algo == "gpi-ls",
            eval_env=eval_env,
            eval_freq=1000,
        )

        if algo == "ols":
            value = policy_evaluation_mo(agent, eval_env, w, rep=5)
            linear_support.add_solution(value, w)
        elif algo == "gpi-ls":
            for wcw in M:
                n_value = policy_evaluation_mo(agent, eval_env, wcw, rep=5)
                linear_support.add_solution(n_value, wcw)

        # Evaluation
        gpi_returns_test_tasks = [
            policy_evaluation_mo(agent, eval_env, w, rep=5, return_scalarized_value=False) for w in test_tasks
        ]
        mean_gpi_returns_test_tasks = np.mean([np.dot(w, q) for w, q in zip(test_tasks, gpi_returns_test_tasks)], axis=0)
        wb.log(
            {"eval/Mean Utility - GPI": mean_gpi_returns_test_tasks, "iteration": iter}
        )  # This is the EU computed in the paper
        eu = expected_utility(gpi_returns_test_tasks, test_tasks)
        wb.log({"eval/EU - GPI": eu, "iteration": iter})

        agent.save(filename=f"{algo}-g={g}-gpi-pd={gpi_pd}-it={iter}-seed={seed}-minecart", save_replay_buffer=False)

    agent.close_wandb()


if __name__ == "__main__":
    fire.Fire(main)
