import fire
import mo_gymnasium as mo_gym
import numpy as np
import wandb as wb
from mo_gymnasium.evaluation import policy_evaluation_mo

from morl_baselines.common.utils import extrema_weights, random_weights
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport


# from gymnasium.wrappers.record_video import RecordVideo


def best_vector(values, w):
    max_v = values[0]
    for i in range(1, len(values)):
        if values[i] @ w > max_v @ w:
            max_v = values[i]
    return max_v


def main(algo: str, gpi_pd: bool, g: int, timesteps_per_iter: int = 10000, seed: int = 0):
    def make_env():
        env = mo_gym.make("minecart-v0")
        env = mo_gym.MORecordEpisodeStatistics(env, gamma=0.98)
        return env

    env = make_env()
    eval_env = make_env()  # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)

    # print(env.convex_coverage_set(frame_skip=4, discount=0.98, incremental_frame_skip=True, symmetric=True))

    def model_training_schedule(timestep):
        if timestep < 100000:
            return 250
        else:
            return 250

    if algo == "ols" or algo == "gpi-ls":
        agent = GPIPD(
            env,
            num_nets=2,
            max_grad_norm=None,
            learning_rate=3e-4,
            gamma=0.98,
            batch_size=128,
            net_arch=[256, 256, 256, 256],
            buffer_size=int(2e5),
            initial_epsilon=1.0,
            final_epsilon=0.05,
            epsilon_decay_steps=50000,
            learning_starts=100,
            alpha_per=0.6,
            min_priority=0.01,
            per=gpi_pd,
            gpi_pd=gpi_pd,
            envelope=False,  # algo != 'ols',
            use_gpi=True,
            gradient_updates=g,
            target_net_update_freq=200,
            tau=1,
            dyna=gpi_pd,
            dynamics_uncertainty_threshold=1.5,
            dynamics_net_arch=[256, 256, 256],  # [200, 200, 200, 200],
            dynamics_buffer_size=int(1e5),
            dynamics_rollout_batch_size=25000,
            dynamics_train_freq=model_training_schedule,
            dynamics_rollout_freq=250,
            dynamics_rollout_starts=5000,
            dynamics_rollout_len=1,
            real_ratio=0.5,
            log=True,
            project_name="MORL Baselines - MineCart",
            experiment_name=f"{algo} + gpid-pd={gpi_pd} g={g}",
        )
    elif algo == "envelope":
        agent = Envelope(
            env,
            num_nets=2,
            max_grad_norm=None,
            learning_rate=3e-4,
            gamma=0.98,
            batch_size=64,
            n_sample_w=4,
            net_arch=[256, 256, 256, 256],
            buffer_size=int(2e5),
            initial_epsilon=1.0,
            final_epsilon=0.05,
            epsilon_decay_steps=50000,
            learning_starts=100,
            alpha_per=0.6,
            min_priority=0.01,
            per=False,
            use_gpi=False,
            dyna=False,
            envelope=True,
            gradient_updates=g,
            target_net_update_freq=200,
            tau=1,
            log=True,
            project_name="MORL Baselines - MineCart",
            experiment_name=f"{algo} g={g}",
        )

    if algo == "ols" or algo == "gpi-ls":
        linear_support = LinearSupport(num_objectives=3, epsilon=0.0 if algo == "ols" else None)

    weight_history = []

    test_tasks = list(random_weights(dim=3, seed=42, n=100 - 3, dist="dirichlet")) + extrema_weights(3)
    ccs = eval_env.convex_coverage_set(frame_skip=4, discount=0.98, incremental_frame_skip=True, symmetric=True)
    max_iter = 15
    for iter in range(1, max_iter + 1):
        if algo == "ols" or algo == "gpi-ls":
            if algo == "gpi-ls":
                agent.set_weight_support(linear_support.get_weight_support())
                w = linear_support.next_weight(algo="gpi-ls", gpi_agent=agent, env=eval_env)
            else:
                w = linear_support.next_weight(algo="ols")

            if w is None:
                break
        else:
            w = None

        print("Next weight vector:", w)
        weight_history.append(w)
        if algo == "gpi-ls":
            M = linear_support.get_weight_support() + linear_support.get_corner_weights(top_k=4) + [w]
        elif algo == "ols":
            M = linear_support.get_weight_support() + [w]
        else:
            M = None

        agent.learn(
            total_timesteps=timesteps_per_iter,
            weight=w,
            weight_support=M,
            change_w_every_episode=algo == "gpi-ls",
            eval_env=eval_env,
            eval_freq=1000,
            reset_num_timesteps=False,
            reset_learning_starts=False,
        )

        if algo != "envelope":
            if algo == "ols":
                value = policy_evaluation_mo(agent, eval_env, w, rep=5)
                linear_support.add_solution(value, w)
            elif algo == "gpi-ls":
                for wcw in M:
                    n_value = policy_evaluation_mo(agent, eval_env, wcw, rep=5)
                    linear_support.add_solution(n_value, wcw, add_not_improved=False)

        if algo != "envelope":
            gpi_returns_test_tasks = [
                policy_evaluation_mo(agent, eval_env, w, rep=5, return_scalarized_value=False) for w in test_tasks
            ]
            mean_gpi_returns_test_tasks = np.mean([np.dot(w, q) for w, q in zip(test_tasks, gpi_returns_test_tasks)], axis=0)
            wb.log({"eval/Mean Utility GPI": mean_gpi_returns_test_tasks, "iteration": iter})

            max_gap_gpi = max(
                [np.dot(best_vector(ccs, w), w) - np.dot(best_vector(gpi_returns_test_tasks, w), w) for w in test_tasks]
            )
            wb.log({"eval/Max Gap GPI": max_gap_gpi, "iteration": iter})

        agent.use_gpi = False
        no_gpi_returns_test_tasks = [
            policy_evaluation_mo(agent, eval_env, w, rep=5, return_scalarized_value=False) for w in test_tasks
        ]
        mean_no_gpi_returns_test_tasks = np.mean(
            [np.dot(psi, w) for (psi, w) in zip(no_gpi_returns_test_tasks, test_tasks)], axis=0
        )
        agent.use_gpi = algo != "envelope"
        wb.log({"eval/Mean Utility No GPI": mean_no_gpi_returns_test_tasks, "iteration": iter})

        max_gap_no_gpi = max(
            [np.dot(best_vector(ccs, w), w) - np.dot(best_vector(no_gpi_returns_test_tasks, w), w) for w in test_tasks]
        )
        wb.log({"eval/Max Gap No GPI": max_gap_no_gpi, "iteration": iter})

        agent.save(filename=f"{algo}-g={g}-gpi-pd={gpi_pd}-it={iter}-seed={seed}-minecart", save_replay_buffer=False)

    agent.close_wandb()


if __name__ == "__main__":
    fire.Fire(main)
