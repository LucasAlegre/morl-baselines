"""CAPQL training script with vectorized environments."""

import numpy as np
import torch
import gymnasium as gym
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv
from morl_baselines.multi_policy.capql.capql import CAPQL
import argparse
import os

torch.set_num_threads(1)

def make_env(env_id, seed, idx):
    def thunk():
        env = mo_gym.make(env_id)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return thunk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="mo-ant-v5")
    parser.add_argument("--total-timesteps", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Create vectorized environment
    env = MOSyncVectorEnv([make_env(args.env_id, args.seed, i) for i in range(args.num_envs)])

    # Initialize CAPQL policy
    policy = CAPQL(
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=256,
        gradient_updates=1,
        log=True,
        seed=args.seed,
    )

    env_eval = mo_gym.make(args.env_id)
    reward_dim = env_eval.unwrapped.reward_space.shape[0]
    ref_point = np.array([-100.0] * reward_dim)
    
    # Train
    print(f"Starting training loop for {args.env_id} with {args.num_envs} envs...")
    policy.train(
        total_timesteps=args.total_timesteps,
        eval_env=env_eval,
        ref_point=ref_point,
        known_pareto_front=None,
    )

    if args.eval:
        from morl_baselines.common.performance_indicators import hypervolume, sparsity, expected_utility
        from morl_baselines.common.weights import equally_spaced_weights
        from morl_baselines.common.evaluation import policy_evaluation_mo

        print(f"Running evaluation for {args.env_id}...")

        # 1. Capture Pareto points
        eval_weights = equally_spaced_weights(reward_dim, n=21)
        points = []
        for w in eval_weights:
            _, _, _, disc_vec_return = policy_evaluation_mo(policy, env_eval, w=w, rep=5)
            points.append(disc_vec_return)

        # 2. Compute metrics
        points = np.array(points)
        hv = hypervolume(ref_point, points)
        sp = sparsity(points)
        eu = expected_utility(points, weights_set=equally_spaced_weights(reward_dim, n=100))

        print(f"Evaluation results for {args.env_id}:")
        print(f"Hypervolume: {hv:.4f}")
        print(f"Sparsity: {sp:.4f}")
        print(f"Expected Utility: {eu:.4f}")

    env.close()

if __name__ == "__main__":
    main()
