"""Launches an experiment on a given environment and algorithm.

Many parameters can be given in the command line, see the help for more infos.

Examples:
    python benchmark/launch_experiment.py --algo pcn --env-id deep-sea-treasure-v0 --num-timesteps 1000000 --gamma 0.99 --ref-point 0 -25 --auto-tag True --wandb-entity openrlbenchmark --seed 0 --init-hyperparams "scaling_factor:np.array([1, 1, 1])"
"""

import argparse
import os
import subprocess
from distutils.util import strtobool

import mo_gymnasium as mo_gym
import numpy as np
import requests
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import FlattenObservation, RecordVideo
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

from morl_baselines.common.evaluation import seed_everything
from morl_baselines.common.experiments import (
    ALGOS,
    ENVS_WITH_KNOWN_PARETO_FRONT,
    StoreDict,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="Name of the algorithm to run", choices=ALGOS.keys(), required=True)
    parser.add_argument("--env-id", type=str, help="MO-Gymnasium id of the environment to run", required=True)
    parser.add_argument("--num-timesteps", type=int, help="Number of timesteps to train for", required=True)
    parser.add_argument("--gamma", type=float, help="Discount factor to apply to the environment and algorithm", required=True)
    parser.add_argument(
        "--ref-point", type=float, nargs="+", help="Reference point to use for the hypervolume calculation", required=True
    )
    parser.add_argument("--seed", type=int, help="Random seed to use", default=42)
    parser.add_argument("--wandb-entity", type=str, help="Wandb entity to use", required=False)
    parser.add_argument(
        "--auto-tag",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, the runs will be tagged with git tags, commit, and pull request number if possible",
    )
    parser.add_argument(
        "--record-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, the runs will be recorded with RecordVideo wrapper.",
    )
    parser.add_argument("--record-video-ep-freq", type=int, default=5, help="Record video frequency (in episodes).")
    parser.add_argument(
        "--init-hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Override hyperparameters to use for the initiation of the algorithm. Example: --init-hyperparams learning_rate:0.001 final_epsilon:0.1",
        default={},
    )

    parser.add_argument(
        "--train-hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Override hyperparameters to use for the train method algorithm. Example: --train-hyperparams num_eval_weights_for_front:10 timesteps_per_iter:10000",
        default={},
    )

    return parser.parse_args()


def autotag() -> str:
    """This adds a tag to the wandb run marking the commit number, allows to versioning of experiments. From CleanRL's benchmark utility."""
    wandb_tag = ""
    print("autotag feature is enabled")
    try:
        git_tag = subprocess.check_output(["git", "describe", "--tags"]).decode("ascii").strip()
        wandb_tag = f"{git_tag}"
        print(f"identified git tag: {git_tag}")
    except subprocess.CalledProcessError:
        return wandb_tag

    git_commit = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"]).decode("ascii").strip()
    try:
        # try finding the pull request number on github
        prs = requests.get(
            f"https://api.github.com/search/issues?q=repo:LucasAlegre/morl-baselines+is:pr+{git_commit}"  # noqa
        )
        if prs.status_code == 200:
            prs = prs.json()
            if len(prs["items"]) > 0:
                pr = prs["items"][0]
                pr_number = pr["number"]
                wandb_tag += f",pr-{pr_number}"  # noqa
        print(f"identified github pull request: {pr_number}")
    except Exception as e:
        print(e)

    return wandb_tag


def main():
    args = parse_args()
    print(args)

    seed_everything(args.seed)

    if args.auto_tag:
        if "WANDB_TAGS" in os.environ:
            raise ValueError(
                "WANDB_TAGS is already set. Please unset it before running this script or run the script with --auto-tag False"
            )
        wandb_tag = autotag()
        if len(wandb_tag) > 0:
            os.environ["WANDB_TAGS"] = wandb_tag

    if args.algo == "pgmorl":
        # PGMORL creates its own environments because it requires wrappers
        print(f"Instantiating {args.algo} on {args.env_id}")
        eval_env = mo_gym.make(args.env_id)
        algo = ALGOS[args.algo](
            env_id=args.env_id,
            origin=np.array(args.ref_point),
            gamma=args.gamma,
            log=True,
            seed=args.seed,
            wandb_entity=args.wandb_entity,
            **args.init_hyperparams,
        )
        print(algo.get_config())

        print("Training starts... Let's roll!")
        algo.train(
            total_timesteps=args.num_timesteps,
            eval_env=eval_env,
            ref_point=np.array(args.ref_point),
            known_pareto_front=None,
            **args.train_hyperparams,
        )

    else:
        if "mario" in args.env_id:
            env = mo_gym.make(args.env_id, death_as_penalty=True)
            eval_env = mo_gym.make(args.env_id, death_as_penalty=True, render_mode="rgb_array" if args.record_video else None)
        else:
            env = mo_gym.make(args.env_id)
            eval_env = mo_gym.make(args.env_id, render_mode="rgb_array" if args.record_video else None)
        env = MORecordEpisodeStatistics(env, gamma=args.gamma)

        if "highway" in args.env_id:
            env = FlattenObservation(env)
            eval_env = FlattenObservation(eval_env)
        elif "mario" in args.env_id:

            def wrap_mario(env):
                from gymnasium.wrappers import (
                    FrameStack,
                    GrayScaleObservation,
                    ResizeObservation,
                    TimeLimit,
                )
                from mo_gymnasium.envs.mario.joypad_space import JoypadSpace
                from mo_gymnasium.wrappers import MOMaxAndSkipObservation

                env = JoypadSpace(env, SIMPLE_MOVEMENT)
                env = MOMaxAndSkipObservation(env, skip=4)
                env = ResizeObservation(env, (84, 84))
                env = GrayScaleObservation(env)
                env = FrameStack(env, 4)
                env = TimeLimit(env, max_episode_steps=1000)
                return env

            env = wrap_mario(env)
            eval_env = wrap_mario(eval_env)

        if args.record_video:
            eval_env = RecordVideo(
                eval_env,
                video_folder=f"videos/{args.algo}-{args.env_id}",
                episode_trigger=lambda ep: ep % args.record_video_ep_freq == 0,
            )

        print(f"Instantiating {args.algo} on {args.env_id}")
        if args.algo == "ols":
            args.init_hyperparams["experiment_name"] = "MultiPolicy MO Q-Learning (OLS)"
        elif args.algo == "gpi-ls":
            args.init_hyperparams["experiment_name"] = "MultiPolicy MO Q-Learning (GPI-LS)"

        algo = ALGOS[args.algo](
            env=env,
            gamma=args.gamma,
            log=True,
            seed=args.seed,
            wandb_entity=args.wandb_entity,
            **args.init_hyperparams,
        )
        if args.env_id in ENVS_WITH_KNOWN_PARETO_FRONT:
            known_pareto_front = env.unwrapped.pareto_front(gamma=args.gamma)
        else:
            known_pareto_front = None

        print(algo.get_config())

        print("Training starts... Let's roll!")
        algo.train(
            total_timesteps=args.num_timesteps,
            eval_env=eval_env,
            ref_point=np.array(args.ref_point),
            known_pareto_front=known_pareto_front,
            **args.train_hyperparams,
        )


if __name__ == "__main__":
    main()
