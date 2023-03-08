import argparse
import os
import subprocess
from distutils.util import strtobool

import mo_gymnasium as mo_gym
import numpy as np
import requests
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import (
    GPIPDContinuousAction,
)
from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import (
    MPMOQLearning,
)
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
from morl_baselines.multi_policy.pcn.pcn import PCN
from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL


ALGOS = {
    "pgmorl": PGMORL,
    "envelope": Envelope,
    "gpi_pd_continuous": GPIPDContinuousAction,
    "gpi_pd_discrete": GPIPD,
    "mpmoql": MPMOQLearning,
    "pcn": PCN,
    "pql": PQL,
    "ols": MPMOQLearning,
}

ENVS_WITH_KNOWN_PARETO_FRONT = [
    "deep-sea-treasure-concave-v0",
    "deep-sea-treasure-v0",
    "minecart-v0",
    "resource-gathering-v0",
]


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
        prs = requests.get(f"https://api.github.com/search/issues?q=repo:LucasAlegre/morl-baselines+is:pr+{git_commit}")
        if prs.status_code == 200:
            prs = prs.json()
            if len(prs["items"]) > 0:
                pr = prs["items"][0]
                pr_number = pr["number"]
                wandb_tag += f",pr-{pr_number}"
        print(f"identified github pull request: {pr_number}")
    except Exception as e:
        print(e)

    return wandb_tag


def main():
    args = parse_args()
    print(args)

    if args.auto_tag:
        if "WANDB_TAGS" in os.environ:
            raise ValueError(
                "WANDB_TAGS is already set. Please unset it before running this script or run the script with --auto-tag False"
            )
        wandb_tag = autotag()
        if len(wandb_tag) > 0:
            os.environ["WANDB_TAGS"] = wandb_tag

    if args.algo != "pgmorl":
        env = MORecordEpisodeStatistics(mo_gym.make(args.env_id), gamma=args.gamma)
        eval_env = mo_gym.make(args.env_id)
        print(f"Instantiating {args.algo} on {args.env_id}")
        algo = ALGOS[args.algo](
            env=env,
            gamma=args.gamma,
            log=True,
            seed=args.seed,
            wandb_entity=args.wandb_entity,
        )
        if args.env_id in ENVS_WITH_KNOWN_PARETO_FRONT:
            known_pareto_front = env.unwrapped.pareto_front(gamma=args.gamma)
        else:
            known_pareto_front = None

        print("Training starts... Let's roll!")
        algo.train(
            total_timesteps=args.num_timesteps,
            eval_env=eval_env,
            ref_point=np.array(args.ref_point),
            known_pareto_front=known_pareto_front,
        )

    else:
        # PGMORL creates its own environments because it requires wrappers
        algo = ALGOS[args.algo](
            env_id=args.env_id,
            ref_point=np.array(args.ref_point),
            gamma=args.gamma,
            log=True,
            seed=args.seed,
            wandb_entity=args.wandb_entity,
        )
        algo.train(total_timesteps=args.num_timesteps)


if __name__ == "__main__":
    main()
