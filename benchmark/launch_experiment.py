import argparse

import mo_gymnasium as mo_gym
import numpy as np
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
    parser.add_argument("--algo", type=str, help="Name of the algorithm to run", choices=ALGOS.keys())
    parser.add_argument("--env-id", type=str, help="MO-Gymnasium id of the environment to run")
    parser.add_argument("--num-timesteps", type=int, help="Number of timesteps to train for")
    parser.add_argument("--gamma", type=float, help="Discount factor to apply to the environment and algorithm")
    parser.add_argument("--ref-point", type=list, nargs="+", help="Reference point to use for the hypervolume calculation")
    parser.add_argument("--seed", type=int, help="Random seed to use", default=42)
    parser.add_argument("--wandb-entity", type=str, help="Wandb entity to use", required=False)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.algo != "pgmorl":
        env = MORecordEpisodeStatistics(mo_gym.make(args.env_id), gamma=args.gamma)
        eval_env = mo_gym.make(args.env_id)
        algo = ALGOS[args.algo](
            env=env,
            gamma=args.gamma,
            log=True,
            seed=args.seed,
            wandb_entity=args.wandb_entity,
        )
        if env.env_id in ENVS_WITH_KNOWN_PARETO_FRONT:
            known_pareto_front = env.unwrapped.pareto_front(gamma=args.gamma)
        else:
            known_pareto_front = None

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
