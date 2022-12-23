from typing import Callable, Optional

import gym
import mo_gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from morl_baselines.multi_policy.morld.morld import MORLD, Policy
from morl_baselines.single_policy.esr.eupg import EUPG
import torch as th
from morl_baselines.single_policy.ser.mosac_continuous_action import MOSAC


def policy_factory(
    id: int,
    env: gym.Env,
    weight: np.ndarray,
    scalarization: Callable[[np.ndarray, np.ndarray], float],
    gamma: float,
    parent_writer: Optional[SummaryWriter],
) -> Policy:
    wrapped = MOSAC(
        id=id,
        envs=env,
        scalarization=th.matmul,
        weights=weight,
        gamma=gamma,
        log=True,
        seed=None,
        parent_writer=parent_writer,
    )
    return Policy(id, weights=weight, wrapped=wrapped)


def main():

    gamma = 0.99

    known_front = []

    algo = MORLD(
        env_name="mo-halfcheetah-v4",
        num_envs=1,
        exchange_every=int(5e4),
        pop_size=6,
        policy_factory=policy_factory,
        scalarization_method="ws",
        evaluation_mode="ser",
        ref_point=np.array([-100.0, -100.0]),
        gamma=gamma,
        log=True,
        neighborhood_size=1,
        update_passes=10,
        shared_buffer=True,
        sharing_mechanism=[],
        weight_adaptation_method=None,
        seed=None,
        experiment_name="MORL-D HalfCheetah",
        front=known_front,
    )

    algo.train(total_timesteps=int(3e6) + 1)


if __name__ == "__main__":
    for i in range(3):
        main()
