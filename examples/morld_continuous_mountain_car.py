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
        parent_writer=parent_writer,
        learning_starts=10_000,
        seed=None,
        net_arch=[32, 32],
    )
    return Policy(id, weights=weight, wrapped=wrapped)


def main():

    gamma = 0.99

    known_front = []

    algo = MORLD(
        env_name="mo-MountainCarContinuous-v0",
        num_envs=1,
        exchange_every=30_000,
        pop_size=6,
        policy_factory=policy_factory,
        scalarization_method="ws",
        evaluation_mode="ser",
        ref_point=np.array([-1000.0, -1000.0]),
        gamma=gamma,
        log=True,
        seed=None,
        neighborhood_size=1,
        shared_buffer=False,
        sharing_mechanism=[],
        weight_adaptation_method=None,
        front=known_front,
    )

    algo.train(total_timesteps=int(1e6))


if __name__ == "__main__":
    for i in range(3):
        main()
