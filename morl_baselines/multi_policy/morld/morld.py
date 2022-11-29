import time
from typing import Union, Callable, Optional

import gym
import mo_gym
import numpy as np
import torch as th
import wandb
from gym.wrappers import TimeLimit
from mo_gym import MORecordEpisodeStatistics
from mo_gym.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure, CONCAVE_MAP
from pymoo.util.ref_dirs import get_reference_directions
from torch.utils.tensorboard import SummaryWriter

from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.pareto import ParetoArchive
from morl_baselines.common.performance_indicators import hypervolume, sparsity
from morl_baselines.common.scalarization import weighted_sum, tchebicheff
from morl_baselines.common.utils import random_weights, nearest_neighbors
from morl_baselines.single_policy.ser.mo_ppo import make_env


class Policy:
    def __init__(self, id: int, weights: np.ndarray, wrapped: MOPolicy):
        self.id = id
        self.weights = weights
        self.wrapped = wrapped


def make_env(env_id, seed, idx, run_name, gamma):
    def thunk():
        env = TimeLimit(DeepSeaTreasure(render_mode=None, dst_map=CONCAVE_MAP), max_episode_steps=500)
        # env = mo_gym.make(env_id, render_mode=None)
        env = MORecordEpisodeStatistics(env, gamma=gamma)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class MORLD(MOAgent):
    """
    MORL/D implementation, decomposition based technique for MORL
    """

    def __init__(
        self,
        env_name: str,
        policy_factory: Callable[
            [int, gym.Env, np.ndarray, Callable[[np.ndarray, np.ndarray], float], float, Optional[SummaryWriter]], Policy
        ],
        scalarization_method: str,  # "ws" or "tch"
        evaluation_mode: str,  # "esr" or "ser"
        ref_point: np.ndarray,
        gamma: float = 0.995,
        pop_size: int = 10,
        num_envs: int = 1,
        seed: int = 42,
        exchange_every: int = int(1e5),
        neighborhood_size: int = 2,  # n = "n closest neighbors", 0=none
        dist_metric: Callable[[np.ndarray, np.ndarray], float] = np.dot,  # distance metric between neighbors
        shared_buffer: bool = False,
        weight_init_method: str = "uniform",
        project_name: str = "MORL-Baselines",
        experiment_name: str = "MORL-D",
        log: bool = True,
        device: Union[th.device, str] = "auto",
    ):
        """
        :param env_name: environment id
        :param policy_factory: factory method to create low level, single objective policies, takes
            int: id
            gym.Env: environment
            np.ndarray: weight,
            Callable: scalarization function
            float: gamma
        :param scalarization_method: scalarization method to apply. "ws" or "tch".
        :param evaluation_mode: esr or ser (for evaluation env)
        :param ref_point: reference point for the hypervolume metric
        :param gamma: gamma
        :param pop_size: size of population
        :param num_envs: number of parallel environments
        :param seed: seed for RNG
        :param exchange_every: exchange trigger (timesteps based)
        :param neighborhood_size: size of the neighbordhood ( in [0, pop_size)
        :param dist_metric: distance metric
        :param shared_buffer: whether buffer should be shared or not
        :param weight_init_method: weight initialization method. "uniform" or "random"
        :param project_name: For wandb logging
        :param experiment_name: For wandb logging
        :param log: For wandb logging
        :param device: torch device
        """
        self.env_name = env_name
        env = mo_gym.make(env_name)
        super().__init__(env, device)
        env.close()
        self.num_envs = num_envs
        self.gamma = gamma
        self.seed = seed

        self.env = make_env(self.env_name, self.seed, 0, experiment_name, self.gamma)()
        self.eval_env = make_env(self.env_name, self.seed, 1, experiment_name, self.gamma)()

        self.evaluation_mode = evaluation_mode
        self.ref_point = ref_point
        self.pop_size = pop_size

        # Scalarization
        self.weight_init_method = weight_init_method
        if self.weight_init_method == "uniform":
            self.weights = get_reference_directions("energy", self.reward_dim, self.pop_size)
        elif self.weight_init_method == "random":
            self.weights = random_weights(self.reward_dim, n=self.pop_size, dist="dirichlet")
        else:
            raise f"Unsupported weight init method: ${self.weight_init_method}"

        self.scalarization_method = scalarization_method
        if scalarization_method == "ws":
            self.scalarization: Callable[[np.ndarray, np.ndarray], float] = weighted_sum
        elif scalarization_method == "tch":
            self.scalarization: Callable[[np.ndarray, np.ndarray], float] = tchebicheff(tau=0.5, reward_dim=self.reward_dim)
        else:
            raise f"Unsupported scalarization method: ${self.scalarization_method}"

        # Sharing schemes
        self.neighborhood_size = neighborhood_size
        self.exchange_every = exchange_every
        self.shared_buffer = shared_buffer
        self.dist_metric = dist_metric
        self.neighborhoods = [
            nearest_neighbors(n=self.neighborhood_size, current_weight=w, all_weights=self.weights, sim=self.dist_metric)
            for w in self.weights
        ]
        print("Weights:", self.weights)
        print("Neighborhoods:", self.neighborhoods)

        # Logging
        self.global_step = 0
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log

        if self.log:
            wandb.tensorboard.patch(root_logdir="/tmp/" + self.experiment_name, pytorch=True)
            self.setup_wandb(project_name=self.project_name, experiment_name=self.experiment_name)

        # Policies
        self.current_policy = 0  # For selection
        self.policy_factory = policy_factory
        self.population = [
            self.policy_factory(i, env, w, self.scalarization, gamma, self.writer) for i, w in enumerate(self.weights)
        ]
        self.archive = ParetoArchive()

    def get_config(self) -> dict:
        return {
            "env_name": self.env_name,
            "scalarization_method": self.scalarization_method,
            "evaluation_mode": self.evaluation_mode,
            "ref_point": self.ref_point,
            "gamma": self.gamma,
            "pop_size": self.pop_size,
            "num_envs": self.num_envs,
            "exchange_every": self.exchange_every,
            "neighborhood_size": self.neighborhood_size,
            "shared_buffer": self.shared_buffer,
            "weight_init_method": self.weight_init_method,
            "project_name": self.project_name,
            "experiment_name": self.experiment_name,
            "log": self.log,
            "device": self.device,
        }

    def __select_candidate(self):
        candidate = self.population[self.current_policy]
        self.current_policy = (self.current_policy + 1) % self.pop_size  # turn by turn selection
        return candidate

    def __eval_all_agents(self):
        """
        Evaluates all agents and store their current performances on the buffer and pareto archive
        """
        for i, agent in enumerate(self.population):
            if self.evaluation_mode == "ser":
                _, _, _, discounted_reward = agent.wrapped.policy_eval(
                    self.eval_env, weights=agent.weights, writer=self.writer
                )
            elif self.evaluation_mode == "esr":
                _, _, _, discounted_reward = agent.wrapped.policy_eval_esr(
                    self.eval_env, weights=agent.weights, scalarization=self.scalarization, writer=self.writer
                )
            else:
                raise "Evaluation mode must either be esr or ser."
            # Storing current results
            self.archive.add(agent, discounted_reward)

        print("Current pareto archive:")
        print(self.archive.evaluations)
        hv = hypervolume(self.ref_point, self.archive.evaluations)
        sp = sparsity(self.archive.evaluations)
        self.writer.add_scalar("charts/hypervolume", hv, self.global_step)
        self.writer.add_scalar("charts/sparsity", sp, self.global_step)

    def train(self, total_timesteps: int, reset_num_timesteps: bool = False):
        # Init
        start_time = time.time()

        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes

        obs, _ = self.env.reset()
        self.__eval_all_agents()

        while self.global_step < total_timesteps:
            policy = self.__select_candidate()
            policy.wrapped.train(self.exchange_every, eval_env=self.eval_env)
            self.global_step += self.exchange_every

            # TODO sharing mechanism
            # TODO adaptation
            self.__eval_all_agents()

            # Logging speed
            print("SPS:", int(self.global_step / (time.time() - start_time)))
            self.writer.add_scalar(
                "charts/SPS",
                int(self.global_step / (time.time() - start_time)),
                self.global_step,
            )

        print("done!")
        self.env.close()
        self.eval_env.close()
        self.close_wandb()
