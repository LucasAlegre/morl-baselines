import math
import time
from typing import Union, Callable, Optional, List, Tuple

import gym
import mo_gym
import numpy as np
import torch as th
import wandb
from gym.wrappers import TimeLimit
from mo_gym import MORecordEpisodeStatistics, MOSyncVectorEnv, MONormalizeReward
from mo_gym.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure, CONCAVE_MAP
from pymoo.util.ref_dirs import get_reference_directions
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.pareto import ParetoArchive
from morl_baselines.common.performance_indicators import hypervolume, sparsity, igd
from morl_baselines.common.scalarization import weighted_sum, tchebicheff
from morl_baselines.common.utils import random_weights, nearest_neighbors, polyak_update


class Policy:
    def __init__(self, id: int, weights: np.ndarray, wrapped: MOPolicy):
        self.id = id
        self.weights = weights
        self.wrapped = wrapped


def make_env(env_id, seed, idx, capture_video, run_name, gamma):
    def thunk():
        env = mo_gym.make(env_id, render_mode=None, dst_map=CONCAVE_MAP)
        env = MORecordEpisodeStatistics(env, gamma=gamma)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        # Rewards are normalized to make the scalarization easier
        # for i in range(env.reward_space.shape[0]):
        #     env = MONormalizeReward(env, idx=i)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)
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
        eval_reps: int = 10,
        gamma: float = 0.995,
        pop_size: int = 10,
        num_envs: int = 1,
        seed: int = 42,
        exchange_every: int = int(1e5),
        neighborhood_size: int = 2,  # n = "n closest neighbors", 0=none
        dist_metric: Callable[[np.ndarray, np.ndarray], float] = lambda a, b: np.sum(
            np.square(a - b)
        ),  # distance metric between neighbors
        shared_buffer: bool = False,
        sharing_mechanism: List[str] = [],
        weight_init_method: str = "uniform",
        weight_adaptation_method: Optional[str] = None,  # "PSA" or None
        front: List[np.ndarray] = [],
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
        :param eval_reps: number of policy evaluation repetitions
        :param ref_point: reference point for the hypervolume metric
        :param gamma: gamma
        :param pop_size: size of population
        :param num_envs: number of parallel environments
        :param seed: seed for RNG
        :param exchange_every: exchange trigger (timesteps based)
        :param neighborhood_size: size of the neighbordhood ( in [0, pop_size)
        :param dist_metric: distance metric between weight vectors to determine neighborhood
        :param shared_buffer: whether buffer should be shared or not
        :param sharing_mechanism: list containing potential sharing mechanisms: "transfer" is only supported for now.
        :param weight_init_method: weight initialization method. "uniform" or "random"
        :param weight_adaptation_method: weight adaptation method. "PSA" or None.
        :param front: Known pareto front, if any.
        :param project_name: For wandb logging
        :param experiment_name: For wandb logging
        :param log: For wandb logging
        :param device: torch device
        """
        self.env_name = env_name
        self.gamma = gamma
        self.seed = seed
        self.num_envs = num_envs

        self.envs = MOSyncVectorEnv(
            [
                make_env(self.env_name, self.seed, i, capture_video=False, run_name=experiment_name, gamma=self.gamma)
                for i in range(self.num_envs)
            ]
        )
        super().__init__(self.envs, device)
        self.eval_env = make_env(
            self.env_name, self.seed, 0, capture_video=False, run_name=experiment_name, gamma=self.gamma
        )()

        self.evaluation_mode = evaluation_mode
        self.eval_reps = eval_reps
        self.ref_point = ref_point
        self.pop_size = pop_size

        # Scalarization and weights
        self.weight_init_method = weight_init_method
        self.weight_adaptation_method = weight_adaptation_method
        if self.weight_adaptation_method == "PSA":
            self.delta = 0.05
        else:
            self.delta = None
        if self.weight_init_method == "uniform":
            self.weights = get_reference_directions("energy", self.reward_dim, self.pop_size).astype(np.float32)
        elif self.weight_init_method == "random":
            self.weights = random_weights(self.reward_dim, n=self.pop_size, dist="dirichlet")
        else:
            raise Exception(f"Unsupported weight init method: ${self.weight_init_method}")

        self.scalarization_method = scalarization_method
        if scalarization_method == "ws":
            self.scalarization: Callable[[np.ndarray, np.ndarray], float] = weighted_sum
        elif scalarization_method == "tch":
            self.scalarization: Callable[[np.ndarray, np.ndarray], float] = tchebicheff(tau=0.5, reward_dim=self.reward_dim)
        else:
            raise Exception(f"Unsupported scalarization method: ${self.scalarization_method}")

        # Sharing schemes
        self.neighborhood_size = neighborhood_size
        self.transfer = True if "transfer" in sharing_mechanism else False
        self.exchange_every = exchange_every
        self.shared_buffer = shared_buffer
        self.dist_metric = dist_metric
        self.neighborhoods = [
            nearest_neighbors(
                n=self.neighborhood_size, current_weight=w, all_weights=self.weights, dist_metric=self.dist_metric
            )
            for w in self.weights
        ]
        print("Weights:", self.weights)
        print("Neighborhoods:", self.neighborhoods)

        # Logging
        self.global_step = 0
        self.iteration = 0
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log

        if self.log:
            self.setup_wandb(project_name=self.project_name, experiment_name=self.experiment_name)
            self.known_front = front
        else:
            self.writer = None

        # Policies' population
        self.current_policy = 0  # For turn by turn selection
        self.policy_factory = policy_factory
        self.population = [
            self.policy_factory(i, self.envs, w, self.scalarization, gamma, self.writer) for i, w in enumerate(self.weights)
        ]
        self.archive = ParetoArchive()

        if self.shared_buffer:
            self.__share_buffers()

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
            "transfer": self.transfer,
            "weight_init_method": self.weight_init_method,
            "weight_adapt_method": self.weight_adaptation_method,
            "delta_adapt": self.delta,
            "project_name": self.project_name,
            "experiment_name": self.experiment_name,
            "log": self.log,
            "device": self.device,
        }

    def __share_buffers(self, neighborhood: bool = False):
        """
        Shares replay buffer among all policies
        :param neighborhood: whether we should share only with closest neighbors. False = share with everyone.
        """
        if neighborhood:
            # Sharing only with neighbors
            for p in self.population:
                shared_buffer = p.wrapped.get_buffer()
                for n in self.neighborhoods[p.id]:
                    self.population[n].wrapped.set_buffer(shared_buffer)
        else:
            # Sharing with everyone
            shared_buffer = self.population[0].wrapped.get_buffer()
            for p in self.population:
                p.wrapped.set_buffer(shared_buffer)

    def __select_candidate(self):
        """
        Candidate selection at every iteration
        Turn by turn in this case.
        """
        candidate = self.population[self.current_policy]
        if self.current_policy + 1 == self.pop_size:
            self.iteration += 1
        self.current_policy = (self.current_policy + 1) % self.pop_size
        return candidate

    def __eval_policy(self, policy: Policy):
        """
        Evaluates a policy
        :param policy: to evaluate
        :return: the discounted reward
        """
        if self.evaluation_mode == "ser":
            acc = np.zeros(self.reward_dim)
            for _ in range(self.eval_reps):
                _, _, _, discounted_reward = policy.wrapped.policy_eval(
                    self.eval_env, weights=policy.weights, scalarization=self.scalarization, writer=self.writer
                )
                acc += discounted_reward

        elif self.evaluation_mode == "esr":
            acc = np.zeros(self.reward_dim)
            acc = np.zeros(self.reward_dim)
            for _ in range(self.eval_reps):
                _, _, _, discounted_reward = policy.wrapped.policy_eval_esr(
                    self.eval_env, weights=policy.weights, scalarization=self.scalarization, writer=self.writer
                )
                acc += discounted_reward
        else:
            raise Exception("Evaluation mode must either be esr or ser.")
        return acc / self.eval_reps

    def __eval_all_policies(self):
        """
        Evaluates all policies and store their current performances on the buffer and pareto archive
        """
        for i, agent in enumerate(self.population):
            discounted_reward = self.__eval_policy(agent)
            # Storing current results
            self.archive.add(agent, discounted_reward)

        print("Current pareto archive:")
        print(self.archive.evaluations)
        hv = hypervolume(self.ref_point, self.archive.evaluations)
        sp = sparsity(self.archive.evaluations)
        self.writer.add_scalar("charts/hypervolume", hv, self.global_step)
        self.writer.add_scalar("charts/sparsity", sp, self.global_step)

        if self.known_front:
            igd_metric = igd(known_front=self.known_front, current_estimate=self.archive.evaluations)
            self.writer.add_scalar("charts/IGD", igd_metric, self.global_step)

    def __share(self, last_trained: Policy):
        """
        Shares information between neighbor policies
        :param last_trained: last trained policy
        """
        if self.transfer and self.iteration == 0:
            # Transfer weights from trained policy to closest neighbors
            neighbors = self.neighborhoods[last_trained.id]
            last_trained_net = last_trained.wrapped.get_policy_net()
            for n in neighbors:
                # Filtering, makes no sense to transfer back to already trained policies
                # Relies on the assumption that we're making turn by turn
                if n > last_trained.id:
                    print(f"Transferring weights from {last_trained.id} to {n}")
                    neighbor_policy = self.population[n]
                    neighbor_net = neighbor_policy.wrapped.get_policy_net()

                    # Polyak update with tau=1 -> copy
                    # Can do something in the middle with tau < 1., which will be soft copies, similar to neuroevolution.
                    polyak_update(
                        params=last_trained_net.parameters(),
                        target_params=neighbor_net.parameters(),
                        tau=1.0,
                    )
                    # Set optimizer to point to the right parameters
                    neighbor_policy.wrapped.optimizer = optim.Adam(
                        neighbor_net.parameters(), lr=neighbor_policy.wrapped.learning_rate
                    )

    def __adapt_weights(self):
        """
        Weight adaptation mechanism. Many strategies exist e.g. MOEA/D-AWA.
        """

        def closest_non_dominated(eval_policy: np.ndarray) -> Tuple[Policy, np.ndarray]:
            """
            Returns the closest policy to eval_policy currently in the Pareto Archive
            :param eval_policy: evaluation where we want to find the closest one
            :return: closest individual and evaluation in the pareto archive
            """
            closest_distance = math.inf
            closest_nd = None
            closest_eval = None
            for eval_candidate, candidate in zip(self.archive.evaluations, self.archive.individuals):
                distance = np.sum(np.square(eval_policy - eval_candidate))
                if closest_distance > distance > 0.0:
                    closest_distance = distance
                    closest_nd = candidate
                    closest_eval = eval_candidate
            return closest_nd, closest_eval

        if self.weight_adaptation_method == "PSA":
            print("Adapting weights using PSA's method")
            # P. Czyzżak and A. Jaszkiewicz,
            # “Pareto simulated annealing—a metaheuristic technique for multiple-objective combinatorial optimization,”
            # Journal of Multi-Criteria Decision Analysis, vol. 7, no. 1, pp. 34–47, 1998,
            # doi: 10.1002/(SICI)1099-1360(199801)7:1<34::AID-MCDA161>3.0.CO;2-6.
            for p in self.population:
                eval_policy = self.__eval_policy(p)
                closest_nd, closest_eval = closest_non_dominated(eval_policy)

                new_weights = p.weights
                if closest_eval is not None:
                    for i in range(len(eval_policy)):
                        # Increases on the weights which are better than closest_eval, decreases on the others
                        if eval_policy[i] >= closest_eval[i]:
                            new_weights[i] = p.weights[i] * (1 + self.delta)
                        else:
                            new_weights[i] = p.weights[i] / (1 + self.delta)
                # Renormalizes so that the weights sum to 1.
                normalized = np.array(new_weights) / np.linalg.norm(np.array(new_weights), ord=1)
                p.wrapped.set_weights(normalized)
                p.weights = normalized
            new_weights = [p.weights for p in self.population]
            print(f"New weights {new_weights}")

    def __adapt_ref_point(self):
        # TCH ref point is automatically adapted in the TCH itself function for now.
        pass

    def train(self, total_timesteps: int, reset_num_timesteps: bool = False):
        # Init
        start_time = time.time()

        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes

        obs, _ = self.envs.reset()
        self.__eval_all_policies()

        while self.global_step < total_timesteps:
            policy = self.__select_candidate()
            policy.wrapped.train(self.exchange_every, eval_env=self.eval_env)
            self.global_step += self.exchange_every
            print(f"Switching... global_steps: {self.global_step}")
            self.__eval_all_policies()

            self.__share(policy)
            if self.current_policy % self.pop_size == 0:
                # Adapts weights and ref point after a full iteration
                self.__adapt_weights()
                self.__adapt_ref_point()

            # Logging speed
            print("SPS:", int(self.global_step / (time.time() - start_time)))
            self.writer.add_scalar(
                "charts/SPS",
                int(self.global_step / (time.time() - start_time)),
                self.global_step,
            )

        print("done!")
        self.envs.close()
        self.eval_env.close()
        self.close_wandb()
