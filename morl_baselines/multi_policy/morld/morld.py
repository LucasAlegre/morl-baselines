"""MORL/D Multi-Objective Reinforcement Learning based on Decomposition.

See Felten, Talbi & Danoy (2024): https://jair.org/index.php/jair/article/view/15702.
"""

import math
import os
import time
from typing import Callable, List, Optional, Tuple, Union
from typing_extensions import override

import gymnasium as gym
import numpy as np
import torch as th
from mo_gymnasium.wrappers import MONormalizeReward
from torch import optim

from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.networks import polyak_update
from morl_baselines.common.pareto import ParetoArchive
from morl_baselines.common.scalarization import tchebicheff, weighted_sum
from morl_baselines.common.utils import nearest_neighbors
from morl_baselines.common.weights import equally_spaced_weights, random_weights
from morl_baselines.single_policy.esr.eupg import EUPG
from morl_baselines.single_policy.ser.mosac_continuous_action import MOSAC
from morl_baselines.single_policy.ser.mosac_discrete_action import MOSACDiscrete


POLICIES = {
    "MOSAC": MOSAC,
    "MOSACDiscrete": MOSACDiscrete,
    "EUPG": EUPG,
}


class Policy:
    """Individual policy for MORL/D."""

    def __init__(self, id: int, weights: np.ndarray, wrapped: MOPolicy):
        """Initializes a policy.

        Args:
            id (int): Policy ID
            weights (np.ndarray): Weight vector
            wrapped (MOPolicy): Wrapped policy
        """
        self.id = id
        self.weights = weights
        self.wrapped = wrapped


class MORLD(MOAgent):
    """MORL/D implementation, decomposition based technique for MORL."""

    def __init__(
        self,
        env: gym.Env,
        scalarization_method: str = "ws",  # "ws" or "tch"
        evaluation_mode: str = "ser",  # "esr" or "ser"
        policy_name: str = "MOSAC",
        policy_args: dict = {},
        gamma: float = 0.995,
        pop_size: int = 6,
        seed: int = 42,
        rng: Optional[np.random.Generator] = None,
        exchange_every: int = int(4e4),
        neighborhood_size: int = 1,  # n = "n closest neighbors", 0=none
        dist_metric: Callable[[np.ndarray, np.ndarray], float] = lambda a, b: np.sum(
            np.square(a - b)
        ),  # distance metric between neighbors
        shared_buffer: bool = False,
        sharing_mechanism: List[str] = [],
        update_passes: int = 10,
        weight_init_method: str = "uniform",
        weight_adaptation_method: Optional[str] = None,  # "PSA" or None
        project_name: str = "MORL-Baselines",
        experiment_name: str = "MORL-D",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        device: Union[th.device, str] = "auto",
    ):
        """Initializes MORL/D.

        Args:
            env: environment
            scalarization_method: scalarization method to apply. "ws" or "tch".
            evaluation_mode: esr or ser (for evaluation env)
            policy_name: name of the underlying policy to use: "MOSAC", EUPG can be easily adapted.
            policy_args: arguments for the policy
            gamma: gamma
            pop_size: size of population
            seed: seed for RNG
            rng: RNG
            exchange_every: exchange trigger (timesteps based)
            neighborhood_size: size of the neighbordhood ( in [0, pop_size)
            dist_metric: distance metric between weight vectors to determine neighborhood
            shared_buffer: whether buffer should be shared or not
            sharing_mechanism: list containing potential sharing mechanisms: "transfer" is only supported for now.
            update_passes: number of times to update all policies after sampling from one policy.
            weight_init_method: weight initialization method. "uniform" or "random"
            weight_adaptation_method: weight adaptation method. "PSA" or None.
            project_name: For wandb logging
            experiment_name: For wandb logging
            wandb_entity: For wandb logging
            log: For wandb logging
            device: torch device
        """
        self.env = env
        super().__init__(self.env, device, seed=seed)
        self.gamma = gamma
        self.seed = seed

        if rng is not None:
            self.np_random = rng
        else:
            self.np_random = np.random.default_rng(self.seed)

        # (!) This is helpful for scalarization (!)
        for i in range(env.unwrapped.reward_space.shape[0]):
            env = MONormalizeReward(env, idx=i)

        self.evaluation_mode = evaluation_mode
        self.pop_size = pop_size

        # Scalarization and weights
        self.weight_init_method = weight_init_method
        self.weight_adaptation_method = weight_adaptation_method
        if self.weight_adaptation_method == "PSA":
            self.delta = 0.1
        else:
            self.delta = None
        if self.weight_init_method == "uniform":
            self.weights = np.array(equally_spaced_weights(self.reward_dim, self.pop_size, self.seed))
        elif self.weight_init_method == "random":
            self.weights = random_weights(self.reward_dim, n=self.pop_size, dist="dirichlet", rng=self.np_random)
        else:
            raise Exception(f"Unsupported weight init method: ${self.weight_init_method}")

        self.scalarization_method = scalarization_method
        if scalarization_method == "ws":
            self.scalarization = weighted_sum
        elif scalarization_method == "tch":
            self.scalarization = tchebicheff(tau=0.5, reward_dim=self.reward_dim)
        else:
            raise Exception(f"Unsupported scalarization method: ${self.scalarization_method}")

        # Sharing schemes
        self.neighborhood_size = neighborhood_size
        self.transfer = True if "transfer" in sharing_mechanism else False
        self.update_passes = update_passes
        self.exchange_every = exchange_every
        self.shared_buffer = shared_buffer
        self.dist_metric = dist_metric
        self.neighborhoods = [
            nearest_neighbors(
                n=self.neighborhood_size,
                current_weight=w,
                all_weights=self.weights,
                dist_metric=self.dist_metric,
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

        self.experiment_name += f"({policy_name})"
        if shared_buffer:
            self.experiment_name += "-SB"
        if self.weight_adaptation_method is not None and shared_buffer:
            self.experiment_name += f"+{self.weight_adaptation_method}"
        elif self.weight_adaptation_method is not None:
            self.experiment_name += f"-{self.weight_adaptation_method}"
        if self.transfer:
            self.experiment_name += "+transfer"

        self.policy_factory = POLICIES[policy_name]
        self.policy_name = policy_name
        self.policy_args = policy_args

        # Policies' population
        self.current_policy = 0  # For turn by turn selection
        self.population = [
            Policy(
                id=i,
                weights=w,
                wrapped=self.policy_factory(
                    id=i,
                    env=self.env,
                    weights=w,
                    scalarization=th.matmul if scalarization_method == "ws" else self.scalarization,
                    gamma=gamma,
                    log=self.log,
                    seed=self.seed,
                    parent_rng=self.np_random,
                    **policy_args,
                ),
            )
            for i, w in enumerate(self.weights)
        ]
        self.archive = ParetoArchive()
        if self.log:
            self.setup_wandb(
                project_name=self.project_name,
                experiment_name=self.experiment_name,
                entity=wandb_entity,
            )

        if self.shared_buffer:
            self.__share_buffers()

    @override
    def get_config(self) -> dict:
        return {
            "env_id": self.env.unwrapped.spec.id,
            "scalarization_method": self.scalarization_method,
            "evaluation_mode": self.evaluation_mode,
            "gamma": self.gamma,
            "pop_size": self.pop_size,
            "exchange_every": self.exchange_every,
            "neighborhood_size": self.neighborhood_size,
            "shared_buffer": self.shared_buffer,
            "update_passes": self.update_passes,
            "transfer": self.transfer,
            "weight_init_method": self.weight_init_method,
            "weight_adapt_method": self.weight_adaptation_method,
            "delta_adapt": self.delta,
            "project_name": self.project_name,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "log": self.log,
            "device": self.device,
            "policy_name": self.policy_name,
            **self.population[0].wrapped.get_config(),
            **self.policy_args,
        }

    def __share_buffers(self, neighborhood: bool = False):
        """Shares replay buffer among all policies.

        Args:
            neighborhood: whether we should share only with closest neighbors. False = share with everyone.
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

    def __select_candidate(self) -> Policy:
        """Candidate selection at every iteration. Turn by turn in this case."""
        candidate = self.population[self.current_policy]
        if self.current_policy + 1 == self.pop_size:
            self.iteration += 1
        self.current_policy = (self.current_policy + 1) % self.pop_size
        return candidate

    def __eval_policy(self, policy: Policy, eval_env: gym.Env, num_eval_episodes_for_front: int) -> np.ndarray:
        """Evaluates a policy.

        Args:
            policy: to evaluate
            eval_env: environment to evaluate on
            num_eval_episodes_for_front: number of episodes to evaluate on
        Return:
             the discounted returns of the policy
        """
        if self.evaluation_mode == "ser":
            acc = np.zeros(self.reward_dim)
            for _ in range(num_eval_episodes_for_front):
                _, _, _, discounted_reward = policy.wrapped.policy_eval(
                    eval_env,
                    weights=policy.weights,
                    scalarization=self.scalarization,
                    log=self.log,
                )
                acc += discounted_reward

        elif self.evaluation_mode == "esr":
            acc = np.zeros(self.reward_dim)
            for _ in range(num_eval_episodes_for_front):
                _, _, _, discounted_reward = policy.wrapped.policy_eval_esr(
                    eval_env,
                    weights=policy.weights,
                    scalarization=self.scalarization,
                    log=self.log,
                )
                acc += discounted_reward
        else:
            raise Exception("Evaluation mode must either be esr or ser.")
        return acc / num_eval_episodes_for_front

    def __eval_all_policies(
        self,
        eval_env: gym.Env,
        num_eval_episodes_for_front: int,
        num_eval_weights_for_eval: int,
        ref_point: np.ndarray,
        known_front: Optional[List[np.ndarray]] = None,
    ):
        """Evaluates all policies and store their current performances on the buffer and pareto archive."""
        evals = []
        for i, agent in enumerate(self.population):
            discounted_reward = self.__eval_policy(agent, eval_env, num_eval_episodes_for_front)
            evals.append(discounted_reward)
            # Storing current results
            self.archive.add(agent, discounted_reward)

        print("Current pareto archive:")
        print(self.archive.evaluations[:50])
        print(self.archive.evaluations[50:])

        if self.log:
            log_all_multi_policy_metrics(
                self.archive.evaluations,
                ref_point,
                self.reward_dim,
                self.global_step,
                n_sample_weights=num_eval_weights_for_eval,
                ref_front=known_front,
            )
        return evals

    def __share(self, last_trained: Policy):
        """Shares information between neighbor policies.

        Args:
            last_trained: last trained policy
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
                        neighbor_net.parameters(),
                        lr=neighbor_policy.wrapped.learning_rate,
                    )

    def __adapt_weights(self, evals: List[np.ndarray]):
        """Weight adaptation mechanism, many strategies exist e.g. MOEA/D-AWA.

        Args:
            evals: current evaluations of the population
        """

        def closest_non_dominated(eval_policy: np.ndarray) -> Tuple[Policy, np.ndarray]:
            """Returns the closest policy to eval_policy currently in the Pareto Archive.

            Args:
                eval_policy: evaluation where we want to find the closest one
            Return:
                closest individual and evaluation in the pareto archive
            """
            closest_distance = math.inf
            closest_nd = None
            closest_eval = None
            for eval_candidate, candidate in zip(self.archive.evaluations, self.archive.individuals):
                distance = np.sum(np.square(eval_policy - eval_candidate))
                if closest_distance > distance > 0.01:
                    closest_distance = distance
                    closest_nd = candidate
                    closest_eval = eval_candidate
            return closest_nd, closest_eval

        if self.weight_adaptation_method == "PSA":
            print("Adapting weights using PSA's method")
            # P. Czyzżak and A. Jaszkiewicz,
            # "Pareto simulated annealing—a metaheuristic technique for multiple-objective combinatorial optimization,"
            # Journal of Multi-Criteria Decision Analysis, vol. 7, no. 1, pp. 34–47, 1998,
            # doi: 10.1002/(SICI)1099-1360(199801)7:1<34::AID-MCDA161>3.0.CO;2-6.
            for i, p in enumerate(self.population):
                eval_policy = evals[i]
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

    def __update_others(self, current: Policy):
        """Runs policy improvements on all policies in the population except current.

        Args:
            current: current policy
        """
        print("Updating other policies...")
        for i in range(self.update_passes):
            for p in self.population:
                if len(p.wrapped.get_buffer()) > 0 and p != current:
                    p.wrapped.update()

    def save(self, save_dir="weights/", filename=None, save_replay_buffer=True):
        """Save the agent's weights and replay buffer for all policies in both the population and the archive."""
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if filename is None:
            filename = "morld_save"

        saved_params = {}

        # Save population
        print("Saving population...")
        for i, policy in enumerate(self.population):
            saved_params[f"population_policy_{i}"] = policy.wrapped.get_save_dict(save_replay_buffer)

        # Save archive
        print("Saving archive...")
        for i, (policy, eval_) in enumerate(zip(self.archive.individuals, self.archive.evaluations)):
            saved_params[f"archive_policy_{i}"] = policy.wrapped.get_save_dict(save_replay_buffer=False)
            saved_params[f"archive_policy_{i}_eval"] = eval_

        th.save(saved_params, os.path.join(save_dir, filename + ".tar"))

    def load(self, path, load_replay_buffer=True):
        """Load the agent weights from a file. Loads both population and archive."""
        params = th.load(path, map_location=self.device, weights_only=False)

        # Load population
        for i, policy in enumerate(self.population):
            key = f"population_policy_{i}"
            if key in params:
                policy.wrapped.load(params[key], load_replay_buffer=load_replay_buffer)
                policy.weights = policy.wrapped.weights
            else:
                print(f"Warning: {key} not found in save file.")

        # Load archive
        self.archive.individuals = []
        self.archive.evaluations = []
        i = 0
        while True:
            policy_key = f"archive_policy_{i}"
            eval_key = f"archive_policy_{i}_eval"
            if policy_key in params and eval_key in params:
                if len(self.population) > 0:
                    import copy

                    template = self.population[0]
                    archive_policy = copy.deepcopy(template)
                    archive_policy.wrapped.load(params[policy_key], load_replay_buffer=False)
                    archive_policy.weights = archive_policy.wrapped.weights
                    self.archive.individuals.append(archive_policy)
                    self.archive.evaluations.append(params[eval_key])
                else:
                    print("Warning: No population to use as template for archive policy loading.")
                    break
                i += 1
            else:
                break

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_episodes_for_front: int = 5,
        num_eval_weights_for_eval: int = 50,
        reset_num_timesteps: bool = False,
        checkpoints: bool = True,
        save_freq: int = 10000,
    ):
        """Trains the algorithm.

        Args:
            total_timesteps: total number of timesteps
            eval_env: evaluation environment
            ref_point: reference point for the hypervolume metric
            known_pareto_front: optimal pareto front for the problem if known
            num_eval_episodes_for_front: number of episodes for each policy evaluation
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            reset_num_timesteps: whether to reset the number of timesteps or not
            checkpoints (bool): Whether to save checkpoints.
            save_freq (int): Number of timesteps between checkpoints.
        """
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "num_eval_episodes_for_front": num_eval_episodes_for_front,
                }
            )

        # Init
        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
        start_time = time.time()

        obs, _ = self.env.reset()
        print("Starting training...")
        self.__eval_all_policies(
            eval_env,
            num_eval_episodes_for_front,
            num_eval_weights_for_eval,
            ref_point,
            known_pareto_front,
        )

        while self.global_step < total_timesteps:
            # selection
            policy = self.__select_candidate()
            # policy improvement
            policy.wrapped.train(self.exchange_every, eval_env=eval_env, start_time=start_time)
            self.global_step += self.exchange_every
            print(f"Switching... global_steps: {self.global_step}")
            for p in self.population:
                p.wrapped.global_step = self.global_step
            self.__update_others(policy)

            # Update archive
            evals = self.__eval_all_policies(
                eval_env,
                num_eval_episodes_for_front,
                num_eval_weights_for_eval,
                ref_point,
                known_pareto_front,
            )

            # cooperation
            self.__share(policy)
            # Adaptation
            self.__adapt_weights(evals)
            self.__adapt_ref_point()

            # Checkpoint
            print("Global step:", self.global_step)
            print("Save freq:", save_freq)
            if checkpoints and self.global_step % save_freq == 0:
                print(f"Saving checkpoint at step {self.global_step}")
                self.save(
                    filename=f"{self.experiment_name} step={self.global_step}",
                    save_replay_buffer=False,
                )

        print("done!")
        self.env.close()
        eval_env.close()
        self.close_wandb()
