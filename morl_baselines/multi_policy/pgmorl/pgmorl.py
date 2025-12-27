"""PGMORL algorithm implementation.

Some code in this file has been adapted from the original code provided by the authors of the paper https://github.com/mit-gfx/PGMORL.
(!) The post-processing phase has not been implemented yet.
"""

import time
from copy import deepcopy
from itertools import product
from typing import List, Optional, Tuple, Union
from typing_extensions import override

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import torch as th
import wandb
from scipy.optimize import least_squares

from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.pareto import ParetoArchive
from morl_baselines.common.performance_indicators import hypervolume, sparsity
from morl_baselines.single_policy.ser.mo_ppo import MOPPO, MOPPONet, make_env


class PerformancePredictor:
    """Performance prediction model.

    Stores the performance deltas along with the used weights after each generation.
    Then, uses these stored samples to perform a regression for predicting the performance of using a given weight
    to train a given policy.
    Predicts: Weight & performance -> delta performance
    """

    def __init__(
        self,
        neighborhood_threshold: float = 0.1,
        sigma: float = 0.03,
        A_bound_min: float = 1.0,
        A_bound_max: float = 500.0,
        f_scale: float = 20.0,
    ):
        """Initialize the performance predictor.

        Args:
            neighborhood_threshold: The threshold for the neighborhood of an evaluation.
            sigma: The sigma value for the prediction model
            A_bound_min: The minimum value for the A parameter of the prediction model.
            A_bound_max: The maximum value for the A parameter of the prediction model.
            f_scale: The scale value for the prediction model.
        """
        # Memory
        self.previous_performance = []
        self.next_performance = []
        self.used_weight = []

        # Prediction model parameters
        self.neighborhood_threshold = neighborhood_threshold
        self.A_bound_min = A_bound_min
        self.A_bound_max = A_bound_max
        self.f_scale = f_scale
        self.sigma = sigma

    def add(self, weight: np.ndarray, eval_before_pg: np.ndarray, eval_after_pg: np.ndarray) -> None:
        """Add a new sample to the performance predictor.

        Args:
            weight: The weight used to train the policy.
            eval_before_pg: The evaluation before training the policy.
            eval_after_pg: The evaluation after training the policy.

        Returns:
            None
        """
        self.previous_performance.append(eval_before_pg)
        self.next_performance.append(eval_after_pg)
        self.used_weight.append(weight)

    def __build_model_and_predict(
        self,
        training_weights,
        training_deltas,
        training_next_perfs,
        current_dim,
        current_eval: np.ndarray,
        weight_candidate: np.ndarray,
        sigma: float,
    ):
        """Uses the hyperbolic model on the training data: weights, deltas and next_perfs to predict the next delta given the current evaluation and weight.

        Returns:
             The expected delta from current_eval by using weight_candidate.
        """

        def __f(x, A, a, b, c):
            return A * (np.exp(a * (x - b)) - 1) / (np.exp(a * (x - b)) + 1) + c

        def __hyperbolic_model(params, x, y):
            # f = A * (exp(a(x - b)) - 1) / (exp(a(x - b)) + 1) + c
            return (
                params[0] * (np.exp(params[1] * (x - params[2])) - 1.0) / (np.exp(params[1] * (x - params[2])) + 1)
                + params[3]
                - y
            ) * w

        def __jacobian(params, x, y):
            A, a, b, _ = params[0], params[1], params[2], params[3]
            J = np.zeros([len(params), len(x)])
            # df_dA = (exp(a(x - b)) - 1) / (exp(a(x - b)) + 1)
            J[0] = ((np.exp(a * (x - b)) - 1) / (np.exp(a * (x - b)) + 1)) * w
            # df_da = A(x - b)(2exp(a(x-b)))/(exp(a(x-b)) + 1)^2
            J[1] = (A * (x - b) * (2.0 * np.exp(a * (x - b))) / ((np.exp(a * (x - b)) + 1) ** 2)) * w
            # df_db = A(-a)(2exp(a(x-b)))/(exp(a(x-b)) + 1)^2
            J[2] = (A * (-a) * (2.0 * np.exp(a * (x - b))) / ((np.exp(a * (x - b)) + 1) ** 2)) * w
            # df_dc = 1
            J[3] = w

            return np.transpose(J)

        train_x = []
        train_y = []
        w = []
        for i in range(len(training_weights)):
            train_x.append(training_weights[i][current_dim])
            train_y.append(training_deltas[i][current_dim])
            diff = np.abs(training_next_perfs[i] - current_eval)
            dist = np.linalg.norm(diff / np.abs(current_eval))
            coef = np.exp(-((dist / sigma) ** 2) / 2.0)
            w.append(coef)

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        w = np.array(w)

        A_upperbound = np.clip(np.max(train_y) - np.min(train_y), 1.0, 500.0)
        initial_guess = np.ones(4)
        res_robust = least_squares(
            __hyperbolic_model,
            initial_guess,
            loss="soft_l1",
            f_scale=self.f_scale,
            args=(train_x, train_y),
            jac=__jacobian,
            bounds=([0, 0.1, -5.0, -500.0], [A_upperbound, 20.0, 5.0, 500.0]),
        )

        return __f(weight_candidate[current_dim], *res_robust.x)

    def predict_next_evaluation(self, weight_candidate: np.ndarray, policy_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the next evaluation of the policy.

        Use a part of the collected data (determined by the neighborhood threshold) to predict the performance
        after using weight to train the policy whose current evaluation is policy_eval.

        Args:
            weight_candidate: weight candidate
            policy_eval: current evaluation of the policy

        Returns:
            the delta prediction, along with the predicted next evaluations
        """
        neighbor_weights = []
        neighbor_deltas = []
        neighbor_next_perf = []
        current_sigma = self.sigma / 2.0
        current_neighb_threshold = self.neighborhood_threshold / 2.0
        # Iterates until we find at least 4 neighbors, enlarges the neighborhood at each iteration
        while len(neighbor_weights) < 4:
            # Enlarging neighborhood
            current_sigma *= 2.0
            current_neighb_threshold *= 2.0

            print(f"current_neighb_threshold: {current_neighb_threshold}")
            print(f"np.abs(policy_eval): {np.abs(policy_eval)}")
            if current_neighb_threshold == np.inf or current_sigma == np.inf:
                raise ValueError("Cannot find at least 4 neighbors by enlarging the neighborhood.")

            # Filtering for neighbors
            for previous_perf, next_perf, neighb_w in zip(self.previous_performance, self.next_performance, self.used_weight):
                if np.all(np.abs(previous_perf - policy_eval) < current_neighb_threshold * np.abs(policy_eval)) and tuple(
                    next_perf
                ) not in list(map(tuple, neighbor_next_perf)):
                    neighbor_weights.append(neighb_w)
                    neighbor_deltas.append(next_perf - previous_perf)
                    neighbor_next_perf.append(next_perf)

        # constructing a prediction model for each objective dimension, and using it to construct the delta predictions
        delta_predictions = [
            self.__build_model_and_predict(
                training_weights=neighbor_weights,
                training_deltas=neighbor_deltas,
                training_next_perfs=neighbor_next_perf,
                current_dim=obj_num,
                current_eval=policy_eval,
                weight_candidate=weight_candidate,
                sigma=current_sigma,
            )
            for obj_num in range(weight_candidate.size)
        ]
        delta_predictions = np.array(delta_predictions)
        return delta_predictions, delta_predictions + policy_eval


def generate_weights(delta_weight: float, dimensions: int = 2) -> np.ndarray:
    """Generates weights uniformly distributed over the objective dimensions. These weight vectors are separated by delta_weight distance.

    Args:
        delta_weight: distance between weight vectors
        dimensions: number of objectives
    Returns:
        all the candidate weights
    """
    # Generate a list of possible weight values based on delta_weight
    possible_weights = np.arange(0.0, 1.0 + delta_weight, delta_weight, dtype=np.float32)

    # Generate all possible combinations of these weights for the given number of dimensions
    weight_combinations = np.array(list(product(possible_weights, repeat=dimensions)), dtype=np.float32)

    # Filter out combinations that do not sum to 1
    valid_weight_vectors = weight_combinations[np.isclose(weight_combinations.sum(axis=1), 1.0)]

    return valid_weight_vectors


class PerformanceBuffer2d:
    """Stores the population. Divides the objective space in to n bins of size max_size.

    (!) restricted to 2D objective space (!)
    """

    def __init__(self, num_bins: int, max_size: int, origin: np.ndarray):
        """Initializes the buffer.

        Args:
            num_bins: number of bins
            max_size: maximum size of each bin
            origin: origin of the objective space (to have only positive values)
        """
        self.num_bins = num_bins
        self.max_size = max_size
        self.origin = -origin
        self.dtheta = np.pi / 2.0 / self.num_bins
        self.bins = [[] for _ in range(self.num_bins)]
        self.bins_evals = [[] for _ in range(self.num_bins)]

    @property
    def evaluations(self) -> List[np.ndarray]:
        """Returns the evaluations of the individuals in the buffer."""
        # flatten
        return [e for l in self.bins_evals for e in l]

    @property
    def individuals(self) -> list:
        """Returns the individuals in the buffer."""
        return [i for l in self.bins for i in l]

    def add(self, candidate, evaluation: np.ndarray):
        """Adds a candidate to the buffer.

        Args:
            candidate: candidate to add
            evaluation: evaluation of the candidate
        """

        def center_eval(eval):
            # Objectives must be positive
            return np.clip(eval + self.origin, 0.0, float("inf"))

        centered_eval = center_eval(evaluation)
        norm_eval = np.linalg.norm(centered_eval)
        theta = np.arccos(np.clip(centered_eval[1] / (norm_eval + 1e-3), -1.0, 1.0))
        buffer_id = int(theta // self.dtheta)

        if buffer_id < 0 or buffer_id >= self.num_bins:
            return

        inserted = False
        # Inserts the new candidate at the right position in the bin, in ascending order of the norm
        for idx, existing_eval in enumerate(self.bins_evals[buffer_id]):
            existing_norm = np.linalg.norm(center_eval(existing_eval))
            if norm_eval < existing_norm:
                self.bins[buffer_id].insert(idx, deepcopy(candidate))
                self.bins_evals[buffer_id].insert(idx, evaluation)
                inserted = True
                break

        # If bin is not full or candidate's norm is better than entire bin, append it
        if not inserted:
            self.bins[buffer_id].append(deepcopy(candidate))
            self.bins_evals[buffer_id].append(evaluation)

        # Remove worst (first) if bin exceeds max size
        if len(self.bins[buffer_id]) > self.max_size:
            self.bins[buffer_id].pop(0)
            self.bins_evals[buffer_id].pop(0)


class PerformanceBuffer3d:
    """Stores the population. Divides the objective space in to n bins of size max_size."""

    def __init__(self, num_bins: int, max_size: int, origin: np.ndarray):
        """Initializes the buffer.

        Args:
            num_bins: number of bins
            max_size: maximum size of each bin
            origin: origin of the objective space (to have only positive values)
        """
        self.max_size = max_size
        self.origin = -origin
        self.pbuffer_vec = generate_weights(1.0 / (num_bins - 1), 3)
        for i in range(len(self.pbuffer_vec)):
            self.pbuffer_vec[i] = self.pbuffer_vec[i] / np.linalg.norm(self.pbuffer_vec[i])
        self.num_bins = len(self.pbuffer_vec)
        self.bins = [[] for _ in range(self.num_bins)]
        self.bins_evals = [[] for _ in range(self.num_bins)]

    @property
    def evaluations(self) -> List[np.ndarray]:
        """Returns the evaluations of the individuals in the buffer."""
        # flatten
        return [e for l in self.bins_evals for e in l]

    @property
    def individuals(self) -> list:
        """Returns the individuals in the buffer."""
        return [i for l in self.bins for i in l]

    def add(self, candidate, evaluation: np.ndarray):
        """Adds a candidate to the buffer.

        Args:
            candidate: candidate to add
            evaluation: evaluation of the candidate
        """

        def center_eval(eval):
            # Objectives must be positive
            return np.clip(eval + self.origin, 0.0, float("inf"))

        centered_eval = center_eval(evaluation)
        norm_eval = np.linalg.norm(centered_eval)
        max_dot, buffer_id = -np.inf, -1
        for i in range(self.num_bins):
            dot = np.dot(self.pbuffer_vec[i], centered_eval)
            if dot > max_dot:
                max_dot, buffer_id = dot, i

        inserted = False
        # Inserts the new candidate at the right position in the bin, in ascending order of the norm
        for idx, existing_eval in enumerate(self.bins_evals[buffer_id]):
            existing_norm = np.linalg.norm(center_eval(existing_eval))
            if norm_eval < existing_norm:
                self.bins[buffer_id].insert(idx, deepcopy(candidate))
                self.bins_evals[buffer_id].insert(idx, evaluation)
                inserted = True
                break

        # If bin is not full or candidate's norm is better than entire bin, append it
        if not inserted:
            self.bins[buffer_id].append(deepcopy(candidate))
            self.bins_evals[buffer_id].append(evaluation)

        # Remove worst (first) if bin exceeds max size
        if len(self.bins[buffer_id]) > self.max_size:
            self.bins[buffer_id].pop(0)
            self.bins_evals[buffer_id].pop(0)


class PGMORL(MOAgent):
    """Prediction Guided Multi-Objective Reinforcement Learning.

    Reference: J. Xu, Y. Tian, P. Ma, D. Rus, S. Sueda, and W. Matusik,
    “Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control,”
    in Proceedings of the 37th International Conference on Machine Learning,
    Nov. 2020, pp. 10607–10616. Available: https://proceedings.mlr.press/v119/xu20h.html

    Paper: https://people.csail.mit.edu/jiex/papers/PGMORL/paper.pdf
    Supplementary materials: https://people.csail.mit.edu/jiex/papers/PGMORL/supp.pdf
    """

    def __init__(
        self,
        env_id: str,
        origin: np.ndarray,
        num_envs: int = 4,
        pop_size: int = 6,
        warmup_iterations: int = 80,
        steps_per_iteration: int = 2048,
        evolutionary_iterations: int = 20,
        num_weight_candidates: int = 7,
        num_performance_buffer: int = 100,
        performance_buffer_size: int = 2,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        delta_weight: float = 0.2,
        sparsity_coef: float = -1.0,
        env=None,
        gamma: float = 0.995,
        project_name: str = "MORL-baselines",
        experiment_name: str = "PGMORL",
        wandb_entity: Optional[str] = None,
        seed: Optional[int] = None,
        log: bool = True,
        net_arch: List = [64, 64],
        num_minibatches: int = 32,
        update_epochs: int = 10,
        learning_rate: float = 3e-4,
        anneal_lr: bool = False,
        clip_coef: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        clip_vloss: bool = True,
        max_grad_norm: float = 0.5,
        norm_adv: bool = True,
        target_kl: Optional[float] = None,
        gae: bool = True,
        gae_lambda: float = 0.95,
        device: Union[th.device, str] = "auto",
        group: Optional[str] = None,
    ):
        """Initializes the PGMORL agent.

        Args:
            env_id: environment id
            origin: reference point to make the objectives positive in the performance buffer
            num_envs: number of environments to use (VectorizedEnvs)
            pop_size: population size
            warmup_iterations: number of warmup iterations
            steps_per_iteration: number of steps per iteration
            evolutionary_iterations: number of evolutionary iterations
            num_weight_candidates: number of weight candidates
            num_performance_buffer: number of performance buffers
            performance_buffer_size: size of the performance buffers
            min_weight: minimum weight
            max_weight: maximum weight
            delta_weight: delta weight for weight generation
            sparsity_coef: sparsity coefficient (alpha in the paper)
            env: environment
            gamma: discount factor
            project_name: name of the project. Usually MORL-baselines.
            experiment_name: name of the experiment. Usually PGMORL.
            wandb_entity: wandb entity, defaults to None.
            seed: seed for the random number generator
            log: whether to log the results
            net_arch: number of units per layer
            num_minibatches: number of minibatches
            update_epochs: number of update epochs
            learning_rate: learning rate
            anneal_lr: whether to anneal the learning rate
            clip_coef: coefficient for the policy gradient clipping
            ent_coef: coefficient for the entropy term
            vf_coef: coefficient for the value function loss
            clip_vloss: whether to clip the value function loss
            max_grad_norm: maximum gradient norm
            norm_adv: whether to normalize the advantages
            target_kl: target KL divergence
            gae: whether to use generalized advantage estimation
            gae_lambda: lambda parameter for GAE
            device: device on which the code should run
            group: The wandb group to use for logging.
        """
        super().__init__(env, device=device, seed=seed)
        # Env dimensions
        self.tmp_env = mo_gym.make(env_id)
        self.extract_env_info(self.tmp_env)
        self.env_id = env_id
        self.num_envs = num_envs
        assert isinstance(self.action_space, gym.spaces.Box), "only continuous action space is supported"
        self.tmp_env.close()
        self.gamma = gamma

        # EA parameters
        self.pop_size = pop_size
        self.warmup_iterations = warmup_iterations
        self.steps_per_iteration = steps_per_iteration
        self.evolutionary_iterations = evolutionary_iterations
        self.num_weight_candidates = num_weight_candidates
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.delta_weight = delta_weight
        self.sparsity_coef = sparsity_coef
        self.num_performance_buffer = num_performance_buffer
        self.performance_buffer_size = performance_buffer_size
        self.archive = ParetoArchive()
        if self.reward_dim == 2:
            self.population = PerformanceBuffer2d(
                num_bins=self.num_performance_buffer,
                max_size=self.performance_buffer_size,
                origin=origin,
            )
        elif self.reward_dim == 3:
            self.population = PerformanceBuffer3d(
                num_bins=self.num_performance_buffer,
                max_size=self.performance_buffer_size,
                origin=origin,
            )
        else:
            raise ValueError("Only 2D and 3D objectives are supported.")
        self.predictor = PerformancePredictor()

        # PPO Parameters
        self.net_arch = net_arch
        self.batch_size = int(self.num_envs * self.steps_per_iteration)
        self.num_minibatches = num_minibatches
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.update_epochs = update_epochs
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.norm_adv = norm_adv
        self.target_kl = target_kl
        self.clip_vloss = clip_vloss
        self.gae_lambda = gae_lambda
        self.gae = gae

        # env setup
        if env is None:
            if self.seed is not None:
                envs = [make_env(env_id, self.seed + i, i, experiment_name, self.gamma) for i in range(self.num_envs)]
            else:
                envs = [make_env(env_id, i, i, experiment_name, self.gamma) for i in range(self.num_envs)]
            self.env = mo_gym.wrappers.vector.MOSyncVectorEnv(envs)
        else:
            raise ValueError("Environments should be vectorized for PPO. You should provide an environment id instead.")

        # Logging
        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity, group)

        self.networks = [
            MOPPONet(
                self.observation_shape,
                self.action_space.shape,
                self.reward_dim,
                self.net_arch,
            ).to(self.device)
            for _ in range(self.pop_size)
        ]

        weights = generate_weights(self.delta_weight, self.reward_dim)
        print(f"Warmup phase - sampled weights: {weights}")

        self.agents = [
            MOPPO(
                i,
                self.networks[i],
                weights[i],
                self.env,
                log=self.log,
                gamma=self.gamma,
                device=self.device,
                seed=self.seed,
                steps_per_iteration=self.steps_per_iteration,
                num_minibatches=self.num_minibatches,
                update_epochs=self.update_epochs,
                learning_rate=self.learning_rate,
                anneal_lr=self.anneal_lr,
                clip_coef=self.clip_coef,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                clip_vloss=self.clip_vloss,
                max_grad_norm=self.max_grad_norm,
                norm_adv=self.norm_adv,
                target_kl=self.target_kl,
                gae=self.gae,
                gae_lambda=self.gae_lambda,
                rng=self.np_random,
            )
            for i in range(self.pop_size)
        ]

    @override
    def get_config(self) -> dict:
        return {
            "env_id": self.env_id,
            "num_envs": self.num_envs,
            "pop_size": self.pop_size,
            "warmup_iterations": self.warmup_iterations,
            "evolutionary_iterations": self.evolutionary_iterations,
            "num_weight_candidates": self.num_weight_candidates,
            "num_performance_buffer": self.num_performance_buffer,
            "performance_buffer_size": self.performance_buffer_size,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "delta_weight": self.delta_weight,
            "sparsity_coef": self.sparsity_coef,
            "gamma": self.gamma,
            "seed": self.seed,
            "net_arch": self.net_arch,
            "batch_size": self.batch_size,
            "minibatch_size": self.minibatch_size,
            "update_epochs": self.update_epochs,
            "learning_rate": self.learning_rate,
            "anneal_lr": self.anneal_lr,
            "clip_coef": self.clip_coef,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "norm_adv": self.norm_adv,
            "target_kl": self.target_kl,
            "clip_vloss": self.clip_vloss,
            "gae": self.gae,
            "gae_lambda": self.gae_lambda,
        }

    def __train_all_agents(self, iteration: int, max_iterations: int):
        for i, agent in enumerate(self.agents):
            agent.global_step = self.global_step
            agent.train(self.start_time, iteration, max_iterations)
            self.global_step += self.steps_per_iteration * self.num_envs

    def __eval_all_agents(
        self,
        eval_env: gym.Env,
        evaluations_before_train: List[np.ndarray],
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        add_to_prediction: bool = True,
    ):
        """Evaluates all agents and store their current performances on the buffer and pareto archive."""
        for i, agent in enumerate(self.agents):
            _, _, _, discounted_reward = agent.policy_eval(eval_env, weights=agent.np_weights, log=self.log)
            # Storing current results
            self.population.add(agent, discounted_reward)
            self.archive.add(agent, discounted_reward)
            if add_to_prediction:
                self.predictor.add(
                    agent.weights.detach().cpu().numpy(),
                    evaluations_before_train[i],
                    discounted_reward,
                )
            evaluations_before_train[i] = discounted_reward

        if self.log:
            print("Current pareto archive:")
            print(self.archive.evaluations)
            log_all_multi_policy_metrics(
                current_front=self.archive.evaluations,
                hv_ref_point=ref_point,
                reward_dim=self.reward_dim,
                global_step=self.global_step,
                n_sample_weights=self.num_eval_weights_for_eval,
                ref_front=known_pareto_front,
            )

    def __task_weight_selection(self, ref_point: np.ndarray):
        """Chooses agents and weights to train at the next iteration based on the current population and prediction model."""
        candidate_weights = generate_weights(self.delta_weight / 2.0, self.reward_dim)  # Generates more weights than agents
        self.np_random.shuffle(candidate_weights)  # Randomize

        current_front = deepcopy(self.archive.evaluations)
        population = self.population.individuals
        population_eval = self.population.evaluations
        selected_tasks = []
        # For each worker, select a (policy, weight) tuple
        for i in range(len(self.agents)):
            max_improv = float("-inf")
            best_candidate = None
            best_eval = None
            best_predicted_eval = None

            # In each selection, look at every possible candidate in the current population and every possible weight generated
            for candidate, last_candidate_eval in zip(population, population_eval):
                # Pruning the already selected (candidate, weight) pairs
                candidate_tuples = [
                    (last_candidate_eval, weight)
                    for weight in candidate_weights
                    if (tuple(last_candidate_eval), tuple(weight)) not in selected_tasks
                ]

                # Prediction of improvements of each pair
                delta_predictions, predicted_evals = map(
                    list,
                    zip(
                        *[
                            self.predictor.predict_next_evaluation(weight, candidate_eval)
                            for candidate_eval, weight in candidate_tuples
                        ]
                    ),
                )
                # optimization criterion is a hypervolume - sparsity
                hypervolumes = [hypervolume(ref_point, current_front + [predicted_eval]) for predicted_eval in predicted_evals]
                sparsity_values = [sparsity(current_front + [predicted_eval]) for predicted_eval in predicted_evals]
                mixture_metrics = [
                    hv + self.sparsity_coef * sparsity_val for hv, sparsity_val in zip(hypervolumes, sparsity_values)
                ]

                if self.log:
                    wandb.log(
                        {
                            "metrics/hypervolume_improvement": np.mean(hypervolumes),
                            "metrics/sparsity_improvement": np.mean(sparsity_values),
                            "metrics/mixture_improvement": np.mean(mixture_metrics),
                            "global_step": self.global_step,
                        },
                    )
                # Best among all the weights for the current candidate
                current_candidate_weight = np.argmax(np.array(mixture_metrics))
                current_candidate_improv = np.max(np.array(mixture_metrics))

                # Best among all candidates, weight tuple update
                if max_improv < current_candidate_improv:
                    max_improv = current_candidate_improv
                    best_candidate = (
                        candidate,
                        candidate_tuples[current_candidate_weight][1],
                    )
                    best_eval = last_candidate_eval
                    best_predicted_eval = predicted_evals[current_candidate_weight]

            selected_tasks.append((tuple(best_eval), tuple(best_candidate[1])))
            # Append current estimate to the estimated front (to compute the next predictions)
            current_front.append(best_predicted_eval)

            # Assigns best predicted (weight-agent) pair to the worker
            copied_agent = deepcopy(best_candidate[0])
            copied_agent.global_step = self.agents[i].global_step
            copied_agent.id = i
            copied_agent.change_weights(deepcopy(best_candidate[1]))
            self.agents[i] = copied_agent

            print(f"Agent #{self.agents[i].id} - weights {best_candidate[1]}")
            print(
                f"current eval: {best_eval} - estimated next: {best_predicted_eval} - deltas {(best_predicted_eval - best_eval)}"
            )

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_eval: int = 50,
    ):
        """Trains the agents."""
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                }
            )
        self.num_eval_weights_for_eval = num_eval_weights_for_eval
        # 1 iteration is a full batch for each agents
        # -> (steps_per_iteration * num_envs * pop_size)  timesteps per iteration
        max_iterations = total_timesteps // self.steps_per_iteration // self.num_envs // self.pop_size
        iteration = 0
        # Init
        current_evaluations = [np.zeros(self.reward_dim) for _ in range(len(self.agents))]
        self.__eval_all_agents(
            eval_env=eval_env,
            evaluations_before_train=current_evaluations,
            ref_point=ref_point,
            known_pareto_front=known_pareto_front,
            add_to_prediction=False,
        )
        self.start_time = time.time()

        # Warmup
        for i in range(1, self.warmup_iterations + 1):
            print(f"Warmup iteration #{iteration}, global step: {self.global_step}")
            if self.log:
                wandb.log({"charts/warmup_iterations": i, "global_step": self.global_step})
            self.__train_all_agents(iteration=iteration, max_iterations=max_iterations)
            iteration += 1
        self.__eval_all_agents(
            eval_env=eval_env,
            evaluations_before_train=current_evaluations,
            ref_point=ref_point,
            known_pareto_front=known_pareto_front,
        )

        # Evolution
        max_iterations = max(max_iterations, self.warmup_iterations + self.evolutionary_iterations)
        evolutionary_generation = 1
        while iteration < max_iterations:
            # Every evolutionary iterations, change the task - weight assignments
            self.__task_weight_selection(ref_point=ref_point)
            print(f"Evolutionary generation #{evolutionary_generation}")
            if self.log:
                wandb.log(
                    {
                        "charts/evolutionary_generation": evolutionary_generation,
                        "global_step": self.global_step,
                    },
                )

            for _ in range(self.evolutionary_iterations):
                # Run training of every agent for evolutionary iterations.
                if self.log:
                    print(f"Evolutionary iteration #{iteration - self.warmup_iterations}")
                    wandb.log(
                        {
                            "charts/evolutionary_iterations": iteration - self.warmup_iterations,
                            "global_step": self.global_step,
                        },
                    )
                self.__train_all_agents(iteration=iteration, max_iterations=max_iterations)
                iteration += 1
            self.__eval_all_agents(
                eval_env=eval_env,
                evaluations_before_train=current_evaluations,
                ref_point=ref_point,
                known_pareto_front=known_pareto_front,
            )
            evolutionary_generation += 1

        print("Done training!")
        self.env.close()
        if self.log:
            self.close_wandb()
