import random
from copy import deepcopy
from typing import List, Optional, Union, Tuple

import gym
import mo_gym
import numpy as np
import torch as th
from scipy.optimize import least_squares

from morl_baselines.common.performance_indicators import sparsity, hypervolume
from morl_baselines.common.morl_algorithm import MORLAlgorithm
from morl_baselines.common.pareto import ParetoArchive
from morl_baselines.mo_algorithms.mo_ppo import make_env, MOPPONet, MOPPOAgent


# Some code in this file has been adapted from the original code provided by the authors of the paper
# https://github.com/mit-gfx/PGMORL


class PerformancePredictor:
    def __init__(self, neighborhood_threshold: float = 0.1, sigma: float = 0.03, A_bound_min: float = 1.,
                 A_bound_max: float = 500., f_scale: float = 20.):
        """
        Stores the performance deltas along with the used weights after each generation.
        Then, uses these stored samples to perform a regression for predicting the performance of using a given weight
        to train a given policy.
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

    def add(self, weight: np.ndarray, eval_before_pg: np.ndarray, eval_after_pg: np.ndarray):
        self.previous_performance.append(eval_before_pg)
        self.next_performance.append(eval_after_pg)
        self.used_weight.append(weight)

    def __build_model_and_predict(self, training_weights, training_deltas, training_next_perfs, current_dim,
                                  current_eval: np.ndarray, weight_candidate: np.ndarray, sigma: float):
        """
        Uses the hyperbolic model on the training data: weights, deltas and next_perfs to predict the next delta
        given the current evaluation and weight.
        :return: The expected delta from current_eval by using weight_candidate.
        """

        def __f(x, A, a, b, c):
            return A * (np.exp(a * (x - b)) - 1) / (np.exp(a * (x - b)) + 1) + c

        def __hyperbolic_model(params, x, y):
            # f = A * (exp(a(x - b)) - 1) / (exp(a(x - b)) + 1) + c
            return (params[0] * (np.exp(params[1] * (x - params[2])) - 1.) / (np.exp(params[1] * (x - params[2])) + 1) +
                    params[3] - y) * w

        def __jacobian(params, x, y):
            A, a, b, c = params[0], params[1], params[2], params[3]
            J = np.zeros([len(params), len(x)])
            # df_dA = (exp(a(x - b)) - 1) / (exp(a(x - b)) + 1)
            J[0] = ((np.exp(a * (x - b)) - 1) / (np.exp(a * (x - b)) + 1)) * w
            # df_da = A(x - b)(2exp(a(x-b)))/(exp(a(x-b)) + 1)^2
            J[1] = (A * (x - b) * (2. * np.exp(a * (x - b))) / ((np.exp(a * (x - b)) + 1) ** 2)) * w
            # df_db = A(-a)(2exp(a(x-b)))/(exp(a(x-b)) + 1)^2
            J[2] = (A * (-a) * (2. * np.exp(a * (x - b))) / ((np.exp(a * (x - b)) + 1) ** 2)) * w
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
        res_robust = least_squares(__hyperbolic_model, initial_guess, loss='soft_l1', f_scale=self.f_scale,
                                   args=(train_x, train_y), jac=__jacobian,
                                   bounds=([0, 0.1, -5., -500.], [A_upperbound, 20., 5., 500.]))

        return __f(weight_candidate.T[current_dim], *res_robust.x)

    def predict_next_evaluation(self, weight_candidate: np.ndarray, policy_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use a part of the collected data (determined by the neighborhood threshold) to predict the performance
        after using weight to train the policy whose current evaluation is policy_eval.
        :param weight_candidate: weight candidate
        :param policy_eval: current evaluation of the policy
        :return: the delta prediction, along with the predicted next evaluations
        """
        neighbor_weights = []
        neighbor_deltas = []
        neighbor_next_perf = []
        current_sigma = self.sigma / 2.
        current_neighb_threshold = self.neighborhood_threshold / 2.
        # Iterates until we find at least 4 neighbors, enlarges the neighborhood at each iteration
        while len(neighbor_weights) < 4:
            if current_neighb_threshold >= 1.:
                break
            else:
                # Enlarging neighborhood
                current_sigma *= 2.
                current_neighb_threshold *= 2.
            # Filtering for neighbors
            for previous_perf, next_perf, w in zip(self.previous_performance, self.next_performance, self.used_weight):
                if np.all(np.abs(previous_perf - policy_eval) < current_neighb_threshold * np.abs(previous_perf)) and \
                        weight_candidate not in neighbor_weights:
                    neighbor_weights.append(weight_candidate)
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
                sigma=current_sigma
            ) for obj_num in range(weight_candidate.size)
        ]
        delta_predictions = np.array(delta_predictions).T
        return delta_predictions, delta_predictions + policy_eval


class PGMORL(MORLAlgorithm):
    """
    J. Xu, Y. Tian, P. Ma, D. Rus, S. Sueda, and W. Matusik,
    “Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control,”
    in Proceedings of the 37th International Conference on Machine Learning,
    Nov. 2020, pp. 10607–10616. Available: https://proceedings.mlr.press/v119/xu20h.html

    https://people.csail.mit.edu/jiex/papers/PGMORL/paper.pdf
    https://people.csail.mit.edu/jiex/papers/PGMORL/supp.pdf
    """

    def __init__(
            self,
            env_id: str = "mo-halfcheetah-v4",
            ref_point: np.ndarray = np.array([0., -5.]),
            num_envs: int = 4,
            pop_size: int = 6,
            warmup_iterations: int = 80,
            steps_per_iteration: int = 2048,
            limit_env_steps: int = int(5e6),
            evolutionary_iterations: int = 20,
            num_weight_candidates: int = 7,
            min_weight: float = 0.,
            max_weight: float = 1.,
            delta_weight: float = 0.2,
            env=None,
            gamma: float = 0.995,
            project_name: str = "PGMORL",
            experiment_name: str = "PGMORL",
            seed: int = 0,
            torch_deterministic: bool = True,
            log: bool = True,
            net_arch: List = [64, 64],
            num_minibatches: int = 32,
            update_epochs: int = 10,
            learning_rate: float = 3e-4,
            anneal_lr: bool = False,
            clip_coef: float = .2,
            ent_coef: float = 0.,
            vf_coef: float = .5,
            clip_vloss: bool = True,
            max_grad_norm: float = .5,
            norm_adv: bool = True,
            target_kl: Optional[float] = None,
            gae: bool = True,
            gae_lambda: float = .95,
            device: Union[th.device, str] = "auto",
    ):
        super().__init__(env, device)
        # Env dimensions
        self.tmp_env = mo_gym.make(env_id)
        self.extract_env_info(self.tmp_env)
        self.env_id = env_id
        self.num_envs = num_envs
        assert isinstance(self.action_space, gym.spaces.Box), "only continuous action space is supported"
        self.tmp_env.close()
        self.gamma = gamma
        self.ref_point = ref_point

        # EA parameters
        self.pop_size = pop_size
        self.warmup_iterations = warmup_iterations
        self.steps_per_iteration = steps_per_iteration
        self.evolutionary_iterations = evolutionary_iterations
        self.num_weight_candidates = num_weight_candidates
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.delta_weight = delta_weight
        self.limit_env_steps = limit_env_steps
        self.max_iterations = self.limit_env_steps // self.steps_per_iteration // self.num_envs
        self.iteration = 0
        self.pareto_archive = ParetoArchive()
        self.predictor = PerformancePredictor()

        # PPO Parameters
        self.net_arch = net_arch
        self.batch_size = int(self.num_envs * self.steps_per_iteration)
        self.minibatch_size = int(self.batch_size // num_minibatches)
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

        # seeding
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        th.manual_seed(self.seed)
        th.backends.cudnn.deterministic = torch_deterministic

        # env setup
        self.num_envs = num_envs
        if env is None:
            self.env = mo_gym.MOSyncVectorEnv(
                [make_env(env_id, self.seed + i, i, experiment_name, self.gamma) for i in range(self.num_envs)]
            )
        else:
            raise ValueError("Environments should be vectorized for PPO. You should provide an environment id instead.")

        # Logging
        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name)

        self.networks = [
            MOPPONet(self.observation_shape, self.action_space.shape, self.reward_dim, self.net_arch).to(self.device)
            for _ in range(self.pop_size)
        ]

        weights = self.generate_weights(self.delta_weight)
        print(f"Warmup phase - sampled weights: {weights}")
        self.pop_size = len(weights)

        self.agents = [
            MOPPOAgent(i, self.networks[i], weights[i], self.env, self.writer, gamma=self.gamma, device=self.device)
            for i in range(self.pop_size)
        ]

    def eval(self, obs):
        pass

    def generate_weights(self, delta_weight: float) -> np.ndarray:
        """
        Generates weights uniformly distributed over the objective dimensions. These weight vectors are separated by
        delta_weight distance.
        :param delta_weight: distance between weight vectors
        :return: all the candidate weights
        """
        return np.linspace((0., 1.), (1., 0.), int(1/delta_weight) + 1, dtype=np.float32)

    def get_config(self) -> dict:
        return {
            "env_id": self.env_id,
            "ref_point": self.ref_point,
            "num_envs": self.num_envs,
            "pop_size": self.pop_size,
            "warmup_iterations": self.warmup_iterations,
            "evolutionary_iterations": self.evolutionary_iterations,
            "steps_per_iteration": self.steps_per_iteration,
            "limit_env_steps": self.limit_env_steps,
            "max_iterations": self.max_iterations,
            "num_weight_candidates": self.num_weight_candidates,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "delta_weight": self.delta_weight,
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

    def __eval_agent(self, agent, render: bool = False):
        scalarized, discounted_scalarized, reward, discounted_reward = mo_gym.eval_mo(agent=agent, env=self.env.envs[0],
                                                                                      w=agent.weights, render=render)
        print(f"Agent #{agent.id} - {reward} - discounted: {discounted_reward}")
        for i, (r, dr) in enumerate(zip(reward, discounted_reward)):
            self.writer.add_scalar(f"charts_{agent.id}/evaluated_reward_{i}", r, self.iteration)
            self.writer.add_scalar(f"charts_{agent.id}/evaluated_discounted_reward_{i}", dr, self.iteration)
        self.writer.add_scalar(f"charts_{agent.id}/scalarized_discounted_reward", scalarized, self.iteration)
        self.writer.add_scalar(f"charts_{agent.id}/scalarized_reward_", discounted_scalarized, self.iteration)
        self.pareto_archive.add(agent, discounted_reward)
        return discounted_reward

    def __train_all_agents(self, evaluations_before_train):
        self.writer.add_scalar("charts/iteration", self.iteration)
        for i, agent in enumerate(self.agents):
            agent.train(self.iteration, self.max_iterations)
            evaluation_after_train = self.__eval_agent(agent)
            self.predictor.add(agent.weights.detach().numpy(), evaluations_before_train[i], evaluation_after_train)
            evaluations_before_train[i] = evaluation_after_train
        print("Current pareto archive:")
        print(self.pareto_archive.evaluations)
        hv = hypervolume(self.ref_point, self.pareto_archive.evaluations)
        sp = sparsity(self.pareto_archive.evaluations)
        self.writer.add_scalar("charts/hypervolume", hv, self.iteration)
        self.writer.add_scalar("charts/sparsity", sp, self.iteration)

    def __update_weights(self, current_evals: List[np.ndarray]):
        """
        Update the weights of each agent based on prediction of PF improvement given current_evals and their current weights.
        :param current_evals: current evaluations of the agents' policies
        """
        candidate_weights = self.generate_weights(self.delta_weight / 2.)  # Generates more weights than agents
        np.random.shuffle(candidate_weights)  # Randomize

        current_front = deepcopy(self.pareto_archive.evaluations)
        for i, a in enumerate(self.agents):
            delta_predictions, predicted_evals = \
                map(list, zip(*[self.predictor.predict_next_evaluation(weight, current_evals[i]) for weight in candidate_weights]))
            print(f"Agent #{a.id} - Delta predictions:")
            print(delta_predictions)
            mixture_metrics = [
                hypervolume(self.ref_point, current_front + predicted_eval) - sparsity(current_front + predicted_eval)
                for predicted_eval in predicted_evals
            ]
            best_candidate = np.argmax(np.array(mixture_metrics))
            # Assigns best predicted weights to the agent
            a.change_weights(deepcopy(candidate_weights[best_candidate]))
            # Append current estimate to the estimated front (for computing the next predictions)
            current_front.append(predicted_evals[best_candidate])

    def train(self):
        # Warmup
        current_evaluations = [np.zeros(self.reward_dim) for _ in range(len(self.agents))]
        for i in range(self.warmup_iterations):
            self.writer.add_scalar("charts/warmup_iterations", i)
            print(f"Warmup iteration #{self.iteration}")
            self.__train_all_agents(current_evaluations)
            self.iteration += 1

        # Evolution
        remaining_iterations = max(self.max_iterations - self.warmup_iterations, self.evolutionary_iterations)
        for generation in range(remaining_iterations):
            print(f"Evolutionary iteration #{generation}")
            self.writer.add_scalar("charts/evolutionary_iterations", generation)
            self.__update_weights(current_evaluations)
            weights = [agent.weights for agent in self.agents]
            print(f"Current weights: {weights}")
            self.__train_all_agents(current_evaluations)
            self.iteration += 1

        print("Done training!")
        self.env.close()
        self.close_wandb()
