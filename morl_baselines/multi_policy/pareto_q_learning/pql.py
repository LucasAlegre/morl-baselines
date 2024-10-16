"""Pareto Q-Learning."""

import numbers
from typing import Callable, List, Optional

import gymnasium as gym
import numpy as np
import wandb

from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.pareto import get_non_dominated
from morl_baselines.common.performance_indicators import hypervolume
from morl_baselines.common.utils import linearly_decaying_value


class PQL(MOAgent):
    """Pareto Q-learning.

    Tabular method relying on pareto pruning.
    Paper: K. Van Moffaert and A. Nowé, “Multi-objective reinforcement learning using sets of pareto dominating policies,” The Journal of Machine Learning Research, vol. 15, no. 1, pp. 3483–3512, 2014.
    """

    def __init__(
        self,
        env,
        ref_point: np.ndarray,
        gamma: float = 0.8,
        initial_epsilon: float = 1.0,
        epsilon_decay_steps: int = 100000,
        final_epsilon: float = 0.1,
        seed: Optional[int] = None,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "Pareto Q-Learning",
        wandb_entity: Optional[str] = None,
        log: bool = True,
    ):
        """Initialize the Pareto Q-learning algorithm.

        Args:
            env: The environment.
            ref_point: The reference point for the hypervolume metric.
            gamma: The discount factor.
            initial_epsilon: The initial epsilon value.
            epsilon_decay_steps: The number of steps to decay epsilon.
            final_epsilon: The final epsilon value.
            seed: The random seed.
            project_name: The name of the project used for logging.
            experiment_name: The name of the experiment used for logging.
            wandb_entity: The wandb entity used for logging.
            log: Whether to log or not.
        """
        super().__init__(env, seed=seed)
        # Learning parameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon

        # Algorithm setup
        self.ref_point = ref_point

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.num_actions = self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            self.num_actions = np.prod(self.env.action_space.nvec)
        else:
            raise Exception("PQL only supports (multi)discrete action spaces.")

        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.env_shape = (self.env.observation_space.n,)
        elif isinstance(self.env.observation_space, gym.spaces.MultiDiscrete):
            self.env_shape = self.env.observation_space.nvec
        elif (
            isinstance(self.env.observation_space, gym.spaces.Box)
            and self.env.observation_space.is_bounded(manner="both")
            and issubclass(self.env.observation_space.dtype.type, numbers.Integral)
        ):
            low_bound = np.array(self.env.observation_space.low)
            high_bound = np.array(self.env.observation_space.high)
            self.env_shape = high_bound - low_bound + 1
        else:
            raise Exception("PQL only supports discretizable observation spaces.")

        self.num_states = np.prod(self.env_shape)
        self.num_objectives = self.env.unwrapped.reward_space.shape[0]
        self.counts = np.zeros((self.num_states, self.num_actions))
        self.non_dominated = [
            [{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)] for _ in range(self.num_states)
        ]
        self.avg_reward = np.zeros((self.num_states, self.num_actions, self.num_objectives))

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log

        if self.log:
            self.setup_wandb(
                project_name=self.project_name,
                experiment_name=self.experiment_name,
                entity=wandb_entity,
            )

    def get_config(self) -> dict:
        """Get the configuration dictionary.

        Returns:
            Dict: A dictionary of parameters and values.
        """
        return {
            "env_id": self.env.unwrapped.spec.id,
            "ref_point": list(self.ref_point),
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "final_epsilon": self.final_epsilon,
            "seed": self.seed,
        }

    def score_pareto_cardinality(self, state: int):
        """Compute the action scores based upon the Pareto cardinality metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        non_dominated = get_non_dominated(candidates)
        scores = np.zeros(self.num_actions)

        for vec in non_dominated:
            for action, q_set in enumerate(q_sets):
                if vec in q_set:
                    scores[action] += 1

        return scores

    def score_hypervolume(self, state: int):
        """Compute the action scores based upon the hypervolume metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        action_scores = [hypervolume(self.ref_point, list(q_set)) for q_set in q_sets]
        return action_scores

    def get_q_set(self, state: int, action: int):
        """Compute the Q-set for a given state-action pair.

        Args:
            state (int): The current state.
            action (int): The action.

        Returns:
            A set of Q vectors.
        """
        nd_array = np.array(list(self.non_dominated[state][action]))
        q_array = self.avg_reward[state, action] + self.gamma * nd_array
        return {tuple(vec) for vec in q_array}

    def select_action(self, state: int, score_func: Callable):
        """Select an action in the current state.

        Args:
            state (int): The current state.
            score_func (callable): A function that returns a score per action.

        Returns:
            int: The selected action.
        """
        if self.np_random.uniform(0, 1) < self.epsilon:
            return self.np_random.integers(self.num_actions)
        else:
            action_scores = score_func(state)
            return self.np_random.choice(np.argwhere(action_scores == np.max(action_scores)).flatten())

    def calc_non_dominated(self, state: int):
        """Get the non-dominated vectors in a given state.

        Args:
            state (int): The current state.

        Returns:
            Set: A set of Pareto non-dominated vectors.
        """
        candidates = set().union(*[self.get_q_set(state, action) for action in range(self.num_actions)])
        non_dominated = get_non_dominated(candidates)
        return non_dominated

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: Optional[np.ndarray] = None,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_eval: int = 50,
        log_every: Optional[int] = 10000,
        action_eval: Optional[str] = "hypervolume",
    ):
        """Learn the Pareto front.

        Args:
            total_timesteps (int, optional): The number of episodes to train for.
            eval_env (gym.Env): The environment to evaluate the policies on.
            eval_ref_point (ndarray, optional): The reference point for the hypervolume metric during evaluation. If none, use the same ref point as training.
            known_pareto_front (List[ndarray], optional): The optimal Pareto front, if known.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            log_every (int, optional): Log the results every number of timesteps. (Default value = 1000)
            action_eval (str, optional): The action evaluation function name. (Default value = 'hypervolume')

        Returns:
            Set: The final Pareto front.
        """
        if action_eval == "hypervolume":
            score_func = self.score_hypervolume
        elif action_eval == "pareto_cardinality":
            score_func = self.score_pareto_cardinality
        else:
            raise Exception("No other method implemented yet")
        if ref_point is None:
            ref_point = self.ref_point
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "log_every": log_every,
                    "action_eval": action_eval,
                }
            )

        while self.global_step < total_timesteps:
            state, _ = self.env.reset()
            state = int(np.ravel_multi_index(state, self.env_shape))
            terminated = False
            truncated = False

            while not (terminated or truncated) and self.global_step < total_timesteps:
                action = self.select_action(state, score_func)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.global_step += 1
                next_state = int(np.ravel_multi_index(next_state, self.env_shape))

                self.counts[state, action] += 1
                self.non_dominated[state][action] = self.calc_non_dominated(next_state)
                self.avg_reward[state, action] += (reward - self.avg_reward[state, action]) / self.counts[state, action]
                state = next_state

                if self.log and self.global_step % log_every == 0:
                    wandb.log({"global_step": self.global_step})
                    pf = self._eval_all_policies(eval_env)
                    log_all_multi_policy_metrics(
                        current_front=pf,
                        hv_ref_point=ref_point,
                        reward_dim=self.reward_dim,
                        global_step=self.global_step,
                        n_sample_weights=num_eval_weights_for_eval,
                        ref_front=known_pareto_front,
                    )

            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                0,
                self.final_epsilon,
            )

        return self.get_local_pcs(state=0)

    def _eval_all_policies(self, env: gym.Env) -> List[np.ndarray]:
        """Evaluate all learned policies by tracking them."""
        pf = []
        for vec in self.get_local_pcs(state=0):
            pf.append(self.track_policy(vec, env))

        return pf

    def track_policy(self, vec, env: gym.Env, tol=1e-3):
        """Track a policy from its return vector.

        Args:
            vec (array_like): The return vector to track.
            env (gym.Env): The environment to track the policy in.
            tol (float, optional): The tolerance for the return vector. (Default value = 1e-3)
        """
        target = np.array(vec)
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_rew = np.zeros(self.num_objectives)
        current_gamma = 1.0

        while not (terminated or truncated):
            state = np.ravel_multi_index(state, self.env_shape)
            closest_dist = np.inf
            closest_action = 0
            found_action = False
            new_target = target

            for action in range(self.num_actions):
                im_rew = self.avg_reward[state, action]
                non_dominated_set = self.non_dominated[state][action]

                for q in non_dominated_set:
                    q = np.array(q)
                    dist = np.sum(np.abs(self.gamma * q + im_rew - target))
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_action = action
                        new_target = q

                        if dist < tol:
                            found_action = True
                            break

                if found_action:
                    break

            state, reward, terminated, truncated, _ = env.step(closest_action)
            total_rew += current_gamma * reward
            current_gamma *= self.gamma
            target = new_target

        return total_rew

    def get_local_pcs(self, state: int = 0):
        """Collect the local PCS in a given state.

        Args:
            state (int): The state to get a local PCS for. (Default value = 0)

        Returns:
            Set: A set of Pareto optimal vectors.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        return get_non_dominated(candidates)
