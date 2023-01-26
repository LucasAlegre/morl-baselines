"""Pareto Q-Learning."""
from typing import Callable, Optional

import numpy as np

from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.pareto import get_non_dominated
from morl_baselines.common.performance_indicators import hypervolume


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
        epsilon_decay: float = 0.99,
        final_epsilon: float = 0.1,
        seed: int = None,
        project_name: str = "MORL-baselines",
        experiment_name: str = "Pareto Q-Learning",
        log: bool = True,
    ):
        """Initialize the Pareto Q-learning algorithm.

        Args:
            env: The environment.
            ref_point: The reference point for the hypervolume metric.
            gamma: The discount factor.
            initial_epsilon: The initial epsilon value.
            epsilon_decay: The epsilon decay rate.
            final_epsilon: The final epsilon value.
            seed: The random seed.
            project_name: The name of the project used for logging.
            experiment_name: The name of the experiment used for logging.
            log: Whether to log or not.
        """
        super().__init__(env)
        # Learning parameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Algorithm setup
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.ref_point = ref_point

        self.num_actions = self.env.action_space.n
        low_bound = self.env.observation_space.low
        high_bound = self.env.observation_space.high
        self.env_shape = (high_bound[0] - low_bound[0] + 1, high_bound[1] - low_bound[1] + 1)
        self.num_states = np.prod(self.env_shape)
        self.num_objectives = self.env.reward_space.shape[0]
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
            self.setup_wandb(project_name=self.project_name, experiment_name=self.experiment_name)

    def get_config(self) -> dict:
        """Get the configuration dictionary.

        Returns:
            Dict: A dictionary of parameters and values.
        """
        return {
            "ref_point": list(self.ref_point),
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay": self.epsilon_decay,
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
        if self.rng.uniform(0, 1) < self.epsilon:
            return self.rng.integers(self.num_actions)
        else:
            action_scores = score_func(state)
            return self.rng.choice(np.argwhere(action_scores == np.max(action_scores)).flatten())

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
        self, num_episodes: Optional[int] = 3000, log_every: Optional[int] = 100, action_eval: Optional[str] = "hypervolume"
    ):
        """Learn the Pareto front.

        Args:
            num_episodes (int, optional): The number of episodes to train for.
            log_every (int, optional): Log the results every number of episodes. (Default value = 100)
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

        for episode in range(num_episodes):
            if episode % log_every == 0:
                print(f"Training episode {episode + 1}")

            state, _ = self.env.reset()
            state = int(np.ravel_multi_index(state, self.env_shape))
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = self.select_action(state, score_func)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = int(np.ravel_multi_index(next_state, self.env_shape))

                self.counts[state, action] += 1
                self.non_dominated[state][action] = self.calc_non_dominated(next_state)
                self.avg_reward[state, action] += (reward - self.avg_reward[state, action]) / self.counts[state, action]
                state = next_state

            self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

            if self.log and episode % log_every == 0:
                pf = self.get_local_pcs(state=0)
                value = hypervolume(self.ref_point, list(pf))
                print(f"Hypervolume in episode {episode}: {value}")
                self.writer.add_scalar("train/hypervolume", value, episode)

        return self.get_local_pcs(state=0)

    def track_policy(self, vec):
        """Track a policy from its return vector.

        Args:
            vec (array_like): The return vector to track.
        """
        target = np.array(vec)
        state, _ = self.env.reset()
        terminated = False
        truncated = False
        total_rew = np.zeros(self.num_objectives)

        while not (terminated or truncated):
            state = np.ravel_multi_index(state, self.env_shape)
            new_target = False

            for action in range(self.num_actions):
                im_rew = self.avg_reward[state, action]
                non_dominated_set = self.non_dominated[state][action]
                for q in non_dominated_set:
                    q = np.array(q)
                    if np.all(self.gamma * q + im_rew == target):
                        state, reward, terminated, truncated, _ = self.env.step(action)
                        total_rew += reward
                        target = q
                        new_target = True
                        break

                if new_target:
                    break

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
