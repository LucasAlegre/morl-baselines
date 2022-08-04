import itertools
from typing import Callable, List, Optional

import gym
import numpy as np

from morl_baselines.common.morl_algorithm import MORLAlgorithm


def get_non_dominated(candidates):
    """
    This function returns the non-dominated subset of elements.
    :param candidates: The input set of candidate vectors.
    :return: The non-dominated subset of this input set.
    Source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    The code provided in all the stackoverflow answers is wrong. Important changes have been made in this function.
    """
    candidates = np.array(list(candidates))  # Turn the input set into a numpy array.
    candidates = candidates[candidates.sum(1).argsort()[::-1]]  # Sort candidates by decreasing sum of coordinates.
    for i in range(candidates.shape[0]):  # Process each point in turn.
        n = candidates.shape[0]  # Check current size of the candidates.
        if i >= n:  # If we've eliminated everything up until this size we stop.
            break
        nd = np.ones(candidates.shape[0], dtype=bool)  # Initialize a boolean mask for undominated points.
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        nd[i + 1:] = np.any(candidates[i + 1:] > candidates[i], axis=1)
        candidates = candidates[nd]  # Grab only the non-dominated vectors using the generated bitmask.

    non_dominated = set()
    for candidate in candidates:
        non_dominated.add(tuple(candidate))  # Add the non dominated vectors to a set again.
    return non_dominated


def get_best(candidates, max_points=10):
    """
    This function gets the best points from the candidate set.
    :param candidates: The set of candidates.
    :param max_points: The maximum number of points in the final set.
    :return: A non dominated set that is potentially further pruned using crowding distance.
    """
    non_dominated = get_non_dominated(candidates)  # Get the non dominated points.

    if max_points is None:  # If we want to keep everything return the non-dominated vectors already.
        return non_dominated

    points_to_remove = len(non_dominated) - max_points  # Calculate the number of points left to remove.

    if points_to_remove > 0:  # If we still need to discard points.
        nd_array = np.array(list(non_dominated))  # Transform the set to an array.
        crowding_distances = crowding_distance_assignment(nd_array)  # Calculate the crowding distances.
        max_ind = np.argsort(crowding_distances)[points_to_remove:]  # Get the indices of the best points.
        best_points = nd_array[max_ind]  # Select the best points using these indices.

        best_set = set()  # Place everything back into a set.
        for point in best_points:
            best_set.add(tuple(point))  # Add the non dominated vectors to a set again.
        return best_set
    else:
        return non_dominated


def crowding_distance_assignment(nd_array):
    """
    This function calculates the crowding distance for each point in the set.
    :param nd_array: The non-dominated set as an array.
    :return: The crowding distances.
    """
    size = nd_array.shape[0]
    num_objectives = nd_array.shape[1]
    crowding_distances = np.zeros(size)

    sorted_ind = np.argsort(nd_array, axis=0)  # The indexes of each column sorted.
    maxima = np.max(nd_array, axis=0)  # The maxima of each objective.
    minima = np.min(nd_array, axis=0)  # The minima of each objective.

    for obj in range(num_objectives):  # Loop over all objectives.
        crowding_distances[sorted_ind[0, obj]] = np.inf  # Always include the outer points.
        crowding_distances[sorted_ind[-1, obj]] = np.inf
        norm_factor = maxima[obj] - minima[obj]

        for i in range(1, size - 1):  # Loop over all other points.
            distance = nd_array[sorted_ind[i + 1, obj], obj] - nd_array[sorted_ind[i - 1, obj], obj]
            crowding_distances[sorted_ind[i, obj]] += distance / norm_factor

    return crowding_distances


class ParetoQ(MORLAlgorithm):
    """
    An implementation for a pareto Q learning agent that is able to deal with stochastic environments.
    """

    def __init__(self,
                 env: Optional[gym.Env],
                 perf_indic: Callable,
                 gamma: float = 0.8,
                 init_epsilon: float = 1.,
                 epsilon_decay: float = 0.99,
                 decay_every: int = 10,
                 min_epsilon: float = 0.2,
                 decimals: int = 2,
                 novec: int = 30) -> None:
        super().__init__(env)

        self.env = env
        self.perf_indic = perf_indic
        try:
            self.num_actions = env.action_space.n
            self.num_states = 121  # env.observation_space.shape
            self.num_objectives = env.reward_space.shape[0]
        except Exception:
            raise Exception('Pareto Q-learning is only supported on the deep sea treasure environment.')

        self.gamma = gamma
        self.init_epsilon = init_epsilon
        self.epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay
        self.decay_every = decay_every
        self.min_epsilon = min_epsilon
        self.decimals = decimals
        self.novec = novec

        # Implemented as recommended by Van Moffaert et al. by substituting (s, a) with (s, a, s').
        self.non_dominated = [
            [[{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_states)] for _ in range(self.num_actions)]
            for _ in range(self.num_states)]
        self.avg_r = np.zeros((self.num_states, self.num_actions, self.num_states, self.num_objectives))
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))

    def calc_q_set(self, state, action):
        """Calculate a Q-set for the state-action pair.

        Args:
            state (int): A state.
            action (int): An action.

        Returns:
            Set: A set of vectorial Q-values for the state action pair.
        """
        q_set = set()

        transition_probs = self.transitions[state, action] / max(1, np.sum(self.transitions[state, action]))
        next_states = np.where(self.transitions[state, action, :] > 0)[0]  # Next states with prob > 0

        next_sets = []
        for next_state in next_states:
            next_sets.append(list(self.non_dominated[state][action][next_state]))

        cartesian_product = itertools.product(*next_sets)

        for next_vectors in cartesian_product:
            expected_vec = np.zeros(self.num_objectives)
            for idx, next_vector in enumerate(next_vectors):
                next_state = next_states[idx]
                transition_prob = transition_probs[next_state]
                disc_future_reward = self.gamma * np.array(next_vector)
                expected_vec += transition_prob * (self.avg_r[state, action, next_state] + disc_future_reward)
            expected_vec = tuple(np.around(expected_vec, decimals=self.decimals))  # Round the future reward.
            q_set.add(tuple(expected_vec))
        return q_set

    def select_action(self, state):
        """Select an action using epsilon greedy on the performance metric.

        Args:
            state (int): The current state.

        Returns:
            int: The selected action.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            values = []
            for action in range(self.num_actions):
                q_set = self.calc_q_set(state, action)
                q_set = get_non_dominated(q_set)
                value = self.perf_indic(q_set)
                values.append(value)
            action = np.random.choice(np.argwhere(values == np.max(values)).flatten())
            return action

    def update(self, state, action, next_state, r):
        """Perform an update of the learner.

        Args:
            state (int): The previous state.
            action (int): The action that was taken.
            next_state (int): The next state.
            r (ndarray): The reward vector for this transition.

        Returns:
            None
        """
        self.transitions[state, action, next_state] += 1
        q_sets = []
        for a in range(self.num_actions):
            q_sets.append(self.calc_q_set(next_state, a))
        self.non_dominated[state][action][next_state] = get_best(set().union(*q_sets), max_points=self.novec)
        self.avg_r[state, action, next_state] += (r - self.avg_r[state, action, next_state]) / self.transitions[
            state, action, next_state]

    def construct_pcs(self):
        """Construct the Pareto Coverage Set.

        Returns:
            List[List[Set]]: A set of undominated rewards per state-action pair.
        """
        pcs = [[{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)] for _ in
               range(self.num_states)]
        for state in range(self.num_states):
            for action in range(self.num_actions):
                pcs[state][action] = get_non_dominated(self.calc_q_set(state, action))
        return pcs

    def construct_pf(self):
        """Construct the Pareto front.

        Returns:
            Set: A set of rewards which are on the Pareto front.
        """
        q_sets = []
        for action in range(self.num_actions):
            q_set = self.calc_q_set(0, action)
            q_sets.append(q_set)
        pareto_front = set().union(*q_sets)
        pareto_front = get_non_dominated(pareto_front)
        return pareto_front

    def get_config(self):
        """Get the config parameters for this agent.

        Returns:
            Dict: A dictionary of configuration parameters.
        """
        return {
            "env_id": self.env.unwrapped.spec.id,
            "init_epsilon": self.init_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "decay_every": self.decay_every,
            "min_epsilon": self.min_epsilon,
            "gamma:": self.gamma,
            "decimals": self.decimals,
            "novec": self.novec,
        }

    def flatten_observation(self, obs):
        return int(np.ravel_multi_index(obs, (11, 11)))

    def train(self,
              iterations: int = 3000,
              max_timesteps: int = 1000,
              log: bool = False,
              log_every: int = 1,
              project_name: str = "PQL",
              experiment_name: str = "PQL"):
        """Train the agent.

        Args:
            iterations (int, optional): The number of iterations to execute. (Default value = 3000)
            max_timesteps (int, optional): The maximum timesteps for each iteration. (Default value = 100)
            log (bool, optional): Whether to log the results or not. (Default value = True)
            log_every (int, optional): Log the performance metric every number of iterations. (Default value = 1)
            project_name (str, optional): The name for the wandb project. (Default value = "PQL")
            experiment_name (str, optional): The name for the wandb experiment. (Default value = "PQL")

        Returns:
            /
        """

        if log:
            self.setup_wandb(project_name, experiment_name)

        for i in range(iterations):
            state = self.flatten_observation(self.env.reset())
            done = False
            timestep = 0

            while not done and timestep < max_timesteps:
                action = self.select_action(state)
                next_state, r, done, prob = self.env.step(action)
                next_state = self.flatten_observation(next_state)
                self.update(state, action, next_state, r)
                state = next_state
                timestep += 1

            if log and i % log == 0:
                pf = self.construct_pf()
                value = self.perf_indic(pf)
                self.writer.add_scalar("train/hypervolume", value, i)

            if i % self.decay_every == 0:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        if log:
            self.close_wandb()

    def eval(self, obs):
        pass
