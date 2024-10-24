"""Scalarized Q-learning for single policy multi-objective reinforcement learning."""

import time
from typing import Optional
from typing_extensions import override

import gymnasium as gym
import numpy as np
import wandb

from morl_baselines.common.evaluation import log_episode_info
from morl_baselines.common.model_based.tabular_model import TabularModel
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.scalarization import weighted_sum
from morl_baselines.common.utils import linearly_decaying_value


class MOQLearning(MOPolicy, MOAgent):
    """Scalarized Q learning for single policy multi-objective reinforcement learning.

    Maintains one Q-table per objective, rely on a scalarization function to choose the moves.
    Paper: K. Van Moffaert, M. Drugan, and A. Nowe, Scalarized Multi-Objective Reinforcement Learning: Novel Design Techniques. 2013. doi: 10.1109/ADPRL.2013.6615007.
    """

    def __init__(
        self,
        env,
        id: Optional[int] = None,
        weights: np.ndarray = np.array([0.5, 0.5]),
        scalarization=weighted_sum,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        initial_epsilon: float = 0.1,
        final_epsilon: float = 0.1,
        epsilon_decay_steps: int = None,
        learning_starts: int = 0,
        use_gpi_policy: bool = False,
        dyna: bool = False,
        dyna_updates: int = 5,
        model: Optional[TabularModel] = None,
        gpi_pd: bool = False,
        min_priority: float = 0.0001,
        alpha: float = 0.6,
        parent=None,
        project_name: str = "MORL-baselines",
        experiment_name: str = "MO Q-Learning",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
        parent_rng: Optional[np.random.Generator] = None,
    ):
        """Initializes the MOQ-learning algorithm.

        Args:
            env: The environment to train on.
            id: The id of the policy.
            weights: The weights to use for the scalarization function.
            scalarization: The scalarization function to use.
            learning_rate: The learning rate.
            gamma: The discount factor.
            initial_epsilon: The initial epsilon value.
            final_epsilon: The final epsilon value.
            epsilon_decay_steps: The number of steps to decay epsilon over.
            learning_starts: The number of steps to wait before starting to learn.
            use_gpi_policy: Whether to use Generalized Policy Improvement (GPI) or not.
            dyna: Whether to use Dyna-Q or not.
            dyna_updates: The number of Dyna-Q updates to perform each step.
            model: The model to use for Dyna. If None and dyna==True, a new one is created.
            gpi_pd: Whether to use the GPI-PD method to prioritize Dyna updates.
            min_priority: The minimum priority to use for GPI-PD.
            alpha: The alpha value to use to smooth GPI-PD priorities.
            parent: The parent MPMOQLearning class in the case of multi-policy training.
            project_name: The name of the project used for logging.
            experiment_name: The name of the experiment used for logging.
            wandb_entity: The entity to use for logging.
            log: Whether to log or not.
            seed: The seed to use for the experiment.
            parent_rng: The random number generator to use. If None, a new one is created.
        """
        MOAgent.__init__(self, env)
        MOPolicy.__init__(self, id)
        self.learning_rate = learning_rate
        self.id = id
        self.seed = seed
        if parent_rng is not None:
            self.np_random = parent_rng
        else:
            self.np_random = np.random.default_rng(self.seed)

        if self.id is not None:
            self.idstr = f"_{self.id}"
        else:
            self.idstr = ""
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.learning_starts = learning_starts
        self.use_gpi_policy = use_gpi_policy
        self.dyna = dyna
        self.dyna_updates = dyna_updates
        self.gpi_pd = gpi_pd
        self.min_priority = min_priority
        self.alpha = alpha
        self.parent = parent

        self.weights = weights
        self.scalarization = scalarization

        self.q_table = dict()

        if model is not None:
            self.model = model
        else:
            self.model = TabularModel(prioritize=self.gpi_pd) if self.dyna else None

        self.log = log
        if self.log and parent_rng is None:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def __act(self, obs: np.array) -> int:
        # epsilon-greedy
        coin = self.np_random.random()
        if coin < self.epsilon:
            return int(self.env.action_space.sample())
        else:
            return self.eval(obs, self.weights)

    def scalarized_q_values(self, obs, w: np.ndarray) -> np.ndarray:
        """Returns the scalarized Q values for each action, given observation and weights."""
        t_obs = tuple(obs)
        if t_obs not in self.q_table:
            return np.zeros(self.action_dim)
        return np.array([self.scalarization(state_action_value, w) for state_action_value in self.q_table[t_obs]])

    def _gpi_pd_priority(
        self, obs: np.ndarray, action: int, reward: np.ndarray, next_obs: np.ndarray, terminal: bool, weights: np.ndarray
    ) -> float:
        """Computes the priority of GPI-PD for a given transition.

        priority = |r.w + gamma * max_a' max_pi' Q^pi'(s', a').w - Q^pi(s, a).w|
        """
        priority = (
            np.dot(reward, weights)
            + (1 - terminal) * self.gamma * self.parent.max_scalar_q_value(next_obs, weights)
            - np.dot(self.q_table[tuple(obs)][action], weights)
        )
        priority = max(np.abs(priority), self.min_priority) ** self.alpha
        return priority

    @override
    def eval(self, obs: np.array, w: Optional[np.ndarray] = None) -> int:
        if self.use_gpi_policy:
            return self.parent.eval(obs, w)
        """Greedily chooses best action using the scalarization method"""
        t_obs = tuple(obs)
        if t_obs not in self.q_table:
            return int(self.env.action_space.sample())
        scalarized = np.array(
            [self.scalarization(state_action_value, self.weights) for state_action_value in self.q_table[t_obs]]
        )
        return int(np.argmax(scalarized))

    @override
    def update(self):
        """Updates the Q table."""
        obs = tuple(self.obs)
        next_obs = tuple(self.next_obs)
        if obs not in self.q_table:
            self.q_table[obs] = np.zeros((self.action_dim, self.reward_dim))
        if next_obs not in self.q_table:
            self.q_table[next_obs] = np.zeros((self.action_dim, self.reward_dim))

        max_q = self.q_table[next_obs][self.eval(self.next_obs, self.weights)]
        td_error = self.reward + (1 - self.terminated) * self.gamma * max_q - self.q_table[obs][self.action]
        self.q_table[obs][self.action] += self.learning_rate * td_error

        # Dyna updates
        if self.dyna:
            if self.gpi_pd:
                priority = self._gpi_pd_priority(obs, self.action, self.reward, next_obs, self.terminated, self.weights)
            else:
                priority = None

            self.model.update(obs, self.action, self.reward, next_obs, self.terminated, priority)
            for _ in range(self.dyna_updates):
                if self.gpi_pd:
                    s, a, r, next_s, terminal, ind = self.model.random_transition()
                else:
                    s, a, r, next_s, terminal = self.model.random_transition()
                if s not in self.q_table:
                    self.q_table[s] = np.zeros((self.action_dim, self.reward_dim))
                if next_s not in self.q_table:
                    self.q_table[next_s] = np.zeros((self.action_dim, self.reward_dim))
                max_q = self.q_table[next_s][self.eval(next_s, self.weights)]
                model_td = r + (1 - terminal) * self.gamma * max_q - self.q_table[s][a]
                self.q_table[s][a] += self.learning_rate * model_td
                if self.gpi_pd:
                    priority = self._gpi_pd_priority(s, a, r, next_s, terminal, self.weights)
                    self.model.update_priority(ind, priority)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_epsilon,
            )

        if self.log and self.global_step % 1000 == 0:
            wandb.log(
                {
                    f"charts{self.idstr}/epsilon": self.epsilon,
                    f"losses{self.idstr}/scalarized_td_error": self.scalarization(td_error, self.weights),
                    f"losses{self.idstr}/mean_td_error": np.mean(td_error),
                    "global_step": self.global_step,
                },
            )

    @override
    def get_config(self) -> dict:
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "final_epsilon": self.final_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "use_gpi_policy": self.use_gpi_policy,
            "dyna": self.dyna,
            "dyna_updates": self.dyna_updates,
            "gpi_pd": self.gpi_pd,
            "min_priority": self.min_priority,
            "alpha": self.alpha,
            "weight": self.weights,
            "scalarization": self.scalarization.__name__,
            "seed": self.seed,
        }

    def train(
        self,
        start_time,
        total_timesteps: int = int(5e5),
        reset_num_timesteps: bool = True,
        eval_env: gym.Env = None,
        eval_freq: int = 1000,
    ):
        """Learning for the agent.

        Args:
            start_time: time when the training started
            total_timesteps: max number of timesteps to learn
            reset_num_timesteps: whether to reset timesteps or not when recalling learn
            eval_env: other environment to launch greedy evaluations
            eval_freq: number of timesteps between each policy evaluation
        """
        num_episodes = 0
        self.obs, _ = self.env.reset()

        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes

        for _ in range(1, total_timesteps + 1):
            self.global_step += 1

            self.action = self.__act(self.obs)
            (
                self.next_obs,
                self.reward,
                self.terminated,
                self.truncated,
                info,
            ) = self.env.step(self.action)

            self.update()

            if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                self.policy_eval(eval_env, scalarization=self.scalarization, weights=self.weights, log=self.log)

            if self.terminated or self.truncated:
                self.obs, _ = self.env.reset()
                num_episodes += 1
                self.num_episodes += 1

                if self.log and self.global_step % 1000 == 0:
                    wandb.log(
                        {
                            f"charts{self.idstr}/SPS": int(self.global_step / (time.time() - start_time)),
                            "global_step": self.global_step,
                        },
                    )
                    if "episode" in info:
                        log_episode_info(
                            info["episode"],
                            self.scalarization,
                            self.weights,
                            self.global_step,
                            self.id,
                            verbose=False,
                        )
            else:
                self.obs = self.next_obs
