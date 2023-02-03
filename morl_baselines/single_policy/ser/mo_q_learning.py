"""Scalarized Q-learning for single policy multi-objective reinforcement learning."""
import time
from typing import Optional
from typing_extensions import override

import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.scalarization import weighted_sum
from morl_baselines.common.utils import linearly_decaying_value, log_episode_info


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
        project_name: str = "MORL-baselines",
        experiment_name: str = "MO-Q-Learning",
        log: bool = True,
        parent_writer: Optional[SummaryWriter] = None,
    ):
        """Initializes the MOQ-learning algorithm.

        Args:
            env: The environment to train on
            id: The id of the policy
            weights: The weights to use for the scalarization function
            scalarization: The scalarization function to use
            learning_rate: The learning rate
            gamma: The discount factor
            initial_epsilon: The initial epsilon value
            final_epsilon: The final epsilon value
            epsilon_decay_steps: The number of steps to decay epsilon over
            learning_starts: The number of steps to wait before starting to learn
            project_name: The name of the project used for logging
            experiment_name: The name of the experiment used for logging
            log: Whether to log or not
            parent_writer: The writer to use for logging. If None, a new writer is created.
        """
        MOAgent.__init__(self, env)
        MOPolicy.__init__(self, id)
        self.learning_rate = learning_rate
        self.id = id
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

        self.weights = weights
        self.scalarization = scalarization

        self.q_table = dict()

        self.log = log
        if parent_writer is not None:
            self.writer = parent_writer
        if self.log and parent_writer is None:
            self.setup_wandb(project_name, experiment_name)

    def __act(self, obs: np.array):
        # epsilon-greedy
        coin = np.random.rand()
        if coin < self.epsilon:
            return int(self.env.action_space.sample())
        else:
            return self.eval(obs)

    @override
    def eval(self, obs: np.array, w: Optional[np.ndarray] = None) -> int:
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
        """
        Updates the Q table
        """
        obs = tuple(self.obs)
        next_obs = tuple(self.next_obs)
        if obs not in self.q_table:
            self.q_table[obs] = np.zeros((self.action_dim, self.reward_dim))
        if next_obs not in self.q_table:
            self.q_table[next_obs] = np.zeros((self.action_dim, self.reward_dim))

        max_q = self.q_table[next_obs][self.eval(self.next_obs)]
        td_error = self.reward + (1 - self.terminated) * self.gamma * max_q - self.q_table[obs][self.action]
        self.q_table[obs][self.action] += self.learning_rate * td_error

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_epsilon,
            )

        if self.log and self.global_step % 1000 == 0:
            self.writer.add_scalar(f"charts{self.idstr}/epsilon", self.epsilon, self.global_step)
            self.writer.add_scalar(
                f"losses{self.idstr}/scalarized_td_error",
                self.scalarization(td_error, self.weights),
                self.global_step,
            )
            self.writer.add_scalar(f"losses{self.idstr}/mean_td_error", np.mean(td_error), self.global_step)

    @override
    def get_config(self) -> dict:
        return {
            "alpha": self.learning_rate,
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "final_epsilon": self.final_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "weight": self.weights,
            "scalarization": self.scalarization.__name__,
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
                self.policy_eval(eval_env, weights=self.weights, writer=self.writer)

            if self.terminated or self.truncated:
                self.obs, _ = self.env.reset()
                num_episodes += 1
                self.num_episodes += 1

                if self.log and self.global_step % 1000 == 0:
                    print("SPS:", int(self.global_step / (time.time() - start_time)))
                    self.writer.add_scalar(
                        f"charts{self.idstr}/SPS",
                        int(self.global_step / (time.time() - start_time)),
                        self.global_step,
                    )
                    if "episode" in info:
                        log_episode_info(
                            info["episode"],
                            self.scalarization,
                            self.weights,
                            self.global_step,
                            self.id,
                            self.writer,
                        )
            else:
                self.obs = self.next_obs
