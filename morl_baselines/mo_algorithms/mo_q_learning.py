import time
from typing import Optional

import gym
import numpy as np
from mo_gym import eval_mo
from torch.utils.tensorboard import SummaryWriter

from morl_baselines.common.utils import linearly_decaying_epsilon
from morl_baselines.common.scalarization import weighted_sum, tchebicheff

from morl_baselines.common.morl_algorithm import MORLAlgorithm


class MOQLearning(MORLAlgorithm):
    """
    Scalarized Q learning:
    Maintains one Q-table per objective, rely on a scalarization function to choose the moves.
    K. Van Moffaert, M. Drugan, and A. Nowe, Scalarized Multi-Objective Reinforcement Learning: Novel Design Techniques. 2013. doi: 10.1109/ADPRL.2013.6615007.
    """
    def __init__(
            self,
            env,
            id: int,
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
            parent_writer: Optional[SummaryWriter] = None
    ):

        super().__init__(env)
        self.learning_rate = learning_rate
        self.id = id
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

    def eval(self, obs: np.array, w: Optional[np.ndarray] = None) -> int:
        """Greedily chooses best action using the scalarization method"""
        obs = tuple(obs)
        if obs not in self.q_table:
            return int(self.env.action_space.sample())
        return int(np.argmax(self.scalarization(self.q_table[obs], self.weights)))

    def __update(self):
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
        td_error = self.reward + (1 - self.terminal) * self.gamma * max_q - self.q_table[obs][self.action]
        self.q_table[obs][self.action] += self.learning_rate * td_error

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_epsilon(self.initial_epsilon, self.epsilon_decay_steps, self.num_timesteps,
                                                     self.learning_starts, self.final_epsilon)

        if self.log and self.num_timesteps % 1000 == 0:
            self.writer.add_scalar(f"losses_{self.id}/scalarized_td_error", self.scalarization(td_error, self.weights), self.num_timesteps)
            self.writer.add_scalar(f"losses_{self.id}/mean_td_error", np.mean(td_error), self.num_timesteps)

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

    def train(self):
        pass

    def policy_eval(self, eval_env, log=True):
        # TODO, make eval_mo generic to scalarization?
        scalarized_reward, scalarized_discounted_reward, vec_reward, discounted_vec_reward = eval_mo(self, eval_env, self.weights)
        if log:
            self.writer.add_scalar(f"eval_{self.id}/scalarized_reward", scalarized_reward, self.num_timesteps)
            self.writer.add_scalar(f"eval_{self.id}/scalarized_discounted_reward", scalarized_discounted_reward,
                                   self.num_timesteps)
            for i in range(vec_reward.shape[0]):
                self.writer.add_scalar(f"eval_{self.id}/vec_{i}", vec_reward[i], self.num_timesteps)
                self.writer.add_scalar(f"eval_{self.id}/discounted_vec_{i}", discounted_vec_reward[i], self.num_timesteps)

        return (
            scalarized_reward,
            scalarized_discounted_reward,
            vec_reward,
            discounted_vec_reward
        )

    def learn(
            self,
            start_time,
            total_timesteps: int = int(5e5),
            reset_num_timesteps: bool = True,
            eval_env: gym.Env = None,
            eval_freq: int = 1000,
    ):
        """
        Learning for the agent
        :param total_timesteps: max number of timesteps to learn
        :param reset_num_timesteps: whether to reset timesteps or not when recalling learn
        :param eval_env: other environment to launch greedy evaluations
        :param eval_freq: number of timesteps between each policy evaluation
        """
        episode_reward = 0.0
        episode_vec_reward = np.zeros_like(self.weights)
        num_episodes = 0
        self.obs, done = self.env.reset(), False

        self.num_timesteps = 0 if reset_num_timesteps else self.num_timesteps
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes

        for _ in range(1, total_timesteps + 1):
            self.num_timesteps += 1

            self.action = self.__act(self.obs)
            self.next_obs, self.reward, done, info = self.env.step(self.action)
            self.terminal = done if "TimeLimit.truncated" not in info else not info["TimeLimit.truncated"]

            self.__update()

            if eval_env is not None and self.log and self.num_timesteps % eval_freq == 0:
                self.policy_eval(eval_env, self.log)

            episode_reward += self.scalarization(self.reward, self.weights)
            episode_vec_reward += self.reward
            if done:
                self.obs, done = self.env.reset(), False
                num_episodes += 1
                self.num_episodes += 1

                if num_episodes % 1000 == 0:
                    print(
                        f"Episode: {self.num_episodes} Step: {self.num_timesteps}, Ep. Total Reward: {episode_reward}, {episode_vec_reward}")
                if self.log:
                    print("SPS:", int(self.num_timesteps / (time.time() - start_time)))
                    self.writer.add_scalar(f"charts_{self.id}/SPS", int(self.num_timesteps / (time.time() - start_time)), self.num_timesteps)

                    self.writer.add_scalar(f"charts_{self.id}/timesteps", self.num_timesteps)
                    self.writer.add_scalar(f"metrics_{self.id}/episode", self.num_episodes, self.num_timesteps)
                    self.writer.add_scalar(f"metrics_{self.id}/scalarized_episode_reward", episode_reward, self.num_timesteps)
                    self.writer.add_scalar(f"charts_{self.id}/learning_rate", self.learning_rate, self.num_timesteps)
                    self.writer.add_scalar(f"charts_{self.id}/epsilon", self.epsilon, self.num_timesteps)
                    for i in range(episode_vec_reward.shape[0]):
                        self.writer.add_scalar(f"metrics_{self.id}/episode_reward_obj{i}", episode_vec_reward[i],
                                               self.num_timesteps)

                episode_reward = 0.0
                episode_vec_reward = np.zeros(self.weights.shape[0])
            else:
                self.obs = self.next_obs
