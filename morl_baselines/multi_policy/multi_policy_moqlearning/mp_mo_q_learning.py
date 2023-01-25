"""Outer-loop MOQ-learning algorithm (uses multiple weights)."""
import time
from typing_extensions import override

import numpy as np

from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.performance_indicators import hypervolume
from morl_baselines.common.scalarization import weighted_sum
from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning


class MPMOQLearning(MOAgent):
    """Multi-policy MOQ-Learning: Outer loop version of mo_q_learning.

    Paper: Paper: K. Van Moffaert, M. Drugan, and A. Nowe, Scalarized Multi-Objective Reinforcement Learning: Novel Design Techniques. 2013. doi: 10.1109/ADPRL.2013.6615007.
    """

    def __init__(
        self,
        env,
        ref_point: np.ndarray,
        weights_step_size: float = 0.1,
        scalarization=weighted_sum,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        initial_epsilon: float = 0.1,
        final_epsilon: float = 0.1,
        epsilon_decay_steps: int = None,
        learning_starts: int = 0,
        num_timesteps: int = int(5e5),
        eval_freq: int = 1000,
        project_name: str = "MORL-baselines",
        experiment_name: str = "MultiPolicy MO Q-Learning",
        log: bool = True,
    ):
        """Initialize the Multi-policy MOQ-learning algorithm.

        Args:
            env: The environment to learn from.
            ref_point: The reference point for the hypervolume calculation.
            weights_step_size: The step size for the weights creation.
            scalarization: The scalarization function to use.
            learning_rate: The learning rate.
            gamma: The discount factor.
            initial_epsilon: The initial epsilon value.
            final_epsilon: The final epsilon value.
            epsilon_decay_steps: The number of steps for epsilon decay.
            learning_starts: The number of steps before learning starts.
            num_timesteps: The number of timesteps for each agent to train for.
            eval_freq: The frequency of evaluation.
            project_name: The name of the project for logging.
            experiment_name: The name of the experiment for logging.
            log: Whether to log or not.
        """
        super().__init__(env)
        # Learning
        self.scalarization = scalarization
        self.weights_step_size = weights_step_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.learning_starts = learning_starts
        self.num_timesteps = num_timesteps
        self.eval_freq = eval_freq

        # Logging
        self.ref_point = ref_point
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log

        self.weights = self.__generate_weights(self.weights_step_size)
        print(f"Generated weights: {self.weights}")
        if self.log:
            self.setup_wandb(project_name=self.project_name, experiment_name=self.experiment_name)
        else:
            self.writer = None

        self.agents = [
            MOQLearning(
                env,
                id=i,
                weights=w,
                scalarization=scalarization,
                learning_rate=learning_rate,
                gamma=gamma,
                initial_epsilon=initial_epsilon,
                final_epsilon=final_epsilon,
                epsilon_decay_steps=epsilon_decay_steps,
                learning_starts=learning_starts,
                project_name=project_name,
                experiment_name=experiment_name,
                log=log,
                parent_writer=self.writer,
            )
            for i, w in enumerate(self.weights)
        ]

    @override
    def get_config(self) -> dict:
        return {
            "alpha": self.learning_rate,
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "final_epsilon": self.final_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "weight_step_size": self.weights_step_size,
            "num_weights": len(self.weights),
            "scalarization": self.scalarization.__name__,
        }

    def __generate_weights(self, step_size):
        return np.linspace((0.0, 1.0), (1.0, 0.0), int(1 / step_size) + 1, dtype=np.float32)

    def eval_all_agents(self):
        """Evaluate all agents and return the rewards and discounted rewards.

        Returns:
            a tuple of rewards and discounted rewards.
        """
        discounted_rewards = []
        rewards = []
        for a in self.agents:
            _, _, vec, discounted_vec = a.policy_eval(eval_env=self.env, weights=a.weights, writer=self.writer)
            discounted_rewards.append(discounted_vec)
            rewards.append(vec)
        if self.log:
            print(f"Evaluation of all agents: {rewards}")
            print(f"discounted: {discounted_rewards}")
        return rewards, discounted_rewards

    def train(self):
        """Train the algorithm."""
        start_time = time.time()
        training_epoch = int(self.num_timesteps / self.eval_freq)
        for e in range(training_epoch):
            print(f"Training epoch #{e}")
            for a in self.agents:
                a.train(
                    start_time,
                    total_timesteps=self.eval_freq,
                    reset_num_timesteps=False,
                )
            self.global_step += len(self.agents) * self.eval_freq
            rewards, disc_rewards = self.eval_all_agents()
            hv = hypervolume(self.ref_point, rewards)
            if self.log:
                self.writer.add_scalar("metrics/hypervolume", hv, self.global_step)

        if self.writer is not None:
            self.writer.close()
