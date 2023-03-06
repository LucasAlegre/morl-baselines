"""Outer-loop MOQ-learning algorithm (uses multiple weights)."""
import time
from copy import deepcopy
from typing import List, Optional
from typing_extensions import override

import numpy as np
from mo_gymnasium import policy_evaluation_mo

from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.performance_indicators import (
    expected_utility,
    hypervolume,
    maximum_utility_loss,
)
from morl_baselines.common.scalarization import weighted_sum
from morl_baselines.common.utils import equally_spaced_weights, random_weights
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport
from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning


class MPMOQLearning(MOAgent, MOPolicy):
    """Multi-policy MOQ-Learning: Outer loop version of mo_q_learning.

    Paper: Paper: K. Van Moffaert, M. Drugan, and A. Nowe, Scalarized Multi-Objective Reinforcement Learning: Novel Design Techniques. 2013. doi: 10.1109/ADPRL.2013.6615007.
    """

    def __init__(
        self,
        env,
        scalarization=weighted_sum,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        initial_epsilon: float = 0.1,
        final_epsilon: float = 0.1,
        epsilon_decay_steps: int = None,
        weight_selection_algo: str = "random",
        epsilon_ols: Optional[float] = None,
        use_gpi: bool = False,
        reuse_q_table: bool = True,
        dyna: bool = False,
        dyna_updates: int = 5,
        project_name: str = "MORL Baselines",
        experiment_name: str = "MultiPolicy MO Q-Learning",
        log: bool = True,
    ):
        """Initialize the Multi-policy MOQ-learning algorithm.

        Args:
            env: The environment to learn from.
            scalarization: The scalarization function to use.
            learning_rate: The learning rate.
            gamma: The discount factor.
            initial_epsilon: The initial epsilon value.
            final_epsilon: The final epsilon value.
            epsilon_decay_steps: The number of steps for epsilon decay.
            weight_selection_algo: The algorithm to use for weight selection. Options: "random", "ols", "gpi-ls"
            epsilon_ols: The epsilon value for the optimistic linear support.
            use_gpi: Whether to use the Generalized Policy Improvement (GPI) or not.
            reuse_q_table: Whether to reuse a Q-table from a previous learned policy when initializing a new policy.
            dyna: Whether to use Dyna-Q or not.
            dyna_updates: The number of Dyna-Q updates to perform.
            project_name: The name of the project for logging.
            experiment_name: The name of the experiment for logging.
            log: Whether to log or not.
        """
        MOAgent.__init__(self, env)
        MOPolicy.__init__(self)
        # Learning
        self.scalarization = scalarization
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.use_gpi = use_gpi
        self.dyna = dyna
        self.dyna_updates = dyna_updates
        self.reuse_q_table = reuse_q_table
        # Linear support
        self.policies = []
        self.weight_selection_algo = weight_selection_algo
        self.epsilon_ols = epsilon_ols
        assert self.weight_selection_algo in [
            "random",
            "ols",
            "gpi-ls",
        ], f"Unknown weight selection algorithm: {self.weight_selection_algo}."
        self.linear_support = LinearSupport(num_objectives=self.reward_dim, epsilon=epsilon_ols)

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log

        if self.log:
            self.setup_wandb(project_name=self.project_name, experiment_name=self.experiment_name)
        else:
            self.writer = None

    @override
    def get_config(self) -> dict:
        return {
            "alpha": self.learning_rate,
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "final_epsilon": self.final_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "scalarization": self.scalarization.__name__,
            "use_gpi": self.use_gpi,
            "weight_selection_algo": self.weight_selection_algo,
            "epsilon_ols": self.epsilon_ols,
            "reuse_q_table": self.reuse_q_table,
            "dyna": self.dyna,
            "dyna_updates": self.dyna_updates,
        }

    def update(self) -> None:
        """This class does not implement the update method."""
        pass

    def _gpi_action(self, state: np.ndarray, w: np.ndarray) -> int:
        """Get the action given by the GPI policy.

        GPI(s, w) = argmax_a max_pi Q^pi(s, a, w) .

        Args:
            state: The state to get the action for.
            weights: The weights to use for the scalarization.

        Returns:
            The action to take.
        """
        q_vals = np.stack([policy.scalarized_q_values(state, w) for policy in self.policies])
        _, action = np.unravel_index(np.argmax(q_vals), q_vals.shape)
        return int(action)

    @override
    def eval(self, obs: np.array, w: Optional[np.ndarray] = None, policy_ind: Optional[int] = None) -> int:
        """If use_gpi is True, return the action given by the GPI policy. Otherwise, return the action given by the policy with the given index."""
        if self.use_gpi:
            return self._gpi_action(obs, w)
        else:
            return self.policies[policy_ind].eval(obs, w)

    def delete_policies(self, delete_indx: List[int]):
        """Delete the policies with the given indices."""
        for i in sorted(delete_indx, reverse=True):
            self.policies.pop(i)

    def train(
        self,
        num_iterations: int,
        timesteps_per_iteration: int,
        eval_freq: int = 1000,
        eval_env=None,
        ref_point: Optional[np.ndarray] = None,
        test_weights: Optional[np.ndarray] = None,
        num_episodes_eval: int = 10,
    ):
        """Learn a set of policies.

        Args:
            num_iterations: The number of iterations/policies to train.
            timesteps_per_iteration: The number of timesteps per iteration.
            eval_freq: The frequency of evaluation.
            eval_env: The environment to use for evaluation.
            ref_point: The reference point for the hypervolume calculation.
            test_weights: The weight vectors to use for evaluation (e.g. for expected utility).
            epsilon_linear_support: The epsilon value for the linear support algorithm.
            num_episodes_eval: The number of episodes used to evaluate the value of a policy.
        """
        if eval_env is None:
            eval_env = deepcopy(self.env)

        if test_weights is None:
            test_weights = equally_spaced_weights(self.reward_dim, n=64)

        for iter in range(num_iterations):
            if self.weight_selection_algo == "ols" or self.weight_selection_algo == "gpi-ls":
                w = self.linear_support.next_weight(
                    algo=self.weight_selection_algo,
                    gpi_agent=self if self.weight_selection_algo == "gpi-ls" else None,
                    env=eval_env if self.weight_selection_algo == "gpi-ls" else None,
                    rep_eval=num_episodes_eval,
                )
            elif self.weight_selection_algo == "random":
                w = random_weights(self.reward_dim)

            new_agent = MOQLearning(
                env=self.env,
                id=iter,
                weights=w,
                scalarization=self.scalarization,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                initial_epsilon=self.initial_epsilon,
                final_epsilon=self.final_epsilon,
                epsilon_decay_steps=self.epsilon_decay_steps,
                dyna=self.dyna,
                dyna_updates=self.dyna_updates,
                log=self.log,
                parent_writer=self.writer,
            )
            if self.reuse_q_table and len(self.policies) > 0:
                reuse_ind = np.argmax([np.dot(w, v) for v in self.linear_support.ccs])
                new_agent.q_table = deepcopy(self.policies[reuse_ind].q_table)
            self.policies.append(new_agent)

            start_time = time.time()
            new_agent.global_step = self.global_step
            new_agent.train(
                start_time=start_time,
                total_timesteps=timesteps_per_iteration,
                reset_num_timesteps=False,
                eval_freq=eval_freq,
                eval_env=eval_env,
            )
            self.global_step = new_agent.global_step

            value = policy_evaluation_mo(agent=new_agent, env=eval_env, w=w, rep=num_episodes_eval)
            removed_inds = self.linear_support.add_solution(value, w)
            self.delete_policies(removed_inds)

            if self.log:
                front = self.linear_support.ccs
                self.writer.add_scalar("metrics/eu", expected_utility(front, test_weights), self.global_step)
                self.writer.add_scalar(
                    "metrics/mul",
                    maximum_utility_loss(front, eval_env.pareto_front(gamma=self.gamma), test_weights),
                    self.global_step,
                )
                if self.use_gpi:
                    front_gpi = [
                        policy_evaluation_mo(agent=self, env=eval_env, w=w, rep=num_episodes_eval) for w in test_weights
                    ]
                    self.writer.add_scalar("metrics/eu_gpi", expected_utility(front_gpi, test_weights), self.global_step)
                    self.writer.add_scalar(
                        "metrics/mul_gpi",
                        maximum_utility_loss(front_gpi, eval_env.pareto_front(gamma=self.gamma), test_weights),
                        self.global_step,
                    )
                if ref_point is not None:
                    self.writer.add_scalar("metrics/hv", hypervolume(ref_point, front), self.global_step)
                    if self.use_gpi:
                        self.writer.add_scalar("metrics/hv_gpi", hypervolume(ref_point, front_gpi), self.global_step)

        if self.writer is not None:
            self.close_wandb()
