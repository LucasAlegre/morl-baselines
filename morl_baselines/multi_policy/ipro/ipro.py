"""IPRO algorithm."""

from typing import Literal, Optional, Union

import gymnasium as gym
import numpy as np
import torch

from morl_baselines.common.pareto import (
    batched_strict_pareto_dominates,
    filter_pareto_dominated,
    pareto_dominates,
    strict_pareto_dominates,
)
from morl_baselines.multi_policy.ipro.box import Box
from morl_baselines.multi_policy.ipro.outer_loop import (
    OuterLoop,
    Subproblem,
    Subsolution,
)


class IPRO(OuterLoop):
    """IPRO algorithm for solving multi-objective problems."""

    def __init__(
        self,
        env: gym.Env,
        direction: Literal["maximize", "minimize"] = "maximize",
        offset: float = 1,
        tolerance: float = 1e-6,
        max_iterations: Optional[int] = None,
        update_freq: int = 1,
        reset_agent: bool = False,
        aug: float = 0.1,
        scale: float = 100,
        iter_total_timesteps: int = 500000,
        learning_rate: float = 2.5e-4,
        num_steps: int = 128,
        anneal_lr: bool = True,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        num_minibatches: int = 4,
        update_epochs: int = 4,
        norm_adv: bool = True,
        clip_coef: float = 0.2,
        clip_vloss: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = None,
        mc_k: int = 32,
        device: Union[torch.device, str] = "auto",
        log: bool = False,
        experiment_name: Optional[str] = "IPRO",
        project_name: str = "MORL-Baselines",
        wandb_entity: str = None,
        wandb_mode: Literal["online", "offline", "disabled"] = "online",
        seed: int = 1,
        rng: Union[np.random.Generator, None] = None,
    ):
        """Initialize the IPRO algorithm.

        Args:
            env (gym.Env): The environment to solve.
            direction (str): The direction of the objectives, either "maximize" or "minimize
            offset (float): The offset to apply to the extrema.
            tolerance (float): The tolerance for the algorithm.
            max_iterations (int, optional): The maximum number of iterations to run the algorithm.
            update_freq (int): The frequency of updating hypervolume improvement heuristic for computing the next referent.
            reset_agent (bool): Whether to reset the agent after each iteration.
            aug (float): The augmentation factor for the AASF.
            scale (float): The scale factor for the AASF.
            iter_total_timesteps (int): The total number of timesteps for each iteration.
            learning_rate (float): The learning rate for the PPO algorithm.
            num_steps (int): The number of rollout steps to take.
            anneal_lr (bool): Whether to anneal the learning rate.
            gamma (float): The discount factor for the PPO algorithm.
            gae_lambda (float): The lambda parameter for Generalized Advantage Estimation.
            num_minibatches (int): The number of minibatches to use for training.
            update_epochs (int): The number of epochs to update the policy.
            norm_adv (bool): Whether to normalize the advantages.
            clip_coef (float): The clipping coefficient for the PPO algorithm.
            clip_vloss (bool): Whether to clip the value loss.
            ent_coef (float): The entropy coefficient for the PPO algorithm.
            vf_coef (float): The value function coefficient for the PPO algorithm.
            max_grad_norm (float): The maximum gradient norm for the PPO algorithm.
            target_kl (float, optional): The target KL divergence for the PPO algorithm.
            mc_k (int): The number of Monte Carlo samples to use for the AASF.
            device (torch.device or str): The device to use for training.
            log (bool): Whether to log the training progress.
            experiment_name (str, optional): The name of the experiment for logging.
            project_name (str): The name of the project for logging.
            wandb_entity (str, optional): The entity for Weights & Biases logging.
            wandb_mode (str): The mode for Weights & Biases logging, either "online", "offline", or "disabled".
            seed (int): The random seed for reproducibility.
            rng (np.random.Generator, optional): A random number generator for reproducibility.
        """
        super().__init__(
            env,
            method="IPRO",
            direction=direction,
            offset=offset,
            tolerance=tolerance,
            max_iterations=max_iterations,
            reset_agent=reset_agent,
            aug=aug,
            scale=scale,
            total_timesteps=iter_total_timesteps,
            learning_rate=learning_rate,
            num_steps=num_steps,
            anneal_lr=anneal_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            num_minibatches=num_minibatches,
            update_epochs=update_epochs,
            norm_adv=norm_adv,
            clip_coef=clip_coef,
            clip_vloss=clip_vloss,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            mc_k=mc_k,
            device=device,
            log=log,
            experiment_name=experiment_name,
            project_name=project_name,
            wandb_entity=wandb_entity,
            wandb_mode=wandb_mode,
            seed=seed,
            rng=rng,
        )
        self.update_freq = update_freq
        self.lower_points = []
        self.upper_points = []

        self.rng = rng if rng is not None else np.random.default_rng(seed)

    def reset(self):
        """Reset the algorithm."""
        self.lower_points = []
        self.upper_points = []
        super().reset()

    def init_phase(
        self,
        extrema: Optional[tuple[np.ndarray, np.ndarray]] = None,
        deterministic: bool = False,
        eval_env: Optional[gym.Env] = None,
    ) -> tuple[list[Subsolution], bool]:
        """Run the initialisation phase of the algorithm.

        This phase computes the bounding box of the Pareto front by solving the maximisation and minimisation problems
        for all objectives. For the ideal, this is exact and for the nadir this is guaranteed to be a pessimistic
        estimate. The lower points are then computed and initial hypervolume improvements.
        """
        subsolutions = []

        if extrema is None:
            nadir = np.zeros(self.dim)
            ideal = np.zeros(self.dim)
            pf = []
            weight_vecs = np.eye(self.dim)

            for i, weight_vec in enumerate(weight_vecs):
                ideal_vec, ideal_sol = self.linear_train(
                    weight_vec=weight_vec,
                    deterministic=deterministic,
                    eval_env=eval_env,
                )
                print(f"Found solution {ideal_vec} for weight vector {weight_vec}")
                nadir_vec, _ = self.linear_train(
                    weight_vec=-1 * weight_vec,
                    deterministic=deterministic,
                    eval_env=eval_env,
                )
                ideal_vec *= self.sign
                nadir_vec *= self.sign
                ideal[i] = ideal_vec[i]
                nadir[i] = nadir_vec[i]
                pf.append(ideal_vec)
                subsolutions.append((weight_vec, ideal_vec, ideal_sol))

            self.pf = filter_pareto_dominated(np.array(pf))
            nadir = nadir - self.offset  # Necessary to ensure every Pareto optimal point strictly dominates the nadir.
            ideal = ideal + self.offset
            self.nadir = np.copy(nadir)
            self.ideal = np.copy(ideal)

            if len(self.pf) == 1:  # If the Pareto front is the ideal.
                return subsolutions, True
        else:
            self.nadir, self.ideal = extrema

        self.ref_point = np.copy(self.nadir) if self.ref_point is None else np.array(self.ref_point)
        self.hv = self.compute_hypervolume(-self.sign * self.pf, -self.sign * self.ref_point)

        self.bounding_box = Box(self.nadir, self.ideal)
        self.total_hv = self.bounding_box.volume
        self.lower_points = np.array([self.nadir])

        for point in self.pf:  # Initialise the lower points.
            self.update_lower_points(np.array(point))

        self.upper_points = np.array([self.ideal])  # Initialise the upper points.
        self.error = max(self.ideal - self.nadir)
        self.compute_hvis()

        return subsolutions, False

    def compute_hvis(self, num=50):
        """Compute the hypervolume improvements of the lower points.

        An optional num parameter can be given as computing the hypervolume for a large number of potential points is
        expensive.
        """
        discarded_extrema = np.vstack((self.pf, self.completed))
        hvis = np.zeros(len(self.lower_points))

        for lower_id in self.rng.choice(len(self.lower_points), min(num, len(self.lower_points)), replace=False):
            hv = self.compute_hypervolume(np.vstack((discarded_extrema, self.lower_points[lower_id])), self.ideal)
            hvis[lower_id] = hv  # We don't have to compute the difference as it is proportional to the hypervolume.

        sorted_args = np.argsort(hvis)[::-1]
        self.lower_points = self.lower_points[sorted_args]

    def max_hypervolume_improvement(self):
        """Recompute the hypervolume improvements and return the point that maximises it."""
        self.compute_hvis()
        return self.lower_points[0]

    def estimate_error(self):
        """Estimate the error of the algorithm."""
        if len(self.upper_points) == 0:
            error = 0
        else:
            pf = np.array(list(self.pf))
            diffs = self.upper_points[:, None, :] - pf[None, :, :]
            error = np.max(np.min(np.max(diffs, axis=2), axis=1))
        self.error = error

    def update_upper_points(self, vec):
        """Update the upper set."""
        strict_dominates = batched_strict_pareto_dominates(self.upper_points, vec)
        to_keep = self.upper_points[strict_dominates == 0]
        shifted = np.stack([self.upper_points[strict_dominates == 1]] * self.dim)
        shifted[range(self.dim), :, range(self.dim)] = np.expand_dims(vec, -1)
        shifted = shifted.reshape(-1, self.dim)
        shifted = shifted[np.all(shifted > self.nadir, axis=-1)]

        new_upper_points = np.vstack((to_keep, shifted))
        self.upper_points = filter_pareto_dominated(new_upper_points)

    def update_lower_points(self, vec):
        """Update the upper set."""
        strict_dominates = batched_strict_pareto_dominates(vec, self.lower_points)
        to_keep = self.lower_points[strict_dominates == 0]
        shifted = np.stack([self.lower_points[strict_dominates == 1]] * self.dim)
        shifted[range(self.dim), :, range(self.dim)] = np.expand_dims(vec, -1)
        shifted = shifted.reshape(-1, self.dim)
        shifted = shifted[np.all(self.ideal > shifted, axis=-1)]

        new_lower_points = np.vstack((to_keep, shifted))
        self.lower_points = -filter_pareto_dominated(-new_lower_points)

    def select_referent(self, method="random"):
        """The method to select a new referent."""
        if method == "random":
            return self.lower_points[self.rng.integers(0, len(self.lower_points))]
        if method == "first":
            return self.lower_points[0]
        else:
            raise ValueError(f"Unknown method {method}")

    def get_iterable_for_replay(self):
        """Get an iterable for replaying the algorithm."""
        return np.copy(self.lower_points)

    def maybe_add_solution(
        self,
        subproblem: Subproblem,
        point: np.ndarray,
        lower: np.ndarray,
    ) -> Subproblem | bool:
        """Check and add a new solution to the Pareto front if possible."""
        if strict_pareto_dominates(point, lower):
            new_subproblem = Subproblem(referent=lower, nadir=self.nadir, ideal=self.ideal)
            self.update_found(new_subproblem, point)
            return new_subproblem
        else:
            return False

    def maybe_add_completed(
        self,
        subproblem: Subproblem,
        point: np.ndarray,
        lower: np.ndarray,
    ) -> Subproblem | bool:
        """Check and add to the completed set if possible."""
        if pareto_dominates(lower, subproblem.referent):
            new_subproblem = Subproblem(referent=lower, nadir=self.nadir, ideal=self.ideal)
            self.update_not_found(new_subproblem, point)
            return new_subproblem
        else:
            return False

    def update_found(self, subproblem, vec):
        """The update to perform when the Pareto oracle found a new Pareto dominant vector."""
        self.pf = np.vstack((self.pf, vec))
        self.update_lower_points(vec)
        self.update_upper_points(vec)

    def update_not_found(self, subproblem, vec):
        """The update to perform when the Pareto oracle did not find a new Pareto dominant vector."""
        self.completed = np.vstack((self.completed, subproblem.referent))
        self.lower_points = self.lower_points[np.any(self.lower_points != subproblem.referent, axis=1)]
        self.update_upper_points(subproblem.referent)
        if strict_pareto_dominates(vec, self.nadir):
            self.robust_points = np.vstack((self.robust_points, vec))

    def decompose_problem(self, iteration, method="first"):
        """Decompose the problem into a subproblem."""
        if iteration % self.update_freq == 0:
            self.compute_hvis()
        referent = self.select_referent(method=method)
        subproblem = Subproblem(referent=referent, nadir=self.nadir, ideal=self.ideal)
        return subproblem

    def update_excluded_volume(self):
        """Update the excluded volume based on the completed solutions."""
        self.dominated_hv = self.compute_hypervolume(-self.pf, -self.nadir)
        self.discarded_hv = self.compute_hypervolume(np.vstack((self.pf, self.completed)), self.ideal)
