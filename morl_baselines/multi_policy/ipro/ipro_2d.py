"""IPRO-2D algorithm for solving bi-objective problems."""

from copy import deepcopy
from typing import Literal, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from sortedcontainers import SortedKeyList

from morl_baselines.common.pareto import (
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


class IPRO2D(OuterLoop):
    """IPRO-2D algorithm for solving bi-objective problems."""

    def __init__(
        self,
        env: gym.Env,
        direction: Literal["maximize", "minimize"] = "maximize",
        offset: float = 1,
        tolerance: float = 1e-6,
        max_iterations: Optional[int] = None,
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
        experiment_name: Optional[str] = "IPRO-2D",
        project_name: str = "MORL-Baselines",
        wandb_entity: str = None,
        wandb_mode: Literal["online", "offline", "disabled"] = "online",
        seed: int = 1,
        rng: Union[np.random.Generator, None] = None,
    ):
        """Initialize the IPRO-2D algorithm.

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
            method="IPRO-2D",
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

        self.box_queue = SortedKeyList([], key=lambda x: x.volume)

    def reset(self):
        """Reset the algorithm."""
        self.box_queue = SortedKeyList([], key=lambda x: x.volume)
        super().reset()

    def estimate_error(self):
        """Estimate the error of the algorithm."""
        if len(self.box_queue) == 0:
            self.error = 0
        else:
            self.error = max(box.max_dist for box in self.box_queue)

    def split_box(self, box, point):
        """Split a box into two new boxes."""
        nadir1 = np.array([box.nadir[0], point[1]])
        ideal1 = np.array([point[0], box.ideal[1]])
        new_box1 = Box(ideal1, nadir1)

        nadir2 = np.array([point[0], box.nadir[1]])
        ideal2 = np.array([box.ideal[0], point[1]])
        new_box2 = Box(ideal2, nadir2)

        self.dominated_hv += Box(box.nadir, point).volume
        self.discarded_hv += Box(point, box.ideal).volume
        return [new_box1, new_box2]

    def update_box_queue(self, box, point):
        """Update the algorithm with a new point."""
        for box in self.split_box(box, point):
            if box.volume > self.tolerance and pareto_dominates(box.ideal, box.nadir):
                self.box_queue.add(box)

    def init_phase(
        self,
        extrema: Optional[tuple[np.ndarray, np.ndarray]] = None,
        deterministic: bool = False,
        eval_env: Optional[gym.Env] = None,
    ) -> tuple[list[Subsolution], bool]:
        """The initial phase in solving the problem."""
        subsolutions = []
        if extrema is None:
            extrema = []

            weight_vecs = np.eye(2)
            for weight_vec in weight_vecs:
                vec, sol = self.linear_train(
                    weight_vec=weight_vec,
                    deterministic=deterministic,
                    eval_env=eval_env,
                )
                print(f"Found solution {vec} for weight vector {weight_vec}")
                vec *= self.sign
                extrema.append(vec)
                subsolutions.append((weight_vec, vec, sol))
            extrema = np.array(extrema)

            self.nadir = np.min(extrema, axis=0) - self.offset
            self.ideal = np.max(extrema, axis=0) + self.offset
            self.pf = filter_pareto_dominated(np.array(extrema))
        else:
            self.nadir, self.ideal = extrema

        self.ref_point = np.copy(self.nadir) if self.ref_point is None else np.array(self.ref_point)
        self.bounding_box = Box(self.nadir, self.ideal)

        self.box_queue.add(self.bounding_box)
        self.estimate_error()
        self.total_hv = self.bounding_box.volume
        self.hv = self.compute_hypervolume(-self.sign * self.pf, -self.sign * self.ref_point)
        self.agent.reset_agent(pref_dim=self.dim)  # Reset since the utility function will change.
        return subsolutions, len(self.pf) == 1

    def is_done(self, step):
        """Check if the algorithm is done."""
        return not self.box_queue or super().is_done(step)

    def get_iterable_for_replay(self):
        """Get an iterable for replaying the algorithm."""
        box_queue = deepcopy(self.box_queue)
        return reversed(list(enumerate(box_queue)))

    def maybe_add_solution(
        self,
        subproblem: Subproblem,
        point: np.ndarray,
        item: tuple[int, Box],
    ) -> Subproblem | bool:
        """Check and add a new solution to the Pareto front if possible."""
        open_box_idx, open_box = item
        if strict_pareto_dominates(point, open_box.nadir):
            new_subproblem = Subproblem(referent=open_box.nadir, nadir=open_box.nadir, ideal=open_box.ideal)
            self.update_found(new_subproblem, point, box_idx=open_box_idx)
            return new_subproblem
        else:
            return False

    def maybe_add_completed(
        self,
        subproblem: Subproblem,
        point: np.ndarray,
        item: tuple[int, Box],
    ) -> Subproblem | bool:
        """Check and add to the completed set if possible."""
        open_box_idx, open_box = item
        if pareto_dominates(open_box.nadir, subproblem.referent):
            new_subproblem = Subproblem(referent=open_box.nadir, nadir=open_box.nadir, ideal=open_box.ideal)
            self.update_not_found(new_subproblem, point, box_idx=open_box_idx)
            return new_subproblem
        else:
            return False

    def update_found(self, subproblem, vec, box_idx=-1):
        """The update to perform when the Pareto oracle found a new Pareto dominant vector."""
        self.update_box_queue(self.box_queue.pop(box_idx), vec)
        self.pf = np.vstack((self.pf, vec))

    def update_not_found(self, subproblem, vec, box_idx=-1):
        """The update to perform when the Pareto oracle did not find a new Pareto dominant vector."""
        box = self.box_queue.pop(box_idx)
        self.discarded_hv += box.volume
        self.completed = np.vstack((self.completed, np.copy(subproblem.referent)))
        if strict_pareto_dominates(vec, self.nadir):
            self.robust_points = np.vstack((self.robust_points, vec))

    def decompose_problem(self, iteration, method="first"):
        """Decompose the problem into a subproblem."""
        box = self.box_queue[-1]
        subproblem = Subproblem(referent=box.nadir, nadir=box.nadir, ideal=box.ideal)
        return subproblem

    def update_excluded_volume(self):
        """This is already handled when splitting the boxes."""
        pass
