"""Outer loop for the IPRO algorithm."""

import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterable, Literal, Optional, TypeAlias

import gymnasium as gym
import numpy as np
import torch
import wandb
from pymoo.config import Config
from pymoo.indicators.hv import Hypervolume

from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.pareto import (
    batched_pareto_dominates,
    batched_strict_pareto_dominates,
    filter_pareto_dominated,
    strict_pareto_dominates,
)
from morl_baselines.single_policy.ser.nl_mo_ppo import NLMOPPO


Config.warnings["not_compiled"] = False


@dataclass
class Subproblem:
    """A subproblem."""

    referent: np.ndarray
    nadir: np.ndarray
    ideal: np.ndarray


Subsolution: TypeAlias = tuple[Subproblem, np.ndarray, Any]
IPROCallback: TypeAlias = Callable[[int, float, float, float, float, float], Any]


def linear_scalarization(batch: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Perform linear scalarization of a batch of points."""
    return torch.sum(batch * weights, dim=-1)


def aasf(batch, referent, nadir, ideal, aug=0.0, scale=100):
    """Compute the Augmented Achievement Scalarizing Function (AASF)."""
    pos_vec = ideal - nadir
    frac_improvement = scale * (batch - referent) / pos_vec
    return torch.min(frac_improvement, dim=-1)[0] + aug * torch.mean(frac_improvement, dim=-1)


class OuterLoop(MOAgent):
    """The outer loop for IPRO. This is not meant to be used directly."""

    def __init__(
        self,
        env: gym.Env,
        method: str = "IPRO",
        direction: Literal["maximize", "minimize"] = "maximize",
        offset: float = 1,
        tolerance: float = 1e-1,
        max_iterations: Optional[int] = None,
        aug: float = 0.1,
        scale: float = 100,
        reset_agent: bool = False,
        log: bool = False,
        experiment_name: Optional[str] = None,
        project_name: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_mode: Literal["online", "offline", "disabled"] = "online",
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the outer loop for multi-objective optimization.

        Args:
            env (gym.Env): The environment to solve.
            method (str): The method to use for the outer loop.
            direction (Literal["maximize", "minimize"]): The optimization direction.
            offset (float): Offset for the Augmented Achievement Scalarizing Function.
            tolerance (float): Tolerance for convergence.
            max_iterations (Optional[int]): Maximum number of iterations.
            aug (float): Augmentation factor for AASF.
            scale (float): Scaling factor for AASF.
            reset_agent (bool): Whether to reset the agent before training.
            log (bool): Whether to log results with wandb.
            experiment_name (Optional[str]): Name of the experiment for logging.
            project_name (Optional[str]): Name of the wandb project.
            wandb_entity (Optional[str]): Wandb entity name.
            wandb_mode (Literal["online", "offline", "disabled"]): Wandb mode.
            seed (Optional[int]): Random seed for reproducibility.
            **kwargs: Additional keyword arguments for the agent.
        """
        MOAgent.__init__(self, env, device=kwargs["device"], seed=seed)

        self.env = env
        self.dim = self.env.reward_space.shape[0]
        self.nadir, self.ideal = None, None
        self.agent = NLMOPPO(0, env, seed=seed, **kwargs)

        self.method = method
        self.direction = direction
        self.ref_point = None
        self.offset = offset
        self.tolerance = tolerance
        self.max_iterations = max_iterations if max_iterations is not None else np.inf
        self.aug = aug
        self.scale = scale
        self.reset_agent = reset_agent

        self.sign = 1 if direction == "maximize" else -1
        self.bounding_box = None
        self.ideal = None
        self.nadir = None
        self.pf = np.empty((0, self.dim))
        self.robust_points = np.empty((0, self.dim))
        self.completed = np.empty((0, self.dim))

        self.hv = 0
        self.total_hv = 0
        self.dominated_hv = 0
        self.discarded_hv = 0
        self.coverage = 0
        self.error = np.inf
        self.replay_triggered = 0

        self.track = log
        self.run_id = None
        self.exp_name = experiment_name
        self.wandb_project_name = project_name
        self.wandb_entity = wandb_entity
        self.wandb_mode = wandb_mode

        self.seed = seed

    def reset(self):
        """Reset the algorithm."""
        self.bounding_box = None
        self.ideal = None
        self.nadir = None
        self.pf = np.empty((0, self.dim))
        self.robust_points = np.empty((0, self.dim))
        self.completed = np.empty((0, self.dim))

        self.hv = 0
        self.total_hv = 0
        self.dominated_hv = 0
        self.discarded_hv = 0
        self.coverage = 0
        self.error = np.inf
        self.replay_triggered = 0

    def get_config(self) -> dict:
        """Get the config of the algorithm."""
        return {
            "method": self.method,
            "env_id": self.env.spec.id,
            "dimensions": self.dim,
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
            "seed": self.seed,
        }

    def setup(self) -> float:
        """Setup wandb."""
        if self.track:
            super().setup_wandb(
                project_name=self.wandb_project_name,
                experiment_name=self.exp_name,
                entity=self.wandb_entity,
                mode=self.wandb_mode,
            )

            wandb.define_metric("iteration")
            wandb.define_metric("outer/hypervolume", step_metric="iteration")
            wandb.define_metric("outer/dominated_hv", step_metric="iteration")
            wandb.define_metric("outer/discarded_hv", step_metric="iteration")
            wandb.define_metric("outer/coverage", step_metric="iteration")
            wandb.define_metric("outer/error", step_metric="iteration")
            self.run_id = wandb.run.id

        return time.time()

    def get_pareto_set(self, subsolutions: list[Subsolution]) -> list[tuple[np.ndarray, Any]]:
        """Get the Pareto set from the subsolutions."""
        pareto_set = []
        for subsolution in subsolutions:
            if np.any(np.all(np.isclose(subsolution[1], self.pf), axis=1)):
                pareto_set.append((self.sign * subsolution[1], subsolution[2]))
        return pareto_set

    def get_pareto_front(self) -> np.ndarray:
        """Get the Pareto front."""
        return self.pf * self.sign

    def finish(self, start_time: float, iteration: int):
        """Finish the algorithm."""
        self.pf = filter_pareto_dominated(np.vstack((self.pf, self.robust_points)))
        self.dominated_hv = self.compute_hypervolume(-self.sign * self.pf, -self.sign * self.nadir)
        self.hv = self.compute_hypervolume(-self.sign * self.pf, -self.sign * self.ref_point)
        self.log_iteration(iteration + 1)

        end_str = f"Iterations {iteration + 1} | Time {time.time() - start_time:.2f} | "
        end_str += f"HV {self.hv:.2f} | PF size {len(self.pf)} |"
        print(end_str)

        self.close_wandb()

    def close_wandb(self):
        """Close wandb."""
        if self.track:
            pf_table = wandb.Table(data=self.pf, columns=[f"obj_{i}" for i in range(self.dim)])
            wandb.log({"pareto_front": pf_table})
            wandb.run.summary["PF_size"] = len(self.pf)
            wandb.finish()

    def log_iteration(
        self, iteration: int, subproblem: Optional[Subproblem] = None, pareto_point: Optional[np.ndarray] = None
    ):
        """Log the iteration."""
        if self.track:
            while True:
                try:
                    wandb.log(
                        {
                            "outer/hypervolume": self.hv,
                            "outer/dominated_hv": self.dominated_hv,
                            "outer/discarded_hv": self.discarded_hv,
                            "outer/coverage": self.coverage,
                            "outer/error": self.error,
                            "iteration": iteration,
                        }
                    )
                    break
                except wandb.Error as e:
                    print(f"wandb got error {e}")
                    time.sleep(random.randint(10, 100))

            if subproblem is not None:
                wandb.run.summary[f"referent_{iteration}"] = self.sign * subproblem.referent
                wandb.run.summary[f"ideal_{iteration}"] = self.sign * subproblem.ideal
                wandb.run.summary[f"pareto_point_{iteration}"] = self.sign * pareto_point

            wandb.run.summary["hypervolume"] = self.hv
            wandb.run.summary["PF_size"] = len(self.pf)
            wandb.run.summary["replay_triggered"] = self.replay_triggered

    def compute_hypervolume(self, points: np.ndarray, ref: np.ndarray) -> float:
        """Compute the hypervolume of a set of points."""
        points = points[batched_pareto_dominates(ref, points)]
        if points.size == 0:
            return 0
        ind = Hypervolume(ref_point=ref)
        return ind(points)

    def init_phase(
        self,
        extrema: Optional[tuple[np.ndarray, np.ndarray]] = None,
        deterministic: bool = False,
        eval_env: Optional[gym.Env] = None,
    ) -> tuple[list[Subsolution], bool]:
        """Initialize the outer loop."""
        raise NotImplementedError

    def is_done(self, step: int) -> bool:
        """Check if the algorithm is done."""
        return 1 - self.coverage <= self.tolerance or step >= self.max_iterations

    def decompose_problem(self, iteration: int, method: str = "first") -> Subproblem:
        """Decompose the problem into a subproblem."""
        raise NotImplementedError

    def update_found(self, subproblem: Subproblem, vec: np.ndarray):
        """The update that is called when a Pareto optimal solution is found."""
        raise NotImplementedError

    def update_not_found(self, subproblem: Subproblem, vec: np.ndarray):
        """The update that is called when no Pareto optimal solution is found."""
        raise NotImplementedError

    def update_excluded_volume(self):
        """Update the dominated and infeasible sets."""
        raise NotImplementedError

    def estimate_error(self):
        """Estimate the error of the algorithm."""
        raise NotImplementedError

    def get_iterable_for_replay(self) -> Iterable[Any]:
        """Get an iterable for replaying the algorithm."""
        raise NotImplementedError

    def maybe_add_solution(
        self,
        subproblem: Subproblem,
        vec: np.ndarray,
        item: Any,
    ) -> Subproblem | bool:
        """Check and add a new solution to the Pareto front if possible."""
        raise NotImplementedError

    def maybe_add_completed(
        self,
        subproblem: Subproblem,
        vec: np.ndarray,
        item: Any,
    ) -> Subproblem | bool:
        """Check and add to the completed set if possible."""
        raise NotImplementedError

    def replay(self, vec: np.ndarray, sol: Any, iter_pairs: list[Subsolution]) -> list[Subsolution]:
        """Replay the algorithm while accounting for the non-optimal Pareto oracle."""
        replay_triggered = self.replay_triggered
        nadir, ideal = self.nadir, self.ideal
        self.reset()
        self.replay_triggered = replay_triggered + 1
        self.init_phase(extrema=(nadir, ideal), eval_env=None)
        idx = 0
        new_subsolutions = []

        for old_subproblem, old_vec, old_sol in iter_pairs:  # Replay the points that were added correctly
            idx += 1
            if strict_pareto_dominates(old_vec, old_subproblem.referent):
                if strict_pareto_dominates(vec, old_vec):
                    self.update_found(old_subproblem, vec)
                    new_subsolutions.append((old_subproblem, vec, sol))
                    break
                else:
                    self.update_found(old_subproblem, old_vec)
                    new_subsolutions.append((old_subproblem, old_vec, old_sol))
            else:
                if strict_pareto_dominates(vec, old_subproblem.referent):
                    self.update_found(old_subproblem, vec)
                    new_subsolutions.append((old_subproblem, vec, sol))
                    break
                else:
                    self.update_not_found(old_subproblem, old_vec)
                    new_subsolutions.append((old_subproblem, old_vec, old_vec))

        for old_subproblem, old_vec, old_sol in iter_pairs[
            idx:
        ]:  # Process the remaining points to see if we can still add them.
            items = self.get_iterable_for_replay()
            if strict_pareto_dominates(old_vec, old_subproblem.referent):
                maybe_add = self.maybe_add_solution
            else:
                maybe_add = self.maybe_add_completed
            for item in items:
                res = maybe_add(old_subproblem, old_vec, item)
                if res:
                    new_subsolutions.append((res, old_vec, old_sol))
                    break

        return new_subsolutions

    def eval(self, obs, disc_vec_return, pref=None):
        """Evaluate policy action for a given observation."""
        return self.agent.eval(obs, disc_vec_return, pref=pref)

    def linear_train(
        self,
        weight_vec: np.ndarray,
        deterministic: bool,
        eval_env: gym.Env,
    ) -> tuple[np.ndarray, Any]:
        """Train the agent using linear scalarization."""
        if self.reset_agent:
            self.agent.reset_agent(pref_dim=self.dim)

        weights = torch.tensor(weight_vec, device=self.device, dtype=torch.float32)
        u_func = partial(linear_scalarization, weights=weights)
        vec = self.agent.train(eval_env, u_func, pref=weights, deterministic=deterministic)
        return vec, self.agent.agent

    def oracle_train(
        self,
        referent: np.ndarray,
        deterministic: bool,
        eval_env: gym.Env,
    ) -> tuple[np.ndarray, Any]:
        """Train the agent using the Augmented Achievement Scalarizing Function (AASF)."""
        if self.reset_agent:
            self.agent.reset_agent(pref_dim=self.dim)

        referent = self.sign * torch.tensor(referent, device=self.device, dtype=torch.float32)
        nadir = self.sign * torch.tensor(self.nadir, device=self.device, dtype=torch.float32)
        ideal = self.sign * torch.tensor(self.ideal, device=self.device, dtype=torch.float32)

        u_func = partial(aasf, referent=referent, nadir=nadir, ideal=ideal, aug=self.aug, scale=self.scale)
        vec = self.agent.train(eval_env, u_func, pref=referent, deterministic=deterministic)
        vec *= self.sign
        return vec, self.agent.agent

    def train(
        self,
        eval_env: gym.Env,
        ref_point: np.ndarray,
        deterministic: bool = False,
        extrema: Optional[tuple[np.ndarray, np.ndarray]] = None,
        callback: Optional[IPROCallback] = None,
    ) -> list[tuple[np.ndarray, Any]]:
        """Solve the problem."""
        self.ref_point = ref_point

        start = self.setup()
        linear_subsolutions, done = self.init_phase(extrema=extrema, deterministic=deterministic, eval_env=eval_env)
        iteration = 0

        if done:
            print("The problem is solved in the initial phase.")
            pareto_set = self.get_pareto_set(linear_subsolutions)
            return pareto_set

        self.log_iteration(iteration)
        subsolutions = []

        while not self.is_done(iteration):
            begin_loop = time.time()
            print(f"Iter {iteration} - Covered {self.coverage:.5f}% - Error {self.error:.5f}")

            subproblem = self.decompose_problem(iteration)
            vec, sol = self.oracle_train(
                referent=subproblem.referent,
                deterministic=deterministic,
                eval_env=eval_env,
            )

            if strict_pareto_dominates(vec, subproblem.referent):
                if np.any(batched_strict_pareto_dominates(vec, np.vstack((self.pf, self.completed)))):
                    subsolutions = self.replay(vec, sol, subsolutions)
                else:
                    self.update_found(subproblem, vec)
                    subsolutions.append((subproblem, vec, sol))
            else:
                if np.any(batched_strict_pareto_dominates(vec, self.completed)):
                    subsolutions = self.replay(vec, sol, subsolutions)
                else:
                    self.update_not_found(subproblem, vec)
                    subsolutions.append((subproblem, vec, sol))

            self.update_excluded_volume()
            self.estimate_error()
            self.coverage = (self.dominated_hv + self.discarded_hv) / self.total_hv
            self.hv = self.compute_hypervolume(-self.sign * self.pf, -self.sign * self.ref_point)

            iteration += 1
            self.log_iteration(iteration, subproblem=subproblem, pareto_point=vec)

            if callback is not None:
                callback(iteration, self.hv, self.dominated_hv, self.discarded_hv, self.coverage, self.error)

            duration = time.time() - begin_loop
            print(f"Ref {self.sign * subproblem.referent} - Found {self.sign * vec} - Time {duration:.2f}s")
            print("---------------------")

        self.finish(start, iteration)
        pareto_set = self.get_pareto_set(linear_subsolutions + subsolutions)
        return pareto_set
