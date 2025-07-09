"""Utility functions for the model."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as th
import torch.nn.functional as F
from gymnasium.spaces import Discrete


def termination_fn_false(obs, act, next_obs, rew):
    """Returns a vector of False values of the same length as the batch size."""
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == len(rew.shape) == 2
    done = np.array([False]).repeat(len(obs))
    done = done[:, np.newaxis]
    return done


def termination_fn_dst(obs, act, next_obs, rew):
    """Termination function of DST."""
    from mo_gymnasium.deep_sea_treasure.deep_sea_treasure import CONCAVE_MAP

    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == len(rew.shape) == 2
    done = np.array([False]).repeat(len(obs))
    next_obs_int = (next_obs * 10).astype(int)
    for i in range(len(done)):
        if next_obs_int[i, 0] < 0 or next_obs_int[i, 0] > 10 or next_obs_int[i, 1] < 0 or next_obs_int[i, 1] > 10:
            done[i] = False
        else:
            done[i] = CONCAVE_MAP[next_obs_int[i, 0]][next_obs_int[i, 1]] > 0.1
    done = done[:, np.newaxis]
    return done


def termination_fn_mountaincar(obs, act, next_obs, rew):
    """Termination function of mountain car."""
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == len(rew.shape) == 2
    position = next_obs[:, 0]
    velocity = next_obs[:, 1]
    done = (position >= 0.45) * (velocity >= 0.0)
    done = done[:, np.newaxis]
    return done


def termination_fn_minecart(obs, act, next_obs, rew):
    """Termination function of minecart."""
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == len(rew.shape) == 2
    old_pos = obs[:, 0:2]
    pos = next_obs[:, 0:2]
    # had_ore = (obs[:,-2] > 0) + (obs[:,-1] > 0)
    in_base = np.sqrt(np.einsum("ij,ij->i", pos, pos)) < 0.15
    was_out_base = np.sqrt(np.einsum("ij,ij->i", old_pos, old_pos)) >= 0.15
    done = was_out_base * in_base
    done = done[:, np.newaxis]
    return done


def termination_fn_hopper(obs, act, next_obs, rew):
    """Termination function of hopper."""
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (
        np.isfinite(next_obs).all(axis=-1)
        * np.abs(next_obs[:, 1:] < 100).all(axis=-1)
        * (height > 0.7)
        * (np.abs(angle) < 0.2)
    )
    done = ~not_done
    done = done[:, np.newaxis]
    return done


def termination_fn_lunarlander(obs, act, next_obs, rew):
    """Termination function of lunarlander. Use reward prediction to determine termination."""
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == len(rew.shape) == 2

    # Condition 1: out of screen
    has_exited_screen = abs(next_obs[:, 0]) >= 1.0

    # Condition 2: all legs have landed (supposed to be 1.0 but we allow for some margin of error) and reward is non-zero
    has_crashed_or_landed = (rew[:, 0] != 0) & (next_obs[:, 6] >= 0.95) & (next_obs[:, 7] >= 0.95)

    not_done = ~(has_exited_screen | has_crashed_or_landed)
    done = ~not_done
    done = done[:, np.newaxis]
    return done


def termination_fn_humanoid(obs, act, next_obs, rew):
    """Termination function of hopper."""
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == len(rew.shape) == 2
    min_z, max_z = 1.0, 2.0  # if u change healthy_z_range in the humanoid env, change this too

    # index needs to be +2 if you unset the exclude_current_positions_from_observation
    # parameter in the humanoid env
    not_done = (min_z < next_obs[:, 0]) & (next_obs[:, 0] < max_z)
    done = ~not_done
    done = done[:, np.newaxis]
    return done


class ModelEnv:
    """Wrapper for the model to be used as an environment."""

    def __init__(self, model, env_id=None, rew_dim=1):
        """Initialize the environment.

        Args:
            model: model to be used as an environment.
            env_id: environment id.
            rew_dim: reward dimension.
        """
        self.model = model
        self.rew_dim = rew_dim
        if "hopper" in env_id:
            self.termination_func = termination_fn_hopper
        elif "halfcheetah" in env_id:
            self.termination_func = termination_fn_false
        elif "humanoid" in env_id:
            self.termination_func = termination_fn_humanoid
        elif "lunar-lander" in env_id:
            self.termination_func = termination_fn_lunarlander
        elif "mo-reacher" in env_id:
            self.termination_func = termination_fn_false
        elif "mountaincar" in env_id:
            self.termination_func = termination_fn_mountaincar
        elif "minecart" in env_id:
            self.termination_func = termination_fn_minecart
        elif env_id == "mo-highway-fast-v0" or env_id == "mo-highway-v0":
            self.termination_func = termination_fn_false
        elif env_id == "deep-sea-treasure-v0":
            self.termination_func = termination_fn_dst
        else:
            raise NotImplementedError

    def step(
        self, obs: th.Tensor, act: th.Tensor, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Step the environment.

        Args:
            obs (th.Tensor): current bservation.
            act (th.Tensor): current action.
            deterministic (bool): whether to use deterministic model prediction.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: next observation, reward, terminals, info.
        """
        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
            act = act.unsqueeze(0)
            return_single = True
        else:
            return_single = False

        inputs = th.cat((obs, act), dim=-1).float().to(self.model.device)
        with th.no_grad():
            samples, vars, uncertainties = self.model.sample(inputs, deterministic=deterministic)

        obs = obs.detach().cpu().numpy()

        samples[:, self.rew_dim :] += obs

        rewards, next_obs = samples[:, : self.rew_dim], samples[:, self.rew_dim :]
        terminals = self.termination_func(obs, act, next_obs, rewards)
        var_rewards, var_obs = vars[:, : self.rew_dim], vars[:, self.rew_dim :]

        if return_single:
            next_obs = next_obs[0]
            rewards = rewards[0]
            terminals = terminals[0]
            uncertainties = uncertainties[0]
            var_obs = var_obs[0]
            var_rewards = var_rewards[0]

        info = {
            "uncertainty": uncertainties,
            "var_obs": var_obs,
            "var_rewards": var_rewards,
        }

        # info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info


def visualize_eval(
    agent,
    env,
    model=None,
    w=None,
    horizon=10,
    init_obs=None,
    compound=True,
    deterministic=False,
    show=False,
    filename=None,
):
    """Generates a plot of the evolution of the state, reward and model predictions over time.

    Args:
        agent: agent to be evaluated
        env: environment to be evaluated
        model: model to be evaluated
        w: weights to be used for the evaluation
        horizon: number of time steps
        init_obs: initial observation
        compound: whether to use compound model predictions
        deterministic: whether to use deterministic model predictions
        show: whether to show the plot
        filename: filename to save the plot

    Returns:
        plt: plt object with the figure
    """
    if init_obs is None:
        init_obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    actions = []
    real_obs = []
    real_rewards = []
    real_vec_rewards = []
    obs = init_obs.copy()
    for step in range(horizon):
        if w is not None:
            act = agent.eval(obs, w)
        else:
            act = agent.eval(obs)
        actions.append(act)
        obs, r, terminated, truncated, info = env.step(act)
        done = terminated or truncated
        real_obs.append(obs.copy())
        if type(r) is float:
            real_rewards.append(r)
        else:
            real_rewards.append(np.dot(r, w))
        if "vector_reward" in info:
            real_vec_rewards.append(info["vector_reward"])
        elif type(r) is np.ndarray:
            real_vec_rewards.append(r)
        if done:
            break

    model_obs = []
    model_obs_stds = []
    model_rewards_stds = []
    model_rewards = []
    if model is not None:
        obs = init_obs.copy()
        model_env = ModelEnv(model, env_id=env.unwrapped.spec.id, rew_dim=1 if w is None else len(w))
        acts = th.tensor(actions).to(agent.device)
        if isinstance(env.action_space, Discrete):
            acts = F.one_hot(acts, num_classes=env.action_space.n).squeeze(1)
        for step in range(len(real_obs)):
            if compound or step == 0:
                obs, r, done, info = model_env.step(
                    th.tensor(obs).to(agent.device),
                    acts[step],
                    deterministic=deterministic,
                )
            else:
                obs, r, done, info = model_env.step(
                    th.tensor(real_obs[step - 1]).to(agent.device),
                    acts[step],
                    deterministic=deterministic,
                )
            model_obs.append(obs.copy())
            model_obs_stds.append(np.sqrt(info["var_obs"].copy()))
            model_rewards_stds.append(np.sqrt(info["var_rewards"].copy()))
            model_rewards.append(r)
            # if done:
            #    break

    num_plots = obs_dim + (1 if w is None else len(w)) + 1
    num_cols = int(np.ceil(np.sqrt(num_plots)))
    num_rows = int(np.ceil(num_plots / num_cols))
    x = np.arange(0, len(real_obs))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 15))
    axs = np.array(axs).reshape(-1)
    for i in range(num_plots):
        if i == num_plots - 1:
            axs[i].set_ylabel("Action")
            axs[i].grid(alpha=0.25)
            axs[i].plot(x, [actions[step] for step in x], label="Action", color="orange")
        elif i >= obs_dim:
            axs[i].set_ylabel(f"Reward {i - obs_dim}")
            axs[i].grid(alpha=0.25)
            if w is not None:
                axs[i].plot(
                    x,
                    [real_vec_rewards[step][i - obs_dim] for step in x],
                    label="Environment",
                    color="black",
                )
            else:
                axs[i].plot(
                    x,
                    [real_rewards[step] for step in x],
                    label="Environment",
                    color="black",
                )
            if model is not None:
                axs[i].plot(
                    x,
                    [model_rewards[step][i - obs_dim] for step in x],
                    label="Model",
                    color="blue",
                )
                axs[i].fill_between(
                    x,
                    [model_rewards[step][i - obs_dim] + model_rewards_stds[step][i - obs_dim] for step in x],
                    [model_rewards[step][i - obs_dim] - model_rewards_stds[step][i - obs_dim] for step in x],
                    alpha=0.2,
                    facecolor="blue",
                )
        else:
            axs[i].set_ylabel(f"State {i}")
            axs[i].grid(alpha=0.25)
            axs[i].plot(x, [real_obs[step][i] for step in x], label="Environment", color="black")
            if model is not None:
                axs[i].plot(x, [model_obs[step][i] for step in x], label="Model", color="blue")
                axs[i].fill_between(
                    x,
                    [model_obs[step][i] + model_obs_stds[step][i] for step in x],
                    [model_obs[step][i] - model_obs_stds[step][i] for step in x],
                    alpha=0.2,
                    facecolor="blue",
                )
    sns.despine()
    if filename is not None:
        plt.savefig(filename + ".pdf", format="pdf", bbox_inches="tight")
    if show:
        plt.show()
    return plt
