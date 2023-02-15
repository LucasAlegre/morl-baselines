"""Pareto Conditioned Network. Code adapted from https://github.com/mathieu-reymond/pareto-conditioned-networks"""
import os
import heapq
from dataclasses import dataclass
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.performance_indicators import hypervolume


def crowding_distance(points):
    # first normalize across dimensions
    points = (points - points.min(axis=0)) / (points.ptp(axis=0) + 1e-8)
    # sort points per dimension
    dim_sorted = np.argsort(points, axis=0)
    point_sorted = np.take_along_axis(points, dim_sorted, axis=0)
    # compute distances between lower and higher point
    distances = np.abs(point_sorted[:-2] - point_sorted[2:])
    # pad extrema's with 1, for each dimension
    distances = np.pad(distances, ((1,), (0,)), constant_values=1)
    # sum distances of each dimension of the same point
    crowding = np.zeros(points.shape)
    crowding[dim_sorted, np.arange(points.shape[-1])] = distances
    crowding = np.sum(crowding, axis=-1)
    return crowding


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminal: bool


def get_non_dominated(solutions):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1
    return is_efficient


def compute_hypervolume(q_set, ref):
    nA = len(q_set)
    q_values = np.zeros(nA)
    for i in range(nA):
        points = np.array(q_set[i])
        hv = hypervolume(ref, points)
        # use negative ref-point for minimization
        q_values[i] = hv
    return q_values


def run_episode(env, agent, desired_return, desired_horizon, max_return):
    transitions = []
    obs, _ = env.reset()
    done = False
    while not done:
        action = agent.act(obs, desired_return, desired_horizon)
        n_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        transitions.append(
            Transition(observation=obs, action=action, reward=np.float32(reward).copy(), next_observation=n_obs, terminal=terminated)
        )

        obs = n_obs
        # clip desired return, to return-upper-bound,
        # to avoid negative returns giving impossible desired returns
        desired_return = np.clip(desired_return - reward, None, max_return, dtype=np.float32)
        # clip desired horizon to avoid negative horizons
        desired_horizon = np.float32(max(desired_horizon - 1, 1.0))
    return transitions


def evaluate(env, agent, max_return, gamma=1.0, n=10):
    episodes = nlargest(n, agent.experience_replay)
    returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
    returns = np.float32(returns)
    horizons = np.float32(horizons)
    e_returns = []
    for i in range(n):
        transitions = run_episode(env, agent, returns[i], np.float32(horizons[i] - 2), max_return)
        # compute return
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].reward += gamma * transitions[i + 1].reward
        e_returns.append(transitions[0].reward)

    e_returns = np.array(e_returns)
    distances = np.linalg.norm(np.array(returns) - e_returns, axis=-1)
    return e_returns, np.array(returns), distances


def nlargest(n, experience_replay, threshold=0.2):
    returns = np.array([e[2][0].reward for e in experience_replay])
    # crowding distance of each point, check ones that are too close together
    distances = crowding_distance(returns)
    sma = np.argwhere(distances <= threshold).flatten()

    non_dominated_i = get_non_dominated(returns)
    non_dominated = returns[non_dominated_i]
    # we will compute distance of each point with each non-dominated point,
    # duplicate each point with number of non_dominated to compute respective distance
    returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(non_dominated), 1))
    # distance to closest non_dominated point
    l2 = np.min(np.linalg.norm(returns_exp - non_dominated, axis=-1), axis=-1) * -1
    # all points that are too close together (crowding distance < threshold) get a penalty
    non_dominated_i = np.nonzero(non_dominated_i)[0]
    _, unique_i = np.unique(non_dominated, axis=0, return_index=True)
    unique_i = non_dominated_i[unique_i]
    duplicates = np.ones(len(l2), dtype=bool)
    duplicates[unique_i] = False
    l2[duplicates] -= 1e-5
    l2[sma] *= 2

    sorted_i = np.argsort(l2)
    largest = [experience_replay[i] for i in sorted_i[-n:]]
    # before returning largest elements, update all distances in heap
    for i in range(len(l2)):
        experience_replay[i] = (l2[i], experience_replay[i][1], experience_replay[i][2])
    heapq.heapify(experience_replay)
    return largest


class Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, scaling_factor: np.ndarray, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.scaling_factor = nn.Parameter(th.tensor(scaling_factor).float(), requires_grad=False)
        self.hidden_dim = hidden_dim

        self.s_emb = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(self.reward_dim + 1, self.hidden_dim), nn.Sigmoid())
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.LogSoftmax(1),
        )

    def forward(self, state, desired_return, desired_horizon):
        c = th.cat((desired_return, desired_horizon), dim=-1)
        # commands are scaled by a fixed factor
        c = c * self.scaling_factor
        s = self.s_emb(state.float())
        c = self.c_emb(c)
        # element-wise multiplication of state-embedding and command
        log_prob = self.fc(s * c)
        return log_prob


class PCN(MOAgent, MOPolicy):
    """Pareto Conditioned Networks (PCN)

    Reymond, M., Bargiacchi, E., & Nowé, A. (2022, May). Pareto Conditioned Networks.
    In Proceedings of the 21st International Conference on Autonomous Agents
    and Multiagent Systems (pp. 1110-1118).
    https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1110.pdf

    ## Credits

    This code is a refactor of the code from the authors of the paper, available at:
    https://github.com/mathieu-reymond/pareto-conditioned-networks
    """

    def __init__(
        self,
        env: Optional[gym.Env],
        scaling_factor: np.ndarray,
        learning_rate: float = 1e-2,
        gamma: float = 1.0,
        batch_size: int = 32,
        hidden_dim: int = 64,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "PCN",
        log: bool = False,
        device: Union[th.device, str] = "auto",
    ) -> None:
        MOAgent.__init__(self, env, device=device)
        MOPolicy.__init__(self, device)

        self.experience_replay = []
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.scaling_factor = scaling_factor

        self.model = Model(
            self.observation_dim, self.action_dim, self.reward_dim, self.scaling_factor, hidden_dim=self.hidden_dim
        ).to(self.device)
        self.opt = th.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.log = log
        if log:
            self.setup_wandb(project_name, experiment_name)

    def get_config(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "scaling_factor": self.scaling_factor,
        }

    def update(self) -> None:
        batch = []
        # randomly choose episodes from experience buffer
        s_i = np.random.choice(np.arange(len(self.experience_replay)), size=self.batch_size, replace=True)
        for i in s_i:
            # episode is tuple (return, transitions)
            ep = self.experience_replay[i][2]
            # choose random timestep from episode,
            # use it's return and leftover timesteps as desired return and horizon
            t = np.random.randint(0, len(ep))
            # reward contains return until end of episode
            s_t, a_t, r_t, h_t = ep[t].observation, ep[t].action, np.float32(ep[t].reward), np.float32(len(ep) - t)
            batch.append((s_t, a_t, r_t, h_t))

        obs, actions, desired_return, desired_horizon = zip(*batch)
        log_prob = self.model(
            th.tensor(obs).to(self.device),
            th.tensor(desired_return).to(self.device),
            th.tensor(desired_horizon).unsqueeze(1).to(self.device),
        )

        self.opt.zero_grad()
        # one-hot of action for CE loss
        actions = F.one_hot(th.tensor(actions).long().to(self.device), len(log_prob[0]))
        # cross-entropy loss
        l = th.sum(-actions * log_prob, -1)
        l = l.mean()
        l.backward()
        self.opt.step()

        return l, log_prob

    def add_episode(self, transitions, max_size: int, step: int) -> None:
        # compute return
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].reward += self.gamma * transitions[i + 1].reward
        # pop smallest episode of heap if full, add new episode
        # heap is sorted by negative distance, (updated in nlargest)
        # put positive number to ensure that new item stays in the heap
        if len(self.experience_replay) == max_size:
            heapq.heappushpop(self.experience_replay, (1, step, transitions))
        else:
            heapq.heappush(self.experience_replay, (1, step, transitions))

    def choose_commands(self, num_episodes: int):
        # get best episodes, according to their crowding distance
        episodes = nlargest(num_episodes, self.experience_replay)
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        # keep only non-dominated returns
        nd_i = get_non_dominated(np.array(returns))
        returns = np.array(returns)[nd_i]
        horizons = np.array(horizons)[nd_i]
        # pick random return from random best episode
        r_i = np.random.randint(0, len(returns))
        desired_horizon = np.float32(horizons[r_i] - 2)
        # mean and std per objective
        _, s = np.mean(returns, axis=0), np.std(returns, axis=0)
        # desired return is sampled from [M, M+S], to try to do better than mean return
        desired_return = returns[r_i].copy()
        # random objective
        r_i = np.random.randint(0, len(desired_return))
        desired_return[r_i] += np.random.uniform(high=s[r_i])
        desired_return = np.float32(desired_return)
        return desired_return, desired_horizon

    def act(self, obs: np.ndarray, desired_return, desired_horizon) -> int:
        log_probs = self.model(
            th.tensor([obs]).to(self.device),
            th.tensor([desired_return]).to(self.device),
            th.tensor([desired_horizon]).unsqueeze(1).to(self.device),
        )
        log_probs = log_probs.detach().cpu().numpy()[0]
        action = np.random.choice(np.arange(len(log_probs)), p=np.exp(log_probs))
        return action

    def eval(self, obs):
        pass

    def train(
        self,
        env: gym.Env,
        num_er_episodes: int = 500,
        total_time_steps: int = 1e7,
        num_step_episodes: int = 10,
        num_model_updates: int = 100,
        max_return: float = 250.0,
        max_buffer_size: int = 500,
        ref_point: np.ndarray = np.array([0.0, 0.0]),
    ):
        self.global_step = 0
        total_episodes = num_er_episodes
        n_checkpoints = 0

        # fill buffer with random episodes
        self.experience_replay = []
        for _ in range(num_er_episodes):
            transitions = []
            obs, _ = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                n_obs, reward, terminated, truncated, _ = env.step(action)
                transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, terminated))
                done = terminated or truncated
                obs = n_obs
                self.global_step += 1
            # add episode in-place
            self.add_episode(transitions, max_size=max_buffer_size, step=self.global_step)

        while self.global_step < total_time_steps:
            loss = []
            entropy = []
            for _ in range(num_model_updates):
                l, lp = self.update()
                loss.append(l.detach().cpu().numpy())
                lp = lp.detach().cpu().numpy()
                ent = np.sum(-np.exp(lp) * lp)
                entropy.append(ent)

            desired_return, desired_horizon = self.choose_commands(num_er_episodes)

            # get all leaves, contain biggest elements, experience_replay got heapified in choose_commands
            leaves_r = np.array([e[2][0].reward for e in self.experience_replay[len(self.experience_replay) // 2 :]])
            # leaves_h = np.array([len(e[2]) for e in self.experience_replay[len(self.experience_replay) // 2 :]])
            try:
                # if len(self.experience_replay) == max_buffer_size:
                #    logger.put('train/leaves/r', leaves_r, self.global_step, f'{leaves_r.shape[-1]}d')
                #    logger.put('train/leaves/h', leaves_h, self.global_step, f'{leaves_h.shape[-1]}d')
                hv = hypervolume(ref_point, leaves_r)
                hv_est = hv
                self.writer.add_scalar("train/hypervolume", hv_est, self.global_step)
            except ValueError:
                pass

            returns = []
            horizons = []
            for _ in range(num_step_episodes):
                transitions = run_episode(env, self, desired_return, desired_horizon, max_return)
                self.global_step += len(transitions)
                self.add_episode(transitions, max_size=max_buffer_size, step=self.global_step)
                returns.append(transitions[0].reward)
                horizons.append(len(transitions))

            total_episodes += num_step_episodes
            self.writer.add_scalar("train/episode", total_episodes, self.global_step)
            self.writer.add_scalar("train/loss", np.mean(loss), self.global_step)
            self.writer.add_scalar("train/entropy", np.mean(entropy), self.global_step)
            self.writer.add_scalar("train/horizon/desired", desired_horizon, self.global_step)
            self.writer.add_scalar(
                "train/horizon/distance", np.linalg.norm(np.mean(horizons) - desired_horizon), self.global_step
            )

            for o in range(len(desired_return)):
                self.writer.add_scalar(f"train/return/{o}/value", desired_horizon, self.global_step)
                self.writer.add_scalar(f"train/return/{o}/desired", np.mean(np.array(returns)[:, o]), self.global_step)
                self.writer.add_scalar(
                    f"train/return/{o}/distance",
                    np.linalg.norm(np.mean(np.array(returns)[:, o]) - desired_return[o]),
                    self.global_step,
                )
            print(
                f"step {self.global_step} \t return {np.mean(returns, axis=0)}, ({np.std(returns, axis=0)}) \t loss {np.mean(loss):.3E}"
            )

            if self.global_step >= (n_checkpoints + 1) * total_time_steps / 100:
                if not os.path.isdir("weights"):
                    os.makedirs("weights")
                th.save(self.model, f"weights/model_{n_checkpoints+1}.pt")
                n_checkpoints += 1

                e_r, e_dr, e_d = evaluate(env, self, max_return, gamma=self.gamma)
                s = "desired return vs evaluated return\n" + 33 * "=" + "\n"
                for i in range(len(e_r)):
                    s += f"{e_dr[i]}  \t  {e_r[i]}  \n"

                for o in range(len(desired_return)):
                    self.writer.add_scalar(f"eval/return/{o}/desired", e_dr[o], self.global_step)
                    self.writer.add_scalar(f"eval/return/{o}/value", e_r[o], self.global_step)
                    self.writer.add_scalar(f"eval/return/{o}/distance", e_d[o], self.global_step)
