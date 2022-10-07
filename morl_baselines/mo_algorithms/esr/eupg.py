from typing import List, Optional, Union

import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from mo_gym.evaluation import eval_mo_esr
from torch.distributions import Categorical

from morl_baselines.common.accrued_reward_buffer import AccruedRewardReplayBuffer
from morl_baselines.common.morl_algorithm import MOPolicy, MOAgent
from morl_baselines.common.networks import mlp
from torch.utils.tensorboard import SummaryWriter
from morl_baselines.common.utils import layer_init, log_episode_info


# EUPG is an ESR algorithm based on Policy Gradient (REINFORCE like)
# The idea is to condition the network on the accrued reward and to
# Scalarize the rewards based on the episodic return (accrued + future rewards)
# Paper: D. Roijers, D. Steckelmacher, and A. Nowe, Multi-objective Reinforcement Learning for the Expected Utility of the Return. 2018.

class PolicyNet(nn.Module):
    def __init__(self, obs_shape, action_dim, rew_dim, net_arch):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.rew_dim = rew_dim

        # Conditioned on accrued reward, so input takes reward
        input_dim = obs_shape[0] + rew_dim

        # |S|+|R| -> ... -> |A|
        self.net = mlp(input_dim, action_dim, net_arch, activation_fn=nn.Tanh)
        self.apply(layer_init)

    def forward(self, obs: th.Tensor, acc_reward: th.Tensor):
        if len(obs.size()) == 0 or len(acc_reward.size()) == 0:
            print(obs)
            print(acc_reward)
        input = th.cat((obs, acc_reward), dim=acc_reward.dim() - 1)
        pi = self.net(input)
        # Normalized sigmoid
        x_exp = th.sigmoid(pi)
        probas = x_exp / th.sum(x_exp)
        return probas.view(-1, self.action_dim)  # Batch Size x |Actions|

    def distribution(self, obs: th.Tensor, acc_reward: th.Tensor):
        probas = self.forward(obs, acc_reward)
        distribution = Categorical(probas)
        return distribution


class EUPG(MOPolicy, MOAgent):
    def __init__(self,
                 env: gym.Env,
                 scalarization,
                 buffer_size: int = int(1e5),
                 net_arch: List = [50],
                 gamma: float = 0.99,
                 learning_rate: float = 1e-3,
                 project_name: str = "MORL-Baselines",
                 experiment_name: str = "EUPG",
                 log: bool = True,
                 device: Union[th.device, str] = "auto",
                 ):
        MOAgent.__init__(self, env, device)
        MOPolicy.__init__(self, None, device)

        self.env = env
        # RL
        self.scalarization = scalarization
        self.gamma = gamma

        # Learning
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_rate = learning_rate
        self.buffer = AccruedRewardReplayBuffer(
            obs_shape=self.observation_shape,
            action_shape=self.action_shape,
            rew_dim=self.reward_dim,
            max_size=self.buffer_size,
            obs_dtype=np.int32,
            action_dtype=np.int32
        )
        self.net = PolicyNet(obs_shape=self.observation_shape, rew_dim=self.reward_dim, action_dim=self.action_dim,
                             net_arch=self.net_arch)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log
        if log:
            self.setup_wandb(project_name, experiment_name)

    def eval(self, obs: np.ndarray, accrued_reward: Optional[np.ndarray]) -> Union[int, np.ndarray]:
        if type(obs) is int:
            obs = th.as_tensor([obs]).to(self.device)
        else:
            obs = th.as_tensor(obs).to(self.device)
        accrued_reward = th.as_tensor(accrued_reward).float().to(self.device)
        return self.max_action(obs, accrued_reward)

    @th.no_grad()
    def max_action(self, obs: th.Tensor, accrued_reward: th.Tensor) -> int:
        probs = self.net(obs, accrued_reward)
        max_act = th.argmax(probs, dim=1)
        return max_act.detach().item()

    def update(self):
        obs, accrued_rewards, actions, rewards, next_obs, terminateds = self.buffer.get_all_data(to_tensor=True,
                                                                                                 device=self.device)
        # Scalarized episodic reward, our target :-)
        episodic_reward = th.sum(rewards, dim=0)
        scalarized_reward = self.scalarization(episodic_reward)

        current_distribution = self.net.distribution(obs, accrued_rewards)
        # Policy gradient
        log_probs = -current_distribution.log_prob(actions)
        loss = th.sum(log_probs * scalarized_reward)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.log:
            self.writer.add_scalar("losses/loss", loss, self.global_step)
            self.writer.add_scalar("metrics/scalarized_episodic_return", scalarized_reward, self.global_step)

    # TODO we should find a way to unify ESR, SER, multi-policy and single-policy into one. The problems are:
    #  some require conditioning on weights, some require the scalarization, some require conditioning on accrued
    #  reward
    def policy_eval(self, eval_env, writer: SummaryWriter):
        """
        Runs a policy evaluation (typically on one episode) on eval_env and logs some metrics using writer.
        :param eval_env: evaluation environment
        :param writer: wandb writer
        :return: a tuple containing the evaluations
        """

        scalarized_reward, scalarized_discounted_reward, vec_reward, discounted_vec_reward = eval_mo_esr(self, eval_env,
                                                                                                         self.scalarization)
        if self.id is None:
            idstr = ""
        else:
            idstr = f"_{self.id}"

        writer.add_scalar(f"eval{idstr}/scalarized_reward", scalarized_reward, self.global_step)
        writer.add_scalar(f"eval{idstr}/scalarized_discounted_reward", scalarized_discounted_reward, self.global_step)
        for i in range(vec_reward.shape[0]):
            writer.add_scalar(f"eval{idstr}/vec_{i}", vec_reward[i], self.global_step)
            writer.add_scalar(f"eval{idstr}/discounted_vec_{i}", discounted_vec_reward[i], self.global_step)

        return (
            scalarized_reward,
            scalarized_discounted_reward,
            vec_reward,
            discounted_vec_reward
        )

    def train(
            self,
            total_timesteps: int,
            eval_env: Optional[gym.Env] = None,
            eval_freq: int = 1000
    ):
        # Init
        obs, _, = self.env.reset()
        terminated = False
        accrued_reward_tensor = th.zeros(self.reward_dim, dtype=th.float32).float().to(self.device)

        # Training loop
        for _ in range(1, total_timesteps + 1):
            self.global_step += 1

            with th.no_grad():
                # For training, takes action randomly according to the policy
                action = self.net.distribution(th.Tensor([obs]).to(self.device), accrued_reward_tensor).sample().item()
            next_obs, vec_reward, terminated, _, info = self.env.step(action)

            # Memory update
            self.buffer.add(obs, accrued_reward_tensor, action, vec_reward, next_obs, terminated)
            accrued_reward_tensor += th.from_numpy(vec_reward).to(self.device)

            if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                self.policy_eval(eval_env, self.writer)

            if terminated:
                # NN is updated at the end of each episode
                self.update()
                obs, _ = self.env.reset()
                terminated = False
                self.num_episodes += 1
                accrued_reward_tensor = th.zeros(self.reward_dim).float().to(self.device)

                if self.log and "episode" in info.keys():
                    log_episode_info(info["episode"], self.scalarization, None, self.global_step, self.writer)

            else:
                obs = next_obs

    def get_config(self) -> dict:
        return {
            # "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
        }
