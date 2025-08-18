"""Non-linear Multi-Objective Proximal Policy Optimization (NLMOPPO) agent."""

import time
from math import ceil
from typing import Callable, Literal, Optional, Union

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.distributions.categorical import Categorical

from morl_baselines.common.morl_algorithm import MOPolicy


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialise a layer with orthogonal weights and constant bias."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Neural network agent with actor & critic.

    Observations are always augmented with accrued reward.
    Preference vector is included if provided; otherwise a zero-vector is used
    (keeping input dimensionality fixed).
    """

    def __init__(self, envs, num_objectives: int, pref_dim: int):
        """Initialise the agent network."""
        super().__init__()
        self.num_objectives = num_objectives
        self.pref_dim = pref_dim  # usually == num_objectives; may be 0 if unused

        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        in_dim = obs_dim + num_objectives + pref_dim  # obs || accrued_reward || pref(=zeros if None)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_objectives), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def _build_aug_obs(self, x, acc_reward, pref):
        x = x.float()
        acc_reward = acc_reward.float().to(x.device)

        # Ensure batch shape consistency:
        batched = x.ndim >= 2
        if not batched:
            # keep everything 1D
            parts = [x, acc_reward]
            if self.pref_dim > 0:
                if pref is None:
                    parts.append(torch.zeros(self.pref_dim, device=x.device, dtype=x.dtype))
                else:
                    parts.append(pref.float().to(x.device).view(-1))  # 1D
            return torch.cat(parts, dim=-1)

        # batched: [B, ...]
        if acc_reward.ndim == 1:
            acc_reward = acc_reward.expand(x.shape[0], -1)
        parts = [x, acc_reward]
        if self.pref_dim > 0:
            if pref is None:
                parts.append(torch.zeros((x.shape[0], self.pref_dim), device=x.device, dtype=x.dtype))
            else:
                pref = pref.float().to(x.device)
                if pref.ndim == 1:
                    pref = pref.expand(x.shape[0], -1)
                elif pref.shape[0] != x.shape[0]:
                    pref = pref.expand(x.shape[0], -1)
                parts.append(pref)
        return torch.cat(parts, dim=-1)

    def get_value(self, x, acc_reward, pref=None):
        """Get the value of the state."""
        aug = self._build_aug_obs(x, acc_reward, pref)
        return self.critic(aug)  # [B, D]

    def get_action_and_value(self, x, acc_reward, action=None, pref=None):
        """Get action and value for the given state."""
        aug = self._build_aug_obs(x, acc_reward, pref)
        logits = self.actor(aug)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(aug)

    def get_greedy_action(self, x, acc_reward, pref=None):
        """Get the greedy action for the given state."""
        aug = self._build_aug_obs(x, acc_reward, pref)
        logits = self.actor(aug)
        return torch.argmax(logits, dim=-1)


class NLMOPPO(MOPolicy):
    """Non-linear Multi-Objective Proximal Policy Optimization (NLMOPPO) agent."""

    def __init__(
        self,
        id: int,
        envs: mo_gym.wrappers.vector.MOSyncVectorEnv,
        log: bool = False,
        experiment_name: Optional[str] = "NLMOPPO",
        wandb_project_name: str = "MORL-Baselines",
        wandb_entity: str = None,
        wandb_mode: Literal["online", "offline", "disabled"] = "online",
        total_timesteps: int = 500000,
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
        seed: int = 1,
        rng: Union[np.random.Generator, None] = None,
    ):
        """Initialize the NLMOPPO agent.

        Args:
            id (int): Unique identifier for the agent.
            envs (mo_gym.wrappers.vector.MOSyncVectorEnv): Environment wrapper for multi-objective tasks.
            log (bool): Whether to log training metrics to Weights & Biases.
            experiment_name (str): Name of the experiment for logging.
            wandb_project_name (str): Weights & Biases project name.
            wandb_entity (str): Weights & Biases entity name.
            wandb_mode (Literal["online", "offline", "disabled"]): Logging mode for Weights & Biases.
            total_timesteps (int): Total number of timesteps to train the agent.
            learning_rate (float): Learning rate for the optimizer.
            num_steps (int): Number of steps per update.
            anneal_lr (bool): Whether to anneal the learning rate during training.
            gamma (float): Discount factor for future rewards.
            gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
            num_minibatches (int): Number of minibatches per update.
            update_epochs (int): Number of epochs to update the policy per batch.
            norm_adv (bool): Whether to normalize advantages.
            clip_coef (float): Clipping coefficient for PPO.
            clip_vloss (bool): Whether to clip value loss.
            ent_coef (float): Coefficient for entropy loss.
            vf_coef (float): Coefficient for value function loss.
            max_grad_norm (float): Maximum gradient norm for clipping.
            target_kl (Optional[float]): Target KL divergence for early stopping.
            mc_k (int): Number of Monte Carlo samples for utility-gradient evaluation.
            device: Device to run the model on ("auto" uses CUDA if available).
            seed: Random seed for reproducibility.
            rng: Optional NumPy random generator instance.
        """
        super().__init__(id, device)

        self.envs = envs
        self.seed = seed
        self.rng = rng or np.random.default_rng(seed)
        self.log = log
        self.experiment_name = experiment_name
        self.wandb_project_name = wandb_project_name
        self.wandb_entity = wandb_entity
        self.wandb_mode = wandb_mode
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.num_envs = self.envs.num_envs
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # Runtime derived quantities
        self.num_objectives = self.envs.reward_space.shape[0]
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = self.total_timesteps // self.batch_size

        # Seed observations for utility-gradient eval
        self.init_obs = torch.as_tensor(
            np.concatenate([envs.reset(seed=self.seed + i)[0] for i in range(ceil(mc_k / self.num_envs))])[:mc_k],
            device=self.device,
            dtype=torch.float32,
        )

        # Preference initially unknown
        self.pref: Union[torch.Tensor, None] = None

        self.reset_agent(pref_dim=self.num_objectives)  # default pref_dim; zeros used when pref=None
        self._setup_storage()

    def reset_agent(self, pref_dim: int):
        """Initialize the agent and optimizer and reset the utility function and preference vector."""
        self.agent = Agent(self.envs, self.num_objectives, pref_dim=pref_dim).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        self.u_func = None
        self.pref = None

    def _setup_storage(self):
        shp_obs = (self.num_steps, self.num_envs) + self.envs.single_observation_space.shape
        self.obs = torch.zeros(shp_obs, device=self.device, dtype=torch.float32)

        self.acc_rewards = torch.zeros(
            (self.num_steps, self.num_envs, self.num_objectives), device=self.device, dtype=torch.float32
        )
        self.actions = torch.zeros(
            (self.num_steps, self.num_envs) + self.envs.single_action_space.shape, device=self.device, dtype=torch.long
        )
        self.logprobs = torch.zeros((self.num_steps, self.num_envs), device=self.device, dtype=torch.float32)
        self.rewards = torch.zeros(
            (self.num_steps, self.num_envs, self.num_objectives), device=self.device, dtype=torch.float32
        )
        self.dones = torch.zeros((self.num_steps, self.num_envs), device=self.device, dtype=torch.float32)
        self.values = torch.zeros(
            (self.num_steps, self.num_envs, self.num_objectives), device=self.device, dtype=torch.float32
        )

        self.returns = None
        self.advantages = None

    def _collect_rollouts(self, next_obs, next_acc_reward, next_done, timestep, global_step):
        for step in range(self.num_steps):
            global_step += self.num_envs
            self.obs[step] = next_obs
            self.acc_rewards[step] = next_acc_reward
            self.dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(
                    next_obs, acc_reward=next_acc_reward, pref=self.pref
                )
                self.values[step] = value

            self.actions[step] = action
            self.logprobs[step] = logprob

            next_np_action = action.detach().cpu().numpy()
            next_obs_np, reward_np, term_np, trunc_np, infos = self.envs.step(next_np_action)
            next_done_np = np.logical_or(term_np, trunc_np)

            self.rewards[step] = torch.as_tensor(reward_np, device=self.device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=self.device, dtype=torch.float32)
            next_done = torch.as_tensor(next_done_np, device=self.device, dtype=torch.float32)

            # Accrued discounted reward per env
            # timestep: [N] int32 ; next_done: [N] float
            next_acc_reward = (next_acc_reward + (self.gamma**timestep) * self.rewards[step]) * (1.0 - next_done.unsqueeze(-1))
            timestep = (timestep + 1) * (1 - next_done.int().unsqueeze(-1))

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info and self.log:
                        wandb.log(
                            {
                                "charts/episodic_return": info["episode"]["r"],
                                "charts/episodic_length": info["episode"]["l"],
                            },
                            step=global_step,
                        )

        return next_obs, next_acc_reward, next_done, timestep, global_step

    def _compute_advantages_and_returns(self, next_obs, next_acc_reward, next_done):
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs, next_acc_reward, self.pref).reshape(
                1, self.num_envs, self.num_objectives
            )  # [1, N, D]
            advantages = torch.zeros_like(self.rewards, device=self.device)  # [T, N, D]
            lastgaelam = torch.zeros((self.num_envs, self.num_objectives), device=self.device)
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = (1.0 - next_done).unsqueeze(-1)  # [N, 1]
                    nextvalues = next_value[0]  # [N, D]
                else:
                    nextnonterminal = (1.0 - self.dones[t + 1]).unsqueeze(-1)  # [N, 1]
                    nextvalues = self.values[t + 1]  # [N, D]
                delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]  # [N, D]
                lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam  # [N, D]
                advantages[t] = lastgaelam
            returns = advantages + self.values  # [T, N, D]
        return advantages, returns

    def _compute_loss_weights(self) -> torch.Tensor:
        """Compute the loss weights for each objective according to the non-linear PG theorem.

        w = du/dv evaluated at v^pi(s0).
        Use zero accrued reward at s0 for the init_obs (common choice).
        """
        B = self.init_obs.shape[0]
        zero_acc = torch.zeros((B, self.num_objectives), device=self.device, dtype=torch.float32)
        v0 = (
            self.agent.get_value(self.init_obs, acc_reward=zero_acc, pref=self.pref).mean(0).detach().requires_grad_(True)
        )  # [D]
        u = self.u_func(v0)  # scalar
        (w,) = torch.autograd.grad(u, v0, retain_graph=False, create_graph=False)
        return w.detach()  # [D]

    def update(self):
        """Update the policy using PPO algorithm."""
        loss_weights = self._compute_loss_weights()  # [D]

        # Flatten batches
        b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)  # [B]
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = self.advantages.reshape(self.batch_size, self.num_objectives)  # [B, D]
        b_returns = self.returns.reshape(self.batch_size, self.num_objectives)  # [B, D]
        b_values = self.values.reshape(self.batch_size, self.num_objectives)  # [B, D]
        b_acc_rewards = self.acc_rewards.reshape(self.batch_size, self.num_objectives)  # [B, D]
        b_prefs = self.pref.expand(self.batch_size, -1) if self.pref is not None else None  # [B, Dp] or None

        b_inds = np.arange(self.batch_size)
        clipfracs = []
        approx_kl = torch.tensor(0.0)
        old_approx_kl = torch.tensor(0.0)

        for epoch in range(self.update_epochs):
            self.rng.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds],
                    acc_reward=b_acc_rewards[mb_inds],
                    action=b_actions.long()[mb_inds],
                    pref=b_prefs[mb_inds] if b_prefs is not None else None,
                )  # newvalue: [mb, D]

                logratio = newlogprob - b_logprobs[mb_inds]  # [mb]
                ratio = logratio.exp()  # [mb]

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

                # Per-objective advantages
                mb_adv_vec = b_advantages[mb_inds]  # [mb, D]
                if self.norm_adv:
                    mean = mb_adv_vec.mean(dim=0, keepdim=True)
                    std = mb_adv_vec.std(dim=0, keepdim=True) + 1e-8
                    mb_adv_vec = (mb_adv_vec - mean) / std

                # PPO surrogate per objective
                pg_loss1 = -mb_adv_vec * ratio.unsqueeze(-1)  # [mb, D]
                pg_loss2 = -mb_adv_vec * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef).unsqueeze(-1)
                per_obj_pg = torch.max(pg_loss1, pg_loss2).mean(dim=0)  # [D]
                pg_loss = (per_obj_pg * loss_weights).sum()

                # Value loss
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2  # [mb, D]
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -self.clip_coef, self.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2  # [mb, D]
                    # First take the max element-wise, then average everything
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        return v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, float(np.mean(clipfracs) if clipfracs else 0.0)

    def eval(self, obs, disc_vec_return, pref=None):
        """Evaluate policy action for a given observation."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        disc_vec_return = torch.as_tensor(disc_vec_return, device=self.device, dtype=torch.float32)
        pref = None if pref is None else torch.as_tensor(pref, device=self.device, dtype=torch.float32)
        return self.agent.get_action_and_value(obs, acc_reward=disc_vec_return, pref=pref)[0]

    def policy_evaluate(self, eval_env, eval_episodes=100, deterministic=False):
        """Evaluate the policy in a given environment."""
        if deterministic:

            def policy(o, a, p):
                """Deterministic policy: get greedy action."""
                return self.agent.get_greedy_action(o, acc_reward=a, pref=p)

        else:

            def policy(o, a, p):
                """Stochastic policy: get action and value."""
                return self.agent.get_action_and_value(o, acc_reward=a, pref=p)[0]

        pareto_point = np.zeros(self.num_objectives)

        for _ in range(eval_episodes):
            obs, _ = eval_env.reset(seed=self.seed)
            terminated = False
            truncated = False
            accrued_reward = np.zeros(self.num_objectives, dtype=np.float32)
            timestep = 0

            while not (terminated or truncated):
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                    acc_t = torch.as_tensor(accrued_reward, device=self.device, dtype=torch.float32)
                    action = policy(obs_t, acc_t, self.pref).item()
                next_obs, reward, terminated, truncated, _ = eval_env.step(action)
                accrued_reward += (self.gamma**timestep) * reward
                obs = next_obs
                timestep += 1

            pareto_point += accrued_reward

        return pareto_point / eval_episodes

    def train(
        self,
        eval_env: gym.Env,
        u_func: Callable[[torch.Tensor], torch.Tensor],
        pref: torch.Tensor = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Main training loop."""
        self.u_func = u_func
        self.pref = None if pref is None else torch.as_tensor(pref, device=self.device, dtype=torch.float32)

        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.seed)
        next_obs = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)
        next_acc_reward = torch.zeros((self.num_envs, self.num_objectives), dtype=torch.float32, device=self.device)
        next_done = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        timestep = torch.zeros((self.num_envs, 1), dtype=torch.int32, device=self.device)

        for iteration in range(1, self.num_iterations + 1):
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            next_obs, next_acc_reward, next_done, timestep, global_step = self._collect_rollouts(
                next_obs, next_acc_reward, next_done, timestep, global_step
            )
            self.advantages, self.returns = self._compute_advantages_and_returns(next_obs, next_acc_reward, next_done)
            v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfrac = self.update()

            sps = int(global_step / (time.time() - start_time))
            if self.log:
                wandb.log(
                    {
                        "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "losses/value_loss": float(v_loss.item()),
                        "losses/policy_loss": float(pg_loss.item()),
                        "losses/entropy": float(entropy_loss.item()),
                        "losses/old_approx_kl": float(old_approx_kl.item()),
                        "losses/approx_kl": float(approx_kl.item()),
                        "losses/clipfrac": clipfrac,
                        "charts/SPS": sps,
                    },
                    step=global_step,
                )

        return self.policy_evaluate(eval_env, deterministic=deterministic)
