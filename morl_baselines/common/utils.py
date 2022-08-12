from os import waitid_result
from typing import Iterable, Optional

import numpy as np
import torch as th
from torch import nn


@th.no_grad()
def layer_init(layer, method="orthogonal", weight_gain: float = 1, bias_const: float = 0) -> None:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        if method == "xavier":
            th.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif method == "orthogonal":
            th.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        th.nn.init.constant_(layer.bias, bias_const)


@th.no_grad()
def polyak_update(
    params: Iterable[th.nn.Parameter],
    target_params: Iterable[th.nn.Parameter],
    tau: float,
) -> None:
    for param, target_param in zip(params, target_params):
        if tau == 1:
            target_param.data.copy_(param.data)
        else:
            target_param.data.mul_(1.0 - tau)
            th.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def get_grad_norm(params: Iterable[th.nn.Parameter]) -> th.Tensor:
    """This is how the grad norm is computed inside torch.nn.clip_grad_norm_()"""
    parameters = [p for p in params if p.grad is not None]
    if len(parameters) == 0:
        return th.tensor(0.)
    device = parameters[0].grad.device
    total_norm = th.norm(th.stack([th.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
    return total_norm


def huber(x, min_priority=0.01):
    return th.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).mean()


def linearly_decaying_value(initial_value, decay_period, step, warmup_steps, final_value):
    """Returns the current value for a linearly decaying parameter.
    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.
    Args:
    decay_period: float, the period over which the value is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before the value is decayed.
    final value: float, the final value to which to decay the value parameter.
    Returns:
    A float, the current value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (initial_value - final_value) * steps_left / decay_period
    value = final_value + bonus
    value = np.clip(value, min(initial_value, final_value), max(initial_value, final_value))
    return value


def random_weights(dim: int, seed: Optional[int] = None, n: int = 1, dist: str = 'gaussian') -> np.ndarray:
    """Generate random normalized weight vectors from a Gaussian or Dirichlet distribution alpha=1
    Args:
        dim: size of the weight vector
        seed: random seed
        n : number of weight vectors to generate
        dist: distribution to use, either 'gaussian' or 'dirichlet'
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random

    if dist == 'gaussian':
        w = np.random.randn(n, dim)
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1, keepdims=True)
    elif dist == 'dirichlet':
        w = rng.dirichlet(np.ones(dim), n)
    else:
        raise ValueError(f'Unknown distribution {dist}')

    if n == 1:
        return w[0]
    return w