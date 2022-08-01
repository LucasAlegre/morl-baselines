import os
import random
from typing import Iterable, List, Optional, Union

import numpy as np
import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from pymoo.factory import get_reference_directions


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


def linearly_decaying_epsilon(initial_epsilon, decay_period, step, warmup_steps, final_epsilon):
    """Returns the current epsilon for the agent's epsilon-greedy policy.
    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.
    Args:
    decay_period: float, the period over which epsilon is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    epsilon: float, the final value to which to decay the epsilon parameter.
    Returns:
    A float, the current epsilon value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (initial_epsilon - final_epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0.0, 1.0 - final_epsilon)
    return final_epsilon + bonus


def get_weights(num_objs: int, num_weights: int) -> List:
    """
    Generates weights spread over the unit simplex
    :param num_objs: number of objectives in the problem
    :param num_weights: number of weights desired
    :return: a list of np.array containing the weights.
    ! There might be a different number of weights being returned than what is specified !
    See https://pymoo.org/misc/reference_directions.html#Reference-Directions
    """
    weights = get_reference_directions("das-dennis", num_objs, n_partitions=num_weights)
    if len(weights) != num_weights + 1:
        print(f"WARNING: the number of weights {num_weights} you have chosen cannot be sampled using Das-Dennis approach. "
              f"We have sampled {len(weights)} instead.")
    return [np.array(weight, dtype=np.float32) for weight in weights]