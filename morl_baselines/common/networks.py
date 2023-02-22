"""Utilities for Neural Networks."""

from typing import List, Type

import numpy as np
import torch as th
from torch import nn


def mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    drop_rate: float = 0.0,
    layer_norm: bool = False,
) -> nn.Sequential:
    """Create a multi layer perceptron (MLP), which is a collection of fully-connected layers each followed by an activation function.

    Args:
        input_dim: Dimension of the input vector
        output_dim: Dimension of the output vector
        net_arch: Architecture of the neural net. It represents the number of units per layer. The length of this list is the number of layers.
        activation_fn: The activation function to use after each layer.
        drop_rate: Dropout rate
        layer_norm: Whether to use layer normalization
    """
    assert len(net_arch) > 0
    modules = [nn.Linear(input_dim, net_arch[0])]
    if drop_rate > 0.0:
        modules.append(nn.Dropout(p=drop_rate))
    if layer_norm:
        modules.append(nn.LayerNorm(net_arch[0]))
    modules.append(activation_fn())

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        if drop_rate > 0.0:
            modules.append(nn.Dropout(p=drop_rate))
        if layer_norm:
            modules.append(nn.LayerNorm(net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1]
        modules.append(nn.Linear(last_layer_dim, output_dim))

    return nn.Sequential(*modules)


class NatureCNN(nn.Module):
    """CNN from DQN nature paper: Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533."""

    def __init__(self, observation_shape: np.ndarray, features_dim: int = 512):
        """CNN from DQN Nature.

        Args:
            observation_shape: Shape of the observation.
            features_dim: Number of features extracted. This corresponds to the number of unit for the last layer.
        """
        super().__init__()
        self.features_dim = features_dim
        n_input_channels = 1 if len(observation_shape) == 2 else observation_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(np.zeros(observation_shape)[np.newaxis]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Predicts the features from the observations.

        Args:
            observations: current observations
        """
        if observations.dim() == 3:
            observations = observations.unsqueeze(0)
        return self.linear(self.cnn(observations / 255.0))
