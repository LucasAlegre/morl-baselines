import torch
from torch import nn


# Based on implementation from: https://github.com/zpschang/DPMORL
# Credits for the original code go to the author: Xin-Qiang Cai


class UtilityFunction(nn.Module):
    def __init__(
            self,
            min_val,
            max_val,
            frozen=True,
            normalise=True,
            max_weight=0.1,
            size_factor=1
    ):
        super().__init__()
        # Initialize the variables
        self.reward_shape = len(min_val)
        self.min_val = min_val
        self.max_val = max_val
        self.normalise = normalise

        # Initialize the utility function
        self.mlp1 = nn.Linear(self.reward_shape, 24 * size_factor)
        self.mlp2 = nn.Linear(72 * size_factor, 24 * size_factor)
        self.mlp3 = nn.Linear(72 * size_factor, 24 * size_factor)
        self.mlp4 = nn.Linear(72 * size_factor, 1)
        self.max_weight = max_weight

        # Initialize positive weight
        self.make_monotone_init()
        self.make_monotone()
        if frozen:
            self.make_frozen()

        # Compute min and max utility values
        self.min_u, self.max_u = self.compute_utility(torch.stack([min_val, max_val]))

    def forward(self, x):
        utilities = self.compute_utility(x)
        if self.normalise:
            utilities = (utilities - self.min_u) / (self.max_u - self.min_u)
        return utilities

    def compute_utility(self, input_x):
        x = self.mlp1(input_x)
        x = torch.cat([x.clamp(min=-0.5), x.clamp(min=-0.5, max=0.5), x.clamp(max=0.5)], -1)
        x = self.mlp2(x)
        x = torch.cat([x.clamp(min=-0.5), x.clamp(min=-0.5, max=0.5), x.clamp(max=0.5)], -1)
        x = self.mlp3(x)
        x = torch.cat([x.clamp(min=-0.5), x.clamp(min=-0.5, max=0.5), x.clamp(max=0.5)], -1)
        x = self.mlp4(x)
        return x[:, 0]

    def make_monotone_init(self):
        for layer in [self.mlp1, self.mlp2, self.mlp3, self.mlp4]:
            layer.weight.data = layer.weight.data.abs()

    def make_monotone(self):
        for layer in [self.mlp1, self.mlp2, self.mlp3, self.mlp4]:
            layer.weight.data = torch.maximum(layer.weight.data, torch.tensor(0.0))
        for layer in [self.mlp1, self.mlp2, self.mlp3, self.mlp4]:
            layer.weight.data = torch.minimum(layer.weight.data, torch.tensor(self.max_weight))

    def make_frozen(self):
        for param in self.parameters():
            param.requires_grad = False
