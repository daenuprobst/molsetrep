from typing import List, Optional
import torch
from torch.nn import Module, Parameter, Linear, BatchNorm1d, LeakyReLU, Linear
from molsetrep.models.set_rep import SetRep
from molsetrep.models.mlp import MLP


class SRClassifier(Module):
    def __init__(
        self,
        n_hidden_sets: int,
        n_elements: int,
        d: int,
        n_classes: int,
        n_hidden_channels: Optional[List] = None,
    ) -> None:
        super(SRClassifier, self).__init__()

        if n_hidden_channels is None:
            n_hidden_channels = [32, 16]

        self.set_rep = SetRep(n_hidden_sets, n_elements, d, n_hidden_channels[0])
        self.mlp = MLP(n_hidden_channels[0], n_hidden_channels[1], n_classes)

    def forward(self, x):
        x = self.set_rep(x)
        return self.mlp(x)


class SRRegressor(Module):
    def __init__(
        self,
        n_hidden_sets: int,
        n_elements: int,
        d: int,
        n_hidden_channels: Optional[List] = None,
    ) -> None:
        super(SRRegressor, self).__init__()

        if n_hidden_channels is None:
            n_hidden_channels = [32, 16]

        self.set_rep = SetRep(n_hidden_sets, n_elements, d, n_hidden_channels[0])
        self.mlp = MLP(n_hidden_channels[0], n_hidden_channels[1], 1)

    def forward(self, x):
        x = self.set_rep(x)
        return self.mlp(x).squeeze(1)
