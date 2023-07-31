import torch
from torch.nn import Module, Parameter, Linear, BatchNorm1d, LeakyReLU, Linear


class DualSRClassifier(Module):
    def __init__(self) -> None:
        super(DualSRClassifier, self).__init__()

    def forward(self, X1, X2):
        ...


class DualSRRegressor(Module):
    def __init__(self) -> None:
        super(DualSRRegressor, self).__init__()

    def forward(self, X, X2):
        ...
