import torch
from torch.nn import Module, Parameter, Linear, BatchNorm1d, LeakyReLU, Linear


class TripleSRClassifier(Module):
    def __init__(self) -> None:
        super(TripleSRClassifier, self).__init__()

    def forward(self, X1, X2, X3):
        ...


class TripleSRRegressor(Module):
    def __init__(self) -> None:
        super(TripleSRRegressor, self).__init__()

    def forward(self, X1, X2, X3):
        ...
