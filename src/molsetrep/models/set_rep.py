import torch
from torch.nn import (
    Module,
    Parameter,
    Linear,
    BatchNorm1d,
    ReLU,
    LeakyReLU,
    Dropout,
)


class SetRep(Module):
    def __init__(
        self,
        n_hidden_sets: int,
        n_elements: int,
        d: int,
        n_out_channels: int = 32,
    ):
        super(SetRep, self).__init__()

        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements
        self.d = d
        self.n_out_channels = n_out_channels

        self.Wc = Parameter(
            torch.FloatTensor(self.d, self.n_hidden_sets * self.n_elements)
        )
        self.bn = BatchNorm1d(self.n_hidden_sets)
        self.fc1 = Linear(self.n_hidden_sets, self.n_out_channels)
        self.relu = LeakyReLU()

        # Init weights
        self.Wc.data.normal_()

    def forward(self, X):
        t = self.relu(torch.matmul(X, self.Wc))
        t = t.view(t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets)
        t, _ = torch.max(t, dim=2)
        t = torch.sum(t, dim=1)
        t = self.bn(t)
        t = self.fc1(t)
        out = self.relu(t)

        return out
