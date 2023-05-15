import torch
from torch.nn import Parameter, Linear, BatchNorm1d, LeakyReLU, Linear


class SetRepRegressor(torch.nn.Module):
    def __init__(self, n_hidden_sets, n_elements, d):
        super(SetRepRegressor, self).__init__()
        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements

        self.Wc = Parameter(torch.FloatTensor(d, n_hidden_sets * n_elements))
        self.fc1 = Linear(n_hidden_sets, 32)
        self.bn = BatchNorm1d(32, affine=True)
        self.relu = LeakyReLU()
        self.fc2 = Linear(32, 1)

        self.init_weights()

    def init_weights(self):
        self.Wc.data.uniform_(-1, 1)

    def forward(self, X):
        t = self.relu(torch.matmul(X, self.Wc))
        t = t.view(t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets)
        t, _ = torch.max(t, dim=2)
        t = torch.sum(t, dim=1)
        t = self.fc1(t)
        t = self.bn(t)
        t = self.relu(t)
        out = self.fc2(t)

        return out.squeeze(1)
