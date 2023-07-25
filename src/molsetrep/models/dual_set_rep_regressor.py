import torch
from torch.nn import Parameter, Linear, BatchNorm1d, ReLU, LeakyReLU, Linear
from torch.nn.functional import log_softmax


class DualSetRepRegressor(torch.nn.Module):
    def __init__(
        self, n_hidden_sets, n_hidden_sets_2, n_elements, n_elements_2, d, d_2
    ):
        super(DualSetRepRegressor, self).__init__()
        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements

        self.n_hidden_sets_2 = n_hidden_sets_2
        self.n_elements_2 = n_elements_2

        self.Wc = Parameter(torch.FloatTensor(d, n_hidden_sets * n_elements))
        self.Wc_2 = Parameter(torch.FloatTensor(d_2, n_hidden_sets_2 * n_elements_2))
        self.fc1 = Linear(n_hidden_sets, 32)
        self.fc1_2 = Linear(n_hidden_sets_2, 32)
        self.bn = BatchNorm1d(n_hidden_sets, affine=True)
        self.bn_2 = BatchNorm1d(n_hidden_sets_2, affine=True)
        self.fc2 = Linear(32 * 2, 32)
        self.fc3 = Linear(32, 1)
        self.relu = ReLU()
        self.relu_2 = ReLU()

        self.init_weights()

    def init_weights(self):
        self.Wc.data.normal_()
        self.Wc_2.data.normal_()

    def forward(self, X, X2):
        # First sets (e.g. atoms)
        t = torch.matmul(X, self.Wc)
        t = self.relu(t)
        t = t.view(t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets)
        t, _ = torch.max(t, dim=2)
        t = torch.sum(t, dim=1)
        t = self.bn(t)
        t = self.fc1(t)
        t = self.relu(t)

        # Second sets (e.g. bonds)
        t_2 = torch.matmul(X2, self.Wc_2)
        t_2 = self.relu_2(t_2)
        t_2 = t_2.view(
            t_2.size()[0], t_2.size()[1], self.n_elements_2, self.n_hidden_sets_2
        )
        t_2, _ = torch.max(t_2, dim=2)
        t_2 = torch.sum(t_2, dim=1)
        t_2 = self.bn_2(t_2)
        t_2 = self.fc1_2(t_2)
        t_2 = self.relu_2(t_2)

        # Concat and softmax
        out = self.fc2(torch.cat((t, t_2), 1))
        out = self.fc3(out)

        return out.squeeze(1)
