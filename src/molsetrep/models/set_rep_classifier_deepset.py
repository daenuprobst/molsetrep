import torch
from torch.nn import Parameter, Linear, BatchNorm1d, ReLU, LeakyReLU, Linear
from torch.nn.functional import log_softmax
from set_transformer.models import DeepSet


class SetRepClassifierDeepSet(torch.nn.Module):
    # changed
    def __init__(self, n_hidden_sets, n_elements, d, n_classes):
        super(SetRepClassifierDeepSet, self).__init__()
        self.d = d
        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements
        self.n_classes=n_classes

        ### removed
        # self.Wc = Parameter(torch.FloatTensor(d, n_hidden_sets * n_elements))
        ### changed from self.fc1 = Linear(n_hidden_sets, 32)
        self.fc1 = Linear(n_hidden_sets * n_classes, 32)
        self.bn = BatchNorm1d(n_hidden_sets, affine=True)  # Set affome tp false
        self.fc2 = Linear(32, n_classes)
        self.relu = ReLU()

        ### added
        self.deep_set = DeepSet(self.d, n_hidden_sets, n_classes)
        ## removed
        # self.init_weights()

    ### removed
    # def init_weights(self):
    #     self.Wc.data.normal_()

    def forward(self, X):
        ### removed
        # t = torch.matmul(X, self.Wc)
        # t = self.relu(t)
        # t = t.view(t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets)
        # t, _ = torch.max(t, dim=2)
        # t = torch.sum(t, dim=1)
        ### added

        t = self.deep_set(X)

        t = self.bn(t)
        ### added
        t = torch.reshape(t, (X.size(dim=0), self.n_hidden_sets * self.n_classes)) # reshape to (batch size. hidden_sets * n_classes)

        t = self.fc1(t)
        t = self.relu(t)
        out = self.fc2(t)

        return log_softmax(out, dim=1)
