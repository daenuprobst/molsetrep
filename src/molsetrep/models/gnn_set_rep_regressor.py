from typing import Optional
import torch
from torch.nn import Parameter, Linear, BatchNorm1d, LeakyReLU, Linear
from torch.nn.functional import log_softmax
from torch_geometric.nn import GIN
from torch_geometric.utils import unbatch
from molsetrep.models.gine import GINE


class GNNSetRepRegressor(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        in_edge_channels,
        n_hidden_sets,
        n_elements,
        gnn: Optional[torch.nn.Module] = None,
    ):
        super(GNNSetRepRegressor, self).__init__()

        self.d = hidden_channels
        self.in_edge_channels = in_edge_channels
        self.gnn = gnn

        if self.gnn is None:
            if self.in_edge_channels > 0:
                self.gnn = GINE(
                    in_channels,
                    hidden_channels,
                    num_layers,
                    edge_dim=in_edge_channels,
                    jk="cat",
                )
            else:
                self.gnn = GIN(in_channels, hidden_channels, num_layers)

        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements

        self.Wc = Parameter(torch.FloatTensor(self.d, n_hidden_sets * n_elements))
        self.fc1 = Linear(n_hidden_sets, 32)
        self.bn = BatchNorm1d(32, affine=True, track_running_stats=False)
        self.relu = LeakyReLU()
        self.fc2 = Linear(32, 1)

        self.init_weights()

    def init_weights(self):
        self.Wc.data.uniform_(-1, 1)

    def forward(self, batch):
        if self.in_edge_channels > 0:
            out = self.gnn(
                batch.x.float(), batch.edge_index, edge_attr=batch.edge_attr.float()
            )
        else:
            out = self.gnn(batch.x.float(), batch.edge_index)

        out_unbatched = unbatch(out, batch.batch)

        batch_size = len(out_unbatched)
        max_cardinality = max([b.shape[0] for b in out_unbatched])

        X = torch.zeros((batch_size, max_cardinality, self.d), device=batch.x.device)

        for i, b in enumerate(out_unbatched):
            X[i, : b.shape[0], :] = b

        t = self.relu(torch.matmul(X, self.Wc))
        t = t.view(t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets)
        t, _ = torch.max(t, dim=2)
        t = torch.sum(t, dim=1)
        t = self.fc1(t)
        t = self.bn(t)
        t = self.relu(t)
        out = self.fc2(t)
        return out.squeeze(1)
