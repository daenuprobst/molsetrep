from typing import Optional
import torch
from torch.nn import Parameter, Linear, BatchNorm1d, LeakyReLU, Linear, Dropout
from torch.nn.functional import log_softmax
from torch_geometric.nn import GIN
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import unbatch
from molsetrep.models.gine import GINE


class GNNRegressor(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        in_edge_channels,
        dropout: float = 0.0,
        gnn: Optional[torch.nn.Module] = None,
    ):
        super(GNNRegressor, self).__init__()

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

        self.dropout = Dropout(dropout)
        self.bn = BatchNorm1d(hidden_channels, affine=True, track_running_stats=False)
        self.relu = LeakyReLU()
        self.fc2 = Linear(hidden_channels, 1)

        self.Wc = torch.FloatTensor([0])

        # self.init_weights()

    # def init_weights(self):
    #     self.Wc.data.uniform_(-1, 1)

    def forward(self, batch):
        if self.in_edge_channels > 0:
            out = self.gnn(
                batch.x.float(), batch.edge_index, edge_attr=batch.edge_attr.float()
            )
        else:
            out = self.gnn(batch.x.float(), batch.edge_index)

        t = global_mean_pool(out, batch.batch)

        t = self.dropout(t)
        t = self.bn(t)
        t = self.relu(t)
        out = self.fc2(t)

        return out.squeeze(1)
