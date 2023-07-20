from typing import Optional
import torch
import networkx as nx
from torch.nn import Parameter, Linear, BatchNorm1d, LeakyReLU, Linear, Dropout
from torch.nn.functional import log_softmax
from torch_geometric.nn import GIN, global_mean_pool
from torch_geometric.utils import unbatch, to_networkx
from torch_geometric.data import Batch
from molsetrep.models.gine import GINE


class GNNClassifier(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        in_edge_channels: int,
        n_classes: int = 2,
        dropout: float = 0.0,
        gnn: Optional[torch.nn.Module] = None,
    ):
        super(GNNClassifier, self).__init__()

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
        self.fc1 = Linear(hidden_channels, 32)
        self.bn = BatchNorm1d(32, affine=True, track_running_stats=False)
        self.relu = LeakyReLU()
        self.fc2 = Linear(32, n_classes)

    def forward(self, batch):
        if self.in_edge_channels > 0:
            out = self.gnn(
                batch.x.float(), batch.edge_index, edge_attr=batch.edge_attr.float()
            )
        else:
            out = self.gnn(batch.x.float(), batch.edge_index)

        t = global_mean_pool(out, batch.batch)
        t = self.dropout(t)
        t = self.fc1(t)
        t = self.bn(t)
        t = self.relu(t)
        out = self.fc2(t)
        return log_softmax(out, dim=1)
