from typing import Optional
import torch
import networkx as nx
from torch.nn import Parameter, Linear, BatchNorm1d, LeakyReLU, Linear, Dropout
from torch.nn.functional import log_softmax, sigmoid
from torch_geometric.nn import GIN
from torch_geometric.utils import unbatch, to_networkx
from torch_geometric.data import Batch
from molsetrep.models.gine import GINE
from set_transformer.models import DeepSet


class GNNDeepSetClassifierGIN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        in_edge_channels: int,
        n_hidden_sets: int,
        n_elements: int,
        n_classes: int = 2,
        ##### CHANGE #####
        dropout: float = 0.0, #default usually at p=0.5 maybe add , p=0.2
        gnn: Optional[torch.nn.Module] = None,
    ):
        super(GNNDeepSetClassifierGIN, self).__init__()

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
        self.n_classes = n_classes
##############
        ### removed
        # self.Wc = Parameter(torch.FloatTensor(self.d, n_hidden_sets * n_elements))
        self.dropout = Dropout(dropout)
        ### changed from self.fc1 = Linear(n_hidden_sets, 32)
        self.fc1 = Linear(n_hidden_sets * n_classes, 32)
        self.bn = BatchNorm1d(32, affine=True, track_running_stats=False)
        self.relu = LeakyReLU()
        self.fc2 = Linear(32, n_classes) 
##############
# can be replaced w. DeepSet or SetTransformer (also don't forget to define the forward part)

    ### removed
    #     self.init_weights()

    # def init_weights(self):
    #     self.Wc.data.uniform_(-1, 1)
    ### added
        self.deep_set = DeepSet(self.d, n_hidden_sets, n_classes)

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

        # t = self.relu(torch.matmul(X, self.Wc))
        # t = t.view(t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets)
        # t, _ = torch.max(t, dim=2)
        # t = torch.sum(t, dim=1)

        ### added

        t = self.deep_set(X)

        ### added
        t = torch.reshape(t, (X.size(dim=0), self.n_hidden_sets * self.n_classes)) # reshape to (batch size. hidden_sets * n_classes)

        t = self.dropout(t)
        t = self.fc1(t)
        t = self.bn(t)
        t = self.relu(t)
        out = self.fc2(t)
        return log_softmax(out, dim=1)
