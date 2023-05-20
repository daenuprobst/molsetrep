from typing import Optional
import torch
import networkx as nx
from torch.nn import Parameter, Linear, BatchNorm1d, LeakyReLU, Linear
from torch.nn.functional import log_softmax
from torch_geometric.nn import GIN
from torch_geometric.utils import unbatch, to_networkx
from torch_geometric.data import Batch
from molsetrep.models.gine import GINE


class GNNSetRepClassifierSubstruct(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        in_edge_channels: int,
        n_hidden_sets: int,
        n_elements: int,
        n_classes: int = 2,
        gnn: Optional[torch.nn.Module] = None,
    ):
        super(GNNSetRepClassifierSubstruct, self).__init__()

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
        self.fc2 = Linear(32, n_classes)

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

        cc_lengths = []
        for data in batch.to_data_list():
            G = to_networkx(data)
            cc_lengths.append(
                [len(cc) for cc in nx.connected_components(G.to_undirected())]
            )

        ub_lengths = []
        for ub in out_unbatched:
            ub_lengths.append(len(ub))

        # print(cc_lengths)
        # print(ub_lengths)

        fps = []
        for ub, ccs in zip(out_unbatched, cc_lengths):
            fp = []
            offset = 0
            for cc in ccs:
                fp.append(ub[offset:cc])
                offset += cc

            max_cardinality = max([b.shape[0] for b in fp])
            fp_to = torch.zeros(
                (len(fp), max_cardinality, self.d), device=batch.x.device
            )
            for i, b in enumerate(fp):
                fp_to[i, : b.shape[0], :] = b

            fps.append(torch.mean(fp_to, dim=-2))

        # for fp in fps:
        #     print(fp.shape)
        # print("-------------------------------")

        out_unbatched = fps

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
        return log_softmax(out, dim=1)
