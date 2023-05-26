from torch_geometric.nn import GINEConv, MLP, MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN


class GINE(BasicGNN):
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(
        self, in_channels: int, out_channels: int, edge_dim: int, **kwargs
    ) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINEConv(mlp, train_eps=True, edge_dim=edge_dim, **kwargs)
