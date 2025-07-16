from torch.nn import Module, Linear, LeakyReLU, Sequential, BatchNorm1d, Dropout


class MLP(Module):
    def __init__(
        self,
        n_input_channels,
        n_hidden_channels,
        n_out_channels,
        dropout: float = 0.0,
        bn: bool = True,
    ):
        super().__init__()

        layers = []

        if bn:
            layers.append(BatchNorm1d(n_input_channels))

        layers.append(Linear(n_input_channels, n_hidden_channels))
        layers.append(LeakyReLU())
        layers.append(Dropout(dropout))
        layers.append(Linear(n_hidden_channels, n_out_channels))

        self.layers = Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
