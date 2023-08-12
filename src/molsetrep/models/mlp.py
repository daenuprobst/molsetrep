from torch.nn import Module, Linear, ReLU, Sequential, BatchNorm1d, Dropout


class MLP(Module):
    def __init__(self, n_input_channels, n_hidden_channels, n_out_channels):
        super().__init__()

        self.layers = Sequential(
            BatchNorm1d(n_input_channels),
            Linear(n_input_channels, n_hidden_channels),
            ReLU(),
            # Linear(n_hidden_channels, n_hidden_channels),
            # ReLU(),
            Linear(n_hidden_channels, n_out_channels),
        )

    def forward(self, x):
        return self.layers(x)
