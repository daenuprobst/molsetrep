from typing import Optional
import torch
from torch.nn import Parameter, Linear, BatchNorm1d, LeakyReLU, Linear
from torch.nn.functional import log_softmax
from torch_geometric.nn import GIN
from torch_geometric.utils import unbatch
from molsetrep.models.gine import GINE
