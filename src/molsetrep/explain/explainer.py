from typing import Tuple, Optional
import torch
from torch import FloatTensor
from torch.nn import Module
from torch_geometric.loader import DataLoader
from sklearn.neighbors import NearestNeighbors

import numpy as np


class Explainer:
    def __init__(
        self,
        model: Module,
        loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.loader = loader

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

    def update(
        self,
    ) -> None:
        raise NotImplementedError()

    def get_knns(self, hidden_sets: np.ndarray, loader_sets: np.ndarray, k: int = 1):
        hidden_means = np.mean(hidden_sets, axis=1)

        loader_means = []
        for s in loader_sets:
            loader_means.append(np.mean(s, axis=0))

        loader_means = np.array(loader_means)

        nbrs = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="cosine").fit(
            loader_means
        )
        _, indices = nbrs.kneighbors(hidden_means)
        return indices.flatten()

    def get_hidden_sets(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        hidden_sets = (
            self.model.Wc.cpu()
            .detach()
            .numpy()
            .reshape((self.model.d, self.model.n_hidden_sets, self.model.n_elements))
        )
        hidden_sets = np.moveaxis(hidden_sets, 1, 0)
        hidden_sets = np.moveaxis(hidden_sets, 1, 2)

        output = (
            self.model.forward_set_only(FloatTensor(hidden_sets).to(self.device))
            .cpu()
            .detach()
            .numpy()
        )

        return (hidden_sets, output)
