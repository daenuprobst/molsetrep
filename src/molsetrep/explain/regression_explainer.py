from typing import Optional
from IPython.display import display, clear_output

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.nn import Module
from torch_geometric.loader import DataLoader
from molsetrep.explain.explainer import Explainer


class RegressionExplainer(Explainer):
    def __init__(
        self,
        model: Module,
        loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(model, loader, device)

    def update(self) -> None:
        y_gt = []
        y_pred = []
        embeddings = []
        for batch in self.loader:
            batch.to(self.device)
            output, embs = self.model(batch, get_embeddings=True)
            embeddings.append([emb.cpu().detach().numpy() for emb in embs])
            y_gt.append(batch.y.cpu().detach().numpy())
            y_pred.append(output.cpu().detach().numpy())

        y_gt = np.concatenate(y_gt)
        y_pred = np.concatenate(y_pred)
        embeddings = np.concatenate(embeddings)

        hidden_sets, y_hidden = self.get_hidden_sets()

        knns = self.get_knns(hidden_sets, embeddings)
        print(knns)
        y_hidden_nn = y_gt[knns]

        clear_output(wait=True)
        plt.cla()
        plt.scatter(y_gt, y_pred)
        plt.scatter(y_hidden_nn, y_hidden, color="red")
        # plt.title(f"Epoch: {epoch + 1}, RMSE: {np.sqrt(valid_mse.compute())}")
        plt.show()
