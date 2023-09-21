from typing import List, Optional
import torch
import lightning.pytorch as pl


from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch_geometric.nn import GIN
from torch_geometric.nn.pool import global_mean_pool

from torchmetrics.classification import (
    Accuracy,
    AUROC,
    F1Score,
)
from torchmetrics.regression import (
    R2Score,
    MeanSquaredError,
    MeanAbsoluteError,
    PearsonCorrCoef,
)

from molsetrep.models import GINE, MLP
from molsetrep.metrics import AUPRC


class GNNClassifier(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        in_edge_channels: int,
        n_hidden_channels: Optional[List] = None,
        n_classes: int = 2,
        gnn: Optional[torch.nn.Module] = None,
    ):
        super(GNNClassifier, self).__init__()

        if n_hidden_channels is None or len(n_hidden_channels) < 2:
            n_hidden_channels = [32, 16]

        self.in_edge_channels = in_edge_channels
        self.gnn = gnn

        if self.gnn is None:
            if self.in_edge_channels > 0:
                self.gnn = GINE(
                    in_channels,
                    n_hidden_channels[0],
                    num_layers,
                    edge_dim=in_edge_channels,
                    jk="cat",
                )
            else:
                self.gnn = GIN(in_channels, n_hidden_channels[0], num_layers)

        self.mlp = MLP(n_hidden_channels[0], n_hidden_channels[1], n_classes)

    def forward(self, batch):
        if self.in_edge_channels > 0:
            out = self.gnn(
                batch.x.float(), batch.edge_index, edge_attr=batch.edge_attr.float()
            )
        else:
            out = self.gnn(batch.x.float(), batch.edge_index)

        t = global_mean_pool(out, batch.batch)
        out = self.mlp(t)
        return F.log_softmax(out, dim=1)


class GNNRegressor(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_layers,
        in_edge_channels,
        n_hidden_channels: Optional[List] = None,
        gnn: Optional[torch.nn.Module] = None,
    ):
        super(GNNRegressor, self).__init__()

        if n_hidden_channels is None or len(n_hidden_channels) < 2:
            n_hidden_channels = [32, 16]

        self.in_edge_channels = in_edge_channels
        self.gnn = gnn

        if self.gnn is None:
            if self.in_edge_channels > 0:
                self.gnn = GINE(
                    in_channels,
                    n_hidden_channels[0],
                    num_layers,
                    edge_dim=in_edge_channels,
                    jk="cat",
                )
            else:
                self.gnn = GIN(in_channels, n_hidden_channels[0], num_layers)

        self.mlp = MLP(n_hidden_channels[0], n_hidden_channels[1], 1)

    def forward(self, batch):
        if self.in_edge_channels > 0:
            out = self.gnn(
                batch.x.float(), batch.edge_index, edge_attr=batch.edge_attr.float()
            )
        else:
            out = self.gnn(batch.x.float(), batch.edge_index)

        t = global_mean_pool(out, batch.batch)
        out = self.mlp(t)
        return out.squeeze(1)


class LightningGNNClassifier(pl.LightningModule):
    def __init__(
        self,
        n_layers: int,
        n_in_channels: int,
        n_edge_channels: int,
        n_classes: int,
        n_hidden_channels: Optional[List] = None,
        class_weights: Optional[List] = None,
        gnn: Optional[torch.nn.Module] = None,
        learning_rate: float = 0.001,
    ) -> None:
        super(LightningGNNClassifier, self).__init__()
        self.save_hyperparameters()

        self.class_weights = class_weights
        self.learning_rate = learning_rate

        self.gnn_classifier = GNNClassifier(
            n_in_channels, n_layers, n_edge_channels, n_hidden_channels, n_classes, gnn
        )

        # Criterions
        self.criterion = CrossEntropyLoss()
        if class_weights is not None:
            self.criterion = CrossEntropyLoss(
                weight=torch.FloatTensor(self.class_weights).to(self.device)
            )

        self.criterion_eval = CrossEntropyLoss()

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.train_auroc = AUROC(task="multiclass", num_classes=n_classes)
        self.train_auprc = AUPRC(task="multiclass", num_classes=n_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=n_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_auroc = AUROC(task="multiclass", num_classes=n_classes)
        self.val_auprc = AUPRC(task="multiclass", num_classes=n_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=n_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=n_classes)
        self.test_auprc = AUPRC(task="multiclass", num_classes=n_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        return self.gnn_classifier(x)

    def training_step(self, batch, batch_idx):
        y = batch.y

        out = self(batch)
        loss = self.criterion(out, y)

        # Metrics
        self.train_accuracy(out, y)
        self.train_auroc(out, y)
        self.train_auprc(out, y)
        self.train_f1(out, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=len(y))
        self.log(
            "train/acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )
        self.log(
            "train/auroc",
            self.train_auroc,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )
        self.log(
            "train/auprc",
            self.train_auprc,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )
        self.log(
            "train/f1", self.train_f1, on_step=False, on_epoch=True, batch_size=len(y)
        )

        return loss

    def validation_step(self, val_batch, batch_idx):
        y = val_batch.y

        out = self(val_batch)
        loss = self.criterion_eval(out, y)

        # Metrics
        self.val_accuracy(out, y)
        self.val_auroc(out, y)
        self.val_auprc(out, y)
        self.val_f1(out, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=len(y))
        self.log(
            "val/acc",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )
        self.log(
            "val/auroc", self.val_auroc, on_step=False, on_epoch=True, batch_size=len(y)
        )
        self.log(
            "val/auprc", self.val_auprc, on_step=False, on_epoch=True, batch_size=len(y)
        )
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, batch_size=len(y))

    def test_step(self, test_batch, batch_idx):
        y = test_batch.y

        out = self(test_batch)
        loss = self.criterion_eval(out, y)

        # Metrics
        self.test_accuracy(out, y)
        self.test_auroc(out, y)
        self.test_auprc(out, y)
        self.test_f1(out, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=len(y))
        self.log(
            "test/acc",
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )
        self.log(
            "test/auroc",
            self.test_auroc,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )
        self.log(
            "test/auprc",
            self.test_auprc,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )
        self.log(
            "test/f1", self.test_f1, on_step=False, on_epoch=True, batch_size=len(y)
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LightningGNNRegressor(pl.LightningModule):
    def __init__(
        self,
        n_layers: int,
        n_in_channels: int,
        n_edge_channels: int,
        n_hidden_channels: Optional[List] = None,
        gnn: Optional[torch.nn.Module] = None,
        learning_rate: float = 0.001,
        scaler: Optional[any] = None,
    ) -> None:
        super(LightningGNNRegressor, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.scaler = scaler

        self.gnn_regressor = GNNRegressor(
            n_in_channels, n_layers, n_edge_channels, n_hidden_channels, gnn
        )

        # Metrics
        self.train_r2 = R2Score()
        self.train_pearson = PearsonCorrCoef()
        self.train_rmse = MeanSquaredError(squared=False)
        self.train_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self.val_pearson = PearsonCorrCoef()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()
        self.test_r2 = R2Score()
        self.test_pearson = PearsonCorrCoef()
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_mae = MeanAbsoluteError()

    def forward(self, x):
        return self.gnn_regressor(x)

    def training_step(self, batch, batch_idx):
        y = batch.y

        out = self(batch)
        loss = F.mse_loss(out, y)

        if self.scaler:
            out = torch.FloatTensor(
                self.scaler.inverse_transform(
                    out.detach().cpu().reshape(-1, 1)
                ).flatten()
            )
            y = torch.FloatTensor(
                self.scaler.inverse_transform(y.detach().cpu().reshape(-1, 1)).flatten()
            )

        # Metrics
        self.train_r2(out, y)
        self.train_pearson(out.to(self.device), y.to(self.device))
        self.train_rmse(out, y)
        self.train_mae(out, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=len(y))
        self.log(
            "train/r2", self.train_r2, on_step=False, on_epoch=True, batch_size=len(y)
        )
        self.log(
            "train/pearson",
            self.train_pearson,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )
        self.log(
            "train/rmse",
            self.train_rmse,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )
        self.log(
            "train/mae", self.train_mae, on_step=False, on_epoch=True, batch_size=len(y)
        )

        return loss

    def validation_step(self, val_batch, batch_idx):
        y = val_batch.y

        out = self(val_batch)
        loss = F.mse_loss(out, y)

        if self.scaler:
            out = torch.FloatTensor(
                self.scaler.inverse_transform(
                    out.detach().cpu().reshape(-1, 1)
                ).flatten()
            )
            y = torch.FloatTensor(
                self.scaler.inverse_transform(y.detach().cpu().reshape(-1, 1)).flatten()
            )

        # Metrics
        self.val_r2(out, y)
        self.val_pearson(out.to(self.device), y.to(self.device))
        self.val_rmse(out, y)
        self.val_mae(out, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True, batch_size=len(y))
        self.log(
            "val/pearson",
            self.val_pearson,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )
        self.log(
            "val/rmse", self.val_rmse, on_step=False, on_epoch=True, batch_size=len(y)
        )
        self.log(
            "val/mae", self.val_mae, on_step=False, on_epoch=True, batch_size=len(y)
        )

    def test_step(self, test_batch, batch_idx):
        y = test_batch.y

        out = self(test_batch)
        loss = F.mse_loss(out, y)

        if self.scaler:
            out = torch.FloatTensor(
                self.scaler.inverse_transform(
                    out.detach().cpu().reshape(-1, 1)
                ).flatten()
            )
            y = torch.FloatTensor(
                self.scaler.inverse_transform(y.detach().cpu().reshape(-1, 1)).flatten()
            )

        # Metrics
        self.test_r2(out, y)
        self.test_pearson(out.to(self.device), y.to(self.device))
        self.test_rmse(out, y)
        self.test_mae(out, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=len(y))
        self.log(
            "test/r2", self.test_r2, on_step=False, on_epoch=True, batch_size=len(y)
        )
        self.log(
            "test/pearson",
            self.test_pearson,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )
        self.log(
            "test/rmse", self.test_rmse, on_step=False, on_epoch=True, batch_size=len(y)
        )
        self.log(
            "test/mae", self.test_mae, on_step=False, on_epoch=True, batch_size=len(y)
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
