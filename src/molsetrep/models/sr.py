from typing import List, Optional, Callable

import torch
from torch.nn import Module, CrossEntropyLoss
import torch.nn.functional as F

import lightning.pytorch as pl

from torchmetrics.classification import Accuracy, AUROC, AveragePrecision
from torchmetrics.regression import R2Score, MeanSquaredError, MeanAbsoluteError

from molsetrep.models import SetRep, SetTransformer, DeepSet, MLP


class SRClassifier(Module):
    def __init__(
        self,
        n_hidden_sets: int,
        n_elements: int,
        d: int,
        n_classes: int,
        n_hidden_channels: Optional[List] = None,
    ) -> None:
        super(SRClassifier, self).__init__()

        if n_hidden_channels is None:
            n_hidden_channels = [32, 16]

        self.set_rep = SetRep(n_hidden_sets, n_elements, d, n_hidden_channels[0])
        self.mlp = MLP(n_hidden_channels[0], n_hidden_channels[1], n_classes)

    def forward(self, x):
        x = self.set_rep(x)
        return self.mlp(x)


class SRRegressor(Module):
    def __init__(
        self,
        n_hidden_sets: int,
        n_elements: int,
        d: int,
        n_hidden_channels: Optional[List] = None,
    ) -> None:
        super(SRRegressor, self).__init__()

        if n_hidden_channels is None:
            n_hidden_channels = [32, 16]

        self.set_rep = SetRep(n_hidden_sets, n_elements, d, n_hidden_channels[0])
        self.mlp = MLP(n_hidden_channels[0], n_hidden_channels[1], 1)

    def forward(self, x):
        x = self.set_rep(x)
        return self.mlp(x).squeeze(1)


class LightningSRClassifier(pl.LightningModule):
    def __init__(
        self,
        n_hidden_sets: List[int],
        n_elements: List[int],
        d: List[int],
        n_classes: int,
        n_hidden_channels: Optional[List] = None,
        class_weights: Optional[List] = None,
        learning_rate: float = 0.001,
    ) -> None:
        super(LightningSRClassifier, self).__init__()
        self.save_hyperparameters()

        self.class_weights = class_weights
        self.learning_rate = learning_rate

        self.sr_classifier = SRClassifier(
            n_hidden_sets[0], n_elements[0], d[0], n_classes, n_hidden_channels
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
        self.train_ap = AveragePrecision(
            task="multiclass", num_classes=n_classes, average=None
        )
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_auroc = AUROC(task="multiclass", num_classes=n_classes)
        self.val_ap = AveragePrecision(
            task="multiclass", num_classes=n_classes, average=None
        )
        self.test_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=n_classes)
        self.test_ap = AveragePrecision(
            task="multiclass", num_classes=n_classes, average=None
        )

    def forward(self, x):
        return self.sr_classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)
        loss = self.criterion(out, y)

        # Metrics
        self.train_accuracy(out, y)
        self.train_auroc(out, y)
        self.train_ap(out, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_accuracy, on_step=False, on_epoch=True)
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True)
        self.log("train/ap", self.train_ap, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        out = self(x)
        loss = self.criterion_eval(out, y)

        # Metrics
        self.val_accuracy(out, y)
        self.val_auroc(out, y)
        self.val_ap(out, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True)
        self.log("val/ap", self.val_ap, on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch

        out = self(x)
        loss = self.criterion_eval(out, y)

        # Metrics
        self.test_accuracy(out, y)
        self.test_auroc(out, y)
        self.test_ap(out, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_accuracy)
        self.log("test/auroc", self.test_auroc)
        self.log("test/ap", self.test_ap)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LightningSRRegressor(pl.LightningModule):
    def __init__(
        self,
        n_hidden_sets: List[int],
        n_elements: List[int],
        d: List[int],
        n_hidden_channels: Optional[List] = None,
        learning_rate: float = 0.001,
        scaler: Optional[any] = None,
    ) -> None:
        super(LightningSRRegressor, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.scaler = scaler

        self.sr_regressor = SRRegressor(
            n_hidden_sets[0], n_elements[0], d[0], n_hidden_channels
        )

        # Metrics
        self.train_r2 = R2Score()
        self.train_rmse = MeanSquaredError(squared=False)
        self.train_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()
        self.test_r2 = R2Score()
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_mae = MeanAbsoluteError()

    def forward(self, x):
        return self.sr_regressor(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)
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
        self.train_rmse(out, y)
        self.train_mae(out, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/r2", self.train_r2, on_step=False, on_epoch=True)
        self.log("train/rmse", self.train_rmse, on_step=False, on_epoch=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        out = self(x)
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
        self.val_rmse(out, y)
        self.val_mae(out, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True)
        self.log("val/rmse", self.val_rmse, on_step=False, on_epoch=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch

        out = self(x)
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
        self.test_rmse(out, y)
        self.test_mae(out, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/r2", self.test_r2, on_step=False, on_epoch=True)
        self.log("test/rmse", self.test_rmse, on_step=False, on_epoch=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
