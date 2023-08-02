from typing import List, Optional

import torch
from torch.nn import Module, CrossEntropyLoss
import torch.nn.functional as F

import lightning.pytorch as pl

from torchmetrics.classification import Accuracy, AUROC
from torchmetrics.regression import R2Score, MeanSquaredError, MeanAbsoluteError

from molsetrep.models.set_rep import SetRep
from molsetrep.models.mlp import MLP


class TripleSRClassifier(Module):
    def __init__(
        self,
        n_hidden_sets: List[int],
        n_elements: List[int],
        d: List[int],
        n_classes: int,
        n_hidden_channels: Optional[List] = None,
    ) -> None:
        super(TripleSRClassifier, self).__init__()

        if n_hidden_channels is None:
            n_hidden_channels = [32, 16]

        self.set_rep_0 = SetRep(
            n_hidden_sets[0], n_elements[0], d[0], n_hidden_channels[0]
        )
        self.set_rep_1 = SetRep(
            n_hidden_sets[1], n_elements[1], d[1], n_hidden_channels[0]
        )
        self.set_rep_2 = SetRep(
            n_hidden_sets[2], n_elements[2], d[2], n_hidden_channels[0]
        )
        self.mlp = MLP(n_hidden_channels[0] * 3, n_hidden_channels[1], n_classes)

    def forward(self, x0, x1, x2):
        x0 = self.set_rep_0(x0)
        x1 = self.set_rep_1(x1)
        x2 = self.set_rep_2(x2)
        return self.mlp(torch.cat((x0, x1, x2), 1))


class TripleSRRegressor(Module):
    def __init__(
        self,
        n_hidden_sets: List[int],
        n_elements: List[int],
        d: List[int],
        n_hidden_channels: Optional[List] = None,
    ) -> None:
        super(TripleSRRegressor, self).__init__()

        if n_hidden_channels is None:
            n_hidden_channels = [32, 16]

        self.set_rep_0 = SetRep(
            n_hidden_sets[0], n_elements[0], d[0], n_hidden_channels[0]
        )
        self.set_rep_1 = SetRep(
            n_hidden_sets[1], n_elements[1], d[1], n_hidden_channels[0]
        )
        self.set_rep_2 = SetRep(
            n_hidden_sets[2], n_elements[2], d[2], n_hidden_channels[0]
        )

        self.mlp = MLP(n_hidden_channels[0] * 3, n_hidden_channels[1], 1)

    def forward(self, x0, x1, x2):
        x0 = self.set_rep_0(x0)
        x1 = self.set_rep_1(x1)
        x2 = self.set_rep_2(x2)

        return self.mlp(torch.cat((x0, x1, x2), 1)).squeeze(1)


class LightningTripleSRClassifier(pl.LightningModule):
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
        super(LightningTripleSRClassifier, self).__init__()
        self.save_hyperparameters()

        self.class_weights = class_weights
        self.learning_rate = learning_rate

        self.sr_classifier = TripleSRClassifier(
            n_hidden_sets, n_elements, d, n_classes, n_hidden_channels
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
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_auroc = AUROC(task="multiclass", num_classes=n_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=n_classes)

    def forward(self, x0, x1, x2):
        return self.sr_classifier(x0, x1, x2)

    def training_step(self, batch, batch_idx):
        x0, x1, x2, y = batch

        out = self(x0, x1, x2)
        loss = self.criterion(out, y)

        # Metrics
        self.train_accuracy(out, y)
        self.train_auroc(out, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_accuracy, on_step=False, on_epoch=True)
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x0, x1, x2, y = val_batch

        out = self(x0, x1, x2)
        loss = self.criterion_eval(out, y)

        # Metrics
        self.val_accuracy(out, y)
        self.val_auroc(out, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        x0, x1, x2, y = test_batch

        out = self(x0, x1, x2)
        loss = self.criterion_eval(out, y)

        # Metrics
        self.test_accuracy(out, y)
        self.test_auroc(out, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_accuracy)
        self.log("test/auroc", self.test_auroc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LightningTripleSRRegressor(pl.LightningModule):
    def __init__(
        self,
        n_hidden_sets: List[int],
        n_elements: List[int],
        d: List[int],
        n_hidden_channels: Optional[List] = None,
        learning_rate: float = 0.001,
    ) -> None:
        super(LightningTripleSRRegressor, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.sr_regressor = TripleSRRegressor(
            n_hidden_sets, n_elements, d, n_hidden_channels
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

    def forward(self, x0, x1, x2):
        return self.sr_regressor(x0, x1, x2)

    def training_step(self, batch, batch_idx):
        x0, x1, x2, y = batch

        out = self(x0, x1, x2)
        loss = F.mse_loss(out, y)

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
        x0, x1, x2, y = val_batch

        out = self(x0, x1, x2)
        loss = F.mse_loss(out, y)

        # Metrics
        self.val_r2(out, y)
        self.val_rmse(out, y)
        self.val_mae(out, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True)
        self.log("val/rmse", self.val_rmse, on_step=False, on_epoch=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        x0, x1, x2, y = test_batch

        out = self(x0, x1, x2)
        loss = F.mse_loss(out, y)

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
