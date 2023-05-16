from typing import Optional, Iterable

import torch
import torch.nn.functional as F

from torch_geometric.data import DataLoader

import numpy as np

from torcheval.metrics import R2Score, MeanSquaredError, BinaryAccuracy, BinaryAUROC
from torcheval.metrics.metric import Metric

from molsetrep.utils.loss_meter import LossMeter


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        n_epochs: int,
        train_metrics: Iterable[Metric],
        valid_metrics: Iterable[Metric],
        test_metrics: Iterable[Metric],
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.train_metrics = train_metrics
        self.valid_metrics = valid_metrics
        self.test_metrics = test_metrics

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model.to(self.device)

        self.train_loss = LossMeter()
        self.valid_loss = LossMeter()
        self.test_loss = LossMeter()

        self.best_model = None
        self.best_epoch = 0
        self.lowest_valid_loss = 99999

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(batch)
        loss = self.criterion(output, batch.y)
        loss.backward()
        self.optimizer.step()
        return output, loss

    def valid_step(self, batch):
        self.model.eval()
        output = self.model(batch)
        loss = self.criterion(output, batch.y)
        return output, loss

    def test_step(self, batch):
        self.model.eval()
        output = self.model(batch)
        loss = self.criterion(output, batch.y)
        return output, loss

    def train(
        self, train_loader: DataLoader, valid_loader: Optional[DataLoader] = None
    ) -> None:
        for epoch in range(self.n_epochs):
            for batch in train_loader:
                batch.to(self.device)
                output, loss = self.train_step(batch)
                self.train_loss.update(loss)

                y_pred = output

                if output.size(dim=1) == 2:
                    y_pred = output.max(1)[1]
                for metric in self.train_metrics:
                    metric.update(y_pred.cpu(), batch.y.cpu())

            for batch in valid_loader:
                batch.to(self.device)
                output, loss = self.valid_step(batch)
                self.valid_loss.update(loss)

                y_pred = output

                if output.size(dim=1) == 2:
                    y_pred = output.max(1)[1]
                for metric in self.valid_metrics:
                    metric.update(y_pred.cpu(), batch.y.cpu())

            best_epoch = False
            if self.valid_loss.compute() < self.lowest_valid_loss:
                self.best_model = self.model.state_dict()
                self.lowest_valid_loss = self.valid_loss.compute()
                self.best_epoch = epoch
                best_epoch = True

            best_epoch_prefix = "* " if best_epoch else "| "

            train_metrics = []
            for metric in self.train_metrics:
                train_metrics.append(
                    f"{type(metric).__name__}: {round(metric.compute().item(), 3)}"
                )

            valid_metrics = []
            for metric in self.valid_metrics:
                valid_metrics.append(
                    f"{type(metric).__name__}: {round(metric.compute().item(), 3)}"
                )

            print(
                best_epoch_prefix,
                f"Epoch {epoch + 1}:",
                f"Train loss: {round(self.train_loss.compute(), 3)}",
                "(" + ", ".join(train_metrics) + ")",
                f" Valid loss: {round(self.valid_loss.compute(), 3)}",
                "(" + ", ".join(valid_metrics) + ")",
            )

            self.train_loss.reset()
            self.valid_loss.reset()

            for metric in self.train_metrics:
                metric.reset()
            for metric in self.valid_metrics:
                metric.reset()

    def test(self, test_loader: DataLoader) -> None:
        self.model.load_state_dict(self.best_model)
        for batch in test_loader:
            batch.to(self.device)
            output, loss = self.test_step(batch)
            self.test_loss.update(loss)

            if output.size(dim=1) == 2:
                y_pred = output.max(1)[1]
            for metric in self.test_metrics:
                metric.update(y_pred.cpu(), batch.y.cpu())

        print("------------------------------------------------")
        print(f"Using Epoch {self.best_epoch + 1} for testing...")
        print(f"Test loss: {round(self.test_loss.compute(), 3)}")

        for metric in self.test_metrics:
            print(f"Test {type(metric).__name__}:", round(metric.compute().item(), 3))

        self.test_loss.reset()

        for metric in self.test_metrics:
            metric.reset()
