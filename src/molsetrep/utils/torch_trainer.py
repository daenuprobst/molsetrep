from typing import Optional, Iterable

import torch

from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import DataLoader
from torcheval.metrics.metric import Metric

from molsetrep.utils.loss_meter import LossMeter


class TorchTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        n_epochs: int,
        train_metrics: Iterable[Metric],
        valid_metrics: Iterable[Metric],
        test_metrics: Iterable[Metric],
        scheduler: Optional[LRScheduler] = None,
        monitor_metric: Optional[int] = None,
        monitor_lower_is_better: bool = True,
        device: Optional[torch.device] = None,
        silent: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.train_metrics = train_metrics
        self.valid_metrics = valid_metrics
        self.test_metrics = test_metrics
        self.scheduler = scheduler
        self.monitor_metric = monitor_metric
        self.monitor_lower_is_better = monitor_lower_is_better

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model.to(self.device)

        self.silent = silent

        self.train_loss = LossMeter()
        self.valid_loss = LossMeter()
        self.test_loss = LossMeter()

        self.best_model = None
        self.best_epoch = 0
        self.best_value = 99999

        if not self.monitor_lower_is_better:
            self.best_value = -self.best_value

        self.hidden_set_history = []

    def train_step(self, X, y):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return output, loss

    def valid_step(self, X, y):
        self.model.eval()
        output = self.model(X)
        loss = self.criterion(output, y)
        return output, loss

    def test_step(self, X, y):
        self.model.eval()
        output = self.model(X)
        loss = self.criterion(output, y)
        return output, loss

    def train(
        self, train_loader: DataLoader, valid_loader: Optional[DataLoader] = None
    ) -> None:
        for epoch in range(self.n_epochs):
            for X, y in train_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                output, loss = self.train_step(X, y)
                self.train_loss.update(loss)

                y_pred = output

                if output.size(dim=-1) == 2:
                    y_pred = output.max(1)[1]
                for metric in self.train_metrics:
                    metric.update(y_pred.cpu(), y.cpu())

            if self.scheduler is not None:
                self.scheduler.step()

            for X, y in valid_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                output, loss = self.valid_step(X, y)
                self.valid_loss.update(loss)

                y_pred = output

                if output.size(dim=-1) == 2:
                    y_pred = output.max(1)[1]
                for metric in self.valid_metrics:
                    metric.update(y_pred.cpu(), y.cpu())

            best_epoch = False
            monitored_metric = self.valid_loss

            if self.monitor_metric is not None:
                monitored_metric = self.valid_metrics[self.monitor_metric]

            monitored_value = monitored_metric.compute().item()

            # Build history of hidden sets
            self.hidden_set_history.append(
                (epoch, monitored_value, self.model.Wc.cpu().detach().numpy())
            )

            # Remember the best epoch
            if self.monitor_lower_is_better:
                best_epoch = monitored_value < self.best_value
            else:
                best_epoch = monitored_value > self.best_value

            if best_epoch:
                self.best_model = self.model.state_dict()
                self.best_value = monitored_value
                self.best_epoch = epoch

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

            if not self.silent:
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

            self.hidden_set_history = sorted(
                self.hidden_set_history,
                reverse=not self.monitor_lower_is_better,
                key=lambda x: x[1],
            )

    def test(self, test_loader: DataLoader, average_n_epochs: int = 0) -> None:
        self.model.load_state_dict(self.best_model)

        if average_n_epochs > 0:
            self.model.Wc = torch.nn.Parameter(
                torch.mean(
                    torch.FloatTensor(
                        [hs[2] for hs in self.hidden_set_history[:average_n_epochs]]
                    ),
                    dim=0,
                ).to(self.device)
            )

        for X, y in test_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            output, loss = self.test_step(X, y)
            self.test_loss.update(loss)

            y_pred = output

            if output.size(dim=-1) == 2:
                y_pred = output.max(1)[1]
            for metric in self.test_metrics:
                metric.update(y_pred.cpu(), y.cpu())

        results = []
        if not self.silent:
            print("------------------------------------------------")
            print(f"Using Epoch {self.best_epoch + 1} for testing...")
            print(f"Test loss: {round(self.test_loss.compute(), 3)}")

            for metric in self.test_metrics:
                print(
                    f"Test {type(metric).__name__}:", round(metric.compute().item(), 3)
                )
        else:
            result = {
                "best_epoch": self.best_epoch + 1,
                "loss": self.test_loss.compute(),
            }

            for metric in self.test_metrics:
                result[type(metric).__name__] = metric.compute().item()

            results.append(result)

        self.test_loss.reset()

        for metric in self.test_metrics:
            metric.reset()

        return results
