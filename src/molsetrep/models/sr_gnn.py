from typing import List, Optional, Literal
import torch
import lightning.pytorch as pl


from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch_geometric.nn import GIN
from torch_geometric.utils import unbatch

from torchmetrics.classification import (
    Accuracy,
    AUROC,
    AveragePrecision,
    F1Score,
)
from torchmetrics.regression import (
    R2Score,
    MeanSquaredError,
    MeanAbsoluteError,
    PearsonCorrCoef,
)

from molsetrep.models import GINE, MLP, SetRep, SetTransformer, DeepSet
from molsetrep.metrics import AUPRC


def graph_batch_to_set(X, batch, dim):
    out_unbatched = unbatch(X, batch.batch)

    batch_size = len(out_unbatched)
    max_cardinality = max([b.shape[0] for b in out_unbatched])

    X_set = torch.zeros((batch_size, max_cardinality, dim), device=batch.x.device)

    for i, b in enumerate(out_unbatched):
        X_set[i, : b.shape[0], :] = b

    return X_set


class SRGNNClassifier(torch.nn.Module):
    def __init__(
        self,
        n_hidden_sets: int,
        n_elements: int,
        in_channels: int,
        num_layers: int,
        in_edge_channels: int,
        n_hidden_channels: Optional[List] = None,
        n_classes: int = 2,
        gnn: Optional[torch.nn.Module] = None,
        set_layer: str = "setrep",
    ):
        super(SRGNNClassifier, self).__init__()

        if n_hidden_channels is None or len(n_hidden_channels) < 2:
            n_hidden_channels = [32, 16]

        self.n_hidden_channels = n_hidden_channels
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
                self.gnn = GIN(in_channels, self.n_hidden_channels[0], num_layers)

        if set_layer == "setrep":
            self.set_rep = SetRep(
                n_hidden_sets,
                n_elements,
                self.n_hidden_channels[0],
                self.n_hidden_channels[0],
            )
        elif set_layer == "transformer":
            self.set_rep = SetTransformer(
                self.n_hidden_channels[0], self.n_hidden_channels[0], 1
            )
        elif set_layer == "deepset":
            self.set_rep = DeepSet(
                self.n_hidden_channels[0], self.n_hidden_channels[0], 1
            )
        else:
            raise ValueError(f"Set layer '{set_layer}' not implemented.")

        self.mlp = MLP(self.n_hidden_channels[0], self.n_hidden_channels[1], n_classes)

    def forward(self, batch):
        if self.in_edge_channels > 0:
            out = self.gnn(
                batch.x.float(), batch.edge_index, edge_attr=batch.edge_attr.float()
            )
        else:
            out = self.gnn(batch.x.float(), batch.edge_index)

        t = graph_batch_to_set(out, batch, self.n_hidden_channels[0])
        t = self.set_rep(t)
        out = self.mlp(t)

        return F.log_softmax(out, dim=1)


class SRGNNRegressor(torch.nn.Module):
    def __init__(
        self,
        n_hidden_sets: int,
        n_elements: int,
        in_channels: int,
        num_layers: int,
        in_edge_channels: int,
        n_hidden_channels: Optional[List] = None,
        gnn: Optional[torch.nn.Module] = None,
        set_layer: str = "setrep",
        n_tasks: int = 1,
    ):
        super(SRGNNRegressor, self).__init__()

        if n_hidden_channels is None or len(n_hidden_channels) < 2:
            n_hidden_channels = [32, 16]

        self.n_hidden_channels = n_hidden_channels
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

        if set_layer == "setrep":
            self.set_rep = SetRep(
                n_hidden_sets,
                n_elements,
                self.n_hidden_channels[0],
                self.n_hidden_channels[0],
            )
        elif set_layer == "transformer":
            self.set_rep = SetTransformer(
                self.n_hidden_channels[0], self.n_hidden_channels[0], 1
            )
        elif set_layer == "deepset":
            self.set_rep = DeepSet(
                self.n_hidden_channels[0], self.n_hidden_channels[0], 1
            )
        else:
            raise ValueError(f"Set layer '{set_layer}' not implemented.")

        self.mlp = MLP(n_hidden_channels[0], n_hidden_channels[1], n_tasks)

    def forward(self, batch):
        if self.in_edge_channels > 0:
            out = self.gnn(
                batch.x.float(), batch.edge_index, edge_attr=batch.edge_attr.float()
            )
        else:
            out = self.gnn(batch.x.float(), batch.edge_index)

        t = graph_batch_to_set(out, batch, self.n_hidden_channels[0])
        t = self.set_rep(t)
        out = self.mlp(t)

        return out.squeeze(1)


class LightningSRGNNClassifier(pl.LightningModule):
    def __init__(
        self,
        n_hidden_sets: List[int],
        n_elements: List[int],
        n_layers: int,
        n_in_channels: int,
        n_edge_channels: int,
        n_classes: int,
        n_hidden_channels: Optional[List] = None,
        gnn_layer: Optional[torch.nn.Module] = None,
        set_layer: str = "setrep",
        class_weights: Optional[List] = None,
        learning_rate: float = 0.001,
        metrics: Optional[List[str]] = None,
        metrics_task: Literal["binary", "multiclass", "multilabel"] = "multiclass",
    ) -> None:
        super(LightningSRGNNClassifier, self).__init__()
        self.save_hyperparameters()

        self.class_weights = class_weights
        self.learning_rate = learning_rate

        self.gnn_classifier = SRGNNClassifier(
            n_hidden_sets[0],
            n_elements[0],
            n_in_channels,
            n_layers,
            n_edge_channels,
            n_hidden_channels,
            n_classes,
            gnn=gnn_layer,
            set_layer=set_layer,
        )

        self.metrics = metrics
        if self.metrics is None:
            self.metrics = ["Accuracy", "AUROC", "AUPRC", "F1Score", "AveragePrecision"]

        self.metrics_task = metrics_task

        # Criterions
        self.criterion = CrossEntropyLoss()
        if class_weights is not None:
            self.criterion = CrossEntropyLoss(
                weight=torch.FloatTensor(self.class_weights).to(self.device)
            )

        self.criterion_eval = CrossEntropyLoss()

        # Metrics
        self.train_accuracy = Accuracy(task=self.metrics_task, num_classes=n_classes)
        self.train_auroc = AUROC(task=self.metrics_task, num_classes=n_classes)
        self.train_auprc = AUPRC(num_classes=n_classes)
        self.train_f1 = F1Score(task=self.metrics_task, num_classes=n_classes)
        self.train_ap = AveragePrecision(task=self.metrics_task, num_classes=n_classes)
        self.val_accuracy = Accuracy(task=self.metrics_task, num_classes=n_classes)
        self.val_auroc = AUROC(task=self.metrics_task, num_classes=n_classes)
        self.val_auprc = AUPRC(num_classes=n_classes)
        self.val_f1 = F1Score(task=self.metrics_task, num_classes=n_classes)
        self.val_ap = AveragePrecision(task=self.metrics_task, num_classes=n_classes)
        self.test_accuracy = Accuracy(task=self.metrics_task, num_classes=n_classes)
        self.test_auroc = AUROC(task=self.metrics_task, num_classes=n_classes)
        self.test_auprc = AUPRC(num_classes=n_classes)
        self.test_f1 = F1Score(task=self.metrics_task, num_classes=n_classes)
        self.test_ap = AveragePrecision(task=self.metrics_task, num_classes=n_classes)

    def forward(self, x):
        return self.gnn_classifier(x)

    def training_step(self, batch, batch_idx):
        y = batch.y

        out = self(batch)
        loss = self.criterion(out, y)

        # Metrics
        if self.metrics is not None and "Accuracy" in self.metrics:
            self.train_accuracy(out, y)
            self.log(
                "train/acc",
                self.train_accuracy,
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

        if self.metrics is not None and "AUROC" in self.metrics:
            self.train_auroc(out, y)
            self.log(
                "train/auroc",
                self.train_auroc,
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

        if self.metrics is not None and "AUPRC" in self.metrics:
            self.train_auprc(out, y)
            self.log(
                "train/auprc",
                self.train_auprc,
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

        if self.metrics is not None and "F1Score" in self.metrics:
            self.train_f1(out, y)
            self.log(
                "train/f1",
                self.train_f1,
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

        if self.metrics is not None and "AveragePrecision" in self.metrics:
            self.train_ap(out, y)
            self.log(
                "train/ap",
                self.train_ap,
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=len(y))

        return loss

    def validation_step(self, val_batch, batch_idx):
        y = val_batch.y

        out = self(val_batch)
        loss = self.criterion_eval(out, y)

        # Metrics
        if self.metrics is not None and "Accuracy" in self.metrics:
            self.val_accuracy(out, y)
            self.log(
                "val/acc",
                self.train_accuracy,
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

        if self.metrics is not None and "AUROC" in self.metrics:
            self.val_auroc(out, y)
            self.log(
                "val/auroc",
                self.train_auroc,
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

        if self.metrics is not None and "AUPRC" in self.metrics:
            self.val_auprc(out, y)
            self.log(
                "val/auprc",
                self.train_auprc,
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

        if self.metrics is not None and "F1Score" in self.metrics:
            self.val_f1(out, y)
            self.log(
                "val/f1",
                self.train_f1,
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

        if self.metrics is not None and "AveragePrecision" in self.metrics:
            self.val_ap(out, y)
            self.log(
                "val/ap",
                self.train_ap,
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=len(y))

    def test_step(self, test_batch, batch_idx):
        y = test_batch.y

        out = self(test_batch)
        loss = self.criterion_eval(out, y)

        # Metrics
        if self.metrics is not None and "Accuracy" in self.metrics:
            self.test_accuracy(out, y)
            self.log(
                "test/acc",
                self.train_accuracy,
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

        if self.metrics is not None and "AUROC" in self.metrics:
            self.test_auroc(out, y)
            self.log(
                "test/auroc",
                self.train_auroc,
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

        if self.metrics is not None and "AUPRC" in self.metrics:
            self.test_auprc(out, y)
            self.log(
                "test/auprc",
                self.train_auprc,
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

        if self.metrics is not None and "F1Score" in self.metrics:
            self.test_f1(out, y)
            self.log(
                "test/f1",
                self.train_f1,
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

        if self.metrics is not None and "AveragePrecision" in self.metrics:
            self.test_ap(out, y)
            self.log(
                "test/ap",
                self.train_ap,
                on_step=False,
                on_epoch=True,
                batch_size=len(y),
            )

        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=len(y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LightningSRGNNRegressor(pl.LightningModule):
    def __init__(
        self,
        n_hidden_sets: List[int],
        n_elements: List[int],
        n_layers: int,
        n_in_channels: int,
        n_edge_channels: int,
        n_hidden_channels: Optional[List] = None,
        gnn_layer: Optional[torch.nn.Module] = None,
        set_layer: str = "setrep",
        learning_rate: float = 0.001,
        scaler: Optional[any] = None,
        n_tasks: int = 1,
    ) -> None:
        super(LightningSRGNNRegressor, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.scaler = scaler

        self.gnn_regressor = SRGNNRegressor(
            n_hidden_sets[0],
            n_elements[0],
            n_in_channels,
            n_layers,
            n_edge_channels,
            n_hidden_channels,
            gnn=gnn_layer,
            set_layer=set_layer,
            n_tasks=n_tasks,
        )

        # Metrics
        self.train_r2 = R2Score()
        self.train_pearson = PearsonCorrCoef()
        self.train_rmse = MeanSquaredError(squared=False)
        self.train_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self.val_pearson = PearsonCorrCoef(num_outputs=11)
        self.val_rmse = MeanSquaredError(squared=False, num_outputs=11)
        self.val_mae = MeanAbsoluteError(num_outputs=11)
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

        print(y)
        print("---")
        print(out)
        print("===")

        # Metrics
        # self.val_r2(out, y)
        self.val_pearson(out.to(self.device), y.to(self.device))
        self.val_rmse(out, y)
        self.val_mae(out, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=len(y))
        # self.log("val/r2", self.val_r2, on_step=False, on_epoch=True, batch_size=len(y))
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
