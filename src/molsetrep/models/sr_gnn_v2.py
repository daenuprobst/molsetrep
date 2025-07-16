from typing import List, Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch_geometric.nn import (
    GAT,
    MLP,
    global_mean_pool,
)
from torch_geometric.utils import unbatch
from torcheval.metrics import BinaryAUPRC, MulticlassAUPRC
from torchmetrics.classification import AUROC, Accuracy, BinaryCalibrationError, F1Score
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
    SpearmanCorrCoef,
    R2Score,
)

from molsetrep.models import DeepSet, SetRep, SetTransformer
from molsetrep.utils.spearman_loss import spearman_loss


def graph_batch_to_set(X, batch, dim):
    out_unbatched = unbatch(X, batch.batch)

    batch_size = len(out_unbatched)
    max_cardinality = max([b.shape[0] for b in out_unbatched])

    X_set = torch.zeros((batch_size, max_cardinality, dim), device=batch.x.device)

    for i, b in enumerate(out_unbatched):
        X_set[i, : b.shape[0], :] = b

    return X_set


class SRGNNClassifierV2(torch.nn.Module):
    def __init__(
        self,
        n_hidden_sets: int,
        n_elements: int,
        num_layers: int,
        in_channels: int,
        in_edge_channels: int,
        n_hidden_channels: Optional[List] = None,
        n_classes: int = 2,
        gnn: Optional[torch.nn.Module] = None,
        set_layer: str = "setrep",
        gnn_dropout: float = 0.0,
        node_encoder_out: int = 0,
        edge_encoder_out: int = 0,
        descriptors: bool = False,
        n_descriptors: int = 200,
        descriptor_mlp: bool = True,
        descriptor_mlp_dropout: float = 0.25,
        descriptor_mlp_bn: bool = True,
        descriptor_mlp_out: int = 32,
        pool: bool = True,
    ):
        super(SRGNNClassifierV2, self).__init__()

        if n_classes == 2:
            n_classes = 1

        if n_hidden_channels is None or len(n_hidden_channels) < 2:
            n_hidden_channels = [64, 32]

        self.channels = in_channels if node_encoder_out == 0 else node_encoder_out

        self.edge_channels = (
            in_edge_channels if edge_encoder_out == 0 else edge_encoder_out
        )

        self.n_hidden_channels = n_hidden_channels
        self.gnn = gnn
        self.gnn_dropout = gnn_dropout
        self.descriptors = descriptors
        self.descriptor_mlp = descriptor_mlp
        self.pool = pool

        self.node_encoder = torch.nn.Linear(in_channels, self.channels)
        self.edge_encoder = torch.nn.Linear(in_edge_channels, self.edge_channels)

        if self.descriptors and self.descriptor_mlp:
            self.desc_mlp = MLP(
                in_channels=n_descriptors,
                hidden_channels=n_descriptors // 2,
                out_channels=descriptor_mlp_out,
                num_layers=2,
            )

        if self.gnn is None:
            self.gnn = GAT(
                self.channels,
                n_hidden_channels[0],
                num_layers,
                out_channels=n_hidden_channels[0],
                edge_dim=self.edge_channels,
                act="leakyrelu",
                jk="cat",
                dropout=self.gnn_dropout,
                v2=True,
                residual=True,
                heads=2,
            )

        if set_layer == "setrep":
            self.set_rep = SetRep(
                n_hidden_sets,
                n_elements,
                self.n_hidden_channels[0],
                self.n_hidden_channels[0],
            )
        elif set_layer == "transformer":
            self.set_rep = SetTransformer(
                self.n_hidden_channels[0],
                self.n_hidden_channels[0],
                1,
                num_heads=8,
                num_inds=16,
                dim_hidden=256,
            )
        elif set_layer == "deepset":
            self.set_rep = DeepSet(
                self.n_hidden_channels[0], self.n_hidden_channels[0], 1
            )
        else:
            raise ValueError(f"Set layer '{set_layer}' not implemented.")

        descriptors_dim = 0
        if descriptors and descriptor_mlp:
            descriptors_dim = descriptor_mlp_out
        elif descriptors:
            descriptors_dim = n_descriptors

        pool_factor = 2
        if not pool:
            pool_factor = 1

        # self.mlp = MLP(
        #     in_channels=n_hidden_channels[0] * pool_factor + descriptors_dim,
        #     hidden_channels=n_hidden_channels[1],
        #     out_channels=1,
        #     num_layers=5,
        #     dropout=0.15,
        # )

        self.mlp = torch.nn.Linear(
            n_hidden_channels[0] * pool_factor + descriptors_dim, 1
        )

    def forward_no_desc(self, batch):
        x = self.node_encoder(batch.x.float())
        e = self.edge_encoder(batch.edge_attr.float())

        out = self.gnn(x, batch.edge_index, edge_attr=e)

        t = graph_batch_to_set(out, batch, self.n_hidden_channels[0])
        t = self.set_rep(t)

        if self.pool:
            p = global_mean_pool(out, batch.batch)
            out = self.mlp(torch.cat((t, p), -1))
        else:
            out = self.mlp(t)

        return out.squeeze(1)

    def forward_desc(self, batch):
        # x = self.node_encoder(batch.x.float())
        # e = self.edge_encoder(batch.edge_attr.float())
        x = batch.x.float()
        e = batch.edge_attr.float()
        d = self.desc_mlp(batch.descriptors)

        out = self.gnn(x, batch.edge_index, edge_attr=e)

        t = graph_batch_to_set(out, batch, self.n_hidden_channels[0])
        t = self.set_rep(t)

        if self.pool:
            p = global_mean_pool(out, batch.batch)
            out = self.mlp(torch.cat((t, p, d), -1))
        else:
            out = self.mlp(torch.cat((t, d), -1))

        return torch.out.squeeze(1)

    def forward(self, batch):
        if self.descriptors:
            return self.forward_desc(batch)

        return self.forward_no_desc(batch)


class SRGNNRegressorV2(torch.nn.Module):
    def __init__(
        self,
        n_hidden_sets: int,
        n_elements: int,
        num_layers: int,
        in_channels: int,
        in_edge_channels: int,
        n_hidden_channels: Optional[List] = None,
        gnn: Optional[torch.nn.Module] = None,
        set_layer: str = "setrep",
        gnn_dropout: float = 0.0,
        node_encoder_out: int = 0,
        edge_encoder_out: int = 0,
        descriptors: bool = False,
        n_descriptors: int = 200,
        descriptor_mlp: bool = True,
        descriptor_mlp_dropout: float = 0.25,
        descriptor_mlp_bn: bool = True,
        descriptor_mlp_out: int = 32,
        pool: bool = True,
    ):
        super(SRGNNRegressorV2, self).__init__()

        if n_hidden_channels is None or len(n_hidden_channels) < 2:
            n_hidden_channels = [64, 32]

        self.channels = in_channels if node_encoder_out == 0 else node_encoder_out

        self.edge_channels = (
            in_edge_channels if edge_encoder_out == 0 else edge_encoder_out
        )

        self.n_hidden_channels = n_hidden_channels
        self.gnn = gnn
        self.gnn_dropout = gnn_dropout
        self.descriptors = descriptors
        self.descriptor_mlp = descriptor_mlp
        self.pool = pool

        self.node_encoder = torch.nn.Linear(in_channels, self.channels)
        self.edge_encoder = torch.nn.Linear(in_edge_channels, self.edge_channels)

        if self.descriptors and self.descriptor_mlp:
            self.desc_mlp = MLP(
                in_channels=n_descriptors,
                hidden_channels=n_descriptors // 2,
                out_channels=descriptor_mlp_out,
                num_layers=2,
            )

        if self.gnn is None:
            self.gnn = GAT(
                self.channels,
                n_hidden_channels[0],
                num_layers,
                out_channels=n_hidden_channels[0],
                edge_dim=self.edge_channels,
                act="leakyrelu",
                jk="cat",
                dropout=self.gnn_dropout,
                v2=True,
                residual=True,
                heads=2,
            )

        if set_layer == "setrep":
            self.set_rep = SetRep(
                n_hidden_sets,
                n_elements,
                self.n_hidden_channels[0],
                self.n_hidden_channels[0],
            )
        elif set_layer == "transformer":
            self.set_rep = SetTransformer(
                self.n_hidden_channels[0],
                self.n_hidden_channels[0],
                1,
                num_heads=8,
                num_inds=16,
                dim_hidden=256,
            )
        elif set_layer == "deepset":
            self.set_rep = DeepSet(
                self.n_hidden_channels[0], self.n_hidden_channels[0], 1
            )
        else:
            raise ValueError(f"Set layer '{set_layer}' not implemented.")

        descriptors_dim = 0
        if descriptors and descriptor_mlp:
            descriptors_dim = descriptor_mlp_out
        elif descriptors:
            descriptors_dim = n_descriptors

        pool_factor = 2
        if not pool:
            pool_factor = 1

        # self.mlp = MLP(
        #     in_channels=n_hidden_channels[0] * pool_factor + descriptors_dim,
        #     hidden_channels=n_hidden_channels[1],
        #     out_channels=1,
        #     num_layers=5,
        #     dropout=0.15,
        # )

        self.mlp = torch.nn.Linear(
            n_hidden_channels[0] * pool_factor + descriptors_dim, 1
        )

    def forward_no_desc(self, batch):
        x = self.node_encoder(batch.x.float())
        e = self.edge_encoder(batch.edge_attr.float())

        out = self.gnn(x, batch.edge_index, edge_attr=e)

        t = graph_batch_to_set(out, batch, self.n_hidden_channels[0])
        t = self.set_rep(t)

        if self.pool:
            p = global_mean_pool(out, batch.batch)
            out = self.mlp(torch.cat((t, p), -1))
        else:
            out = self.mlp(t)

        return out.squeeze(1)

    def forward_desc(self, batch):
        x = self.node_encoder(batch.x.float())
        e = self.edge_encoder(batch.edge_attr.float())
        d = self.desc_mlp(batch.descriptors)

        out = self.gnn(x, batch.edge_index, edge_attr=e)

        t = graph_batch_to_set(out, batch, self.n_hidden_channels[0])
        t = self.set_rep(t)

        if self.pool:
            p = global_mean_pool(out, batch.batch)
            out = self.mlp(torch.cat((t, p, d), -1))
        else:
            out = self.mlp(torch.cat((t, d), -1))

        return out.squeeze(1)

    def forward(self, batch):
        if self.descriptors:
            return self.forward_desc(batch)

        return self.forward_no_desc(batch)


class LightningSRGNNClassifierV2(pl.LightningModule):
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
        gnn_dropout: float = 0.0,
        node_encoder_out: int = 0,
        edge_encoder_out: int = 0,
        descriptors: bool = False,
        n_descriptors: int = 200,
        descriptor_mlp: bool = True,
        descriptor_mlp_dropout: float = 0.25,
        descriptor_mlp_bn: bool = True,
        descriptor_mlp_out: int = 32,
    ) -> None:
        super(LightningSRGNNClassifierV2, self).__init__()
        self.save_hyperparameters()

        self.n_classes = n_classes
        self.class_weights = class_weights
        self.learning_rate = learning_rate

        self.gnn_classifier = SRGNNClassifierV2(
            n_hidden_sets[0],
            n_elements[0],
            n_layers,
            n_in_channels,
            n_edge_channels,
            n_hidden_channels,
            n_classes,
            gnn=gnn_layer,
            set_layer=set_layer,
            gnn_dropout=gnn_dropout,
            node_encoder_out=node_encoder_out,
            edge_encoder_out=edge_encoder_out,
            descriptors=descriptors,
            n_descriptors=n_descriptors,
            descriptor_mlp=descriptor_mlp,
            descriptor_mlp_dropout=descriptor_mlp_dropout,
            descriptor_mlp_bn=descriptor_mlp_bn,
            descriptor_mlp_out=descriptor_mlp_out,
        )

        # Criterions
        args = {}
        if class_weights is not None:
            args = {"weight": torch.FloatTensor(self.class_weights).to(self.device)}

            if n_classes == 2:
                args = {
                    "weight": torch.FloatTensor(
                        [self.class_weights[1] / self.class_weights[0]]
                    ).to(self.device)
                }

        self.criterion = CrossEntropyLoss(**args)
        self.criterion_eval = CrossEntropyLoss()

        if n_classes == 2:
            self.criterion = BCEWithLogitsLoss(**args)
            self.criterion_eval = BCEWithLogitsLoss()

        args = {"task": "multiclass", "num_classes": n_classes}
        if n_classes == 2:
            self.train_auprc = BinaryAUPRC()
            self.val_auprc = BinaryAUPRC()
            self.test_auprc = BinaryAUPRC()
            args = {"task": "binary"}
        else:
            self.train_auprc = MulticlassAUPRC(num_classes=n_classes)
            self.val_auprc = MulticlassAUPRC(num_classes=n_classes)
            self.test_auprc = MulticlassAUPRC(num_classes=n_classes)

        # Metrics
        self.train_accuracy = Accuracy(**args)
        self.train_auroc = AUROC(**args)
        self.train_f1 = F1Score(**args)
        self.val_accuracy = Accuracy(**args)
        self.val_auroc = AUROC(**args)
        self.val_f1 = F1Score(**args)
        self.test_accuracy = Accuracy(**args)
        self.test_auroc = AUROC(**args)
        self.test_f1 = F1Score(**args)

    def get_pred(self, logits):
        if self.n_classes > 2:
            return torch.argmax(logits, dim=1)
        else:
            return torch.sigmoid(logits).round().long()

    def forward(self, x):
        return self.gnn_classifier(x)

    def training_step(self, batch, batch_idx):
        y = batch.y

        out = self(batch)
        loss = self.criterion(out, y.float())

        pred = self.get_pred(out)
        # Metrics
        self.train_accuracy(pred, y)
        self.train_auroc(pred, y)
        self.train_auprc.update(pred, y)
        self.train_f1(pred, y)

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
            self.train_auprc.compute(),
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

        loss = self.criterion_eval(out, y.float())
        pred = self.get_pred(out)

        # Metrics
        self.val_accuracy(pred, y)
        self.val_auroc(pred, y)
        self.val_auprc.update(pred, y)
        self.val_f1(pred, y)

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
            "val/auprc",
            self.val_auprc.compute(),
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, batch_size=len(y))

    def test_step(self, test_batch, batch_idx):
        y = test_batch.y

        out = self(test_batch)
        loss = self.criterion_eval(out, y.float())
        pred = self.get_pred(out)

        # Metrics
        self.test_accuracy(pred, y)
        self.test_auroc(pred, y)
        self.test_auprc.update(pred, y)
        self.test_f1(pred, y)

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
            self.test_auprc.compute(),
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )
        self.log(
            "test/f1", self.test_f1, on_step=False, on_epoch=True, batch_size=len(y)
        )

    def predict_step(self, test_batch, batch_idx):
        y = test_batch.y

        out = self(test_batch)
        loss = self.criterion_eval(out, y.float())
        pred = self.get_pred(out)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, "min", patience=25, factor=0.75, min_lr=0.00001
        # )
        # return [optimizer], [
        #     {"scheduler": scheduler, "interval": "epoch", "monitor": "val/loss"}
        # ]


class LightningSRGNNRegressorV2(pl.LightningModule):
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
        gnn_dropout: float = 0.0,
        node_encoder_out: int = 0,
        edge_encoder_out: int = 0,
        descriptors: bool = False,
        n_descriptors: int = 200,
        descriptor_mlp: bool = True,
        descriptor_mlp_dropout: float = 0.25,
        descriptor_mlp_bn: bool = True,
        descriptor_mlp_out: int = 32,
        pool: bool = True,
        hybrid_loss: bool = False,
        ranking_loss_weight: float = 1.5,
    ) -> None:
        super(LightningSRGNNRegressorV2, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.scaler = scaler
        self.hybrid_loss = hybrid_loss
        self.ranking_loss_weight = ranking_loss_weight

        self.gnn_regressor = SRGNNRegressorV2(
            n_hidden_sets[0],
            n_elements[0],
            n_layers,
            n_in_channels,
            n_edge_channels,
            n_hidden_channels,
            gnn=gnn_layer,
            set_layer=set_layer,
            gnn_dropout=gnn_dropout,
            node_encoder_out=node_encoder_out,
            edge_encoder_out=edge_encoder_out,
            descriptors=descriptors,
            n_descriptors=n_descriptors,
            descriptor_mlp=descriptor_mlp,
            descriptor_mlp_dropout=descriptor_mlp_dropout,
            descriptor_mlp_bn=descriptor_mlp_bn,
            descriptor_mlp_out=descriptor_mlp_out,
            pool=pool,
        )

        # Metrics
        self.train_r2 = R2Score()
        self.train_pearson = PearsonCorrCoef()
        self.train_spearman = SpearmanCorrCoef()
        self.train_rmse = MeanSquaredError(squared=False)
        self.train_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self.val_pearson = PearsonCorrCoef()
        self.val_spearman = SpearmanCorrCoef()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()
        self.test_r2 = R2Score()
        self.test_pearson = PearsonCorrCoef()
        self.test_spearman = SpearmanCorrCoef()
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_mae = MeanAbsoluteError()

    def forward(self, x):
        return self.gnn_regressor(x)

    def training_step(self, batch, batch_idx):
        y = batch.y

        out = self(batch)

        if self.hybrid_loss:
            loss = self.ranking_loss_weight * spearman_loss(out, y) + F.l1_loss(out, y)
        else:
            loss = F.l1_loss(out, y)

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
        self.train_spearman(out.to(self.device), y.to(self.device))
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
            "train/spearman",
            self.train_spearman,
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

        if self.hybrid_loss:
            loss = self.ranking_loss_weight * spearman_loss(out, y) + F.l1_loss(out, y)
        else:
            loss = F.l1_loss(out, y)

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
        self.val_spearman(out.to(self.device), y.to(self.device))
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
            "val/spearman",
            self.val_spearman,
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

        if self.hybrid_loss:
            loss = self.ranking_loss_weight * spearman_loss(out, y) + F.l1_loss(out, y)
        else:
            loss = F.l1_loss(out, y)

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
        self.test_spearman(out.to(self.device), y.to(self.device))
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
            "test/spearman",
            self.test_spearman,
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, "min", patience=25, factor=0.75, min_lr=0.0001
        # )
        # return [optimizer], [
        #     {"scheduler": scheduler, "interval": "epoch", "monitor": "val/loss"}
        # ]
