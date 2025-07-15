import os
import pickle
from typing import List, Optional
import lightning.pytorch as pl
import numpy as np
import torch
import typer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import wandb
from rdkit import RDLogger
from tdc.benchmark_group import admet_group
from tdc.metadata import admet_metrics
from wandb import finish as wandb_finish

from molsetrep.encoders import (
    GraphEncoder,
)
from molsetrep.models import (
    LightningSRGNNClassifierV2 as LightningSRGNNClassifier,
)
from molsetrep.models import (
    LightningSRGNNRegressorV2 as LightningSRGNNRegressor,
)
from molsetrep.utils.datasets import (
    CustomDataset,
)

from molsetrep.utils.log_standard_scaler import LogStandardScaler

RDLogger.DisableLog("rdApp.*")


app = typer.Typer(pretty_exceptions_enable=False)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

# torch.set_float32_matmul_precision("medium")

metric_map = {
    "mae": "mae",
    "roc-auc": "auroc",
    "spearman": "spearman",
    "pr-auc": "auprc",
}

task_type_map = {
    "mae": "regression",
    "roc-auc": "classification",
    "spearman": "regression",
    "pr-auc": "classification",
}

mode_map = {
    "mae": "min",
    "roc-auc": "max",
    "spearman": "max",
    "pr-auc": "max",
}

log_scale_sets = [
    "vdss_lombardo",
    "half_life_obach",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
]


def get_model(
    task_type: str,
    n_hidden_sets: List[int],
    n_elements: List[int],
    d: List[int],
    n_classes: int,
    n_hidden_channels: Optional[List[int]] = None,
    class_weights: Optional[List[float]] = None,
    scaler: Optional[any] = None,
    set_layer: str = "setrep",
    learning_rate: float = 0.001,
    n_layers: int = 6,
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
    hybrid_loss: bool = True,
    ranking_loss_weight: float = 1.5,
    **kwargs,
) -> torch.nn.Module:
    if task_type == "classification":
        return LightningSRGNNClassifier(
            n_hidden_sets,
            n_elements,
            n_layers,
            d[0],
            d[1],
            n_classes,
            n_hidden_channels,
            None,
            set_layer,
            class_weights,
            learning_rate,
            gnn_dropout=gnn_dropout,
            node_encoder_out=node_encoder_out,
            edge_encoder_out=edge_encoder_out,
            descriptors=descriptors,
            n_descriptors=n_descriptors,
            descriptor_mlp=descriptor_mlp,
            descriptor_mlp_dropout=descriptor_mlp_dropout,
            descriptor_mlp_bn=descriptor_mlp_bn,
            descriptor_mlp_out=descriptor_mlp_out,
            **kwargs,
        )
    elif task_type == "regression":
        return LightningSRGNNRegressor(
            n_hidden_sets,
            n_elements,
            n_layers,
            d[0],
            d[1],
            n_hidden_channels,
            None,
            set_layer,
            learning_rate,
            scaler,
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
            hybrid_loss=hybrid_loss,
            ranking_loss_weight=ranking_loss_weight,
            **kwargs,
        )


def get_datasets(
    train,
    valid,
    test,
    task_type,
    use_class_weights,
    data_set_name,
    batch_size=64,
    charges=False,
    descriptors=False,
):
    encoder = GraphEncoder(charges=charges, descriptors=descriptors)

    label_dtype = torch.long
    if task_type == "regression":
        label_dtype = torch.float

    scaler = None

    train_y = train.y
    valid_y = valid.y
    test_y = test.y

    if len(train_y.shape) == 1:
        train_y = np.expand_dims(train_y, -1)
        valid_y = np.expand_dims(valid_y, -1)
        test_y = np.expand_dims(test_y, -1)

    train_y = np.array(train_y[:, 0])
    valid_y = np.array(valid_y[:, 0])
    test_y = np.array(test_y[:, 0])

    if task_type == "regression":
        log_scale = data_set_name in log_scale_sets
        scaler = LogStandardScaler(log=log_scale)
        scaler.fit(train_y.reshape(-1, 1))

        train_y = scaler.transform(train_y.reshape(-1, 1)).flatten()
        valid_y = scaler.transform(valid_y.reshape(-1, 1)).flatten()
        test_y = scaler.transform(test_y.reshape(-1, 1)).flatten()

    train_dataset = encoder.encode(
        train.ids,
        train_y,
        label_dtype=label_dtype,
        batch_size=batch_size,
        shuffle=True,
        weighted_sampler=use_class_weights and task_type == "classification",
    )

    valid_dataset = encoder.encode(
        valid.ids,
        valid_y,
        label_dtype=label_dtype,
        batch_size=batch_size,
    )

    test_dataset = encoder.encode(
        test.ids,
        test_y,
        label_dtype=label_dtype,
        batch_size=batch_size,
    )

    return train_dataset, valid_dataset, test_dataset, scaler


def tdc_benchmark(
    use_class_weights: bool = True,
    max_epochs: int = 50,
    batch_size: int = 64,
    n_hidden_sets: Optional[List[int]] = None,
    n_elements: Optional[List[int]] = None,
    n_hidden_channels: Optional[List[int]] = None,
    n_layers: int = 6,
    learning_rate: float = 0.001,
    set_layer: str = "setrep",
    gnn_type: str = "gine",
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
    charges: bool = False,
    variant: Optional[str] = None,
    ckpt_path: str = "best",
):
    group = admet_group(path="data/")
    predictions_list = []

    for seed in [1, 2, 3, 4, 5]:
        predictions = {}
        for benchmark in group:
            name = benchmark["name"]

            if name != "clearance_microsome_az".lower():
                continue

            admet_metric = admet_metrics[name]
            metric = metric_map[admet_metric]
            task_type = task_type_map[admet_metric]
            mode = mode_map[admet_metric]

            if task_type != "regression":
                continue

            if metric == "spearman":
                hybrid_loss = True
            else:
                hybrid_loss = False

            train_val, test = benchmark["train_val"], benchmark["test"]

            train, valid = group.get_train_valid_split(
                benchmark=name, split_type="default", seed=seed
            )

            train_split = CustomDataset.from_df(train, "Drug", ["Y"])
            valid_split = CustomDataset.from_df(valid, "Drug", ["Y"])
            test_split = CustomDataset.from_df(test, "Drug", ["Y"])

            train_loader, valid_loader, test_loader, scaler = get_datasets(
                train_split,
                valid_split,
                test_split,
                task_type,
                use_class_weights,
                name,
                batch_size,
                charges,
                descriptors,
            )

            d = [
                train_loader.dataset[0].num_node_features,
                train_loader.dataset[0].num_edge_features,
            ]

            model = get_model(
                task_type,
                list(n_hidden_sets),
                list(n_elements),
                d,
                2,
                learning_rate=learning_rate,
                scaler=scaler,
                n_hidden_channels=n_hidden_channels,
                set_layer=set_layer,
                n_layers=n_layers,
                gnn_dropout=gnn_dropout,
                node_encoder_out=node_encoder_out,
                edge_encoder_out=edge_encoder_out,
                descriptors=descriptors,
                n_descriptors=n_descriptors,
                descriptor_mlp=descriptor_mlp,
                descriptor_mlp_dropout=descriptor_mlp_dropout,
                descriptor_mlp_bn=descriptor_mlp_bn,
                descriptor_mlp_out=descriptor_mlp_out,
                hybrid_loss=hybrid_loss,
                ranking_loss_weight=ranking_loss_weight,
                pool=pool,
            )

            checkpoint_callback = ModelCheckpoint(monitor=f"val/{metric}", mode=mode)
            learning_rate_callback = LearningRateMonitor(logging_interval="step")
            early_stopping_callback = EarlyStopping(
                monitor=f"val/{metric}", patience=50, mode=mode
            )

            project_name = f"TDC_Regression_Final"
            wandb_logger = wandb.WandbLogger(project=project_name)
            wandb_logger.experiment.config.update(
                {
                    "model": "",
                    "variant": variant,
                    "dataset": name,
                    "batch_size": batch_size,
                    "set_layer": set_layer,
                    "gnn_dropout": gnn_dropout,
                    "node_encoder_out": node_encoder_out,
                    "edge_encoder_out": edge_encoder_out,
                    "descriptors": descriptors,
                    "n_descriptors": n_descriptors,
                    "descriptor_mlp": descriptor_mlp,
                    "descriptor_mlp_dropout": descriptor_mlp_dropout,
                    "descriptor_mlp_bn": descriptor_mlp_bn,
                    "descriptor_mlp_out": descriptor_mlp_out,
                },
                allow_val_change=True,
            )
            wandb_logger.watch(model, log="all")

            trainer = pl.Trainer(
                callbacks=[
                    checkpoint_callback,
                    learning_rate_callback,
                    # early_stopping_callback,
                ],
                max_epochs=max_epochs,
                log_every_n_steps=1,
                logger=wandb_logger,
            )

            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=valid_loader
            )
            test_out = trainer.test(ckpt_path=ckpt_path, dataloaders=test_loader)
            y_pred_test = trainer.predict(ckpt_path=ckpt_path, dataloaders=test_loader)
            y_pred_test = torch.cat(y_pred_test)

            if task_type == "regression":
                y_pred_test = scaler.inverse_transform(
                    y_pred_test.detach().cpu().reshape(-1, 1)
                ).flatten()

            if wandb_logger:
                wandb_logger.finalize("success")
                wandb_finish()

            predictions[name] = y_pred_test
            predictions_list.append(predictions)

    result = group.evaluate_many(predictions_list)
    print(result)
    with open("result.pkl", "wb") as f:
        pickle.dump(result, f)


@app.command()
def main(
    use_class_weights: bool = True,
    max_epochs: int = 150,
    batch_size: int = 64,
    n_hidden_sets: Optional[List[int]] = None,
    n_elements: Optional[List[int]] = None,
    n_hidden_channels: Optional[List[int]] = None,
    n_layers: int = 1,
    learning_rate: float = 0.001,
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
    hybrid_loss: bool = False,
    ranking_loss_weight: float = 1.5,
    charges: bool = False,
    variant: Optional[str] = None,
    ckpt_path: str = "best",
):
    if n_hidden_sets is None:
        n_hidden_sets = [8]

    if n_elements is None:
        n_elements = [8]

    if n_hidden_channels is None:
        n_hidden_channels = [64, 32]

    tdc_benchmark(
        use_class_weights=use_class_weights,
        max_epochs=max_epochs,
        batch_size=batch_size,
        n_hidden_sets=n_hidden_sets,
        n_elements=n_elements,
        n_hidden_channels=n_hidden_channels,
        n_layers=n_layers,
        learning_rate=learning_rate,
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
        hybrid_loss=hybrid_loss,
        ranking_loss_weight=ranking_loss_weight,
        charges=charges,
        variant=variant,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    app()
