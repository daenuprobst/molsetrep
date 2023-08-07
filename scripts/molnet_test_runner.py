from typing import List, Optional
from functools import partial
import typer
import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from wandb import finish as wandb_finish
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import wandb

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from deepchem.trans import undo_transforms

from molsetrep.utils.datasets import (
    get_class_weights,
    molnet_task_loader,
    molnet_loader,
)

from molsetrep.encoders import (
    Encoder,
    SingleSetEncoder,
    DualSetEncoder,
    TripleSetEncoder,
    LigandProtEncoder,
    GraphEncoder,
)
from molsetrep.models import (
    LightningSRClassifier,
    LightningSRRegressor,
    LightningDualSRClassifier,
    LightningDualSRRegressor,
    LightningTripleSRClassifier,
    LightningTripleSRRegressor,
    LightningGNNClassifier,
    LightningGNNRegressor,
    LightningSRGNNClassifier,
    LightningSRGNNRegressor,
)

from molsetrep.data import PDBBindFeaturizer

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

app = typer.Typer(pretty_exceptions_enable=False)


def get_encoder(model_name: str, data_set_name: str, charges: bool = True) -> Encoder:
    if data_set_name == "pdbbind":
        return LigandProtEncoder()
    elif model_name == "msr1":
        return SingleSetEncoder(charges=charges)
    elif model_name == "msr2":
        return DualSetEncoder(charges=charges)
    elif model_name == "msr3":
        return TripleSetEncoder(charges=charges)
    elif model_name == "gnn" or "srgnn":
        return GraphEncoder(charges=charges)
    else:
        raise ValueError(f"No model named '{model_name}' available.")


def get_model(
    model_name: str,
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
    **kwargs,
) -> torch.nn.Module:
    if model_name == "gnn":
        if task_type == "classification":
            return LightningGNNClassifier(
                6,
                d[0],
                d[1],
                n_classes,
                n_hidden_channels,
                class_weights,
                learning_rate,
                **kwargs,
            )
        elif task_type == "regression":
            return LightningGNNRegressor(
                6, d[0], d[1], n_hidden_channels, learning_rate, scaler, **kwargs
            )
        else:
            raise ValueError(
                f"No task type '{task_type}' for model named '{model_name}' available."
            )
    elif model_name == "srgnn":
        if task_type == "classification":
            return LightningSRGNNClassifier(
                n_hidden_sets,
                n_elements,
                6,
                d[0],
                d[1],
                n_classes,
                n_hidden_channels,
                set_layer,
                class_weights,
                learning_rate,
                **kwargs,
            )
        elif task_type == "regression":
            return LightningSRGNNRegressor(
                n_hidden_sets,
                n_elements,
                6,
                d[0],
                d[1],
                n_hidden_channels,
                set_layer,
                learning_rate,
                scaler,
                **kwargs,
            )
        else:
            raise ValueError(
                f"No task type '{task_type}' for model named '{model_name}' available."
            )
    elif model_name == "msr1":
        if task_type == "classification":
            return LightningSRClassifier(
                n_hidden_sets,
                n_elements,
                d,
                n_classes,
                n_hidden_channels,
                class_weights,
                learning_rate,
                set_layer,
                **kwargs,
            )
        elif task_type == "regression":
            return LightningSRRegressor(
                n_hidden_sets,
                n_elements,
                d,
                n_hidden_channels,
                learning_rate=learning_rate,
                set_layer=set_layer,
                scaler=scaler,
                **kwargs,
            )
        else:
            raise ValueError(
                f"No task type '{task_type}' for model named '{model_name}' available."
            )
    elif model_name == "msr2":
        if task_type == "classification":
            return LightningDualSRClassifier(
                n_hidden_sets,
                n_elements,
                d,
                n_classes,
                n_hidden_channels,
                class_weights,
                learning_rate,
                set_layer,
                **kwargs,
            )
        elif task_type == "regression":
            return LightningDualSRRegressor(
                n_hidden_sets,
                n_elements,
                d,
                n_hidden_channels,
                learning_rate=learning_rate,
                scaler=scaler,
                set_layer=set_layer,
                **kwargs,
            )
        else:
            raise ValueError(
                f"No task type '{task_type}' for model named '{model_name}' available."
            )
    elif model_name == "msr3":
        if task_type == "classification":
            return LightningTripleSRClassifier(
                n_hidden_sets,
                n_elements,
                d,
                n_classes,
                n_hidden_channels,
                class_weights,
                learning_rate,
                set_layer,
                **kwargs,
            )
        elif task_type == "regression":
            return LightningTripleSRRegressor(
                n_hidden_sets,
                n_elements,
                d,
                n_hidden_channels,
                learning_rate=learning_rate,
                scaler=scaler,
                set_layer=set_layer,
                **kwargs,
            )
        else:
            raise ValueError(
                f"No task type '{task_type}' for model named '{model_name}' available."
            )
    else:
        raise ValueError(f"No model named '{model_name}' available.")


@app.command()
def main(
    data_set_name: str,
    model_name: str,
    use_class_weights: bool = False,
    max_epochs: int = 50,
    n: int = 1,
    batch_size: int = 64,
    splitter: str = "random",
    task_type: str = "regression",
    n_hidden_sets: Optional[List[int]] = None,
    n_elements: Optional[List[int]] = None,
    n_hidden_channels: Optional[List[int]] = None,
    learning_rate: float = 0.001,
    monitor: Optional[str] = None,
    set_layer: str = "setrep",
    charges: bool = True,
):
    featurizer = None
    set_name = None
    if data_set_name == "pdbbind":
        featurizer = PDBBindFeaturizer()
        set_name = "refined"

    tasks = molnet_task_loader(data_set_name, featurizer=featurizer, set_name=set_name)

    print(f'\nDataset "{data_set_name}" contains {len(tasks)} task(s).')

    label_dtype = torch.long
    if task_type == "regression":
        label_dtype = torch.float

    for task_idx in range(len(tasks)):
        for _ in range(n):
            train, valid, test, _, transforms = molnet_loader(
                data_set_name,
                splitter=splitter,
                reload=False,
                transformers=[],
                featurizer=featurizer,
                set_name=set_name,
            )

            class_weights = None
            if task_type == "classification" and use_class_weights:
                class_weights, _ = get_class_weights(train.y, task_idx)

            scaler = None

            train_y = train.y
            valid_y = valid.y
            test_y = test.y

            if len(train_y.shape) == 1:
                train_y = np.expand_dims(train_y, -1)
                valid_y = np.expand_dims(valid_y, -1)
                test_y = np.expand_dims(test_y, -1)

            train_y = np.array(train_y[:, task_idx])
            valid_y = np.array(valid_y[:, task_idx])
            test_y = np.array(test_y[:, task_idx])

            if task_type == "regression":
                scaler = StandardScaler()
                scaler.fit(np.concatenate((train_y, valid_y, test_y)).reshape(-1, 1))

                train_y = scaler.transform(train_y.reshape(-1, 1)).flatten()
                valid_y = scaler.transform(valid_y.reshape(-1, 1)).flatten()
                test_y = scaler.transform(test_y.reshape(-1, 1)).flatten()

            enc = get_encoder(model_name, data_set_name, charges)

            train_dataset = enc.encode(
                train.X if data_set_name == "pdbbind" else train.ids,
                train_y,
                label_dtype=label_dtype,
            )
            valid_dataset = enc.encode(
                valid.X if data_set_name == "pdbbind" else valid.ids,
                valid_y,
                label_dtype=label_dtype,
            )
            test_dataset = enc.encode(
                test.X if data_set_name == "pdbbind" else test.ids,
                test_y,
                label_dtype=label_dtype,
            )

            if model_name == "gnn" or model_name == "srgnn":
                train_loader = train_dataset
                valid_loader = valid_dataset
                test_loader = test_dataset
            else:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=8,
                    drop_last=True,
                )
                valid_loader = DataLoader(
                    valid_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=8,
                    drop_last=True,
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=8,
                    drop_last=True,
                )

            if model_name == "gnn" or model_name == "srgnn":
                d = [
                    train_loader.dataset[0].num_node_features,
                    train_loader.dataset[0].num_edge_features,
                ]
            else:
                d = [
                    len(train_dataset[0][i][0])
                    for i in range(len(train_dataset[0]) - 1)
                ]

            if n_hidden_sets is None or len(n_hidden_sets) == 0:
                n_hidden_sets = [8] * len(d)

            if n_elements is None or len(n_elements) == 0:
                n_elements = [8] * len(d)

            model = get_model(
                model_name,
                task_type,
                list(n_hidden_sets),
                list(n_elements),
                d,
                2,
                class_weights=class_weights,
                learning_rate=learning_rate,
                scaler=scaler,
                n_hidden_channels=n_hidden_channels,
                set_layer=set_layer,
            )

            if monitor is None:
                monitor = "auroc" if task_type == "classification" else "rmse"

            checkpoint_callback = ModelCheckpoint(monitor=f"val/{monitor}", mode="max")
            if task_type == "regression":
                checkpoint_callback = ModelCheckpoint(
                    monitor=f"val/{monitor}", mode="min"
                )

            wandb_logger = wandb.WandbLogger(project=f"MSR_{data_set_name}_{splitter}")
            wandb_logger.experiment.config.update(
                {"model": model_name, "task": tasks[task_idx]}
            )
            wandb_logger.watch(model, log="all")

            trainer = pl.Trainer(
                callbacks=[checkpoint_callback],
                max_epochs=max_epochs,
                log_every_n_steps=1,
                logger=wandb_logger,
            )

            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=valid_loader
            )
            trainer.test(ckpt_path="best", dataloaders=test_loader)

            wandb_logger.finalize("success")
            wandb_finish()


if __name__ == "__main__":
    app()
