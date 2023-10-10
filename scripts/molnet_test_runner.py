import os
import random
from typing import List, Optional
from multiprocessing import cpu_count
import typer
import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from wandb import finish as wandb_finish
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import wandb

from sklearn.preprocessing import StandardScaler

from molsetrep.utils.datasets import (
    get_class_weights,
    molnet_task_loader,
    molnet_loader,
    ocelot_loader,
    ocelot_task_loader,
    doyle_loader,
    doyle_test_loader,
    doyle_task_loader,
    az_loader,
    az_task_loader,
    suzuki_loader,
    suzuki_task_loader,
    uspto_loader,
    uspto_task_loader,
    adme_loader,
    adme_task_loader,
    custom_molnet_loader,
    custom_molnet_task_loader,
    pdbbind_custom_loader,
    pdbbind_custom_task_loader,
)

from molsetrep.encoders import (
    Encoder,
    SingleSetEncoder,
    DualSetEncoder,
    LigandProtEncoder,
    LigandProtPairEncoder,
    GraphEncoder,
    RXNSetEncoder,
    RXNGraphEncoder,
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

from torch_geometric.nn.models import GAT, GCN

from molsetrep.data import PDBBindFeaturizer

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

app = typer.Typer(pretty_exceptions_enable=False)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

# torch.set_float32_matmul_precision("medium")


def get_encoder(model_name: str, data_set_name: str, charges: bool = True) -> Encoder:
    if data_set_name in [
        "doyle",
        "doyle_test",
        "az",
        "suzuki",
        "uspto",
    ] and model_name in ["gnn", "srgnn"]:
        return RXNGraphEncoder(charges=charges)
    elif data_set_name in ["doyle", "doyle_test", "az", "suzuki", "uspto"]:
        return RXNSetEncoder()
    elif data_set_name in ["pdbbind", "pdbbind-custom"]:
        if model_name == "msr1":
            return LigandProtPairEncoder(charges=charges)
        elif model_name == "msr2":
            return LigandProtEncoder(coords=True, charges=charges)
    elif model_name == "msr1":
        return SingleSetEncoder(charges=charges)
    elif model_name == "msr2":
        return DualSetEncoder(charges=charges)
    elif model_name == "gnn" or model_name == "srgnn":
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
    gnn_type: str = "gine",
    learning_rate: float = 0.001,
    n_layers: int = 6,
    **kwargs,
) -> torch.nn.Module:
    gnn_layer = None

    if gnn_type.lower() == "gat":
        gnn_layer = GAT(
            d[0],
            n_hidden_channels[0],
            n_layers,
            edge_dim=d[1],
            jk="cat",
        )
    elif gnn_type.lower() == "gcn":
        gnn_layer = GCN(d[0], n_hidden_channels[0], n_layers, jk="cat")

    if model_name == "gnn":
        if task_type == "classification":
            return LightningGNNClassifier(
                n_layers,
                d[0],
                d[1],
                n_classes,
                n_hidden_channels,
                class_weights,
                gnn_layer,
                learning_rate,
                **kwargs,
            )
        elif task_type == "regression":
            return LightningGNNRegressor(
                n_layers,
                d[0],
                d[1],
                n_hidden_channels,
                gnn_layer,
                learning_rate,
                scaler,
                **kwargs,
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
                n_layers,
                d[0],
                d[1],
                n_classes,
                n_hidden_channels,
                gnn_layer,
                set_layer,
                class_weights,
                learning_rate,
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
                gnn_layer,
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
    start_n: int = 0,
    batch_size: int = 64,
    splitter: str = "random",
    task_type: str = "regression",
    n_hidden_sets: Optional[List[int]] = None,
    n_elements: Optional[List[int]] = None,
    n_hidden_channels: Optional[List[int]] = None,
    n_layers: int = 6,
    learning_rate: float = 0.001,
    monitor: Optional[str] = None,
    set_layer: str = "setrep",
    gnn_type: str = "gine",
    charges: bool = False,
    project: Optional[str] = None,
    variant: Optional[str] = None,
    split_ratio: float = 0.9,
    task: Optional[List[str]] = None,
    pdbbind_subset: str = "core",
    ckpt_path: str = "best",
):
    featurizer = None
    set_name = None
    data_loader = molnet_loader
    task_loader = molnet_task_loader

    if data_set_name == "ocelot":
        data_loader = ocelot_loader
        task_loader = ocelot_task_loader

    if data_set_name == "adme":
        data_loader = adme_loader
        task_loader = adme_task_loader

    if data_set_name == "doyle":
        data_loader = doyle_loader
        task_loader = doyle_task_loader

    if data_set_name == "doyle_test":
        data_loader = doyle_test_loader
        task_loader = doyle_task_loader

    if data_set_name == "az":
        data_loader = az_loader
        task_loader = az_task_loader

    if data_set_name == "suzuki":
        data_loader = suzuki_loader
        task_loader = suzuki_task_loader

    if data_set_name == "uspto":
        data_loader = uspto_loader
        task_loader = uspto_task_loader

    if data_set_name == "pdbbind-custom":
        featurizer = PDBBindFeaturizer()
        data_loader = pdbbind_custom_loader
        task_loader = pdbbind_custom_task_loader

    if data_set_name == "pdbbind":
        featurizer = PDBBindFeaturizer()
        set_name = pdbbind_subset

    if splitter == "custom-scaffold":
        data_loader = custom_molnet_loader
        task_loader = custom_molnet_task_loader

    tasks = task_loader(data_set_name, featurizer=featurizer, set_name=set_name)
    print(f'\nDataset "{data_set_name}" contains {len(tasks)} task(s).')

    label_dtype = torch.long
    if task_type == "regression":
        label_dtype = torch.float

    for task_idx, task_name in enumerate(tasks):
        if task is not None and len(task) > 0 and task_name not in task:
            continue

        print(f"Running task '{task_name}' ({task_idx})")
        # if task_idx < 8:
        #     continue
        for experiment_idx in range(n):
            seed = experiment_idx + start_n
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            train, valid, test, _, __builtins__ = data_loader(
                data_set_name,
                splitter=splitter,
                reload=False,
                transformers=[],
                featurizer=featurizer,
                set_name=set_name,
                seed=seed,
                fold_idx=experiment_idx,
                n_folds=n,
                task_name=task_name,
                split_ratio=split_ratio,
            )

            # In case tasks are loaded separately (e.g. ADME data)
            if (
                data_set_name not in ["pdbbind", "pdbbind-custom"]
                and len(train.y[0]) == 1
            ):
                task_idx = 0

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
                train.X
                if data_set_name in ["pdbbind", "pdbbind-custom"]
                else train.ids,
                train_y,
                label_dtype=label_dtype,
                batch_size=batch_size,
            )
            valid_dataset = enc.encode(
                valid.X
                if data_set_name in ["pdbbind", "pdbbind-custom"]
                else valid.ids,
                valid_y,
                label_dtype=label_dtype,
                batch_size=batch_size,
            )
            test_dataset = enc.encode(
                test.X if data_set_name in ["pdbbind", "pdbbind-custom"] else test.ids,
                test_y,
                label_dtype=label_dtype,
                batch_size=batch_size,
            )

            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            g = torch.Generator()
            g.manual_seed(0)

            if model_name == "gnn" or model_name == "srgnn":
                train_loader = train_dataset
                valid_loader = valid_dataset
                test_loader = test_dataset
            else:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=cpu_count() if cpu_count() < 8 else 8,
                    drop_last=True,
                    worker_init_fn=seed_worker,
                    generator=g,
                )
                valid_loader = DataLoader(
                    valid_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=cpu_count() if cpu_count() < 8 else 8,
                    drop_last=True,
                    worker_init_fn=seed_worker,
                    generator=g,
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=cpu_count() if cpu_count() < 8 else 8,
                    drop_last=True,
                    worker_init_fn=seed_worker,
                    generator=g,
                )

            if model_name == "gnn" or model_name == "srgnn":
                d = [
                    train_loader.dataset[0].num_node_features,
                    train_loader.dataset[0].num_edge_features,
                ]
            else:
                if len(train_dataset[0][0].shape) > 1:
                    d = [
                        len(train_dataset[0][i][0])
                        for i in range(len(train_dataset[0]) - 1)
                    ]
                else:
                    d = [train_dataset[0][0].shape[0]]

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
                gnn_type=gnn_type,
                set_layer=set_layer,
                n_layers=n_layers,
            )

            if monitor is None:
                monitor = "auroc" if task_type == "classification" else "rmse"

            checkpoint_callback = ModelCheckpoint(monitor=f"val/{monitor}", mode="max")
            if task_type == "regression":
                checkpoint_callback = ModelCheckpoint(
                    monitor=f"val/{monitor}", mode="min"
                )

            if monitor == "loss":
                checkpoint_callback = ModelCheckpoint(
                    monitor=f"val/{monitor}", mode="min"
                )

            project_name = f"MSR_{data_set_name}_{splitter}"
            if project is not None:
                project_name = project

            wandb_logger = wandb.WandbLogger(project=project_name)
            wandb_logger.experiment.config.update(
                {
                    "model": model_name,
                    "task": task_name,
                    "variant": variant,
                    "dataset": data_set_name,
                    "batch_size": batch_size,
                    "splitter": splitter,
                    "experiment_idx": experiment_idx,
                    "split_ratio": split_ratio,
                    "set_layer": set_layer,
                    "gnn_type": gnn_type,
                }
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
            trainer.test(ckpt_path=ckpt_path, dataloaders=test_loader)

            wandb_logger.finalize("success")
            wandb_finish()


if __name__ == "__main__":
    app()
