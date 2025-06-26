import os
import random
from multiprocessing import cpu_count
from typing import List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
import typer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import wandb
from rdkit import RDLogger
from sklearn.preprocessing import StandardScaler
from tdc.benchmark_group import admet_group
from torch.utils.data import DataLoader
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
    tdc_adme_loader,
    tdc_adme_task_loader,
)

RDLogger.DisableLog("rdApp.*")

app = typer.Typer(pretty_exceptions_enable=False)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

# torch.set_float32_matmul_precision("medium")


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
    gnn_dropout: float = 0.0,
    node_encoder_out: int = 0,
    edge_encoder_out: int = 0,
    descriptors: bool = False,
    n_descriptors: int = 200,
    descriptor_mlp: bool = True,
    descriptor_mlp_dropout: float = 0.25,
    descriptor_mlp_bn: bool = True,
    descriptor_mlp_out: int = 32,
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
            **kwargs,
        )


def get_datasets(
    train,
    valid,
    test,
    task_type,
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
        scaler = StandardScaler()
        scaler.fit(train_y.reshape(-1, 1))

        train_y = scaler.transform(train_y.reshape(-1, 1)).flatten()
        valid_y = scaler.transform(valid_y.reshape(-1, 1)).flatten()
        test_y = scaler.transform(test_y.reshape(-1, 1)).flatten()

    train_dataset = encoder.encode(
        train.ids,
        train_y,
        label_dtype=label_dtype,
        batch_size=batch_size,
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

    return train_dataset, valid_dataset, test_dataset


def tdc_benchmark():
    group = admet_group(path="data/")
    predictions_list = []

    for seed in [1, 2, 3, 4, 5]:
        predictions = {}
        for benchmark in group:
            name = benchmark["name"]
            train_val, test = benchmark["train_val"], benchmark["test"]
            train, valid = group.get_train_valid_split(
                benchmark=name, split_type="default", seed=seed
            )

            train_split = CustomDataset.from_df(train, "Drug", ["Y"])
            valid_split = CustomDataset.from_df(valid, "Drug", ["Y"])
            test_split = CustomDataset.from_df(test, "Drug", ["Y"])

            train_dataset, valid_dataset, test_dataset = get_datasets(
                train_split, valid_split, test_split
            )

            predictions[name] = y_pred_test
        predictions_list.append(predictions)

    group.evaluate_many(predictions_list)


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
    gnn_dropout: float = 0.0,
    node_encoder_out: int = 0,
    edge_encoder_out: int = 0,
    descriptors: bool = False,
    n_descriptors: int = 200,
    descriptor_mlp: bool = True,
    descriptor_mlp_dropout: float = 0.25,
    descriptor_mlp_bn: bool = True,
    descriptor_mlp_out: int = 32,
    charges: bool = False,
    project: Optional[str] = None,
    variant: Optional[str] = None,
    split_ratio: float = 0.9,
    task: Optional[List[str]] = None,
    pdbbind_radius: float = 5.5,
    pdbbind_subset: str = "core",
    ckpt_path: str = "best",
    custom_path: Optional[str] = None,
):
    tdc_benchmark()
    # set_name = None
    #
    # data_loader = tdc_adme_loader
    # task_loader = tdc_adme_task_loader
    #
    # tasks = task_loader(data_set_name, set_name=set_name)
    # print(f'\nDataset "{data_set_name}" contains {len(tasks)} task(s).')
    #
    # label_dtype = torch.long
    # if task_type == "regression":
    #     label_dtype = torch.float
    #
    # for task_idx, task_name in enumerate(tasks):
    #     if task is not None and len(task) > 0 and task_name not in task:
    #         continue
    #
    #     print(f"Running task '{task_name}' ({task_idx})")
    #     # if task_idx < 8:
    #     #     continue
    #     for experiment_idx in range(n):
    #         seed = experiment_idx + start_n
    #         torch.manual_seed(seed)
    #         random.seed(seed)
    #         np.random.seed(seed)
    #
    #         train, valid, test, _, __builtins__ = data_loader(
    #             data_set_name,
    #             splitter=splitter,
    #             reload=False,
    #             transformers=[],
    #             featurizer=featurizer,
    #             set_name=set_name,
    #             seed=seed,
    #             fold_idx=experiment_idx,
    #             n_folds=n,
    #             task_name=task_name,
    #             split_ratio=split_ratio,
    #             custom_path=custom_path,
    #         )
    #
    #         # In case tasks are loaded separately (e.g. ADME data)
    #         if (
    #             data_set_name not in ["pdbbind", "pdbbind-custom"]
    #             and len(train.y[0]) == 1
    #         ):
    #             task_idx = 0
    #
    #         class_weights = None
    #         if task_type == "classification" and use_class_weights:
    #             class_weights, _ = get_class_weights(train.y, task_idx)
    #
    #         scaler = None
    #
    #         train_y = train.y
    #         valid_y = valid.y
    #         test_y = test.y
    #
    #         if len(train_y.shape) == 1:
    #             train_y = np.expand_dims(train_y, -1)
    #             valid_y = np.expand_dims(valid_y, -1)
    #             test_y = np.expand_dims(test_y, -1)
    #
    #         train_y = np.array(train_y[:, task_idx])
    #         valid_y = np.array(valid_y[:, task_idx])
    #         test_y = np.array(test_y[:, task_idx])
    #
    #         if task_type == "regression":
    #             scaler = StandardScaler()
    #             scaler.fit(train_y.reshape(-1, 1))
    #
    #             train_y = scaler.transform(train_y.reshape(-1, 1)).flatten()
    #             valid_y = scaler.transform(valid_y.reshape(-1, 1)).flatten()
    #             test_y = scaler.transform(test_y.reshape(-1, 1)).flatten()
    #
    #         train_dataset = encoder.encode(
    #             (
    #                 train.X
    #                 if data_set_name in ["pdbbind", "pdbbind-custom"]
    #                 else train.ids
    #             ),
    #             train_y,
    #             label_dtype=label_dtype,
    #             batch_size=batch_size,
    #         )
    #         valid_dataset = encoder.encode(
    #             (
    #                 valid.X
    #                 if data_set_name in ["pdbbind", "pdbbind-custom"]
    #                 else valid.ids
    #             ),
    #             valid_y,
    #             label_dtype=label_dtype,
    #             batch_size=batch_size,
    #         )
    #         test_dataset = encoder.encode(
    #             test.X if data_set_name in ["pdbbind", "pdbbind-custom"] else test.ids,
    #             test_y,
    #             label_dtype=label_dtype,
    #             batch_size=batch_size,
    #         )
    #
    #         def seed_worker(worker_id):
    #             worker_seed = torch.initial_seed() % 2**32
    #             np.random.seed(worker_seed)
    #             random.seed(worker_seed)
    #
    #         g = torch.Generator()
    #         g.manual_seed(0)
    #
    #         if model_name == "gnn" or model_name == "srgnn":
    #             train_loader = train_dataset
    #             valid_loader = valid_dataset
    #             test_loader = test_dataset
    #         else:
    #             train_loader = DataLoader(
    #                 train_dataset,
    #                 batch_size=batch_size,
    #                 shuffle=True,
    #                 num_workers=cpu_count() if cpu_count() < 8 else 8,
    #                 drop_last=True,
    #                 worker_init_fn=seed_worker,
    #                 generator=g,
    #             )
    #             valid_loader = DataLoader(
    #                 valid_dataset,
    #                 batch_size=batch_size,
    #                 shuffle=False,
    #                 num_workers=cpu_count() if cpu_count() < 8 else 8,
    #                 drop_last=True,
    #                 worker_init_fn=seed_worker,
    #                 generator=g,
    #             )
    #             test_loader = DataLoader(
    #                 test_dataset,
    #                 batch_size=batch_size,
    #                 shuffle=False,
    #                 num_workers=cpu_count() if cpu_count() < 8 else 8,
    #                 drop_last=True,
    #                 worker_init_fn=seed_worker,
    #                 generator=g,
    #             )
    #
    #         if model_name == "gnn" or model_name == "srgnn":
    #             d = [
    #                 train_loader.dataset[0].num_node_features,
    #                 train_loader.dataset[0].num_edge_features,
    #             ]
    #         else:
    #             if len(train_dataset[0][0].shape) > 1:
    #                 d = [
    #                     len(train_dataset[0][i][0])
    #                     for i in range(len(train_dataset[0]) - 1)
    #                 ]
    #             else:
    #                 d = [train_dataset[0][0].shape[0]]
    #
    #         if n_hidden_sets is None or len(n_hidden_sets) == 0:
    #             n_hidden_sets = [8] * len(d)
    #
    #         if n_elements is None or len(n_elements) == 0:
    #             n_elements = [8] * len(d)
    #
    #         model = get_model(
    #             model_name,
    #             task_type,
    #             list(n_hidden_sets),
    #             list(n_elements),
    #             d,
    #             2,
    #             class_weights=class_weights,
    #             learning_rate=learning_rate,
    #             scaler=scaler,
    #             n_hidden_channels=n_hidden_channels,
    #             gnn_type=gnn_type,
    #             set_layer=set_layer,
    #             n_layers=n_layers,
    #             gnn_dropout=gnn_dropout,
    #             node_encoder_out=node_encoder_out,
    #             edge_encoder_out=edge_encoder_out,
    #             descriptors=descriptors,
    #             n_descriptors=n_descriptors,
    #             descriptor_mlp=descriptor_mlp,
    #             descriptor_mlp_dropout=descriptor_mlp_dropout,
    #             descriptor_mlp_bn=descriptor_mlp_bn,
    #             descriptor_mlp_out=descriptor_mlp_out,
    #         )
    #
    #         if monitor is None:
    #             monitor = "auroc" if task_type == "classification" else "mae"
    #
    #         checkpoint_callback = ModelCheckpoint(monitor=f"val/{monitor}", mode="max")
    #         learning_rate_callback = LearningRateMonitor(logging_interval="step")
    #
    #         if task_type == "regression":
    #             checkpoint_callback = ModelCheckpoint(
    #                 monitor=f"val/{monitor}", mode="min"
    #             )
    #
    #         if monitor == "loss":
    #             checkpoint_callback = ModelCheckpoint(
    #                 monitor=f"val/{monitor}", mode="min"
    #             )
    #
    #         project_name = f"MSR_REV_{data_set_name}_{splitter}"
    #         if project is not None:
    #             project_name = project
    #
    #         wandb_logger = wandb.WandbLogger(project=project_name)
    #         wandb_logger.experiment.config.update(
    #             {
    #                 "model": model_name,
    #                 "task": task_name,
    #                 "variant": variant,
    #                 "dataset": data_set_name,
    #                 "batch_size": batch_size,
    #                 "splitter": splitter,
    #                 "experiment_idx": experiment_idx,
    #                 "split_ratio": split_ratio,
    #                 "set_layer": set_layer,
    #                 "gnn_type": gnn_type,
    #                 "gnn_dropout": gnn_dropout,
    #                 "node_encoder_out": node_encoder_out,
    #                 "edge_encoder_out": edge_encoder_out,
    #                 "descriptors": descriptors,
    #                 "n_descriptors": n_descriptors,
    #                 "descriptor_mlp": descriptor_mlp,
    #                 "descriptor_mlp_dropout": descriptor_mlp_dropout,
    #                 "descriptor_mlp_bn": descriptor_mlp_bn,
    #                 "descriptor_mlp_out": descriptor_mlp_out,
    #                 "pdbbind_radius": pdbbind_radius,
    #             }
    #         )
    #         wandb_logger.watch(model, log="all")
    #
    #         trainer = pl.Trainer(
    #             callbacks=[checkpoint_callback],
    #             max_epochs=max_epochs,
    #             log_every_n_steps=1,
    #             logger=wandb_logger,
    #         )
    #
    #         trainer.fit(
    #             model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    #         )
    #         trainer.test(ckpt_path=ckpt_path, dataloaders=test_loader)
    #
    #         wandb_logger.finalize("success")
    #         wandb_finish()


if __name__ == "__main__":
    app()
