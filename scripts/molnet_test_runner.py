from typing import Tuple, List, Optional
import typer
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from wandb import finish as wandb_finish
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import wandb

from molsetrep.utils.datasets import (
    get_class_weights,
    molnet_task_loader,
    molnet_encoded_loader,
    molnet_loader,
)

from molsetrep.encoders import (
    Encoder,
    SingleSetEncoder,
    DualSetEncoder,
    TripleSetEncoder,
)
from molsetrep.models import (
    SRClassifier,
    SRRegressor,
    LightningSRClassifier,
    LightningSRRegressor,
    DualSRClassifier,
    DualSRRegressor,
    TripleSRClassifier,
    TripleSRRegressor,
)

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def get_encoder(
    model_name: str,
) -> Encoder:
    if model_name == "msr1":
        return SingleSetEncoder()
    elif model_name == "msr2":
        return DualSetEncoder()
    elif model_name == "msr3":
        return TripleSetEncoder()
    else:
        raise ValueError(f"No model named '{model_name}' available.")


def get_model(
    model_name: str,
    task_type: str,
    n_hidden_sets: int,
    n_elements: int,
    d: int,
    n_classes: int,
    n_hidden_channels: Optional[List[int]] = None,
    class_weights: Optional[List[float]] = None,
    **kwargs,
) -> torch.nn.Module:
    if model_name == "msr1":
        if task_type == "classification":
            return LightningSRClassifier(
                n_hidden_sets,
                n_elements,
                d,
                n_classes,
                n_hidden_channels,
                class_weights,
                **kwargs,
            )
        elif task_type == "regression":
            return LightningSRRegressor(
                n_hidden_sets, n_elements, d, n_hidden_channels, **kwargs
            )
        else:
            raise ValueError(
                f"No task type '{task_type}' for model named '{model_name}' available."
            )
    elif model_name == "msr2":
        if task_type == "classification":
            return DualSRClassifier(**kwargs)
        elif task_type == "regression":
            return DualSRRegressor(**kwargs)
        else:
            raise ValueError(
                f"No task type '{task_type}' for model named '{model_name}' available."
            )
    elif model_name == "msr3":
        if task_type == "classification":
            return TripleSRClassifier(**kwargs)
        elif task_type == "regression":
            return TripleSRRegressor(**kwargs)
        else:
            raise ValueError(
                f"No task type '{task_type}' for model named '{model_name}' available."
            )
    else:
        raise ValueError(f"No model named '{model_name}' available.")


def main(
    data_set_name: str,
    model_name: str,
    use_class_weights: bool = False,
    max_epochs: int = 50,
    n: int = 4,
    batch_size: int = 64,
    splitter: str = "random",
    task_type: str = "regression",
):
    tasks = molnet_task_loader(data_set_name)

    print(f'\nDataset "{data_set_name}" contains {len(tasks)} task(s).')

    label_dtype = torch.long
    if task_type == "regression":
        label_dtype = torch.float

    for task_idx in range(len(tasks)):
        train, valid, test, _ = molnet_loader(
            data_set_name, splitter=splitter, reload=False
        )

        class_weights = None
        if task_type == "classification" and use_class_weights:
            class_weights, _ = get_class_weights(train.y, task_idx)

        enc = get_encoder(model_name)

        train_dataset = enc.encode(
            train.ids, [y[task_idx] for y in train.y], label_dtype=label_dtype
        )
        valid_dataset = enc.encode(
            valid.ids, [y[task_idx] for y in valid.y], label_dtype=label_dtype
        )
        test_dataset = enc.encode(
            test.ids, [y[task_idx] for y in test.y], label_dtype=label_dtype
        )

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

        d = [len(train_dataset[0][i][0]) for i in range(len(train_dataset[0]) - 1)]

        model = get_model(
            model_name, task_type, 8, 8, d[0], 2, class_weights=class_weights
        )

        checkpoint_callback = ModelCheckpoint(monitor="val/auroc", mode="max")
        if task_type == "regression":
            checkpoint_callback = ModelCheckpoint(monitor="val/rmse", mode="min")

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            max_epochs=max_epochs,
            log_every_n_steps=1,
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        trainer.test(ckpt_path="best", dataloaders=test_loader)


if __name__ == "__main__":
    typer.run(main)
