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
    LightningSRClassifier,
    LightningSRRegressor,
    LightningDualSRClassifier,
    LightningDualSRRegressor,
    LightningTripleSRClassifier,
    LightningTripleSRRegressor,
)

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

app = typer.Typer(pretty_exceptions_enable=False)


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
    n_hidden_sets: List[int],
    n_elements: List[int],
    d: List[int],
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
            return LightningDualSRClassifier(
                n_hidden_sets,
                n_elements,
                d,
                n_classes,
                n_hidden_channels,
                class_weights,
                **kwargs,
            )
        elif task_type == "regression":
            return LightningDualSRRegressor(
                n_hidden_sets, n_elements, d, n_hidden_channels, **kwargs
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
                **kwargs,
            )
        elif task_type == "regression":
            return LightningTripleSRRegressor(
                n_hidden_sets, n_elements, d, n_hidden_channels, **kwargs
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
    n: int = 4,
    batch_size: int = 64,
    splitter: str = "random",
    task_type: str = "regression",
    n_hidden_sets: Optional[List[int]] = None,
    n_elements: Optional[List[int]] = None,
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
        )

        checkpoint_callback = ModelCheckpoint(monitor="val/auroc", mode="max")
        if task_type == "regression":
            checkpoint_callback = ModelCheckpoint(monitor="val/rmse", mode="min")

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

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        trainer.test(ckpt_path="best", dataloaders=test_loader)

        wandb_logger.finalize("success")
        wandb_finish()


if __name__ == "__main__":
    app()
