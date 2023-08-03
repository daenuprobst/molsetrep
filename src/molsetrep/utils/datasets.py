import torch
from torch.utils.data import DataLoader
import numpy as np
import deepchem.molnet as mn

from molsetrep.encoders import Encoder


def molnet_task_loader(name: str, **kwargs):
    mn_loader = getattr(mn, f"load_{name}")
    dc_set = mn_loader(**kwargs)

    tasks, _, _ = dc_set
    return tasks


def molnet_loader(name: str, **kwargs):
    mn_loader = getattr(mn, f"load_{name}")
    dc_set = mn_loader(**kwargs)

    tasks, dataset, transformers = dc_set
    train, valid, test = dataset
    return train, valid, test, tasks, transformers


def get_class_weights(y, task_idx):
    y_t = y.T

    _, counts = np.unique(y_t[task_idx], return_counts=True)
    weights = [1 - c / y_t[task_idx].shape[0] for c in counts]

    return np.array(weights), np.array(counts)


def molnet_encoded_loader(
    name: str,
    encoder: Encoder,
    task_idx: int,
    label_dtype=torch.long,
    batch_size=64,
    **kwargs,
):
    train, valid, test, _ = molnet_loader(name, kwargs)

    class_weights, class_counts = get_class_weights(train.y, task_idx)

    train_dataset = encoder.encode(
        train.ids, [y[task_idx] for y in train.y], label_dtype=label_dtype
    )
    valid_dataset = encoder.encode(
        valid.ids, [y[task_idx] for y in valid.y], label_dtype=label_dtype
    )
    test_dataset = encoder.encode(
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

    d = [len(train_dataset[0][i][0]) for i in range(len(train_dataset[0]))]

    return train_loader, valid_loader, test_loader, d, [y[task_idx] for y in test.y]
